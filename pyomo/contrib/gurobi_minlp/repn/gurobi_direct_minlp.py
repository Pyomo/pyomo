#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import datetime
import io
from operator import attrgetter, itemgetter

from pyomo.common.dependencies import attempt_import
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.numeric_types import native_complex_types
from pyomo.common.timing import HierarchicalTimer

# ESJ TODO: We should move this somewhere sensible
from pyomo.contrib.cp.repn.docplex_writer import collect_valid_components
from pyomo.contrib.solver.common.solution_loader import SolutionLoaderBase
from pyomo.contrib.solver.solvers.gurobi_direct import GurobiDirect

from pyomo.core.base import (
    Binary,
    Block,
    BooleanVar,
    Constraint,
    Expression,
    Integers,
    minimize,
    maximize,
    NonNegativeIntegers,
    NonNegativeReals,
    NonPositiveIntegers,
    NonPositiveReals,
    Objective,
    Param,
    Reals,
    SortComponents,
    Suffix,
    Var,
    value,
)
import pyomo.core.expr as EXPR
from pyomo.core.expr.numeric_expr import (
    NegationExpression,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    AbsExpression,
    UnaryFunctionExpression,
    Expr_ifExpression,
    LinearExpression,
    MonomialTermExpression,
    SumExpression,
)
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.staleflag import StaleFlagManager

from pyomo.opt import SolverFactory, WriterFactory
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.util import (
    apply_node_operation,
    ExprType,
    ExitNodeDispatcher,
    BeforeChildDispatcher,
    complex_number_error,
    initialize_exit_node_dispatcher,
    InvalidNumber,
    nan,
    OrderedVarRecorder,
)

## DEBUG
from pytest import set_trace

"""
Even in Gurobi 12:

If you have f(x) == 0, you must write it as z == f(x) and then write z == 0.
Basically, you must introduce auxiliary variables for all the general nonlinear
parts. (And no worries about additively separable or anything--they do that 
under the hood).

Radhakrishna thinks we should replace the *entire* LHS of the constraint with the
auxiliary variable rather than just the nonlinear part. Otherwise we would really
need to keep track of what nonlinear subexpressions we had already replaced and make
sure to use the same auxiliary variables.

Conclusion: So I think I should actually build on top of the linear walker and then
replace anything that has a nonlinear part...

Model.addConstr() doesn't have the three-arg version anymore.

Let's not use the '.nl' attribute at all for now--seems like the exception rather than
the rule that you would want to specifically tell Gurobi *not* to expand the expression.
"""

_CONSTANT = ExprType.CONSTANT
_GENERAL = ExprType.GENERAL
_LINEAR = ExprType.LINEAR
_VARIABLE = ExprType.VARIABLE

_function_map = {}

gurobipy, gurobipy_available = attempt_import('gurobipy', minimum_version='12.0.0')
if gurobipy_available:
    from gurobipy import GRB, nlfunc

    _function_map.update(
        {
            'exp': (_GENERAL, nlfunc.exp),
            'log': (_GENERAL, nlfunc.log),
            'log10': (_GENERAL, nlfunc.log10),
            'sin': (_GENERAL, nlfunc.sin),
            'cos': (_GENERAL, nlfunc.cos),
            'tan': (_GENERAL, nlfunc.tan),
            'sqrt': (_GENERAL, nlfunc.sqrt),
            # TODO: We'll have to do functional programming things if we want to support
            # any of these...
            # 'asin': None,
            # 'sinh': None,
            # 'asinh': None,
            # 'acos': None,
            # 'cosh': None,
            # 'acosh': None,
            # 'atan': None,
            # 'tanh': None,
            # 'atanh': None,
            # 'ceil': None,
            # 'floor': None,
        }
    )

### FIXME: Remove the following as soon as non-active components no
### longer report active==True
from pyomo.network import Port
from pyomo.core.base import RangeSet, Set

###


_domain_map = ComponentMap(
    (
        (Binary, (GRB.BINARY, -float('inf'), float('inf'))),
        (Integers, (GRB.INTEGER, -float('inf'), float('inf'))),
        (NonNegativeIntegers, (GRB.INTEGER, 0, float('inf'))),
        (NonPositiveIntegers, (GRB.INTEGER, -float('inf'), 0)),
        (NonNegativeReals, (GRB.CONTINUOUS, 0, float('inf'))),
        (NonPositiveReals, (GRB.CONTINUOUS, -float('inf'), 0)),
        (Reals, (GRB.CONTINUOUS, -float('inf'), float('inf'))),
    )
)


def _create_grb_var(visitor, pyomo_var, name=""):
    pyo_domain = pyomo_var.domain
    if pyo_domain in _domain_map:
        domain, domain_lb, domain_ub = _domain_map[pyo_domain]
    else:
        raise ValueError(
            "Unsupported domain for Var '%s': %s" % (pyomo_var.name, pyo_domain)
        )
    lb = max(domain_lb, pyomo_var.lb) if pyomo_var.lb is not None else domain_lb
    ub = min(domain_ub, pyomo_var.ub) if pyomo_var.ub is not None else domain_ub
    return visitor.grb_model.addVar(lb=lb, ub=ub, vtype=domain, name=name)


class GurobiMINLPBeforeChildDispatcher(BeforeChildDispatcher):
    @staticmethod
    def _before_var(visitor, child):
        if child not in visitor.var_map:
            if child.fixed:
                # ESJ TODO: I want the linear walker implementation of
                # check_constant... Could it be in the base class or something?
                return False, (_CONSTANT, visitor.check_constant(child.value, child))
            grb_var = _create_grb_var(
                visitor,
                child,
                name=child.name if visitor.symbolic_solver_labels else "",
            )
            visitor.var_map[child] = grb_var
        return False, (_VARIABLE, visitor.var_map[child])

    @staticmethod
    def _before_named_expression(visitor, child):
        _id = id(child)
        if _id in visitor.subexpression_cache:
            _type, expr = visitor.subexpression_cache[_id]
            return False, (_type, expr)
        else:
            return True, None


def _handle_node_with_eval_expr_visitor_invariant(visitor, node, data):
    """
    Calls expression evaluation visitor on nodes that have an invariant
    expression type in the return.
    """
    return (data[0], visitor._eval_expr_visitor.visit(node, (data[1],)))


def _handle_node_with_eval_expr_visitor_unknown(visitor, node, *data):
    # ESJ: Is this cheating?
    expr_type = max(map(itemgetter(0), data))
    return (expr_type, visitor._eval_expr_visitor.visit(node, tuple(map(itemgetter(1), data))))


def _handle_node_with_eval_expr_visitor_constant(visitor, node, *data):
    return (_CONSTANT, visitor._eval_expr_visitor.visit(node, tuple(map(itemgetter(1), data))))


def _handle_node_with_eval_expr_visitor_linear(visitor, node, *data):
    return (_LINEAR, visitor._eval_expr_visitor.visit(node, tuple(map(itemgetter(1), data))))


def _handle_node_with_eval_expr_visitor_nonlinear(visitor, node, *data):
    # ESJ: _apply_operation for DivisionExpression expects that result is indexed, so
    # I'm making it a tuple rather than a map.
    return (
        _GENERAL,
        visitor._eval_expr_visitor.visit(node, tuple(map(itemgetter(1), data))),
    )


def _handle_unary(visitor, node, data):
    if node._name in _function_map:
        expr_type, fcn = _function_map[node._name]
        return expr_type, fcn(data[1])
    raise ValueError(
        "The unary function '%s' is not supported by the Gurobi MINLP writer."
        % node._name
    )


def _handle_named_expression(visitor, node, arg1):
    # Record this common expression
    visitor.subexpression_cache[id(node)] = arg1
    _type, arg1 = arg1
    return _type, arg1


def _handle_abs_constant(visitor, node, arg1):
    return (_CONSTANT, abs(arg1[1]))


def _handle_abs_var(visitor, node, arg1):
    aux_abs = visitor.grb_model.addVar()
    visitor.grb_model.addConstr(aux_abs == gurobipy.abs_(arg1[1]))

    return (_VARIABLE, aux_abs)


def _handle_abs_expression(visitor, node, arg1):
    # we need auxiliary variable
    aux_arg = visitor.grb_model.addVar()
    visitor.grb_model.addConstr(aux_arg == arg1[1])
    aux_abs = visitor.grb_model.addVar()
    visitor.grb_model.addConstr(aux_abs == gurobipy.abs_(aux_arg))

    return (_VARIABLE, aux_abs)


def define_exit_node_handlers(_exit_node_handlers=None):
    if _exit_node_handlers is None:
        _exit_node_handlers = {}

    # We can rely on operator overloading for many, but not all expressions.
    _exit_node_handlers[SumExpression] = {
        None: _handle_node_with_eval_expr_visitor_unknown
    }
    _exit_node_handlers[LinearExpression] = {
        None: _handle_node_with_eval_expr_visitor_linear
    }
    _exit_node_handlers[NegationExpression] = {
        None: _handle_node_with_eval_expr_visitor_invariant
    }
    _exit_node_handlers[ProductExpression] = {
        None: _handle_node_with_eval_expr_visitor_nonlinear,
        (_CONSTANT, _CONSTANT): _handle_node_with_eval_expr_visitor_constant,
        (_CONSTANT, _LINEAR): _handle_node_with_eval_expr_visitor_linear,
        (_LINEAR, _CONSTANT): _handle_node_with_eval_expr_visitor_linear,
        (_CONSTANT, _VARIABLE): _handle_node_with_eval_expr_visitor_linear,
        (_VARIABLE, _CONSTANT): _handle_node_with_eval_expr_visitor_linear,
    }
    _exit_node_handlers[MonomialTermExpression] = _exit_node_handlers[ProductExpression]
    _exit_node_handlers[DivisionExpression] = {
        None: _handle_node_with_eval_expr_visitor_nonlinear,
        (_CONSTANT, _CONSTANT): _handle_node_with_eval_expr_visitor_constant,
        (_LINEAR, _CONSTANT): _handle_node_with_eval_expr_visitor_linear,
        (_VARIABLE, _CONSTANT): _handle_node_with_eval_expr_visitor_linear,
    }
    _exit_node_handlers[PowExpression] = {
        None: _handle_node_with_eval_expr_visitor_nonlinear,
        (_CONSTANT, _CONSTANT): _handle_node_with_eval_expr_visitor_constant,
    }
    _exit_node_handlers[UnaryFunctionExpression] = {None: _handle_unary}

    ## TODO: ExprIf, RangedExpressions (if we do exprif...
    _exit_node_handlers[Expression] = {None: _handle_named_expression}

    # These are special because of quirks of Gurobi's current support for general
    # nonlinear:
    _exit_node_handlers[AbsExpression] = {
        None: _handle_abs_expression,
        (_CONSTANT,): _handle_abs_constant,
        (_VARIABLE,): _handle_abs_var,
    }

    return _exit_node_handlers


class GurobiMINLPVisitor(StreamBasedExpressionVisitor):
    before_child_dispatcher = GurobiMINLPBeforeChildDispatcher()
    exit_node_dispatcher = ExitNodeDispatcher(
        initialize_exit_node_dispatcher(define_exit_node_handlers())
    )

    def __init__(self, grb_model, symbolic_solver_labels=False):
        super().__init__()
        self.grb_model = grb_model
        self.symbolic_solver_labels = symbolic_solver_labels
        self.var_map = ComponentMap()
        self.subexpression_cache = {}
        self._eval_expr_visitor = _EvaluationVisitor(True)
        self.evaluate = self._eval_expr_visitor.dfs_postorder_stack

    def initializeWalker(self, expr):
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, self.finalizeResult(result)
        return True, expr

    def beforeChild(self, node, child, child_idx):
        # # Return native types
        # if child.__class__ in EXPR.native_types:
        #     return False, child

        return self.before_child_dispatcher[child.__class__](self, child)

    def exitNode(self, node, data):
        return self.exit_node_dispatcher[(node.__class__, *map(itemgetter(0), data))](
            self, node, *data
        )

    def finalizeResult(self, result):
        self.grb_model.update()
        return result[1]

    # ESJ TODO: THIS IS COPIED FROM THE LINEAR WALKER--CAN WE PUT IT IN UTIL OR
    # SOMETHING?
    def check_constant(self, ans, obj):
        if ans.__class__ not in EXPR.native_numeric_types:
            # None can be returned from uninitialized Var/Param objects
            if ans is None:
                return InvalidNumber(
                    None, f"'{obj}' evaluated to a nonnumeric value '{ans}'"
                )
            if ans.__class__ is InvalidNumber:
                return ans
            elif ans.__class__ in native_complex_types:
                return complex_number_error(ans, self, obj)
            else:
                # It is possible to get other non-numeric types.  Most
                # common are bool and 1-element numpy.array().  We will
                # attempt to convert the value to a float before
                # proceeding.
                #
                # TODO: we should check bool and warn/error (while bool is
                # convertible to float in Python, they have very
                # different semantic meanings in Pyomo).
                try:
                    ans = float(ans)
                except:
                    return InvalidNumber(
                        ans, f"'{obj}' evaluated to a nonnumeric value '{ans}'"
                    )
        if ans != ans:
            return InvalidNumber(
                nan, f"'{obj}' evaluated to a nonnumeric value '{ans}'"
            )
        return ans


@WriterFactory.register(
    'gurobi_minlp',
    'Direct interface to Gurobi that allows for general nonlinear expressions',
)
class GurobiMINLPWriter(object):
    CONFIG = ConfigDict('gurobi_minlp_writer')
    CONFIG.declare(
        'symbolic_solver_labels',
        ConfigValue(
            default=False,
            domain=bool,
            description='Write Pyomo Var and Constraint names to Gurobi model',
        ),
    )

    def __init__(self):
        self.config = self.CONFIG()

    def _create_gurobi_expression(
        self, expr, src, src_index, grb_model, quadratic_visitor, grb_visitor
    ):
        """
        Uses the quadratic walker to determine if the expression is a general
        nonlinear (non-quadratic) expression, and returns a gurobipy representation
        of the expression
        """
        repn = quadratic_visitor.walk_expression(expr)
        if repn.nonlinear is None:
            grb_expr = grb_visitor.walk_expression(expr)
            return grb_expr, False, None
        else:
            # It's general nonlinear
            grb_expr = grb_visitor.walk_expression(expr)
            aux = grb_model.addVar()
            return grb_expr, True, aux

    def write(self, model, **options):
        config = options.pop('config', self.config)(options)

        components, unknown = collect_valid_components(
            model,
            active=True,
            sort=SortComponents.deterministic,
            valid={
                Block,
                Objective,
                Constraint,
                Expression,
                Var,
                BooleanVar,
                Param,
                Suffix,
                # FIXME: Non-active components should not report as Active
                Set,
                RangeSet,
                Port,
            },
            targets={Objective, Constraint},
        )
        if unknown:
            raise ValueError(
                "The model ('%s') contains the following active components "
                "that the Gurobi MINLP writer does not know how to "
                "process:\n\t%s"
                % (
                    model.name,
                    "\n\t".join(
                        "%s:\n\t\t%s" % (k, "\n\t\t".join(map(attrgetter('name'), v)))
                        for k, v in unknown.items()
                    ),
                )
            )

        # Get a quadratic walker instance
        quadratic_visitor = QuadraticRepnVisitor(
            subexpression_cache={}, var_recorder=OrderedVarRecorder({}, {}, None)
        )

        # create Gurobi model
        grb_model = gurobipy.Model()
        visitor = GurobiMINLPVisitor(
            grb_model, symbolic_solver_labels=config.symbolic_solver_labels
        )

        active_objs = components[Objective]
        if len(active_objs) > 1:
            raise ValueError(
                "More than one active objective defined for "
                "input model '%s': Cannot write to gurobipy." % model.name
            )
        elif len(active_objs) == 1:
            obj = active_objs[0]
            pyo_obj = [obj]
            if obj.sense is minimize:
                sense = GRB.MINIMIZE
            else:
                sense = GRB.MAXIMIZE
            obj_expr, nonlinear, aux = self._create_gurobi_expression(
                obj.expr, obj, 0, grb_model, quadratic_visitor, visitor
            )
            if nonlinear:
                # The objective must be linear or quadratic, so we move the nonlinear
                # one to the constraints
                grb_model.setObjective(aux, sense=sense)
                grb_model.addConstr(aux == obj_expr)
            else:
                grb_model.setObjective(obj_expr, sense=sense)
        # else it's fine--Gurobi doesn't require us to give an objective, so we don't
        # either, but we do have to pass the info through for the results object
        else:
            pyo_obj = []

        # write constraints
        for cons in components[Constraint]:
            expr, nonlinear, aux = self._create_gurobi_expression(
                cons.body, cons, 0, grb_model, quadratic_visitor, visitor
            )
            if nonlinear:
                grb_model.addConstr(aux == expr)
                expr = aux
            if cons.equality:
                grb_model.addConstr(value(cons.lower) == expr)
            else:
                if cons.lb is not None:
                    grb_model.addConstr(value(cons.lb) <= expr)
                if cons.ub is not None:
                    grb_model.addConstr(value(cons.ub) >= expr)

        grb_model.update()
        return grb_model, visitor.var_map, pyo_obj


class GurobiMINLPSolutionLoader(SolutionLoaderBase):
    def __init__(self, grb_model, var_map, pyo_obj):
        self._grb_model = grb_model
        self._pyo_to_grb_var_map = var_map
        self._pyo_obj = pyo_obj

    def load_vars(self, vars_to_load=None, solution_number=0):
        assert solution_number == 0
        if self._grb_model.SolCount == 0:
            raise NoSolutionError()

        if vars_to_load:
            vars_to_load = ComponentSet(vars_to_load)
        else:
            vars_to_load = ComponentSet(self._pyo_to_grb_var_map.keys())

        for pyo_var, grb_var in self._pyo_to_grb_var_map.items():
            if pyo_var in vars_to_load:
                pyo_var.set_value(grb_var.x, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)


# ESJ TODO: I just did the most convenient inheritence for the moment--if this is the
# right thing to do is a different question.
@SolverFactory.register(
    'gurobi_direct_minlp',
    doc='Direct interface to Gurobi version 12 and up '
    'supporting general nonlinear expressions',
)
class GurobiMINLPSolver(GurobiDirect):
    def solve(self, model, **kwds):
        """Solve the model.

        Args:
            model (Block): a Pyomo model or Block to be solved
        """
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        config = self.config(value=kwds, preserve_implicit=True)
        if not self.available():
            c = self.__class__
            raise ApplicationError(
                f'Solver {c.__module__}.{c.__qualname__} is not available '
                f'({self.available()}).'
            )
        if config.timer is None:
            config.timer = HierarchicalTimer()
        timer = config.timer

        StaleFlagManager.mark_all_as_stale()

        timer.start('compile_model')

        writer = GurobiMINLPWriter()
        grb_model, var_map, pyo_obj = writer.write(
            model, symbolic_solver_labels=config.symbolic_solver_labels
        )

        timer.stop('compile_model')

        ostreams = [io.StringIO()] + config.tee

        # set options
        options = config.solver_options

        grb_model.setParam('LogToConsole', 1)

        if config.threads is not None:
            grb_model.setParam('Threads', config.threads)
        if config.time_limit is not None:
            grb_model.setParam('TimeLimit', config.time_limit)
        if config.rel_gap is not None:
            grb_model.setParam('MIPGap', config.rel_gap)
        if config.abs_gap is not None:
            grb_model.setParam('MIPGapAbs', config.abs_gap)

        if config.use_mipstart:
            raise MouseTrap("MIPSTART not yet supported")

        for key, option in options.items():
            grb_model.setParam(key, option)

        grbsol = grb_model.optimize()

        res = self._postsolve(
            timer, config, GurobiMINLPSolutionLoader(grb_model, var_map, pyo_obj)
        )

        res.solver_config = config
        res.solver_name = 'Gurobi'
        res.solver_version = self.version()
        res.solver_log = ostreams[0].getvalue()

        end_timestamp = datetime.datetime.now(datetime.timezone.utc)
        res.timing_info.start_timestamp = start_timestamp
        res.timing_info.wall_time = (end_timestamp - start_timestamp).total_seconds()
        res.timing_info.timer = timer
        return res
