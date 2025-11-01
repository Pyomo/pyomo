#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
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
from pyomo.common.errors import InvalidValueError
from pyomo.common.numeric_types import native_complex_types
from pyomo.common.timing import HierarchicalTimer

from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.solution_loader import SolutionLoaderBase
from pyomo.contrib.solver.common.util import NoSolutionError
from pyomo.contrib.solver.solvers.gurobi_direct import (
    GurobiDirect,
    GurobiDirectSolutionLoader,
)

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

from pyomo.opt import WriterFactory
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.util import (
    apply_node_operation,
    categorize_valid_components,
    ExprType,
    ExitNodeDispatcher,
    BeforeChildDispatcher,
    complex_number_error,
    initialize_exit_node_dispatcher,
    nan,
    OrderedVarRecorder,
    check_constant,
)

import sys

### FIXME: Remove the following as soon as non-active components no
### longer report active==True
from pyomo.network import Port
from pyomo.core.base import RangeSet, Set

###

_CONSTANT = ExprType.CONSTANT
_GENERAL = ExprType.GENERAL
_LINEAR = ExprType.LINEAR
_QUADRATIC = ExprType.QUADRATIC
_VARIABLE = ExprType.VARIABLE

_function_map = {}


def _finalize_gurobipy(gurobipy, available):
    if not available:
        return
    _function_map.update(
        {
            'exp': (_GENERAL, gurobipy.nlfunc.exp),
            'log': (_GENERAL, gurobipy.nlfunc.log),
            'log10': (_GENERAL, gurobipy.nlfunc.log10),
            'sin': (_GENERAL, gurobipy.nlfunc.sin),
            'cos': (_GENERAL, gurobipy.nlfunc.cos),
            'tan': (_GENERAL, gurobipy.nlfunc.tan),
            'sqrt': (_GENERAL, gurobipy.nlfunc.sqrt),
            # Not supporting any of these right now--we'd have to build them from the
            # above:
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


gurobipy, gurobipy_available = attempt_import(
    'gurobipy',
    deferred_submodules=['GRB'],
    callback=_finalize_gurobipy,
    minimum_version='12.0.0',
)
GRB = gurobipy.GRB


"""
In Gurobi 12:

If you have f(x) == 0, you must write it as z == f(x) and then write z == 0.
Basically, you must introduce auxiliary variables for all the general nonlinear
parts. (And no worries about additively separable or anything--they do that 
under the hood).

In this implementation, we replace the *entire* LHS of the constraint with the
auxiliary variable rather than just the nonlinear part. Otherwise we would really
need to keep track of what nonlinear subexpressions we had already replaced and make
sure to use the same auxiliary variables, and from what we know, this is probably not
worth it.

We are not using Gurobi's '.nl' attribute at all for now--its usage seems like the
exception rather than the rule, so we will let Gurobi expand the expressions for now.
"""


def _create_grb_var(visitor, pyomo_var, name=""):
    pyo_domain = pyomo_var.domain
    domain_lb, domain_ub, domain = pyo_domain.get_interval()
    if domain not in (0, 1):
        raise ValueError(
            "Unsupported domain for Var '%s': %s" % (pyomo_var.name, pyomo_var.domain)
        )
    domain = GRB.INTEGER if domain else GRB.CONTINUOUS
    # We set binaries to be binary right now because we don't know if Gurbi cares
    if pyo_domain is Binary:
        domain = GRB.BINARY

    # returns tigter of bounds from domain and bounds set on variable
    lb, ub = pyomo_var.bounds
    if lb is None:
        lb = -float("inf")
    if ub is None:
        ub = float("inf")

    return visitor.grb_model.addVar(lb=lb, ub=ub, vtype=domain, name=name)


class GurobiMINLPBeforeChildDispatcher(BeforeChildDispatcher):
    @staticmethod
    def _before_var(visitor, child):
        if child not in visitor.var_map:
            if child.fixed:
                return False, (_CONSTANT, check_constant(child.value, child, visitor))
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
    # The expression type is whatever the highest one of the incoming arguments
    # was.
    expr_type = max(map(itemgetter(0), data))
    return (
        expr_type,
        visitor._eval_expr_visitor.visit(node, tuple(map(itemgetter(1), data))),
    )


def _handle_node_with_eval_expr_visitor_constant(visitor, node, *data):
    return (
        _CONSTANT,
        visitor._eval_expr_visitor.visit(node, tuple(map(itemgetter(1), data))),
    )


def _handle_node_with_eval_expr_visitor_linear(visitor, node, *data):
    return (
        _LINEAR,
        visitor._eval_expr_visitor.visit(node, tuple(map(itemgetter(1), data))),
    )


def _handle_node_with_eval_expr_visitor_quadratic(visitor, node, *data):
    return (
        _QUADRATIC,
        visitor._eval_expr_visitor.visit(node, tuple(map(itemgetter(1), data))),
    )


def _handle_node_with_eval_expr_visitor_nonlinear(visitor, node, *data):
    # ESJ: _apply_operation for DivisionExpression expects that result
    # supports __getitem__, so I'm expanding the map to a tuple.
    return (
        _GENERAL,
        visitor._eval_expr_visitor.visit(node, tuple(map(itemgetter(1), data))),
    )


def _handle_linear_constant_pow_expr(visitor, node, arg1, arg2):
    expr_type = _GENERAL
    if arg2[1] == 1:
        expr_type = _LINEAR
    if arg2[1] == 2:
        expr_type = _QUADRATIC
    return (
        expr_type,
        visitor._eval_expr_visitor.visit(node, tuple(map(itemgetter(1), (arg1, arg2)))),
    )


def _handle_quadratic_constant_pow_expr(visitor, node, arg1, arg2):
    expr_type = _GENERAL
    if arg2[1] == 1:
        expr_type = _QUADRATIC
    return (
        expr_type,
        visitor._eval_expr_visitor.visit(node, tuple(map(itemgetter(1), (arg1, arg2)))),
    )


def _handle_unary(visitor, node, data):
    if node._name in _function_map:
        expr_type, fcn = _function_map[node._name]
        return expr_type, fcn(data[1])
    raise ValueError(
        "The unary function '%s' is not supported by the Gurobi MINLP writer."
        % node._name
    )


def _handle_unary_constant(visitor, node, data):
    try:
        return _CONSTANT, node._fcn(value(data[1]))
    except:
        raise InvalidValueError(
            f"Invalid number encountered evaluating constant unary expression "
            f"{node}: {sys.exc_info()[1]}"
        )


def _handle_named_expression(visitor, node, arg1):
    # Record this common expression
    visitor.subexpression_cache[id(node)] = arg1
    _type, arg1 = arg1
    return _type, arg1


def _handle_abs_constant(visitor, node, arg1):
    return (_CONSTANT, abs(arg1[1]))


def _handle_abs_var(visitor, node, arg1):
    # This auxiliary variable actually is non-negative, yay absolute value!
    aux_abs = visitor.grb_model.addVar()
    visitor.grb_model.addConstr(aux_abs == gurobipy.abs_(arg1[1]))

    return (_VARIABLE, aux_abs)


def _handle_abs_expression(visitor, node, arg1):
    # we need an auxiliary variable
    aux_arg = visitor.grb_model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
    visitor.grb_model.addConstr(aux_arg == arg1[1])
    # This one truly is non-negative because it's an absolute value
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
        # Can come back LINEAR or CONSTANT, so we use the 'unknown' version
        None: _handle_node_with_eval_expr_visitor_unknown
    }
    _exit_node_handlers[NegationExpression] = {
        None: _handle_node_with_eval_expr_visitor_invariant
    }
    _exit_node_handlers[ProductExpression] = {
        None: _handle_node_with_eval_expr_visitor_nonlinear,
        (_CONSTANT, _CONSTANT): _handle_node_with_eval_expr_visitor_constant,
        (_CONSTANT, _LINEAR): _handle_node_with_eval_expr_visitor_linear,
        (_CONSTANT, _QUADRATIC): _handle_node_with_eval_expr_visitor_quadratic,
        (_CONSTANT, _VARIABLE): _handle_node_with_eval_expr_visitor_linear,
        (_LINEAR, _CONSTANT): _handle_node_with_eval_expr_visitor_linear,
        (_LINEAR, _LINEAR): _handle_node_with_eval_expr_visitor_quadratic,
        (_LINEAR, _VARIABLE): _handle_node_with_eval_expr_visitor_quadratic,
        (_VARIABLE, _CONSTANT): _handle_node_with_eval_expr_visitor_linear,
        (_VARIABLE, _LINEAR): _handle_node_with_eval_expr_visitor_quadratic,
        (_VARIABLE, _VARIABLE): _handle_node_with_eval_expr_visitor_quadratic,
    }
    _exit_node_handlers[MonomialTermExpression] = _exit_node_handlers[ProductExpression]
    _exit_node_handlers[DivisionExpression] = {
        None: _handle_node_with_eval_expr_visitor_nonlinear,
        (_CONSTANT, _CONSTANT): _handle_node_with_eval_expr_visitor_constant,
        (_LINEAR, _CONSTANT): _handle_node_with_eval_expr_visitor_linear,
        (_VARIABLE, _CONSTANT): _handle_node_with_eval_expr_visitor_linear,
        (_QUADRATIC, _CONSTANT): _handle_node_with_eval_expr_visitor_quadratic,
    }
    _exit_node_handlers[PowExpression] = {
        None: _handle_node_with_eval_expr_visitor_nonlinear,
        (_CONSTANT, _CONSTANT): _handle_node_with_eval_expr_visitor_constant,
        (_VARIABLE, _CONSTANT): _handle_linear_constant_pow_expr,
        (_LINEAR, _CONSTANT): _handle_linear_constant_pow_expr,
        (_QUADRATIC, _CONSTANT): _handle_quadratic_constant_pow_expr,
    }
    _exit_node_handlers[UnaryFunctionExpression] = {
        None: _handle_unary,
        (_CONSTANT,): _handle_unary_constant,
    }

    ## TODO: ExprIf, RangedExpressions (if we do exprif...)
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
        return self.before_child_dispatcher[child.__class__](self, child)

    def exitNode(self, node, data):
        return self.exit_node_dispatcher[(node.__class__, *map(itemgetter(0), data))](
            self, node, *data
        )

    def finalizeResult(self, result):
        return result


@WriterFactory.register(
    'gurobi_minlp',
    'Direct interface to Gurobi that allows for general nonlinear expressions',
)
class GurobiMINLPWriter:
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
        Returns a gurobipy representation of the expression
        """
        expr_type, grb_expr = grb_visitor.walk_expression(expr)
        if expr_type is not _GENERAL:
            return expr_type, grb_expr, False, None
        else:
            aux = grb_model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
            return expr_type, grb_expr, True, aux

    def write(self, model, **options):
        config = options.pop('config', self.config)(options)

        components, unknown = categorize_valid_components(
            model,
            active=True,
            sort=SortComponents.deterministic,
            valid={
                Block,
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
                        sorted(
                            "%s:\n\t\t%s"
                            % (k, "\n\t\t".join(sorted(map(attrgetter('name'), v))))
                            for k, v in unknown.items()
                        )
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

        active_objs = []
        if components[Objective]:
            for block in components[Objective]:
                for obj in block.component_data_objects(
                    Objective,
                    active=True,
                    descend_into=False,
                    sort=SortComponents.deterministic,
                ):
                    active_objs.append(obj)
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
            expr_type, obj_expr, nonlinear, aux = self._create_gurobi_expression(
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
        pyo_cons = []
        grb_cons = []

        if components[Constraint]:
            for block in components[Constraint]:
                for cons in block.component_data_objects(
                    Constraint,
                    active=True,
                    descend_into=False,
                    sort=SortComponents.deterministic,
                ):
                    lb, body, ub = cons.to_bounded_expression(evaluate_bounds=True)
                    expr_type, expr, nonlinear, aux = self._create_gurobi_expression(
                        body, cons, 0, grb_model, quadratic_visitor, visitor
                    )
                    if nonlinear:
                        grb_model.addConstr(aux == expr)
                        expr = aux
                    elif expr_type == _CONSTANT:
                        # cast everything to a float in case there are numpy
                        # types because you can't do addConstr(np.True_)
                        expr = float(expr)
                        if lb is not None:
                            lb = float(lb)
                        if ub is not None:
                            ub = float(ub)
                    if cons.equality:
                        grb_cons.append(grb_model.addConstr(expr == lb))
                        pyo_cons.append(cons)
                    else:
                        # TODO: should we have special handling if expr is a
                        # GRB.LinExpr so that we can use the ranged linear
                        # constraint syntax (expr == [lb, ub])?
                        if lb is not None:
                            grb_cons.append(grb_model.addConstr(expr >= lb))
                            pyo_cons.append(cons)
                        if ub is not None:
                            grb_cons.append(grb_model.addConstr(expr <= ub))
                            pyo_cons.append(cons)

        grb_model.update()
        return grb_model, visitor.var_map, pyo_obj, grb_cons, pyo_cons


@SolverFactory.register(
    'gurobi_direct_minlp',
    doc='Direct interface to Gurobi version 12 and up '
    'supporting general nonlinear expressions',
)
class GurobiDirectMINLP(GurobiDirect):
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
        grb_model, var_map, pyo_obj, grb_cons, pyo_cons = writer.write(
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
            timer,
            config,
            GurobiDirectSolutionLoader(
                grb_model,
                grb_cons=grb_cons,
                grb_vars=var_map.values(),
                pyo_cons=pyo_cons,
                pyo_vars=var_map.keys(),
                pyo_obj=pyo_obj,
            ),
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
