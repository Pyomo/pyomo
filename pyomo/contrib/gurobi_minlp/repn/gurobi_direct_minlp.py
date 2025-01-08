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

from pyomo.common.dependencies import attempt_import
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.numeric_types import native_complex_types

# ESJ TODO: We should move this somewhere sensible
from pyomo.contrib.cp.repn.docplex_writer import collect_valid_components

from pyomo.core.base import (
    Binary,
    Block,
    Constraint,
    Expression,
    Integers,
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

from pyomo.opt import SolverFactory, WriterFactory
from pyomo.repn.util import (
    apply_node_operation,
    BeforeChildDispatcher,
    complex_number_error,
    ExitNodeDispatcher,
    initialize_exit_node_dispatcher,
)


gurobipy, gurobipy_available = attempt_import('gurobipy', minimum_version='12.0.0')
if gurobipy_available:
    from gurobipy import GRB

### FIXME: Remove the following as soon as non-active components no
### longer report active==True
from pyomo.network import Port
from pyomo.core.base import RangeSet, Set
###


_domain_map = ComponentMap((
    (Binary, (GRB.BINARY, -float('inf'), float('inf'))),
    (Integers, (GRB.INTEGER, -float('inf'), float('inf'))),
    (NonNegativeIntegers, (GRB.INTEGER, 0, float('inf'))),
    (NonPositiveIntegers, (GRB.INTEGER, -float('inf'), 0)),
    (NonNegativeReals, (GRB.CONTINUOUS, 0, float('inf'))),
    (NonPositiveReals, (GRB.CONTINUOUS, -float('inf'), 0)),
    (Reals, (GRB.CONTINUOUS, -float('inf'), float('inf'))),
))


def _create_grb_var(visitor, pyomo_var, name=None):
    pyo_domain = pyomo_var.domain
    if pyo_domain in _domain_map:
        domain, domain_lb, domain_ub = _domain_map[pyo_domain]
    else:
        raise ValueError(
            "Unsupported domain for Var '%s': %s" % (pyomo_var.name, pyo_domain)
        )
    lb = max(domain_lb, pyomo_var.lb) if pyomo_var.lb is not None else domain_lb
    ub = min(domain_ub, pyomo_var.ub) if pyomo_var.ub is not None else domain_ub
    return visitor.grb_model.addVar(
        lb=lb,
        ub=ub,
        vtype=domain,
        name=name
    )


class GurobiMINLPBeforeChildDispatcher(BeforeChildDispatcher):
    @staticmethod
    def _before_var(visitor, child):
        _id = id(child)
        if _id not in visitor.var_map:
            if child.fixed:
                # ESJ TODO: I want the linear walker implementation of
                # check_constant... Could it be in the base class or something?
                return False, visitor.check_constant(child.value, child)
            grb_var = _create_grb_var(
                visitor,
                child, name=child.name if visitor.symbolic_solver_labels else None
            )
            visitor.var_map[_id] = grb_var
        return False, visitor.var_map[_id]


def _handle_sum(visitor, node, *args):
    return sum(arg for arg in args)


def _handle_negation(visitor, node, arg):
    return -arg


def _handle_product(visitor, node, arg1, arg2):
    return arg1 * arg2


def _handle_division(visitor, node, arg1, arg2):
    # ESJ TODO: Not 100% sure that this is the right operator overloading in grbpy
    return arg1 / arg2


def _handle_pow(visitor, node, arg1, arg2):
    return arg1 ** arg2


def _handle_unary(visitor, node, arg):
    ans = apply_node_operation(node, (arg[1],))
    # Unary includes sqrt() which can return complex numbers
    if ans.__class__ in native_complex_types:
        ans = complex_number_error(ans, visitor, node)
    return ans


def _handle_abs(visitor, node, arg):
    # TODO
    pass


def _handle_named_expression(visitor, node, arg):
    # TODO
    pass


def _handle_expr_if(visitor, node, arg1, arg2, arg3):
    # TODO
    pass


# TODO: We have to handle relational expression if we support Expr_If :(


def define_exit_node_handlers(_exit_node_handlers=None):
    if _exit_node_handlers is None:
        _exit_node_handlers = {}
    _exit_node_handlers[NegationExpression] = {None: _handle_negation}
    _exit_node_handlers[SumExpression] = {None: _handle_sum}
    _exit_node_handlers[LinearExpression] = {None: _handle_sum}
    _exit_node_handlers[ProductExpression] = {None: _handle_product}
    _exit_node_handlers[MonomialTermExpression] = {None: _handle_product}
    _exit_node_handlers[DivisionExpression] = {None: _handle_division}
    _exit_node_handlers[PowExpression] = {None: _handle_pow}
    _exit_node_handlers[UnaryFunctionExpression] = {None: _handle_unary}
    _exit_node_handlers[AbsExpression] = {None: _handle_abs}
    _exit_node_handlers[Expression] = {None: _handle_named_expression}
    _exit_node_handlers[Expr_ifExpression] = {None: _handle_expr_if}

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
        self.var_map = {}
        self._named_expressions = {}
        self._eval_expr_visitor = _EvaluationVisitor(True)
        self.evaluate = self._eval_expr_visitor.dfs_postorder_stack

    def initializeWalker(self, expr):
        expr, src, src_index = expr
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, self.finalizeResult(result)
        return True, expr

    def beforeChild(self, node, child, child_idx):
        # Return native types
        if child.__class__ in EXPR.native_types:
            return False, child

        return self.before_child_dispatcher[child.__class__](self, child)

    def exitNode(self, node, data):
        return self.exit_node_dispatcher[node.__class__](self, node, *data)

    def finalizeResult(self, result):
        self.grb_model.update()
        return result

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
    'Direct interface to Gurobi that allows for general nonlinear expressions'
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
                Var,
                Param,
                Suffix,
                # FIXME: Non-active components should not report as Active
                Set,
                RangeSet,
                Port,
            },
            targets={
                Objective,
                Constraint,
            },
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

        grb_model = grb.model()
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
            obj_expr = visitor.walk_expression((obj.expr, obj, 0))
            if obj.sense is minimize:
                # TODO
                pass
            else:
                # TODO
                pass
        else:
            # TODO: We have no objective--we should put in a dummy, consistent
            # with the other writers?
            pass

        # write constraints
        for cons in components[Constraint]:
            expr = visitor.walk_expression((cons.body, cons, 0))
            # TODO

        return grb_model, visitor.pyomo_to_gurobipy

# ESJ TODO: We should probably not do this and actually tack this on to another
# solver? But I'm not sure. In any case, it should probably at least inerhit
# from another direct interface to Gurobi since all the handling of licenses and
# termination conditions and things should be common.
@SolverFactory.register('gurobi_direct_minlp',
                        doc='Direct interface to Gurobi version 12 and up '
                        'supporting general nonlinear expressions')
class GurobiMINLPSolver(object):
    CONFIG = ConfigDict("gurobi_minlp_solver")
    CONFIG.declare(
        'symbolic_solver_labels',
        ConfigValue(
            default=False,
            domain=bool,
            description='Write Pyomo Var and Constraint names to gurobipy model',
        ),
    )
    CONFIG.declare(
        'tee',
        ConfigValue(
            default=False, domain=bool, description="Stream solver output to terminal."
        ),
    )
    CONFIG.declare(
        'options', ConfigValue(default={}, description="Dictionary of solver options.")
    )

    def __init__(self, **kwds):
        self.config = self.CONFIG()
        self.config.set_value(kwds)
        # TODO termination conditions and things

    # Support use as a context manager under current solver API
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    def available(self, exception_flag=True):
        # TODO
        pass

    def license_is_valid(self):
        # TODO
        pass

    def solve(self, model, **kwds):
        """Solve the model.

        Args:
            model (Block): a Pyomo model or Block to be solved
        """
        config = self.config()
        config.set_value(kwds)

        writer = GurobiMINLPWriter()
        grb_model, var_map = writer.write(
            model, symbolic_solver_labels=config.symbolic_solver_labels
        )
        # TODO: Is this right??
        grbsol = grb_model.optimize(**self.options)

        # TODO: handle results status
        #return results
