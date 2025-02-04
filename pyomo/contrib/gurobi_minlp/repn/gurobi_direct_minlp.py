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
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.util import (
    apply_node_operation,
    ExprType,
    BeforeChildDispatcher,
    complex_number_error,
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


_function_map = {}

gurobipy, gurobipy_available = attempt_import('gurobipy', minimum_version='12.0.0')
if gurobipy_available:
    from gurobipy import GRB, nlfunc

    _function_map.update(
        {
            'exp': nlfunc.exp,
            'log': nlfunc.log,
            'log10': nlfunc.log10,
            'sin': nlfunc.sin,
            'cos': nlfunc.cos,
            'tan': nlfunc.tan,
            'sqrt': nlfunc.sqrt,
            # TODO: We'll have to do functional programming things if we want to support
            # any of these...
            'asin': None,
            'sinh': None,
            'asinh': None,
            'acos': None,
            'cosh': None,
            'acosh': None,
            'atan': None,
            'tanh': None,
            'atanh': None,
            'ceil': None,
            'floor': None,
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
        _id = id(child)
        if _id not in visitor.var_map:
            if child.fixed:
                # ESJ TODO: I want the linear walker implementation of
                # check_constant... Could it be in the base class or something?
                return False, visitor.check_constant(child.value, child)
            grb_var = _create_grb_var(
                visitor,
                child,
                name=child.name if visitor.symbolic_solver_labels else "",
            )
            visitor.var_map[_id] = grb_var
        return False, visitor.var_map[_id]

    @staticmethod
    def _before_native_numeric(visitor, child):
        return False, child

    @staticmethod
    def _before_native_logical(visitor, child):
        return False, InvalidNumber(
            child, f"{child!r} ({type(child).__name__}) is not a valid numeric type"
        )

    @staticmethod
    def _before_complex(visitor, child):
        return False, complex_number_error(child, visitor, child)

    @staticmethod
    def _before_invalid(visitor, child):
        return False, InvalidNumber(
            child, f"{child!r} ({type(child).__name__}) is not a valid numeric type"
        )

    @staticmethod
    def _before_string(visitor, child):
        return False, InvalidNumber(
            child, f"{child!r} ({type(child).__name__}) is not a valid numeric type"
        )

    @staticmethod
    def _before_npv(visitor, child):
        try:
            return False, visitor.check_constant(visitor.evaluate(child), child)
        except (ValueError, ArithmeticError):
            return True, None

    @staticmethod
    def _before_param(visitor, child):
        return False, visitor.check_constant(child.value, child)


class GurobiMINLPVisitor(StreamBasedExpressionVisitor):
    before_child_dispatcher = GurobiMINLPBeforeChildDispatcher()

    def __init__(self, grb_model, symbolic_solver_labels=False):
        super().__init__()
        self.grb_model = grb_model
        self.symbolic_solver_labels = symbolic_solver_labels
        self.var_map = {}
        self._named_expressions = {}
        self._eval_expr_visitor = _EvaluationVisitor(True)
        self.evaluate = self._eval_expr_visitor.dfs_postorder_stack

    def initializeWalker(self, expr):
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
        if node.__class__ is EXPR.UnaryFunctionExpression:
            return _function_map[node._name](data[0])
            # import pdb
            # pdb.set_trace()
            # return apply_node_operation(node, data)
        return self._eval_expr_visitor.visit(node, data)

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
        print("Creating Gurobi expression")
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
                Var,
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

        # write constraints
        for cons in components[Constraint]:
            expr, nonlinear, aux = self._create_gurobi_expression(
                cons.body, cons, 0, grb_model, quadratic_visitor, visitor
            )
            if nonlinear:
                grb_model.addConstr(aux == expr)
                expr = aux
            if cons.equality:
                grb_model.addConstr(cons.lower == expr)
            else:
                if cons.lb is not None:
                    grb_model.addConstr(cons.lb <= expr)
                if cons.ub is not None:
                    grb_model.addConstr(cons.ub >= expr)

        return grb_model, visitor.var_map


# ESJ TODO: We should probably not do this and actually tack this on to another
# solver? But I'm not sure. In any case, it should probably at least inerhit
# from another direct interface to Gurobi since all the handling of licenses and
# termination conditions and things should be common.
@SolverFactory.register(
    'gurobi_direct_minlp',
    doc='Direct interface to Gurobi version 12 and up '
    'supporting general nonlinear expressions',
)
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
        # return results
