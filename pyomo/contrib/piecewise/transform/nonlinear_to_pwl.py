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

import numpy as np
import itertools

from pyomo.environ import (
    TransformationFactory,
    Transformation,
    Var,
    Constraint,
    Objective,
    Any,
    value,
)
from pyomo.core.expr.numeric_expr import SumExpression
from pyomo.core.expr import identify_variables
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.core.expr import SumExpression
from pyomo.contrib.piecewise import PiecewiseLinearExpression


# TODO remove
MAX_DIM = 5

# This should be safe to use many times; declare it globally
_quadratic_repn_visitor = QuadraticRepnVisitor(
    subexpression_cache={}, var_map={}, var_order={}, sorter=None
)


def get_pwl_function_approximation(func, method, n, bounds, **kwargs):
    """
    Get a piecewise-linear approximation to a function, given:

    func: function to approximate
    method: method to use for the approximation, current options are:
        - 'simple_random_point_grid'
        - 'simple_uniform_point_grid'
        - 'naive_lmt'
    n: parameter controlling fineness of the approximation based on the specified method
    bounds: list of tuples giving upper and lower bounds for each of func's arguments
    kwargs: additional arguments to be specified to the method used
    """

    points = None
    match (method):
        case 'simple_random_point_grid':
            points = get_simple_random_point_grid(bounds, n)
        case 'simple_uniform_point_grid':
            points = get_simple_uniform_point_grid(bounds, n)
        case 'naive_lmt':
            points = get_points_naive_lmt(bounds, n, func, randomize=True)
        case 'naive_lmt_uniform':
            points = get_points_naive_lmt(bounds, n, func, randomize=False)
        case _:
            raise NotImplementedError(f"Invalid method: {method}")

    # Default path: after getting the points, construct PWLF using the
    # function-and-list-of-points constructor

    # DUCT TAPE WARNING: work around deficiency in PiecewiseLinearFunction constructor. TODO
    dim = len(points[0])
    if dim == 1:
        points = [pt[0] for pt in points]

    print(
        f"    Constructing PWLF with {len(points)} points, each of which are {dim}-dimensional"
    )
    return PiecewiseLinearFunction(points=points, function=func)


def get_simple_random_point_grid(bounds, n, seed=42):
    # Generate randomized grid of points
    linspaces = []
    for b in bounds:
        np.random.seed(seed)
        linspaces.append(np.random.uniform(b[0], b[1], n))
    return list(itertools.product(*linspaces))


def get_simple_uniform_point_grid(bounds, n):
    # Generate non-randomized grid of points
    linspaces = []
    for b in bounds:
        # Issues happen when exactly using the boundary
        nudge = (b[1] - b[0]) * 1e-4
        linspaces.append(
            # np.linspace(b[0], b[1], n)
            np.linspace(b[0] + nudge, b[1] - nudge, n)
        )
    return list(itertools.product(*linspaces))


# TODO this was copypasted from shumeng; make it better
def get_points_naive_lmt(bounds, n, func, seed=42, randomize=True):
    from lineartree import LinearTreeRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    import PWLTransformation.lmt as lmtutils

    points = None
    if randomize:
        points = get_simple_random_point_grid(bounds, n, seed=seed)
    else:
        points = get_simple_uniform_point_grid(bounds, n)
        # perturb(points, 0.01)
    x_list = np.array(points)
    y_list = []
    for point in points:
        y_list.append(func(*point))
    regr = LinearTreeRegressor(
        LinearRegression(),
        criterion='mse',
        max_bins=120,
        min_samples_leaf=4,
        max_depth=5,
    )

    # Using train_test_split is silly. TODO: remove this and just sample my own
    # extra points if I want to estimate the error.
    X_train, X_test, y_train, y_test = train_test_split(
        x_list, y_list, test_size=0.2, random_state=seed
    )
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    error = mean_squared_error(y_test, y_pred)

    leaves, splits, ths = lmtutils.parse_linear_tree_regressor(regr, bounds)

    # This was originally part of the LMT_Model_component and used to calculate
    # avg_leaves for the output data. TODO: get this back
    # self.total_leaves += len(leaves)

    # bound_point_list = lmt.generate_bound(leaves)
    bound_point_list = lmtutils.generate_bound_points(leaves, bounds)
    # duct tape to fix possible issues from unknown bugs. TODO should this go
    # here?
    return bound_point_list


@TransformationFactory.register(
    'contrib.piecewise.nonlinear_to_pwl',
    doc="Convert nonlinear constraints and objectives to piecewise-linear approximations.",
)
class NonlinearToPWL(Transformation):
    """
    Convert nonlinear constraints and objectives to piecewise-linear approximations.
    """

    def __init__(self):
        super(Transformation).__init__()

    def _apply_to(
        self,
        model,
        n=3,
        method='simple_uniform_point_grid',
        allow_quadratic_cons=True,
        allow_quadratic_objs=True,
        additively_decompose=True,
    ):
        """Apply the transformation"""

        # Check ahead of time whether there are any unbounded variables. If
        # there are, we'll have to bail out
        # But deactivated variables can be left alone -- or should they be?
        # Let's not, for now.
        for v in model.component_objects(Var):
            if None in v.bounds:
                print(
                    "Error: cannot apply transformation to model with unbounded variables"
                )
                raise NotImplementedError(
                    "Cannot apply transformation to model with unbounded variables"
                )

        # Upcoming steps will trash the values of the vars, since I don't know
        # a better way. But what if the user set them with initialize= ? We'd
        # better restore them after we're done.
        orig_var_map = {id(var): var.value for var in model.component_objects(Var)}

        # Now we are ready to start
        original_cons = list(model.component_data_objects(Constraint))
        original_objs = list(model.component_data_objects(Objective))

        model._pwl_quadratic_count = 0
        model._pwl_nonlinear_count = 0

        # Let's put all our new constraints in one big index
        model._pwl_cons = Constraint(Any)

        for con in original_cons:
            repn = _quadratic_repn_visitor.walk_expression(con.body)
            if repn.nonlinear is None:
                if repn.quadratic is None:
                    # Linear constraint. Always skip.
                    continue
                else:
                    model._pwl_quadratic_count += 1
                    if allow_quadratic_cons:
                        continue
            else:
                model._pwl_nonlinear_count += 1
            _replace_con(
                model, con, method, n, allow_quadratic_cons, additively_decompose
            )

        # And do the same for objectives
        for obj in original_objs:
            repn = _quadratic_repn_visitor.walk_expression(obj)
            if repn.nonlinear is None:
                if repn.quadratic is None:
                    # Linear objective. Skip.
                    continue
                else:
                    model._pwl_quadratic_count += 1
                    if allow_quadratic_objs:
                        continue
            else:
                model._pwl_nonlinear_count += 1
            _replace_obj(
                model, obj, method, n, allow_quadratic_objs, additively_decompose
            )

        # Before we're done, replace the old variable values
        for var in model.component_objects(Var):
            var.value = orig_var_map[id(var)]


# Check whether a term should be skipped for approximation. Do not touch
# model's quadratic or nonlinear counts; those are only for top-level
# expressions which were already checked
def _check_skip_approx(expr, allow_quadratic, model):
    repn = _quadratic_repn_visitor.walk_expression(expr)
    if repn.nonlinear is None:
        if repn.quadratic is None:
            # Linear expression. Skip.
            return True
        else:
            # model._pwl_quadratic_count += 1
            if allow_quadratic:
                return True
    else:
        pass
        # model._pwl_nonlinear_count += 1
    dim = len(list(identify_variables(expr)))
    if dim > MAX_DIM:
        print(f"Refusing to approximate function with {dim}-dimensional component.")
        raise RuntimeError(
            f"Refusing to approximate function with {dim}-dimensional component."
        )
    return False


def _replace_con(model, con, method, n, allow_quadratic_cons, additively_decompose):
    vars = list(identify_variables(con.body))
    bounds = [(v.bounds[0], v.bounds[1]) for v in vars]

    # Alright, let's do it like this. Additively decompose con.body and work on the pieces
    func_pieces = []
    for k, expr in enumerate(
        _additively_decompose_expr(con.body) if additively_decompose else [con.body]
    ):
        # First, check if we actually need to do anything
        if _check_skip_approx(expr, allow_quadratic_cons, model):
            # We're skipping this term. Just add expr directly to the pieces
            func_pieces.append(expr)
            continue

        vars_inner = list(identify_variables(expr))
        bounds = [(v.bounds[0], v.bounds[1]) for v in vars_inner]

        def eval_con_func(*args):
            # sanity check
            assert len(args) == len(
                vars_inner
            ), f"eval_con_func was called with {len(args)} arguments, but expected {len(vars_inner)}"
            for i, v in enumerate(vars_inner):
                v.value = args[i]
            return value(con.body)

        pwlf = get_pwl_function_approximation(eval_con_func, method, n, bounds)

        con_name = con.getname(fully_qualified=False)
        model.add_component(f"_pwle_{con_name}_{k}", pwlf)
        # func_pieces.append(pwlf(*vars_inner).expr)
        func_pieces.append(pwlf(*vars_inner))

    pwl_func = sum(func_pieces)

    # Change the constraint. This is hard to do in-place, so I'll
    # remake it and deactivate the old one as was done originally.

    # Now we need a ton of if statements to properly set up the constraint
    if con.equality:
        model._pwl_cons[str(con)] = pwl_func == con.ub
    elif con.strict_lower:
        model._pwl_cons[str(con)] = pwl_func > con.lb
    elif con.strict_upper:
        model._pwl_cons[str(con)] = pwl_func < con.ub
    elif con.has_lb():
        if con.has_ub():  # constraint is of the form lb <= expr <= ub
            model._pwl_cons[str(con)] = (con.lb, pwl_func, con.ub)
        else:
            model._pwl_cons[str(con)] = pwl_func >= con.lb
    elif con.has_ub():
        model._pwl_cons[str(con)] = pwl_func <= con.ub
    else:
        assert (
            False
        ), f"unreachable: original Constraint '{con_name}' did not have any upper or lower bound"
    con.deactivate()


def _replace_obj(model, obj, method, n, allow_quadratic_obj, additively_decompose):
    vars = list(identify_variables(obj))
    bounds = [(v.bounds[0], v.bounds[1]) for v in vars]

    func_pieces = []
    for k, expr in enumerate(
        _additively_decompose_expr(obj.expr) if additively_decompose else [obj.expr]
    ):
        # First, check if we actually need to do anything
        if _check_skip_approx(expr, allow_quadratic_obj, model):
            # We're skipping this term. Just add expr directly to the pieces
            func_pieces.append(expr)
            continue

        vars_inner = list(identify_variables(expr))
        bounds = [(v.bounds[0], v.bounds[1]) for v in vars_inner]

        def eval_obj_func(*args):
            # sanity check
            assert len(args) == len(
                vars_inner
            ), f"eval_obj_func was called with {len(args)} arguments, but expected {len(vars_inner)}"
            for i, v in enumerate(vars_inner):
                v.value = args[i]
            return value(obj)

        pwlf = get_pwl_function_approximation(eval_obj_func, method, n, bounds)

        obj_name = obj.getname(fully_qualified=False)
        model.add_component(f"_pwle_{obj_name}_{k}", pwlf)
        func_pieces.append(pwlf(*vars_inner))

    pwl_func = sum(func_pieces[1:], func_pieces[0])

    # Add the new objective
    obj_name = obj.getname(fully_qualified=False)
    # model.add_component(f"_pwle_{obj_name}", pwl_func)
    model.add_component(
        f"_pwl_obj_{obj_name}", Objective(expr=pwl_func, sense=obj.sense)
    )
    obj.deactivate()


# Copypasted from gdp/plugins/partition_disjuncts.py for now. This is the
# stupid approach that will not properly catch all additive separability; to do
# it better we need a walker.
def _additively_decompose_expr(input_expr):
    if input_expr.__class__ is not SumExpression:
        # print(f"couldn't decompose: input_expr.__class__ was {input_expr.__class__}, not SumExpression")
        # This isn't separable, so we just have the one expression
        return [input_expr]
    # else, it was a SumExpression, and we will break it into the summands
    summands = list(input_expr.args)
    # print(f"len(summands) is {len(summands)}")
    # print(f"summands is {summands}")
    return summands
