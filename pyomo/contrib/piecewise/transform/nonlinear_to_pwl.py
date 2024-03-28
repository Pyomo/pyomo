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

import itertools

from lineartree import LinearTreeRegressor
import lineartree

import numpy as np

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
from pyomo.contrib.piecewise import (
    PiecewiseLinearExpression,
    PiecewiseLinearFunction
)

from sklearn.linear_model import LinearRegression
import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


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
    for (lb, ub) in bounds:
        np.random.seed(seed)
        linspaces.append(np.random.uniform(lb, ub, n))
    return list(itertools.product(*linspaces))


def get_simple_uniform_point_grid(bounds, n):
    # Generate non-randomized grid of points
    linspaces = []
    for (lb, ub) in bounds:
        # Issues happen when exactly using the boundary
        nudge = (ub - lb) * 1e-4
        linspaces.append(
            # np.linspace(b[0], b[1], n)
            np.linspace(lb + nudge, ub - nudge, n)
        )
    return list(itertools.product(*linspaces))


# TODO this was copypasted from shumeng; make it better
def get_points_naive_lmt(bounds, n, func, seed=42, randomize=True):
    
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

    leaves, splits, ths = parse_linear_tree_regressor(regr, bounds)

    # This was originally part of the LMT_Model_component and used to calculate
    # avg_leaves for the output data. TODO: get this back
    # self.total_leaves += len(leaves)

    # bound_point_list = lmt.generate_bound(leaves)
    bound_point_list = generate_bound_points(leaves, bounds)
    # duct tape to fix possible issues from unknown bugs. TODO should this go
    # here?
    return bound_point_list


# TODO: this is still horrible. Maybe I should put these back together into
# a wrapper class again, but better this time?


# Given a leaves dict (as generated by parse_tree) and a list of tuples
# representing variable bounds, generate the set of vertices separating each
# subset of the domain
def generate_bound_points(leaves, bounds):
    bound_points = []
    for leaf in leaves.values():
        lower_corner_list = []
        upper_corner_list = []
        for var_bound in leaf['bounds'].values():
            lower_corner_list.append(var_bound[0])
            upper_corner_list.append(var_bound[1])
        
        # Duct tape to fix issues from unknown bugs
        for pt in [lower_corner_list, upper_corner_list]:
            for i in range(len(pt)):
                # clamp within bounds range
                pt[i] = max(pt[i], bounds[i][0])
                pt[i] = min(pt[i], bounds[i][1])

        if tuple(lower_corner_list) not in bound_points:
            bound_points.append(tuple(lower_corner_list))
        if tuple(upper_corner_list) not in bound_points:
            bound_points.append(tuple(upper_corner_list))

    # This process should have gotten every interior bound point. However, all
    # but two of the corners of the overall bounding box should have been
    # missed. Let's fix that now.
    for outer_corner in itertools.product(*bounds):
        if outer_corner not in bound_points:
            bound_points.append(outer_corner)
    return bound_points


# Parse a LinearTreeRegressor and identify features such as bounds, slope, and
# intercept for leaves. Return some dicts.
def parse_linear_tree_regressor(linear_tree_regressor, bounds):
    leaves = linear_tree_regressor.summary(only_leaves=True)
    splits = linear_tree_regressor.summary()

    for key, leaf in leaves.items():
        del splits[key]
        leaf['bounds'] = {}
        leaf['slope'] = list(leaf['models'].coef_)
        leaf['intercept'] = leaf['models'].intercept_

    L = np.array(list(leaves.keys()))
    features = np.arange(0, len(leaves[L[0]]['slope']))

    for node in splits.values():
        left_child_node = node['children'][0]  # find its left child
        right_child_node = node['children'][1]  # find its right child
        # create the list to save leaves
        node['left_leaves'], node['right_leaves'] = [], []
        if left_child_node in leaves:  # if left child is a leaf node
            node['left_leaves'].append(left_child_node)
        else:  # traverse its left node by calling function to find all the leaves from its left node
            node['left_leaves'] = find_leaves(splits, leaves, splits[left_child_node])
        if right_child_node in leaves:  # if right child is a leaf node
            node['right_leaves'].append(right_child_node)
        else:  # traverse its right node by calling function to find all the leaves from its right node
            node['right_leaves'] = find_leaves(splits, leaves, splits[right_child_node])

    # For each feature in each leaf, initialize lower and upper bounds to None
    for th in features:
        for leaf in leaves:
            leaves[leaf]['bounds'][th] = [None, None]
    for split in splits:
        var = splits[split]['col']
        for leaf in splits[split]['left_leaves']:
            leaves[leaf]['bounds'][var][1] = splits[split]['th']

        for leaf in splits[split]['right_leaves']:
            leaves[leaf]['bounds'][var][0] = splits[split]['th']

    leaves_new = reassign_none_bounds(leaves, bounds)
    splitting_thresholds = {}
    for split in splits:
        var = splits[split]['col']
        splitting_thresholds[var] = {}
    for split in splits:
        var = splits[split]['col']
        splitting_thresholds[var][split] = splits[split]['th']
    # Make sure every nested dictionary in the splitting_thresholds dictionary
    # is sorted by value
    for var in splitting_thresholds:
        splitting_thresholds[var] = dict(
            sorted(splitting_thresholds[var].items(), key=lambda x: x[1])
        )

    return leaves_new, splits, splitting_thresholds


# Populate the "None" bounds with the bounding box bounds for a leaves-dict-tree
# amalgamation.
def reassign_none_bounds(leaves, input_bounds):
    L = np.array(list(leaves.keys()))
    features = np.arange(0, len(leaves[L[0]]['slope']))

    for l in L:
        for f in features:
            if leaves[l]['bounds'][f][0] == None:
                leaves[l]['bounds'][f][0] = input_bounds[f][0]
            if leaves[l]['bounds'][f][1] == None:
                leaves[l]['bounds'][f][1] = input_bounds[f][1]
    return leaves


def find_leaves(splits, leaves, input_node):
    root_node = input_node
    leaves_list = []
    queue = [root_node]
    while queue:
        node = queue.pop()
        node_left = node['children'][0]
        node_right = node['children'][1]
        if node_left in leaves:
            leaves_list.append(node_left)
        else:
            queue.append(splits[node_left])
        if node_right in leaves:
            leaves_list.append(node_right)
        else:
            queue.append(splits[node_right])
    return leaves_list


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
    # TODO: ConfigDict
    def _apply_to(
        self,
        model,
        n=3,
        method='simple_uniform_point_grid',
        allow_quadratic_cons=True,
        allow_quadratic_objs=True,
        additively_decompose=True,
    ):
        """TODO: docstring"""

        # Upcoming steps will trash the values of the vars, since I don't know
        # a better way. But what if the user set them with initialize= ? We'd
        # better restore them after we're done.
        orig_var_map = {id(var): var.value for var in model.component_data_objects(Var)}

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
        for var in model.component_data_objects(Var):
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


def _generate_bounds_list(vars_inner, con):
    bounds = []
    for v in vars_inner:
        if v.fixed:
            bounds.append((value(v), value(v)))
        elif None in v.bounds:
            raise ValueError(
                "Cannot automatically approximate constraints with unbounded "
                "variables. Var '%s' appearining in component '%s' is missing "
                "at least one bound" % (con.name, v.name))
        else:
            bounds.append(v.bounds)
    return bounds


def _replace_con(model, con, method, n, allow_quadratic_cons, additively_decompose):
    # Additively decompose con.body and work on the pieces
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
        bounds = _generate_bounds_list(vars_inner, con)

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
        bounds = _generate_bounds_list(vars_inner, obj)

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
