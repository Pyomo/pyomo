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

from collections import defaultdict
import itertools

import logging

from pyomo.environ import (
    TransformationFactory,
    Transformation,
    Var,
    Constraint,
    Objective,
    Any,
    value,
    BooleanVar,
    Connector,
    Expression,
    Suffix,
    Param,
    Set,
    SetOf,
    RangeSet,
    Block,
    ExternalFunction,
    SortComponents,
    LogicalConstraint,
)
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue, PositiveInt, InEnum
from pyomo.common.dependencies import attempt_import
from pyomo.common.dependencies import numpy as np
from pyomo.common.enums import IntEnum
from pyomo.common.modeling import unique_component_name
from pyomo.core.expr.numeric_expr import SumExpression
from pyomo.core.expr import identify_variables
from pyomo.core.expr import SumExpression
from pyomo.core.util import target_list
from pyomo.contrib.piecewise import PiecewiseLinearExpression, PiecewiseLinearFunction
from pyomo.gdp import Disjunct, Disjunction
from pyomo.network import Port
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.util import ExprType


lineartree, lineartree_available = attempt_import('lineartree')
sklearn_lm, sklearn_available = attempt_import('sklearn.linear_model')

logger = logging.getLogger(__name__)


class DomainPartitioningMethod(IntEnum):
    RANDOM_GRID = 1
    UNIFORM_GRID = 2
    LINEAR_MODEL_TREE_UNIFORM = 3
    LINEAR_MODEL_TREE_RANDOM = 4


class _NonlinearToPWLTransformationData(AutoSlots.Mixin):
    __slots__ = (
        'transformed_component',
        'src_component',
        'transformed_constraints',
        'transformed_objectives',
    )

    def __init__(self):
        self.transformed_component = ComponentMap()
        self.src_component = ComponentMap()
        self.transformed_constraints = defaultdict(ComponentSet)
        self.transformed_objectives = defaultdict(ComponentSet)


Block.register_private_data_initializer(_NonlinearToPWLTransformationData)


def _get_random_point_grid(bounds, n, func, config, seed=42):
    # Generate randomized grid of points
    linspaces = []
    np.random.seed(seed)
    for (lb, ub), is_integer in bounds:
        if not is_integer:
            linspaces.append(np.random.uniform(lb, ub, n))
        else:
            size = min(n, ub - lb + 1)
            linspaces.append(
                np.random.choice(range(lb, ub + 1), size=size, replace=False)
            )
    return list(itertools.product(*linspaces))


def _get_uniform_point_grid(bounds, n, func, config):
    # Generate non-randomized grid of points
    linspaces = []
    for (lb, ub), is_integer in bounds:
        if not is_integer:
            # Issues happen when exactly using the boundary
            nudge = (ub - lb) * 1e-4
            linspaces.append(np.linspace(lb + nudge, ub - nudge, n))
        else:
            size = min(n, ub - lb + 1)
            pts = np.linspace(lb, ub, size)
            linspaces.append(np.array([round(i) for i in pts]))
    return list(itertools.product(*linspaces))


def _get_points_lmt_random_sample(bounds, n, func, config, seed=42):
    points = _get_random_point_grid(bounds, n, func, config, seed=seed)
    return _get_points_lmt(points, bounds, func, config, seed)


def _get_points_lmt_uniform_sample(bounds, n, func, config, seed=42):
    points = _get_uniform_point_grid(bounds, n, func, config)
    return _get_points_lmt(points, bounds, func, config, seed)


def _get_points_lmt(points, bounds, func, config, seed):
    x_list = np.array(points)
    y_list = []

    for point in points:
        y_list.append(func(*point))
    max_depth = config.linear_tree_max_depth
    if max_depth is None:
        # Want the tree to grow with increasing points but not get too large.
        max_depth = max(4, int(np.log2(len(points) / 4)))
    regr = lineartree.LinearTreeRegressor(
        sklearn_lm.LinearRegression(),
        criterion='mse',
        max_bins=120,
        min_samples_leaf=4,
        max_depth=max_depth,
    )
    regr.fit(x_list, y_list)

    leaves, splits, thresholds = _parse_linear_tree_regressor(regr, bounds)

    bound_point_list = _generate_bound_points(leaves, bounds)
    return bound_point_list


_partition_method_dispatcher = {
    DomainPartitioningMethod.RANDOM_GRID: _get_random_point_grid,
    DomainPartitioningMethod.UNIFORM_GRID: _get_uniform_point_grid,
    DomainPartitioningMethod.LINEAR_MODEL_TREE_UNIFORM: _get_points_lmt_uniform_sample,
    DomainPartitioningMethod.LINEAR_MODEL_TREE_RANDOM: _get_points_lmt_random_sample,
}


def _get_pwl_function_approximation(func, config, bounds):
    """
    Get a piecewise-linear approximation of a function, given:

    func: function to approximate
    config: ConfigDict for transformation, specifying domain_partitioning_method,
       num_points, and max_depth (if using linear trees)
    bounds: list of tuples giving upper and lower bounds and a boolean indicating
       if the variable's domain is discrete or not, for each of func's arguments
    """
    method = config.domain_partitioning_method
    n = config.num_points
    points = _partition_method_dispatcher[method](bounds, n, func, config)

    # Don't confuse PiecewiseLinearFunction constructor...
    dim = len(points[0])
    if dim == 1:
        points = [pt[0] for pt in points]

    # After getting the points, construct PWLF using the
    # function-and-list-of-points constructor
    logger.debug(
        f"Constructing PWLF with {len(points)} points, each of which "
        f"are {dim}-dimensional"
    )
    return PiecewiseLinearFunction(points=points, function=func)


# Given a leaves dict (as generated by parse_tree) and a list of tuples
# representing variable bounds, generate the set of vertices separating each
# subset of the domain
def _generate_bound_points(leaves, bounds):
    bound_points = []
    for leaf in leaves.values():
        lower_corner_list = []
        upper_corner_list = []
        for var_bound in leaf['bounds'].values():
            lower_corner_list.append(var_bound[0])
            upper_corner_list.append(var_bound[1])

        for pt in [lower_corner_list, upper_corner_list]:
            for i in range(len(pt)):
                # clamp within bounds range
                pt[i] = max(pt[i], bounds[i][0][0])
                pt[i] = min(pt[i], bounds[i][0][1])

        if tuple(lower_corner_list) not in bound_points:
            bound_points.append(tuple(lower_corner_list))
        if tuple(upper_corner_list) not in bound_points:
            bound_points.append(tuple(upper_corner_list))

    # This process should have gotten every interior bound point. However, all
    # but two of the corners of the overall bounding box should have been
    # missed. Let's fix that now.
    for outer_corner in itertools.product(*[b[0] for b in bounds]):
        if outer_corner not in bound_points:
            bound_points.append(outer_corner)
    return bound_points


# Parse a LinearTreeRegressor and identify features such as bounds, slope, and
# intercept for leaves. Return some dicts.
def _parse_linear_tree_regressor(linear_tree_regressor, bounds):
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
        else:  # traverse its left node by calling function to find all the
            # leaves from its left node
            node['left_leaves'] = _find_leaves(splits, leaves, splits[left_child_node])
        if right_child_node in leaves:  # if right child is a leaf node
            node['right_leaves'].append(right_child_node)
        else:  # traverse its right node by calling function to find all the
            # leaves from its right node
            node['right_leaves'] = _find_leaves(
                splits, leaves, splits[right_child_node]
            )

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

    leaves_new = _reassign_none_bounds(leaves, bounds)
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


# This doesn't catch all additively separable expressions--we really need a
# walker (as does gdp.partition_disjuncts)
def _additively_decompose_expr(input_expr, min_dimension):
    dimension = len(list(identify_variables(input_expr)))
    if input_expr.__class__ is not SumExpression or dimension < min_dimension:
        # This isn't separable or we don't want to separate it, so we just have
        # the one expression
        return [input_expr]
    # else, it was a SumExpression, and we will break it into the summands
    return list(input_expr.args)


# Populate the "None" bounds with the bounding box bounds for a leaves-dict-tree
# amalgamation.
def _reassign_none_bounds(leaves, input_bounds):
    L = np.array(list(leaves.keys()))
    features = np.arange(0, len(leaves[L[0]]['slope']))

    for l in L:
        for f in features:
            if leaves[l]['bounds'][f][0] == None:
                leaves[l]['bounds'][f][0] = input_bounds[f][0][0]
            if leaves[l]['bounds'][f][1] == None:
                leaves[l]['bounds'][f][1] = input_bounds[f][0][1]
    return leaves


def _find_leaves(splits, leaves, input_node):
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
    doc="Convert nonlinear constraints and objectives to piecewise-linear "
    "approximations.",
)
class NonlinearToPWL(Transformation):
    """
    Convert nonlinear constraints and objectives to piecewise-linear approximations.
    """

    CONFIG = ConfigDict('contrib.piecewise.nonlinear_to_pwl')
    CONFIG.declare(
        'targets',
        ConfigValue(
            default=None,
            domain=target_list,
            description="target or list of targets that will be approximated",
            doc="""
            This specifies the list of components to approximate. If None (default),
            the entire model is transformed. Note that if the transformation is
            done out of place, the list of targets should be attached to the model
            before it is cloned, and the list will specify the targets on the cloned
            instance.""",
        ),
    )
    CONFIG.declare(
        'num_points',
        ConfigValue(
            default=3,
            domain=PositiveInt,
            description="Number of breakpoints for each piecewise-linear approximation",
            doc="""
            Specifies the number of points in each function domain to triangulate in 
            order to construct the piecewise-linear approximation. Must be an integer
            greater than 1.""",
        ),
    )
    CONFIG.declare(
        'domain_partitioning_method',
        ConfigValue(
            default=DomainPartitioningMethod.UNIFORM_GRID,
            domain=InEnum(DomainPartitioningMethod),
            description="Method for sampling points that will partition function "
            "domains.",
            doc="""
            The method by which the points used to partition each function domain
            are selected. By default, the range of each variable is partitioned
            uniformly, however it is possible to sample randomly or to use the
            partitions from training a linear model tree based on either uniform
            or random samples of the ranges.""",
        ),
    )
    CONFIG.declare(
        'approximate_quadratic_constraints',
        ConfigValue(
            default=True,
            domain=bool,
            description="Whether or not to approximate quadratic constraints.",
            doc="""
            Whether or not to calculate piecewise-linear approximations for
            quadratic constraints. If True, the resulting approximation will be
            a mixed-integer linear program. If False, the resulting approximation
            will be a mixed-integer quadratic program.""",
        ),
    )
    CONFIG.declare(
        'approximate_quadratic_objectives',
        ConfigValue(
            default=True,
            domain=bool,
            description="Whether or not to approximate quadratic objectives.",
            doc="""
            Whether or not to calculate piecewise-linear approximations for
            quadratic objectives. If True, the resulting approximation will be
            a mixed-integer linear program. If False, the resulting approximation
            will be a mixed-integer quadratic program.""",
        ),
    )
    CONFIG.declare(
        'additively_decompose',
        ConfigValue(
            default=False,
            domain=bool,
            description="Whether or not to additively decompose constraint expressions "
            "and approximate the summands separately.",
            doc="""
            If False, each nonlinear constraint expression will be approximated by
            exactly one piecewise-linear function. If True, constraints will be 
            additively decomposed, and each of the resulting summands will be
            approximated by a separate piecewise-linear function.

            It is recommended to leave this False as long as no nonlinear constraint 
            involves more than about 5-6 variables. For constraints with higher-
            dimmensional nonlinear functions, additive decomposition will improve
            the scalability of the approximation (since partitioning the domain is
            subject to the curse of dimensionality).""",
        ),
    )
    CONFIG.declare(
        'max_dimension',
        ConfigValue(
            default=5,
            domain=PositiveInt,
            description="The maximum dimension of functions that will be approximated.",
            doc="""
            Specifies the maximum dimension function the transformation should
            attempt to approximate. If a nonlinear function dimension exceeds
            'max_dimension' the transformation will log a warning and leave the
            expression as-is. For functions with dimension significantly above the
            default (5), it is likely that this transformation will stall
            triangulating the points in order to partition the function domain.""",
        ),
    )
    CONFIG.declare(
        'min_dimension_to_additively_decompose',
        ConfigValue(
            default=1,
            domain=PositiveInt,
            description="The minimum dimension of functions that will be additively "
            "decomposed.",
            doc="""
            Specifies the minimum dimension of a function that the transformation
            should attempt to additively decompose. If a nonlinear function dimension
            exceeds 'min_dimension_to_additively_decompose' the transformation will
            additively decompose. If a the dimension of an expression is less than
            the 'min_dimension_to_additively_decompose' then it will not be additively
            decomposed""",
        ),
    )
    CONFIG.declare(
        'linear_tree_max_depth',
        ConfigValue(
            default=None,
            domain=PositiveInt,
            description="Maximum depth for linear tree training, used if using a "
            "domain partitioning method based on linear model trees.",
            doc="""
            Only used if 'domain_partitioning_method' is LINEAR_MODEL_TREE_UNIFORM or
            LINEAR_MODEL_TREE_RANDOM: Specifies the maximum depth of the linear model
            trees trained to determine the points to be triangulated to form the 
            domain of the piecewise-linear approximations. If None (the default),
            the max depth will be given as max(4, ln(num_points / 4)).
            """,
        ),
    )

    def __init__(self):
        super(Transformation).__init__()
        self._handlers = {
            Constraint: self._transform_constraint,
            Objective: self._transform_objective,
            Var: False,
            BooleanVar: False,
            Connector: False,
            Expression: False,
            Suffix: False,
            Param: False,
            Set: False,
            SetOf: False,
            RangeSet: False,
            Disjunction: False,
            Disjunct: self._transform_block_components,
            Block: self._transform_block_components,
            ExternalFunction: False,
            Port: False,
            PiecewiseLinearFunction: False,
            LogicalConstraint: False,
        }
        self._transformation_blocks = {}
        self._transformation_block_set = ComponentSet()
        self._quadratic_repn_visitor = QuadraticRepnVisitor(
            subexpression_cache={}, var_map={}, var_order={}, sorter=None
        )

    def _apply_to(self, instance, **kwds):
        try:
            self._apply_to_impl(instance, **kwds)
        finally:
            self._transformation_blocks.clear()
            self._transformation_block_set.clear()

    def _apply_to_impl(self, model, **kwds):
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        targets = config.targets
        if targets is None:
            targets = (model,)

        for target in targets:
            if target.ctype is Block or target.ctype is Disjunct:
                self._transform_block_components(target, config)
            elif target.ctype is Constraint:
                self._transform_constraint(target, config)
            elif target.ctype is Objective:
                self._transform_objective(target, config)
            else:
                raise ValueError(
                    "Target '%s' is not a Block, Constraint, or Objective. It "
                    "is of type '%s' and cannot be transformed."
                    % (target.name, type(target))
                )

    def _get_transformation_block(self, parent):
        if parent in self._transformation_blocks:
            return self._transformation_blocks[parent]

        nm = unique_component_name(parent, '_pyomo_contrib_nonlinear_to_pwl')
        self._transformation_blocks[parent] = transBlock = Block()
        parent.add_component(nm, transBlock)
        self._transformation_block_set.add(transBlock)

        transBlock._pwl_cons = Constraint(Any)
        return transBlock

    def _transform_block_components(self, block, config):
        blocks = block.values() if block.is_indexed() else (block,)
        for b in blocks:
            for obj in b.component_objects(
                active=True, descend_into=False, sort=SortComponents.deterministic
            ):
                if obj in self._transformation_block_set:
                    # This is a Block we created--we know we don't need to look
                    # on it.
                    continue
                handler = self._handlers.get(obj.ctype, None)
                if not handler:
                    if handler is None:
                        raise RuntimeError(
                            "No transformation handler registered for modeling "
                            "components of type '%s'." % obj.ctype
                        )
                    continue
                handler(obj, config)

    def _transform_constraint(self, cons, config):
        trans_block = self._get_transformation_block(cons.parent_block())
        trans_data_dict = trans_block.private_data()
        src_data_dict = cons.parent_block().private_data()
        constraints = cons.values() if cons.is_indexed() else (cons,)
        for c in constraints:
            pw_approx, expr_type = self._approximate_expression(
                c.body, c, trans_block, config, config.approximate_quadratic_constraints
            )

            if pw_approx is None:
                # Didn't need approximated, nothing to do
                continue
            c.model().private_data().transformed_constraints[expr_type].add(c)

            idx = len(trans_block._pwl_cons)
            trans_block._pwl_cons[c.name, idx] = (c.lower, pw_approx, c.upper)
            new_cons = trans_block._pwl_cons[c.name, idx]
            trans_data_dict.src_component[new_cons] = c
            src_data_dict.transformed_component[c] = new_cons

            # deactivate original
            c.deactivate()

    def _transform_objective(self, objective, config):
        trans_block = self._get_transformation_block(objective.parent_block())
        trans_data_dict = trans_block.private_data()
        objectives = objective.values() if objective.is_indexed() else (objective,)
        src_data_dict = objective.parent_block().private_data()
        for obj in objectives:
            pw_approx, expr_type = self._approximate_expression(
                obj.expr,
                obj,
                trans_block,
                config,
                config.approximate_quadratic_objectives,
            )

            if pw_approx is None:
                # Didn't need approximated, nothing to do
                continue
            obj.model().private_data().transformed_objectives[expr_type].add(obj)

            new_obj = Objective(expr=pw_approx, sense=obj.sense)
            trans_block.add_component(
                unique_component_name(trans_block, obj.name), new_obj
            )
            trans_data_dict.src_component[new_obj] = obj
            src_data_dict.transformed_component[obj] = new_obj

            obj.deactivate()

    def _get_bounds_list(self, var_list, obj):
        bounds = []
        for v in var_list:
            if None in v.bounds:
                raise ValueError(
                    "Cannot automatically approximate constraints with unbounded "
                    "variables. Var '%s' appearing in component '%s' is missing "
                    "at least one bound" % (v.name, obj.name)
                )
            else:
                bounds.append((v.bounds, v.is_integer()))
        return bounds

    def _needs_approximating(self, expr, approximate_quadratic):
        repn = self._quadratic_repn_visitor.walk_expression(expr)
        if repn.nonlinear is None:
            if repn.quadratic is None:
                # Linear constraint. Always skip.
                return ExprType.LINEAR, False
            else:
                if not approximate_quadratic:
                    # Didn't need approximated, nothing to do
                    return ExprType.QUADRATIC, False
                return ExprType.QUADRATIC, True
        return ExprType.GENERAL, True

    def _approximate_expression(
        self, expr, obj, trans_block, config, approximate_quadratic
    ):
        expr_type, needs_approximating = self._needs_approximating(
            expr, approximate_quadratic
        )
        if not needs_approximating:
            return None, expr_type

        # Additively decompose expr and work on the pieces
        pwl_summands = []
        for k, subexpr in enumerate(
            _additively_decompose_expr(
                expr, config.min_dimension_to_additively_decompose
            )
            if config.additively_decompose
            else (expr,)
        ):
            # First check if this is a good idea
            expr_vars = list(identify_variables(subexpr, include_fixed=False))
            orig_values = ComponentMap((v, v.value) for v in expr_vars)

            dim = len(expr_vars)
            if dim > config.max_dimension:
                raise ValueError(
                    "Not approximating expression for component '%s' as "
                    "it exceeds the maximum dimension of %s. Try increasing "
                    "'max_dimension' or additively separating the expression."
                    % (obj.name, config.max_dimension)
                )
                pwl_summands.append(subexpr)
                continue
            elif not self._needs_approximating(subexpr, approximate_quadratic)[1]:
                pwl_summands.append(subexpr)
                continue
            # else we approximate subexpr

            def eval_expr(*args):
                for i, v in enumerate(expr_vars):
                    v.value = args[i]
                return value(subexpr)

            pwlf = _get_pwl_function_approximation(
                eval_expr, config, self._get_bounds_list(expr_vars, obj)
            )
            name = unique_component_name(
                trans_block, obj.getname(fully_qualified=False)
            )
            trans_block.add_component(f"_pwle_{name}_{k}", pwlf)
            # NOTE: We are *not* using += because it will hit the NamedExpression
            # implementation of iadd and dereference the ExpressionData holding
            # the PiecewiseLinearExpression that we later transform my remapping
            # it to a Var...
            pwl_summands.append(pwlf(*expr_vars))

            # restore var values
            for v, val in orig_values.items():
                v.value = val

        return sum(pwl_summands), expr_type

    def get_src_component(self, cons):
        data = cons.parent_block().private_data().src_component
        if cons in data:
            return data[cons]
        else:
            raise ValueError(
                "It does not appear that '%s' is a transformed Constraint "
                "created by the 'nonlinear_to_pwl' transformation." % cons.name
            )

    def get_transformed_component(self, cons):
        data = cons.parent_block().private_data().transformed_component
        if cons in data:
            return data[cons]
        else:
            raise ValueError(
                "It does not appear that '%s' is a Constraint that was "
                "transformed by the 'nonlinear_to_pwl' transformation." % cons.name
            )

    def get_transformed_nonlinear_constraints(self, model):
        """
        Given a model that has been transformed with contrib.piecewise.nonlinear_to_pwl,
        return the list of general (not quadratic) nonlinear Constraints that were
        approximated with PiecewiseLinearFunctions
        """
        return model.private_data().transformed_constraints[ExprType.GENERAL]

    def get_transformed_quadratic_constraints(self, model):
        """
        Given a model that has been transformed with contrib.piecewise.nonlinear_to_pwl,
        return the list of quadratic Constraints that were approximated with
        PiecewiseLinearFunctions
        """
        return model.private_data().transformed_constraints[ExprType.QUADRATIC]

    def get_transformed_nonlinear_objectives(self, model):
        """
        Given a model that has been transformed with contrib.piecewise.nonlinear_to_pwl,
        return the list of general (not quadratic) nonlinear Constraints that were
        approximated with PiecewiseLinearFunctions
        """
        return model.private_data().transformed_objectives[ExprType.GENERAL]

    def get_transformed_quadratic_objectives(self, model):
        """
        Given a model that has been transformed with contrib.piecewise.nonlinear_to_pwl,
        return the list of quadratic Constraints that were approximated with
        PiecewiseLinearFunctions
        """
        return model.private_data().transformed_objectives[ExprType.QUADRATIC]
