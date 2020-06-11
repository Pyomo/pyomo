#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# this module contains various utilities for creating PH weighted penalty
# objectives, e.g., through quadratic or linearized penalty terms.

from math import fabs, log, exp
from six.moves import xrange
import sys

from pyomo.core import Set, Constraint, Expression, BooleanSet, value
from pyomo.pysp.phutils import indexToString

# IMPT: In general, the breakpoint computation codes can return a
#       2-list even if the lb equals the ub. This case happens quite
#       often in real models (although typically lb=xvag=ub).  See the
#       code for constructing the pieces on how this case is handled
#       in the linearization.

#
# routine to compute linearization breakpoints uniformly between the
# bounds and the mean.
#

def compute_uniform_breakpoints(lb, node_min, xavg, node_max, ub, num_breakpoints_per_side, tolerance):

    breakpoints = []

    # add the lower bound - the first breakpoint.
    breakpoints.append(lb)

    # determine the breakpoints to the left of the mean.
    left_step = (xavg - lb) / num_breakpoints_per_side
    current_x = lb
    for i in range(1,num_breakpoints_per_side+1):
        this_lb = current_x
        this_ub = current_x+left_step
        if (fabs(this_lb-lb) > tolerance) and (fabs(this_lb-xavg) > tolerance):
            breakpoints.append(this_lb)
        current_x += left_step

    # add the mean - it's always a breakpoint. unless!
    # the lb or ub and the avg are the same.
    if (fabs(lb-xavg) > tolerance) and (fabs(ub-xavg) > tolerance):
        breakpoints.append(xavg)

    # determine the breakpoints to the right of the mean.
    right_step = (ub - xavg) / num_breakpoints_per_side
    current_x = xavg
    for i in range(1,num_breakpoints_per_side+1):
        this_lb = current_x
        this_ub = current_x+right_step
        if (fabs(this_ub-xavg) > tolerance) and (fabs(this_ub-ub) > tolerance):
            breakpoints.append(this_ub)
        current_x += right_step

    # add the upper bound - the last breakpoint.
    # the upper bound should always be different than the lower bound (I say with some
    # hesitation - it really depends on what plugins are doing to modify the bounds dynamically).
    breakpoints.append(ub)

    return breakpoints

#
# routine to compute linearization breakpoints uniformly between the current node min/max bounds.
#

def compute_uniform_between_nodestat_breakpoints(lb, node_min, xavg, node_max, ub, num_breakpoints, tolerance):

    breakpoints = []

    # add the lower bound - the first breakpoint.
    breakpoints.append(lb)

    # add the node-min - the second breakpoint. but only if it is different than the lower bound and the mean.
    if (fabs(node_min-lb) > tolerance) and (fabs(node_min-xavg) > tolerance):
        breakpoints.append(node_min)

    step = (node_max - node_min) / num_breakpoints
    current_x = node_min
    for i in range(1,num_breakpoints+1):
        this_lb = current_x
        this_ub = current_x+step
        if (fabs(this_lb-node_min) > tolerance) and (fabs(this_lb-node_max) > tolerance) and (fabs(this_lb-xavg) > tolerance):
            breakpoints.append(this_lb)
        current_x += step

    # add the node-max - the second-to-last breakpoint. but only if it is different than the upper bound and the mean.
    if (fabs(node_max-ub) > tolerance) and (fabs(node_max-xavg) > tolerance):
        breakpoints.append(node_max)

    # add the upper bound - the last breakpoint.
    breakpoints.append(ub)

    # add the mean - it's always a breakpoint. unless! -
    # it happens to be equal to (within tolerance) the lower or upper bounds.
    # sort to insert it in the correct place.
    if (fabs(xavg - lb) > tolerance) and (fabs(xavg - ub) > tolerance):
        breakpoints.append(xavg)
    breakpoints.sort()

    return breakpoints

#
# routine to compute linearization breakpoints using "Woodruff" relaxation of the compute_uniform_between_nodestat_breakpoints.
#

def compute_uniform_between_woodruff_breakpoints(lb, node_min, xavg, node_max, ub, num_breakpoints, tolerance):

    breakpoints = []

    # add the lower bound - the first breakpoint.
    breakpoints.append(lb)

    # be either three "differences" from the mean, or else "halfway to the bound", whichever is closer to the mean.
    left = max(xavg - 3.0 * (xavg - node_min), xavg - 0.5 * (xavg - lb))
    right = min(xavg + 3.0 * (node_max - xavg), xavg + 0.5 * (ub - xavg))

    # add the left bound - the second breakpoint. but only if it is different than the lower bound and the mean.
    if (fabs(left-lb) > tolerance) and (fabs(left-xavg) > tolerance):
        breakpoints.append(left)

    step = (right - left) / num_breakpoints
    current_x = left
    for i in range(1,num_breakpoints+1):
        this_lb = current_x
        this_ub = current_x+step
        if (fabs(this_lb-left) > tolerance) and (fabs(this_lb-right) > tolerance) and (fabs(this_lb-xavg) > tolerance):
            breakpoints.append(this_lb)
        current_x += step

    # add the right bound - the second-to-last breakpoint. but only if it is different than the upper bound and the mean.
    if (fabs(right-ub) > tolerance) and (fabs(right-xavg) > tolerance):
        breakpoints.append(right)

    # add the upper bound - the last breakpoint.
    breakpoints.append(ub)

    # add the mean - it's always a breakpoint.
    # sort to insert it in the correct place.
    breakpoints.append(xavg)
    breakpoints.sort()

    return breakpoints

#
# routine to compute linearization breakpoints based on an exponential distribution from the mean in each direction.
#

def compute_exponential_from_mean_breakpoints(lb, node_min, xavg, node_max, ub, num_breakpoints_per_side, tolerance):

    breakpoints = []

    # add the lower bound - the first breakpoint.
    breakpoints.append(lb)

    # determine the breakpoints to the left of the mean.
    left_delta = xavg - lb
    base = exp(log(left_delta) / num_breakpoints_per_side)
    current_offset = base
    for i in range(1,num_breakpoints_per_side+1):
        current_x = xavg - current_offset
        if (fabs(current_x-lb) > tolerance) and (fabs(current_x-xavg) > tolerance):
            breakpoints.append(current_x)
        current_offset *= base

    # add the mean - it's always a breakpoint.
    breakpoints.append(xavg)

    # determine the breakpoints to the right of the mean.
    right_delta = ub - xavg
    base = exp(log(right_delta) / num_breakpoints_per_side)
    current_offset = base
    for i in range(1,num_breakpoints_per_side+1):
        current_x = xavg + current_offset
        if (fabs(current_x-xavg) > tolerance) and (fabs(current_x-ub) > tolerance):
            breakpoints.append(current_x)
        current_offset *= base

    # add the upper bound - the last breakpoint.
    breakpoints.append(ub)

    return breakpoints

#
# a utility to create piece-wise linear constraint expressions for a given variable, for
# use in constructing the augmented (penalized) PH objective. lb and ub are the bounds
# on this piece, variable is the actual instance variable, and average is the instance
# parameter specifying the average of this variable across instances sharing passing
# through a common tree node. lb and ub are floats.
# IMPT: There are cases where lb=ub, in which case the slope is 0 and the intercept
#       is simply the penalty at the lower(or upper) bound.
#

def create_piecewise_constraint_tuple(lb, ub, instance_variable, variable_average, quad_variable, tolerance):

    penalty_at_lb = (lb - variable_average) * (lb - variable_average)
    penalty_at_ub = (ub - variable_average) * (ub - variable_average)
    slope = None
    if fabs(ub-lb) > tolerance:
        slope = (penalty_at_ub - penalty_at_lb) / (ub - lb)
    else:
        slope = 0.0
    intercept = penalty_at_lb - slope * lb
    expression = (0.0, quad_variable - slope * instance_variable - intercept, None)

    return expression

#
# Add the PH weight terms to the objective, guided by various options.
#

def add_ph_objective_weight_terms(instance_name, instance, scenario_tree):

    scenario = scenario_tree.get_scenario(instance.name)

    # cache for efficiency purposes.
    objective = scenario._instance_objective

    # linear weight penalty expressions.
    weight_expression = 0.0 # indicates unassigned

    nodeid_to_vardata_map = instance._ScenarioTreeSymbolMap.bySymbol

    for tree_node in scenario._node_list[:-1]:

        w_parameter_name = "PHWEIGHT_"+str(tree_node._name)
        w_parameter = instance.find_component(w_parameter_name)

        for variable_id in tree_node._variable_ids:

            # don't add weight terms for derived variables at the tree node.
            if variable_id in tree_node._derived_variable_ids:
                continue

            instance_vardata = nodeid_to_vardata_map[variable_id]

            if instance_vardata.fixed is False:
                # add the linear (w-weighted) term in a consistent fashion, independent of variable type.
                # don't adjust the sign of the weight here - that has already been accounted for in the main PH routine.
                # TODO: Should blend_paramter be used in this expression?
                weight_expression += w_parameter[variable_id] * instance_vardata

    weight_expression_component = instance.PHWEIGHT_EXPRESSION = Expression(initialize=weight_expression)

    # augment the original objective with the new linear terms.
    objective.expr += weight_expression_component

    return weight_expression_component, weight_expression

#
# Add the PH proximal terms to the objective, guided by various options.
#

def add_ph_objective_proximal_terms(instance_name,
                                    instance, scenario_tree,
                                    linearize_nonbinary_penalty_terms,
                                    retain_quadratic_binary_terms):

    scenario = scenario_tree.get_scenario(instance.name)

    # cache for efficiency purposes.
    objective = scenario._instance_objective
    is_minimizing = objective.is_minimizing()

    # proximal penalty expressions.
    proximal_expression = 0.0 # indicates unassigned

    nodeid_to_vardata_map = instance._ScenarioTreeSymbolMap.bySymbol

    for tree_node in scenario._node_list[:-1]:

        xbar_parameter_name = "PHXBAR_"+str(tree_node._name)
        xbar_parameter = instance.find_component(xbar_parameter_name)

        rho_parameter_name = "PHRHO_"+str(tree_node._name)
        rho_parameter = instance.find_component(rho_parameter_name)

        blend_parameter_name = "PHBLEND_"+str(tree_node._name)
        blend_parameter = instance.find_component(blend_parameter_name)

        quad_penalty_term_variable = None
        # if linearizing, then we have previously defined a variable
        # associated with the result of the linearized approximation
        # of the penalty term - this is simply added to the objective
        # function.
        if linearize_nonbinary_penalty_terms > 0:
            quad_penalty_term_variable_name = "PHQUADPENALTY_"+str(tree_node._name)
            quad_penalty_term_variable = instance.find_component(quad_penalty_term_variable_name)

        for variable_id in tree_node._variable_ids:

            # don't add weight terms for derived variables at the tree node.
            if variable_id in tree_node._derived_variable_ids:
                continue

            instance_vardata = nodeid_to_vardata_map[variable_id]

            if instance_vardata.fixed is False:

                # deal with binaries
                if isinstance(instance_vardata.domain, BooleanSet):

                    if retain_quadratic_binary_terms is False:
                        if is_minimizing:
                            proximal_expression += (blend_parameter[variable_id] * rho_parameter[variable_id] / 2.0 * (instance_vardata - 2.0 * xbar_parameter[variable_id] * instance_vardata + xbar_parameter[variable_id] * xbar_parameter[variable_id]))
                        else:
                            proximal_expression -= (blend_parameter[variable_id] * rho_parameter[variable_id] / 2.0 * (instance_vardata - 2.0 * xbar_parameter[variable_id] * instance_vardata + xbar_parameter[variable_id] * xbar_parameter[variable_id]))
                    else:
                        if is_minimizing:
                            proximal_expression += (blend_parameter[variable_id] * (rho_parameter[variable_id] / 2.0) * (instance_vardata - xbar_parameter[variable_id]) * (instance_vardata - xbar_parameter[variable_id]))
                        else:
                            proximal_expression -= (blend_parameter[variable_id] * (rho_parameter[variable_id] / 2.0) * (instance_vardata - xbar_parameter[variable_id]) * (instance_vardata - xbar_parameter[variable_id]))

                # deal with everything else
                else:

                    if linearize_nonbinary_penalty_terms > 0:

                        # the variables are easy - just a single entry.
                        # GAH: Should blend_paramter be used in this expression?
                        if is_minimizing:
                            proximal_expression += (rho_parameter[variable_id] / 2.0 * quad_penalty_term_variable[variable_id])
                        else:
                            proximal_expression -= (rho_parameter[variable_id] / 2.0 * quad_penalty_term_variable[variable_id])

                    else:

                        # deal with the baseline quadratic case.
                        if is_minimizing:
                            proximal_expression += (blend_parameter[variable_id] * (rho_parameter[variable_id] / 2.0) * (instance_vardata - xbar_parameter[variable_id]) * (instance_vardata - xbar_parameter[variable_id]))
                        else:
                            proximal_expression -= (blend_parameter[variable_id] * (rho_parameter[variable_id] / 2.0) * (instance_vardata - xbar_parameter[variable_id]) * (instance_vardata - xbar_parameter[variable_id]))

    proximal_expression_component = instance.PHPROXIMAL_EXPRESSION = Expression(initialize=proximal_expression)

    # augment the original objective with the new proximal terms
    objective.expr += proximal_expression_component

    return proximal_expression_component, proximal_expression

#
# form the constraints required to compute the cost variable values
# when linearizing PH objectives.
#

def form_linearized_objective_constraints(instance_name,
                                          instance,
                                          scenario_tree,
                                          linearize_nonbinary_penalty_terms,
                                          breakpoint_strategy,
                                          tolerance):


    # keep track and return what was added to the instance, so
    # it can be cleaned up if necessary.
    new_instance_attributes = []

    linearization_index_set_name = "PH_LINEARIZATION_INDEX_SET"
    linearization_index_set = instance.find_component(linearization_index_set_name)
    if linearization_index_set is None:
        linearization_index_set = Set(initialize=range(0, linearize_nonbinary_penalty_terms*2), dimen=1, name=linearization_index_set_name)
        instance.add_component(linearization_index_set_name, linearization_index_set)

    scenario = scenario_tree.get_scenario(instance_name)

    nodeid_to_vardata_map = instance._ScenarioTreeSymbolMap.bySymbol

    for tree_node in scenario._node_list[:-1]:

        xbar_dict = tree_node._xbars

        # if linearizing, then we have previously defined a variable
        # associated with the result of the linearized approximation
        # of the penalty term - this is simply added to the objective
        # function.
        linearized_cost_variable_name = "PHQUADPENALTY_"+str(tree_node._name)
        linearized_cost_variable = instance.find_component(linearized_cost_variable_name)

        # grab the linearization constraint associated with the
        # linearized cost variable, if it exists. otherwise, create it
        # - but an empty variety. the constraints are stage-specific -
        # we could index by constraint, but we don't know if that is
        # really worth the additional effort.
        linearization_constraint_name = "PH_LINEARIZATION_"+str(tree_node._name)
        linearization_constraint = instance.find_component(linearization_constraint_name)
        if linearization_constraint is not None:
            # clear whatever constraint components are there - there
            # may be fewer breakpoints, due to tolerances, and we
            # don't want to the old pieces laying around.
            linearization_constraint.clear()
        else:
            # this is the first time the constraint is being added -
            # add it to the list of PH-specific constraints for this
            # instance.
            new_instance_attributes.append(linearization_constraint_name)
            nodal_index_set_name = "PHINDEX_"+str(tree_node._name)
            nodal_index_set = instance.find_component(nodal_index_set_name)
            assert nodal_index_set is not None
            linearization_constraint = \
                Constraint(nodal_index_set,
                           linearization_index_set,
                           name=linearization_constraint_name)
            linearization_constraint.construct()
            instance.add_component(linearization_constraint_name, linearization_constraint)

        for variable_id in tree_node._variable_ids:

            # don't add weight terms for derived variables at the tree
            # node.
            if variable_id in tree_node._derived_variable_ids:
                continue

            if variable_id not in tree_node._minimums:
                variable_name, index = tree_node._variable_ids[variable_id]
                raise RuntimeError("No minimum value statistic found for variable=%s "
                                   "on tree node=%s; cannot form linearized PH objective"
                                   % (variable_name+indexToString(index),
                                      tree_node._name))
            if variable_id not in tree_node._maximums:
                variable_name, index = tree_node._variable_ids[variable_id]
                raise RuntimeError("No maximums value statistic found for "
                                   "variable=%s on tree node=%s; cannot "
                                   "form linearized PH objective"
                                   % (variable_name+indexToString(index),
                                      tree_node._name))


            xbar = xbar_dict[variable_id]
            node_min = tree_node._minimums[variable_id]
            node_max = tree_node._maximums[variable_id]

            instance_vardata = nodeid_to_vardata_map[variable_id]

            if (instance_vardata.stale is False) and (instance_vardata.fixed is False):

                # binaries have already been dealt with in the process of PH objective function formation.
                if isinstance(instance_vardata.domain, BooleanSet) is False:

                    x = instance_vardata

                    if x.lb is None or x.ub is None:
                        msg = "Missing bound for variable '%s'\n"         \
                              'Both lower and upper bounds required when' \
                              ' piece-wise approximating quadratic '      \
                              'penalty terms'
                        raise ValueError(msg % instance_vardata.name)
                    lb = value(x.lb)
                    ub = value(x.ub)

                    # compute the breakpoint sequence according to the specified strategy.
                    try:
                        strategy = (compute_uniform_breakpoints,
                                    compute_uniform_between_nodestat_breakpoints,
                                    compute_uniform_between_woodruff_breakpoints,
                                    compute_exponential_from_mean_breakpoints,
                                    )[ breakpoint_strategy ]
                        args = ( lb, node_min, xbar, node_max, ub, \
                                     linearize_nonbinary_penalty_terms, tolerance )
                        breakpoints = strategy( *args )
                    except ValueError:
                        e = sys.exc_info()[1]
                        msg = 'A breakpoint distribution strategy (%s) '  \
                              'is currently not supported within PH!'
                        raise ValueError(msg % breakpoint_strategy)

                    for i in xrange(len(breakpoints)-1):

                        this_lb = breakpoints[i]
                        this_ub = breakpoints[i+1]

                        segment_tuple = create_piecewise_constraint_tuple(this_lb,
                                                                          this_ub,
                                                                          x,
                                                                          xbar,
                                                                          linearized_cost_variable[variable_id],
                                                                          tolerance)

                        linearization_constraint.add((variable_id,i), segment_tuple)

    return new_instance_attributes
