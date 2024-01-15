#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""This module provides functions for cut generation used across multiple
algorithms."""
from math import fabs
from pyomo.common.collections import ComponentSet
from pyomo.core import TransformationFactory, value, Constraint, Block


def _record_binary_value(var, var_value_is_one, var_value_is_zero, int_tol):
    val = value(var)
    if fabs(val - 1) <= int_tol:
        var_value_is_one.add(var)
    elif fabs(val) <= int_tol:
        var_value_is_zero.add(var)
    else:
        raise ValueError(
            'Binary %s = %s is not 0 or 1 within integer tolerance %s'
            % (var.name, val, int_tol)
        )


def add_no_good_cut(target_model_util_block, config):
    """Cut the current integer solution from the target model."""
    var_value_is_one = ComponentSet()
    var_value_is_zero = ComponentSet()
    for var in target_model_util_block.transformed_boolean_variable_list:
        _record_binary_value(
            var, var_value_is_one, var_value_is_zero, config.integer_tolerance
        )

    disjuncts = []
    if config.force_subproblem_nlp:
        # We need to also cut the solutions for the other discrete variables
        for var in target_model_util_block.discrete_variable_list:
            # This has possible duplicates with the above, but few enough that
            # we'll leave it for now.
            if var.is_binary():
                _record_binary_value(
                    var, var_value_is_one, var_value_is_zero, config.integer_tolerance
                )
            else:
                # It's integer. It still has to be in the no-good cut because
                # else the algorithm is wrong (We're cutting more than just this
                # solution), so we're going to do this as a Disjunction for
                # now. I think this is the *wrong* way to model if you are using
                # GDPopt, so if it's a little inefficient then... We'll deal
                # with that if it becomes an issue.
                val = round(value(var), 0)
                less = var <= val - 1
                more = var >= val + 1
                disjuncts.extend([less, more])

    # It shouldn't be possible to get here unless there's a solution to be cut.
    assert (var_value_is_one or var_value_is_zero) or len(disjuncts) == 0

    int_cut = (
        sum(1 - v for v in var_value_is_one) + sum(v for v in var_value_is_zero)
    ) >= 1

    if len(disjuncts) > 0:
        idx = len(target_model_util_block.no_good_disjunctions)
        target_model_util_block.no_good_disjunctions[idx] = [
            [disj] for disj in disjuncts
        ] + [[int_cut]]
        config.logger.debug(
            'Adding no-good disjunction: %s'
            % _disjunction_to_str(target_model_util_block.no_good_disjunctions[idx])
        )
        # transform it
        TransformationFactory(config.discrete_problem_transformation).apply_to(
            target_model_util_block,
            targets=[target_model_util_block.no_good_disjunctions[idx]],
        )
    else:
        config.logger.debug('Adding no-good cut: %s' % int_cut)
        # Exclude the current binary combination
        target_model_util_block.no_good_cuts.add(expr=int_cut)


def _disjunction_to_str(disjunction):
    pretty = []
    for disjunct in disjunction.disjuncts:
        exprs = []
        for cons in disjunct.component_data_objects(
            Constraint, active=True, descend_into=Block
        ):
            exprs.append(str(cons.expr))
        pretty.append("[%s]" % ", ".join(exprs))
    return " v ".join(pretty)
