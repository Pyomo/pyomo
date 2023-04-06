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
"""A library of possible callbacks to use for initializing the NLP subproblems.
However, it seems likely that problem-specific routines might be better, in
which case you can write your own, and specify it in the
'subproblem_initialization_method' argument."""


def _restore_vars_from_nlp_block_saved_values(nlp_util_block):
    for var, old_value in nlp_util_block.initial_var_values.items():
        if not var.fixed and var.is_continuous():
            if old_value is not None:
                # Adjust value if it falls outside the bounds
                if var.has_lb() and old_value < var.lb:
                    old_value = var.lb
                if var.has_ub() and old_value > var.ub:
                    old_value = var.ub
                # Set the value
                var.set_value(old_value)


# This is the original GDPopt behavior:
def restore_vars_to_original_values(solver, nlp_util_block, mip_util_block):
    """Perform initialization of the subproblem.

    This just restores the continuous variables to the original
    model values, which were saved on the subproblem's utility block when it
    was created.
    """
    # restore original continuous variable values
    _restore_vars_from_nlp_block_saved_values(nlp_util_block)


# This is the default for the enumerate algorithm
def restore_vars_to_original_values_enumerate(
    true_disjuncts, boolean_var_values, discrete_var_values, nlp_util_block
):
    """Perform initialization of the subproblem.

    This just restores the continuous variables to the original
    model values, which were saved on the subproblem's utility block when it
    was created.
    """
    # restore original continuous variable values
    _restore_vars_from_nlp_block_saved_values(nlp_util_block)
