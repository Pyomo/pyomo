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

from pyomo.core import Constraint
from pyomo.core.base.constraint import ConstraintList
from pyomo.contrib.mindtpy.util import generate_norm1_norm_constraint


def fp_converged(working_model, mip_model, config, discrete_only=True):
    """Calculates the euclidean norm between the discrete variables in the MIP and NLP models.

    Parameters
    ----------
    working_model : Pyomo model
        The working model(original model).
    mip_model : Pyomo model
        The mip model.
    config : ConfigBlock
        The specific configurations for MindtPy.
    discrete_only : bool, optional
        Whether to only optimize on distance between the discrete variables, by default True.

    Returns
    -------
    distance : float
        The euclidean norm between the discrete variables in the MIP and NLP models.
    """
    distance = (max((nlp_var.value - milp_var.value)**2
                    for (nlp_var, milp_var) in
                    zip(working_model.MindtPy_utils.variable_list,
                        mip_model.MindtPy_utils.variable_list)
                    if (not discrete_only) or milp_var.is_integer()))
    return distance <= config.fp_projzerotol


def add_orthogonality_cuts(working_model, mip_model, config):
    """Add orthogonality cuts.

    This function adds orthogonality cuts to avoid cycling when the independence constraint qualification is not satisfied.

    Parameters
    ----------
    working_model : Pyomo model
        The working model(original model).
    mip_model : Pyomo model
        The mip model.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    mip_integer_vars = mip_model.MindtPy_utils.discrete_variable_list
    nlp_integer_vars = working_model.MindtPy_utils.discrete_variable_list
    orthogonality_cut = sum((nlp_v.value-mip_v.value)*(mip_v-nlp_v.value)
                            for mip_v, nlp_v in zip(mip_integer_vars, nlp_integer_vars)) >= 0
    mip_model.MindtPy_utils.cuts.fp_orthogonality_cuts.add(
        orthogonality_cut)
    if config.fp_projcuts:
        orthogonality_cut = sum((nlp_v.value-mip_v.value)*(nlp_v-nlp_v.value)
                                for mip_v, nlp_v in zip(mip_integer_vars, nlp_integer_vars)) >= 0
        working_model.MindtPy_utils.cuts.fp_orthogonality_cuts.add(
            orthogonality_cut)


def generate_norm_constraint(fp_nlp_model, mip_model, config):
    """Generate the norm constraint for the FP-NLP subproblem.

    Parameters
    ----------
    fp_nlp_model : Pyomo model
        The feasibility pump NLP subproblem.
    mip_model : Pyomo model
        The mip_model model.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    if config.fp_main_norm == 'L1':
        # TODO: check if we can access the block defined in FP-main problem
        generate_norm1_norm_constraint(
            fp_nlp_model, mip_model, config, discrete_only=True)
    elif config.fp_main_norm == 'L2':
        fp_nlp_model.norm_constraint = Constraint(expr=sum((nlp_var - mip_var.value)**2 - config.fp_norm_constraint_coef*(nlp_var.value - mip_var.value)**2
                                                     for nlp_var, mip_var in zip(fp_nlp_model.MindtPy_utils.discrete_variable_list, mip_model.MindtPy_utils.discrete_variable_list)) <= 0)
    elif config.fp_main_norm == 'L_infinity':
        fp_nlp_model.norm_constraint = ConstraintList()
        rhs = config.fp_norm_constraint_coef * max(nlp_var.value - mip_var.value for nlp_var, mip_var in zip(
            fp_nlp_model.MindtPy_utils.discrete_variable_list, mip_model.MindtPy_utils.discrete_variable_list))
        for nlp_var, mip_var in zip(fp_nlp_model.MindtPy_utils.discrete_variable_list, mip_model.MindtPy_utils.discrete_variable_list):
            fp_nlp_model.norm_constraint.add(nlp_var - mip_var.value <= rhs)
