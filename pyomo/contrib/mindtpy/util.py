#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Utility functions and classes for the MindtPy solver."""
from __future__ import division

import logging
from pyomo.common.collections import ComponentMap
from pyomo.core import Objective, Suffix
from pyomo.core.expr import differentiate
from pyomo.core.expr import current as EXPR
from pyomo.opt import SolverFactory
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver

logger = logging.getLogger('pyomo.contrib')


class MindtPySolveData(object):
    """Data container to hold solve-instance data.
    Key attributes:
        - original_model: the original model that the user gave us to solve
        - working_model: the original model after preprocessing
    """
    pass


def model_is_valid(solve_data, config):
    """
    Determines whether the model is solveable by MindtPy.

    This function returns True if the given model is solveable by MindtPy (and performs some preprocessing such
    as moving the objective to the constraints).

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: MindtPy configurations
        contains the specific configurations for the algorithm

    Returns
    -------
    Boolean value (True if model is solveable in MindtPy else False)
    """
    m = solve_data.working_model
    MindtPy = m.MindtPy_utils

    # Handle LP/NLP being passed to the solver
    prob = solve_data.results.problem
    if (prob.number_of_binary_variables == 0 and
        prob.number_of_integer_variables == 0 and
            prob.number_of_disjunctions == 0):
        config.logger.info('Problem has no discrete decisions.')
        obj = next(m.component_data_objects(ctype=Objective, active=True))
        if (any(c.body.polynomial_degree() not in (1, 0) for c in MindtPy.constraint_list) or
                obj.expr.polynomial_degree() not in (1, 0)):
            config.logger.info(
                "Your model is an NLP (nonlinear program). "
                "Using NLP solver %s to solve." % config.nlp_solver)
            SolverFactory(config.nlp_solver).solve(
                solve_data.original_model, tee=config.solver_tee, **config.nlp_solver_args)
            return False
        else:
            config.logger.info(
                "Your model is an LP (linear program). "
                "Using LP solver %s to solve." % config.mip_solver)
            mipopt = SolverFactory(config.mip_solver)
            if isinstance(mipopt, PersistentSolver):
                mipopt.set_instance(solve_data.original_model)
            if config.threads > 0:
                mipopt.options["threads"] = config.threads
            mipopt.solve(solve_data.original_model,
                         tee=config.solver_tee, **config.mip_solver_args)
            return False

    if not hasattr(m, 'dual') and config.use_dual:  # Set up dual value reporting
        m.dual = Suffix(direction=Suffix.IMPORT)

    # TODO if any continuous variables are multiplied with binary ones,
    #  need to do some kind of transformation (Glover?) or throw an error message
    return True


def calc_jacobians(solve_data, config):
    """
    Generates a map of jacobians for the variables in the model

    This function generates a map of jacobians corresponding to the variables in the model and adds this
    ComponentMap to solve_data

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: MindtPy configurations
        contains the specific configurations for the algorithm
    """
    # Map nonlinear_constraint --> Map(
    #     variable --> jacobian of constraint wrt. variable)
    solve_data.jacobians = ComponentMap()
    if config.differentiate_mode == "reverse_symbolic":
        mode = differentiate.Modes.reverse_symbolic
    elif config.differentiate_mode == "sympy":
        mode = differentiate.Modes.sympy
    for c in solve_data.mip.MindtPy_utils.constraint_list:
        if c.body.polynomial_degree() in (1, 0):
            continue  # skip linear constraints
        vars_in_constr = list(EXPR.identify_variables(c.body))
        jac_list = differentiate(
            c.body, wrt_list=vars_in_constr, mode=mode)
        solve_data.jacobians[c] = ComponentMap(
            (var, jac_wrt_var)
            for var, jac_wrt_var in zip(vars_in_constr, jac_list))


def add_feas_slacks(m, config):
    """
    Adds feasibility slack variables according to config.feasibility_norm (given an infeasible problem)

    Parameters
    ----------
    m: model
        Pyomo model
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    MindtPy = m.MindtPy_utils
    # generate new constraints
    for i, constr in enumerate(MindtPy.constraint_list, 1):
        if constr.body.polynomial_degree() not in [0, 1]:
            if constr.has_ub():
                if config.feasibility_norm in {'L1', 'L2'}:
                    c = MindtPy.MindtPy_feas.feas_constraints.add(
                        constr.body - constr.upper
                        <= MindtPy.MindtPy_feas.slack_var[i])
                else:
                    c = MindtPy.MindtPy_feas.feas_constraints.add(
                        constr.body - constr.upper
                        <= MindtPy.MindtPy_feas.slack_var)
            if constr.has_lb():
                if config.feasibility_norm in {'L1', 'L2'}:
                    c = MindtPy.MindtPy_feas.feas_constraints.add(
                        constr.body - constr.lower
                        >= -MindtPy.MindtPy_feas.slack_var[i])
                else:
                    c = MindtPy.MindtPy_feas.feas_constraints.add(
                        constr.body - constr.lower
                        >= -MindtPy.MindtPy_feas.slack_var)


def var_bound_add(solve_data, config):
    """
    This function will add bounds for variables in nonlinear constraints if they are not bounded. (This is to avoid
    an unbounded master problem in the LP/NLP algorithm.) Thus, the model will be updated to include bounds for the
    unbounded variables in nonlinear constraints.

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm

    """
    m = solve_data.working_model
    MindtPy = m.MindtPy_utils
    for c in MindtPy.constraint_list:
        if c.body.polynomial_degree() not in (1, 0):
            for var in list(EXPR.identify_variables(c.body)):
                if var.has_lb() and var.has_ub():
                    continue
                elif not var.has_lb():
                    if var.is_integer():
                        var.setlb(-config.integer_var_bound - 1)
                    else:
                        var.setlb(-config.continuous_var_bound - 1)
                elif not var.has_ub():
                    if var.is_integer():
                        var.setub(config.integer_var_bound)
                    else:
                        var.setub(config.continuous_var_bound)
