#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo
from pyomo.opt import check_optimal_termination
from pyomo.common.dependencies import attempt_import
from .interface import InteriorPointInterface
from .linalg.scipy_interface import ScipyInterface

np, numpy_available = attempt_import('numpy',
                                     'Interior point requires numpy',
                                     minimum_version='1.13.0')


# Todo: This function currently used IPOPT for the initial solve - should accept solver
def inv_reduced_hessian_barrier(model, independent_variables, bound_tolerance=1e-6, solver_options=None, tee=False):
    """
    This function computes the inverse of the reduced Hessian of a problem at the
    solution. This function first solves the problem with Ipopt and then generates
    the KKT system for the barrier subproblem to compute the inverse reduced hessian.

    For more information on the reduced Hessian, see "Numerical Optimization", 2nd Edition
    Nocedal and Wright, 2006.
    
    The approach used in this method can be found in, "Computational Strategies for 
    the Optimal Operation of Large-Scale Chemical Processes", Dissertation, V. Zavala
    2008. See section 3.2.1.

    Parameters
    ----------
    model : Pyomo model
        The Pyomo model that we want to solve and analyze
    independent_variables : list of Pyomo variables
        This is the list of independent variables for computing the reduced hessian.
        These variables must not be at their bounds at the solution of the 
        optimization problem.
    bound_tolerance : float
       The tolerance to use when checking if the variables are too close to their bound.
       If they are too close, then the routine will exit without a reduced hessian.
    solver_options: dictionary
        Additional solver options to consider.
    tee : bool
       This flag is sent to the tee option of the solver. If true, then the solver
       log is output to the console.
    """
    m = model

    # make sure the necessary suffixes are added
    # so the reduced hessian kkt system is setup correctly from
    # the ipopt solution
    if not hasattr(m, 'ipopt_zL_out'):
        m.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    if not hasattr(m, 'ipopt_zU_out'):
        m.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    if not hasattr(m, 'ipopt_zL_in'):
        m.ipopt_zL_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    if not hasattr(m, 'ipopt_zU_in'):
        m.ipopt_zU_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    if not hasattr(m, 'dual'):
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    # create the ipopt solver
    solver = pyo.SolverFactory('ipopt')

    # copy additional solver options
    if solver_options is not None:
        for key in solver_options:
            solver.options[key] = solver_options[key]

    # set options to prevent bounds relaxation (and 0 slacks)
    solver.options['bound_relax_factor']=0
    solver.options['honor_original_bounds']='no'

    # solve the problem
    status = solver.solve(m, tee=tee)
    if not check_optimal_termination(status):
        return status, None

    # compute the barrier parameter
    # ToDo: this needs to eventually come from the solver itself
    estimated_mu = list()
    for v in m.ipopt_zL_out:
        if v.has_lb():
            estimated_mu.append((pyo.value(v) - v.lb)*m.ipopt_zL_out[v])
    for v in m.ipopt_zU_out:
        if v.has_ub():
            estimated_mu.append((v.ub - pyo.value(v))*m.ipopt_zU_out[v])
    if len(estimated_mu) == 0:
        mu = 10**-8.6
    else:
        mu = sum(estimated_mu)/len(estimated_mu)
        # check to make sure these estimates were all reasonable
        if any([abs(mu-estmu) > 1e-7 for estmu in estimated_mu]):
            print('Warning: estimated values of mu do not seem consistent - using mu=10^(-8.6)')
            mu = 10**-8.6

    # collect the list of var data objects for the independent variables
    ind_vardatas = list()
    for v in independent_variables:
        if v.is_indexed():
            for k in v:
                ind_vardatas.append(v[k])
        else:
            ind_vardatas.append(v)

    # check that none of the independent variables are at their bounds
    for v in ind_vardatas:
        if (v.has_lb() and pyo.value(v) - v.lb <= bound_tolerance) or \
           (v.has_ub() and v.ub - pyo.value(v) <= bound_tolerance):
                raise ValueError("Independent variable: {} has a solution value that is near"
                                 " its bound (according to tolerance). The reduced hessian"
                                 " computation does not support this at this time. All"
                                 " independent variables should be in their interior.".format(v))

    # find the list of indices that we need to make up the reduced hessian
    kkt_builder = InteriorPointInterface(m)
    pyomo_nlp = kkt_builder.pyomo_nlp()
    ind_var_indices = pyomo_nlp.get_primal_indices(ind_vardatas)

    # setup the computation of the reduced hessian
    kkt_builder.set_barrier_parameter(mu)
    kkt = kkt_builder.evaluate_primal_dual_kkt_matrix()
    linear_solver = ScipyInterface(compute_inertia=False)
    linear_solver.do_symbolic_factorization(kkt)
    linear_solver.do_numeric_factorization(kkt)

    n_rh = len(ind_var_indices)
    rhs = np.zeros(kkt.shape[0])
    inv_red_hess = np.zeros((n_rh, n_rh))
    
    for rhi, vari in enumerate(ind_var_indices):
        rhs[vari] = 1
        v = linear_solver.do_back_solve(rhs)
        rhs[vari] = 0
        for rhj, varj in enumerate(ind_var_indices):
            inv_red_hess[rhi,rhj] = v[varj]

    return status, inv_red_hess
