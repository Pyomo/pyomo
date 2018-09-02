import numpy as np


def print_eval_count(iter):
    print("Number of objective function evaluations               = {}".format(iter + 1))
    print("Number of objective gradient evaluations               = {}".format(iter + 1))
    print("Number of equality constraint evaluations              = {}".format(iter + 1))
    print("Number of inequality constraint evaluations            = {}".format(0))
    print("Number of equality constraint Jacobian evaluations     = {}".format(iter + 1))
    print("Number of inequality constraint Jacobian evaluations   = {}".format(0))
    print("Number of Lagrangian Hessian evaluations               = {}".format(iter))
    print("Total CPU secs in PyNumero (w/o function evaluations)  = TODO")
    print("Total CPU secs in NLP function evaluations             = TODO")


def print_nlp_info(nlp):

    print("\n\n******************************************************************************")
    print("******************************************************************************")
    print("******************************************************************************")
    print("******************************************************************************")
    print("******************************************************************************")

    print("\nThis is PyNumero running with linear solver ma27.\n")
    print("Number of nonzeros in equality constraint Jacobian...:{:>9d}".format(nlp.nnz_jacobian_c))
    print("Number of nonzeros in inequality constraint Jacobian.:{:>9d}".format(nlp.nnz_jacobian_d))
    print("Number of nonzeros in Lagrangian Hessian.............:{:>9d}\n".format(nlp.nnz_hessian_lag))
    print("Total number of variables............................:{:>9d}".format(nlp.nx))

    xl_finite_idx = np.where(np.isfinite(nlp.xl()))
    xu_finite_idx = np.where(np.isfinite(nlp.xu()))
    xlu_finite_idx = np.intersect1d(xl_finite_idx, xu_finite_idx)
    num_upper_and_lower = xlu_finite_idx.size
    num_only_lower = len(np.setdiff1d(xl_finite_idx, xlu_finite_idx))
    num_only_upper = len(np.setdiff1d(xu_finite_idx, xlu_finite_idx))

    print("                     variables with only lower bounds:{:>9d}".format(num_only_lower))
    print("                variables with lower and upper bounds:{:>9d}".format(num_upper_and_lower))
    print("                     variables with only upper bounds:{:>9d}".format(num_only_upper))
    print("Total number of equality constraints.................:{:>9d}".format(nlp.nc))
    print("Total number of inequality constraints...............:{:>9d}".format(nlp.nd))
    print("        inequality constraints with only lower bounds:{:>9d}".format(0))
    print("   inequality constraints with lower and upper bounds:{:>9d}".format(0))
    print("        inequality constraints with only upper bounds:{:>9d}\n".format(0))


def print_summary(iter_count, obj, primal_inf, dual_inf):

    print("\nNumber of Iterations....:{:>4d}\n".format(iter_count))
    print("                                   (scaled)                 (unscaled)")
    print("Objective...............:  {:>2.16e}     {:>16.16e}".format(obj, obj))
    print("Dual infeasibility......:  {:>2.16e}     {:>16.16e}".format(dual_inf, dual_inf))
    print("Constraint violation....:  {:>2.16e}     {:>16.16e}".format(primal_inf, primal_inf))
    print("Complementarity.........:  {:>2.16e}     {:>16.16e}".format(0.0, 0.0))
    print("\n")

    print_eval_count(iter_count)