#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import numpy as np
import pyomo.contrib.pynumero as pn
from scipy.sparse import tril


def print_eval_count(iter):
    print("\t\t Number of objective function evaluations..............:  {:>6d}".format(iter + 1))
    print("\t\t Number of objective gradient evaluations..............:  {:>6d}".format(iter + 1))
    print("\t\t Number of equality constraint evaluations.............:  {:>6d}".format(iter + 1))
    print("\t\t Number of inequality constraint evaluations...........:  {:>6d}".format(iter + 1))
    print("\t\t Number of equality constraint Jacobian evaluations....:  {:>6d}".format(iter + 1))
    print("\t\t Number of inequality constraint Jacobian evaluations..:  {:>6d}".format(iter + 1))
    print("\t\t Number of Lagrangian Hessian evaluations..............:  {:>6d}".format(iter + 1))
    #print("\t\t Total CPU secs in PyNumero (w/o function evaluations).:  TODO")
    #print("\t\t Total CPU secs in NLP function evaluations............:  TODO")


def print_nlp_info(nlp, header='Interior-Point', linear_solver='mumps'):

    print("\n\n********************************************************************************************")
    print("********************************************************************************************")
    print("********************** THIS ALGORITHM WAS WRITTEN IN PYNUMERO ******************************")
    print("********************************************************************************************")
    print("********************************************************************************************")

    x = nlp.x_init()
    y = nlp.y_init()
    hess = nlp.hessian_lag(x, y)
    flat_hess = hess.tocoo()
    lower_hess = tril(flat_hess)

    print("\nALGORITHM NAME:", header)
    print("\nLINEAR SOLVER NAME:", linear_solver)
    print("\nPROBLEM STATISTICS:\n")
    print("Number of nonzeros in equality constraint Jacobian...:{:>9d}".format(nlp.nnz_jacobian_c))
    print("Number of nonzeros in inequality constraint Jacobian.:{:>9d}".format(nlp.nnz_jacobian_d))
    print("Number of nonzeros in Lagrangian Hessian.............:{:>9d}".format(lower_hess.nnz))
    print("Number of primal variables...........................:{:>9d}".format(nlp.nx))

    xl_finite_idx = pn.where(np.isfinite(nlp.xl()))
    xu_finite_idx = pn.where(np.isfinite(nlp.xu()))
    xlu_finite_idx = pn.intersect1d(xl_finite_idx, xu_finite_idx)
    num_upper_and_lower = xlu_finite_idx.size
    num_only_lower = len(pn.setdiff1d(xl_finite_idx, xlu_finite_idx))
    num_only_upper = len(pn.setdiff1d(xu_finite_idx, xlu_finite_idx))

    print("Variables with only lower bounds:{:>30d}".format(num_only_lower))
    print("Variables with lower and upper bounds:{:>25d}".format(num_upper_and_lower))
    print("Variables with only upper bounds:{:>30d}".format(num_only_upper))
    print("Total number of equality constraints.................:{:>9d}".format(nlp.nc))
    print("Total number of inequality constraints...............:{:>9d}".format(nlp.nd))
    print("Inequality constraints with only lower bounds:{:>17d}".format(0))
    print("Inequality constraints with lower and upper bounds:{:>12d}".format(0))
    print("Inequality constraints with only upper bounds:{:>17d}\n".format(0))

    print("ITERATIONS:\n")


def print_summary(iter_count, obj, primal_inf, dual_inf, comp_inf):

    print("\nSOLUTION STATISTICS:\n")
    print("Number of Iterations...............................:       {:>5d}".format(iter_count))
    print("Objective..........................................:          {:>16.16e}".format(obj, obj))
    print("Dual infeasibility.................................:          {:>16.16e}".format(dual_inf, dual_inf))
    print("Constraint violation...............................:          {:>16.16e}".format(primal_inf, primal_inf))
    print("Complementarity....................................:          {:>16.16e}".format(comp_inf, comp_inf))
    print("\n")

    #print_eval_count(iter_count)
