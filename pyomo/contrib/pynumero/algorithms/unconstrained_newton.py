from pyomo.contrib.pynumero.algorithms.print_utils import (print_nlp_info,
                                                           print_summary)
from pyomo.contrib.pynumero.algorithms import (InertiaCorrectionParams,
                                 UnconstrainedLineSearch)
from pyomo.contrib.pynumero.linalg.solvers import ma27_solver
import pyomo.environ as pe
import math as pymath
import numpy as np


def newton_unconstrained(nlp, **kwargs):

    max_iter = kwargs.pop('max_iter', 100)
    reg_max_iter = kwargs.pop('reg_max_iter', 100)
    wls = kwargs.pop('wls', True)
    tee = kwargs.pop('tee', True)

    nxlb = len(nlp.xl(condensed=True))
    nxub = len(nlp.xu(condensed=True))

    assert nlp.ng == 0, 'Newton unconstrained only supports problems without constraints'
    assert nxlb == 0, 'Newton unconstrained does not support lower bounds'
    assert nxub == 0, 'Newton unconstrained does not support upper bounds'

    if tee:
        print_nlp_info(nlp)

    # create vector of variables
    x = np.copy(nlp.x_init())
    dx = nlp.create_vector_x()
    lam = nlp.create_vector_y()

    # evaluate all components
    f = nlp.objective(x)
    df = nlp.grad_objective(x)
    hess = nlp.hessian_lag(x, lam)

    # create linear solver
    lsolver = ma27_solver.MA27LinearSolver()
    lsolver.do_symbolic_factorization(hess, include_diagonal=True)

    # line search helper
    line_searcher = UnconstrainedLineSearch(nlp.objective, nlp.grad_objective)

    # define parameters
    alpha_primal = 0.0
    alpha_dual = 0.0
    norm_d = 0.0
    inertia_params = InertiaCorrectionParams()
    nx = nlp.nx
    diag_correction = np.zeros(nx)
    val_reg = 0.0

    if tee:
        print("iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls")

    for i in range(max_iter):

        # compute infeasibility
        dual_inf = np.linalg.norm(df, ord=np.inf)
        # format the line for printing
        if val_reg == 0.0:
            formating = "{:>4d} {:>14.7e} {:>7.2e} {:>7.2e}  {} {:>9.2e} {:>4} {:>10.2e} {:>8.2e} {:>3d}"
            regularization = "--"
        else:
            formating = "{:>4d} {:>14.7e} {:>7.2e} {:>7.2e}  {} {:>9.2e} {:>4.1f} {:>7.2e} {:>7.2e} {:>3d}"
            regularization = pymath.log10(val_reg)

        line = formating.format(i, f, 0.0, dual_inf, "--", norm_d,
                                regularization, alpha_dual, alpha_primal, line_searcher.num_backtrack)
        if tee:
            print(line)

        # Compute search direction
        diag_correction.fill(0.0)
        status = lsolver.do_numeric_factorization(hess, diagonal=diag_correction)
        done = inertia_params.ibr1(status)
        j = 0
        val_reg = 0.0
        while not done:
            diag_correction[0: nx] = inertia_params.delta_w
            status = lsolver.do_numeric_factorization(hess, diagonal=diag_correction)
            if inertia_params.delta_w > 0.0:
                val_reg = inertia_params.delta_w
            done = inertia_params.ibr4(status)
            j += 1
            if j > reg_max_iter:
                break

        dx = lsolver.do_back_solve(-df)

        if dual_inf < 1e-6:
            if tee:
                print_summary(i, f, 0.0, dual_inf)
                print("\nEXIT: Optimal Solution Found\n")
            return x, lam

        # step
        if wls:
            alpha = line_searcher.search(x, dx)
        else:
            alpha = 1.0
        x += dx * alpha

        norm_d = np.linalg.norm(dx, ord=np.inf)
        alpha_dual = alpha
        alpha_primal = alpha

        # evaluate objective
        f = nlp.objective(x)
        # evaluate gradient of the objective
        nlp.grad_objective(x, df)
        # evaluate hessian of the lagrangian
        nlp.hessian_lag(x, lam, hess, eval_f_c=False)

    print("Reach limit iterations")
    return x, lam

if __name__ == "__main__":

    """
    import pyomo.contrib.pynumero.algorithms.bdexp_cute as mod

    model = mod.model
    opt = pe.SolverFactory('ipopt')

    nlp = PyomoNLP(model)

    # for now simply redirect stderr to suppress regularization warnings
    x, lam = newton_unconstrained(nlp)

    opt.options['nlp_scaling_method'] = 'none'
    opt.options['linear_system_scaling'] = 'none'
    opt.solve(model, tee=True)
    """




