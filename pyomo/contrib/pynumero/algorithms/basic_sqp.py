from pyomo.contrib.pynumero.algorithms.print_utils import (print_nlp_info,
                                                           print_summary)
from pyomo.contrib.pynumero.sparse import BlockVector, BlockSymMatrix
from pyomo.contrib.pynumero.algorithms import (InertiaCorrectionParams,
                                               BasicFilterLineSearch,
                                               newton_unconstrained)
from pyomo.contrib.pynumero.linalg.solvers import ma27_solver
import pyomo.environ as pe
import math as pymath
import numpy as np
import logging

from pyutilib.misc.timing import tic, toc


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(__name__ + logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    return l

sqp_logger = setup_logger("", "sqp.log", level=logging.DEBUG)
vector_logger = setup_logger("vector_", "sqp_vector.log", level=logging.DEBUG)
ic_logger = setup_logger("ic_", "sqp_ic.log", level=logging.DEBUG)


def grad_x_lagrangian(grad_objective, jacobian, lam):
    return grad_objective + jacobian.transpose()*lam


def basic_sqp(nlp, **kwargs):

    """
    Runs a basic sqp algorithm

    Parameters
    ----------
    nlp : NLP
    kwargs

    Returns
    -------

    """

    tee = kwargs.pop('tee', True)

    nxlb = len(nlp.xl(condensed=True))
    nxub = len(nlp.xu(condensed=True))

    if nlp.ng == 0 and nxlb == 0 and nxub == 0:
        sqp_logger.info("Solve as unconstrained problem")
        return newton_unconstrained(nlp, tee=tee, **kwargs)

    if nxlb > 0 or nxub > 0:
        raise RuntimeError('basic_sqp cannot handle bounds yet')
    if nlp.nd > 0:
        raise RuntimeError('basic_sqp cannot handle general inequalities')

    max_iter = kwargs.pop('max_iter', 100)
    reg_max_iter = kwargs.pop('reg_max_iter', 100)
    wls = kwargs.pop('wls', True)
    debug_mode = kwargs.pop('debug_mode', False)

    # create vector of variables
    x = nlp.x_init()
    lam = nlp.create_vector_y()  # initializes multipliers at zero ToDo: change this?

    if tee:
        print_nlp_info(nlp)

    # function evaluations
    f = nlp.objective(x)
    # evaluate gradient of the objective
    df = nlp.grad_objective(x)
    # evaluate residual constraints
    res_c = nlp.evaluate_g(x)
    # evaluate jacobian constraints
    jac_c = nlp.jacobian_g(x)
    # evaluate hessian of the lagrangian
    hess_lag = nlp.hessian_lag(x, lam, eval_f_c=False)
    # create gradient of lagrangian
    grad_x_lag = grad_x_lagrangian(df, jac_c, lam)

    # create KKT system
    kkt = BlockSymMatrix(2)
    kkt[0, 0] = hess_lag
    kkt[1, 0] = jac_c

    # create RHS for newton step
    rhs = BlockVector([-grad_x_lag, -res_c])

    # create block vector of steps
    dxl = BlockVector(2)
    dxl[0] = nlp.create_vector_x() # delta x
    dxl[1] = nlp.create_vector_y() # delta lambda

    # create linear solver
    lsolver = ma27_solver.MA27LinearSolver()
    lsolver.do_symbolic_factorization(kkt, include_diagonal=True)

    # line search helper
    def rule_feasibility(var_x):
        residual_c = nlp.evaluate_g(var_x)
        return np.linalg.norm(residual_c, ord=1)

    line_searcher = BasicFilterLineSearch(nlp.objective, rule_feasibility, nlp.grad_objective)
    line_searcher.add_to_filter(f, rule_feasibility(x))

    # set iteration parameters
    inertia_params = InertiaCorrectionParams()
    diag_correction = np.zeros(kkt.shape[0])
    nx = nlp.nx
    nc = nlp.nc

    norm_d = 0.0
    alpha_dual = 0.0
    alpha_primal = 0.0
    val_reg = 0.0
    num_ls = 0

    if debug_mode:
        sqp_logger.info('Setting up problem')
        sqp_logger.info('Number of variables: {}'.format(nlp.nx))
        sqp_logger.info('Number of constraints: {}'.format(nlp.ng))
        sqp_logger.info('Number of nnz Jacobian constraints: {}'.format(jac_c.nnz))
        sqp_logger.info('Number of nnz Hessian Lagrangian: {}'.format(hess_lag.nnz))
        sqp_logger.info('Number of nnz Lower triangular KKT: {}\n'.format(kkt.nnz))

    for i in range(max_iter):
        if debug_mode:
            vector_logger.debug("\n**************************************************")
            vector_logger.debug("*** Beginning Iteration {} from the following point:".format(i))
            vector_logger.debug("**************************************************\n")
            vector_logger.debug("||curr_x||_inf = {}".format(np.linalg.norm(x, ord=np.inf)))
            vector_logger.debug("||curr_y||_inf = {}".format(np.linalg.norm(lam, ord=np.inf)))
            vector_logger.debug("||curr_c||_inf = {}".format(np.linalg.norm(res_c, ord=np.inf)))
            vector_logger.debug("||curr_grad_f||_inf = {}".format(np.linalg.norm(df, ord=np.inf)))
            vector_logger.debug("\ncurr_x\n")
            vector_logger.debug(x)
            vector_logger.debug("\ncurr_y\n")
            vector_logger.debug(lam)
            vector_logger.debug("\ngrad_f\n")
            vector_logger.debug(df)
            vector_logger.debug("\ncurr_c\n")
            vector_logger.debug(res_c)

        if i%10 == 0 and tee:
            print("iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls")

        primal_inf = np.linalg.norm(rhs[1], ord=np.inf)
        dual_inf = np.linalg.norm(rhs[0], ord=np.inf)

        if val_reg == 0.0:
            formating = "{:>4d} {:>14.7e} {:>7.2e} {:>7.2e}  {} {:>9.2e} {:>4} {:>10.2e} {:>8.2e} {:>3d}"
            regularization = "--"
        else:
            formating = "{:>4d} {:>14.7e} {:>7.2e} {:>7.2e}  {} {:>9.2e}  {:>4.1f}  {:>7.2e} {:>7.2e} {:>3d}"
            regularization = pymath.log10(val_reg)

        line = formating.format(i, f, primal_inf, dual_inf, "--", norm_d,
                                regularization, alpha_dual, alpha_primal, num_ls)
        if tee:
            print(line)

        # Compute search direction
        diag_correction.fill(0.0)
        status = lsolver.do_numeric_factorization(kkt, diagonal=diag_correction, desired_num_neg_eval=nc)
        done = inertia_params.ibr1(status)
        j = 0
        val_reg = 0.0

        if debug_mode:
            ic_logger.debug("\n**************************************************")
            ic_logger.debug("*** Solving the Primal Dual System for Iteration: {}".format(i))
            ic_logger.debug("**************************************************\n")

        while not done:

            if debug_mode:
                ic_logger.debug("Solving system with:")
                ic_logger.debug("\tdelta_x={}".format(inertia_params.delta_w))
                ic_logger.debug("\tdelta_c={}".format(inertia_params.delta_a))
                ic_logger.debug("\tLast solve status: {}".format(status))
                if status == 2:
                    ic_logger.debug("\tNumber of negative eigenvalues: {} ".format(lsolver._get_num_neg_evals()))

            diag_correction[0: nx] = inertia_params.delta_w
            diag_correction[nx: nx + nc] = inertia_params.delta_a
            status = lsolver.do_numeric_factorization(kkt, diagonal=diag_correction, desired_num_neg_eval=nc)
            if inertia_params.delta_w > 0.0:
                val_reg = inertia_params.delta_w
            done = inertia_params.ibr4(status)

            j += 1
            if j > reg_max_iter:
                break

        # solve kkt
        dxl = lsolver.do_back_solve(rhs)

        if dual_inf < 1e-6 and primal_inf < 1e-8:
            if tee:
                print_summary(i, f, primal_inf, dual_inf)
                print("\nEXIT: Optimal Solution Found\n")
            return x, lam

        # Compute step size

        if debug_mode:
            ic_logger.debug("||rhs||_inf = {}".format(np.linalg.norm(rhs.flatten(), ord=np.inf)))
            ic_logger.debug("||sol||_inf = {}".format(np.linalg.norm(dxl.flatten(), ord=np.inf)))

        if wls:
            alpha = line_searcher.search(x, dxl[0])
        else:
            alpha = 1.0
        x += dxl[0] * alpha
        lam += dxl[1] * alpha

        norm_d = np.linalg.norm(dxl[0], ord=np.inf)
        alpha_dual = alpha
        alpha_primal = alpha
        num_ls = line_searcher.num_backtrack + 1

        # evaluate objective
        f = nlp.objective(x)
        # evaluate gradient of the objective
        nlp.grad_objective(x, df)
        # evaluate residual constraints
        nlp.evaluate_g(x, res_c)
        # evaluate jacobian equality constraints
        nlp.jacobian_g(x, jac_c)
        # evaluate hessian of the lagrangian
        nlp.hessian_lag(x, lam, hess_lag, eval_f_c=False)

        # update rhs
        rhs[0] = -grad_x_lagrangian(df, kkt[1, 0], lam)
        rhs[1] = -res_c

    print("Reach limit iterations")
    return x, lam


if __name__ == "__main__":
    """
    import pyomo.contrib.pynumero.algorithms.basic3 as mod

    model = mod.model
    opt = pe.SolverFactory('ipopt')

    start = tic(msg=False)
    nlp = PyomoNLP(model)

    x, lam = basic_sqp(nlp, wls=True, max_iter=300)
    end = toc(msg=False)
    print("time {}".format(end))
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)
    for c in model.component_map(pe.Constraint, active=True):
        model.dual[c] = 1.0001

    opt.options['warm_start_init_point'] = 'yes'
    opt.options['linear_system_scaling'] = 'none'
    opt.solve(model, tee=True)
    """