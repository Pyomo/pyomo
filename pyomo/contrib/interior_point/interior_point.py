from pyomo.contrib.interior_point.interface import InteriorPointInterface, BaseInteriorPointInterface
from pyomo.contrib.interior_point.linalg.mumps_interface import MumpsInterface
from scipy.sparse import tril


def solve_interior_point(pyomo_model, max_iter=100):
    interface = InteriorPointInterface(pyomo_model)
    primals = interface.init_primals().copy()
    slacks = interface.init_slacks().copy()
    duals_eq = interface.init_duals_eq().copy()
    duals_ineq = interface.init_duals_ineq().copy()
    duals_primals_lb = interface.init_duals_primals_lb().copy()
    duals_primals_ub = interface.init_duals_primals_ub().copy()
    duals_slacks_lb = interface.init_duals_slacks_lb().copy()
    duals_slacks_ub = interface.init_duals_slacks_ub().copy()
    
    minimum_barrier_parameter = 1e-9
    barrier_parameter = 0.1
    interface.set_barrier_parameter(barrier_parameter)

    for _iter in range(max_iter):
        print('*********************************')
        print('primals: ', primals)
        print('slacks: ', slacks)
        print('duals_eq: ', duals_eq)
        print('duals_ineq: ', duals_ineq)
        print('duals_primals_lb: ', duals_primals_lb)
        print('duals_primals_ub: ', duals_primals_ub)
        print('duals_slacks_lb: ', duals_slacks_lb)
        print('duals_slacks_ub: ', duals_slacks_ub)
        interface.set_primals(primals)
        interface.set_slacks(slacks)
        interface.set_duals_eq(duals_eq)
        interface.set_duals_ineq(duals_ineq)
        interface.set_duals_primals_lb(duals_primals_lb)
        interface.set_duals_primals_ub(duals_primals_ub)
        interface.set_duals_slacks_lb(duals_slacks_lb)
        interface.set_duals_slacks_ub(duals_slacks_ub)
        interface.set_barrier_parameter(barrier_parameter)

        kkt = interface.evaluate_primal_dual_kkt_matrix()
        print(kkt.toarray())
        kkt = tril(kkt.tocoo())
        rhs = interface.evaluate_primal_dual_kkt_rhs()
        print(rhs.flatten())
        linear_solver = MumpsInterface()  # icntl_options={1: 6, 2: 6, 3: 6, 4: 4})
        linear_solver.do_symbolic_factorization(kkt)
        linear_solver.do_numeric_factorization(kkt)
        delta = linear_solver.do_back_solve(rhs)

        interface.set_primal_dual_kkt_solution(delta)
        delta_primals = interface.get_delta_primals()
        delta_slacks = interface.get_delta_slacks()
        delta_duals_eq = interface.get_delta_duals_eq()
        delta_duals_ineq = interface.get_delta_duals_ineq()
        delta_duals_primals_lb = interface.get_delta_duals_primals_lb()
        delta_duals_primals_ub = interface.get_delta_duals_primals_ub()
        delta_duals_slacks_lb = interface.get_delta_duals_slacks_lb()
        delta_duals_slacks_ub = interface.get_delta_duals_slacks_ub()

        alpha = 1
        primals += alpha * delta_primals
        slacks += alpha * delta_slacks
        duals_eq += alpha * delta_duals_eq
        duals_ineq += alpha * delta_duals_ineq
        duals_primals_lb += alpha * delta_duals_primals_lb
        duals_primals_ub += alpha * delta_duals_primals_ub
        duals_slacks_lb += alpha * delta_duals_slacks_lb
        duals_slacks_ub += alpha * delta_duals_slacks_ub

        barrier_parameter = max(minimum_barrier_parameter, min(0.5*barrier_parameter, barrier_parameter**1.5))
