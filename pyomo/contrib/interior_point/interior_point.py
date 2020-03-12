from pyomo.contrib.interior_point.interface import InteriorPointInterface, BaseInteriorPointInterface
from pyomo.contrib.interior_point.linalg.mumps_interface import MumpsInterface


class InteriorPointSolver(object):
    def __init__(self, pyomo_model):
        self._interface = InteriorPointInterface(pyomo_model)

    def solve(self, max_iter=100):
        interface = self._interface
        primals = interface.init_primals()
        slacks = interface.init_slacks()
        duals_eq = interface.init_duals_eq()
        duals_ineq = interface.init_duals_ineq()
        duals_primals_lb = interface.init_duals_primals_lb()
        duals_primals_ub = interface.init_duals_primals_ub()
        duals_slacks_lb = interface.init_duals_slacks_lb()
        duals_slacks_ub = interface.init_duals_slacks_ub()

        barrier_parameter = 0.1

        linear_solver = MumpsInterface()

        for _iter in range(max_iter):
            interface.set_primals(primals)
            interface.set_slacks(slacks)
            interface.set_duals_eq(duals_eq)
            interface.set_duals_ineq(duals_ineq)
            interface.set_duals_primals_lb(duals_primals_lb)
            interface.set_duals_primals_ub(duals_primals_ub)
            interface.set_duals_slacks_lb(duals_slacks_lb)
            interface.set_duals_slacks_ub(duals_slacks_ub)

            kkt = interface.evaluate_primal_dual_kkt_matrix(barrier_parameter)
            rhs = interface.evaluate_primal_dual_kkt_rhs(barrier_parameter)
            linear_solver.do_symbolic_factorization(kkt)
            linear_solver.do_numeric_factorization(kkt)
            delta = linear_solver.do_back_solve(rhs)

            interface.set_primal_dual_kkt_solution(delta)
            delta_primals = interface.get_delta_primals()
            delta_slacks = interface.get_delta_slacks()
            delta_duals_eq = interface.get_delta_duals_eq()
            delta_duals_ineq = interface.get_delta_duals_ineq()
            delta_duals_primals_lb = -duals_primals_lb + 
