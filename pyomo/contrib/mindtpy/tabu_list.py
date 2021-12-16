from cplex.callbacks import IncumbentCallback
from pyomo.core import Var


class IncumbentCallback_cplex(IncumbentCallback):
    """Inherent class in Cplex to call Incumbent callback."""

    def __call__(self):
        """
        This is an inherent function in LazyConstraintCallback in cplex. 
        This callback will be used after each new potential incumbent is found.
        https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/ilog.odms.cplex.help/refpythoncplex/html/cplex.callbacks.IncumbentCallback-class.html
        IncumbentCallback will be activated after Lazyconstraint callback, when the potential incumbent solution is satisfies the lazyconstraints.
        TODO: need to handle GOA same integer combination check in lazyconstraint callback in single_tree.py
        TODO: integer_var_value_tuple can be replaced by solve_data.curr_int_sol
        """
        solve_data = self.solve_data
        opt = self.opt
        config = self.config
        if config.single_tree:
            self.reject()
        else:
            temp = []
            for var in solve_data.mip.MindtPy_utils.discrete_variable_list:
                value = self.get_values(
                    opt._pyomo_var_to_solver_var_map[var])
                temp.append(int(round(value)))
            integer_var_value = tuple(temp)

            if integer_var_value in set(solve_data.integer_list):
                self.reject()
