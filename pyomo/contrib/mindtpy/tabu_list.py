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
#

from pyomo.common.dependencies import attempt_import, UnavailableClass

cplex, cplex_available = attempt_import('cplex')


class IncumbentCallback_cplex(
    cplex.callbacks.IncumbentCallback if cplex_available else UnavailableClass(cplex)
):
    """Inherent class in Cplex to call Incumbent callback."""

    def __call__(self):
        """
        This is an inherent function in LazyConstraintCallback in CPLEX.
        This callback will be used after each new potential incumbent is found.
        https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/ilog.odms.cplex.help/refpythoncplex/html/cplex.callbacks.IncumbentCallback-class.html
        IncumbentCallback will be activated after Lazyconstraint callback, when the potential incumbent solution is satisfies the lazyconstraints.
        TODO: need to handle GOA same integer combination check in lazyconstraint callback in single_tree.py
        """
        mindtpy_solver = self.mindtpy_solver
        opt = self.opt
        config = self.config
        if config.single_tree:
            self.reject()
        else:
            temp = []
            for var in mindtpy_solver.mip.MindtPy_utils.discrete_variable_list:
                value = self.get_values(opt._pyomo_var_to_solver_var_map[var])
                temp.append(int(round(value)))
            mindtpy_solver.curr_int_sol = tuple(temp)

            if mindtpy_solver.curr_int_sol in set(mindtpy_solver.integer_list):
                self.reject()
