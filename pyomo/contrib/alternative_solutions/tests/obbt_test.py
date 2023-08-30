

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

"""Tests for the GDPopt solver plugin."""

# from contextlib import redirect_stdout
# from io import StringIO
# import logging
# from math import fabs
# from os.path import join, normpath

# import pyomo.common.unittest as unittest
# from pyomo.common.log import LoggingIntercept
# from pyomo.common.collections import Bunch
# from pyomo.common.config import ConfigDict, ConfigValue
# from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
# from pyomo.contrib.appsi.solvers.gurobi import Gurobi
# from pyomo.contrib.gdpopt.create_oa_subproblems import (
#     add_util_block, add_disjunct_list, add_constraints_by_disjunct,
#     add_global_constraint_list)
# import pyomo.contrib.gdpopt.tests.common_tests as ct
# from pyomo.contrib.gdpopt.util import is_feasible, time_code
# from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available
# from pyomo.contrib.gdpopt.solve_discrete_problem import (
#     solve_MILP_discrete_problem, distinguish_mip_infeasible_or_unbounded)
# from pyomo.environ import (
#     Block, ConcreteModel, Constraint, Integers, LogicalConstraint, maximize,
#     Objective, RangeSet, TransformationFactory, SolverFactory, sqrt, value, Var)
# from pyomo.gdp import Disjunct, Disjunction
# from pyomo.gdp.tests import models
# from pyomo.opt import TerminationCondition

class TestGDPoptUnit(unittest.TestCase):
    """Real unit tests for GDPopt"""

    #@unittest.skipUnless(SolverFactory(mip_solver).available(),
    #                     "MIP solver not available")
    def test_continuous_2d(self):
        m = ConcreteModel()
        m.GDPopt_utils = Block()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.z = Var()
        # Include a disjunction so that we don't default to just a MIP solver
        m.d = Disjunction(expr=[
            [m.x + m.y >= 5], [m.x - m.y <= 3]
        ])
        m.o = Objective(expr=m.z)
        m.GDPopt_utils.variable_list = [m.x, m.y, m.z]
        m.GDPopt_utils.disjunct_list = [m.d._autodisjuncts[0],
                                        m.d._autodisjuncts[1]]
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.WARNING):
            solver = SolverFactory('gdpopt.loa')
            dummy = Block()
            dummy.timing = Bunch()
            with time_code(dummy.timing, 'main', is_main_timer=True):
                tc = solve_MILP_discrete_problem(
                    m.GDPopt_utils,
                    dummy,
                    solver.CONFIG(dict(mip_solver=mip_solver)))
            self.assertIn("Discrete problem was unbounded. Re-solving with "
                          "arbitrary bound values", output.getvalue().strip())
        self.assertIs(tc, TerminationCondition.unbounded)

if __name__ == '__main__':
    unittest.main()


# # -*- coding: utf-8 -*-
# """
# Created on Thu Aug  4 15:59:24 2022

# @author: jlgearh
# """
# import random

# import pyomo.environ as pe

# from pyomo.contrib.alternative_solutions.obbt import obbt_analysis

# def get_random_knapsack_model(num_x_vars, num_y_vars, budget_pct, seed=1000):
#     random.seed(seed)
    
#     W = budget_pct * (num_x_vars + num_y_vars) / 2
    
    
#     model = pe.ConcreteModel()
    
#     model.X_INDEX = pe.RangeSet(1,num_x_vars)
#     model.Y_INDEX = pe.RangeSet(1,num_y_vars)
    
#     model.wu = pe.Param(model.X_INDEX, initialize=lambda model, i : round(random.uniform(0.0,1.0), 2), within=pe.Reals)
#     model.vu = pe.Param(model.X_INDEX, initialize=lambda model, i : round(random.uniform(0.0,1.0), 2), within=pe.Reals)
#     model.x = pe.Var(model.X_INDEX, within=pe.NonNegativeIntegers)
    
#     model.b = pe.Block()
#     model.b.wl = pe.Param(model.Y_INDEX, initialize=lambda model, i : round(random.uniform(0.0,1.0), 2), within=pe.Reals)
#     model.b.vl = pe.Param(model.Y_INDEX, initialize=lambda model, i : round(random.uniform(0.0,1.0), 2), within=pe.Reals)
#     model.b.y = pe.Var(model.Y_INDEX, within=pe.NonNegativeReals)
    
#     model.o = pe.Objective(expr=sum(model.vu[i]*model.x[i] for i in model.X_INDEX) + \
#                            sum(model.b.vl[i]*model.b.y[i] for i in model.Y_INDEX), sense=pe.maximize)
#     model.c = pe.Constraint(expr=sum(model.wu[i]*model.x[i] for i in model.X_INDEX) + \
#                             sum(model.b.wl[i]*model.b.y[i] for i in model.Y_INDEX)<= W)
        
#     return model

# model = get_random_knapsack_model(4, 4, 0.2)
# result = obbt_analysis(model, variables='all', rel_opt_gap=None, 
#                                   abs_gap=None, already_solved=False, 
#                                   solver='gurobi', solver_options={}, 
#                                   use_persistent_solver = False, tee=True,
#                                   refine_bounds=False)