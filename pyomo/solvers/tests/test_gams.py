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

import pyomo.common.unittest as unittest

import pyomo.environ as pyo
import pyomo.solvers.plugins.solvers.GAMS as GAMS
from pyomo.common.tempfiles import TempfileManager

gams_avail = pyo.SolverFactory('gams').available()

@unittest.skipUnless(gams_avail, "Tests require GAMS")
class TestGAMS(unittest.TestCase):
    def test_dat_parser(self):
        # This tests issue 2571
        m = pyo.ConcreteModel()
        m.S = pyo.Set(initialize=list(range(5)))
        m.a_long_var_name = pyo.Var(m.S, bounds=(0, 1), initialize=1)
        m.obj = pyo.Objective(
            expr=2000 * pyo.summation(m.a_long_var_name), sense=pyo.maximize)
        solver = pyo.SolverFactory("gams:conopt")
        res = solver.solve(
            m, symbolic_solver_labels=True, load_solutions=False,
            io_options={'put_results_format': 'dat'})
        self.assertEqual(res.solution[0].Objective['obj']['Value'], 10000)
        for i in range(5):
            self.assertEqual(
                res.solution[0].Variable[f'a_long_var_name_{i}_']['Value'],
                1
            )
