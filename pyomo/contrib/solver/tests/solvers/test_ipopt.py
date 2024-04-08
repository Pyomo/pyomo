#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


import pyomo.environ as pyo
from pyomo.common.fileutils import ExecutableData
from pyomo.common.config import ConfigDict
from pyomo.contrib.solver.ipopt import IpoptConfig
from pyomo.contrib.solver.factory import SolverFactory
from pyomo.common import unittest


"""
TODO:
    - Test unique configuration options
    - Test unique results options
    - Ensure that `*.opt` file is only created when needed
    - Ensure options are correctly parsing to env or opt file
    - Failures at appropriate times
"""


class TestIpopt(unittest.TestCase):
    def create_model(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(initialize=1.5)
        model.y = pyo.Var(initialize=1.5)

        def rosenbrock(m):
            return (1.0 - m.x) ** 2 + 100.0 * (m.y - m.x**2) ** 2

        model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)
        return model

    def test_ipopt_config(self):
        # Test default initialization
        config = IpoptConfig()
        self.assertTrue(config.load_solutions)
        self.assertIsInstance(config.solver_options, ConfigDict)
        self.assertIsInstance(config.executable, ExecutableData)

        # Test custom initialization
        solver = SolverFactory('ipopt', executable='/path/to/exe')
        self.assertFalse(solver.config.tee)
        self.assertTrue(solver.config.executable.startswith('/path'))

        # Change value on a solve call
        # model = self.create_model()
        # result = solver.solve(model, tee=True)
