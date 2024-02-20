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

from pyomo.common import unittest, Executable
from pyomo.repn.plugins.nl_writer import NLWriter
from pyomo.contrib.solver import ipopt


ipopt_available = ipopt.Ipopt().available()


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestIpoptSolverConfig(unittest.TestCase):
    def test_default_instantiation(self):
        config = ipopt.IpoptConfig()
        # Should be inherited
        self.assertIsNone(config._description)
        self.assertEqual(config._visibility, 0)
        self.assertFalse(config.tee)
        self.assertTrue(config.load_solutions)
        self.assertTrue(config.raise_exception_on_nonoptimal_result)
        self.assertFalse(config.symbolic_solver_labels)
        self.assertIsNone(config.timer)
        self.assertIsNone(config.threads)
        self.assertIsNone(config.time_limit)
        # Unique to this object
        self.assertIsInstance(config.executable, type(Executable('path')))
        self.assertIsInstance(config.writer_config, type(NLWriter.CONFIG()))

    def test_custom_instantiation(self):
        config = ipopt.IpoptConfig(description="A description")
        config.tee = True
        self.assertTrue(config.tee)
        self.assertEqual(config._description, "A description")
        self.assertFalse(config.time_limit)
        # Default should be `ipopt`
        self.assertIsNotNone(str(config.executable))
        self.assertIn('ipopt', str(config.executable))
        # Set to a totally bogus path
        config.executable = Executable('/bogus/path')
        self.assertIsNone(config.executable.executable)
        self.assertFalse(config.executable.available())


class TestIpoptResults(unittest.TestCase):
    def test_default_instantiation(self):
        res = ipopt.IpoptResults()
        # Inherited methods/attributes
        self.assertIsNone(res.solution_loader)
        self.assertIsNone(res.incumbent_objective)
        self.assertIsNone(res.objective_bound)
        self.assertIsNone(res.solver_name)
        self.assertIsNone(res.solver_version)
        self.assertIsNone(res.iteration_count)
        self.assertIsNone(res.timing_info.start_timestamp)
        self.assertIsNone(res.timing_info.wall_time)
        # Unique to this object
        self.assertIsNone(res.timing_info.ipopt_excluding_nlp_functions)
        self.assertIsNone(res.timing_info.nlp_function_evaluations)
        self.assertIsNone(res.timing_info.total_seconds)


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestIpoptInterface(unittest.TestCase):
    pass
