#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common import unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.contrib.solver.solvers.sol_reader import SolFileData

currdir = this_file_dir()


class TestSolFileData(unittest.TestCase):
    def test_default_instantiation(self):
        instance = SolFileData()
        self.assertIsInstance(instance.primals, list)
        self.assertIsInstance(instance.duals, list)
        self.assertIsInstance(instance.var_suffixes, dict)
        self.assertIsInstance(instance.con_suffixes, dict)
        self.assertIsInstance(instance.obj_suffixes, dict)
        self.assertIsInstance(instance.problem_suffixes, dict)
        self.assertIsInstance(instance.other, list)


class TestSolParser(unittest.TestCase):
    # I am not sure how to write these tests best since the sol parser requires
    # not only a file but also the nl_info and results objects.
    def setUp(self):
        TempfileManager.push()

    def tearDown(self):
        TempfileManager.pop(remove=True)

    def test_default_behavior(self):
        pass

    def test_custom_behavior(self):
        pass

    def test_infeasible1(self):
        pass

    def test_infeasible2(self):
        pass
