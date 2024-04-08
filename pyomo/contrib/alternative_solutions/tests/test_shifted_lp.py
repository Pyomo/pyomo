from numpy.testing import assert_array_almost_equal

import pyomo.environ as pe
import pyomo.common.unittest as unittest

import pyomo.contrib.alternative_solutions.tests.test_cases as tc
from pyomo.contrib.alternative_solutions import shifted_lp
import pdb


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


mip_solver = "gurobi_appsi"
# mip_solver = 'gurobi'


class TestShiftedIP(unittest.TestCase):

    def test_mip_abs_objective(self):
        """COMMENT"""
        m = tc.get_indexed_pentagonal_pyramid_mip()
        m.x.domain = pe.Reals
        opt = pe.SolverFactory("gurobi")
        old_results = opt.solve(m, tee=True)
        old_obj = pe.value(m.o)
        new_model = shifted_lp.get_shifted_linear_model(m)
        new_results = opt.solve(new_model, tee=True)
        new_obj = pe.value(new_model.objective)
        self.assertAlmostEqual(old_obj, new_obj)

    def test_polyhedron(self):
        m = tc.get_3d_polyhedron_problem()
        opt = pe.SolverFactory("gurobi")
        old_results = opt.solve(m, tee=True)
        old_obj = pe.value(m.o)
        new_model = shifted_lp.get_shifted_linear_model(m)
        new_results = opt.solve(new_model, tee=True)
        new_obj = pe.value(new_model.objective)


if __name__ == "__main__":
    unittest.main()
