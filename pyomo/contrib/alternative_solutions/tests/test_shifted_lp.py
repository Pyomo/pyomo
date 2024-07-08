import pytest

try:
    from numpy.testing import assert_array_almost_equal

    numpy_available = True
except:
    numpy_available = False

import pyomo.environ as pe
import pyomo.opt
import pyomo.common.unittest as unittest

import pyomo.contrib.alternative_solutions.tests.test_cases as tc
from pyomo.contrib.alternative_solutions import shifted_lp

# TODO: add checks that confirm the shifted constraints make sense

#
# Find available solvers. Just use GLPK if it's available.
#
solvers = list(pyomo.opt.check_available_solvers("glpk", "gurobi"))
if "glpk" in solvers:
    solver = ["glpk"]
pytestmark = pytest.mark.parametrize("lp_solver", solvers)


@unittest.pytest.mark.default
class TestShiftedIP:

    @pytest.mark.skipif(not numpy_available, reason="Numpy not installed")
    def test_mip_abs_objective(self, lp_solver):
        m = tc.get_indexed_pentagonal_pyramid_mip()
        m.x.domain = pe.Reals

        opt = pe.SolverFactory(lp_solver)
        old_results = opt.solve(m, tee=False)
        old_obj = pe.value(m.o)

        new_model = shifted_lp.get_shifted_linear_model(m)
        new_results = opt.solve(new_model, tee=False)
        new_obj = pe.value(new_model.objective)

        assert old_obj == pytest.approx(new_obj)

    def test_polyhedron(self, lp_solver):
        m = tc.get_3d_polyhedron_problem()

        opt = pe.SolverFactory(lp_solver)
        old_results = opt.solve(m, tee=False)
        old_obj = pe.value(m.o)

        new_model = shifted_lp.get_shifted_linear_model(m)
        new_results = opt.solve(new_model, tee=False)
        new_obj = pe.value(new_model.objective)

        assert old_obj == pytest.approx(new_obj)


if __name__ == "__main__":
    unittest.main()
