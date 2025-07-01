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

import pyomo.opt
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
import pyomo.contrib.alternative_solutions.aos_utils as au
from pyomo.contrib.alternative_solutions import Solution

mip_solver = "gurobi"
mip_available = pyomo.opt.check_available_solvers(mip_solver)


class TestSolutionUnit(unittest.TestCase):

    def get_model(self):
        """
        Simple model with all variable types and fixed variables to test the
        Solution code.
        """
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.y = pyo.Var(domain=pyo.Binary)
        m.z = pyo.Var(domain=pyo.NonNegativeIntegers)
        m.f = pyo.Var(domain=pyo.Reals)

        m.f.fix(1)
        m.obj = pyo.Objective(expr=m.x + m.y + m.z + m.f, sense=pyo.maximize)

        m.con_x = pyo.Constraint(expr=m.x <= 1.5)
        m.con_y = pyo.Constraint(expr=m.y <= 1)
        m.con_z = pyo.Constraint(expr=m.z <= 3)
        return m

    @unittest.skipUnless(mip_available, "MIP solver not available")
    def test_solution(self):
        """
        Create a Solution Object, call its functions, and ensure the correct
        data is returned.
        """
        model = self.get_model()
        opt = pyo.SolverFactory(mip_solver)
        opt.solve(model)
        all_vars = au.get_model_variables(model, include_fixed=True)

        solution = Solution(model, all_vars, include_fixed=False)
        sol_str = """{
    "fixed_variables": [
        "f"
    ],
    "objective": "obj",
    "objective_value": 6.5,
    "solution": {
        "x": 1.5,
        "y": 1,
        "z": 3
    }
}"""
        assert str(solution) == sol_str

        solution = Solution(model, all_vars)
        sol_str = """{
    "fixed_variables": [
        "f"
    ],
    "objective": "obj",
    "objective_value": 6.5,
    "solution": {
        "f": 1,
        "x": 1.5,
        "y": 1,
        "z": 3
    }
}"""
        assert solution.to_string(round_discrete=True) == sol_str

        sol_val = solution.get_variable_name_values(
            include_fixed=True, round_discrete=True
        )
        self.assertEqual(set(sol_val.keys()), {"x", "y", "z", "f"})
        self.assertEqual(set(solution.get_fixed_variable_names()), {"f"})


if __name__ == "__main__":
    unittest.main()
