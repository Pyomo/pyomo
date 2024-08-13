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

import pyomo.opt
import pyomo.environ as pe
import pyomo.common.unittest as unittest
import pyomo.contrib.alternative_solutions.aos_utils as au
from pyomo.contrib.alternative_solutions import Solution

pyomo.opt.check_available_solvers("gurobi")
mip_solver = "gurobi"


class TestSolutionUnit(unittest.TestCase):

    def get_model(self):
        """
        Simple model with all variable types and fixed variables to test the
        Solution code.
        """
        m = pe.ConcreteModel()
        m.x = pe.Var(domain=pe.NonNegativeReals)
        m.y = pe.Var(domain=pe.Binary)
        m.z = pe.Var(domain=pe.NonNegativeIntegers)
        m.f = pe.Var(domain=pe.Reals)

        m.f.fix(1)
        m.obj = pe.Objective(expr=m.x + m.y + m.z + m.f, sense=pe.maximize)

        m.con_x = pe.Constraint(expr=m.x <= 1.5)
        m.con_y = pe.Constraint(expr=m.y <= 1)
        m.con_z = pe.Constraint(expr=m.z <= 3)
        return m

    @unittest.skipUnless(
        pe.SolverFactory(mip_solver).available(exception_flag=False),
        "MIP solver not available",
    )
    def test_solution(self):
        """
        Create a Solution Object, call its functions, and ensure the correct
        data is returned.
        """
        model = self.get_model()
        opt = pe.SolverFactory(mip_solver)
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
