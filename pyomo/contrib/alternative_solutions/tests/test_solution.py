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
from pyomo.contrib.alternative_solutions import PyomoSolution

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
        all_vars = au.get_model_variables(model, include_fixed=False)
        obj = au.get_active_objective(model)

        solution = PyomoSolution(variables=all_vars, objective=obj)
        sol_str = """{
    "id": null,
    "objectives": [
        {
            "index": 0,
            "name": "obj",
            "suffix": {},
            "value": 6.5
        }
    ],
    "suffix": {},
    "variables": [
        {
            "discrete": false,
            "fixed": false,
            "index": 0,
            "name": "x",
            "suffix": {},
            "value": 1.5
        },
        {
            "discrete": true,
            "fixed": false,
            "index": 1,
            "name": "y",
            "suffix": {},
            "value": 1
        },
        {
            "discrete": true,
            "fixed": false,
            "index": 2,
            "name": "z",
            "suffix": {},
            "value": 3
        }
    ]
}"""
        assert str(solution) == sol_str

        all_vars = au.get_model_variables(model, include_fixed=True)
        solution = PyomoSolution(variables=all_vars, objective=obj)
        sol_str = """{
    "id": null,
    "objectives": [
        {
            "index": 0,
            "name": "obj",
            "suffix": {},
            "value": 6.5
        }
    ],
    "suffix": {},
    "variables": [
        {
            "discrete": false,
            "fixed": false,
            "index": 0,
            "name": "x",
            "suffix": {},
            "value": 1.5
        },
        {
            "discrete": true,
            "fixed": false,
            "index": 1,
            "name": "y",
            "suffix": {},
            "value": 1
        },
        {
            "discrete": true,
            "fixed": false,
            "index": 2,
            "name": "z",
            "suffix": {},
            "value": 3
        },
        {
            "discrete": false,
            "fixed": true,
            "index": 3,
            "name": "f",
            "suffix": {},
            "value": 1
        }
    ]
}"""
        assert solution.to_string() == sol_str

        sol_val = solution.name_to_variable
        self.assertEqual(set(sol_val.keys()), {"x", "y", "z", "f"})
        self.assertEqual(set(solution.fixed_variable_names), {"f"})


if __name__ == "__main__":
    unittest.main()
