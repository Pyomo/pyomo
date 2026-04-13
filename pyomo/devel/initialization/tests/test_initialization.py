# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from typing import Tuple

import pyomo.environ as pyo
import pyomo.devel.initialization as ini
from pyomo.devel.initialization.examples.init_polynomial_ex import main
from pyomo.common import unittest
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import SolutionStatus, Results, TerminationCondition
from pyomo.contrib.solver.common.base import Availability, SolverBase
import pytest


scip = SolverFactory('scip_direct')
ipopt = SolverFactory('ipopt')


class MockNLPSolver(SolverBase):
    def __init__(self, varlist, sol_map, **kwds) -> None:
        super().__init__(**kwds)
        self.varlist = varlist
        self.sol_map = sol_map
        self.iter = 0

    def available(self) -> Availability:
        return Availability.FullLicense
    
    def version(self) -> Tuple:
        return (1, 0, 0)
    
    def check_solution(self):
        expected, rel_tol, abs_tol = self.sol_map[self.iter]
        self.iter += 1
        for v, val in zip(self.varlist, expected):
            # print(v, v.value, val)
            assert v.value == pytest.approx(val, rel=rel_tol, abs=abs_tol)
    
    def solve(self, model, **kwds) -> Results:
        self.check_solution()
        res = Results()
        res.termination_condition = TerminationCondition.error
        res.solution_status = SolutionStatus.noSolution
        res.incumbent_objective = None
        res.objective_bound = None
        res.solver_name = 'MockNLPSolver'
        res.solver_version = self.version()
        return res


@unittest.skipUnless(scip.available(), 'scip is not available')
@unittest.skipUnless(ipopt.available(), 'ipopt is not available')
class TestExamples(unittest.TestCase):
    def test_poly_global(self):
        stat, x = main(method=ini.InitializationMethod.global_opt)
        self.assertEqual(stat, SolutionStatus.optimal)
        self.assertAlmostEqual(x, -9.920159607881597)

    def test_poly_pwl(self):
        stat, x = main(method=ini.InitializationMethod.pwl_approximation)
        self.assertEqual(stat, SolutionStatus.optimal)
        self.assertAlmostEqual(x, -9.920159607881597)

    def test_poly_lp(self):
        stat, x = main(method=ini.InitializationMethod.lp_approximation)
        self.assertEqual(stat, SolutionStatus.optimal)
        self.assertAlmostEqual(x, -9.920159607881597)


class TestInit(unittest.TestCase):
    def test_lp_init(self):
        """
        For this test, we will create a small linear program,
        but we will make it look nonlinear. Then, the linear
        approximation should be exact. The LP is

        max 3*x1 + 2*x2
        s.t.
            x1 + x2 <= 4
            2*x1 + x2 <= 5
            x1 >= 0
            x2 >= 0

        The solution is

            x1 = 1
            x2 = 3
        """
        m = pyo.ConcreteModel()
        m.x1 = pyo.Var(bounds=(0, 100))
        m.x2 = pyo.Var(bounds=(0, 100))
        m.obj = pyo.Objective(expr=(3*m.x1*m.x1 + 2*m.x2*m.x1)/m.x1, sense=pyo.maximize)
        m.c1 = pyo.Constraint(expr=pyo.exp(pyo.log(m.x1 + m.x2)) <= 4)
        m.c2 = pyo.Constraint(expr=((2*m.x1 + m.x2)**2)**0.5 <= 5)

        # all the actual testing happens in the MockNLPSolver
        nlp_solver = MockNLPSolver(
            varlist=[m.x1, m.x2], 
            sol_map={
                0: ([None, None], 0, 0), 
                1: ([1, 3], 1e-6, 1e-6),
            },
        )
        mip_solver = SolverFactory('highs')
        results = ini.initialize_nlp(
            nlp=m,
            nlp_solver=nlp_solver,
            mip_solver=mip_solver,
            method=ini.InitializationMethod.lp_approximation,
        )

    def test_global_init(self):
        """
        Same as test_lp_init
        """
        m = pyo.ConcreteModel()
        m.x1 = pyo.Var(bounds=(0, 100))
        m.x2 = pyo.Var(bounds=(0, 100))
        m.obj = pyo.Objective(expr=(3*m.x1*m.x1 + 2*m.x2*m.x1)/m.x1, sense=pyo.maximize)
        m.c1 = pyo.Constraint(expr=pyo.exp(pyo.log(m.x1 + m.x2)) <= 4)
        m.c2 = pyo.Constraint(expr=((2*m.x1 + m.x2)**2)**0.5 <= 5)

        # all the actual testing happens in the MockNLPSolver
        nlp_solver = MockNLPSolver(
            varlist=[m.x1, m.x2], 
            sol_map={
                0: ([None, None], 0, 0), 
                1: ([1, 3], 1e-6, 1e-6),
            },
        )
        global_solver = SolverFactory('scip_direct')
        results = ini.initialize_nlp(
            nlp=m,
            nlp_solver=nlp_solver,
            global_solver=global_solver,
            method=ini.InitializationMethod.global_opt,
        )

    def test_pwl_init(self):
        """
        Here, we really just want to make sure that the 
        approximation improves as refinement is done.
        """
        m = pyo.ConcreteModel()
        m.x1 = pyo.Var(bounds=(0.5, 1.5))
        m.x2 = pyo.Var(bounds=(2.5, 3.5))
        m.obj = pyo.Objective(expr=(3*m.x1*m.x1 + 2*m.x2*m.x1)/m.x1, sense=pyo.maximize)
        m.c1 = pyo.Constraint(expr=pyo.exp(pyo.log(m.x1 + m.x2)) <= 4)
        m.c2 = pyo.Constraint(expr=((2*m.x1 + m.x2)**2)**0.5 <= 5)

        # all the actual testing happens in the MockNLPSolver
        nlp_solver = MockNLPSolver(
            varlist=[m.x1, m.x2], 
            sol_map={
                0: ([None, None], 0, 0),
                1: ([1, 3], 1e-0, 1e-0),
                2: ([1, 3], 1e-1, 1e-1),
                3: ([1, 3], 1e-2, 1e-2),
                4: ([1, 3], 1e-3, 1e-3),
                5: ([1, 3], 1e-5, 1e-5),
                6: ([1, 3], 1e-6, 1e-6),
                7: ([1, 3], 1e-7, 1e-7),
                8: ([1, 3], 1e-7, 1e-7),
                9: ([1, 3], 1e-7, 1e-7),
                10: ([1, 3], 1e-7, 1e-7),
                11: ([1, 3], 1e-7, 1e-7),
            },
        )
        mip_solver = SolverFactory('highs')
        results = ini.initialize_nlp(
            nlp=m,
            nlp_solver=nlp_solver,
            mip_solver=mip_solver,
            method=ini.InitializationMethod.pwl_approximation,
            max_pwl_refinement_iter=10,
        )


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    t = TestInit()
    t.test_pwl_init()
