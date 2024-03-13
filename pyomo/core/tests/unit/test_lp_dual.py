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

from pyomo.common.dependencies import scipy_available
import pyomo.common.unittest as unittest
from pyomo.environ import (
    ConcreteModel, 
    Constraint,
    NonNegativeReals,
    NonPositiveReals,
    Objective,
    Reals,
    Suffix,
    TerminationCondition,
    TransformationFactory,
    value,
    Var,
)
from pyomo.opt import SolverFactory, WriterFactory

from pytest import set_trace

@unittest.skipUnless(scipy_available, "Scipy not available")
class TestLPDual(unittest.TestCase):
    @unittest.skipUnless(SolverFactory('gurobi').available(exception_flag=False) and
                         SolverFactory('gurobi').license_is_valid(),
                         "Gurobi is not available")
    def test_lp_dual_solve(self):
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=NonPositiveReals)
        m.z = Var(domain=Reals)

        m.obj = Objective(expr=m.x + 2*m.y - 3*m.z)
        m.c1 = Constraint(expr=-4*m.x - 2*m.y - m.z <= -5)
        m.c2 = Constraint(expr=m.x + m.y <= 3)
        m.c3 = Constraint(expr=- m.y - m.z <= -4.2)
        m.c4 = Constraint(expr=m.z <= 42)
        m.dual = Suffix(direction=Suffix.IMPORT)

        lp_dual = TransformationFactory('core.lp_dual')
        dual = lp_dual.create_using(m)
        dual.dual = Suffix(direction=Suffix.IMPORT)

        opt = SolverFactory('gurobi')
        results = opt.solve(m)
        self.assertEqual(results.solver.termination_condition,
                         TerminationCondition.optimal)
        results = opt.solve(dual)
        self.assertEqual(results.solver.termination_condition,
                         TerminationCondition.optimal)

        self.assertAlmostEqual(value(m.obj), value(dual.obj))
        for idx, cons in enumerate([m.c1, m.c2, m.c3, m.c4]):
            self.assertAlmostEqual(value(dual.x[idx]), value(m.dual[cons]))
        # for idx, (mult, v) in enumerate([(1, m.x), (-1, m.y), (1, m.z)]):
        #     self.assertAlmostEqual(mult*value(v), value(dual.dual[dual_cons]))
