#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.gdp import *

import pyomo.opt
from pyomo.repn import generate_standard_repn

import random
from six import StringIO

from nose.tools import set_trace

solvers = pyomo.opt.check_available_solvers('ipopt')

# TODO:
#     - test that deactivated objectives on the model don't get used by the
#       transformation

def check_linear_coef(self, repn, var, coef):
    var_id = None
    for i,v in enumerate(repn.linear_vars):
        if v is var:
            var_id = i
    self.assertIsNotNone(var_id)
    self.assertAlmostEqual(repn.linear_coefs[var_id], coef)

class TwoTermDisj(unittest.TestCase):
    @staticmethod
    def makeModel():
        m = ConcreteModel()
        m.x = Var(bounds=(0,5))
        m.y = Var(bounds=(0,5))
        def d_rule(disjunct, flag):
            m = disjunct.model()
            if flag:
                disjunct.c1 = Constraint(expr=1 <= m.x <= 2)
                disjunct.c2 = Constraint(expr=3 <= m.y <= 4)
            else:
                disjunct.c1 = Constraint(expr=3 <= m.x <= 4)
                disjunct.c2 = Constraint(expr=1 <= m.y <= 2)
        m.d = Disjunct([0,1], rule=d_rule)
        def disj_rule(m):
            return [m.d[0], m.d[1]]
        m.disjunction = Disjunction(rule=disj_rule)

        m.obj = Objective(expr=m.x + 2*m.y)
        return m

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_transformation_block(self):
        m = self.makeModel()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        # we created the block
        transBlock = m._pyomo_gdp_cuttingplane_relaxation
        self.assertIsInstance(transBlock, Block)
        # the cuts are on it
        cuts = transBlock.cuts
        self.assertIsInstance(cuts, Constraint)
        # this one adds 4 cuts
        self.assertEqual(len(cuts), 4)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cut_constraint(self):
        m = self.makeModel()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        cut = m._pyomo_gdp_cuttingplane_relaxation.cuts[0]
        self.assertEqual(cut.lower, 0)
        self.assertIsNone(cut.upper)

        # Var, coef, xhat:
        expected_cut = [
            ( m.x, 0.45, 2.7 ),
            ( m.y, 0.55, 1.3 ),
            ( m.d[0].indicator_var, 0.1, 0.85 ),
            ( m.d[1].indicator_var, -0.1, 0.15 ),
        ]

        # test body
        repn = generate_standard_repn(cut.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 4)
        for v, coef, xhat in expected_cut:
            check_linear_coef(self, repn, v, coef)

        self.assertAlmostEqual(
            repn.constant, -1*sum(c*x for v,c,x in expected_cut), 5)


    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_create_using(self):
        m = self.makeModel()

        # TODO: this is duplicate code with other transformation tests
        modelcopy = TransformationFactory('gdp.cuttingplane').create_using(m)
        modelcopy_buf = StringIO()
        modelcopy.pprint(ostream=modelcopy_buf)
        modelcopy_output = modelcopy_buf.getvalue()

        TransformationFactory('gdp.cuttingplane').apply_to(m)
        model_buf = StringIO()
        m.pprint(ostream=model_buf)
        model_output = model_buf.getvalue()
        self.maxDiff = None
        self.assertMultiLineEqual(modelcopy_output, model_output)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_active_objective_err(self):
        m = self.makeModel()
        m.obj.deactivate()
        self.assertRaisesRegexp(
            GDP_Error,
            "Cannot apply cutting planes transformation without an active "
            "objective in the model*",
            TransformationFactory('gdp.cuttingplane').apply_to,
            m
        )
