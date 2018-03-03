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
from pyomo.core.base import expr_common, expr as EXPR
import pyomo.opt

import random
from six import StringIO

from nose.tools import set_trace
from pyomo.opt import SolverFactory

solvers = pyomo.opt.check_available_solvers('gurobi')

# TODO:
#     - test that deactivated objectives on the model don't get used by the
#       transformation

class TwoTermDisj(unittest.TestCase):
    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.coopr3_trees)
        # set seed so we can test name collisions predictably
        random.seed(666)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)
    
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

    @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
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

    @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
    def test_cut_constraint(self):
        m = self.makeModel()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        cut = m._pyomo_gdp_cuttingplane_relaxation.cuts[0]
        self.assertEqual(cut.lower, 0)
        self.assertIsNone(cut.upper)

        # test body
        self.assertEqual(len(cut.body._coef), 4)
        self.assertEqual(len(cut.body._args), 4)
        self.assertEqual(cut.body._const, 0)
        
        coefs = {
            0: 0.45,
            1: 0.55,
            2: 0.1,
            3: -0.1
        }

        xhat = {
            0: 2.7,
            1: 1.3,
            2: 0.85,
            3: 0.15
        }

        variables = {
            0: m.x,
            1: m.y,
            2: m.d[0].indicator_var,
            3: m.d[1].indicator_var
        }

        for i in range(4):
            self.assertAlmostEqual(cut.body._coef[i], coefs[i])
            self.assertEqual(len(cut.body._args[i]._coef), 1)
            self.assertEqual(len(cut.body._args[i]._args), 1)
            self.assertAlmostEqual(cut.body._args[i]._const, -1*xhat[i])
            self.assertEqual(cut.body._args[i]._coef[0], 1)
            self.assertIs(cut.body._args[i]._args[0], variables[i])

    @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
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

    @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
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
