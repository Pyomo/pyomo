import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.gdp import *

# DEBUG
from nose.tools import set_trace

class TwoTermDisj(unittest.TestCase):
    @staticmethod
    def makeModel():
        m = ConcreteModel()
        m.w = Var(bounds=(2,7))
        m.x = Var(bounds=(1, 8))
        m.y = Var(bounds=(3, 10))
        def d_rule(disjunct, flag):
            m = disjunct.model()
            if flag:
                disjunct.c1 = Constraint(expr=m.x >= 2)
                disjunct.c2 = Constraint(expr=m.w == 3)
            else:
                disjunct.c = Constraint(expr=m.x + m.y**2 <= 14)
        m.d = Disjunct([0,1], rule=d_rule)
        def disj_rule(m):
            return [m.d[0], m.d[1]]
        m.disjunction = Disjunction(rule=disj_rule)
        return m


    def test_transformation_block(self):
        m = self.makeModel()
        TransformationFactory('gdp.chull').apply_to(m)

        transBlock = m._pyomo_gdp_chull_relaxation
        self.assertIsInstance(transBlock, Block)
        lbub = transBlock.lbub
        self.assertIsInstance(lbub, Set)
        self.assertEqual(lbub, ['lb', 'ub'])

        disjBlock = transBlock.relaxedDisjuncts
        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)


    def test_xor_constraint(self):
        m = self.makeModel()
        TransformationFactory('gdp.chull').apply_to(m)

        xorC = m._gdp_chull_relaxation_disjunction_xor
        self.assertIsInstance(xorC, Constraint)
        self.assertEqual(len(xorC), 1)
        
        self.assertEqual(xorC.lower, 1)
        self.assertEqual(xorC.upper, 1)
        self.assertEqual(xorC.body._const, 0)
        self.assertEqual(len(xorC.body._args), 2)
        self.assertEqual(len(xorC.body._coef), 2)
        self.assertIs(xorC.body._args[0], m.d[0].indicator_var)
        self.assertIs(xorC.body._args[1], m.d[1].indicator_var)
        self.assertEqual(xorC.body._coef[0], 1)
        self.assertEqual(xorC.body._coef[1], 1)


    def test_transformed_disjunct_mappings(self):
        m = self.makeModel()
        TransformationFactory('gdp.chull').apply_to(m)

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts

        # the disjuncts will always be transformed in the same order,
        # and d[0] goes first, so we can check in a loop.
        for i in [0,1]:
            infodict = disjBlock[i]._gdp_transformation_info
            self.assertIsInstance(infodict, dict)
            self.assertEqual(len(infodict), 3)
            self.assertIs(infodict['src'], m.d[i])
            self.assertIsInstance(infodict['srcConstraints'], ComponentMap)
            self.assertIsInstance(infodict['srcVars'], ComponentMap)

            disjDict = m.d[i]._gdp_transformation_info
            self.assertIsInstance(disjDict, dict)
            self.assertEqual(len(disjDict), 5)
            self.assertTrue(disjDict['relaxed'])
            self.assertIs(disjDict['chull'], disjBlock[i])
            disaggregatedVars = disjDict['disaggregatedVars']
            self.assertIsInstance(disaggregatedVars, ComponentMap)
            bigmConstraints = disjDict['bigmConstraints']
            self.assertIsInstance(bigmConstraints, ComponentMap)
            relaxedConstraints = disjDict['relaxedConstraints']
            self.assertIsInstance(relaxedConstraints, ComponentMap)


    def test_transformed_constraint_mappings(self):
        m = self.makeModel()
        TransformationFactory('gdp.chull').apply_to(m)

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts

        # first disjunct
        srcConsdict = disjBlock[0]._gdp_transformation_info['srcConstraints']
        transConsdict = m.d[0]._gdp_transformation_info['relaxedConstraints']

        self.assertEqual(len(srcConsdict), 1)
        self.assertEqual(len(transConsdict), 1)
        orig1 = m.d[0].c
        trans1 = disjBlock[0].component("d[0].c")
        self.assertIs(srcConsdict[trans1], orig1)
        self.assertIs(transConsdict[orig1], trans1)
        
        # second disjunct
        srcConsdict = disjBlock[1]._gdp_transformation_info['srcConstraints']
        transConsdict = m.d[1]._gdp_transformation_info['relaxedConstraints']

        self.assertEqual(len(srcConsdict), 2)
        self.assertEqual(len(transConsdict), 2)
        # first constraint
        orig1 = m.d[1].c1
        trans1 = disjBlock[1].component("d[1].c1")
        self.assertIs(srcConsdict[trans1], orig1)
        self.assertIs(transConsdict[orig1], trans1)
        # second constraint
        orig2 = m.d[1].c2
        trans2 = disjBlock[1].component("d[1].c2")
        self.assertIs(srcConsdict[trans2], orig2)
        self.assertIs(transConsdict[orig2], trans2)


    def test_disaggregatedVar_mappings(self):
        m = self.makeModel()
        TransformationFactory('gdp.chull').apply_to(m)

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts

        srcVars = disjBlock[0]._gdp_transformation_info['srcVars']
        disVars = m.d[0]._gdp_transformation_info['disaggregatedVars']

        for i in [0,1]:
            srcVars = disjBlock[i]._gdp_transformation_info['srcVars']
            disVars = m.d[i]._gdp_transformation_info['disaggregatedVars']
            self.assertEqual(len(srcVars), 3)
            self.assertEqual(len(disVars), 3)
            # TODO: there has got to be better syntax for this??
            mappings = ComponentMap()
            mappings[m.w] = disjBlock[i].w
            mappings[m.y] = disjBlock[i].y
            mappings[m.x] = disjBlock[i].x
            for orig, disagg in mappings.iteritems():
                self.assertIs(srcVars[disagg], orig)
                self.assertIs(disVars[orig], disagg)


    def test_bigMConstraint_mappings(self):
        m = self.makeModel()
        TransformationFactory('gdp.chull').apply_to(m)

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts   

        # TODO

        

# class NestedDisjunction(unittest.TestCase):
#     @staticmethod
#     def makeModel():
        

