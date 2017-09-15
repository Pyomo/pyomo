import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.core.base import expr_common, expr as EXPR
from pyomo.gdp import *

import random
from six import iteritems

# DEBUG
from nose.tools import set_trace

EPS = 1e-2

class TwoTermDisj(unittest.TestCase):
    # make sure that we are using coopr3 expressions...
    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.coopr3_trees)
        # set seed to test unique namer
        random.seed(666)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)

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

    def test_transformation_block_name_collision(self):
        m = self.makeModel()
        # add block with the name we are about to try to use
        m._pyomo_gdp_chull_relaxation = Block(Any)
        TransformationFactory('gdp.chull').apply_to(m)

        # check that we got a uniquely named block
        transBlock = m.component("_pyomo_gdp_chull_relaxation_4")
        self.assertIsInstance(transBlock, Block)

        # check that the relaxed disjuncts really are here.
        disjBlock = transBlock.relaxedDisjuncts
        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)
        self.assertIsInstance(disjBlock[0].component("d[0].c"), Constraint)
        self.assertIsInstance(disjBlock[1].component("d[1].c1"), Constraint)
        self.assertIsInstance(disjBlock[1].component("d[1].c2"), Constraint)
        
        # we didn't add to the block that wasn't ours
        self.assertEqual(len(m._pyomo_gdp_chull_relaxation), 0)

    def test_info_dict_name_collision(self):
        m = self.makeModel()
        # we never have a way to know if the dictionary we made was ours. But we
        # should yell if there is a non-dictionary component of the same name.
        m._gdp_transformation_info = Block()
        self.assertRaisesRegexp(
            GDP_Error, 
            "Component unknown contains an attribute named "
            "_gdp_transformation_info. The transformation requires that it can "
            "create this attribute!*", 
            TransformationFactory('gdp.chull').apply_to,
            m)

    def test_indicator_vars_still_active(self):
        m = self.makeModel()
        TransformationFactory('gdp.chull').apply_to(m)

        self.assertIsInstance(m.d[0].indicator_var, Var)
        self.assertTrue(m.d[0].indicator_var.active)
        self.assertTrue(m.d[0].indicator_var.is_binary())
        self.assertIsInstance(m.d[1].indicator_var, Var)
        self.assertTrue(m.d[1].indicator_var.active)
        self.assertTrue(m.d[1].indicator_var.is_binary())

    def test_disaggregated_vars(self):
        m = self.makeModel()
        TransformationFactory('gdp.chull').apply_to(m)

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts
        # same on both disjuncts
        for i in [0,1]:
            relaxationBlock = disjBlock[i]
            w = relaxationBlock.w
            x = relaxationBlock.x
            y = relaxationBlock.y
            # variables created
            self.assertIsInstance(w, Var)
            self.assertIsInstance(x, Var)
            self.assertIsInstance(y, Var)
            # the are in reals
            self.assertIsInstance(w.domain, RealSet)
            self.assertIsInstance(x.domain, RealSet)
            self.assertIsInstance(y.domain, RealSet)
            # they don't have bounds
            self.assertIsNone(w.lb)
            self.assertIsNone(w.ub)
            self.assertIsNone(x.lb)
            self.assertIsNone(x.ub)
            self.assertIsNone(y.lb)
            self.assertIsNone(y.ub)

    def check_furman_et_al_denominator(self, expr, ind_var):
        self.assertEqual(expr._const, EPS)
        self.assertEqual(len(expr._args), 1)
        self.assertEqual(len(expr._coef), 1)
        self.assertEqual(expr._coef[0], 1 - EPS)
        self.assertIs(expr._args[0], ind_var)

    def test_transformed_constraint_nonlinear(self):
        m = self.makeModel()
        TransformationFactory('gdp.chull').apply_to(m)

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts

        # the only constraint on the first block is the non-linear one
        disj1c = disjBlock[0].component("d[0].c")
        self.assertIsInstance(disj1c, Constraint)
        # we only have an upper bound
        self.assertEqual(len(disj1c), 1)
        cons = disj1c['ub']
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        self.assertEqual(len(cons.body._args), 3)
        self.assertEqual(len(cons.body._coef), 3)
        self.assertEqual(cons.body._coef[0], 1)
        # first term
        firstterm = cons.body._args[0]
        self.assertEqual(len(firstterm._numerator), 2)
        self.assertEqual(len(firstterm._denominator), 0)
        self.check_furman_et_al_denominator(firstterm._numerator[0],
                                       m.d[0].indicator_var)
        sub_part = firstterm._numerator[1]
        self.assertEqual(len(sub_part._coef), 2)
        self.assertEqual(len(sub_part._args), 2)
        self.assertEqual(sub_part._coef[0], 1)
        self.assertEqual(sub_part._coef[1], 1)
        x_part = sub_part._args[0]
        self.assertEqual(len(x_part._numerator), 1)
        self.assertIs(x_part._numerator[0], disjBlock[0].x)
        self.assertEqual(len(x_part._denominator), 1)
        self.check_furman_et_al_denominator(x_part._denominator[0],
                                            m.d[0].indicator_var)
        y_part = sub_part._args[1]
        self.assertEqual(len(y_part._args), 2)
        self.assertEqual(y_part._args[1], 2)
        y_frac = y_part._args[0]
        self.assertEqual(len(y_frac._numerator), 1)
        self.assertIs(y_frac._numerator[0], disjBlock[0].y)
        self.assertEqual(len(y_frac._denominator), 1)
        self.check_furman_et_al_denominator(y_frac._denominator[0],
                                            m.d[0].indicator_var)

        self.assertEqual(cons.body._coef[1], -1)
        secondterm = cons.body._args[1]
        self.assertEqual(len(secondterm._numerator), 2)
        self.assertEqual(len(secondterm._denominator), 0)
        self.assertEqual(secondterm._coef, EPS)
        h0 = secondterm._numerator[0]
        self.assertEqual(len(h0._args), 2)
        self.assertEqual(len(h0._coef), 2)
        self.assertEqual(h0._const, 0)
        self.assertEqual(len(h0._args[1]._args), 2)
        self.assertEqual(h0._args[0], 0)
        self.assertEqual(h0._args[1]._args[0], 0)
        self.assertEqual(h0._args[1]._args[1], 2)
        self.assertEqual(h0._coef[0], 1)
        self.assertEqual(h0._coef[1], 1)

        self.assertEqual(cons.body._coef[2], -14)
        thirdterm = cons.body._args[2]
        self.assertIs(thirdterm, m.d[0].indicator_var)

    def test_transformed_constraints_linear(self):
        m = self.makeModel()
        TransformationFactory('gdp.chull').apply_to(m)

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts

        # the only constraint on the first block is the non-linear one
        c1 = disjBlock[1].component("d[1].c1")
        # has only lb
        self.assertEqual(len(c1), 1)
        cons = c1['lb']
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        self.assertEqual(len(cons.body._args), 2)
        self.assertEqual(len(cons.body._coef), 2)
        self.assertEqual(cons.body._coef[0], 2)
        self.assertIs(cons.body._args[0], m.d[1].indicator_var)
        self.assertEqual(cons.body._coef[1], -1)
        self.assertIs(cons.body._args[1], disjBlock[1].x)

        c2 = disjBlock[1].component("d[1].c2")
        # has both lb and ub
        self.assertEqual(len(c2), 2)
        conslb = c2['lb']
        self.assertIsNone(conslb.lower)
        self.assertEqual(conslb.upper, 0)
        self.assertEqual(len(conslb.body._args), 2)
        self.assertEqual(len(conslb.body._coef), 2)
        self.assertEqual(conslb.body._coef[0], 3)
        self.assertIs(conslb.body._args[0], m.d[1].indicator_var)
        self.assertEqual(conslb.body._coef[1], -1)
        self.assertIs(conslb.body._args[1], disjBlock[1].w)
        consub = c2['ub']
        self.assertIsNone(consub.lower)
        self.assertEqual(consub.upper, 0)
        self.assertEqual(len(consub.body._args), 2)
        self.assertEqual(len(consub.body._coef), 2)
        self.assertEqual(consub.body._coef[0], 1)
        self.assertIs(consub.body._args[0], disjBlock[1].w)
        self.assertEqual(consub.body._coef[1], -3)
        self.assertIs(consub.body._args[1], m.d[1].indicator_var)

    def check_bound_constraints(self, cons, disvar, indvar, lb, ub):
        self.assertIsInstance(cons, Constraint)
        # both lb and ub
        self.assertEqual(len(cons), 2)
        varlb = cons['lb']
        self.assertIsNone(varlb.lower)
        self.assertEqual(varlb.upper, 0)
        self.assertEqual(len(varlb.body._args), 2)
        self.assertEqual(len(varlb.body._coef), 2)
        self.assertEqual(varlb.body._coef[0], lb)
        self.assertIs(varlb.body._args[0], indvar)
        self.assertEqual(varlb.body._coef[1], -1)
        self.assertIs(varlb.body._args[1], disvar)
        varub = cons['ub']
        self.assertIsNone(varub.lower)
        self.assertEqual(varub.upper, 0)
        self.assertEqual(len(varub.body._args), 2)
        self.assertEqual(len(varub.body._coef), 2)
        self.assertEqual(varub.body._coef[0], 1)
        self.assertIs(varub.body._args[0], disvar)
        self.assertEqual(varub.body._coef[1], -1*ub)
        self.assertIs(varub.body._args[1], indvar)

    def test_disaggregatedVar_bounds(self):
        m = self.makeModel()
        TransformationFactory('gdp.chull').apply_to(m)
        
        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts
        for i in [0,1]:
            # check bounds constraints for each variable on each of the two
            # disjuncts.
            self.check_bound_constraints(disjBlock[i].w_bounds, disjBlock[i].w,
                                         m.d[i].indicator_var, 2, 7)
            self.check_bound_constraints(disjBlock[i].x_bounds, disjBlock[i].x,
                                         m.d[i].indicator_var, 1, 8)
            self.check_bound_constraints(disjBlock[i].y_bounds, disjBlock[i].y,
                                         m.d[i].indicator_var, 3, 10)

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
        
    def test_error_for_or(self):
        m = self.makeModel()
        m.disjunction.xor = False

        self.assertRaisesRegexp(
            GDP_Error,
            "Cannot do convex hull transformation for disjunction disjunction "
            "with or constraint. Must be an xor!*",
            TransformationFactory('gdp.chull').apply_to,
            m)

    def check_disaggregation_constraint(self, cons, var, disvar1, disvar2):
        self.assertEqual(cons.lower, 0)
        self.assertEqual(cons.upper, 0)
        self.assertEqual(len(cons.body._args), 3)
        self.assertEqual(len(cons.body._coef), 3)
        self.assertEqual(cons.body._coef[0], 1)
        self.assertIs(cons.body._args[0], var)
        self.assertEqual(cons.body._coef[1], -1)
        self.assertIs(cons.body._args[1], disvar1)
        self.assertEqual(cons.body._coef[2], -1)
        self.assertIs(cons.body._args[2], disvar2)

    def test_disaggregation_constraint(self):
        m = self.makeModel()
        TransformationFactory('gdp.chull').apply_to(m)
        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts

        disCons = m._gdp_chull_relaxation_disjunction_disaggregation
        self.assertIsInstance(disCons, Constraint)
        # one for each of the variables
        self.assertEqual(len(disCons), 3)
        self.check_disaggregation_constraint(disCons['w'], m.w, disjBlock[0].w,
                                             disjBlock[1].w)
        self.check_disaggregation_constraint(disCons['x'], m.x, disjBlock[0].x,
                                             disjBlock[1].x)
        self.check_disaggregation_constraint(disCons['y'], m.y, disjBlock[0].y,
                                             disjBlock[1].y)

    def test_original_disjuncts_deactivated(self):
        m = self.makeModel()
        TransformationFactory('gdp.chull').apply_to(m, targets=(m,))

        self.assertFalse(m.d.active)
        self.assertFalse(m.d[0].active)
        self.assertFalse(m.d[1].active)
        self.assertFalse(m.d[0].c.active)
        self.assertFalse(m.d[1].c1.active)
        self.assertFalse(m.d[1].c2.active)

    def test_transformed_disjunct_mappings(self):
        m = self.makeModel()
        TransformationFactory('gdp.chull').apply_to(m)

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts

        # the disjuncts will always be transformed in the same order,
        # and d[0] goes first, so we can check in a loop.
        for i in [0,1]:
            infodict = disjBlock[i]._gdp_transformation_info
            self.assertIsInstance(infodict, dict)
            self.assertEqual(len(infodict), 4)
            self.assertIs(infodict['src'], m.d[i])
            self.assertIsInstance(infodict['srcConstraints'], ComponentMap)
            self.assertIsInstance(infodict['srcVars'], ComponentMap)
            self.assertIsInstance(
                infodict['boundConstraintToSrcVar'], ComponentMap)

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
            for orig, disagg in iteritems(mappings):
                self.assertIs(srcVars[disagg], orig)
                self.assertIs(disVars[orig], disagg)

    def test_bigMConstraint_mappings(self):
        m = self.makeModel()
        TransformationFactory('gdp.chull').apply_to(m)

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts   

        for i in [0,1]:
            srcBigm = disjBlock[i]._gdp_transformation_info[
                'boundConstraintToSrcVar']
            bigm = m.d[i]._gdp_transformation_info['bigmConstraints']
            self.assertEqual(len(srcBigm), 3)
            self.assertEqual(len(bigm), 3)
            # TODO: this too...
            mappings = ComponentMap()
            mappings[m.w] = disjBlock[i].w_bounds
            mappings[m.y] = disjBlock[i].y_bounds
            mappings[m.x] = disjBlock[i].x_bounds
            for var, cons in iteritems(mappings):
                self.assertIs(srcBigm[cons], var)
                self.assertIs(bigm[var], cons)

    def test_do_not_transform_user_deactivated_disjuncts(self):
        # TODO
        pass

    def test_unbounded_var_error(self):
        m = self.makeModel()
        # no bounds
        m.w.setlb(None)
        m.w.setub(None)
        self.assertRaisesRegexp(
            GDP_Error,
            "Variables that appear in disjuncts must be "
            "bounded in order to use the chull "
            "transformation! Missing bound for w.*",
            TransformationFactory('gdp.chull').apply_to,
            m)


class IndexedDisjunction(unittest.TestCase):
    @staticmethod
    def makeModel(self):
        pass


# class NestedDisjunction(unittest.TestCase):
#     @staticmethod
#     def makeModel():
        

