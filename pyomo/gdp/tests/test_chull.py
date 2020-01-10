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
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn

from pyomo.gdp import *
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as common

import pyomo.opt
linear_solvers = pyomo.opt.check_available_solvers(
    'glpk','cbc','gurobi','cplex')

import random
from six import iteritems, iterkeys, StringIO

# DEBUG
from nose.tools import set_trace

EPS = TransformationFactory('gdp.chull').CONFIG.EPS

def check_linear_coef(self, repn, var, coef):
    var_id = None
    for i,v in enumerate(repn.linear_vars):
        if v is var:
            var_id = i
    self.assertIsNotNone(var_id)
    self.assertEqual(repn.linear_coefs[var_id], coef)

class CommonTests:
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def diff_apply_to_and_create_using(self, model):
        modelcopy = TransformationFactory('gdp.chull').create_using(model)
        modelcopy_buf = StringIO()
        modelcopy.pprint(ostream=modelcopy_buf)
        modelcopy_output = modelcopy_buf.getvalue()

        TransformationFactory('gdp.chull').apply_to(model)
        model_buf = StringIO()
        model.pprint(ostream=model_buf)
        model_output = model_buf.getvalue()
        self.assertMultiLineEqual(modelcopy_output, model_output)

class TwoTermDisj(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed to test unique namer
        random.seed(666)

    def test_transformation_block(self):
        m = models.makeTwoTermDisj_Nonlinear()
        TransformationFactory('gdp.chull').apply_to(m)

        transBlock = m._pyomo_gdp_chull_relaxation
        self.assertIsInstance(transBlock, Block)
        lbub = transBlock.lbub
        self.assertIsInstance(lbub, Set)
        self.assertEqual(lbub, ['lb', 'ub', 'eq'])

        disjBlock = transBlock.relaxedDisjuncts
        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)

    def test_transformation_block_name_collision(self):
        m = models.makeTwoTermDisj_Nonlinear()
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

    def test_indicator_vars_still_active(self):
        m = models.makeTwoTermDisj_Nonlinear()
        TransformationFactory('gdp.chull').apply_to(m)

        self.assertIsInstance(m.d[0].indicator_var, Var)
        self.assertTrue(m.d[0].indicator_var.active)
        self.assertTrue(m.d[0].indicator_var.is_binary())
        self.assertIsInstance(m.d[1].indicator_var, Var)
        self.assertTrue(m.d[1].indicator_var.active)
        self.assertTrue(m.d[1].indicator_var.is_binary())

    def test_disaggregated_vars(self):
        m = models.makeTwoTermDisj_Nonlinear()
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
            self.assertEqual(w.lb, 0)
            self.assertEqual(w.ub, 7)
            self.assertEqual(x.lb, 0)
            self.assertEqual(x.ub, 8)
            self.assertEqual(y.lb, -10)
            self.assertEqual(y.ub, 0)

    def check_furman_et_al_denominator(self, expr, ind_var):
        self.assertEqual(expr._const, EPS)
        self.assertEqual(len(expr._args), 1)
        self.assertEqual(len(expr._coef), 1)
        self.assertEqual(expr._coef[0], 1 - EPS)
        self.assertIs(expr._args[0], ind_var)

    def test_transformed_constraint_nonlinear(self):
        m = models.makeTwoTermDisj_Nonlinear()
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
        repn = generate_standard_repn(cons.body)
        self.assertFalse(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 1)
        # This is a weak test, but as good as any to ensure that the
        # substitution was done correctly
        EPS_1 = 1-EPS
        self.assertEqual(
            str(cons.body),
            "(%s*d[0].indicator_var + %s)*("
            "_pyomo_gdp_chull_relaxation.relaxedDisjuncts[0].x"
            "/(%s*d[0].indicator_var + %s) + "
            "(_pyomo_gdp_chull_relaxation.relaxedDisjuncts[0].y/"
            "(%s*d[0].indicator_var + %s))**2) - "
            "%s*(0.0 + 0.0**2)*(1 - d[0].indicator_var) "
            "- 14.0*d[0].indicator_var"
            % (EPS_1, EPS, EPS_1, EPS, EPS_1, EPS, EPS))

    def test_transformed_constraints_linear(self):
        m = models.makeTwoTermDisj_Nonlinear()
        TransformationFactory('gdp.chull').apply_to(m)

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts

        # the only constraint on the first block is the non-linear one
        c1 = disjBlock[1].component("d[1].c1")
        # has only lb
        self.assertEqual(len(c1), 1)
        cons = c1['lb']
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        check_linear_coef(self, repn, disjBlock[1].x, -1)
        check_linear_coef(self, repn, m.d[1].indicator_var, 2)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(disjBlock[1].x.lb, 0)
        self.assertEqual(disjBlock[1].x.ub, 8)

        c2 = disjBlock[1].component("d[1].c2")
        # 'eq' is preserved
        self.assertEqual(len(c2), 1)
        cons = c2['eq']
        self.assertEqual(cons.lower, 0)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        check_linear_coef(self, repn, disjBlock[1].w, 1)
        check_linear_coef(self, repn, m.d[1].indicator_var, -3)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(disjBlock[1].w.lb, 0)
        self.assertEqual(disjBlock[1].w.ub, 7)

        c3 = disjBlock[1].component("d[1].c3")
        # bounded inequality is split
        self.assertEqual(len(c3), 2)
        cons = c3['lb']
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        check_linear_coef(self, repn, disjBlock[1].x, -1)
        check_linear_coef(self, repn, m.d[1].indicator_var, 1)
        self.assertEqual(repn.constant, 0)

        cons = c3['ub']
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        check_linear_coef(self, repn, disjBlock[1].x, 1)
        check_linear_coef(self, repn, m.d[1].indicator_var, -3)
        self.assertEqual(repn.constant, 0)

    def check_bound_constraints(self, cons, disvar, indvar, lb, ub):
        self.assertIsInstance(cons, Constraint)
        # both lb and ub
        self.assertEqual(len(cons), 2)
        varlb = cons['lb']
        self.assertIsNone(varlb.lower)
        self.assertEqual(varlb.upper, 0)
        repn = generate_standard_repn(varlb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        check_linear_coef(self, repn, indvar, lb)
        check_linear_coef(self, repn, disvar, -1)

        varub = cons['ub']
        self.assertIsNone(varub.lower)
        self.assertEqual(varub.upper, 0)
        repn = generate_standard_repn(varub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        check_linear_coef(self, repn, indvar, -ub)
        check_linear_coef(self, repn, disvar, 1)

    def test_disaggregatedVar_bounds(self):
        m = models.makeTwoTermDisj_Nonlinear()
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
                                         m.d[i].indicator_var, -10, -3)

    def test_xor_constraint(self):
        m = models.makeTwoTermDisj_Nonlinear()
        TransformationFactory('gdp.chull').apply_to(m)

        xorC = m._pyomo_gdp_chull_relaxation.disjunction_xor
        self.assertIsInstance(xorC, Constraint)
        self.assertEqual(len(xorC), 1)

        repn = generate_standard_repn(xorC.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        check_linear_coef(self, repn, m.d[0].indicator_var, 1)
        check_linear_coef(self, repn, m.d[1].indicator_var, 1)

    def test_error_for_or(self):
        m = models.makeTwoTermDisj_Nonlinear()
        m.disjunction.xor = False

        self.assertRaisesRegexp(
            GDP_Error,
            "Cannot do convex hull transformation for disjunction disjunction "
            "with OR constraint. Must be an XOR!*",
            TransformationFactory('gdp.chull').apply_to,
            m)

    def check_disaggregation_constraint(self, cons, var, disvar1, disvar2):
        repn = generate_standard_repn(cons.body)
        self.assertEqual(cons.lower, 0)
        self.assertEqual(cons.upper, 0)
        self.assertEqual(len(repn.linear_vars), 3)
        check_linear_coef(self, repn, var, 1)
        check_linear_coef(self, repn, disvar1, -1)
        check_linear_coef(self, repn, disvar2, -1)

    def test_disaggregation_constraint(self):
        m = models.makeTwoTermDisj_Nonlinear()
        chull = TransformationFactory('gdp.chull')
        chull.apply_to(m)
        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts

        self.check_disaggregation_constraint(
            chull.get_disaggregation_constraint(m.w, m.disjunction), m.w,
            disjBlock[0].w, disjBlock[1].w)
        self.check_disaggregation_constraint(
            chull.get_disaggregation_constraint(m.x, m.disjunction), m.x,
            disjBlock[0].x, disjBlock[1].x)
        self.check_disaggregation_constraint(
            chull.get_disaggregation_constraint(m.y, m.disjunction), m.y,
            disjBlock[0].y, disjBlock[1].y)

    def test_original_disjuncts_deactivated(self):
        m = models.makeTwoTermDisj_Nonlinear()
        TransformationFactory('gdp.chull').apply_to(m, targets=(m,))

        self.assertFalse(m.d.active)
        self.assertFalse(m.d[0].active)
        self.assertFalse(m.d[1].active)
        # Constraints aren't deactived: only disjuncts
        self.assertTrue(m.d[0].c.active)
        self.assertTrue(m.d[1].c1.active)
        self.assertTrue(m.d[1].c2.active)

    def test_transformed_disjunct_mappings(self):
        m = models.makeTwoTermDisj_Nonlinear()
        chull = TransformationFactory('gdp.chull')
        chull.apply_to(m)

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts

        # the disjuncts will always be transformed in the same order,
        # and d[0] goes first, so we can check in a loop.
        for i in [0,1]:
            self.assertIs(disjBlock[i]._srcDisjunct(), m.d[i])
            self.assertIs(chull.get_src_disjunct(disjBlock[i]), m.d[i])

    def test_transformed_constraint_mappings(self):
        m = models.makeTwoTermDisj_Nonlinear()
        chull = TransformationFactory('gdp.chull')
        chull.apply_to(m)

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts

        # first disjunct
        orig1 = m.d[0].c
        trans1 = disjBlock[0].component("d[0].c")
        self.assertIs(chull.get_src_constraint(trans1), orig1)
        self.assertIs(chull.get_transformed_constraint(orig1), trans1)

        # second disjunct
        
        # first constraint
        orig1 = m.d[1].c1
        trans1 = disjBlock[1].component("d[1].c1")
        self.assertIs(chull.get_src_constraint(trans1), orig1)
        self.assertIs(chull.get_transformed_constraint(orig1), trans1)
        
        # second constraint
        orig2 = m.d[1].c2
        trans2 = disjBlock[1].component("d[1].c2")
        self.assertIs(chull.get_src_constraint(trans2), orig2)
        self.assertIs(chull.get_transformed_constraint(orig2), trans2)
        
        # third constraint
        orig3 = m.d[1].c3
        trans3 = disjBlock[1].component("d[1].c3")
        self.assertIs(chull.get_src_constraint(trans3), orig3)
        self.assertIs(chull.get_transformed_constraint(orig3), trans3)

    def test_disaggregatedVar_mappings(self):
        m = models.makeTwoTermDisj_Nonlinear()
        chull = TransformationFactory('gdp.chull')
        chull.apply_to(m)

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts

        for i in [0,1]:
            mappings = ComponentMap()
            mappings[m.w] = disjBlock[i].w
            mappings[m.y] = disjBlock[i].y
            mappings[m.x] = disjBlock[i].x

            for orig, disagg in iteritems(mappings):
                self.assertIs(chull.get_src_var(disagg), orig)
                self.assertIs(chull.get_disaggregated_var(orig, m.d[i]), disagg)

    def test_bigMConstraint_mappings(self):
        m = models.makeTwoTermDisj_Nonlinear()
        chull = TransformationFactory('gdp.chull')
        chull.apply_to(m)

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts

        for i in [0,1]:
            mappings = ComponentMap()
            # [ESJ 11/05/2019] I think this test was useless before... I think
            # this *map* was useless before. It should be disaggregated variable
            # to the constraints, not the original variable? Why did this even
            # work??
            mappings[disjBlock[i].w] = disjBlock[i].w_bounds
            mappings[disjBlock[i].y] = disjBlock[i].y_bounds
            mappings[disjBlock[i].x] = disjBlock[i].x_bounds
            for var, cons in iteritems(mappings):
                self.assertIs(chull.get_var_bounds_constraint(var), cons)

    def test_create_using_nonlinear(self):
        m = models.makeTwoTermDisj_Nonlinear()
        self.diff_apply_to_and_create_using(m)

    def test_var_global_because_objective(self):
        m = models.localVar()
        chull = TransformationFactory('gdp.chull')
        chull.apply_to(m)

        #TODO: desired behavior here has got to be an error about not having
        #bounds on y. We don't know how to tranform this, but the issue is that
        #right now we think we do!
        self.assertTrue(False)

    def test_local_var_not_disaggregated(self):
        m = models.localVar()
        m.del_component(m.objective)
        # now it's legal and we can just ask if we transformed it correctly.
        TransformationFactory('gdp.chull').apply_to(m)

        # check that y was not disaggregated
        self.assertIsNone(m._pyomo_gdp_chull_relaxation.relaxedDisjuncts[0].\
                          component("y"))
        self.assertIsNone(m._pyomo_gdp_chull_relaxation.relaxedDisjuncts[1].\
                          component("y"))
        self.assertEqual(
            len(m._pyomo_gdp_chull_relaxation.disaggregationConstraints), 1)

    def test_do_not_transform_user_deactivated_disjuncts(self):
        m = models.makeTwoTermDisj()
        m.d[0].deactivate()
        chull = TransformationFactory('gdp.chull') 
        chull.apply_to(m, targets=(m,))

        self.assertFalse(m.disjunction.active)
        self.assertFalse(m.d[1].active)

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts
        self.assertIs(disjBlock[0], m.d[1].transformation_block())
        self.assertIs(chull.get_src_disjunct(disjBlock[0]), m.d[1])

    def test_unbounded_var_error(self):
        m = models.makeTwoTermDisj_Nonlinear()
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

    def test_indexed_constraints_in_disjunct(self):
        m = models.makeThreeTermDisj_IndexedConstraints()

        TransformationFactory('gdp.chull').apply_to(m)
        transBlock = m._pyomo_gdp_chull_relaxation

        # 2 blocks: the original Disjunct and the transformation block
        self.assertEqual(
            len(list(m.component_objects(Block, descend_into=False))), 2)
        self.assertEqual(
            len(list(m.component_objects(Disjunct))), 0)

        # Each relaxed disjunct should have 3 vars, but i "d[i].c"
        # Constraints
        for i in [1,2,3]:
            relaxed = transBlock.relaxedDisjuncts[i-1]
            self.assertEqual(len(list(relaxed.component_objects(Var))), 3)
            self.assertEqual(len(list(relaxed.component_data_objects(Var))), 3)
            self.assertEqual(
                len(list(relaxed.component_objects(Constraint))), 4)
            # Note: m.x LB == 0, so only 3 bounds constriants (not 6)
            self.assertEqual(
                len(list(relaxed.component_data_objects(Constraint))), 3+i)
            self.assertEqual(len(relaxed.component('d[%s].c'%i)), i)

    def test_virtual_indexed_constraints_in_disjunct(self):
        m = ConcreteModel()
        m.I = [1,2,3]
        m.x = Var(m.I, bounds=(-1,10))
        def d_rule(d,j):
            m = d.model()
            d.c = Constraint(Any)
            for k in range(j):
                d.c[k+1] = m.x[k+1] >= k+1
        m.d = Disjunct(m.I, rule=d_rule)
        m.disjunction = Disjunction(expr=[m.d[i] for i in m.I])

        TransformationFactory('gdp.chull').apply_to(m)
        transBlock = m._pyomo_gdp_chull_relaxation

        # 2 blocks: the original Disjunct and the transformation block
        self.assertEqual(
            len(list(m.component_objects(Block, descend_into=False))), 2)
        self.assertEqual(
            len(list(m.component_objects(Disjunct))), 0)

        # Each relaxed disjunct should have 3 vars, but i "d[i].c"
        # Constraints
        for i in [1,2,3]:
            relaxed = transBlock.relaxedDisjuncts[i-1]
            self.assertEqual(len(list(relaxed.component_objects(Var))), 3)
            self.assertEqual(len(list(relaxed.component_data_objects(Var))), 3)
            self.assertEqual(
                len(list(relaxed.component_objects(Constraint))), 4)
            self.assertEqual(
                len(list(relaxed.component_data_objects(Constraint))), 3*2+i)
            self.assertEqual(len(relaxed.component('d[%s].c'%i)), i)


class IndexedDisjunction(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_disaggregation_constraints(self):
        m = models.makeTwoTermIndexedDisjunction()
        chull = TransformationFactory('gdp.chull')
        chull.apply_to(m)
        relaxedDisjuncts = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts

        disaggregatedVars = {
            1: [relaxedDisjuncts[0].component('x[1]'),
                relaxedDisjuncts[1].component('x[1]')],
            2: [relaxedDisjuncts[2].component('x[2]'),
                relaxedDisjuncts[3].component('x[2]')],
            3: [relaxedDisjuncts[4].component('x[3]'),
                relaxedDisjuncts[5].component('x[3]')],
        }

        for i, disVars in iteritems(disaggregatedVars):
            cons = chull.get_disaggregation_constraint(m.x[i],
                                                       m.disjunction[i])
            self.assertEqual(cons.lower, 0)
            self.assertEqual(cons.upper, 0)
            repn = generate_standard_repn(cons.body)
            self.assertTrue(repn.is_linear())
            self.assertEqual(repn.constant, 0)
            self.assertEqual(len(repn.linear_vars), 3)
            check_linear_coef(self, repn, m.x[i], 1)
            check_linear_coef(self, repn, disVars[0], -1)
            check_linear_coef(self, repn, disVars[1], -1)

    def test_disaggregation_constraints_tuple_indices(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
        chull = TransformationFactory('gdp.chull')
        chull.apply_to(m)
        relaxedDisjuncts = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts

        disaggregatedVars = {
            (1,'A'): [relaxedDisjuncts[0].component('a[1,A]'),
                      relaxedDisjuncts[1].component('a[1,A]')],
            (1,'B'): [relaxedDisjuncts[2].component('a[1,B]'),
                      relaxedDisjuncts[3].component('a[1,B]')],
            (2,'A'): [relaxedDisjuncts[4].component('a[2,A]'),
                      relaxedDisjuncts[5].component('a[2,A]')],
            (2,'B'): [relaxedDisjuncts[6].component('a[2,B]'),
                      relaxedDisjuncts[7].component('a[2,B]')],
        }

        for i, disVars in iteritems(disaggregatedVars):
            cons = chull.get_disaggregation_constraint(m.a[i],
                                                       m.disjunction[i])
            self.assertEqual(cons.lower, 0)
            self.assertEqual(cons.upper, 0)
            # NOTE: fixed variables are evaluated here.
            repn = generate_standard_repn(cons.body)
            self.assertTrue(repn.is_linear())
            self.assertEqual(repn.constant, 0)
            # The flag=1 disjunct disaggregated variable is fixed to 0, so the
            # below is actually correct:
            self.assertEqual(len(repn.linear_vars), 2)
            check_linear_coef(self, repn, m.a[i], 1)
            check_linear_coef(self, repn, disVars[0], -1)
            self.assertTrue(disVars[1].is_fixed())
            self.assertEqual(value(disVars[1]), 0)

    def test_create_using(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
        self.diff_apply_to_and_create_using(m)

    def test_disjunction_data_target(self):
        m = models.makeThreeTermIndexedDisj()
        TransformationFactory('gdp.chull').apply_to(m, 
                                                    targets=[m.disjunction[2]])

        # we got a transformation block on the model
        transBlock = m.component("_pyomo_gdp_chull_relaxation")
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(transBlock.component("disjunction_xor"),
                              Constraint)
        self.assertIsInstance(transBlock.disjunction_xor[2],
                              constraint._GeneralConstraintData)
        self.assertIsInstance(transBlock.component("relaxedDisjuncts"), Block)
        self.assertEqual(len(transBlock.relaxedDisjuncts), 3)

        # suppose we transform the next one separately
        TransformationFactory('gdp.chull').apply_to(m, 
                                                    targets=[m.disjunction[1]])
        # we added to the same XOR constraint before
        self.assertIsInstance(transBlock.disjunction_xor[1], 
                              constraint._GeneralConstraintData)
        # we used the same transformation block, so we have more relaxed
        # disjuncts
        self.assertEqual(len(transBlock.relaxedDisjuncts), 6)

    def check_relaxation_block(self, m, name, numdisjuncts):
        transBlock = m.component(name)
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(transBlock.component("relaxedDisjuncts"), Block)
        self.assertEqual(len(transBlock.relaxedDisjuncts), numdisjuncts)

    def test_disjunction_data_target_any_index(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-100, 100))
        m.disjunct3 = Disjunct(Any)
        m.disjunct4 = Disjunct(Any)
        m.disjunction2=Disjunction(Any)
        for i in range(2):
            m.disjunct3[i].cons = Constraint(expr=m.x == 2)
            m.disjunct4[i].cons = Constraint(expr=m.x <= 3)
            m.disjunction2[i] = [m.disjunct3[i], m.disjunct4[i]]
        
            TransformationFactory('gdp.chull').apply_to(
                m, targets=[m.disjunction2[i]]) 

            if i == 0:
                self.check_relaxation_block(m, "_pyomo_gdp_chull_relaxation", 2)
            if i == 2:
                self.check_relaxation_block(m, "_pyomo_gdp_chull_relaxation", 4)
    
    def check_trans_block_disjunctions_of_disjunct_datas(self, m):
        transBlock1 = m.component("_pyomo_gdp_chull_relaxation")
        self.assertIsInstance(transBlock1, Block)
        self.assertIsInstance(transBlock1.component("relaxedDisjuncts"), Block)
        # We end up with a transformation block for every SimpleDisjunction or
        # IndexedDisjunction.
        self.assertEqual(len(transBlock1.relaxedDisjuncts), 2)
        self.assertIsInstance(transBlock1.relaxedDisjuncts[0].component("x"),
                              Var)
        self.assertTrue(transBlock1.relaxedDisjuncts[0].x.is_fixed())
        self.assertEqual(value(transBlock1.relaxedDisjuncts[0].x), 0)
        self.assertIsInstance(transBlock1.relaxedDisjuncts[0].component(
            "firstTerm[1].cons"), Constraint)
        # No constraint becuase disaggregated variable fixed to 0
        self.assertEqual(len(transBlock1.relaxedDisjuncts[0].component(
            "firstTerm[1].cons")), 0)
        self.assertIsInstance(transBlock1.relaxedDisjuncts[0].component(
            "x_bounds"), Constraint)
        self.assertEqual(len(transBlock1.relaxedDisjuncts[0].component(
            "x_bounds")), 2)

        self.assertIsInstance(transBlock1.relaxedDisjuncts[1].component("x"),
                              Var)
        self.assertIsInstance(transBlock1.relaxedDisjuncts[1].component(
            "secondTerm[1].cons"), Constraint)
        self.assertEqual(len(transBlock1.relaxedDisjuncts[1].component(
            "secondTerm[1].cons")), 1)
        self.assertIsInstance(transBlock1.relaxedDisjuncts[1].component(
            "x_bounds"), Constraint)
        self.assertEqual(len(transBlock1.relaxedDisjuncts[1].component(
            "x_bounds")), 2)

        transBlock2 = m.component("_pyomo_gdp_chull_relaxation_4")
        self.assertIsInstance(transBlock2, Block)
        self.assertIsInstance(transBlock2.component("relaxedDisjuncts"), Block)
        self.assertEqual(len(transBlock2.relaxedDisjuncts), 2)
        self.assertIsInstance(transBlock2.relaxedDisjuncts[0].component("x"),
                              Var)
        self.assertIsInstance(transBlock2.relaxedDisjuncts[0].component(
            "firstTerm[2].cons"), Constraint)
        # we have an equality constraint
        self.assertEqual(len(transBlock2.relaxedDisjuncts[0].component(
            "firstTerm[2].cons")), 1)
        self.assertIsInstance(transBlock2.relaxedDisjuncts[0].component(
            "x_bounds"), Constraint)
        self.assertEqual(len(transBlock2.relaxedDisjuncts[0].component(
            "x_bounds")), 2)

        self.assertIsInstance(transBlock2.relaxedDisjuncts[1].component("x"),
                              Var)
        self.assertIsInstance(transBlock2.relaxedDisjuncts[1].component(
            "secondTerm[2].cons"), Constraint)
        self.assertEqual(len(transBlock2.relaxedDisjuncts[1].component(
            "secondTerm[2].cons")), 1)
        self.assertIsInstance(transBlock2.relaxedDisjuncts[1].component(
            "x_bounds"), Constraint)
        self.assertEqual(len(transBlock2.relaxedDisjuncts[1].component(
            "x_bounds")), 2)
                        
    def test_simple_disjunction_of_disjunct_datas(self):
        # This is actually a reasonable use case if you are generating
        # disjunctions with the same structure. So you might have Disjuncts
        # indexed by Any and disjunctions indexed by Any and be adding a
        # disjunction of two of the DisjunctDatas in every iteration.
        m = models.makeDisjunctionOfDisjunctDatas()
        TransformationFactory('gdp.chull').apply_to(m)

        self.check_trans_block_disjunctions_of_disjunct_datas(m)
        self.assertIsInstance(
            m._pyomo_gdp_chull_relaxation.component("disjunction_xor"),
            Constraint)
        self.assertIsInstance(
            m._pyomo_gdp_chull_relaxation_4.component("disjunction2_xor"),
            Constraint)

    def test_any_indexed_disjunction_of_disjunct_datas(self):
        m = models.makeAnyIndexedDisjunctionOfDisjunctDatas()
        TransformationFactory('gdp.chull').apply_to(m)

        transBlock = m.component("_pyomo_gdp_chull_relaxation")
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(transBlock.component("relaxedDisjuncts"), Block)
        self.assertEqual(len(transBlock.relaxedDisjuncts), 4)
        self.assertIsInstance(transBlock.relaxedDisjuncts[0].component("x"),
                              Var)
        self.assertTrue(transBlock.relaxedDisjuncts[0].x.is_fixed())
        self.assertEqual(value(transBlock.relaxedDisjuncts[0].x), 0)
        self.assertIsInstance(transBlock.relaxedDisjuncts[0].component(
            "firstTerm[1].cons"), Constraint)
        # No constraint becuase disaggregated variable fixed to 0
        self.assertEqual(len(transBlock.relaxedDisjuncts[0].component(
            "firstTerm[1].cons")), 0)
        self.assertIsInstance(transBlock.relaxedDisjuncts[0].component(
            "x_bounds"), Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[0].component(
            "x_bounds")), 2)

        self.assertIsInstance(transBlock.relaxedDisjuncts[1].component("x"),
                              Var)
        self.assertIsInstance(transBlock.relaxedDisjuncts[1].component(
            "secondTerm[1].cons"), Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[1].component(
            "secondTerm[1].cons")), 1)
        self.assertIsInstance(transBlock.relaxedDisjuncts[1].component(
            "x_bounds"), Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[1].component(
            "x_bounds")), 2)

        self.assertIsInstance(transBlock.relaxedDisjuncts[2].component("x"),
                              Var)
        self.assertIsInstance(transBlock.relaxedDisjuncts[2].component(
            "firstTerm[2].cons"), Constraint)
        # we have an equality constraint
        self.assertEqual(len(transBlock.relaxedDisjuncts[2].component(
            "firstTerm[2].cons")), 1)
        self.assertIsInstance(transBlock.relaxedDisjuncts[2].component(
            "x_bounds"), Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[2].component(
            "x_bounds")), 2)

        self.assertIsInstance(transBlock.relaxedDisjuncts[3].component("x"),
                              Var)
        self.assertIsInstance(transBlock.relaxedDisjuncts[3].component(
            "secondTerm[2].cons"), Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[3].component(
            "secondTerm[2].cons")), 1)
        self.assertIsInstance(transBlock.relaxedDisjuncts[3].component(
            "x_bounds"), Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[3].component(
            "x_bounds")), 2)

        self.assertIsInstance(transBlock.component("disjunction_xor"),
                              Constraint)
        self.assertEqual(len(transBlock.component("disjunction_xor")), 2)

    def check_first_iteration(self, model):
        transBlock = model.component("_pyomo_gdp_chull_relaxation")
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(
            transBlock.component("disjunctionList_xor"), Constraint)
        self.assertEqual(len(transBlock.disjunctionList_xor), 1)
        self.assertFalse(model.disjunctionList[0].active)

        self.assertIsInstance(transBlock.relaxedDisjuncts, Block)
        self.assertEqual(len(transBlock.relaxedDisjuncts), 2)

        self.assertIsInstance(transBlock.relaxedDisjuncts[0].x, Var)
        self.assertTrue(transBlock.relaxedDisjuncts[0].x.is_fixed())
        self.assertEqual(value(transBlock.relaxedDisjuncts[0].x), 0)
        self.assertIsInstance(transBlock.relaxedDisjuncts[0].component(
            "firstTerm[0].cons"), Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[0].component(
            "firstTerm[0].cons")), 0)
        self.assertIsInstance(transBlock.relaxedDisjuncts[0].x_bounds,
                              Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[0].x_bounds), 2)

        self.assertIsInstance(transBlock.relaxedDisjuncts[1].x, Var)
        self.assertFalse(transBlock.relaxedDisjuncts[1].x.is_fixed())
        self.assertIsInstance(transBlock.relaxedDisjuncts[1].component(
            "secondTerm[0].cons"), Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[1].component(
            "secondTerm[0].cons")), 1)
        self.assertIsInstance(transBlock.relaxedDisjuncts[1].x_bounds,
                              Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[1].x_bounds), 2)

    def check_second_iteration(self, model):
        transBlock = model.component("_pyomo_gdp_chull_relaxation")
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(transBlock.component("relaxedDisjuncts"), Block)
        self.assertEqual(len(transBlock.relaxedDisjuncts), 4)
        self.assertIsInstance(transBlock.relaxedDisjuncts[2].component(
            "firstTerm[1].cons"), Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[2].component(
            "firstTerm[1].cons")), 1)
        self.assertIsInstance(transBlock.relaxedDisjuncts[3].component(
            "secondTerm[1].cons"), Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[3].component(
            "secondTerm[1].cons")), 1)
        self.assertEqual(
            len(transBlock.disjunctionList_xor), 2)
        self.assertFalse(model.disjunctionList[1].active)
        self.assertFalse(model.disjunctionList[0].active)

    def test_disjunction_and_disjuncts_indexed_by_any(self):
        model = ConcreteModel()
        model.x = Var(bounds=(-100, 100))

        model.firstTerm = Disjunct(Any)
        model.secondTerm = Disjunct(Any)
        model.disjunctionList = Disjunction(Any)

        model.obj = Objective(expr=model.x)
        
        for i in range(2):
            model.firstTerm[i].cons = Constraint(expr=model.x == 2*i)
            model.secondTerm[i].cons = Constraint(expr=model.x >= i + 2)
            model.disjunctionList[i] = [model.firstTerm[i], model.secondTerm[i]]

            TransformationFactory('gdp.chull').apply_to(model)

            if i == 0:
                self.check_first_iteration(model)

            if i == 1:
                self.check_second_iteration(model)

    def test_iteratively_adding_disjunctions_transform_container(self):
        # If you are iteratively adding Disjunctions to an IndexedDisjunction,
        # then if you are lazy about what you transform, you might shoot
        # yourself in the foot because if the whole IndexedDisjunction gets
        # deactivated by the first transformation, the new DisjunctionDatas
        # don't get transformed. Interestingly, this isn't what happens. We
        # deactivate the container and then still transform what's inside. I
        # don't think we should deactivate the container at all, maybe?
        model = ConcreteModel()
        model.x = Var(bounds=(-100, 100))
        model.disjunctionList = Disjunction(Any)
        model.obj = Objective(expr=model.x)
        for i in range(2):
            firstTermName = "firstTerm[%s]" % i
            model.add_component(firstTermName, Disjunct())
            model.component(firstTermName).cons = Constraint(
                expr=model.x == 2*i)
            secondTermName = "secondTerm[%s]" % i
            model.add_component(secondTermName, Disjunct())
            model.component(secondTermName).cons = Constraint(
                expr=model.x >= i + 2)
            model.disjunctionList[i] = [model.component(firstTermName),
                                        model.component(secondTermName)]

            # we're lazy and we just transform the disjunctionList (and in
            # theory we are transforming at every iteration because we are
            # solving at every iteration)
            TransformationFactory('gdp.chull').apply_to(
                model, targets=[model.disjunctionList])
            if i == 0:
                self.check_first_iteration(model)

            if i == 1:
                self.check_second_iteration(model)

    def test_iteratively_adding_disjunctions_transform_model(self):
        # Same as above, but transforming whole model in every iteration
        model = ConcreteModel()
        model.x = Var(bounds=(-100, 100))
        model.disjunctionList = Disjunction(Any)
        model.obj = Objective(expr=model.x)
        for i in range(2):
            firstTermName = "firstTerm[%s]" % i
            model.add_component(firstTermName, Disjunct())
            model.component(firstTermName).cons = Constraint(
                expr=model.x == 2*i)
            secondTermName = "secondTerm[%s]" % i
            model.add_component(secondTermName, Disjunct())
            model.component(secondTermName).cons = Constraint(
                expr=model.x >= i + 2)
            model.disjunctionList[i] = [model.component(firstTermName),
                                        model.component(secondTermName)]

            # we're lazy and we just transform the model (and in
            # theory we are transforming at every iteration because we are
            # solving at every iteration)
            TransformationFactory('gdp.chull').apply_to(model)
            if i == 0:
                self.check_first_iteration(model)

            if i == 1:
                self.check_second_iteration(model)

    def test_iteratively_adding_to_indexed_disjunction_on_block(self):
        m = ConcreteModel()
        m.b = Block()
        m.b.x = Var(bounds=(-100, 100))
        m.b.firstTerm = Disjunct([1,2])
        m.b.firstTerm[1].cons = Constraint(expr=m.b.x == 0)
        m.b.firstTerm[2].cons = Constraint(expr=m.b.x == 2)
        m.b.secondTerm = Disjunct([1,2])
        m.b.secondTerm[1].cons = Constraint(expr=m.b.x >= 2)
        m.b.secondTerm[2].cons = Constraint(expr=m.b.x >= 3)
        m.b.disjunctionList = Disjunction(Any)

        m.b.obj = Objective(expr=m.b.x)

        for i in range(1,3):
            m.b.disjunctionList[i] = [m.b.firstTerm[i], m.b.secondTerm[i]]

            TransformationFactory('gdp.chull').apply_to(m, targets=[m.b])
            m.b.disjunctionList[i] = [m.b.firstTerm[i], m.b.secondTerm[i]]

            TransformationFactory('gdp.chull').apply_to(m, targets=[m.b])
            
            if i == 1:
                self.check_relaxation_block(m.b, "_pyomo_gdp_chull_relaxation",
                                            2)
            if i == 2:
                self.check_relaxation_block(m.b, "_pyomo_gdp_chull_relaxation",
                                            4)

# NOTE: These are copied from bigm...
class TestTargets_SingleDisjunction(unittest.TestCase, CommonTests):
    def test_only_targets_inactive(self):
        m = models.makeTwoSimpleDisjunctions()
        TransformationFactory('gdp.chull').apply_to(
            m,
            targets=[m.disjunction1])

        self.assertFalse(m.disjunction1.active)
        self.assertIsNotNone(m.disjunction1._algebraic_constraint)
        # disjunction2 still active
        self.assertTrue(m.disjunction2.active)
        self.assertIsNone(m.disjunction2._algebraic_constraint)

        self.assertFalse(m.disjunct1[0].active)
        self.assertFalse(m.disjunct1[1].active)
        self.assertFalse(m.disjunct1.active)
        self.assertTrue(m.disjunct2[0].active)
        self.assertTrue(m.disjunct2[1].active)
        self.assertTrue(m.disjunct2.active)

    def test_only_targets_transformed(self):
        m = models.makeTwoSimpleDisjunctions()
        chull = TransformationFactory('gdp.chull')
        chull.apply_to(
            m,
            targets=[m.disjunction1])

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts
        # only two disjuncts relaxed
        self.assertEqual(len(disjBlock), 2)
        # These aren't the only components that get created, but they are a good
        # enough proxy for which disjuncts got relaxed, which is what we want to
        # check.
        self.assertIsInstance(disjBlock[0].component("disjunct1[0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[1].component("disjunct1[1].c"),
                              Constraint)

        pairs = [
            (0, 0),
            (1, 1)
        ]
        for i, j in pairs:
            self.assertIs(disjBlock[i], m.disjunct1[j].transformation_block())
            self.assertIs(chull.get_src_disjunct(disjBlock[i]), m.disjunct1[j])

        self.assertIsNone(m.disjunct2[0].transformation_block)
        self.assertIsNone(m.disjunct2[1].transformation_block)

    def test_target_not_a_component_err(self):
        decoy = ConcreteModel()
        decoy.block = Block()
        m = models.makeTwoSimpleDisjunctions()
        self.assertRaisesRegexp(
            GDP_Error,
            "Target block is not a component on instance unknown!",
            TransformationFactory('gdp.chull').apply_to,
            m,
            targets=[decoy.block])

    # test that cuid targets still work for now. This and the next test should
    # go away when we actually deprecate CUIDs
    def test_cuid_targets_still_work_for_now(self):
        m = models.makeTwoSimpleDisjunctions()
        chull = TransformationFactory('gdp.chull')
        chull.apply_to(
            m,
            targets=[ComponentUID(m.disjunction1)])

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts
        # only two disjuncts relaxed
        self.assertEqual(len(disjBlock), 2)
        self.assertIsInstance(disjBlock[0].component("disjunct1[0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[1].component("disjunct1[1].c"),
                              Constraint)

        pairs = [
            (0, 0),
            (1, 1)
        ]
        for i, j in pairs:
            self.assertIs(disjBlock[i], m.disjunct1[j].transformation_block())
            self.assertIs(chull.get_src_disjunct(disjBlock[i]), m.disjunct1[j])

        self.assertIsNone(m.disjunct2[0].transformation_block)
        self.assertIsNone(m.disjunct2[1].transformation_block)

    def test_cuid_target_error_still_works_for_now(self):
        m = models.makeTwoSimpleDisjunctions()
        m2 = ConcreteModel()
        m2.oops = Block()
        self.assertRaisesRegexp(
            GDP_Error,
            "Target %s is not a component on the instance!" % 
            ComponentUID(m2.oops),
            TransformationFactory('gdp.chull').apply_to,
            m,
            targets=ComponentUID(m2.oops))

# Also copied from bigm...
class TestTargets_IndexedDisjunction(unittest.TestCase, CommonTests):
    # There are a couple tests for targets above, but since I had the patience
    # to make all these for bigm also, I may as well reap the benefits here too.
    def test_indexedDisj_targets_inactive(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        TransformationFactory('gdp.chull').apply_to(
            m,
            targets=[m.disjunction1])

        self.assertFalse(m.disjunction1.active)
        self.assertFalse(m.disjunction1[1].active)
        self.assertFalse(m.disjunction1[2].active)

        self.assertFalse(m.disjunct1[1,0].active)
        self.assertFalse(m.disjunct1[1,1].active)
        self.assertFalse(m.disjunct1[2,0].active)
        self.assertFalse(m.disjunct1[2,1].active)
        self.assertFalse(m.disjunct1.active)

        self.assertTrue(m.b[0].disjunct[0].active)
        self.assertTrue(m.b[0].disjunct[1].active)
        self.assertTrue(m.b[1].disjunct0.active)
        self.assertTrue(m.b[1].disjunct1.active)

    def test_indexedDisj_only_targets_transformed(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        chull = TransformationFactory('gdp.chull')
        chull.apply_to(
            m,
            targets=[m.disjunction1])

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 4)
        self.assertIsInstance(disjBlock[0].component("disjunct1[1,0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[1].component("disjunct1[1,1].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[2].component("disjunct1[2,0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[3].component("disjunct1[2,1].c"),
                              Constraint)

        # This relies on the disjunctions being transformed in the same order
        # every time. These are the mappings between the indices of the original
        # disjuncts and the indices on the indexed block on the transformation
        # block.
        pairs = [
            ((1,0), 0),
            ((1,1), 1),
            ((2,0), 2),
            ((2,1), 3),
        ]
        for i, j in pairs:
            self.assertIs(chull.get_src_disjunct(disjBlock[j]), m.disjunct1[i])
            self.assertIs(disjBlock[j], m.disjunct1[i].transformation_block())

    def test_warn_for_untransformed(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        def innerdisj_rule(d, flag):
            m = d.model()
            if flag:
                d.c = Constraint(expr=m.a[1] <= 2)
            else:
                d.c = Constraint(expr=m.a[1] >= 65)
        m.disjunct1[1,1].innerdisjunct = Disjunct([0,1], rule=innerdisj_rule)
        m.disjunct1[1,1].innerdisjunction = Disjunction([0],
            rule=lambda a,i: [m.disjunct1[1,1].innerdisjunct[0],
                              m.disjunct1[1,1].innerdisjunct[1]])
        # This test relies on the order that the component objects of
        # the disjunct get considered. In this case, the disjunct
        # causes the error, but in another world, it could be the
        # disjunction, which is also active.
        self.assertRaisesRegexp(
            GDP_Error,
            "Found active disjunct disjunct1\[1,1\].innerdisjunct\[0\] "
            "in disjunct disjunct1\[1,1\]!.*",
            TransformationFactory('gdp.chull').create_using,
            m,
            targets=[m.disjunction1[1]])
        #
        # we will make that disjunction come first now...
        #
        tmp = m.disjunct1[1,1].innerdisjunct
        m.disjunct1[1,1].del_component(tmp)
        m.disjunct1[1,1].add_component('innerdisjunct', tmp)
        self.assertRaisesRegexp(
            GDP_Error,
            "Found untransformed disjunction disjunct1\[1,1\]."
            "innerdisjunction\[0\] in disjunct disjunct1\[1,1\]!.*",
            TransformationFactory('gdp.chull').create_using,
            m,
            targets=[m.disjunction1[1]])
        # Deactivating the disjunction will allow us to get past it back
        # to the Disjunct (after we realize there are no active
        # DisjunctionData within the active Disjunction)
        m.disjunct1[1,1].innerdisjunction[0].deactivate()
        self.assertRaisesRegexp(
            GDP_Error,
            "Found active disjunct disjunct1\[1,1\].innerdisjunct\[0\] "
            "in disjunct disjunct1\[1,1\]!.*",
            TransformationFactory('gdp.chull').create_using,
            m,
            targets=[m.disjunction1[1]])

    def test_disjData_targets_inactive(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        TransformationFactory('gdp.chull').apply_to(
            m,
            targets=[m.disjunction1[2]])
        
        self.assertIsNotNone(m.disjunction1[2]._algebraic_constraint)
        self.assertFalse(m.disjunction1[2].active)

        self.assertTrue(m.disjunct1.active)
        self.assertIsNotNone(m.disjunction1._algebraic_constraint)
        self.assertTrue(m.disjunct1[1,0].active)
        self.assertIsNone(m.disjunct1[1,0]._transformation_block)
        self.assertTrue(m.disjunct1[1,1].active)
        self.assertIsNone(m.disjunct1[1,1]._transformation_block)
        self.assertFalse(m.disjunct1[2,0].active)
        self.assertIsNotNone(m.disjunct1[2,0]._transformation_block)
        self.assertFalse(m.disjunct1[2,1].active)
        self.assertIsNotNone(m.disjunct1[2,1]._transformation_block)

        self.assertTrue(m.b[0].disjunct.active)
        self.assertTrue(m.b[0].disjunct[0].active)
        self.assertIsNone(m.b[0].disjunct[0]._transformation_block)
        self.assertTrue(m.b[0].disjunct[1].active)
        self.assertIsNone(m.b[0].disjunct[1]._transformation_block)
        self.assertTrue(m.b[1].disjunct0.active)
        self.assertIsNone(m.b[1].disjunct0._transformation_block)
        self.assertTrue(m.b[1].disjunct1.active)
        self.assertIsNone(m.b[1].disjunct1._transformation_block)

    def test_disjData_only_targets_transformed(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        chull = TransformationFactory('gdp.chull')
        chull.apply_to(
            m,
            targets=[m.disjunction1[2]])

        disjBlock = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 2)
        self.assertIsInstance(disjBlock[0].component("disjunct1[2,0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[1].component("disjunct1[2,1].c"),
                              Constraint)

        # This relies on the disjunctions being transformed in the same order
        # every time. These are the mappings between the indices of the original
        # disjuncts and the indices on the indexed block on the transformation
        # block.
        pairs = [
            ((2,0), 0),
            ((2,1), 1),
        ]
        for i, j in pairs:
            self.assertIs(m.disjunct1[i].transformation_block(), disjBlock[j])
            self.assertIs(chull.get_src_disjunct(disjBlock[j]), m.disjunct1[i])

    def test_indexedBlock_targets_inactive(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        TransformationFactory('gdp.chull').apply_to(
            m,
            targets=[m.b])

        self.assertTrue(m.disjunct1.active)
        self.assertTrue(m.disjunct1[1,0].active)
        self.assertTrue(m.disjunct1[1,1].active)
        self.assertTrue(m.disjunct1[2,0].active)
        self.assertTrue(m.disjunct1[2,1].active)
        self.assertIsNone(m.disjunct1[1,0].transformation_block)
        self.assertIsNone(m.disjunct1[1,1].transformation_block)
        self.assertIsNone(m.disjunct1[2,0].transformation_block)
        self.assertIsNone(m.disjunct1[2,1].transformation_block)

        self.assertFalse(m.b[0].disjunct.active)
        self.assertFalse(m.b[0].disjunct[0].active)
        self.assertFalse(m.b[0].disjunct[1].active)
        self.assertFalse(m.b[1].disjunct0.active)
        self.assertFalse(m.b[1].disjunct1.active)

    def test_indexedBlock_only_targets_transformed(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        chull = TransformationFactory('gdp.chull')
        chull.apply_to(
            m,
            targets=[m.b])

        disjBlock1 = m.b[0]._pyomo_gdp_chull_relaxation.relaxedDisjuncts
        self.assertEqual(len(disjBlock1), 2)
        self.assertIsInstance(disjBlock1[0].component("b[0].disjunct[0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock1[1].component("b[0].disjunct[1].c"),
                              Constraint)
        disjBlock2 = m.b[1]._pyomo_gdp_chull_relaxation.relaxedDisjuncts
        self.assertEqual(len(disjBlock2), 2)
        self.assertIsInstance(disjBlock2[0].component("b[1].disjunct0.c"),
                              Constraint)
        self.assertIsInstance(disjBlock2[1].component("b[1].disjunct1.c"),
                              Constraint)

        # This relies on the disjunctions being transformed in the same order
        # every time. This dictionary maps the block index to the list of
        # pairs of (originalDisjunctIndex, transBlockIndex)
        pairs = {
            0:
            [
                ('disjunct',0,0),
                ('disjunct',1,1),
            ],
            1:
            [
                ('disjunct0',None,0),
                ('disjunct1',None,1),
            ]
        }

        for blocknum, lst in iteritems(pairs):
            for comp, i, j in lst:
                original = m.b[blocknum].component(comp)
                if blocknum == 0:
                    disjBlock = disjBlock1
                if blocknum == 1:
                    disjBlock = disjBlock2
                self.assertIs(original[i].transformation_block(), disjBlock[j])
                self.assertIs(chull.get_src_disjunct(disjBlock[j]), original[i])

    def checkb0TargetsInactive(self, m):
        self.assertTrue(m.disjunct1.active)
        self.assertTrue(m.disjunct1[1,0].active)
        self.assertTrue(m.disjunct1[1,1].active)
        self.assertTrue(m.disjunct1[2,0].active)
        self.assertTrue(m.disjunct1[2,1].active)

        self.assertFalse(m.b[0].disjunct.active)
        self.assertFalse(m.b[0].disjunct[0].active)
        self.assertFalse(m.b[0].disjunct[1].active)
        self.assertTrue(m.b[1].disjunct0.active)
        self.assertTrue(m.b[1].disjunct1.active)

    def checkb0TargetsTransformed(self, m):
        chull = TransformationFactory('gdp.chull')
        disjBlock = m.b[0]._pyomo_gdp_chull_relaxation.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 2)
        self.assertIsInstance(disjBlock[0].component("b[0].disjunct[0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[1].component("b[0].disjunct[1].c"),
                              Constraint)

        # This relies on the disjunctions being transformed in the same order
        # every time. This dictionary maps the block index to the list of
        # pairs of (originalDisjunctIndex, transBlockIndex)
        pairs = [
                (0,0),
                (1,1),
        ]
        for i, j in pairs:
            self.assertIs(m.b[0].disjunct[i].transformation_block(),
                          disjBlock[j])
            self.assertIs(chull.get_src_disjunct(disjBlock[j]), 
                          m.b[0].disjunct[i])

    def test_blockData_targets_inactive(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        TransformationFactory('gdp.chull').apply_to(
            m,
            targets=[m.b[0]])

        self.checkb0TargetsInactive(m)

    def test_blockData_only_targets_transformed(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        TransformationFactory('gdp.chull').apply_to(
            m,
            targets=[m.b[0]])
        self.checkb0TargetsTransformed(m)

    def test_do_not_transform_deactivated_targets(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        m.b[1].deactivate()
        TransformationFactory('gdp.chull').apply_to(
            m,
            targets=[m.b[0], m.b[1]])

        self.checkb0TargetsInactive(m)
        self.checkb0TargetsTransformed(m)

    def test_create_using(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        self.diff_apply_to_and_create_using(m)

        
class DisaggregatedVarNamingConflict(unittest.TestCase):
    @staticmethod
    def makeModel():
        m = ConcreteModel()
        m.b = Block()
        m.b.x = Var(bounds=(0, 10))
        m.add_component("b.x", Var(bounds=(-9, 9)))
        def disjunct_rule(d, i):
            m = d.model()
            if i:
                d.cons_block = Constraint(expr=m.b.x >= 5)
                d.cons_model = Constraint(expr=m.component("b.x")==0)
            else:
                d.cons_model = Constraint(expr=m.component("b.x") <= -5)
        m.disjunct = Disjunct([0,1], rule=disjunct_rule)
        m.disjunction = Disjunction(expr=[m.disjunct[0], m.disjunct[1]])

        return m

    def test_disaggregation_constraints(self):
        m = self.makeModel()
        chull = TransformationFactory('gdp.chull')
        chull.apply_to(m)

        disaggregationConstraints = m._pyomo_gdp_chull_relaxation.\
                                    disaggregationConstraints

        consmap = [
            (m.component("b.x"), disaggregationConstraints[0]),
            (m.b.x, disaggregationConstraints[1]) 
        ]

        for v, cons in consmap:
            disCons = chull.get_disaggregation_constraint(v, m.disjunction)
            self.assertIs(disCons, cons)
    

class NestedDisjunction(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_deactivated_disjunct_leaves_nested_disjuncts_active(self):
        m = models.makeNestedDisjunctions_FlatDisjuncts()
        m.d1.deactivate()
        # Specifying 'targets' prevents the HACK_GDP_Disjunct_Reclassifier
        # transformation of Disjuncts to Blocks
        TransformationFactory('gdp.chull').apply_to(m, targets=[m])

        self.assertFalse(m.d1.active)
        self.assertTrue(m.d1.indicator_var.fixed)
        self.assertEqual(m.d1.indicator_var.value, 0)

        self.assertFalse(m.d2.active)
        self.assertFalse(m.d2.indicator_var.fixed)

        self.assertTrue(m.d3.active)
        self.assertFalse(m.d3.indicator_var.fixed)

        self.assertTrue(m.d4.active)
        self.assertFalse(m.d4.indicator_var.fixed)

        m = models.makeNestedDisjunctions_NestedDisjuncts()
        m.d1.deactivate()
        # Specifying 'targets' prevents the HACK_GDP_Disjunct_Reclassifier
        # transformation of Disjuncts to Blocks
        TransformationFactory('gdp.chull').apply_to(m, targets=[m])

        self.assertFalse(m.d1.active)
        self.assertTrue(m.d1.indicator_var.fixed)
        self.assertEqual(m.d1.indicator_var.value, 0)

        self.assertFalse(m.d2.active)
        self.assertFalse(m.d2.indicator_var.fixed)

        self.assertTrue(m.d1.d3.active)
        self.assertFalse(m.d1.d3.indicator_var.fixed)

        self.assertTrue(m.d1.d4.active)
        self.assertFalse(m.d1.d4.indicator_var.fixed)

    @unittest.skipIf(not linear_solvers, "No linear solver available")
    def test_relaxation_feasibility(self):
        m = models.makeNestedDisjunctions_FlatDisjuncts()
        TransformationFactory('gdp.chull').apply_to(m)

        solver = SolverFactory(linear_solvers[0])

        cases = [
            (1,1,1,1,None),
            (0,0,0,0,None),
            (1,0,0,0,None),
            (0,1,0,0,1.1),
            (0,0,1,0,None),
            (0,0,0,1,None),
            (1,1,0,0,None),
            (1,0,1,0,1.2),
            (1,0,0,1,1.3),
            (1,0,1,1,None),
            ]
        for case in cases:
            m.d1.indicator_var.fix(case[0])
            m.d2.indicator_var.fix(case[1])
            m.d3.indicator_var.fix(case[2])
            m.d4.indicator_var.fix(case[3])
            results = solver.solve(m)
            print(case, results.solver)
            if case[4] is None:
                self.assertEqual(results.solver.termination_condition,
                                 pyomo.opt.TerminationCondition.infeasible)
            else:
                self.assertEqual(results.solver.termination_condition,
                                 pyomo.opt.TerminationCondition.optimal)
                self.assertEqual(value(m.obj), case[4])

    # TODO: This fails because of the name collision stuf. It seems that
    # apply_to and create_using choose different things in the unique namer,
    # even when I set the seed. Does that make any sense?
    def test_create_using(self):
        m = models.makeNestedDisjunctions_FlatDisjuncts()
        self.diff_apply_to_and_create_using(m)

class TestSpecialCases(unittest.TestCase):
    def test_warn_for_untransformed(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        def innerdisj_rule(d, flag):
            m = d.model()
            if flag:
                d.c = Constraint(expr=m.a[1] <= 2)
            else:
                d.c = Constraint(expr=m.a[1] >= 65)
        m.disjunct1[1,1].innerdisjunct = Disjunct([0,1], rule=innerdisj_rule)
        m.disjunct1[1,1].innerdisjunction = Disjunction([0],
            rule=lambda a,i: [m.disjunct1[1,1].innerdisjunct[0],
                              m.disjunct1[1,1].innerdisjunct[1]])
        # This test relies on the order that the component objects of
        # the disjunct get considered. In this case, the disjunct
        # causes the error, but in another world, it could be the
        # disjunction, which is also active.
        self.assertRaisesRegexp(
            GDP_Error,
            "Found active disjunct disjunct1\[1,1\].innerdisjunct\[0\] "
            "in disjunct disjunct1\[1,1\]!.*",
            TransformationFactory('gdp.chull').create_using,
            m,
            targets=[m.disjunction1[1]])
        #
        # we will make that disjunction come first now...
        #
        tmp = m.disjunct1[1,1].innerdisjunct
        m.disjunct1[1,1].del_component(tmp)
        m.disjunct1[1,1].add_component('innerdisjunct', tmp)
        self.assertRaisesRegexp(
            GDP_Error,
            "Found untransformed disjunction disjunct1\[1,1\]."
            "innerdisjunction\[0\] in disjunct disjunct1\[1,1\]!.*",
            TransformationFactory('gdp.chull').create_using,
            m,
            targets=[m.disjunction1[1]])
        # Deactivating the disjunction will allow us to get past it back
        # to the Disjunct (after we realize there are no active
        # DisjunctionData within the active Disjunction)
        m.disjunct1[1,1].innerdisjunction[0].deactivate()
        self.assertRaisesRegexp(
            GDP_Error,
            "Found active disjunct disjunct1\[1,1\].innerdisjunct\[0\] "
            "in disjunct disjunct1\[1,1\]!.*",
            TransformationFactory('gdp.chull').create_using,
            m,
            targets=[m.disjunction1[1]])

    def test_local_vars(self):
        m = ConcreteModel()
        m.x = Var(bounds=(5,100))
        m.y = Var(bounds=(0,100))
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.y >= m.x)
        m.d2 = Disjunct()
        m.d2.z = Var()
        m.d2.c = Constraint(expr=m.y >= m.d2.z)
        m.disj = Disjunction(expr=[m.d1, m.d2])

        self.assertRaisesRegexp(
            GDP_Error,
            ".*Missing bound for d2.z.*",
            TransformationFactory('gdp.chull').create_using,
            m)
        m.d2.z.setlb(7)
        self.assertRaisesRegexp(
            GDP_Error,
            ".*Missing bound for d2.z.*",
            TransformationFactory('gdp.chull').create_using,
            m)
        m.d2.z.setub(9)

        i = TransformationFactory('gdp.chull').create_using(m)
        rd = i._pyomo_gdp_chull_relaxation.relaxedDisjuncts[1]
        self.assertEqual(sorted(rd.component_map(Var)), ['x','y'])
        self.assertEqual(len(rd.component_map(Constraint)), 4)
        self.assertEqual(i.d2.z.bounds, (0,9))
        self.assertEqual(len(rd.z_bounds), 2)
        self.assertEqual(rd.z_bounds['lb'].lower, None)
        self.assertEqual(rd.z_bounds['lb'].upper, 0)
        self.assertEqual(rd.z_bounds['ub'].lower, None)
        self.assertEqual(rd.z_bounds['ub'].upper, 0)
        i.d2.indicator_var = 1
        i.d2.z = 2
        self.assertEqual(rd.z_bounds['lb'].body(), 5)
        self.assertEqual(rd.z_bounds['ub'].body(), -7)

        m.d2.z.setlb(-9)
        m.d2.z.setub(-7)
        i = TransformationFactory('gdp.chull').create_using(m)
        rd = i._pyomo_gdp_chull_relaxation.relaxedDisjuncts[1]
        self.assertEqual(sorted(rd.component_map(Var)), ['x','y'])
        self.assertEqual(len(rd.component_map(Constraint)), 4)
        self.assertEqual(i.d2.z.bounds, (-9,0))
        self.assertEqual(len(rd.z_bounds), 2)
        self.assertEqual(rd.z_bounds['lb'].lower, None)
        self.assertEqual(rd.z_bounds['lb'].upper, 0)
        self.assertEqual(rd.z_bounds['ub'].lower, None)
        self.assertEqual(rd.z_bounds['ub'].upper, 0)
        i.d2.indicator_var = 1
        i.d2.z = 2
        self.assertEqual(rd.z_bounds['lb'].body(), -11)
        self.assertEqual(rd.z_bounds['ub'].body(), 9)


class RangeSetOnDisjunct(unittest.TestCase):
    def test_RangeSet(self):
        m = models.makeDisjunctWithRangeSet()
        TransformationFactory('gdp.chull').apply_to(m)
        self.assertIsInstance(m.d1.s, RangeSet)


# NOTE: This is copied from bigm. The only thing that changes is bigm -> chull
class TransformABlock(unittest.TestCase, CommonTests):
    # If you transform a block as if it is a model, the transformation should
    # only modify the block you passed it, else when you solve the block, you
    # are missing the disjunction you thought was on there.
    def test_transformation_simple_block(self):
        m = models.makeTwoTermDisjOnBlock()
        TransformationFactory('gdp.chull').apply_to(m.b)

        # transformation block not on m
        self.assertIsNone(m.component("_pyomo_gdp_chull_relaxation"))
        
        # transformation block on m.b
        self.assertIsInstance(m.b.component("_pyomo_gdp_chull_relaxation"), 
                              Block)

    def test_transform_block_data(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        TransformationFactory('gdp.chull').apply_to(m.b[0])

        self.assertIsNone(m.component("_pyomo_gdp_chull_relaxation"))

        self.assertIsInstance(m.b[0].component("_pyomo_gdp_chull_relaxation"),
                              Block)

    def test_simple_block_target(self):
        m = models.makeTwoTermDisjOnBlock()
        TransformationFactory('gdp.chull').apply_to(m, targets=[m.b])

        # transformation block not on m
        self.assertIsNone(m.component("_pyomo_gdp_chull_relaxation"))
        
        # transformation block on m.b
        self.assertIsInstance(m.b.component("_pyomo_gdp_chull_relaxation"),
                              Block)

    def test_block_data_target(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        TransformationFactory('gdp.chull').apply_to(m, targets=[m.b[0]])

        self.assertIsNone(m.component("_pyomo_gdp_chull_relaxation"))

        self.assertIsInstance(m.b[0].component("_pyomo_gdp_chull_relaxation"),
                              Block)

    def test_indexed_block_target(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        TransformationFactory('gdp.chull').apply_to(m, targets=[m.b])

        # We expect the transformation block on each of the BlockDatas. Because
        # it is always going on the parent block of the disjunction.

        self.assertIsNone(m.component("_pyomo_gdp_chull_relaxation"))

        for i in [0,1]:
            self.assertIsInstance(
                m.b[i].component("_pyomo_gdp_chull_relaxation"), Block)

    def add_disj_not_on_block(self, m):
        def simpdisj_rule(disjunct):
            m = disjunct.model()
            disjunct.c = Constraint(expr=m.a >= 3)
        m.simpledisj = Disjunct(rule=simpdisj_rule)
        def simpledisj2_rule(disjunct):
            m = disjunct.model()
            disjunct.c = Constraint(expr=m.a <= 3.5)
        m.simpledisj2 = Disjunct(rule=simpledisj2_rule)
        m.disjunction2 = Disjunction(expr=[m.simpledisj, m.simpledisj2])
        return m

    def test_block_targets_inactive(self):
        m = models.makeTwoTermDisjOnBlock()
        m = self.add_disj_not_on_block(m)
        TransformationFactory('gdp.chull').apply_to(
            m,
            targets=[m.b])

        self.assertFalse(m.b.disjunct[0].active)
        self.assertFalse(m.b.disjunct[1].active)
        self.assertFalse(m.b.disjunct.active)
        self.assertTrue(m.simpledisj.active)
        self.assertTrue(m.simpledisj2.active)

    def test_block_only_targets_transformed(self):
        m = models.makeTwoTermDisjOnBlock()
        m = self.add_disj_not_on_block(m)
        bigm = TransformationFactory('gdp.chull')
        bigm.apply_to(
            m,
            targets=[m.b])

        disjBlock = m.b._pyomo_gdp_chull_relaxation.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 2)
        self.assertIsInstance(disjBlock[0].component("b.disjunct[0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[1].component("b.disjunct[1].c"),
                              Constraint)

        # this relies on the disjuncts being transformed in the same order every
        # time
        pairs = [
            (0,0),
            (1,1),
        ]
        for i, j in pairs:
            self.assertIs(m.b.disjunct[i].transformation_block(), disjBlock[j])
            self.assertIs(bigm.get_src_disjunct(disjBlock[j]), m.b.disjunct[i])

    def test_create_using(self):
        m = models.makeTwoTermDisjOnBlock()
        self.diff_apply_to_and_create_using(m)

class TestErrors(unittest.TestCase):
    # copied from bigm
    def test_ask_for_transformed_constraint_from_untransformed_disjunct(self):
        m = models.makeTwoTermIndexedDisjunction()
        chull = TransformationFactory('gdp.chull')
        chull.apply_to(m, targets=m.disjunction[1])

        self.assertRaisesRegexp(
            GDP_Error,
            "Constraint disjunct\[2,b\].cons_b is on a disjunct which has "
            "not been transformed",
            chull.get_transformed_constraint,
            m.disjunct[2, 'b'].cons_b)

    def test_silly_target(self):
        m = models.makeTwoTermDisj()
        self.assertRaisesRegexp(
            GDP_Error,
            "Target d\[1\].c1 was not a Block, Disjunct, or Disjunction. "
            "It was of type "
            "<class 'pyomo.core.base.constraint.SimpleConstraint'> and "
            "can't be transformed.",
            TransformationFactory('gdp.chull').apply_to,
            m,
            targets=[m.d[1].c1])

    def test_retrieving_nondisjunctive_components(self):
        m = models.makeTwoTermDisj()
        m.b = Block()
        m.b.global_cons = Constraint(expr=m.a + m.x >= 8)
        m.another_global_cons = Constraint(expr=m.a + m.x <= 11)

        chull = TransformationFactory('gdp.chull')
        chull.apply_to(m)

        self.assertRaisesRegexp(
            GDP_Error,
            "Constraint b.global_cons is not on a disjunct and so was not "
            "transformed",
            chull.get_transformed_constraint,
            m.b.global_cons)

        self.assertRaisesRegexp(
            GDP_Error,
            "Constraint b.global_cons is not a transformed constraint",
            chull.get_src_constraint,
            m.b.global_cons)

        self.assertRaisesRegexp(
            GDP_Error,
            "Constraint another_global_cons is not a transformed constraint",
            chull.get_src_constraint,
            m.another_global_cons)
        
        self.assertRaisesRegexp(
            GDP_Error,
            "Block b doesn't appear to be a transformation block for a "
            "disjunct. No source disjunct found.",
            chull.get_src_disjunct,
            m.b)

        self.assertRaisesRegexp(
            GDP_Error,
            "It appears that another_global_cons is not an XOR or OR"
            " constraint resulting from transforming a Disjunction.",
            chull.get_src_disjunction,
            m.another_global_cons)

    # TODO: This isn't actually a problem for chull because we don't need to
    # move anything for nested disjunctions... I catch it in bigm because I
    # don't actually know what to do in that case--I can't get the
    # transformation block. Here I don't care, but is it bad if there is
    # different behavior? Because this is silent in chull.
    # def test_transformed_disjunction_all_disjuncts_deactivated(self):
    #     # I'm not sure I like that I can make this happen...
    #     m = ConcreteModel()
    #     m.x = Var(bounds=(0,8))
    #     m.y = Var(bounds=(0,7))
    #     m.disjunction = Disjunction(expr=[m.x == 0, m.x >= 4])
    #     m.disjunction_disjuncts[0].nestedDisjunction = Disjunction(
    #         expr=[m.y == 6, m.y <= 1])
    #     m.disjunction.disjuncts[0].nestedDisjunction.disjuncts[0].deactivate()
    #     m.disjunction.disjuncts[0].nestedDisjunction.disjuncts[1].deactivate()
    #     TransformationFactory('gdp.chull').apply_to(
    #         m, 
    #         targets=m.disjunction.disjuncts[0].nestedDisjunction)

    #     self.assertRaisesRegexp(
    #         GDP_Error,
    #         "Found transformed disjunction "
    #         "disjunction_disjuncts\[0\].nestedDisjunction on disjunct "
    #         "disjunction_disjuncts\[0\], "
    #         "but none of its disjuncts have been transformed. "
    #         "This is very strange.",
    #         TransformationFactory('gdp.chull').apply_to,
    #         m)

    def test_transform_empty_disjunction(self):
        m = ConcreteModel()
        m.empty = Disjunction(expr=[])
    
        self.assertRaisesRegexp(
            GDP_Error,
            "Disjunction empty is empty. This is likely indicative of a "
            "modeling error.*",
            TransformationFactory('gdp.chull').apply_to,
            m)

    def test_deactivated_disjunct_nonzero_indicator_var(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0,8))
        m.disjunction = Disjunction(expr=[m.x == 0, m.x >= 4])

        m.disjunction.disjuncts[0].deactivate()
        m.disjunction.disjuncts[0].indicator_var.fix(1)

        self.assertRaisesRegexp(
            GDP_Error,
            "The disjunct disjunction_disjuncts\[0\] is deactivated, but the "
            "indicator_var is fixed to 1. This makes no sense.",
            TransformationFactory('gdp.chull').apply_to,
            m)

    def test_deactivated_disjunct_unfixed_indicator_var(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0,8))
        m.disjunction = Disjunction(expr=[m.x == 0, m.x >= 4])

        m.disjunction.disjuncts[0].deactivate()
        m.disjunction.disjuncts[0].indicator_var.fixed = False

        self.assertRaisesRegexp(
            GDP_Error,
            "The disjunct disjunction_disjuncts\[0\] is deactivated, but the "
            "indicator_var is not fixed and the disjunct does not "
            "appear to have been relaxed. This makes no sense. "
            "\(If the intent is to deactivate the disjunct, fix its "
            "indicator_var to 0.\)",
            TransformationFactory('gdp.chull').apply_to,
            m)
