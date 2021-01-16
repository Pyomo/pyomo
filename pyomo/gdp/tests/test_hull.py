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
from pyomo.common.log import LoggingIntercept
import logging

from pyomo.environ import (TransformationFactory, Block, Set, Constraint, Var,
                           RealSet, ComponentMap, value, log, ConcreteModel,
                           Any, Suffix, SolverFactory, RangeSet, Param,
                           Objective, TerminationCondition)
from pyomo.repn import generate_standard_repn

from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct

import pyomo.opt
linear_solvers = pyomo.opt.check_available_solvers(
    'glpk','cbc','gurobi','cplex')

import random
from six import iteritems, StringIO

EPS = TransformationFactory('gdp.hull').CONFIG.EPS

class CommonTests:
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def diff_apply_to_and_create_using(self, model):
        ct.diff_apply_to_and_create_using(self, model, 'gdp.hull')

class TwoTermDisj(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed to test unique namer
        random.seed(666)

    def test_transformation_block(self):
        m = models.makeTwoTermDisj_Nonlinear()
        TransformationFactory('gdp.hull').apply_to(m)

        transBlock = m._pyomo_gdp_hull_reformulation
        self.assertIsInstance(transBlock, Block)
        lbub = transBlock.lbub
        self.assertIsInstance(lbub, Set)
        self.assertEqual(lbub, ['lb', 'ub', 'eq'])

        disjBlock = transBlock.relaxedDisjuncts
        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)

    def test_transformation_block_name_collision(self):
        ct.check_transformation_block_name_collision(self, 'hull')

    def test_disaggregated_vars(self):
        m = models.makeTwoTermDisj_Nonlinear()
        TransformationFactory('gdp.hull').apply_to(m)

        disjBlock = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts
        # same on both disjuncts
        for i in [0,1]:
            relaxationBlock = disjBlock[i]
            w = relaxationBlock.disaggregatedVars.w
            x = relaxationBlock.disaggregatedVars.x
            y = relaxationBlock.disaggregatedVars.y
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
        TransformationFactory('gdp.hull').apply_to(m)

        disjBlock = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts

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
            "_pyomo_gdp_hull_reformulation.relaxedDisjuncts[0]."
            "disaggregatedVars.x"
            "/(%s*d[0].indicator_var + %s) + "
            "(_pyomo_gdp_hull_reformulation.relaxedDisjuncts[0]."
            "disaggregatedVars.y/"
            "(%s*d[0].indicator_var + %s))**2) - "
            "%s*(0.0 + 0.0**2)*(1 - d[0].indicator_var) "
            "- 14.0*d[0].indicator_var"
            % (EPS_1, EPS, EPS_1, EPS, EPS_1, EPS, EPS))

    def test_transformed_constraints_linear(self):
        m = models.makeTwoTermDisj_Nonlinear()
        TransformationFactory('gdp.hull').apply_to(m)

        disjBlock = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts

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
        ct.check_linear_coef(self, repn, disjBlock[1].disaggregatedVars.x, -1)
        ct.check_linear_coef(self, repn, m.d[1].indicator_var, 2)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(disjBlock[1].disaggregatedVars.x.lb, 0)
        self.assertEqual(disjBlock[1].disaggregatedVars.x.ub, 8)

        c2 = disjBlock[1].component("d[1].c2")
        # 'eq' is preserved
        self.assertEqual(len(c2), 1)
        cons = c2['eq']
        self.assertEqual(cons.lower, 0)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, disjBlock[1].disaggregatedVars.w, 1)
        ct.check_linear_coef(self, repn, m.d[1].indicator_var, -3)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(disjBlock[1].disaggregatedVars.w.lb, 0)
        self.assertEqual(disjBlock[1].disaggregatedVars.w.ub, 7)

        c3 = disjBlock[1].component("d[1].c3")
        # bounded inequality is split
        self.assertEqual(len(c3), 2)
        cons = c3['lb']
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, disjBlock[1].disaggregatedVars.x, -1)
        ct.check_linear_coef(self, repn, m.d[1].indicator_var, 1)
        self.assertEqual(repn.constant, 0)

        cons = c3['ub']
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, disjBlock[1].disaggregatedVars.x, 1)
        ct.check_linear_coef(self, repn, m.d[1].indicator_var, -3)
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
        ct.check_linear_coef(self, repn, indvar, lb)
        ct.check_linear_coef(self, repn, disvar, -1)

        varub = cons['ub']
        self.assertIsNone(varub.lower)
        self.assertEqual(varub.upper, 0)
        repn = generate_standard_repn(varub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, indvar, -ub)
        ct.check_linear_coef(self, repn, disvar, 1)

    def test_disaggregatedVar_bounds(self):
        m = models.makeTwoTermDisj_Nonlinear()
        TransformationFactory('gdp.hull').apply_to(m)

        disjBlock = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts
        for i in [0,1]:
            # check bounds constraints for each variable on each of the two
            # disjuncts.
            self.check_bound_constraints(disjBlock[i].w_bounds,
                                         disjBlock[i].disaggregatedVars.w,
                                         m.d[i].indicator_var, 2, 7)
            self.check_bound_constraints(disjBlock[i].x_bounds,
                                         disjBlock[i].disaggregatedVars.x,
                                         m.d[i].indicator_var, 1, 8)
            self.check_bound_constraints(disjBlock[i].y_bounds,
                                         disjBlock[i].disaggregatedVars.y,
                                         m.d[i].indicator_var, -10, -3)

    def test_error_for_or(self):
        m = models.makeTwoTermDisj_Nonlinear()
        m.disjunction.xor = False

        self.assertRaisesRegexp(
            GDP_Error,
            "Cannot do hull reformulation for Disjunction "
            "'disjunction' with OR constraint.  Must be an XOR!*",
            TransformationFactory('gdp.hull').apply_to,
            m)

    def check_disaggregation_constraint(self, cons, var, disvar1, disvar2):
        repn = generate_standard_repn(cons.body)
        self.assertEqual(cons.lower, 0)
        self.assertEqual(cons.upper, 0)
        self.assertEqual(len(repn.linear_vars), 3)
        ct.check_linear_coef(self, repn, var, 1)
        ct.check_linear_coef(self, repn, disvar1, -1)
        ct.check_linear_coef(self, repn, disvar2, -1)

    def test_disaggregation_constraint(self):
        m = models.makeTwoTermDisj_Nonlinear()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        disjBlock = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts

        self.check_disaggregation_constraint(
            hull.get_disaggregation_constraint(m.w, m.disjunction), m.w,
            disjBlock[0].disaggregatedVars.w, disjBlock[1].disaggregatedVars.w)
        self.check_disaggregation_constraint(
            hull.get_disaggregation_constraint(m.x, m.disjunction), m.x,
            disjBlock[0].disaggregatedVars.x, disjBlock[1].disaggregatedVars.x)
        self.check_disaggregation_constraint(
            hull.get_disaggregation_constraint(m.y, m.disjunction), m.y,
            disjBlock[0].disaggregatedVars.y, disjBlock[1].disaggregatedVars.y)

    def test_xor_constraint_mapping(self):
        ct.check_xor_constraint_mapping(self, 'hull')

    def test_xor_constraint_mapping_two_disjunctions(self):
        ct.check_xor_constraint_mapping_two_disjunctions(self, 'hull')

    def test_transformed_disjunct_mappings(self):
        ct.check_disjunct_mapping(self, 'hull')

    def test_transformed_constraint_mappings(self):
        # ESJ: Letting bigm and hull test their own constraint mappings
        # because, though the paradigm is the same, hull doesn't always create
        # a transformed constraint when it can instead accomplish an x == 0
        # constraint by fixing the disaggregated variable.
        m = models.makeTwoTermDisj_Nonlinear()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        disjBlock = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts

        # first disjunct
        orig1 = m.d[0].c
        trans1 = disjBlock[0].component("d[0].c")
        self.assertIs(hull.get_src_constraint(trans1), orig1)
        self.assertIs(hull.get_src_constraint(trans1['ub']), orig1)
        trans_list = hull.get_transformed_constraints(orig1)
        self.assertEqual(len(trans_list), 1)
        self.assertIs(trans_list[0], trans1['ub'])

        # second disjunct

        # first constraint
        orig1 = m.d[1].c1
        trans1 = disjBlock[1].component("d[1].c1")
        self.assertIs(hull.get_src_constraint(trans1), orig1)
        self.assertIs(hull.get_src_constraint(trans1['lb']), orig1)
        trans_list = hull.get_transformed_constraints(orig1)
        self.assertEqual(len(trans_list), 1)
        self.assertIs(trans_list[0], trans1['lb'])

        # second constraint
        orig2 = m.d[1].c2
        trans2 = disjBlock[1].component("d[1].c2")
        self.assertIs(hull.get_src_constraint(trans2), orig2)
        self.assertIs(hull.get_src_constraint(trans2['eq']), orig2)
        trans_list = hull.get_transformed_constraints(orig2)
        self.assertEqual(len(trans_list), 1)
        self.assertIs(trans_list[0], trans2['eq'])

        # third constraint
        orig3 = m.d[1].c3
        trans3 = disjBlock[1].component("d[1].c3")
        self.assertIs(hull.get_src_constraint(trans3), orig3)
        self.assertIs(hull.get_src_constraint(trans3['lb']), orig3)
        self.assertIs(hull.get_src_constraint(trans3['ub']), orig3)
        trans_list = hull.get_transformed_constraints(orig3)
        self.assertEqual(len(trans_list), 2)
        self.assertIs(trans_list[0], trans3['lb'])
        self.assertIs(trans_list[1], trans3['ub'])

    def test_disaggregatedVar_mappings(self):
        m = models.makeTwoTermDisj_Nonlinear()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        disjBlock = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts

        for i in [0,1]:
            mappings = ComponentMap()
            mappings[m.w] = disjBlock[i].disaggregatedVars.w
            mappings[m.y] = disjBlock[i].disaggregatedVars.y
            mappings[m.x] = disjBlock[i].disaggregatedVars.x

            for orig, disagg in iteritems(mappings):
                self.assertIs(hull.get_src_var(disagg), orig)
                self.assertIs(hull.get_disaggregated_var(orig, m.d[i]), disagg)

    def test_bigMConstraint_mappings(self):
        m = models.makeTwoTermDisj_Nonlinear()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        disjBlock = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts

        for i in [0,1]:
            mappings = ComponentMap()
            # [ESJ 11/05/2019] I think this test was useless before... I think
            # this *map* was useless before. It should be disaggregated variable
            # to the constraints, not the original variable? Why did this even
            # work??
            mappings[disjBlock[i].disaggregatedVars.w] = disjBlock[i].w_bounds
            mappings[disjBlock[i].disaggregatedVars.y] = disjBlock[i].y_bounds
            mappings[disjBlock[i].disaggregatedVars.x] = disjBlock[i].x_bounds
            for var, cons in iteritems(mappings):
                self.assertIs(hull.get_var_bounds_constraint(var), cons)

    def test_create_using_nonlinear(self):
        m = models.makeTwoTermDisj_Nonlinear()
        self.diff_apply_to_and_create_using(m)

    # [ESJ 02/14/2020] In order to match bigm and the (unfortunate) expectation
    # we have established, we never decide something is local based on where it
    # is declared. We treat variables declared on Disjuncts as if they are
    # declared globally. We need to use the bounds as if they are global and
    # also disaggregate the variable
    def test_locally_declared_var_bounds_used_globally(self):
        m = models.localVar()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        # check that we used the bounds on the local variable as if they are
        # global. Which means checking the bounds constraints...
        y_disagg = m.disj2.transformation_block().disaggregatedVars.y
        cons = hull.get_var_bounds_constraint(y_disagg)
        lb = cons['lb']
        self.assertIsNone(lb.lower)
        self.assertEqual(value(lb.upper), 0)
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        ct.check_linear_coef(self, repn, m.disj2.indicator_var, 1)
        ct.check_linear_coef(self, repn, y_disagg, -1)

        ub = cons['ub']
        self.assertIsNone(ub.lower)
        self.assertEqual(value(ub.upper), 0)
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        ct.check_linear_coef(self, repn, y_disagg, 1)
        ct.check_linear_coef(self, repn, m.disj2.indicator_var, -3)

    def test_locally_declared_variables_disaggregated(self):
        m = models.localVar()

        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        # two birds one stone: test the mappings too
        disj1y = hull.get_disaggregated_var(m.disj2.y, m.disj1)
        disj2y = hull.get_disaggregated_var(m.disj2.y, m.disj2)
        self.assertIs(disj1y, 
                      m.disj1._transformation_block().disaggregatedVars.y)
        self.assertIs(disj2y,
                      m.disj2._transformation_block().disaggregatedVars.y)
        self.assertIs(hull.get_src_var(disj1y), m.disj2.y)
        self.assertIs(hull.get_src_var(disj2y), m.disj2.y)

    def test_global_vars_local_to_a_disjunction_disaggregated(self):
        # The point of this is that where a variable is declared has absolutely
        # nothing to do with whether or not it should be disaggregated. With the
        # only exception being that we can tell disaggregated variables and we
        # know they are really and truly local to only one disjunct (EVER, in the
        # whole model) because we declared them.

        # So here, for some perverse reason, we declare the variables on disj1,
        # but we use them in disj2. Both of them need to be disaggregated in
        # both disjunctions though: Neither is local. (And, unless we want to do
        # a search of the whole model (or disallow this kind of insanity) we
        # can't be smarter because what if you transformed this one disjunction
        # at a time? You can never assume a variable isn't used elsewhere in the
        # model, and if it is, you must disaggregate it.)
        m = ConcreteModel()
        m.disj1 = Disjunct()
        m.disj1.x = Var(bounds=(1, 10))
        m.disj1.y = Var(bounds=(2, 11))
        m.disj1.cons1 = Constraint(expr=m.disj1.x + m.disj1.y <= 5)
        m.disj2 = Disjunct()
        m.disj2.cons = Constraint(expr=m.disj1.y >= 8)
        m.disjunction1 = Disjunction(expr=[m.disj1, m.disj2])

        m.disj3 = Disjunct()
        m.disj3.cons = Constraint(expr=m.disj1.x >= 7)
        m.disj4 = Disjunct()
        m.disj4.cons = Constraint(expr=m.disj1.y == 3)
        m.disjunction2 = Disjunction(expr=[m.disj3, m.disj4])

        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        # check that all the variables are disaggregated
        for disj in [m.disj1, m.disj2, m.disj3, m.disj4]:
            transBlock = disj.transformation_block()
            varBlock = transBlock.disaggregatedVars
            self.assertEqual(len([v for v in
                                  varBlock.component_data_objects(Var)]), 2)
            x = varBlock.component("x")
            y = varBlock.component("y")
            self.assertIsInstance(x, Var)
            self.assertIsInstance(y, Var)
            self.assertIs(hull.get_disaggregated_var(m.disj1.x, disj), x)
            self.assertIs(hull.get_src_var(x), m.disj1.x)
            self.assertIs(hull.get_disaggregated_var(m.disj1.y, disj), y)
            self.assertIs(hull.get_src_var(y), m.disj1.y)

    def check_name_collision_disaggregated_vars(self, m, disj, name):
        hull = TransformationFactory('gdp.hull')
        transBlock = disj.transformation_block()
        varBlock = transBlock.disaggregatedVars
        self.assertEqual(len([v for v in
                              varBlock.component_data_objects(Var)]), 2)
        x = varBlock.component("x")
        x2 = varBlock.component(name)
        self.assertIsInstance(x, Var)
        self.assertIsInstance(x2, Var)
        self.assertIs(hull.get_disaggregated_var(m.disj1.x, disj), x)
        self.assertIs(hull.get_src_var(x), m.disj1.x)
        self.assertIs(hull.get_disaggregated_var(m.x, disj), x2)
        self.assertIs(hull.get_src_var(x2), m.x)

    def test_disaggregated_var_name_collision(self):
        # same model as the test above, but now I am putting what was disj1.y
        # as m.x, just to invite disaster.
        m = ConcreteModel()
        m.x = Var(bounds=(2, 11))
        m.disj1 = Disjunct()
        m.disj1.x = Var(bounds=(1, 10))
        m.disj1.cons1 = Constraint(expr=m.disj1.x + m.x <= 5)
        m.disj2 = Disjunct()
        m.disj2.cons = Constraint(expr=m.x >= 8)
        m.disjunction1 = Disjunction(expr=[m.disj1, m.disj2])

        m.disj3 = Disjunct()
        m.disj3.cons = Constraint(expr=m.disj1.x >= 7)
        m.disj4 = Disjunct()
        m.disj4.cons = Constraint(expr=m.x == 3)
        m.disjunction2 = Disjunction(expr=[m.disj3, m.disj4])

        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        for disj, nm in ((m.disj1, "x_4"), (m.disj2, "x_9"),
                         (m.disj3, "x_5"), (m.disj4, "x_8")):
            self.check_name_collision_disaggregated_vars(m, disj, nm)

    def test_do_not_transform_user_deactivated_disjuncts(self):
        ct.check_user_deactivated_disjuncts(self, 'hull')

    def test_improperly_deactivated_disjuncts(self):
        ct.check_improperly_deactivated_disjuncts(self, 'hull')

    def test_do_not_transform_userDeactivated_IndexedDisjunction(self):
        ct.check_do_not_transform_userDeactivated_indexedDisjunction(self,
                                                                     'hull')

    def test_disjunction_deactivated(self):
        ct.check_disjunction_deactivated(self, 'hull')

    def test_disjunctDatas_deactivated(self):
        ct.check_disjunctDatas_deactivated(self, 'hull')

    def test_deactivated_constraints(self):
        ct.check_deactivated_constraints(self, 'hull')

    def check_no_double_transformation(self):
        ct.check_do_not_transform_twice_if_disjunction_reactivated(self,
                                                                   'hull')

    def test_indicator_vars(self):
        ct.check_indicator_vars(self, 'hull')

    def test_xor_constraints(self):
        ct.check_xor_constraint(self, 'hull')

    def test_unbounded_var_error(self):
        m = models.makeTwoTermDisj_Nonlinear()
        # no bounds
        m.w.setlb(None)
        m.w.setub(None)
        self.assertRaisesRegexp(
            GDP_Error,
            "Variables that appear in disjuncts must be "
            "bounded in order to use the hull "
            "transformation! Missing bound for w.*",
            TransformationFactory('gdp.hull').apply_to,
            m)

    def test_indexed_constraints_in_disjunct(self):
        m = models.makeThreeTermDisj_IndexedConstraints()

        TransformationFactory('gdp.hull').apply_to(m)
        transBlock = m._pyomo_gdp_hull_reformulation

        # 2 blocks: the original Disjunct and the transformation block
        self.assertEqual(
            len(list(m.component_objects(Block, descend_into=False))), 1)
        self.assertEqual(
            len(list(m.component_objects(Disjunct))), 1)

        # Each relaxed disjunct should have 3 disaggregated vars, but i "d[i].c"
        # Constraints
        for i in [1,2,3]:
            relaxed = transBlock.relaxedDisjuncts[i-1]
            self.assertEqual(
                len(list(relaxed.disaggregatedVars.component_objects(Var))), 3)
            self.assertEqual(
                len(list(relaxed.disaggregatedVars.component_data_objects(Var))),
                3)
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

        TransformationFactory('gdp.hull').apply_to(m)
        transBlock = m._pyomo_gdp_hull_reformulation

        # 2 blocks: the original Disjunct and the transformation block
        self.assertEqual(
            len(list(m.component_objects(Block, descend_into=False))), 1)
        self.assertEqual(
            len(list(m.component_objects(Disjunct))), 1)

        # Each relaxed disjunct should have 3 disaggregated vars, but i "d[i].c"
        # Constraints
        for i in [1,2,3]:
            relaxed = transBlock.relaxedDisjuncts[i-1]
            self.assertEqual(
                len(list(relaxed.disaggregatedVars.component_objects( Var))), 3)
            self.assertEqual(
                len(list(relaxed.disaggregatedVars.component_data_objects(
                    Var))), 3)
            self.assertEqual(
                len(list(relaxed.component_objects(Constraint))), 4)
            self.assertEqual(
                len(list(relaxed.component_data_objects(Constraint))), 3*2+i)
            self.assertEqual(len(relaxed.component('d[%s].c'%i)), i)

    def test_do_not_transform_deactivated_constraintDatas(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.a[1].setlb(0)
        m.a[1].setub(100)
        m.a[2].setlb(0)
        m.a[2].setub(100)
        m.b.simpledisj1.c[1].deactivate()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        # can't ask for simpledisj1.c[1]: it wasn't transformed
        log = StringIO()
        with LoggingIntercept(log, 'pyomo.gdp', logging.ERROR):
            self.assertRaisesRegexp(
                KeyError,
                ".*b.simpledisj1.c\[1\]",
                hull.get_transformed_constraints,
                m.b.simpledisj1.c[1])
        self.assertRegexpMatches(log.getvalue(),
                                 ".*Constraint 'b.simpledisj1.c\[1\]' has not "
                                 "been transformed.")

        # this fixes a[2] to 0, so we should get the disggregated var
        transformed = hull.get_transformed_constraints(m.b.simpledisj1.c[2])
        self.assertEqual(len(transformed), 1)
        disaggregated_a2 = hull.get_disaggregated_var(m.a[2], m.b.simpledisj1)
        self.assertIs(transformed[0], disaggregated_a2)
        self.assertIsInstance(disaggregated_a2, Var)
        self.assertTrue(disaggregated_a2.is_fixed())
        self.assertEqual(value(disaggregated_a2), 0)

        transformed = hull.get_transformed_constraints(m.b.simpledisj2.c[1])
        # simpledisj2.c[1] is a <= constraint
        self.assertEqual(len(transformed), 1)
        self.assertIs(transformed[0],
                      m.b.simpledisj2.transformation_block().\
                      component("b.simpledisj2.c")[(1,'ub')])

        transformed = hull.get_transformed_constraints(m.b.simpledisj2.c[2])
        # simpledisj2.c[2] is a <= constraint
        self.assertEqual(len(transformed), 1)
        self.assertIs(transformed[0],
                      m.b.simpledisj2.transformation_block().\
                      component("b.simpledisj2.c")[(2,'ub')])


class MultiTermDisj(unittest.TestCase, CommonTests):
    def test_xor_constraint(self):
        ct.check_three_term_xor_constraint(self, 'hull')

    def test_create_using(self):
        m = models.makeThreeTermIndexedDisj()
        self.diff_apply_to_and_create_using(m)

class IndexedDisjunction(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_disaggregation_constraints(self):
        m = models.makeTwoTermIndexedDisjunction()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        relaxedDisjuncts = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts

        disaggregatedVars = {
            1: [relaxedDisjuncts[0].disaggregatedVars.component('x[1]'),
                relaxedDisjuncts[1].disaggregatedVars.component('x[1]')],
            2: [relaxedDisjuncts[2].disaggregatedVars.component('x[2]'),
                relaxedDisjuncts[3].disaggregatedVars.component('x[2]')],
            3: [relaxedDisjuncts[4].disaggregatedVars.component('x[3]'),
                relaxedDisjuncts[5].disaggregatedVars.component('x[3]')],
        }

        for i, disVars in iteritems(disaggregatedVars):
            cons = hull.get_disaggregation_constraint(m.x[i],
                                                       m.disjunction[i])
            self.assertEqual(cons.lower, 0)
            self.assertEqual(cons.upper, 0)
            repn = generate_standard_repn(cons.body)
            self.assertTrue(repn.is_linear())
            self.assertEqual(repn.constant, 0)
            self.assertEqual(len(repn.linear_vars), 3)
            ct.check_linear_coef(self, repn, m.x[i], 1)
            ct.check_linear_coef(self, repn, disVars[0], -1)
            ct.check_linear_coef(self, repn, disVars[1], -1)

    def test_disaggregation_constraints_tuple_indices(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        relaxedDisjuncts = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts

        disaggregatedVars = {
            (1,'A'): [relaxedDisjuncts[0].disaggregatedVars.component('a[1,A]'),
                      relaxedDisjuncts[1].disaggregatedVars.component('a[1,A]')],
            (1,'B'): [relaxedDisjuncts[2].disaggregatedVars.component('a[1,B]'),
                      relaxedDisjuncts[3].disaggregatedVars.component('a[1,B]')],
            (2,'A'): [relaxedDisjuncts[4].disaggregatedVars.component('a[2,A]'),
                      relaxedDisjuncts[5].disaggregatedVars.component('a[2,A]')],
            (2,'B'): [relaxedDisjuncts[6].disaggregatedVars.component('a[2,B]'),
                      relaxedDisjuncts[7].disaggregatedVars.component('a[2,B]')],
        }

        for i, disVars in iteritems(disaggregatedVars):
            cons = hull.get_disaggregation_constraint(m.a[i],
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
            ct.check_linear_coef(self, repn, m.a[i], 1)
            ct.check_linear_coef(self, repn, disVars[0], -1)
            self.assertTrue(disVars[1].is_fixed())
            self.assertEqual(value(disVars[1]), 0)

    def test_xor_constraints(self):
        ct.check_indexed_xor_constraints(self, 'hull')

    def test_xor_constraints_with_targets(self):
        ct.check_indexed_xor_constraints_with_targets(self, 'hull')

    def test_create_using(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
        ct.diff_apply_to_and_create_using(self, m, 'gdp.hull')

    def test_deactivated_constraints(self):
        ct.check_constraints_deactivated_indexedDisjunction(self, 'hull')

    def test_deactivated_disjuncts(self):
        ct.check_deactivated_disjuncts(self, 'hull')

    def test_deactivated_disjunctions(self):
        ct.check_deactivated_disjunctions(self, 'hull')

    def test_partial_deactivate_indexed_disjunction(self):
        ct.check_partial_deactivate_indexed_disjunction(self, 'hull')

    def test_disjunction_data_target(self):
        ct.check_disjunction_data_target(self, 'hull')

    def test_disjunction_data_target_any_index(self):
        ct.check_disjunction_data_target_any_index(self, 'hull')

    def test_targets_with_container_as_arg(self):
        ct.check_targets_with_container_as_arg(self, 'hull')

    def check_trans_block_disjunctions_of_disjunct_datas(self, m):
        transBlock1 = m.component("_pyomo_gdp_hull_reformulation")
        self.assertIsInstance(transBlock1, Block)
        self.assertIsInstance(transBlock1.component("relaxedDisjuncts"), Block)
        # We end up with a transformation block for every SimpleDisjunction or
        # IndexedDisjunction.
        self.assertEqual(len(transBlock1.relaxedDisjuncts), 2)
        self.assertIsInstance(transBlock1.relaxedDisjuncts[0].disaggregatedVars.\
                              component("x"), Var)
        self.assertTrue(transBlock1.relaxedDisjuncts[0].disaggregatedVars.x.\
                        is_fixed())
        self.assertEqual(value(transBlock1.relaxedDisjuncts[0].\
                               disaggregatedVars.x), 0)
        self.assertIsInstance(transBlock1.relaxedDisjuncts[0].component(
            "firstTerm[1].cons"), Constraint)
        # No constraint becuase disaggregated variable fixed to 0
        self.assertEqual(len(transBlock1.relaxedDisjuncts[0].component(
            "firstTerm[1].cons")), 0)
        self.assertIsInstance(transBlock1.relaxedDisjuncts[0].component(
            "x_bounds"), Constraint)
        self.assertEqual(len(transBlock1.relaxedDisjuncts[0].component(
            "x_bounds")), 2)

        self.assertIsInstance(transBlock1.relaxedDisjuncts[1].disaggregatedVars.\
                              component("x"), Var)
        self.assertIsInstance(transBlock1.relaxedDisjuncts[1].component(
            "secondTerm[1].cons"), Constraint)
        self.assertEqual(len(transBlock1.relaxedDisjuncts[1].component(
            "secondTerm[1].cons")), 1)
        self.assertIsInstance(transBlock1.relaxedDisjuncts[1].component(
            "x_bounds"), Constraint)
        self.assertEqual(len(transBlock1.relaxedDisjuncts[1].component(
            "x_bounds")), 2)

        transBlock2 = m.component("_pyomo_gdp_hull_reformulation_4")
        self.assertIsInstance(transBlock2, Block)
        self.assertIsInstance(transBlock2.component("relaxedDisjuncts"), Block)
        self.assertEqual(len(transBlock2.relaxedDisjuncts), 2)
        self.assertIsInstance(transBlock2.relaxedDisjuncts[0].disaggregatedVars.\
                              component("x"), Var)
        self.assertIsInstance(transBlock2.relaxedDisjuncts[0].component(
            "firstTerm[2].cons"), Constraint)
        # we have an equality constraint
        self.assertEqual(len(transBlock2.relaxedDisjuncts[0].component(
            "firstTerm[2].cons")), 1)
        self.assertIsInstance(transBlock2.relaxedDisjuncts[0].component(
            "x_bounds"), Constraint)
        self.assertEqual(len(transBlock2.relaxedDisjuncts[0].component(
            "x_bounds")), 2)

        self.assertIsInstance(transBlock2.relaxedDisjuncts[1].disaggregatedVars.\
                              component("x"), Var)
        self.assertIsInstance(transBlock2.relaxedDisjuncts[1].component(
            "secondTerm[2].cons"), Constraint)
        self.assertEqual(len(transBlock2.relaxedDisjuncts[1].component(
            "secondTerm[2].cons")), 1)
        self.assertIsInstance(transBlock2.relaxedDisjuncts[1].component(
            "x_bounds"), Constraint)
        self.assertEqual(len(transBlock2.relaxedDisjuncts[1].component(
            "x_bounds")), 2)

    def test_simple_disjunction_of_disjunct_datas(self):
        ct.check_simple_disjunction_of_disjunct_datas(self, 'hull')

    def test_any_indexed_disjunction_of_disjunct_datas(self):
        m = models.makeAnyIndexedDisjunctionOfDisjunctDatas()
        TransformationFactory('gdp.hull').apply_to(m)

        transBlock = m.component("_pyomo_gdp_hull_reformulation")
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(transBlock.component("relaxedDisjuncts"), Block)
        self.assertEqual(len(transBlock.relaxedDisjuncts), 4)
        self.assertIsInstance(transBlock.relaxedDisjuncts[0].disaggregatedVars.\
                              component("x"), Var)
        self.assertTrue(transBlock.relaxedDisjuncts[0].disaggregatedVars.\
                        x.is_fixed())
        self.assertEqual(value(transBlock.relaxedDisjuncts[0].disaggregatedVars.\
                               x), 0)
        self.assertIsInstance(transBlock.relaxedDisjuncts[0].component(
            "firstTerm[1].cons"), Constraint)
        # No constraint becuase disaggregated variable fixed to 0
        self.assertEqual(len(transBlock.relaxedDisjuncts[0].component(
            "firstTerm[1].cons")), 0)
        self.assertIsInstance(transBlock.relaxedDisjuncts[0].component(
            "x_bounds"), Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[0].component(
            "x_bounds")), 2)

        self.assertIsInstance(transBlock.relaxedDisjuncts[1].disaggregatedVars.\
                              component("x"), Var)
        self.assertIsInstance(transBlock.relaxedDisjuncts[1].component(
            "secondTerm[1].cons"), Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[1].component(
            "secondTerm[1].cons")), 1)
        self.assertIsInstance(transBlock.relaxedDisjuncts[1].component(
            "x_bounds"), Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[1].component(
            "x_bounds")), 2)

        self.assertIsInstance(transBlock.relaxedDisjuncts[2].disaggregatedVars.\
                              component("x"), Var)
        self.assertIsInstance(transBlock.relaxedDisjuncts[2].component(
            "firstTerm[2].cons"), Constraint)
        # we have an equality constraint
        self.assertEqual(len(transBlock.relaxedDisjuncts[2].component(
            "firstTerm[2].cons")), 1)
        self.assertIsInstance(transBlock.relaxedDisjuncts[2].component(
            "x_bounds"), Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[2].component(
            "x_bounds")), 2)

        self.assertIsInstance(transBlock.relaxedDisjuncts[3].disaggregatedVars.\
                              component("x"), Var)
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
        transBlock = model.component("_pyomo_gdp_hull_reformulation")
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(
            transBlock.component("disjunctionList_xor"), Constraint)
        self.assertEqual(len(transBlock.disjunctionList_xor), 1)
        self.assertFalse(model.disjunctionList[0].active)

        self.assertIsInstance(transBlock.relaxedDisjuncts, Block)
        self.assertEqual(len(transBlock.relaxedDisjuncts), 2)

        self.assertIsInstance(transBlock.relaxedDisjuncts[0].disaggregatedVars.x,
                              Var)
        self.assertTrue(transBlock.relaxedDisjuncts[0].disaggregatedVars.x.\
                        is_fixed())
        self.assertEqual(value(transBlock.relaxedDisjuncts[0].disaggregatedVars.\
                               x), 0)
        self.assertIsInstance(transBlock.relaxedDisjuncts[0].component(
            "firstTerm[0].cons"), Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[0].component(
            "firstTerm[0].cons")), 0)
        self.assertIsInstance(transBlock.relaxedDisjuncts[0].x_bounds,
                              Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[0].x_bounds), 2)

        self.assertIsInstance(transBlock.relaxedDisjuncts[1].disaggregatedVars.x,
                              Var)
        self.assertFalse(transBlock.relaxedDisjuncts[1].disaggregatedVars.\
                         x.is_fixed())
        self.assertIsInstance(transBlock.relaxedDisjuncts[1].component(
            "secondTerm[0].cons"), Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[1].component(
            "secondTerm[0].cons")), 1)
        self.assertIsInstance(transBlock.relaxedDisjuncts[1].x_bounds,
                              Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[1].x_bounds), 2)

    def check_second_iteration(self, model):
        transBlock = model.component("_pyomo_gdp_hull_reformulation")
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
        ct.check_disjunction_and_disjuncts_indexed_by_any(self, 'hull')

    def test_iteratively_adding_disjunctions_transform_container(self):
        ct.check_iteratively_adding_disjunctions_transform_container(self,
                                                                     'hull')

    def test_iteratively_adding_disjunctions_transform_model(self):
        ct.check_iteratively_adding_disjunctions_transform_model(self, 'hull')

    def test_iteratively_adding_to_indexed_disjunction_on_block(self):
        ct.check_iteratively_adding_to_indexed_disjunction_on_block(self,
                                                                    'hull')

class TestTargets_SingleDisjunction(unittest.TestCase, CommonTests):
    def test_only_targets_inactive(self):
        ct.check_only_targets_inactive(self, 'hull')

    def test_only_targets_transformed(self):
        ct.check_only_targets_get_transformed(self, 'hull')

    def test_target_not_a_component_err(self):
        ct.check_target_not_a_component_error(self, 'hull')

    def test_targets_cannot_be_cuids(self):
        ct.check_targets_cannot_be_cuids(self, 'hull')

class TestTargets_IndexedDisjunction(unittest.TestCase, CommonTests):
    # There are a couple tests for targets above, but since I had the patience
    # to make all these for bigm also, I may as well reap the benefits here too.
    def test_indexedDisj_targets_inactive(self):
        ct.check_indexedDisj_targets_inactive(self, 'hull')

    def test_indexedDisj_only_targets_transformed(self):
        ct.check_indexedDisj_only_targets_transformed(self, 'hull')

    def test_warn_for_untransformed(self):
        ct.check_warn_for_untransformed(self, 'hull')

    def test_disjData_targets_inactive(self):
        ct.check_disjData_targets_inactive(self, 'hull')
        m = models.makeDisjunctionsOnIndexedBlock()

    def test_disjData_only_targets_transformed(self):
        ct.check_disjData_only_targets_transformed(self, 'hull')

    def test_indexedBlock_targets_inactive(self):
        ct.check_indexedBlock_targets_inactive(self, 'hull')

    def test_indexedBlock_only_targets_transformed(self):
        ct.check_indexedBlock_only_targets_transformed(self, 'hull')

    def test_blockData_targets_inactive(self):
        ct.check_blockData_targets_inactive(self, 'hull')

    def test_blockData_only_targets_transformed(self):
        ct.check_blockData_only_targets_transformed(self, 'hull')

    def test_do_not_transform_deactivated_targets(self):
        ct.check_do_not_transform_deactivated_targets(self, 'hull')

    def test_create_using(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        ct.diff_apply_to_and_create_using(self, m, 'gdp.hull')

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
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        disaggregationConstraints = m._pyomo_gdp_hull_reformulation.\
                                    disaggregationConstraints
        disaggregationConstraints.pprint()
        consmap = [
            (m.component("b.x"), disaggregationConstraints[(0, None)]),
            (m.b.x, disaggregationConstraints[(1, None)])
        ]

        for v, cons in consmap:
            disCons = hull.get_disaggregation_constraint(v, m.disjunction)
            self.assertIs(disCons, cons)

class DisjunctInMultipleDisjunctions(unittest.TestCase, CommonTests):
    def test_error_for_same_disjunct_in_multiple_disjunctions(self):
        ct.check_error_for_same_disjunct_in_multiple_disjunctions(self, 'hull')

class NestedDisjunction(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_disjuncts_inactive(self):
        ct.check_disjuncts_inactive_nested(self, 'hull')

    def test_deactivated_disjunct_leaves_nested_disjuncts_active(self):
        ct.check_deactivated_disjunct_leaves_nested_disjunct_active(self,
                                                                    'hull')

    def test_mappings_between_disjunctions_and_xors(self):
        # For the sake of not second-guessing anyone, we will let the inner
        # disjunction point to its original XOR constraint. This constraint
        # itself will be transformed by the outer disjunction, so if you want to
        # find what it became you will have to follow its map to the transformed
        # version. (But this behaves the same as bigm)
        ct.check_mappings_between_disjunctions_and_xors(self, 'hull')

    def test_disjunct_targets_inactive(self):
        ct.check_disjunct_targets_inactive(self, 'hull')

    def test_disjunct_only_targets_transformed(self):
        ct.check_disjunct_only_targets_transformed(self, 'hull')

    def test_disjunctData_targets_inactive(self):
        ct.check_disjunctData_targets_inactive(self, 'hull')

    def test_disjunctData_only_targets_transformed(self):
        ct.check_disjunctData_only_targets_transformed(self, 'hull')

    def test_disjunction_target_err(self):
        ct.check_disjunction_target_err(self, 'hull')

    @unittest.skipIf(not linear_solvers, "No linear solver available")
    def test_relaxation_feasibility(self):
        m = models.makeNestedDisjunctions_FlatDisjuncts()
        TransformationFactory('gdp.hull').apply_to(m)

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
            if case[4] is None:
                self.assertEqual(results.solver.termination_condition,
                                 pyomo.opt.TerminationCondition.infeasible)
            else:
                self.assertEqual(results.solver.termination_condition,
                                 pyomo.opt.TerminationCondition.optimal)
                self.assertEqual(value(m.obj), case[4])

    @unittest.skipIf(not linear_solvers, "No linear solver available")
    def test_relaxation_feasibility_transform_inner_first(self):
        # This test is identical to the above except that the
        # reference_indicator_var transformation will be called on m.d1
        # first. So this makes sure that we are still doing the right thing even
        # if the indicator_var references already exist.
        m = models.makeNestedDisjunctions_FlatDisjuncts()
        TransformationFactory('gdp.hull').apply_to(m.d1)
        TransformationFactory('gdp.hull').apply_to(m)

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
            if case[4] is None:
                self.assertEqual(results.solver.termination_condition,
                                 pyomo.opt.TerminationCondition.infeasible)
            else:
                self.assertEqual(results.solver.termination_condition,
                                 pyomo.opt.TerminationCondition.optimal)
                self.assertEqual(value(m.obj), case[4])

    def test_create_using(self):
        m = models.makeNestedDisjunctions_FlatDisjuncts()
        self.diff_apply_to_and_create_using(m)

    def check_outer_disaggregation_constraint(self, cons, var, disj1, disj2):
        hull = TransformationFactory('gdp.hull')
        self.assertTrue(cons.active)
        self.assertEqual(cons.lower, 0)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        ct.check_linear_coef(self, repn, var, 1)
        ct.check_linear_coef(self, repn, hull.get_disaggregated_var(var, disj1),
                             -1)
        ct.check_linear_coef(self, repn, hull.get_disaggregated_var(var, disj2),
                             -1)

    def check_bounds_constraint_ub(self, constraint, ub, dis_var, ind_var):
        hull = TransformationFactory('gdp.hull')
        self.assertIsInstance(constraint, Constraint)
        self.assertTrue(constraint.active)
        self.assertEqual(len(constraint), 1)
        self.assertTrue(constraint['ub'].active)
        self.assertEqual(constraint['ub'].upper, 0)
        self.assertIsNone(constraint['ub'].lower)
        repn = generate_standard_repn(constraint['ub'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, dis_var, 1)
        ct.check_linear_coef(self, repn, ind_var, -ub)
        self.assertIs(constraint, hull.get_var_bounds_constraint(dis_var))

    def check_inner_disaggregated_var_bounds(self, cons, dis, ind_var,
                                             original_cons):
        hull = TransformationFactory('gdp.hull')
        self.assertIsInstance(cons, Constraint)
        self.assertTrue(cons.active)
        self.assertEqual(len(cons), 1)
        self.assertTrue(cons[('ub', 'ub')].active)
        self.assertIsNone(cons[('ub', 'ub')].lower)
        self.assertEqual(cons[('ub', 'ub')].upper, 0)
        repn = generate_standard_repn(cons[('ub', 'ub')].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, dis, 1)
        ct.check_linear_coef(self, repn, ind_var, -2)

        self.assertIs(hull.get_var_bounds_constraint(dis), original_cons)
        transformed_list = hull.get_transformed_constraints(original_cons['ub'])
        self.assertEqual(len(transformed_list), 1)
        self.assertIs(transformed_list[0], cons[('ub', 'ub')])

    def check_inner_transformed_constraint(self, cons, dis, lb, ind_var,
                                           first_transformed, original):
        hull = TransformationFactory('gdp.hull')
        self.assertIsInstance(cons, Constraint)
        self.assertTrue(cons.active)
        self.assertEqual(len(cons), 1)
        # Ha, this really isn't lovely, but its just chance that it's ub the
        # second time.
        self.assertTrue(cons[('lb', 'ub')].active)
        self.assertIsNone(cons[('lb', 'ub')].lower)
        self.assertEqual(cons[('lb', 'ub')].upper, 0)
        repn = generate_standard_repn(cons[('lb', 'ub')].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, dis, -1)
        ct.check_linear_coef(self, repn, ind_var, lb)

        self.assertIs(hull.get_src_constraint(first_transformed),
                      original)
        trans_list = hull.get_transformed_constraints(original)
        self.assertEqual(len(trans_list), 1)
        self.assertIs(trans_list[0], first_transformed['lb'])
        self.assertIs(hull.get_src_constraint(first_transformed['lb']),
                      original)
        self.assertIs(hull.get_src_constraint(cons), first_transformed)
        trans_list = hull.get_transformed_constraints(first_transformed['lb'])
        self.assertEqual(len(trans_list), 1)
        self.assertIs(trans_list[0], cons[('lb', 'ub')])
        self.assertIs(hull.get_src_constraint(cons[('lb', 'ub')]),
                      first_transformed['lb'])

    def check_outer_transformed_constraint(self, cons, dis, lb, ind_var):
        hull = TransformationFactory('gdp.hull')
        self.assertIsInstance(cons, Constraint)
        self.assertTrue(cons.active)
        self.assertEqual(len(cons), 1)
        self.assertTrue(cons['lb'].active)
        self.assertIsNone(cons['lb'].lower)
        self.assertEqual(cons['lb'].upper, 0)
        repn = generate_standard_repn(cons['lb'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, dis, -1)
        ct.check_linear_coef(self, repn, ind_var, lb)

        orig = ind_var.parent_block().c
        self.assertIs(hull.get_src_constraint(cons), orig)
        trans_list = hull.get_transformed_constraints(orig)
        self.assertEqual(len(trans_list), 1)
        self.assertIs(trans_list[0], cons['lb'])

    def test_transformed_model_nestedDisjuncts(self):
        # This test tests *everything* for a simple nested disjunction case.
        m = models.makeNestedDisjunctions_NestedDisjuncts()

        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        transBlock = m._pyomo_gdp_hull_reformulation
        self.assertTrue(transBlock.active)

        # outer xor should be on this block
        xor = transBlock.disj_xor
        self.assertIsInstance(xor, Constraint)
        self.assertTrue(xor.active)
        self.assertEqual(xor.lower, 1)
        self.assertEqual(xor.upper, 1)
        repn = generate_standard_repn(xor.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        ct.check_linear_coef(self, repn, m.d1.indicator_var, 1)
        ct.check_linear_coef(self, repn, m.d2.indicator_var, 1)
        self.assertIs(xor, m.disj.algebraic_constraint())
        self.assertIs(m.disj, hull.get_src_disjunction(xor))

        # so should the outer disaggregation constraint
        dis = transBlock.disaggregationConstraints
        self.assertIsInstance(dis, Constraint)
        self.assertTrue(dis.active)
        self.assertEqual(len(dis), 3)
        self.check_outer_disaggregation_constraint(dis[0,None], m.x, m.d1,
                                                   m.d2)
        self.assertIs(hull.get_disaggregation_constraint(m.x, m.disj),
                      dis[0, None])
        self.check_outer_disaggregation_constraint(dis[1,None],
                                                   m.d1.d3.indicator_var, m.d1,
                                                   m.d2)
        self.assertIs(hull.get_disaggregation_constraint(m.d1.d3.indicator_var,
                                                          m.disj), dis[1,None])
        self.check_outer_disaggregation_constraint(dis[2,None],
                                                   m.d1.d4.indicator_var, m.d1,
                                                   m.d2)
        self.assertIs(hull.get_disaggregation_constraint(m.d1.d4.indicator_var,
                                                          m.disj), dis[2,None])

        # we should have four disjunct transformation blocks: 2 real ones and
        # then two that are just home to indicator_var and disaggregated var
        # References.
        disjBlocks = transBlock.relaxedDisjuncts
        self.assertTrue(disjBlocks.active)
        self.assertEqual(len(disjBlocks), 4)

        disj1 = disjBlocks[0]
        self.assertTrue(disj1.active)
        self.assertIs(disj1, m.d1.transformation_block())
        self.assertIs(m.d1, hull.get_src_disjunct(disj1))

        # check the disaggregated vars are here
        self.assertIsInstance(disj1.disaggregatedVars.x, Var)
        self.assertEqual(disj1.disaggregatedVars.x.lb, 0)
        self.assertEqual(disj1.disaggregatedVars.x.ub, 2)
        self.assertIs(disj1.disaggregatedVars.x, 
                      hull.get_disaggregated_var(m.x, m.d1))
        self.assertIs(m.x, hull.get_src_var(disj1.disaggregatedVars.x))
        d3 = disj1.disaggregatedVars.component("d1.d3.indicator_var")
        self.assertEqual(d3.lb, 0)
        self.assertEqual(d3.ub, 1)
        self.assertIsInstance(d3, Var)
        self.assertIs(d3, hull.get_disaggregated_var(m.d1.d3.indicator_var,
                                                      m.d1))
        self.assertIs(m.d1.d3.indicator_var, hull.get_src_var(d3))
        d4 = disj1.disaggregatedVars.component("d1.d4.indicator_var")
        self.assertIsInstance(d4, Var)
        self.assertEqual(d4.lb, 0)
        self.assertEqual(d4.ub, 1)
        self.assertIs(d4, hull.get_disaggregated_var(m.d1.d4.indicator_var,
                                                      m.d1))
        self.assertIs(m.d1.d4.indicator_var, hull.get_src_var(d4))

        # check inner disjunction disaggregated vars
        x3 = m.d1._pyomo_gdp_hull_reformulation.relaxedDisjuncts[0].\
             disaggregatedVars.x
        self.assertIsInstance(x3, Var)
        self.assertEqual(x3.lb, 0)
        self.assertEqual(x3.ub, 2)
        self.assertIs(hull.get_disaggregated_var(m.x, m.d1.d3), x3)
        self.assertIs(hull.get_src_var(x3), m.x)

        x4 = m.d1._pyomo_gdp_hull_reformulation.relaxedDisjuncts[1].\
             disaggregatedVars.x
        self.assertIsInstance(x4, Var)
        self.assertEqual(x4.lb, 0)
        self.assertEqual(x4.ub, 2)
        self.assertIs(hull.get_disaggregated_var(m.x, m.d1.d4), x4)
        self.assertIs(hull.get_src_var(x4), m.x)

        # check the bounds constraints
        self.check_bounds_constraint_ub(disj1.x_bounds, 2,
                                        disj1.disaggregatedVars.x,
                                        m.d1.indicator_var)
        self.check_bounds_constraint_ub(
            disj1.component("d1.d3.indicator_var_bounds"), 1, 
            disj1.disaggregatedVars.component("d1.d3.indicator_var"), 
            m.d1.indicator_var)
        self.check_bounds_constraint_ub(
            disj1.component("d1.d4.indicator_var_bounds"), 1,
            disj1.disaggregatedVars.component("d1.d4.indicator_var"), 
            m.d1.indicator_var)

        # check the transformed constraints

        # transformed xor
        xor = disj1.component("d1._pyomo_gdp_hull_reformulation.d1.disj2_xor")
        self.assertIsInstance(xor, Constraint)
        self.assertTrue(xor.active)
        self.assertEqual(len(xor), 1)
        self.assertTrue(xor['eq'].active)
        self.assertEqual(xor['eq'].lower, 0)
        self.assertEqual(xor['eq'].upper, 0)
        repn = generate_standard_repn(xor['eq'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 3)
        ct.check_linear_coef(
            self, repn,
            disj1.disaggregatedVars.component("d1.d3.indicator_var"), 1)
        ct.check_linear_coef(
            self, repn,
            disj1.disaggregatedVars.component("d1.d4.indicator_var"), 1)
        ct.check_linear_coef(self, repn, m.d1.indicator_var, -1)

        # inner disjunction disaggregation constraint
        dis_cons_inner_disjunction = disj1.component(
            "d1._pyomo_gdp_hull_reformulation.disaggregationConstraints")
        self.assertIsInstance(dis_cons_inner_disjunction, Constraint)
        self.assertTrue(dis_cons_inner_disjunction.active)
        self.assertEqual(len(dis_cons_inner_disjunction), 1)
        dis_cons_inner_disjunction.pprint()
        self.assertTrue(dis_cons_inner_disjunction[(0,None,'eq')].active)
        self.assertEqual(dis_cons_inner_disjunction[(0,None,'eq')].lower, 0)
        self.assertEqual(dis_cons_inner_disjunction[(0,None,'eq')].upper, 0)
        repn = generate_standard_repn(dis_cons_inner_disjunction[(0, None,
                                                                  'eq')].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 3)
        ct.check_linear_coef(self, repn, x3, -1)
        ct.check_linear_coef(self, repn, x4, -1)
        ct.check_linear_coef(self, repn, disj1.disaggregatedVars.x, 1)

        # disaggregated d3.x bounds constraints
        x3_bounds = disj1.component(
            "d1._pyomo_gdp_hull_reformulation.relaxedDisjuncts[0].x_bounds")
        original_cons = m.d1._pyomo_gdp_hull_reformulation.relaxedDisjuncts[0].\
                        x_bounds
        self.check_inner_disaggregated_var_bounds(
            x3_bounds, x3, 
            disj1.disaggregatedVars.component("d1.d3.indicator_var"),
            original_cons)


        # disaggregated d4.x bounds constraints
        x4_bounds = disj1.component(
            "d1._pyomo_gdp_hull_reformulation.relaxedDisjuncts[1].x_bounds")
        original_cons = m.d1._pyomo_gdp_hull_reformulation.relaxedDisjuncts[1].\
                        x_bounds
        self.check_inner_disaggregated_var_bounds(
            x4_bounds, x4,
            disj1.disaggregatedVars.component("d1.d4.indicator_var"),
            original_cons)

        # transformed x >= 1.2
        cons = disj1.component(
            "d1._pyomo_gdp_hull_reformulation.relaxedDisjuncts[0].d1.d3.c")
        first_transformed = m.d1._pyomo_gdp_hull_reformulation.\
                            relaxedDisjuncts[0].component("d1.d3.c")
        original = m.d1.d3.c
        self.check_inner_transformed_constraint(
            cons, x3, 1.2,
            disj1.disaggregatedVars.component("d1.d3.indicator_var"),
            first_transformed, original)

        # transformed x >= 1.3
        cons = disj1.component(
            "d1._pyomo_gdp_hull_reformulation.relaxedDisjuncts[1].d1.d4.c")
        first_transformed = m.d1._pyomo_gdp_hull_reformulation.\
                            relaxedDisjuncts[1].component("d1.d4.c")
        original = m.d1.d4.c
        self.check_inner_transformed_constraint(
            cons, x4, 1.3,
            disj1.disaggregatedVars.component("d1.d4.indicator_var"),
            first_transformed, original)

        # outer disjunction transformed constraint
        cons = disj1.component("d1.c")
        self.check_outer_transformed_constraint(cons, disj1.disaggregatedVars.x,
                                                1, m.d1.indicator_var)

        # and last, check the second transformed outer disjunct
        disj2 = disjBlocks[3]
        self.assertTrue(disj2.active)
        self.assertIs(disj2, m.d2.transformation_block())
        self.assertIs(m.d2, hull.get_src_disjunct(disj2))

        # disaggregated var
        x2 = disj2.disaggregatedVars.x
        self.assertIsInstance(x2, Var)
        self.assertEqual(x2.lb, 0)
        self.assertEqual(x2.ub, 2)
        self.assertIs(hull.get_disaggregated_var(m.x, m.d2), x2)
        self.assertIs(hull.get_src_var(x2), m.x)

        # bounds constraint
        x_bounds = disj2.x_bounds
        self.check_bounds_constraint_ub(x_bounds, 2, x2, m.d2.indicator_var)

        # transformed constraint x >= 1.1
        cons = disj2.component("d2.c")
        self.check_outer_transformed_constraint(cons, x2, 1.1,
                                                m.d2.indicator_var)

        # check inner xor mapping: Note that this maps to a now deactivated
        # (transformed again) constraint, but that it is possible to go full
        # circle, like so:
        orig_inner_xor = m.d1._pyomo_gdp_hull_reformulation.component(
            "d1.disj2_xor")
        self.assertIs(m.d1.disj2.algebraic_constraint(), orig_inner_xor)
        self.assertFalse(orig_inner_xor.active)
        trans_list = hull.get_transformed_constraints(orig_inner_xor)
        self.assertEqual(len(trans_list), 1)
        self.assertIs(trans_list[0], xor['eq'])
        self.assertIs(hull.get_src_constraint(xor), orig_inner_xor)
        self.assertIs(hull.get_src_disjunction(orig_inner_xor), m.d1.disj2)

        # the same goes for the disaggregation constraint
        orig_dis_container = m.d1._pyomo_gdp_hull_reformulation.\
                             disaggregationConstraints
        orig_dis = orig_dis_container[0,None]
        self.assertIs(hull.get_disaggregation_constraint(m.x, m.d1.disj2),
                      orig_dis)
        self.assertFalse(orig_dis.active)
        transformedList = hull.get_transformed_constraints(orig_dis)
        self.assertEqual(len(transformedList), 1)
        self.assertIs(transformedList[0], dis_cons_inner_disjunction[(0, None,
                                                                      'eq')])

        self.assertIs(hull.get_src_constraint(
            dis_cons_inner_disjunction[(0, None, 'eq')]), orig_dis)
        self.assertIs(hull.get_src_constraint( dis_cons_inner_disjunction),
                      orig_dis_container)
        # though we don't have a map back from the disaggregation constraint to
        # the variable because I'm not sure why you would... The variable is in
        # the constraint.

        # check the inner disjunct mappings
        self.assertIs(m.d1.d3.transformation_block(),
                      m.d1._pyomo_gdp_hull_reformulation.relaxedDisjuncts[0])
        self.assertIs(hull.get_src_disjunct(
            m.d1._pyomo_gdp_hull_reformulation.relaxedDisjuncts[0]), m.d1.d3)
        self.assertIs(m.d1.d4.transformation_block(),
                      m.d1._pyomo_gdp_hull_reformulation.relaxedDisjuncts[1])
        self.assertIs(hull.get_src_disjunct(
            m.d1._pyomo_gdp_hull_reformulation.relaxedDisjuncts[1]), m.d1.d4)

    @unittest.skipIf(not linear_solvers, "No linear solver available")
    def test_solve_nested_model(self):
        # This is really a test that our variable references have all been moved
        # up correctly.
        m = models.makeNestedDisjunctions_NestedDisjuncts()

        hull = TransformationFactory('gdp.hull')
        m_hull = hull.create_using(m)

        SolverFactory(linear_solvers[0]).solve(m_hull)

        # check solution
        self.assertEqual(value(m_hull.d1.indicator_var), 0)
        self.assertEqual(value(m_hull.d2.indicator_var), 1)
        self.assertEqual(value(m_hull.x), 1.1)

        # transform inner problem with bigm, outer with hull and make sure it
        # still works
        TransformationFactory('gdp.bigm').apply_to(m.d1.disj2)
        hull.apply_to(m)

        SolverFactory(linear_solvers[0]).solve(m)

        # check solution
        self.assertEqual(value(m.d1.indicator_var), 0)
        self.assertEqual(value(m.d2.indicator_var), 1)
        self.assertEqual(value(m.x), 1.1)

    @unittest.skipIf(not linear_solvers, "No linear solver available")
    def test_disaggregated_vars_are_set_to_0_correctly(self):
        m = models.makeNestedDisjunctions_FlatDisjuncts()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        # this should be a feasible integer solution
        m.d1.indicator_var.fix(0)
        m.d2.indicator_var.fix(1)
        m.d3.indicator_var.fix(0)
        m.d4.indicator_var.fix(0)

        results = SolverFactory(linear_solvers[0]).solve(m)
        self.assertEqual(results.solver.termination_condition,
                         TerminationCondition.optimal)
        self.assertEqual(value(m.x), 1.1)

        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d1)), 0)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d2)), 1.1)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d3)), 0)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d4)), 0)

        # and what if one of the inner disjuncts is true?
        m.d1.indicator_var.fix(1)
        m.d2.indicator_var.fix(0)
        m.d3.indicator_var.fix(1)
        m.d4.indicator_var.fix(0)

        results = SolverFactory(linear_solvers[0]).solve(m)
        self.assertEqual(results.solver.termination_condition,
                         TerminationCondition.optimal)
        self.assertEqual(value(m.x), 1.2)

        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d1)), 1.2)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d2)), 0)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d3)), 1.2)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d4)), 0)

class TestSpecialCases(unittest.TestCase):
    def test_local_vars(self):
        """ checks that if nothing is marked as local, we assume it is all
        global. We disaggregate everything to be safe."""
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
            TransformationFactory('gdp.hull').create_using,
            m)
        m.d2.z.setlb(7)
        self.assertRaisesRegexp(
            GDP_Error,
            ".*Missing bound for d2.z.*",
            TransformationFactory('gdp.hull').create_using,
            m)
        m.d2.z.setub(9)

        i = TransformationFactory('gdp.hull').create_using(m)
        rd = i._pyomo_gdp_hull_reformulation.relaxedDisjuncts[1]
        varBlock = rd.disaggregatedVars
        # z should be disaggregated because we can't be sure it's not somewhere
        # else on the model
        self.assertEqual(sorted(varBlock.component_map(Var)), ['x','y','z'])
        self.assertEqual(len(rd.component_map(Constraint)), 4)
        # bounds haven't changed on original
        self.assertEqual(i.d2.z.bounds, (7,9))
        # check disaggregated variable
        self.assertIsInstance(varBlock.component("z"), Var)
        self.assertEqual(varBlock.z.bounds, (0,9))
        self.assertEqual(len(rd.z_bounds), 2)
        self.assertEqual(rd.z_bounds['lb'].lower, None)
        self.assertEqual(rd.z_bounds['lb'].upper, 0)
        self.assertEqual(rd.z_bounds['ub'].lower, None)
        self.assertEqual(rd.z_bounds['ub'].upper, 0)
        i.d2.indicator_var = 1
        varBlock.z = 2
        self.assertEqual(rd.z_bounds['lb'].body(), 5)
        self.assertEqual(rd.z_bounds['ub'].body(), -7)

        m.d2.z.setlb(-9)
        m.d2.z.setub(-7)
        i = TransformationFactory('gdp.hull').create_using(m)
        rd = i._pyomo_gdp_hull_reformulation.relaxedDisjuncts[1]
        varBlock = rd.disaggregatedVars
        self.assertEqual(sorted(varBlock.component_map(Var)), ['x','y','z'])
        self.assertEqual(len(rd.component_map(Constraint)), 4)
        # original bounds unchanged
        self.assertEqual(i.d2.z.bounds, (-9,-7))
        # check disaggregated variable
        self.assertIsInstance(varBlock.component("z"), Var)
        self.assertEqual(varBlock.z.bounds, (-9,0))
        self.assertEqual(len(rd.z_bounds), 2)
        self.assertEqual(rd.z_bounds['lb'].lower, None)
        self.assertEqual(rd.z_bounds['lb'].upper, 0)
        self.assertEqual(rd.z_bounds['ub'].lower, None)
        self.assertEqual(rd.z_bounds['ub'].upper, 0)
        i.d2.indicator_var = 1
        varBlock.z = 2
        self.assertEqual(rd.z_bounds['lb'].body(), -11)
        self.assertEqual(rd.z_bounds['ub'].body(), 9)

    def test_local_var_suffix(self):
        hull = TransformationFactory('gdp.hull')

        model = ConcreteModel()
        model.x = Var(bounds=(5,100))
        model.y = Var(bounds=(0,100))
        model.d1 = Disjunct()
        model.d1.c = Constraint(expr=model.y >= model.x)
        model.d2 = Disjunct()
        model.d2.z = Var(bounds=(-9, -7))
        model.d2.c = Constraint(expr=model.y >= model.d2.z)
        model.disj = Disjunction(expr=[model.d1, model.d2])

        # we don't declare z local
        m = hull.create_using(model)
        self.assertEqual(m.d2.z.lb, -9)
        self.assertEqual(m.d2.z.ub, -7)
        z_disaggregated = m.d2.transformation_block().disaggregatedVars.\
                          component("z")
        self.assertIsInstance(z_disaggregated, Var)
        self.assertIs(z_disaggregated,
                      hull.get_disaggregated_var(m.d2.z, m.d2))

        # we do declare z local
        model.d2.LocalVars = Suffix(direction=Suffix.LOCAL)
        model.d2.LocalVars[model.d2] = [model.d2.z]

        m = hull.create_using(model)

        # make sure we did not disaggregate z
        self.assertEqual(m.d2.z.lb, -9)
        self.assertEqual(m.d2.z.ub, 0)
        # it is its own disaggregated variable
        self.assertIs(hull.get_disaggregated_var(m.d2.z, m.d2), m.d2.z)
        # it does not exist on the transformation block
        self.assertIsNone(m.d2.transformation_block().disaggregatedVars.\
                          component("z"))

class UntransformableObjectsOnDisjunct(unittest.TestCase):
    def test_RangeSet(self):
        ct.check_RangeSet(self, 'hull')

    def test_Expression(self):
        ct.check_Expression(self, 'hull')

class TransformABlock(unittest.TestCase, CommonTests):
    def test_transformation_simple_block(self):
        ct.check_transformation_simple_block(self, 'hull')

    def test_transform_block_data(self):
        ct.check_transform_block_data(self, 'hull')

    def test_simple_block_target(self):
        ct.check_simple_block_target(self, 'hull')

    def test_block_data_target(self):
        ct.check_block_data_target(self, 'hull')

    def test_indexed_block_target(self):
        ct.check_indexed_block_target(self, 'hull')

    def test_block_targets_inactive(self):
        ct.check_block_targets_inactive(self, 'hull')

    def test_block_only_targets_transformed(self):
        ct.check_block_only_targets_transformed(self, 'hull')

    def test_create_using(self):
        m = models.makeTwoTermDisjOnBlock()
        ct.diff_apply_to_and_create_using(self, m, 'gdp.hull')

class DisjOnBlock(unittest.TestCase, CommonTests):
    # when the disjunction is on a block, we want all of the stuff created by
    # the transformation to go on that block also so that solving the block
    # maintains its meaning

    def test_xor_constraint_added(self):
        ct.check_xor_constraint_added(self, 'hull')

    def test_trans_block_created(self):
        ct.check_trans_block_created(self, 'hull')

class TestErrors(unittest.TestCase):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_ask_for_transformed_constraint_from_untransformed_disjunct(self):
        ct.check_ask_for_transformed_constraint_from_untransformed_disjunct(
            self, 'hull')

    def test_silly_target(self):
        ct.check_silly_target(self, 'hull')

    def test_retrieving_nondisjunctive_components(self):
        ct.check_retrieving_nondisjunctive_components(self, 'hull')

    def test_transform_empty_disjunction(self):
        ct.check_transform_empty_disjunction(self, 'hull')

    def test_deactivated_disjunct_nonzero_indicator_var(self):
        ct.check_deactivated_disjunct_nonzero_indicator_var(self,
                                                            'hull')

    def test_deactivated_disjunct_unfixed_indicator_var(self):
        ct.check_deactivated_disjunct_unfixed_indicator_var(self, 'hull')

    def test_infeasible_xor_because_all_disjuncts_deactivated(self):
        m = ct.setup_infeasible_xor_because_all_disjuncts_deactivated(self,
                                                                      'hull')
        hull = TransformationFactory('gdp.hull')
        transBlock = m.component("_pyomo_gdp_hull_reformulation")
        self.assertIsInstance(transBlock, Block)
        self.assertEqual(len(transBlock.relaxedDisjuncts), 2)
        self.assertIsInstance(transBlock.component("disjunction_xor"),
                              Constraint)
        disjunct1 = transBlock.relaxedDisjuncts[0]
        # we disaggregated the (deactivated) indicator variables
        d3_ind = m.disjunction_disjuncts[0].nestedDisjunction_disjuncts[0].\
                 indicator_var
        d4_ind = m.disjunction_disjuncts[0].nestedDisjunction_disjuncts[1].\
                 indicator_var
        self.assertIs(hull.get_disaggregated_var(d3_ind,
                                                  m.disjunction_disjuncts[0]),
                      disjunct1.disaggregatedVars.indicator_var)
        self.assertIs(hull.get_src_var(
            disjunct1.disaggregatedVars.indicator_var), d3_ind)
        self.assertIs(hull.get_disaggregated_var(d4_ind,
                                                  m.disjunction_disjuncts[0]),
                      disjunct1.disaggregatedVars.indicator_var_4)
        self.assertIs(hull.get_src_var(
            disjunct1.disaggregatedVars.indicator_var_4), d4_ind)

        relaxed_xor = disjunct1.component(
            "disjunction_disjuncts[0]._pyomo_gdp_hull_reformulation."
            "disjunction_disjuncts[0].nestedDisjunction_xor")
        self.assertIsInstance(relaxed_xor, Constraint)
        self.assertEqual(len(relaxed_xor), 1)
        repn = generate_standard_repn(relaxed_xor['eq'].body)
        self.assertEqual(relaxed_xor['eq'].lower, 0)
        self.assertEqual(relaxed_xor['eq'].upper, 0)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 3)
        # constraint says that the disaggregated indicator variables of the
        # nested disjuncts sum to the indicator variable of the outer disjunct.
        ct.check_linear_coef( self, repn,
                              m.disjunction.disjuncts[0].indicator_var, -1)
        ct.check_linear_coef(self, repn,
                             disjunct1.disaggregatedVars.indicator_var, 1)
        ct.check_linear_coef(self, repn,
                             disjunct1.disaggregatedVars.indicator_var_4, 1)
        self.assertEqual(repn.constant, 0)

        # but the disaggregation constraints are going to force them to 0 (which
        # will in turn force the outer disjunct indicator variable to 0, which
        # is what we want)
        d3_ind_dis = transBlock.disaggregationConstraints[1, None]
        self.assertEqual(d3_ind_dis.lower, 0)
        self.assertEqual(d3_ind_dis.upper, 0)
        repn = generate_standard_repn(d3_ind_dis.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, 0)
        ct.check_linear_coef(self, repn,
                             disjunct1.disaggregatedVars.indicator_var, -1)
        ct.check_linear_coef(self, repn,
                             transBlock.relaxedDisjuncts[1].disaggregatedVars.\
                             indicator_var, -1)
        d4_ind_dis = transBlock.disaggregationConstraints[2, None]
        self.assertEqual(d4_ind_dis.lower, 0)
        self.assertEqual(d4_ind_dis.upper, 0)
        repn = generate_standard_repn(d4_ind_dis.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, 0)
        ct.check_linear_coef(self, repn,
                             disjunct1.disaggregatedVars.indicator_var_4, -1)
        ct.check_linear_coef(self, repn,
                             transBlock.relaxedDisjuncts[1].disaggregatedVars.\
                             indicator_var_9, -1)

    def test_mapping_method_errors(self):
        m = models.makeTwoTermDisj_Nonlinear()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        log = StringIO()
        with LoggingIntercept(log, 'pyomo.gdp.hull', logging.ERROR):
            self.assertRaisesRegexp(
                AttributeError,
                "'NoneType' object has no attribute '_bigMConstraintMap'",
                hull.get_var_bounds_constraint,
                m.w)
        self.assertRegexpMatches(
            log.getvalue(),
            ".*Either 'w' is not a disaggregated variable, "
            "or the disjunction that disaggregates it has "
            "not been properly transformed.")

        log = StringIO()
        with LoggingIntercept(log, 'pyomo.gdp.hull', logging.ERROR):
            self.assertRaisesRegexp(
                KeyError,
                ".*_pyomo_gdp_hull_reformulation.relaxedDisjuncts\[1\]."
                "disaggregatedVars.w",
                hull.get_disaggregation_constraint,
                m.d[1].transformation_block().disaggregatedVars.w,
                m.disjunction)
        self.assertRegexpMatches(log.getvalue(), ".*It doesn't appear that "
                                 "'_pyomo_gdp_hull_reformulation."
                                 "relaxedDisjuncts\[1\].disaggregatedVars.w' "
                                 "is a variable that was disaggregated by "
                                 "Disjunction 'disjunction'")

        log = StringIO()
        with LoggingIntercept(log, 'pyomo.gdp.hull', logging.ERROR):
            self.assertRaisesRegexp(
                AttributeError,
                "'NoneType' object has no attribute '_disaggregatedVarMap'",
                hull.get_src_var,
                m.w)
        self.assertRegexpMatches(
            log.getvalue(),
            ".*'w' does not appear to be a disaggregated variable")

        log = StringIO()
        with LoggingIntercept(log, 'pyomo.gdp.hull', logging.ERROR):
            self.assertRaisesRegexp(
                KeyError,
                ".*_pyomo_gdp_hull_reformulation.relaxedDisjuncts\[1\]."
                "disaggregatedVars.w",
                hull.get_disaggregated_var,
                m.d[1].transformation_block().disaggregatedVars.w,
                m.d[1])
        self.assertRegexpMatches(log.getvalue(),
                                 ".*It does not appear "
                                 "'_pyomo_gdp_hull_reformulation."
                                 "relaxedDisjuncts\[1\].disaggregatedVars.w' "
                                 "is a variable which appears in disjunct "
                                 "'d\[1\]'")

        m.random_disjunction = Disjunction(expr=[m.w == 2, m.w >= 7])
        self.assertRaisesRegexp(
            GDP_Error,
            "Disjunction 'random_disjunction' has not been properly "
            "transformed: None of its disjuncts are transformed.",
            hull.get_disaggregation_constraint,
            m.w,
            m.random_disjunction)

        self.assertRaisesRegexp(
            GDP_Error,
            "Disjunct 'random_disjunction_disjuncts\[0\]' has not been "
            "transformed",
            hull.get_disaggregated_var,
            m.w,
            m.random_disjunction.disjuncts[0])

class InnerDisjunctionSharedDisjuncts(unittest.TestCase):
    def test_activeInnerDisjunction_err(self):
        ct.check_activeInnerDisjunction_err(self, 'hull')

class BlocksOnDisjuncts(unittest.TestCase):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def makeModel(self):
        # I'm going to multi-task and also check some types of constraints
        # whose expressions need to be tested
        m = ConcreteModel()
        m.x = Var(bounds=(1, 5))
        m.y = Var(bounds=(0, 9))
        m.disj1 = Disjunct()
        m.disj1.add_component("b.any_index", Constraint(expr=m.x >= 1.5))
        m.disj1.b = Block()
        m.disj1.b.any_index = Constraint(Any)
        m.disj1.b.any_index['local'] = m.x <= 2
        m.disj1.b.LocalVars = Suffix(direction=Suffix.LOCAL)
        m.disj1.b.LocalVars[m.disj1] = [m.x]
        m.disj1.b.any_index['nonlin-ub'] = m.y**2 <= 4
        m.disj2 = Disjunct()
        m.disj2.non_lin_lb = Constraint(expr=log(1 + m.y) >= 1)
        m.disjunction = Disjunction(expr=[m.disj1, m.disj2])
        return m

    def test_transformed_constraint_name_conflict(self):
        m = self.makeModel()

        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        transBlock = m.disj1.transformation_block()
        self.assertIsInstance(transBlock.component("disj1.b.any_index"),
                              Constraint)
        self.assertIsInstance(transBlock.component("disj1.b.any_index_4"),
                              Constraint)
        xformed = hull.get_transformed_constraints(
            m.disj1.component("b.any_index"))
        self.assertEqual(len(xformed), 1)
        self.assertIs(xformed[0],
                      transBlock.component("disj1.b.any_index")['lb'])

        xformed = hull.get_transformed_constraints(m.disj1.b.any_index['local'])
        self.assertEqual(len(xformed), 1)
        self.assertIs(xformed[0],
                      transBlock.component("disj1.b.any_index_4")[
                          ('local','ub')])
        xformed = hull.get_transformed_constraints(
            m.disj1.b.any_index['nonlin-ub'])
        self.assertEqual(len(xformed), 1)
        self.assertIs(xformed[0],
                      transBlock.component("disj1.b.any_index_4")[
                          ('nonlin-ub','ub')])

    def test_local_var_handled_correctly(self):
        m = self.makeModel()

        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        # test the local variable was handled correctly.
        self.assertIs(hull.get_disaggregated_var(m.x, m.disj1), m.x)
        self.assertEqual(m.x.lb, 0)
        self.assertEqual(m.x.ub, 5)
        self.assertIsNone(m.disj1.transformation_block().disaggregatedVars.\
                          component("x"))
        self.assertIsInstance(m.disj1.transformation_block().disaggregatedVars.\
                              component("y"), Var)

    # this doesn't require the block, I'm just coopting this test to make sure
    # of some nonlinear expressions.
    def test_transformed_constraints(self):
        m = self.makeModel()

        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        # test the transformed nonlinear constraints
        nonlin_ub_list = hull.get_transformed_constraints(
            m.disj1.b.any_index['nonlin-ub'])
        self.assertEqual(len(nonlin_ub_list), 1)
        cons = nonlin_ub_list[0]
        self.assertEqual(cons.index(), ('nonlin-ub', 'ub'))
        self.assertIs(cons.ctype, Constraint)
        self.assertIsNone(cons.lower)
        self.assertEqual(value(cons.upper), 0)
        repn = generate_standard_repn(cons.body)
        self.assertEqual(str(repn.nonlinear_expr),
                         "(0.9999*disj1.indicator_var + 0.0001)*"
                         "(_pyomo_gdp_hull_reformulation.relaxedDisjuncts[0]."
                         "disaggregatedVars.y/"
                         "(0.9999*disj1.indicator_var + 0.0001))**2")
        self.assertEqual(len(repn.nonlinear_vars), 2)
        self.assertIs(repn.nonlinear_vars[0], m.disj1.indicator_var)
        self.assertIs(repn.nonlinear_vars[1],
                      hull.get_disaggregated_var(m.y, m.disj1))
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], m.disj1.indicator_var)
        self.assertEqual(repn.linear_coefs[0], -4)

        nonlin_lb_list = hull.get_transformed_constraints(m.disj2.non_lin_lb)
        self.assertEqual(len(nonlin_lb_list), 1)
        cons = nonlin_lb_list[0]
        self.assertEqual(cons.index(), 'lb')
        self.assertIs(cons.ctype, Constraint)
        self.assertIsNone(cons.lower)
        self.assertEqual(value(cons.upper), 0)
        repn = generate_standard_repn(cons.body)
        self.assertEqual(str(repn.nonlinear_expr),
                         "- ((0.9999*disj2.indicator_var + 0.0001)*"
                         "log(1 + "
                         "_pyomo_gdp_hull_reformulation.relaxedDisjuncts[1]."
                         "disaggregatedVars.y/"
                         "(0.9999*disj2.indicator_var + 0.0001)))")
        self.assertEqual(len(repn.nonlinear_vars), 2)
        self.assertIs(repn.nonlinear_vars[0], m.disj2.indicator_var)
        self.assertIs(repn.nonlinear_vars[1],
                      hull.get_disaggregated_var(m.y, m.disj2))
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], m.disj2.indicator_var)
        self.assertEqual(repn.linear_coefs[0], 1)

class DisaggregatingFixedVars(unittest.TestCase):
    def test_disaggregate_fixed_variables(self):
        m = models.makeTwoTermDisj()
        m.x.fix(6)
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        # check that we did indeed disaggregate x
        transBlock = m.d[1]._transformation_block()
        self.assertIsInstance(transBlock.disaggregatedVars.component("x"), Var)
        self.assertIs(hull.get_disaggregated_var(m.x, m.d[1]),
                      transBlock.disaggregatedVars.x)
        self.assertIs(hull.get_src_var(transBlock.disaggregatedVars.x), m.x)

    def test_do_not_disaggregate_fixed_variables(self):
        m = models.makeTwoTermDisj()
        m.x.fix(6)
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m, assume_fixed_vars_permanent=True)
        # check that we didn't disaggregate x
        transBlock = m.d[1]._transformation_block()
        self.assertIsNone(transBlock.disaggregatedVars.component("x"))

class NameDeprecationTest(unittest.TestCase):
    def test_name_deprecated(self):
        m = models.makeTwoTermDisj()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.gdp', logging.WARNING):
            TransformationFactory('gdp.chull').apply_to(m)
        self.assertIn("DEPRECATED: The 'gdp.chull' name is deprecated. "
                      "Please use the more apt 'gdp.hull' instead.",
                      output.getvalue().replace('\n', ' '))

    def test_hull_chull_equivalent(self):
        m = models.makeTwoTermDisj()
        out1 = StringIO()
        out2 = StringIO()
        m1 = TransformationFactory('gdp.hull').create_using(m)
        m2 = TransformationFactory('gdp.chull').create_using(m)
        m1.pprint(ostream=out1)
        m2.pprint(ostream=out2)
        self.assertMultiLineEqual(out1.getvalue(), out2.getvalue())

class KmeansTest(unittest.TestCase):
    @unittest.skipIf('gurobi' not in linear_solvers, 
                     "Gurobi solver not available")
    def test_optimal_soln_feasible(self):
        m = ConcreteModel()
        m.Points = RangeSet(3)
        m.Centroids = RangeSet(2)

        m.X = Param(m.Points, initialize={1:0.3672, 2:0.8043, 3:0.3059})

        m.cluster_center = Var(m.Centroids, bounds=(0,2))
        m.distance = Var(m.Points, bounds=(0,2))
        m.t = Var(m.Points, m.Centroids, bounds=(0,2))

        @m.Disjunct(m.Points, m.Centroids)
        def AssignPoint(d, i, k):
            m = d.model()
            d.LocalVars = Suffix(direction=Suffix.LOCAL)
            d.LocalVars[d] = [m.t[i,k]]
            def distance1(d):
                return m.t[i,k] >= m.X[i] - m.cluster_center[k]
            def distance2(d):
                return m.t[i,k] >= - (m.X[i] - m.cluster_center[k])
            d.dist1 = Constraint(rule=distance1)
            d.dist2 = Constraint(rule=distance2)
            d.define_distance = Constraint(expr=m.distance[i] == m.t[i,k])

        @m.Disjunction(m.Points)
        def OneCentroidPerPt(m, i):
            return [m.AssignPoint[i, k] for k in m.Centroids]

        m.obj = Objective(expr=sum(m.distance[i] for i in m.Points))

        TransformationFactory('gdp.hull').apply_to(m)

        # fix an optimal solution
        m.AssignPoint[1,1].indicator_var.fix(1)
        m.AssignPoint[1,2].indicator_var.fix(0)
        m.AssignPoint[2,1].indicator_var.fix(0)
        m.AssignPoint[2,2].indicator_var.fix(1)
        m.AssignPoint[3,1].indicator_var.fix(1)
        m.AssignPoint[3,2].indicator_var.fix(0)

        m.cluster_center[1].fix(0.3059)
        m.cluster_center[2].fix(0.8043)

        m.distance[1].fix(0.0613)
        m.distance[2].fix(0)
        m.distance[3].fix(0)

        m.t[1,1].fix(0.0613)
        m.t[1,2].fix(0)
        m.t[2,1].fix(0)
        m.t[2,2].fix(0)
        m.t[3,1].fix(0)
        m.t[3,2].fix(0)

        results = SolverFactory('gurobi').solve(m)
        
        self.assertEqual(results.solver.termination_condition,
                         TerminationCondition.optimal)
        
        TOL = 1e-8
        for c in m.component_data_objects(Constraint, active=True): 
            if c.lower is not None:
                self.assertGreaterEqual(value(c.body) + TOL, value(c.lower))
            if c.upper is not None:
                self.assertLessEqual(value(c.body) - TOL, value(c.upper))
