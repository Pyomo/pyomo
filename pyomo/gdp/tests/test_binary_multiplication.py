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

import pyomo.common.unittest as unittest

from pyomo.environ import (
    TransformationFactory,
    Block,
    Constraint,
    ConcreteModel,
    Var,
    Any,
    SolverFactory,
)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.compare import assertExpressionsEqual

import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct

import random

gurobi_available = (
    SolverFactory('gurobi').available(exception_flag=False)
    and SolverFactory('gurobi').license_is_valid()
)


class CommonTests:
    def diff_apply_to_and_create_using(self, model):
        ct.diff_apply_to_and_create_using(self, model, 'gdp.binary_multiplication')


class TwoTermDisj(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_new_block_created(self):
        m = models.makeTwoTermDisj()
        TransformationFactory('gdp.binary_multiplication').apply_to(m)

        # we have a transformation block
        transBlock = m.component("_pyomo_gdp_binary_multiplication_reformulation")
        self.assertIsInstance(transBlock, Block)

        disjBlock = transBlock.component("relaxedDisjuncts")
        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)
        # it has the disjuncts on it
        self.assertIs(m.d[0].transformation_block, disjBlock[0])
        self.assertIs(m.d[1].transformation_block, disjBlock[1])

    def test_disjunction_deactivated(self):
        ct.check_disjunction_deactivated(self, 'binary_multiplication')

    def test_disjunctDatas_deactivated(self):
        ct.check_disjunctDatas_deactivated(self, 'binary_multiplication')

    def test_do_not_transform_twice_if_disjunction_reactivated(self):
        ct.check_do_not_transform_twice_if_disjunction_reactivated(
            self, 'binary_multiplication'
        )

    def test_xor_constraint_mapping(self):
        ct.check_xor_constraint_mapping(self, 'binary_multiplication')

    def test_xor_constraint_mapping_two_disjunctions(self):
        ct.check_xor_constraint_mapping_two_disjunctions(self, 'binary_multiplication')

    def test_disjunct_mapping(self):
        ct.check_disjunct_mapping(self, 'binary_multiplication')

    def test_disjunct_and_constraint_maps(self):
        """Tests the actual data structures used to store the maps."""
        m = models.makeTwoTermDisj()
        binary_multiplication = TransformationFactory('gdp.binary_multiplication')
        binary_multiplication.apply_to(m)
        disjBlock = m._pyomo_gdp_binary_multiplication_reformulation.relaxedDisjuncts
        oldblock = m.component("d")

        # we are counting on the fact that the disjuncts get relaxed in the
        # same order every time.
        for i in [0, 1]:
            self.assertIs(oldblock[i].transformation_block, disjBlock[i])
            self.assertIs(
                binary_multiplication.get_src_disjunct(disjBlock[i]), oldblock[i]
            )

        # check constraint dict has right mapping
        c1_list = binary_multiplication.get_transformed_constraints(oldblock[1].c1)
        # this is an equality
        self.assertEqual(len(c1_list), 1)
        self.assertIs(c1_list[0].parent_block(), disjBlock[1])
        self.assertIs(
            binary_multiplication.get_src_constraint(c1_list[0]), oldblock[1].c1
        )

        c2_list = binary_multiplication.get_transformed_constraints(oldblock[1].c2)
        # just ub
        self.assertEqual(len(c2_list), 1)
        self.assertIs(c2_list[0].parent_block(), disjBlock[1])
        self.assertIs(
            binary_multiplication.get_src_constraint(c2_list[0]), oldblock[1].c2
        )

        c_list = binary_multiplication.get_transformed_constraints(oldblock[0].c)
        # just lb
        self.assertEqual(len(c_list), 1)
        self.assertIs(c_list[0].parent_block(), disjBlock[0])
        self.assertIs(
            binary_multiplication.get_src_constraint(c_list[0]), oldblock[0].c
        )

    def test_new_block_nameCollision(self):
        ct.check_transformation_block_name_collision(self, 'binary_multiplication')

    def test_indicator_vars(self):
        ct.check_indicator_vars(self, 'binary_multiplication')

    def test_xor_constraints(self):
        ct.check_xor_constraint(self, 'binary_multiplication')

    def test_or_constraints(self):
        m = models.makeTwoTermDisj()
        m.disjunction.xor = False
        TransformationFactory('gdp.binary_multiplication').apply_to(m)

        # check or constraint is an or (upper bound is None)
        orcons = m._pyomo_gdp_binary_multiplication_reformulation.component(
            "disjunction_xor"
        )
        self.assertIsInstance(orcons, Constraint)
        assertExpressionsEqual(
            self,
            orcons.body,
            EXPR.LinearExpression(
                [m.d[0].binary_indicator_var, m.d[1].binary_indicator_var]
            ),
        )
        self.assertEqual(orcons.lower, 1)
        self.assertIsNone(orcons.upper)

    def test_deactivated_constraints(self):
        ct.check_deactivated_constraints(self, 'binary_multiplication')

    def test_transformed_constraints(self):
        m = models.makeTwoTermDisj()
        binary_multiplication = TransformationFactory('gdp.binary_multiplication')
        binary_multiplication.apply_to(m)
        self.check_transformed_constraints(m, binary_multiplication, -3, 2, 7, 2)

    def test_do_not_transform_userDeactivated_disjuncts(self):
        ct.check_user_deactivated_disjuncts(self, 'binary_multiplication')

    def test_improperly_deactivated_disjuncts(self):
        ct.check_improperly_deactivated_disjuncts(self, 'binary_multiplication')

    def test_do_not_transform_userDeactivated_IndexedDisjunction(self):
        ct.check_do_not_transform_userDeactivated_indexedDisjunction(
            self, 'binary_multiplication'
        )

    def check_transformed_constraints(
        self, model, binary_multiplication, cons1lb, cons2lb, cons2ub, cons3ub
    ):
        disjBlock = (
            model._pyomo_gdp_binary_multiplication_reformulation.relaxedDisjuncts
        )

        # first constraint
        c = binary_multiplication.get_transformed_constraints(model.d[0].c)
        self.assertEqual(len(c), 1)
        c_lb = c[0]
        self.assertTrue(c[0].active)
        ind_var = model.d[0].indicator_var
        assertExpressionsEqual(
            self, c[0].body, (model.a - model.d[0].c.lower) * ind_var
        )
        self.assertEqual(c[0].lower, 0)
        self.assertIsNone(c[0].upper)

        # second constraint
        c = binary_multiplication.get_transformed_constraints(model.d[1].c1)
        self.assertEqual(len(c), 1)
        c_eq = c[0]
        self.assertTrue(c[0].active)
        ind_var = model.d[1].indicator_var
        assertExpressionsEqual(self, c[0].body, model.a * ind_var)
        self.assertEqual(c[0].lower, 0)
        self.assertEqual(c[0].upper, 0)

        # third constraint
        c = binary_multiplication.get_transformed_constraints(model.d[1].c2)
        self.assertEqual(len(c), 1)
        c_ub = c[0]
        self.assertTrue(c_ub.active)
        assertExpressionsEqual(
            self, c_ub.body, (model.x - model.d[1].c2.upper) * ind_var
        )
        self.assertIsNone(c_ub.lower)
        self.assertEqual(c_ub.upper, 0)

    def test_create_using(self):
        m = models.makeTwoTermDisj()
        self.diff_apply_to_and_create_using(m)

    def test_indexed_constraints_in_disjunct(self):
        m = ConcreteModel()
        m.I = [1, 2, 3]
        m.x = Var(m.I, bounds=(0, 10))

        def c_rule(b, i):
            m = b.model()
            return m.x[i] >= i

        def d_rule(d, j):
            m = d.model()
            d.c = Constraint(m.I[:j], rule=c_rule)

        m.d = Disjunct(m.I, rule=d_rule)
        m.disjunction = Disjunction(expr=[m.d[i] for i in m.I])

        TransformationFactory('gdp.binary_multiplication').apply_to(m)
        transBlock = m._pyomo_gdp_binary_multiplication_reformulation

        # 2 blocks: the original Disjunct and the transformation block
        self.assertEqual(len(list(m.component_objects(Block, descend_into=False))), 1)
        self.assertEqual(len(list(m.component_objects(Disjunct))), 1)

        # Each relaxed disjunct should have 1 var (the reference to the
        # indicator var), and i "d[i].c" Constraints
        for i in [1, 2, 3]:
            relaxed = transBlock.relaxedDisjuncts[i - 1]
            self.assertEqual(len(list(relaxed.component_objects(Var))), 1)
            self.assertEqual(len(list(relaxed.component_data_objects(Var))), 1)
            self.assertEqual(len(list(relaxed.component_objects(Constraint))), 1)
            self.assertEqual(len(list(relaxed.component_data_objects(Constraint))), i)

    def test_virtual_indexed_constraints_in_disjunct(self):
        m = ConcreteModel()
        m.I = [1, 2, 3]
        m.x = Var(m.I, bounds=(0, 10))

        def d_rule(d, j):
            m = d.model()
            d.c = Constraint(Any)
            for k in range(j):
                d.c[k + 1] = m.x[k + 1] >= k + 1

        m.d = Disjunct(m.I, rule=d_rule)
        m.disjunction = Disjunction(expr=[m.d[i] for i in m.I])

        TransformationFactory('gdp.binary_multiplication').apply_to(m)
        transBlock = m._pyomo_gdp_binary_multiplication_reformulation

        # 2 blocks: the original Disjunct and the transformation block
        self.assertEqual(len(list(m.component_objects(Block, descend_into=False))), 1)
        self.assertEqual(len(list(m.component_objects(Disjunct))), 1)

        # Each relaxed disjunct should have 1 var (the reference to the
        # indicator var), and i "d[i].c" Constraints
        for i in [1, 2, 3]:
            relaxed = transBlock.relaxedDisjuncts[i - 1]
            self.assertEqual(len(list(relaxed.component_objects(Var))), 1)
            self.assertEqual(len(list(relaxed.component_data_objects(Var))), 1)
            self.assertEqual(len(list(relaxed.component_objects(Constraint))), 1)
            self.assertEqual(len(list(relaxed.component_data_objects(Constraint))), i)

    def test_local_var(self):
        m = models.localVar()
        binary_multiplication = TransformationFactory('gdp.binary_multiplication')
        binary_multiplication.apply_to(m)

        # we just need to make sure that constraint was transformed correctly,
        # which just means that the M values were correct.
        transformedC = binary_multiplication.get_transformed_constraints(m.disj2.cons)
        self.assertEqual(len(transformedC), 1)
        eq = transformedC[0]
        repn = generate_standard_repn(eq.body)
        self.assertIsNone(repn.nonlinear_expr)
        self.assertEqual(len(repn.linear_coefs), 1)
        self.assertEqual(len(repn.quadratic_coefs), 2)
        ct.check_linear_coef(self, repn, m.disj2.indicator_var, -3)
        ct.check_quadratic_coef(self, repn, m.x, m.disj2.indicator_var, 1)
        ct.check_quadratic_coef(self, repn, m.disj2.y, m.disj2.indicator_var, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(eq.lb, 0)
        self.assertEqual(eq.ub, 0)


class TestNestedGDP(unittest.TestCase):
    @unittest.skipUnless(gurobi_available, "Gurobi is not available")
    def test_do_not_assume_nested_indicators_local(self):
        ct.check_do_not_assume_nested_indicators_local(
            self, 'gdp.binary_multiplication'
        )


if __name__ == '__main__':
    unittest.main()
