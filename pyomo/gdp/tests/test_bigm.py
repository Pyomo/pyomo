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

from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.deprecation import RenamedClass

from pyomo.environ import (
    TransformationFactory,
    Block,
    Set,
    Constraint,
    ComponentMap,
    LogicalConstraint,
    Objective,
    SolverFactory,
    Suffix,
    TerminationCondition,
    ConcreteModel,
    Var,
    Any,
    value,
)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.base import constraint, ConstraintData
from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.common.log import LoggingIntercept
import logging

import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct

import pyomo.network as ntwk

import random

from io import StringIO

gurobi_available = (
    SolverFactory('gurobi').available(exception_flag=False)
    and SolverFactory('gurobi').license_is_valid()
)


class CommonTests:
    def diff_apply_to_and_create_using(self, model):
        ct.diff_apply_to_and_create_using(self, model, 'gdp.bigm')


class TwoTermDisj(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_new_block_created(self):
        m = models.makeTwoTermDisj()
        TransformationFactory('gdp.bigm').apply_to(m)

        # we have a transformation block
        transBlock = m.component("_pyomo_gdp_bigm_reformulation")
        self.assertIsInstance(transBlock, Block)

        disjBlock = transBlock.component("relaxedDisjuncts")
        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)
        # it has the disjuncts on it
        self.assertIs(m.d[0].transformation_block, disjBlock[0])
        self.assertIs(m.d[1].transformation_block, disjBlock[1])

    def test_disjunction_deactivated(self):
        ct.check_disjunction_deactivated(self, 'bigm')

    def test_disjunctDatas_deactivated(self):
        ct.check_disjunctDatas_deactivated(self, 'bigm')

    def test_do_not_transform_twice_if_disjunction_reactivated(self):
        ct.check_do_not_transform_twice_if_disjunction_reactivated(self, 'bigm')

    def test_xor_constraint_mapping(self):
        ct.check_xor_constraint_mapping(self, 'bigm')

    def test_xor_constraint_mapping_two_disjunctions(self):
        ct.check_xor_constraint_mapping_two_disjunctions(self, 'bigm')

    def test_disjunct_mapping(self):
        ct.check_disjunct_mapping(self, 'bigm')

    def test_disjunct_and_constraint_maps(self):
        """Tests the actual data structures used to store the maps."""
        # ESJ: Note that despite outward appearances, this test really is unique
        # to bigm. Because hull handles the a == 0 constraint by fixing the
        # disaggregated variable rather than creating a transformed constraint.
        m = models.makeTwoTermDisj()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        disjBlock = m._pyomo_gdp_bigm_reformulation.relaxedDisjuncts
        oldblock = m.component("d")

        # we are counting on the fact that the disjuncts get relaxed in the
        # same order every time.
        for i in [0, 1]:
            self.assertIs(oldblock[i].transformation_block, disjBlock[i])
            self.assertIs(bigm.get_src_disjunct(disjBlock[i]), oldblock[i])

        # check constraint dict has right mapping
        c1_list = bigm.get_transformed_constraints(oldblock[1].c1)
        # this is an equality, so we have both lb and ub
        self.assertEqual(len(c1_list), 2)
        self.assertIs(c1_list[0].parent_block(), disjBlock[1])
        self.assertIs(bigm.get_src_constraint(c1_list[0]), oldblock[1].c1)
        self.assertIs(c1_list[1].parent_block(), disjBlock[1])
        self.assertIs(bigm.get_src_constraint(c1_list[0]), oldblock[1].c1)

        c2_list = bigm.get_transformed_constraints(oldblock[1].c2)
        # just ub
        self.assertEqual(len(c2_list), 1)
        self.assertIs(c2_list[0].parent_block(), disjBlock[1])
        self.assertIs(bigm.get_src_constraint(c2_list[0]), oldblock[1].c2)

        c_list = bigm.get_transformed_constraints(oldblock[0].c)
        # just lb
        self.assertEqual(len(c_list), 1)
        self.assertIs(c_list[0].parent_block(), disjBlock[0])
        self.assertIs(bigm.get_src_constraint(c_list[0]), oldblock[0].c)

    def test_new_block_nameCollision(self):
        ct.check_transformation_block_name_collision(self, 'bigm')

    def test_indicator_vars(self):
        ct.check_indicator_vars(self, 'bigm')

    def test_xor_constraints(self):
        ct.check_xor_constraint(self, 'bigm')

    def test_or_constraints(self):
        m = models.makeTwoTermDisj()
        m.disjunction.xor = False
        TransformationFactory('gdp.bigm').apply_to(m)

        # check or constraint is an or (upper bound is None)
        orcons = m._pyomo_gdp_bigm_reformulation.component("disjunction_xor")
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
        ct.check_deactivated_constraints(self, 'bigm')

    def test_transformed_constraints(self):
        m = models.makeTwoTermDisj()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        self.checkMs(m, bigm, -3, 2, 7, 2)

    def test_do_not_transform_userDeactivated_disjuncts(self):
        ct.check_user_deactivated_disjuncts(self, 'bigm')

    def test_improperly_deactivated_disjuncts(self):
        ct.check_improperly_deactivated_disjuncts(self, 'bigm')

    def test_do_not_transform_userDeactivated_IndexedDisjunction(self):
        ct.check_do_not_transform_userDeactivated_indexedDisjunction(self, 'bigm')

    # helper method to check the M values in all of the transformed
    # constraints (m, M) is the tuple for M.  This also relies on the
    # disjuncts being transformed in the same order every time.
    def checkMs(self, model, bigm, cons1lb, cons2lb, cons2ub, cons3ub):
        disjBlock = model._pyomo_gdp_bigm_reformulation.relaxedDisjuncts

        # first constraint
        c = bigm.get_transformed_constraints(model.d[0].c)
        self.assertEqual(len(c), 1)
        c_lb = c[0]
        self.assertTrue(c[0].active)
        repn = generate_standard_repn(c[0].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, model.a, 1)
        ct.check_linear_coef(self, repn, model.d[0].indicator_var, cons1lb)
        self.assertEqual(repn.constant, -cons1lb)
        self.assertEqual(c[0].lower, model.d[0].c.lower)
        self.assertIsNone(c[0].upper)

        # second constraint
        c = bigm.get_transformed_constraints(model.d[1].c1)
        self.assertEqual(len(c), 2)
        c_lb = c[0]
        c_ub = c[1]
        self.assertTrue(c[0].active)
        repn = generate_standard_repn(c[0].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, model.a, 1)
        ct.check_linear_coef(self, repn, model.d[1].indicator_var, cons2lb)
        self.assertEqual(repn.constant, -cons2lb)
        self.assertEqual(c[0].lower, model.d[1].c1.lower)
        self.assertIsNone(c[0].upper)
        self.assertTrue(c_ub.active)
        repn = generate_standard_repn(c_ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, model.a, 1)
        ct.check_linear_coef(self, repn, model.d[1].indicator_var, cons2ub)
        self.assertEqual(repn.constant, -cons2ub)
        self.assertIsNone(c_ub.lower)
        self.assertEqual(c_ub.upper, model.d[1].c1.upper)

        # third constraint
        c = bigm.get_transformed_constraints(model.d[1].c2)
        self.assertEqual(len(c), 1)
        c_ub = c[0]
        self.assertTrue(c_ub.active)
        repn = generate_standard_repn(c_ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, model.x, 1)
        ct.check_linear_coef(self, repn, model.d[1].indicator_var, cons3ub)
        self.assertEqual(repn.constant, -cons3ub)
        self.assertIsNone(c_ub.lower)
        self.assertEqual(c_ub.upper, model.d[1].c2.upper)

    def test_suffix_M_None(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        self.checkMs(m, bigm, -20, -20, 20, 20)

    def test_suffix_M_None_on_disjunctData(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # override for the first index:
        m.d[0].BigM = Suffix(direction=Suffix.LOCAL)
        m.d[0].BigM[None] = 18

        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        # there should now be different values of m on d[0] and d[1]
        self.checkMs(m, bigm, -18, -20, 20, 20)

    def test_suffix_M_simpleConstraint_on_disjunctData(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # override for the first index:
        m.d[0].BigM = Suffix(direction=Suffix.LOCAL)
        m.d[0].BigM[m.d[0].c] = 18

        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        self.checkMs(m, bigm, -18, -20, 20, 20)

    def test_arg_M_None(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        # give an arg
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m, bigM={None: 19})
        self.checkMs(m, bigm, -19, -19, 19, 19)

    def test_arg_M_singleNum(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        # give an arg
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m, bigM=19.2)
        self.checkMs(m, bigm, -19.2, -19.2, 19.2, 19.2)

    def test_singleArg_M_tuple(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        # give an arg
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m, bigM=(-18, 19.2))
        self.checkMs(m, bigm, -18, -18, 19.2, 19.2)

    def test_singleArg_M_tuple_wrongLength(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        # give an arg
        self.assertRaisesRegex(
            GDP_Error,
            r"Big-M \([^)]*\) for constraint d\[0\].c is not of "
            r"length two. Expected either a single value or "
            r"tuple or list of length two specifying M values for the lower "
            "and upper sides of the constraint respectively.*",
            TransformationFactory('gdp.bigm').apply_to,
            m,
            bigM=(-18, 19.2, 3),
        )

    def test_singleArg_M_list(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        # give an arg
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m, bigM=[-18, 19.2])
        self.checkMs(m, bigm, -18, -18, 19.2, 19.2)

    def test_singleArg_M_list_wrongLength(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        # give an arg
        self.assertRaisesRegex(
            GDP_Error,
            r"Big-M \[[^\]]*\] for constraint d\[0\].c is not of "
            r"length two. Expected either a single value or "
            r"tuple or list of length two*",
            TransformationFactory('gdp.bigm').apply_to,
            m,
            bigM=[-18, 19.2, 3],
        )

    def test_arg_M_simpleConstraint(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # specify a suffix on constraints so we can be happy we overrode them
        m.BigM[m.d[0].c] = 200
        m.BigM[m.d[1].c1] = 200
        m.BigM[m.d[1].c2] = 200

        # give an arg
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m, bigM={None: 19, m.d[0].c: 18, m.d[1].c1: 17, m.d[1].c2: 16})
        self.checkMs(m, bigm, -18, -17, 17, 16)

    def test_tuple_M_arg(self):
        m = models.makeTwoTermDisj()
        # give a tuple arg
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m, bigM={None: (-20, 19)})
        self.checkMs(m, bigm, -20, -20, 19, 19)

    def test_tuple_M_suffix(self):
        m = models.makeTwoTermDisj()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = (-18, 20)
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        self.checkMs(m, bigm, -18, -18, 20, 20)

    def test_list_M_arg(self):
        m = models.makeTwoTermDisj()
        # give a tuple arg
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m, bigM={None: [-20, 19]})
        self.checkMs(m, bigm, -20, -20, 19, 19)

    def test_list_M_suffix(self):
        m = models.makeTwoTermDisj()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = [-18, 20]
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        self.checkMs(m, bigm, -18, -18, 20, 20)

    def test_tuple_wrong_length_err(self):
        m = models.makeTwoTermDisj()
        M = (-20, 19, 32)
        self.assertRaisesRegex(
            GDP_Error,
            r"Big-M \(-20, 19, 32\) for constraint d\[0\].c is not of "
            r"length two. Expected either a single value or "
            r"tuple or list of length two*",
            TransformationFactory('gdp.bigm').apply_to,
            m,
            bigM={None: M},
        )

    def test_list_wrong_length_err(self):
        m = models.makeTwoTermDisj()
        M = [-20, 19, 34]
        self.assertRaisesRegex(
            GDP_Error,
            r"Big-M \[-20, 19, 34\] for constraint d\[0\].c is not of "
            r"length two. Expected either a single value or "
            r"tuple or list of length two*",
            TransformationFactory('gdp.bigm').apply_to,
            m,
            bigM={None: M},
        )

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

        TransformationFactory('gdp.bigm').apply_to(m)
        transBlock = m._pyomo_gdp_bigm_reformulation

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

        TransformationFactory('gdp.bigm').apply_to(m)
        transBlock = m._pyomo_gdp_bigm_reformulation

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
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)

        # we just need to make sure that constraint was transformed correctly,
        # which just means that the M values were correct.
        transformedC = bigm.get_transformed_constraints(m.disj2.cons)
        self.assertEqual(len(transformedC), 2)
        lb = transformedC[0]
        ub = transformedC[1]
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        ct.check_linear_coef(self, repn, m.disj2.indicator_var, -2)
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        ct.check_linear_coef(self, repn, m.disj2.indicator_var, 3)


class TwoTermDisjNonlinear(unittest.TestCase, CommonTests):
    def test_nonlinear_bigM(self):
        m = models.makeTwoTermDisj_Nonlinear()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        disjBlock = m._pyomo_gdp_bigm_reformulation.relaxedDisjuncts

        # first constraint
        c = bigm.get_transformed_constraints(m.d[0].c)
        self.assertEqual(len(c), 1)
        c_ub = c[0]
        self.assertTrue(c_ub.active)
        repn = generate_standard_repn(c_ub.body)
        self.assertFalse(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, m.x, 1)
        ct.check_linear_coef(self, repn, m.d[0].indicator_var, 94)
        self.assertEqual(repn.constant, -94)
        self.assertEqual(c_ub.upper, m.d[0].c.upper)
        self.assertIsNone(c_ub.lower)

    def test_nonlinear_bigM_missing_var_bounds(self):
        m = models.makeTwoTermDisj_Nonlinear()
        m.y.setlb(None)
        self.assertRaisesRegex(
            GDP_Error,
            r"Cannot estimate M for unbounded "
            r"expressions.\n\t\(found while processing "
            r"constraint 'd\[0\].c'\)",
            TransformationFactory('gdp.bigm').apply_to,
            m,
        )

    def test_nonlinear_disjoint(self):
        m = ConcreteModel()
        x = m.x = Var(bounds=(-4, 4))
        y = m.y = Var(bounds=(-10, 10))
        m.disj = Disjunction(
            expr=[
                [x**2 + y**2 <= 2, x**3 + y**2 + x * y >= 1.0 / 2.0],
                [(x - 3) ** 2 + (y - 3) ** 2 <= 1],
            ]
        )
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        disjBlock = m._pyomo_gdp_bigm_reformulation.relaxedDisjuncts

        # first disjunct, first constraint
        c = bigm.get_transformed_constraints(m.disj_disjuncts[0].constraint[1])
        self.assertEqual(len(c), 1)
        c_ub = c[0]
        repn = generate_standard_repn(c_ub.body)
        self.assertFalse(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 1)
        ct.check_linear_coef(self, repn, m.disj_disjuncts[0].indicator_var, 114)
        self.assertEqual(repn.constant, -114)
        self.assertEqual(c_ub.upper, m.disj_disjuncts[0].constraint[1].upper)
        self.assertIsNone(c_ub.lower)
        # first disjunct, second constraint
        c = bigm.get_transformed_constraints(m.disj_disjuncts[0].constraint[2])
        self.assertEqual(len(c), 1)
        c_lb = c[0]
        repn = generate_standard_repn(c_lb.body)
        self.assertFalse(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 1)
        ct.check_linear_coef(self, repn, m.disj_disjuncts[0].indicator_var, -104.5)
        self.assertEqual(repn.constant, 104.5)
        self.assertEqual(c_lb.lower, m.disj_disjuncts[0].constraint[2].lower)
        self.assertIsNone(c_lb.upper)
        # second disjunct, first constraint
        c = bigm.get_transformed_constraints(m.disj_disjuncts[1].constraint[1])
        self.assertEqual(len(c), 1)
        c_ub = c[0]
        repn = generate_standard_repn(c_ub.body)
        self.assertFalse(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 3)
        ct.check_linear_coef(self, repn, m.x, -6)
        ct.check_linear_coef(self, repn, m.y, -6)
        ct.check_linear_coef(self, repn, m.disj_disjuncts[1].indicator_var, 217)
        self.assertEqual(repn.constant, -199)
        self.assertEqual(c_ub.upper, m.disj_disjuncts[1].constraint[1].upper)
        self.assertIsNone(c_ub.lower)


class TwoTermIndexedDisj(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)
        # These are the pairs of which disjunct indices map to which
        # blocks in the list of block on the transformation
        # block. This is needed in multiple tests, so I am storing it
        # here.
        self.pairs = [
            ((0, 1, 'A'), 0),
            ((1, 1, 'A'), 1),
            ((0, 1, 'B'), 2),
            ((1, 1, 'B'), 3),
            ((0, 2, 'A'), 4),
            ((1, 2, 'A'), 5),
            ((0, 2, 'B'), 6),
            ((1, 2, 'B'), 7),
        ]

    def test_xor_constraints(self):
        ct.check_indexed_xor_constraints(self, 'bigm')

    def test_deactivated_constraints(self):
        ct.check_constraints_deactivated_indexedDisjunction(self, 'bigm')

    def test_transformed_block_structure(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
        TransformationFactory('gdp.bigm').apply_to(m)
        transBlock = m.component("_pyomo_gdp_bigm_reformulation")
        self.assertIsInstance(transBlock, Block)

        # check the IndexedBlock of transformed disjuncts
        disjBlock = transBlock.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 8)

        # check that all 8 blocks have exactly one constraint on them.
        for i, j in self.pairs:
            self.assertEqual(len(disjBlock[j].component_map(Constraint)), 1)

    def test_disjunct_and_constraint_maps(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)

        disjBlock = m._pyomo_gdp_bigm_reformulation.relaxedDisjuncts
        oldblock = m.component("disjunct")

        # this test relies on the fact that the disjuncts are going to be
        # relaxed in the same order every time, so they will correspond to
        # these indices on the transformation block:
        for src, dest in self.pairs:
            srcDisjunct = oldblock[src]
            transformedDisjunct = disjBlock[dest]
            self.assertIs(bigm.get_src_disjunct(transformedDisjunct), srcDisjunct)
            self.assertIs(transformedDisjunct, srcDisjunct.transformation_block)

            transformed = bigm.get_transformed_constraints(srcDisjunct.c)
            if src[0]:
                # equality
                self.assertEqual(len(transformed), 2)
                self.assertIsInstance(transformed[0], ConstraintData)
                self.assertIsInstance(transformed[1], ConstraintData)
                self.assertIs(bigm.get_src_constraint(transformed[0]), srcDisjunct.c)
                self.assertIs(bigm.get_src_constraint(transformed[1]), srcDisjunct.c)
            else:
                # >=
                self.assertEqual(len(transformed), 1)
                self.assertIsInstance(transformed[0], ConstraintData)
                # check reverse map from the container
                self.assertIs(bigm.get_src_constraint(transformed[0]), srcDisjunct.c)

    def test_deactivated_disjuncts(self):
        ct.check_deactivated_disjuncts(self, 'bigm')

    def test_deactivated_disjunction(self):
        ct.check_deactivated_disjunctions(self, 'bigm')

    def test_create_using(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
        self.diff_apply_to_and_create_using(m)


class DisjOnBlock(unittest.TestCase, CommonTests):
    # when the disjunction is on a block, we want all of the stuff created by
    # the transformation to go on that block also so that solving the block
    # maintains its meaning

    def test_xor_constraint_added(self):
        ct.check_xor_constraint_added(self, 'bigm')

    def test_trans_block_created(self):
        ct.check_trans_block_created(self, 'bigm')

    def checkFirstDisjMs(self, model, disj1c1lb, disj1c1ub, disj1c2):
        bigm = TransformationFactory('gdp.bigm')

        c1 = bigm.get_transformed_constraints(model.b.disjunct[0].c)
        self.assertEqual(len(c1), 2)
        lb = c1[0]
        ub = c1[1]
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c1lb)
        ct.check_linear_coef(self, repn, model.b.disjunct[0].indicator_var, disj1c1lb)
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c1ub)
        ct.check_linear_coef(self, repn, model.b.disjunct[0].indicator_var, disj1c1ub)

        c2 = bigm.get_transformed_constraints(model.b.disjunct[1].c)
        self.assertEqual(len(c2), 1)
        ub = c2[0]
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c2)
        ct.check_linear_coef(self, repn, model.b.disjunct[1].indicator_var, disj1c2)

    def checkMs(self, model, disj1c1lb, disj1c1ub, disj1c2, disj2c1, disj2c2):
        bigm = TransformationFactory('gdp.bigm')
        self.checkFirstDisjMs(model, disj1c1lb, disj1c1ub, disj1c2)

        c = bigm.get_transformed_constraints(model.simpledisj.c)
        self.assertEqual(len(c), 1)
        lb = c[0]
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj2c1)
        ct.check_linear_coef(self, repn, model.simpledisj.indicator_var, disj2c1)

        c = bigm.get_transformed_constraints(model.simpledisj2.c)
        self.assertEqual(len(c), 1)
        ub = c[0]
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj2c2)
        ct.check_linear_coef(self, repn, model.simpledisj2.indicator_var, disj2c2)

    def test_suffix_M_onBlock(self):
        m = models.makeTwoTermDisjOnBlock()
        # adding something that's not on the block so that I know that only
        # the stuff on the block was changed
        m = models.add_disj_not_on_block(m)
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[None] = 34
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)

        # check m values
        self.checkMs(m, -34, 34, 34, -3, 1.5)

        # check the source of the values
        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.simpledisj.c
        )
        self.assertIsNone(l_src)
        self.assertIsNone(u_src)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertEqual(l_val, -3)
        self.assertIsNone(u_val)
        (l_val, u_val) = bigm.get_M_value(m.simpledisj.c)
        self.assertEqual(l_val, -3)
        self.assertIsNone(u_val)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.simpledisj2.c
        )
        self.assertIsNone(l_src)
        self.assertIsNone(u_src)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 1.5)
        (l_val, u_val) = bigm.get_M_value(m.simpledisj2.c)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 1.5)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.b.disjunct[0].c
        )
        self.assertIs(l_src, m.b.BigM)
        self.assertIs(u_src, m.b.BigM)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertEqual(l_val, -34)
        self.assertEqual(u_val, 34)
        l_val, u_val = bigm.get_M_value(m.b.disjunct[0].c)
        self.assertEqual(l_val, -34)
        self.assertEqual(u_val, 34)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.b.disjunct[1].c
        )
        self.assertIsNone(l_src)
        self.assertIs(u_src, m.b.BigM)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 34)
        l_val, u_val = bigm.get_M_value(m.b.disjunct[1].c)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 34)

    def test_block_M_arg(self):
        m = models.makeTwoTermDisjOnBlock()
        m = models.add_disj_not_on_block(m)
        bigms = {m.b: 100, m.b.disjunct[1].c: 13}
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m, bigM=bigms)
        self.checkMs(m, -100, 100, 13, -3, 1.5)

        # check the source of the values
        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.simpledisj.c
        )
        self.assertIsNone(l_src)
        self.assertIsNone(u_src)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertEqual(l_val, -3)
        self.assertIsNone(u_val)
        (l_val, u_val) = bigm.get_M_value(m.simpledisj.c)
        self.assertEqual(l_val, -3)
        self.assertIsNone(u_val)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.simpledisj2.c
        )
        self.assertIsNone(l_src)
        self.assertIsNone(u_src)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 1.5)
        (l_val, u_val) = bigm.get_M_value(m.simpledisj2.c)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 1.5)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.b.disjunct[0].c
        )
        self.assertIs(l_src, bigms)
        self.assertIs(u_src, bigms)
        self.assertIs(l_key, m.b)
        self.assertIs(u_key, m.b)
        self.assertEqual(l_val, -100)
        self.assertEqual(u_val, 100)
        l_val, u_val = bigm.get_M_value(m.b.disjunct[0].c)
        self.assertEqual(l_val, -100)
        self.assertEqual(u_val, 100)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.b.disjunct[1].c
        )
        self.assertIsNone(l_src)
        self.assertIs(u_src, bigms)
        self.assertIsNone(l_key)
        self.assertIs(u_key, m.b.disjunct[1].c)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 13)
        l_val, u_val = bigm.get_M_value(m.b.disjunct[1].c)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 13)

    def test_disjunct_M_arg(self):
        m = models.makeTwoTermDisjOnBlock()
        m = models.add_disj_not_on_block(m)
        bigm = TransformationFactory('gdp.bigm')
        bigms = {m.b: 100, m.b.disjunct[1]: 13}
        bigm.apply_to(m, bigM=bigms)
        self.checkMs(m, -100, 100, 13, -3, 1.5)

        # check the source of the values
        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.simpledisj.c
        )
        self.assertIsNone(l_src)
        self.assertIsNone(u_src)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertEqual(l_val, -3)
        self.assertIsNone(u_val)
        (l_val, u_val) = bigm.get_M_value(m.simpledisj.c)
        self.assertEqual(l_val, -3)
        self.assertIsNone(u_val)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.simpledisj2.c
        )
        self.assertIsNone(l_src)
        self.assertIsNone(u_src)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 1.5)
        (l_val, u_val) = bigm.get_M_value(m.simpledisj2.c)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 1.5)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.b.disjunct[0].c
        )
        self.assertIs(l_src, bigms)
        self.assertIs(u_src, bigms)
        self.assertIs(l_key, m.b)
        self.assertIs(u_key, m.b)
        self.assertEqual(l_val, -100)
        self.assertEqual(u_val, 100)
        l_val, u_val = bigm.get_M_value(m.b.disjunct[0].c)
        self.assertEqual(l_val, -100)
        self.assertEqual(u_val, 100)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.b.disjunct[1].c
        )
        self.assertIsNone(l_src)
        self.assertIs(u_src, bigms)
        self.assertIsNone(l_key)
        self.assertIs(u_key, m.b.disjunct[1])
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 13)
        l_val, u_val = bigm.get_M_value(m.b.disjunct[1].c)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 13)

    def test_block_M_arg_with_default(self):
        m = models.makeTwoTermDisjOnBlock()
        m = models.add_disj_not_on_block(m)
        bigm = TransformationFactory('gdp.bigm')
        bigms = {
            m.b: 100,
            m.b.disjunct[1].c: 13,
            m.b.disjunct[0].c: (None, 50),
            None: 34,
        }
        bigm.apply_to(m, bigM=bigms)
        self.checkMs(m, -100, 50, 13, -34, 34)

        # check the source of the values
        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.simpledisj.c
        )
        self.assertIs(l_src, bigms)
        self.assertIsNone(u_src)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertEqual(l_val, -34)
        self.assertIsNone(u_val)
        l_val, u_val = bigm.get_M_value(m.simpledisj.c)
        self.assertEqual(l_val, -34)
        self.assertIsNone(u_val)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.simpledisj2.c
        )
        self.assertIsNone(l_src)
        self.assertIs(u_src, bigms)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 34)
        l_val, u_val = bigm.get_M_value(m.simpledisj2.c)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 34)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.b.disjunct[0].c
        )
        self.assertIs(l_src, bigms)
        self.assertIs(u_src, bigms)
        self.assertIs(l_key, m.b)
        self.assertIs(u_key, m.b.disjunct[0].c)
        self.assertEqual(l_val, -100)
        self.assertEqual(u_val, 50)
        l_val, u_val = bigm.get_M_value(m.b.disjunct[0].c)
        self.assertEqual(l_val, -100)
        self.assertEqual(u_val, 50)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.b.disjunct[1].c
        )
        self.assertIsNone(l_src)
        self.assertIs(u_src, bigms)
        self.assertIsNone(l_key)
        self.assertIs(u_key, m.b.disjunct[1].c)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 13)
        l_val, u_val = bigm.get_M_value(m.b.disjunct[1].c)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 13)

    def test_model_M_arg(self):
        m = models.makeTwoTermDisjOnBlock()
        m = models.add_disj_not_on_block(m)
        out = StringIO()
        with LoggingIntercept(out, 'pyomo.gdp.bigm'):
            TransformationFactory('gdp.bigm').apply_to(
                m, bigM={m: 100, m.b.disjunct[1].c: 13}
            )
        self.checkMs(m, -100, 100, 13, -100, 100)
        # make sure we didn't get any warnings when we used all the args
        self.assertEqual(out.getvalue(), '')

    def test_model_M_arg_overrides_None(self):
        m = models.makeTwoTermDisjOnBlock()
        m = models.add_disj_not_on_block(m)
        out = StringIO()
        with LoggingIntercept(out, 'pyomo.gdp.bigm'):
            TransformationFactory('gdp.bigm').apply_to(
                m, bigM={m: 100, m.b.disjunct[1].c: 13, None: 34}
            )
        self.checkMs(m, -100, 100, 13, -100, 100)
        self.assertEqual(
            out.getvalue(),
            "Unused arguments in the bigM map! "
            "These arguments were not used by the "
            "transformation:\n\tNone\n\n",
        )

    def test_warning_for_crazy_bigm_args(self):
        m = models.makeTwoTermDisjOnBlock()
        m = models.add_disj_not_on_block(m)
        out = StringIO()
        bigM = ComponentMap({m: 100, m.b.disjunct[1].c: 13})
        # this is silly
        bigM[m.a] = 34
        with LoggingIntercept(out, 'pyomo.gdp.bigm'):
            TransformationFactory('gdp.bigm').apply_to(m, bigM=bigM)
        self.checkMs(m, -100, 100, 13, -100, 100)
        self.assertEqual(
            out.getvalue(),
            "Unused arguments in the bigM map! "
            "These arguments were not used by the "
            "transformation:\n\ta\n\n",
        )

    def test_use_above_scope_m_value(self):
        m = models.makeTwoTermDisjOnBlock()
        m = models.add_disj_not_on_block(m)
        bigM = ComponentMap({m: 100, m.b.disjunct[1].c: 13})
        out = StringIO()
        # transform just the block. We expect to use the M value specified on
        # the model, and we should comment on nothing.
        with LoggingIntercept(out, 'pyomo.gdp.bigm'):
            TransformationFactory('gdp.bigm').apply_to(m.b, bigM=bigM)
        self.checkFirstDisjMs(m, -100, 100, 13)
        self.assertEqual(out.getvalue(), '')

    def test_unused_arguments_transform_block(self):
        m = models.makeTwoTermDisjOnBlock()
        m = models.add_disj_not_on_block(m)

        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 1e6
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[None] = 15

        out = StringIO()
        with LoggingIntercept(out, 'pyomo.gdp.bigm'):
            TransformationFactory('gdp.bigm').apply_to(
                m.b, bigM={m: 100, m.b: 13, m.simpledisj2.c: 10}
            )

        self.checkFirstDisjMs(m, -13, 13, 13)

        # The order these get printed depends on a dictionary order, so test
        # this way...
        self.assertIn(
            "Unused arguments in the bigM map! "
            "These arguments were not used by the "
            "transformation:",
            out.getvalue(),
        )
        self.assertIn("simpledisj2.c", out.getvalue())
        self.assertIn("unknown", out.getvalue())

    def test_suffix_M_simple_disj(self):
        m = models.makeTwoTermDisjOnBlock()
        m = models.add_disj_not_on_block(m)
        m.simpledisj.BigM = Suffix(direction=Suffix.LOCAL)
        m.simpledisj.BigM[None] = 45
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        self.checkMs(m, -20, 20, 20, -45, 20)

        # check source of the m values
        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.simpledisj.c
        )
        self.assertIs(l_src, m.simpledisj.BigM)
        self.assertIsNone(u_src)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertEqual(l_val, -45)
        self.assertIsNone(u_val)
        l_val, u_val = bigm.get_M_value(m.simpledisj.c)
        self.assertEqual(l_val, -45)
        self.assertIsNone(u_val)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.simpledisj2.c
        )
        self.assertIsNone(l_src)
        self.assertIs(u_src, m.BigM)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 20)
        l_val, u_val = bigm.get_M_value(m.simpledisj2.c)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 20)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.b.disjunct[0].c
        )
        self.assertIs(l_src, m.BigM)
        self.assertIs(u_src, m.BigM)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertEqual(l_val, -20)
        self.assertEqual(u_val, 20)
        l_val, u_val = bigm.get_M_value(m.b.disjunct[0].c)
        self.assertEqual(l_val, -20)
        self.assertEqual(u_val, 20)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.b.disjunct[1].c
        )
        self.assertIsNone(l_src)
        self.assertIs(u_src, m.BigM)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 20)
        l_val, u_val = bigm.get_M_value(m.b.disjunct[1].c)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 20)

    def test_suffix_M_constraintKeyOnBlock(self):
        m = models.makeTwoTermDisjOnBlock()
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[m.b.disjunct[0].c] = 87
        m.b.BigM[None] = 64

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkFirstDisjMs(m, -87, 87, 64)

    def test_suffix_M_constraintKeyOnModel(self):
        m = models.makeTwoTermDisjOnBlock()
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[None] = 64
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[m.b.disjunct[0].c] = 87

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkFirstDisjMs(m, -87, 87, 64)

    def test_suffix_M_constraintKeyOnSimpleDisj(self):
        m = models.makeTwoTermDisjOnBlock()
        m = models.add_disj_not_on_block(m)
        m.simpledisj.BigM = Suffix(direction=Suffix.LOCAL)
        m.simpledisj.BigM[None] = 45
        m.simpledisj.BigM[m.simpledisj.c] = 87
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        bigms = {m.b.disjunct[0].c: (-15, None)}
        bigm = TransformationFactory('gdp.bigm')

        bigm.apply_to(m, bigM=bigms)
        self.checkMs(m, -15, 20, 20, -87, 20)

        # check source of the m values
        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.simpledisj.c
        )
        self.assertIs(l_src, m.simpledisj.BigM)
        self.assertIsNone(u_src)
        self.assertIs(l_key, m.simpledisj.c)
        self.assertIsNone(u_key)
        self.assertEqual(l_val, -87)
        self.assertIsNone(u_val)
        l_val, u_val = bigm.get_M_value(m.simpledisj.c)
        self.assertEqual(l_val, -87)
        self.assertIsNone(u_val)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.simpledisj2.c
        )
        self.assertIsNone(l_src)
        self.assertIs(u_src, m.BigM)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 20)
        l_val, u_val = bigm.get_M_value(m.simpledisj2.c)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 20)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.b.disjunct[0].c
        )
        self.assertIs(l_src, bigms)
        self.assertIs(u_src, m.BigM)
        self.assertIs(l_key, m.b.disjunct[0].c)
        self.assertIsNone(u_key)
        self.assertEqual(l_val, -15)
        self.assertEqual(u_val, 20)
        l_val, u_val = bigm.get_M_value(m.b.disjunct[0].c)
        self.assertEqual(l_val, -15)
        self.assertEqual(u_val, 20)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.b.disjunct[1].c
        )
        self.assertIsNone(l_src)
        self.assertIs(u_src, m.BigM)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 20)
        l_val, u_val = bigm.get_M_value(m.b.disjunct[1].c)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 20)

    def test_suffix_M_constraintKeyOnSimpleDisj_deprecated_m_src_method(self):
        m = models.makeTwoTermDisjOnBlock()
        m = models.add_disj_not_on_block(m)
        m.simpledisj.BigM = Suffix(direction=Suffix.LOCAL)
        m.simpledisj.BigM[None] = 45
        m.simpledisj.BigM[m.simpledisj.c] = 87
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        bigms = {m.b.disjunct[0].c: (-15, None)}
        bigm = TransformationFactory('gdp.bigm')

        bigm.apply_to(m, bigM=bigms)

        # check source of the m values
        (src, key) = bigm.get_m_value_src(m.simpledisj.c)
        self.assertIs(src, m.simpledisj.BigM)
        self.assertIs(key, m.simpledisj.c)
        (src, key) = bigm.get_m_value_src(m.simpledisj2.c)
        self.assertIs(src, m.BigM)
        self.assertIsNone(key)
        self.assertRaisesRegex(
            GDP_Error,
            r"This is why this method is deprecated: The lower "
            r"and upper M values for constraint b.disjunct\[0\].c "
            r"came from different sources, please use the "
            r"get_M_value_src method.",
            bigm.get_m_value_src,
            m.b.disjunct[0].c,
        )
        (src, key) = bigm.get_m_value_src(m.b.disjunct[1].c)
        self.assertIs(src, m.BigM)
        self.assertIsNone(key)

    def test_disjunct_M_arg_deprecated_m_src_method(self):
        m = models.makeTwoTermDisjOnBlock()
        m = models.add_disj_not_on_block(m)
        bigm = TransformationFactory('gdp.bigm')
        bigms = {m.b: 100, m.b.disjunct[1]: 13}
        bigm.apply_to(m, bigM=bigms)
        self.checkMs(m, -100, 100, 13, -3, 1.5)

        # check the source of the values
        (src, key) = bigm.get_m_value_src(m.simpledisj.c)
        self.assertEqual(src, -3)
        self.assertIsNone(key)
        (src, key) = bigm.get_m_value_src(m.simpledisj2.c)
        self.assertIsNone(src)
        self.assertEqual(key, 1.5)
        (src, key) = bigm.get_m_value_src(m.b.disjunct[0].c)
        self.assertIs(src, bigms)
        self.assertIs(key, m.b)
        (src, key) = bigm.get_m_value_src(m.b.disjunct[1].c)
        self.assertIs(src, bigms)
        self.assertIs(key, m.b.disjunct[1])

    def test_largest_M_value(self):
        m = models.makeTwoTermDisjOnBlock()
        m = models.add_disj_not_on_block(m)
        bigm = TransformationFactory('gdp.bigm')
        bigms = {m.b: 100, m.b.disjunct[1]: 13}
        bigm.apply_to(m, bigM=bigms)

        self.assertEqual(bigm.get_largest_M_value(m), 100)

    def test_block_targets_inactive(self):
        ct.check_block_targets_inactive(self, 'bigm')

    def test_block_only_targets_transformed(self):
        ct.check_block_only_targets_transformed(self, 'bigm')

    def test_create_using(self):
        m = models.makeTwoTermDisjOnBlock()
        ct.diff_apply_to_and_create_using(self, m, 'gdp.bigm')


class ScalarDisjIndexedConstraints(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_do_not_transform_deactivated_constraintDatas(self):
        # ESJ: specific to how bigM transforms constraints (so not a common test
        # with hull)
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 30
        m.b.simpledisj1.c[1].deactivate()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)

        # the real test: This wasn't transformed
        with self.assertRaisesRegex(
            GDP_Error, r"Constraint 'b.simpledisj1.c\[1\]' has not been transformed."
        ):
            bigm.get_transformed_constraints(m.b.simpledisj1.c[1])

        # and the rest of the container was transformed
        cons_list = bigm.get_transformed_constraints(m.b.simpledisj1.c[2])
        self.assertEqual(len(cons_list), 2)
        lb = cons_list[0]
        ub = cons_list[1]
        self.assertIsInstance(lb, constraint.ConstraintData)
        self.assertIsInstance(ub, constraint.ConstraintData)

    def checkMs(
        self, m, disj1c1lb, disj1c1ub, disj1c2lb, disj1c2ub, disj2c1ub, disj2c2ub
    ):
        bigm = TransformationFactory('gdp.bigm')
        m_values = bigm.get_all_M_values_by_constraint(m)

        c = bigm.get_transformed_constraints(m.b.simpledisj1.c[1])
        self.assertEqual(len(c), 2)
        lb = c[0]
        ub = c[1]
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c1lb)
        ct.check_linear_coef(self, repn, m.b.simpledisj1.indicator_var, disj1c1lb)
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c1ub)
        ct.check_linear_coef(self, repn, m.b.simpledisj1.indicator_var, disj1c1ub)
        self.assertIn(m.b.simpledisj1.c[1], m_values.keys())
        self.assertEqual(m_values[m.b.simpledisj1.c[1]][0], disj1c1lb)
        self.assertEqual(m_values[m.b.simpledisj1.c[1]][1], disj1c1ub)

        c = bigm.get_transformed_constraints(m.b.simpledisj1.c[2])
        self.assertEqual(len(c), 2)
        lb = c[0]
        ub = c[1]
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c2lb)
        ct.check_linear_coef(self, repn, m.b.simpledisj1.indicator_var, disj1c2lb)
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c2ub)
        ct.check_linear_coef(self, repn, m.b.simpledisj1.indicator_var, disj1c2ub)
        self.assertIn(m.b.simpledisj1.c[2], m_values.keys())
        self.assertEqual(m_values[m.b.simpledisj1.c[2]][0], disj1c2lb)
        self.assertEqual(m_values[m.b.simpledisj1.c[2]][1], disj1c2ub)

        c = bigm.get_transformed_constraints(m.b.simpledisj2.c[1])
        self.assertEqual(len(c), 1)
        ub = c[0]
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj2c1ub)
        ct.check_linear_coef(self, repn, m.b.simpledisj2.indicator_var, disj2c1ub)
        self.assertIn(m.b.simpledisj2.c[1], m_values.keys())
        self.assertEqual(m_values[m.b.simpledisj2.c[1]][1], disj2c1ub)
        self.assertIsNone(m_values[m.b.simpledisj2.c[1]][0])

        c = bigm.get_transformed_constraints(m.b.simpledisj2.c[2])
        self.assertEqual(len(c), 1)
        ub = c[0]
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj2c2ub)
        ct.check_linear_coef(self, repn, m.b.simpledisj2.indicator_var, disj2c2ub)
        self.assertIn(m.b.simpledisj2.c[2], m_values.keys())
        self.assertEqual(m_values[m.b.simpledisj2.c[2]][1], disj2c2ub)
        self.assertIsNone(m_values[m.b.simpledisj2.c[2]][0])

        # verify that we don't have anything extra in this dictionary either.
        self.assertEqual(len(m_values), 4)

    def test_suffix_M_constraintData_on_block(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[None] = 30
        m.b.BigM[m.b.simpledisj1.c[1]] = 15

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -15, 15, -30, 30, 30, 30)

    def test_suffix_M_indexedConstraint_on_block(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[None] = 30
        m.b.BigM[m.b.simpledisj2.c] = 15

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -30, 30, -30, 30, 15, 15)

    def test_suffix_M_constraintData_on_simpleDisjunct(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 65
        m.b.simpledisj1.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.simpledisj1.BigM[m.b.simpledisj1.c[2]] = (-14, 13)

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -65, 65, -14, 13, 65, 65)

    def test_suffix_M_indexedConstraint_on_simpleDisjunct(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 65
        m.b.simpledisj1.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.simpledisj1.BigM[m.b.simpledisj1.c] = (-14, 13)

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -14, 13, -14, 13, 65, 65)

    def test_unbounded_var_m_estimation_err(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        self.assertRaisesRegex(
            GDP_Error,
            r"Cannot estimate M for unbounded "
            r"expressions.\n\t\(found while processing "
            r"constraint 'b.simpledisj1.c\[1\]'\). "
            r"Please specify a value of M "
            r"or ensure all variables that appear in the "
            r"constraint are bounded.",
            TransformationFactory('gdp.bigm').apply_to,
            m,
        )

    def test_create_using(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 100
        self.diff_apply_to_and_create_using(m)


class SimpleDisjIndexedConstraints(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarDisjIndexedConstraints
    __renamed__version__ = '6.0'


class MultiTermDisj(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_xor_constraint(self):
        ct.check_three_term_xor_constraint(self, 'bigm')

    def test_create_using(self):
        m = models.makeThreeTermIndexedDisj()
        self.diff_apply_to_and_create_using(m)


class IndexedConstraintsInDisj(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_transformed_constraints_on_block(self):
        # constraints should still be moved as indexed constraints, and we will
        # just add ['lb', 'ub'] as another index (using both for equality and
        # both bounds and the one that we need when we only have one bound)
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)

        transBlock = m.component("_pyomo_gdp_bigm_reformulation")
        self.assertIsInstance(transBlock, Block)
        disjBlock = transBlock.component("relaxedDisjuncts")
        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)

        cons11 = bigm.get_transformed_constraints(m.disjunct[0].c[1])
        self.assertEqual(len(cons11), 1)
        cons11_lb = cons11[0]
        self.assertIsInstance(cons11_lb.parent_component(), Constraint)
        self.assertTrue(cons11_lb.active)
        cons12 = bigm.get_transformed_constraints(m.disjunct[0].c[2])
        self.assertEqual(len(cons12), 1)
        cons12_lb = cons12[0]
        self.assertIsInstance(cons12_lb.parent_component(), Constraint)
        self.assertTrue(cons12_lb.active)

        cons21 = bigm.get_transformed_constraints(m.disjunct[1].c[1])
        self.assertEqual(len(cons21), 2)
        cons21_lb = cons21[0]
        cons21_ub = cons21[1]
        self.assertIsInstance(cons21_lb.parent_component(), Constraint)
        self.assertIsInstance(cons21_ub.parent_component(), Constraint)
        self.assertTrue(cons21_lb.active)
        self.assertTrue(cons21_ub.active)
        cons22 = bigm.get_transformed_constraints(m.disjunct[1].c[2])
        self.assertEqual(len(cons22), 2)
        cons22_lb = cons22[0]
        cons22_ub = cons22[1]
        self.assertIsInstance(cons22_lb.parent_component(), Constraint)
        self.assertIsInstance(cons22_ub.parent_component(), Constraint)
        self.assertTrue(cons22_lb.active)
        self.assertTrue(cons22_ub.active)

    def checkMs(self, model, c11lb, c12lb, c21lb, c21ub, c22lb, c22ub):
        bigm = TransformationFactory('gdp.bigm')
        c = bigm.get_transformed_constraints(model.disjunct[0].c[1])
        self.assertEqual(len(c), 1)
        lb = c[0]
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c11lb)
        ct.check_linear_coef(self, repn, model.disjunct[0].indicator_var, c11lb)
        c = bigm.get_transformed_constraints(model.disjunct[0].c[2])
        self.assertEqual(len(c), 1)
        lb = c[0]
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c12lb)
        ct.check_linear_coef(self, repn, model.disjunct[0].indicator_var, c12lb)

        c = bigm.get_transformed_constraints(model.disjunct[1].c[1])
        self.assertEqual(len(c), 2)
        lb = c[0]
        ub = c[1]
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c21lb)
        ct.check_linear_coef(self, repn, model.disjunct[1].indicator_var, c21lb)
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c21ub)
        ct.check_linear_coef(self, repn, model.disjunct[1].indicator_var, c21ub)
        c = bigm.get_transformed_constraints(model.disjunct[1].c[2])
        self.assertEqual(len(c), 2)
        lb = c[0]
        ub = c[1]
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c22lb)
        ct.check_linear_coef(self, repn, model.disjunct[1].indicator_var, c22lb)
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c22ub)
        ct.check_linear_coef(self, repn, model.disjunct[1].indicator_var, c22ub)

    def test_arg_M_constraintdata(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # specify a suffix on a componentdata so we can be happy we overrode it
        m.BigM[m.disjunct[0].c[1]] = 19

        # give an arg
        TransformationFactory('gdp.bigm').apply_to(
            m, bigM={None: 19, m.disjunct[0].c[1]: 17, m.disjunct[0].c[2]: 18}
        )

        # check that m values are what we expect
        self.checkMs(m, -17, -18, -19, 19, -19, 19)

    def test_arg_M_indexedConstraint(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # specify a suffix on a component so we can be happy we overrode it
        m.BigM[m.disjunct[0].c] = 19

        # give an arg. Doing this one as a ComponentMap, just to make sure.
        TransformationFactory('gdp.bigm').apply_to(
            m, bigM=ComponentMap({None: 19, m.disjunct[0].c: 17})
        )
        self.checkMs(m, -17, -17, -19, 19, -19, 19)

    def test_suffix_M_None_on_indexedConstraint(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        m.BigM[m.disjunct[0].c] = 19
        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -19, -19, -20, 20, -20, 20)

    def test_suffix_M_None_on_constraintdata(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        m.BigM[m.disjunct[0].c[1]] = 19

        TransformationFactory('gdp.bigm').apply_to(m)

        # check that m values are what we expect
        self.checkMs(m, -19, -20, -20, 20, -20, 20)

    def test_suffix_M_indexedConstraint_on_disjData(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # specify a suffix on a disjunctData
        m.disjunct[0].BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[m.disjunct[0].c] = 19

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -19, -19, -20, 20, -20, 20)

    def test_suffix_M_constraintData_on_disjData(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # specify a suffix on a disjunctData
        m.disjunct[0].BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[m.disjunct[0].c] = 19
        m.BigM[m.disjunct[0].c[1]] = 18

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -18, -19, -20, 20, -20, 20)

    def test_create_using(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        self.diff_apply_to_and_create_using(m)


class DisjunctInMultipleDisjunctions(unittest.TestCase, CommonTests):
    def test_error_for_same_disjunct_in_multiple_disjunctions(self):
        ct.check_error_for_same_disjunct_in_multiple_disjunctions(self, 'bigm')


class TestTargets_SingleDisjunction(unittest.TestCase, CommonTests):
    def test_only_targets_inactive(self):
        ct.check_only_targets_inactive(self, 'bigm')

    def test_only_targets_transformed(self):
        ct.check_only_targets_get_transformed(self, 'bigm')

    def test_target_not_a_component_err(self):
        ct.check_target_not_a_component_error(self, 'bigm')

    def test_targets_cannot_be_cuids(self):
        ct.check_targets_cannot_be_cuids(self, 'bigm')

    # [ESJ 09/14/2019] See my rant in #1072, but I think this is why we cannot
    # actually support this!
    # def test_break_targets_with_cuids(self):
    #     m = models.makeTwoSimpleDisjunctions()
    #     b = Block() # so this guy has no parent, he's some mistake presumably
    #     # But we specify *him* has the target with cuid
    #     TransformationFactory('gdp.bigm').apply_to(m, targets=ComponentUID(b))

    #     # No error, and we've transformed the whole model
    #     m.pprint()


class TestTargets_IndexedDisjunction(unittest.TestCase, CommonTests):
    def test_indexedDisj_targets_inactive(self):
        ct.check_indexedDisj_targets_inactive(self, 'bigm')

    def test_indexedDisj_only_targets_transformed(self):
        ct.check_indexedDisj_only_targets_transformed(self, 'bigm')

    def test_warn_for_untransformed(self):
        ct.check_warn_for_untransformed(self, 'bigm')

    def test_disjData_targets_inactive(self):
        ct.check_disjData_targets_inactive(self, 'bigm')

    def test_disjData_only_targets_transformed(self):
        ct.check_disjData_only_targets_transformed(self, 'bigm')

    def test_indexedBlock_targets_inactive(self):
        ct.check_indexedBlock_targets_inactive(self, 'bigm')

    def test_indexedBlock_only_targets_transformed(self):
        ct.check_indexedBlock_only_targets_transformed(self, 'bigm')

    def test_blockData_targets_inactive(self):
        ct.check_blockData_targets_inactive(self, 'bigm')

    def test_blockData_only_targets_transformed(self):
        ct.check_blockData_only_targets_transformed(self, 'bigm')

    def test_do_not_transform_deactivated_targets(self):
        ct.check_do_not_transform_deactivated_targets(self, 'bigm')

    def test_create_using(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        ct.diff_apply_to_and_create_using(self, m, 'gdp.bigm')


class DisjunctionInDisjunct(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_disjuncts_inactive(self):
        ct.check_disjuncts_inactive_nested(self, 'bigm')

    def test_deactivated_disjunct_leaves_nested_disjuncts_active(self):
        ct.check_deactivated_disjunct_leaves_nested_disjunct_active(self, 'bigm')

    def check_disjunction_transformation_block_structure(self, transBlock, pairs):
        self.assertIsInstance(transBlock, Block)

        disjBlock = transBlock.relaxedDisjuncts
        self.assertIsInstance(disjBlock, Block)
        # All the transformed outer disjuncts should be on Block:
        self.assertEqual(len(disjBlock), len(pairs))

        # This test will also rely on the disjunctions being relaxed in the same
        # order every time
        bigm = TransformationFactory('gdp.bigm')
        for i, j in pairs:
            for comp in j:
                self.assertIs(
                    bigm.get_transformed_constraints(comp)[0].parent_block(),
                    disjBlock[i],
                )

    def test_transformation_block_structure(self):
        m = models.makeNestedDisjunctions()
        TransformationFactory('gdp.bigm').apply_to(m)

        # This is the transformation block for the outer disjunction
        transBlock = m.disjunction.algebraic_constraint.parent_block()
        pairs = [
            (0, [m.simpledisjunct.innerdisjunct0.c]),
            (1, [m.simpledisjunct.innerdisjunct1.c]),
            (2, []),  # No constraints, just a reference to simpledisjunct's
            # indicator_var
            (5, [m.disjunct[0].c]),
            (2, [m.disjunct[1].innerdisjunct[0].c]),
            (3, [m.disjunct[1].innerdisjunct[1].c]),
            (6, []),  # Again no constraints, just indicator var ref
        ]
        self.check_disjunction_transformation_block_structure(transBlock, pairs)
        # we have the XOR constraints for both the outer and inner disjunctions
        self.assertIsInstance(transBlock.component("disjunction_xor"), Constraint)

    def test_mappings_between_disjunctions_and_xors(self):
        m = models.makeNestedDisjunctions()
        transform = TransformationFactory('gdp.bigm')
        transform.apply_to(m)

        transBlock1 = m.component("_pyomo_gdp_bigm_reformulation")
        transBlock2 = m.disjunct[1].component("_pyomo_gdp_bigm_reformulation")
        transBlock3 = m.simpledisjunct.component("_pyomo_gdp_bigm_reformulation")

        disjunctionPairs = [
            (m.disjunction, transBlock1.disjunction_xor),
            (m.disjunct[1].innerdisjunction[0], transBlock2.innerdisjunction_xor[0]),
            (m.simpledisjunct.innerdisjunction, transBlock3.innerdisjunction_xor),
        ]

        # check disjunction mappings
        for disjunction, xor in disjunctionPairs:
            self.assertIs(disjunction.algebraic_constraint, xor)
            self.assertIs(transform.get_src_disjunction(xor), disjunction)

    def test_disjunct_mappings(self):
        m = models.makeNestedDisjunctions()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)

        disjunctBlocks = m._pyomo_gdp_bigm_reformulation.relaxedDisjuncts

        # I want to check that I correctly updated the pointers to the
        # transformation blocks on the inner Disjuncts.
        self.assertIs(
            m.disjunct[1].innerdisjunct[0].transformation_block, disjunctBlocks[2]
        )
        self.assertIs(disjunctBlocks[2]._src_disjunct(), m.disjunct[1].innerdisjunct[0])

        self.assertIs(
            m.disjunct[1].innerdisjunct[1].transformation_block, disjunctBlocks[3]
        )
        self.assertIs(disjunctBlocks[3]._src_disjunct(), m.disjunct[1].innerdisjunct[1])

        self.assertIs(
            m.simpledisjunct.innerdisjunct0.transformation_block, disjunctBlocks[0]
        )
        self.assertIs(
            disjunctBlocks[0]._src_disjunct(), m.simpledisjunct.innerdisjunct0
        )

        self.assertIs(
            m.simpledisjunct.innerdisjunct1.transformation_block, disjunctBlocks[1]
        )
        self.assertIs(
            disjunctBlocks[1]._src_disjunct(), m.simpledisjunct.innerdisjunct1
        )

    def test_m_value_mappings(self):
        m = models.makeNestedDisjunctions()
        bigm = TransformationFactory('gdp.bigm')
        m.simpledisjunct.BigM = Suffix(direction=Suffix.LOCAL)
        m.simpledisjunct.BigM[None] = 58
        m.simpledisjunct.BigM[m.simpledisjunct.innerdisjunct0.c] = 42
        bigms = {m.disjunct[1].innerdisjunct[0]: 89}
        bigm.apply_to(m, bigM=bigms)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.disjunct[1].innerdisjunct[0].c
        )
        self.assertIs(l_src, bigms)
        self.assertIs(u_src, bigms)
        self.assertIs(l_key, m.disjunct[1].innerdisjunct[0])
        self.assertIs(u_key, m.disjunct[1].innerdisjunct[0])
        self.assertEqual(l_val, -89)
        self.assertEqual(u_val, 89)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.disjunct[1].innerdisjunct[1].c
        )
        self.assertIsNone(l_src)
        self.assertIsNone(u_src)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertEqual(l_val, -5)
        self.assertIsNone(u_val)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.disjunct[0].c
        )
        self.assertIsNone(l_src)
        self.assertIsNone(u_src)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertEqual(l_val, -11)
        self.assertEqual(u_val, 7)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.disjunct[1].c
        )
        self.assertIsNone(l_src)
        self.assertIsNone(u_src)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 21)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.simpledisjunct.innerdisjunct0.c
        )
        self.assertIsNone(l_src)
        self.assertIs(u_src, m.simpledisjunct.BigM)
        self.assertIsNone(l_key)
        self.assertIs(u_key, m.simpledisjunct.innerdisjunct0.c)
        self.assertIsNone(l_val)
        self.assertEqual(u_val, 42)

        ((l_val, l_src, l_key), (u_val, u_src, u_key)) = bigm.get_M_value_src(
            m.simpledisjunct.innerdisjunct1.c
        )
        self.assertIs(l_src, m.simpledisjunct.BigM)
        self.assertIsNone(u_src)
        self.assertIsNone(l_key)
        self.assertIsNone(u_key)
        self.assertEqual(l_val, -58)
        self.assertIsNone(u_val)

    # many of the transformed constraints look like this, so can call this
    # function to test them.
    def check_bigM_constraint(self, cons, variable, M, indicator_var):
        assertExpressionsEqual(
            self,
            cons.body,
            variable - float(M) * (1 - indicator_var.get_associated_binary()),
        )

    def check_inner_xor_constraint(self, inner_disjunction, outer_disjunct, bigm):
        inner_xor = inner_disjunction.algebraic_constraint
        sum_indicators = sum(
            d.binary_indicator_var for d in inner_disjunction.disjuncts
        )
        assertExpressionsEqual(self, inner_xor.expr, sum_indicators == 1)
        # this guy has been transformed
        self.assertFalse(inner_xor.active)
        cons = bigm.get_transformed_constraints(inner_xor)
        self.assertEqual(len(cons), 2)
        lb = cons[0]
        ct.check_obj_in_active_tree(self, lb)
        lb_expr = self.simplify_cons(lb, leq=False)
        assertExpressionsEqual(
            self,
            lb_expr,
            1.0 <= sum_indicators - outer_disjunct.binary_indicator_var + 1,
        )
        ub = cons[1]
        ct.check_obj_in_active_tree(self, ub)
        ub_expr = self.simplify_cons(ub, leq=True)
        assertExpressionsEqual(
            self,
            ub_expr,
            sum_indicators + outer_disjunct.binary_indicator_var - 1 <= 1.0,
        )

    def test_transformed_constraints(self):
        # We'll check all the transformed constraints to make sure
        # that nothing was transformed twice. The real key is that the
        # xor constraints created by the inner disjunctions get
        # transformed by the outer ones.
        m = models.makeNestedDisjunctions()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        cons1 = bigm.get_transformed_constraints(m.disjunct[1].innerdisjunct[0].c)
        self.assertEqual(len(cons1), 2)
        cons1lb = cons1[0]
        cons1ub = cons1[1]
        self.assertEqual(cons1lb.lower, 0)
        self.assertIsNone(cons1lb.upper)
        assertExpressionsEqual(
            self,
            cons1lb.body,
            EXPR.SumExpression(
                [
                    m.z,
                    EXPR.NegationExpression(
                        (
                            EXPR.ProductExpression(
                                (
                                    0.0,
                                    EXPR.LinearExpression(
                                        [
                                            1,
                                            EXPR.MonomialTermExpression(
                                                (
                                                    -1,
                                                    m.disjunct[1]
                                                    .innerdisjunct[0]
                                                    .binary_indicator_var,
                                                )
                                            ),
                                            1,
                                            EXPR.MonomialTermExpression(
                                                (-1, m.disjunct[1].binary_indicator_var)
                                            ),
                                        ]
                                    ),
                                )
                            ),
                        )
                    ),
                ]
            ),
        )
        assertExpressionsEqual(
            self,
            cons1ub.expr,
            m.z
            - 10.0
            * (
                1
                - m.disjunct[1].innerdisjunct[0].binary_indicator_var
                + 1
                - m.disjunct[1].binary_indicator_var
            )
            <= 0.0,
        )

        cons2 = bigm.get_transformed_constraints(m.disjunct[1].innerdisjunct[1].c)
        self.assertEqual(len(cons2), 1)
        cons2lb = cons2[0]
        assertExpressionsEqual(
            self,
            cons2lb.expr,
            5.0
            <= m.z
            - (-5.0)
            * (
                1
                - m.disjunct[1].innerdisjunct[1].binary_indicator_var
                + 1
                - m.disjunct[1].binary_indicator_var
            ),
        )

        cons3 = bigm.get_transformed_constraints(m.simpledisjunct.innerdisjunct0.c)
        self.assertEqual(len(cons3), 1)
        cons3ub = cons3[0]
        assertExpressionsEqual(
            self,
            cons3ub.expr,
            m.x
            - 7.0
            * (
                1
                - m.simpledisjunct.innerdisjunct0.binary_indicator_var
                + 1
                - m.simpledisjunct.binary_indicator_var
            )
            <= 2.0,
        )

        cons4 = bigm.get_transformed_constraints(m.simpledisjunct.innerdisjunct1.c)
        self.assertEqual(len(cons4), 1)
        cons4lb = cons4[0]
        assertExpressionsEqual(
            self,
            cons4lb.expr,
            m.x
            - (-13.0)
            * (
                1
                - m.simpledisjunct.innerdisjunct1.binary_indicator_var
                + 1
                - m.simpledisjunct.binary_indicator_var
            )
            >= 4.0,
        )

        # Here we check that the xor constraint from
        # simpledisjunct.innerdisjunction is transformed.
        self.check_inner_xor_constraint(
            m.simpledisjunct.innerdisjunction, m.simpledisjunct, bigm
        )

        cons6 = bigm.get_transformed_constraints(m.disjunct[0].c)
        self.assertEqual(len(cons6), 2)
        cons6lb = cons6[0]
        self.assertIsNone(cons6lb.upper)
        self.assertEqual(cons6lb.lower, 2)
        self.check_bigM_constraint(cons6lb, m.x, -11, m.disjunct[0].indicator_var)
        cons6ub = cons6[1]
        self.assertIsNone(cons6ub.lower)
        self.assertEqual(cons6ub.upper, 2)
        self.check_bigM_constraint(cons6ub, m.x, 7, m.disjunct[0].indicator_var)

        # now we check that the xor constraint from disjunct[1].innerdisjunction
        # is correct.
        self.check_inner_xor_constraint(
            m.disjunct[1].innerdisjunction[0], m.disjunct[1], bigm
        )

        cons8 = bigm.get_transformed_constraints(m.disjunct[1].c)
        self.assertEqual(len(cons8), 1)
        cons8ub = cons8[0]
        self.assertIsNone(cons8ub.lower)
        self.assertEqual(cons8ub.upper, 2)
        self.check_bigM_constraint(cons8ub, m.a, 21, m.disjunct[1].indicator_var)

    def test_unique_reference_to_nested_indicator_var(self):
        ct.check_unique_reference_to_nested_indicator_var(self, 'bigm')

    def test_disjunct_targets_inactive(self):
        ct.check_disjunct_targets_inactive(self, 'bigm')

    def test_disjunct_only_targets_transformed(self):
        ct.check_disjunct_only_targets_transformed(self, 'bigm')

    def test_disjunctData_targets_inactive(self):
        ct.check_disjunctData_targets_inactive(self, 'bigm')

    def test_disjunctData_only_targets_transformed(self):
        ct.check_disjunctData_only_targets_transformed(self, 'bigm')

    def test_cannot_call_transformation_on_disjunction(self):
        ct.check_cannot_call_transformation_on_disjunction(self, 'bigm')

    def test_disjunction_target_err(self):
        ct.check_disjunction_target_err(self, 'bigm')

    def test_nested_disjunction_target(self):
        ct.check_nested_disjunction_target(self, 'bigm')

    def test_target_appears_twice(self):
        ct.check_target_appears_twice(self, 'bigm')

    def test_create_using(self):
        m = models.makeNestedDisjunctions()
        self.diff_apply_to_and_create_using(m)

    def test_indexed_nested_disjunction(self):
        # When we have a nested disjunction inside of a disjunct, we need to
        # make sure that we don't delete the relaxedDisjuncts container because
        # we will end up moving things out of it in two different steps. If that
        # were to happen, this would throw an error when it can't find the block
        # the second time.
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d1.indexedDisjunct1 = Disjunct([0, 1])
        m.d1.indexedDisjunct2 = Disjunct([0, 1])

        @m.d1.Disjunction([0, 1])
        def innerIndexed(d, i):
            return [d.indexedDisjunct1[i], d.indexedDisjunct2[i]]

        m.d2 = Disjunct()
        m.outer = Disjunction(expr=[m.d1, m.d2])

        TransformationFactory('gdp.bigm').apply_to(m)

        # we check that they all ended up on the same Block in the end (I don't
        # really care in what order for this test)
        disjuncts = [
            m.d1,
            m.d2,
            m.d1.indexedDisjunct1[0],
            m.d1.indexedDisjunct1[1],
            m.d1.indexedDisjunct2[0],
            m.d1.indexedDisjunct2[1],
        ]
        for disjunct in disjuncts:
            self.assertIs(
                disjunct.transformation_block.parent_component(),
                m._pyomo_gdp_bigm_reformulation.relaxedDisjuncts,
            )

    def simplify_cons(self, cons, leq):
        visitor = LinearRepnVisitor({}, {}, {}, None)
        repn = visitor.walk_expression(cons.body)
        self.assertIsNone(repn.nonlinear)
        if leq:
            self.assertIsNone(cons.lower)
            ub = cons.upper
            return ub >= repn.to_expression(visitor)
        else:
            self.assertIsNone(cons.upper)
            lb = cons.lower
            return lb <= repn.to_expression(visitor)

    def check_hierarchical_nested_model(self, m, bigm):
        outer_xor = m.disjunction_block.disjunction.algebraic_constraint
        ct.check_two_term_disjunction_xor(
            self, outer_xor, m.disj1, m.disjunct_block.disj2
        )

        self.check_inner_xor_constraint(
            m.disjunct_block.disj2.disjunction, m.disjunct_block.disj2, bigm
        )

        # outer disjunction constraints
        disj1c = bigm.get_transformed_constraints(m.disj1.c)
        self.assertEqual(len(disj1c), 1)
        cons = disj1c[0]
        assertExpressionsEqual(
            self,
            cons.expr,
            m.x[1] ** 2
            + m.x[2] ** 2
            + m.x[3] ** 2
            + m.x[4] ** 2
            - 143.0 * (1 - m.disj1.binary_indicator_var)
            <= 1.0,
        )

        disj2c = bigm.get_transformed_constraints(m.disjunct_block.disj2.c)
        self.assertEqual(len(disj2c), 1)
        cons = disj2c[0]
        assertExpressionsEqual(
            self,
            cons.expr,
            (3 - m.x[1]) ** 2
            + (3 - m.x[2]) ** 2
            + (3 - m.x[3]) ** 2
            + (3 - m.x[4]) ** 2
            - 99.0 * (1 - m.disjunct_block.disj2.binary_indicator_var)
            <= 1.0,
        )

        # inner disjunction constraints
        innerd1c = bigm.get_transformed_constraints(
            m.disjunct_block.disj2.disjunction_disjuncts[0].constraint[1]
        )
        self.assertEqual(len(innerd1c), 1)
        cons = innerd1c[0]
        assertExpressionsEqual(
            self,
            cons.expr,
            m.x[1] ** 2
            + m.x[2] ** 2
            + m.x[3] ** 2
            + m.x[4] ** 2
            - 143.0
            * (
                1
                - m.disjunct_block.disj2.disjunction_disjuncts[0].binary_indicator_var
                + 1
                - m.disjunct_block.disj2.binary_indicator_var
            )
            <= 1.0,
        )

        innerd2c = bigm.get_transformed_constraints(
            m.disjunct_block.disj2.disjunction_disjuncts[1].constraint[1]
        )
        self.assertEqual(len(innerd2c), 1)
        cons = innerd2c[0]
        assertExpressionsEqual(
            self,
            cons.expr,
            (3 - m.x[1]) ** 2
            + (3 - m.x[2]) ** 2
            + (3 - m.x[3]) ** 2
            + (3 - m.x[4]) ** 2
            - 99.0
            * (
                1
                - m.disjunct_block.disj2.disjunction_disjuncts[1].binary_indicator_var
                + 1
                - m.disjunct_block.disj2.binary_indicator_var
            )
            <= 1.0,
        )

    def test_hierarchical_badly_ordered_targets(self):
        m = models.makeHierarchicalNested_DeclOrderMatchesInstantiationOrder()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m, targets=[m.disjunction_block, m.disjunct_block.disj2])

        # the real test here is that the above doesn't scream about there being
        # an untransformed Disjunction inside of a Disjunct it's trying to
        # transform. So let's just check that everything is transformed
        self.check_hierarchical_nested_model(m, bigm)

    def test_decl_order_opposite_instantiation_order(self):
        # In this test, we create the same problem as above, but we don't even
        # need targets!
        m = models.makeHierarchicalNested_DeclOrderOppositeInstantiationOrder()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)

        # Like above, the real test is that the above doesn't scream. We can use
        # the same check to make sure everything is transformed correctly.
        self.check_hierarchical_nested_model(m, bigm)

    @unittest.skipUnless(gurobi_available, "Gurobi is not available")
    def test_do_not_assume_nested_indicators_local(self):
        ct.check_do_not_assume_nested_indicators_local(self, 'gdp.bigm')

    @unittest.skipUnless(gurobi_available, "Gurobi is not available")
    def test_constraints_not_enforced_when_an_ancestor_indicator_is_False(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 30))

        m.left = Disjunct()
        m.left.left = Disjunct()
        m.left.left.c = Constraint(expr=m.x >= 10)
        m.left.right = Disjunct()
        m.left.right.c = Constraint(expr=m.x >= 9)
        m.left.disjunction = Disjunction(expr=[m.left.left, m.left.right])
        m.right = Disjunct()
        m.right.left = Disjunct()
        m.right.left.c = Constraint(expr=m.x >= 11)
        m.right.right = Disjunct()
        m.right.right.c = Constraint(expr=m.x >= 8)
        m.right.disjunction = Disjunction(expr=[m.right.left, m.right.right])
        m.disjunction = Disjunction(expr=[m.left, m.right])

        m.equiv_left = LogicalConstraint(
            expr=m.left.left.indicator_var.equivalent_to(m.right.left.indicator_var)
        )
        m.equiv_right = LogicalConstraint(
            expr=m.left.right.indicator_var.equivalent_to(m.right.right.indicator_var)
        )

        m.obj = Objective(expr=m.x)

        TransformationFactory('gdp.bigm').apply_to(m)
        results = SolverFactory('gurobi').solve(m)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertTrue(value(m.right.indicator_var))
        self.assertFalse(value(m.left.indicator_var))
        self.assertTrue(value(m.right.right.indicator_var))
        self.assertFalse(value(m.right.left.indicator_var))
        self.assertTrue(value(m.left.right.indicator_var))
        self.assertAlmostEqual(value(m.x), 8)


class IndexedDisjunction(unittest.TestCase):
    # this tests that if the targets are a subset of the
    # DisjunctDatas in an IndexedDisjunction that the xor constraint
    # created on the parent block will still be indexed as expected.
    def test_xor_constraint(self):
        ct.check_indexed_xor_constraints_with_targets(self, 'bigm')

    def test_partial_deactivate_indexed_disjunction(self):
        ct.check_partial_deactivate_indexed_disjunction(self, 'bigm')


class BlocksOnDisjuncts(unittest.TestCase):
    # ESJ: All of these tests are specific to bigm because they check how much
    # stuff is on the transformation blocks.
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_transformed_constraint_nameConflicts(self):
        m = models.makeTwoTermDisj_BlockOnDisj()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)

        transBlock = m._pyomo_gdp_bigm_reformulation
        disjBlock = transBlock.relaxedDisjuncts

        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)
        evil0 = bigm.get_transformed_constraints(m.evil[0].c)
        self.assertEqual(len(evil0), 1)
        self.assertIs(evil0[0].parent_block(), disjBlock[0])

        evil1 = bigm.get_transformed_constraints(m.evil[1].component('b.c'))
        self.assertEqual(len(evil1), 1)
        self.assertIs(evil1[0].parent_block(), disjBlock[1])
        evil1 = bigm.get_transformed_constraints(m.evil[1].b.c)
        self.assertEqual(len(evil1), 2)
        self.assertIs(evil1[0].parent_block(), disjBlock[1])
        self.assertIs(evil1[1].parent_block(), disjBlock[1])
        evil1 = bigm.get_transformed_constraints(m.evil[1].b.anotherblock.c)
        self.assertEqual(len(evil1), 1)
        self.assertIs(evil1[0].parent_block(), disjBlock[1])
        evil1 = bigm.get_transformed_constraints(m.evil[1].bb[1].c)
        self.assertEqual(len(evil1), 2)
        self.assertIs(evil1[0].parent_block(), disjBlock[1])
        self.assertIs(evil1[1].parent_block(), disjBlock[1])

    def check_all_but_evil1_b_anotherblock_constraint_transformed(self, m):
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)

        transBlock = m._pyomo_gdp_bigm_reformulation
        disjBlock = transBlock.relaxedDisjuncts

        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)
        evil0 = bigm.get_transformed_constraints(m.evil[0].c)
        self.assertEqual(len(evil0), 1)
        self.assertIs(evil0[0].parent_block(), disjBlock[0])

        evil1 = bigm.get_transformed_constraints(m.evil[1].component('b.c'))
        self.assertEqual(len(evil1), 1)
        self.assertIs(evil1[0].parent_block(), disjBlock[1])
        evil1 = bigm.get_transformed_constraints(m.evil[1].b.c)
        self.assertEqual(len(evil1), 2)
        self.assertIs(evil1[0].parent_block(), disjBlock[1])
        self.assertIs(evil1[1].parent_block(), disjBlock[1])
        with self.assertRaisesRegex(
            GDP_Error,
            r"Constraint 'evil\[1\].b.anotherblock.c' has not been transformed.",
        ):
            bigm.get_transformed_constraints(m.evil[1].b.anotherblock.c)

        evil1 = bigm.get_transformed_constraints(m.evil[1].bb[1].c)
        self.assertEqual(len(evil1), 2)
        self.assertIs(evil1[0].parent_block(), disjBlock[1])
        self.assertIs(evil1[1].parent_block(), disjBlock[1])

    def test_do_not_transform_deactivated_constraint(self):
        m = models.makeTwoTermDisj_BlockOnDisj()
        m.evil[1].b.anotherblock.c.deactivate()
        self.check_all_but_evil1_b_anotherblock_constraint_transformed(m)

    def test_do_not_transform_deactivated_block(self):
        m = models.makeTwoTermDisj_BlockOnDisj()
        m.evil[1].b.anotherblock.deactivate()

        self.check_all_but_evil1_b_anotherblock_constraint_transformed(m)

    def test_pick_up_bigm_suffix_on_block(self):
        m = models.makeTwoTermDisj_BlockOnDisj()
        m.evil[1].b.BigM = Suffix(direction=Suffix.LOCAL)
        m.evil[1].b.BigM[m.evil[1].b.c] = 2000
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)

        # check that the m value got used
        cons_list = bigm.get_transformed_constraints(m.evil[1].b.c)
        ub = cons_list[1]
        self.assertEqual(ub.upper, 0)
        self.assertIsNone(ub.lower)
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -2000)
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertIs(repn.linear_vars[0], m.x)
        self.assertEqual(repn.linear_coefs[0], 1)
        self.assertIs(repn.linear_vars[1], m.evil[1].binary_indicator_var)
        self.assertEqual(repn.linear_coefs[1], 2000)

    def test_use_correct_none_suffix(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-100, 111))
        m.b = Block()
        m.b.d = Disjunct()
        m.b.d.foo = Block()

        m.b.d.c = Constraint(expr=m.x >= 9)

        m.b.BigM = Suffix()
        m.b.BigM[None] = 10
        m.b.d.foo.BigM = Suffix()
        m.b.d.foo.BigM[None] = 1

        m.d = Disjunct()
        m.disj = Disjunction(expr=[m.d, m.b.d])

        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)

        # we should have picked up 10 for m.b.d.c
        cons_list = bigm.get_transformed_constraints(m.b.d.c)
        lb = cons_list[0]
        self.assertEqual(lb.lower, 9)
        self.assertIsNone(lb.upper)
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 10)
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertIs(repn.linear_vars[0], m.x)
        self.assertEqual(repn.linear_coefs[0], 1)
        self.assertIs(repn.linear_vars[1], m.b.d.binary_indicator_var)
        self.assertEqual(repn.linear_coefs[1], -10)


class UntransformableObjectsOnDisjunct(unittest.TestCase):
    def test_RangeSet(self):
        ct.check_RangeSet(self, 'bigm')

    def test_Expression(self):
        ct.check_Expression(self, 'bigm')


class TransformABlock(unittest.TestCase):
    def test_transformation_simple_block(self):
        ct.check_transformation_simple_block(self, 'bigm')

    def test_transform_block_data(self):
        ct.check_transform_block_data(self, 'bigm')

    def test_simple_block_target(self):
        ct.check_simple_block_target(self, 'bigm')

    def test_block_data_target(self):
        ct.check_block_data_target(self, 'bigm')

    def test_indexed_block_target(self):
        ct.check_indexed_block_target(self, 'bigm')


class IndexedDisjunctions(unittest.TestCase):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_disjunction_data_target(self):
        ct.check_disjunction_data_target(self, 'bigm')

    def test_disjunction_data_target_any_index(self):
        ct.check_disjunction_data_target_any_index(self, 'bigm')

    # ESJ: This and the following tests are *very* similar to those in hull,
    # but I actually bothered to check the additional transformed objects in
    # hull (disaggregated variables, bounds constraints...), so they are
    # reproduced independently there.
    def check_trans_block_disjunctions_of_disjunct_datas(self, m):
        transBlock1 = m.component("_pyomo_gdp_bigm_reformulation")
        self.assertIsInstance(transBlock1, Block)
        self.assertIsInstance(transBlock1.component("relaxedDisjuncts"), Block)
        # We end up with a transformation block for every ScalarDisjunction or
        # IndexedDisjunction.
        bigm = TransformationFactory('gdp.bigm')
        self.assertEqual(len(transBlock1.relaxedDisjuncts), 4)
        firstTerm1 = bigm.get_transformed_constraints(m.firstTerm[1].cons)
        self.assertIs(firstTerm1[0].parent_block(), transBlock1.relaxedDisjuncts[0])
        self.assertEqual(len(firstTerm1), 2)
        secondTerm1 = bigm.get_transformed_constraints(m.secondTerm[1].cons)
        self.assertIs(secondTerm1[0].parent_block(), transBlock1.relaxedDisjuncts[1])
        self.assertEqual(len(secondTerm1), 1)
        firstTerm2 = bigm.get_transformed_constraints(m.firstTerm[2].cons)
        self.assertIs(firstTerm2[0].parent_block(), transBlock1.relaxedDisjuncts[2])
        self.assertEqual(len(firstTerm2), 2)
        secondTerm2 = bigm.get_transformed_constraints(m.secondTerm[2].cons)
        self.assertIs(secondTerm2[0].parent_block(), transBlock1.relaxedDisjuncts[3])
        self.assertEqual(len(secondTerm2), 1)

    def test_simple_disjunction_of_disjunct_datas(self):
        ct.check_simple_disjunction_of_disjunct_datas(self, 'bigm')

    def test_any_indexed_disjunction_of_disjunct_datas(self):
        m = models.makeAnyIndexedDisjunctionOfDisjunctDatas()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)

        transBlock = m.component("_pyomo_gdp_bigm_reformulation")
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(transBlock.component("relaxedDisjuncts"), Block)
        self.assertEqual(len(transBlock.relaxedDisjuncts), 4)
        firstTerm1 = bigm.get_transformed_constraints(m.firstTerm[1].cons)
        self.assertIs(firstTerm1[0].parent_block(), transBlock.relaxedDisjuncts[0])
        self.assertEqual(len(firstTerm1), 2)
        secondTerm1 = bigm.get_transformed_constraints(m.secondTerm[1].cons)
        self.assertIs(secondTerm1[0].parent_block(), transBlock.relaxedDisjuncts[1])
        self.assertEqual(len(secondTerm1), 1)
        firstTerm2 = bigm.get_transformed_constraints(m.firstTerm[2].cons)
        self.assertIs(firstTerm2[0].parent_block(), transBlock.relaxedDisjuncts[2])
        self.assertEqual(len(firstTerm1), 2)
        secondTerm2 = bigm.get_transformed_constraints(m.secondTerm[2].cons)
        self.assertIs(secondTerm2[0].parent_block(), transBlock.relaxedDisjuncts[3])
        self.assertEqual(len(secondTerm1), 1)

        self.assertIsInstance(
            m.disjunction[1].algebraic_constraint.parent_component(), Constraint
        )
        self.assertIsInstance(
            m.disjunction[2].algebraic_constraint.parent_component(), Constraint
        )

    def check_first_iteration(self, model):
        transBlock = model.component("_pyomo_gdp_bigm_reformulation")
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(transBlock.component("disjunctionList_xor"), Constraint)
        self.assertEqual(len(transBlock.disjunctionList_xor), 1)
        self.assertFalse(model.disjunctionList[0].active)

    def check_second_iteration(self, model):
        transBlock = model.component("_pyomo_gdp_bigm_reformulation_4")
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(transBlock.component("relaxedDisjuncts"), Block)
        self.assertEqual(len(transBlock.relaxedDisjuncts), 2)
        bigm = TransformationFactory('gdp.bigm')

        if model.component('firstTerm') is None:
            firstTerm1 = model.component('firstTerm[1]')
            secondTerm1 = model.component('secondTerm[1]')
        else:
            firstTerm1 = model.firstTerm[1]
            secondTerm1 = model.secondTerm[1]

        firstTerm = bigm.get_transformed_constraints(firstTerm1.cons)
        self.assertIs(firstTerm[0].parent_block(), transBlock.relaxedDisjuncts[0])
        self.assertEqual(len(firstTerm), 2)

        secondTerm = bigm.get_transformed_constraints(secondTerm1.cons)
        self.assertIs(secondTerm[0].parent_block(), transBlock.relaxedDisjuncts[1])
        self.assertEqual(len(secondTerm), 1)

        self.assertIsInstance(
            model.disjunctionList[1].algebraic_constraint.parent_component(), Constraint
        )
        self.assertIsInstance(
            model.disjunctionList[0].algebraic_constraint.parent_component(), Constraint
        )
        self.assertFalse(model.disjunctionList[1].active)
        self.assertFalse(model.disjunctionList[0].active)

    def test_disjunction_and_disjuncts_indexed_by_any(self):
        ct.check_disjunction_and_disjuncts_indexed_by_any(self, 'bigm')

    def test_iteratively_adding_disjunctions_transform_container(self):
        ct.check_iteratively_adding_disjunctions_transform_container(self, 'bigm')

    def test_iteratively_adding_disjunctions_transform_model(self):
        ct.check_iteratively_adding_disjunctions_transform_model(self, 'bigm')

    def test_iteratively_adding_to_indexed_disjunction_on_block(self):
        ct.check_iteratively_adding_to_indexed_disjunction_on_block(self, 'bigm')


class TestErrors(unittest.TestCase):
    def test_transform_empty_disjunction(self):
        ct.check_transform_empty_disjunction(self, 'bigm')

    def test_deactivated_disjunct_nonzero_indicator_var(self):
        ct.check_deactivated_disjunct_nonzero_indicator_var(self, 'bigm')

    def test_deactivated_disjunct_unfixed_indicator_var(self):
        ct.check_deactivated_disjunct_unfixed_indicator_var(self, 'bigm')

    def test_infeasible_xor_because_all_disjuncts_deactivated(self):
        m = ct.setup_infeasible_xor_because_all_disjuncts_deactivated(self, 'bigm')
        bigm = TransformationFactory('gdp.bigm')

        transBlock = m.component("_pyomo_gdp_bigm_reformulation")
        self.assertIsInstance(transBlock, Block)
        self.assertEqual(len(transBlock.relaxedDisjuncts), 2)
        self.assertIsInstance(transBlock.component("disjunction_xor"), Constraint)
        disjunct1 = transBlock.relaxedDisjuncts[0]
        relaxed_xor = bigm.get_transformed_constraints(
            m.disjunction_disjuncts[0].nestedDisjunction.algebraic_constraint
        )
        # It was an equality
        self.assertEqual(len(relaxed_xor), 2)
        self.assertIsInstance(relaxed_xor[0].parent_component(), Constraint)
        relaxed_xor_lb = relaxed_xor[0]
        relaxed_xor_ub = relaxed_xor[1]
        repn = generate_standard_repn(relaxed_xor_lb.body)
        self.assertEqual(relaxed_xor_lb.lower, 1)
        self.assertIsNone(relaxed_xor_lb.upper)
        # the other variables got eaten in the constant because they are fixed.
        self.assertEqual(len(repn.linear_vars), 1)
        ct.check_linear_coef(self, repn, m.disjunction.disjuncts[0].indicator_var, -1)
        self.assertEqual(repn.constant, 1)
        repn = generate_standard_repn(relaxed_xor_ub.body)
        self.assertIsNone(relaxed_xor_ub.lower)
        self.assertEqual(value(relaxed_xor_ub.upper), 1)
        self.assertEqual(len(repn.linear_vars), 1)
        ct.check_linear_coef(self, repn, m.disjunction.disjuncts[0].indicator_var, 1)

        # and last check that the other constraints here look fine
        x0 = bigm.get_transformed_constraints(m.disjunction_disjuncts[0].constraint[1])
        self.assertEqual(len(x0), 2)
        lb = x0[0]
        ub = x0[1]
        self.assertIsInstance(lb.parent_component(), Constraint)
        # lb = x0[(1, 'lb')]
        self.assertEqual(value(lb.lower), 0)
        self.assertIsNone(lb.upper)
        repn = generate_standard_repn(lb.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        ct.check_linear_coef(self, repn, m.x, 1)

        self.assertIsInstance(ub.parent_component(), Constraint)
        self.assertIsNone(ub.lower)
        self.assertEqual(value(ub.upper), 0)
        repn = generate_standard_repn(ub.body)
        self.assertEqual(repn.constant, -8)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, m.x, 1)
        ct.check_linear_coef(self, repn, m.disjunction_disjuncts[0].indicator_var, 8)

    def test_retrieving_nondisjunctive_components(self):
        ct.check_retrieving_nondisjunctive_components(self, 'bigm')

    def test_ask_for_transformed_constraint_from_untransformed_disjunct(self):
        ct.check_ask_for_transformed_constraint_from_untransformed_disjunct(
            self, 'bigm'
        )

    def test_silly_target(self):
        ct.check_silly_target(self, 'bigm')

    def test_untransformed_arcs(self):
        ct.check_untransformed_network_raises_GDPError(self, 'bigm')


class EstimatingMwithFixedVars(unittest.TestCase):
    def test_tighter_Ms_when_vars_fixed_forever(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.y = Var(bounds=(0, 70))
        m.d = Disjunct()
        m.d.c = Constraint(expr=m.x + m.y <= 13)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x >= 7)
        m.disj = Disjunction(expr=[m.d, m.d2])
        m.y.fix(10)
        bigm = TransformationFactory('gdp.bigm')
        promise = bigm.create_using(m, assume_fixed_vars_permanent=True)
        bigm.apply_to(m, assume_fixed_vars_permanent=False)

        # check the M values in both cases
        # first where y might be unfixed:
        xformed = bigm.get_transformed_constraints(m.d.c)
        self.assertEqual(len(xformed), 1)
        cons = xformed[0]
        self.assertEqual(cons.upper, 13)
        self.assertIsNone(cons.lower)
        repn = generate_standard_repn(cons.body)
        self.assertEqual(repn.constant, -57)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, m.x, 1)
        ct.check_linear_coef(self, repn, m.d.indicator_var, 67)

        # then where it won't
        xformed = bigm.get_transformed_constraints(promise.d.c)
        self.assertEqual(len(xformed), 1)
        cons = xformed[0]
        self.assertEqual(cons.upper, 13)
        self.assertIsNone(cons.lower)
        repn = generate_standard_repn(cons.body)
        self.assertEqual(repn.constant, 3)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, promise.x, 1)
        ct.check_linear_coef(self, repn, promise.d.indicator_var, 7)


class TrivialDisjuncts(unittest.TestCase):
    @unittest.skipIf(not ct.linear_solvers, "No linear solver available")
    def test_trivial_disjuncts_linear(self):
        ct.check_trivial_constraints(self, ct.linear_solvers[0], transformation='bigm')

    @unittest.skipIf(not ct.nonlinear_solvers, "No linear solver available")
    def test_trivial_disjuncts_nonlinear(self):
        ct.check_trivial_constraints(
            self, ct.nonlinear_solvers[0], transformation='bigm'
        )


class NetworkDisjuncts(unittest.TestCase, CommonTests):
    @unittest.skipIf(not ct.linear_solvers, "No linear solver available")
    def test_solution_maximize(self):
        ct.check_network_disjuncts(self, minimize=False, transformation='bigm')

    @unittest.skipIf(not ct.linear_solvers, "No linear solver available")
    def test_solution_minimize(self):
        ct.check_network_disjuncts(self, minimize=True, transformation='bigm')


class LogicalConstraintsOnDisjuncts(unittest.TestCase):
    def test_logical_constraints_transformed(self):
        m = models.makeLogicalConstraintsOnDisjuncts()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)

        y1 = m.Y[1].get_associated_binary()
        y2 = m.Y[2].get_associated_binary()

        # check the bigm transformation of the logical things on the disjuncts

        # first d[1]:
        cons = bigm.get_transformed_constraints(
            m.d[1]._logical_to_disjunctive.transformed_constraints[1]
        )
        # big-M transformation of z = 1 - y1:
        #     z <= 1 - y1 + (1 - d[1].indicator_var)
        #     z >= 1 - y1 - (1 - d[1].indicator_var)
        z = m.d[1]._logical_to_disjunctive.auxiliary_vars[1]
        self.assertEqual(len(cons), 2)
        leq = cons[0]
        self.assertEqual(leq.lower, 0)
        self.assertIsNone(leq.upper)
        repn = generate_standard_repn(leq.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self, simplified, z + y1 - m.d[1].binary_indicator_var
        )
        geq = cons[1]
        self.assertEqual(geq.upper, 0)
        self.assertIsNone(geq.lower)
        repn = generate_standard_repn(geq.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self, simplified, z + y1 + m.d[1].binary_indicator_var - 2
        )

        # then d[4]:
        z1 = m.d[4]._logical_to_disjunctive.auxiliary_vars[1]
        z2 = m.d[4]._logical_to_disjunctive.auxiliary_vars[2]
        z3 = m.d[4]._logical_to_disjunctive.auxiliary_vars[3]  # fixed True
        cons = bigm.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[1]
        )
        self.assertEqual(len(cons), 1)
        c = cons[0]
        # (1 - z1) + (1 - y1) + y2 >= 1 - (1 - d4.ind_var)
        self.assertIsNone(c.upper)
        self.assertEqual(c.lower, 1)
        repn = generate_standard_repn(c.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self, simplified, -z1 - y1 + y2 - m.d[4].binary_indicator_var + 3
        )
        cons = bigm.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[2]
        )
        self.assertEqual(len(cons), 1)
        c = cons[0]
        # z1 + 1 - (1 - y1) >= 1 - (1 - d4.ind_var)
        self.assertIsNone(c.upper)
        self.assertEqual(c.lower, 1)
        repn = generate_standard_repn(c.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self, simplified, y1 + z1 - m.d[4].binary_indicator_var + 1
        )
        cons = bigm.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[3]
        )
        self.assertEqual(len(cons), 1)
        c = cons[0]
        # z1 + (1 - y2) >= 1 - (1 - d4.ind_var)
        self.assertIsNone(c.upper)
        self.assertEqual(c.lower, 1)
        repn = generate_standard_repn(c.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self, simplified, -y2 + z1 - m.d[4].binary_indicator_var + 2
        )
        cons = bigm.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[4]
        )
        self.assertEqual(len(cons), 1)
        c = cons[0]
        # (1 - z2) + y1 + (1 - y2) >= 1 - (1 - d4.ind_var)
        self.assertIsNone(c.upper)
        self.assertEqual(c.lower, 1)
        repn = generate_standard_repn(c.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self, simplified, -z2 - y2 + y1 - m.d[4].binary_indicator_var + 3
        )
        cons = bigm.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[5]
        )
        self.assertEqual(len(cons), 1)
        c = cons[0]
        # z2 + (1 - y1) >= 1 - (1 - d4.ind_var)
        self.assertIsNone(c.upper)
        self.assertEqual(c.lower, 1)
        repn = generate_standard_repn(c.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self, simplified, -y1 + z2 - m.d[4].binary_indicator_var + 2
        )
        cons = bigm.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[6]
        )
        self.assertEqual(len(cons), 1)
        c = cons[0]
        # z2 + 1 - (1 - y2) >= 1 - (1 - d4.ind_var)
        self.assertIsNone(c.upper)
        self.assertEqual(c.lower, 1)
        repn = generate_standard_repn(c.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self, simplified, y2 + z2 - m.d[4].binary_indicator_var + 1
        )
        cons = bigm.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[7]
        )
        self.assertEqual(len(cons), 1)
        c = cons[0]
        # z3 <= z1 + (1 - d4.ind_var)
        self.assertIsNone(c.lower)
        self.assertEqual(c.upper, 0)
        repn = generate_standard_repn(c.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self, simplified, z3 - z1 + m.d[4].binary_indicator_var - 1
        )
        cons = bigm.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[8]
        )
        self.assertEqual(len(cons), 1)
        c = cons[0]
        # z3 <= z2 + (1 - d4.ind_var)
        self.assertIsNone(c.lower)
        self.assertEqual(c.upper, 0)
        repn = generate_standard_repn(c.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self, simplified, z3 - z2 + m.d[4].binary_indicator_var - 1
        )

        # check that the global logical constraints were also transformed.
        self.assertFalse(m.p.active)

    @unittest.skipIf(not ct.linear_solvers, "No linear solver available")
    def test_solution_obeys_logical_constraints(self):
        m = models.makeLogicalConstraintsOnDisjuncts()
        ct.check_solution_obeys_logical_constraints(self, 'bigm', m)

    @unittest.skipIf(not ct.linear_solvers, "No linear solver available")
    def test_boolean_vars_on_disjunct(self):
        # Just to make sure we do everything in the correct order, make sure
        # that we can solve a model where some BooleanVars were declared on one
        # of the Disjuncts
        m = models.makeBooleanVarsOnDisjuncts()
        ct.check_solution_obeys_logical_constraints(self, 'bigm', m)

    def test_pickle(self):
        ct.check_transformed_model_pickles(self, 'bigm')

    @unittest.skipIf(not dill_available, "Dill is not available")
    def test_dill_pickle(self):
        ct.check_transformed_model_pickles_with_dill(self, 'bigm')


@unittest.skipUnless(gurobi_available, "Gurobi is not available")
class NestedDisjunctsInFlatGDP(unittest.TestCase):
    """
    This class tests the fix for #2702
    """

    def test_declare_disjuncts_in_disjunction_rule(self):
        ct.check_nested_disjuncts_in_flat_gdp(self, 'bigm')


if __name__ == '__main__':
    unittest.main()
