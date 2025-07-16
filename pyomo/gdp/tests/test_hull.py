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

import logging
import sys
import random
from io import StringIO

import pyomo.common.unittest as unittest

from pyomo.common.dependencies import dill_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import this_file_dir

from pyomo.environ import (
    TransformationFactory,
    Block,
    Set,
    Constraint,
    Var,
    RealSet,
    ComponentMap,
    value,
    log,
    ConcreteModel,
    Any,
    Suffix,
    SolverFactory,
    RangeSet,
    Param,
    Objective,
    TerminationCondition,
)
from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor

from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct


currdir = this_file_dir()

EPS = TransformationFactory('gdp.hull').CONFIG.EPS
linear_solvers = ct.linear_solvers

gurobi_available = (
    SolverFactory('gurobi').available(exception_flag=False)
    and SolverFactory('gurobi').license_is_valid()
)


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

        transBlock = m._pyomo_gdp_hull_reformulation
        disjBlock = transBlock.relaxedDisjuncts
        # same on both disjuncts
        for i in [0, 1]:
            relaxationBlock = disjBlock[i]
            x = relaxationBlock.disaggregatedVars.x
            if i == 1:  # this disjunct as x, w, and no y
                w = relaxationBlock.disaggregatedVars.w
                y = transBlock._disaggregatedVars[0]
            elif i == 0:  # this disjunct as x, y, and no w
                y = relaxationBlock.disaggregatedVars.y
                w = transBlock._disaggregatedVars[1]
            # variables created (w and y can be Vars or VarDatas depending on
            # the disjunct)
            self.assertIs(w.ctype, Var)
            self.assertIsInstance(x, Var)
            self.assertIs(y.ctype, Var)
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
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        disjBlock = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts

        # the only constraint on the first block is the non-linear one
        disj1c = hull.get_transformed_constraints(m.d[0].c)
        # we only have an upper bound
        self.assertEqual(len(disj1c), 1)
        cons = disj1c[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertFalse(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 1)
        # This is a weak test, but as good as any to ensure that the
        # substitution was done correctly
        EPS_1 = 1 - EPS
        _disj = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts[0]
        assertExpressionsEqual(
            self,
            cons.body,
            EXPR.SumExpression(
                [
                    EXPR.ProductExpression(
                        (
                            EXPR.LinearExpression(
                                [
                                    EXPR.MonomialTermExpression(
                                        (EPS_1, m.d[0].binary_indicator_var)
                                    ),
                                    EPS,
                                ]
                            ),
                            EXPR.SumExpression(
                                [
                                    EXPR.DivisionExpression(
                                        (
                                            _disj.disaggregatedVars.x,
                                            EXPR.LinearExpression(
                                                [
                                                    EXPR.MonomialTermExpression(
                                                        (
                                                            EPS_1,
                                                            m.d[0].binary_indicator_var,
                                                        )
                                                    ),
                                                    EPS,
                                                ]
                                            ),
                                        )
                                    ),
                                    EXPR.PowExpression(
                                        (
                                            EXPR.DivisionExpression(
                                                (
                                                    _disj.disaggregatedVars.y,
                                                    EXPR.LinearExpression(
                                                        [
                                                            EXPR.MonomialTermExpression(
                                                                (
                                                                    EPS_1,
                                                                    m.d[
                                                                        0
                                                                    ].binary_indicator_var,
                                                                )
                                                            ),
                                                            EPS,
                                                        ]
                                                    ),
                                                )
                                            ),
                                            2,
                                        )
                                    ),
                                ]
                            ),
                        )
                    ),
                    EXPR.NegationExpression(
                        (
                            EXPR.ProductExpression(
                                (
                                    0.0,
                                    EXPR.LinearExpression(
                                        [
                                            1,
                                            EXPR.MonomialTermExpression(
                                                (-1, m.d[0].binary_indicator_var)
                                            ),
                                        ]
                                    ),
                                )
                            ),
                        )
                    ),
                    EXPR.MonomialTermExpression((-14.0, m.d[0].binary_indicator_var)),
                ]
            ),
        )

    def test_transformed_constraints_linear(self):
        m = models.makeTwoTermDisj_Nonlinear()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        disjBlock = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts

        # the only constraint on the first block is the non-linear one
        c1 = hull.get_transformed_constraints(m.d[1].c1)
        self.assertEqual(len(c1), 1)
        cons = c1[0]
        # has only lb
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

        c2 = hull.get_transformed_constraints(m.d[1].c2)
        self.assertEqual(len(c2), 1)
        cons = c2[0]
        # 'eq' is preserved
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

        c3 = hull.get_transformed_constraints(m.d[1].c3)
        # bounded inequality is split
        self.assertEqual(len(c3), 2)
        # lb
        cons = c3[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, disjBlock[1].disaggregatedVars.x, -1)
        ct.check_linear_coef(self, repn, m.d[1].indicator_var, 1)
        self.assertEqual(repn.constant, 0)

        # ub
        cons = c3[1]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, disjBlock[1].disaggregatedVars.x, 1)
        ct.check_linear_coef(self, repn, m.d[1].indicator_var, -3)
        self.assertEqual(repn.constant, 0)

    def check_bound_constraints_on_disjBlock(self, cons, disvar, indvar, lb, ub):
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

    def check_bound_constraints_on_disjunctionBlock(
        self, varlb, varub, disvar, indvar, lb, ub
    ):
        self.assertIsNone(varlb.lower)
        self.assertEqual(varlb.upper, 0)
        repn = generate_standard_repn(varlb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, lb)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, indvar, -lb)
        ct.check_linear_coef(self, repn, disvar, -1)

        self.assertIsNone(varub.lower)
        self.assertEqual(varub.upper, 0)
        repn = generate_standard_repn(varub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -ub)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, indvar, ub)
        ct.check_linear_coef(self, repn, disvar, 1)

    def test_disaggregatedVar_bounds(self):
        m = models.makeTwoTermDisj_Nonlinear()
        TransformationFactory('gdp.hull').apply_to(m)

        transBlock = m._pyomo_gdp_hull_reformulation
        disjBlock = transBlock.relaxedDisjuncts
        for i in [0, 1]:
            # check bounds constraints for each variable on each of the two
            # disjuncts.
            self.check_bound_constraints_on_disjBlock(
                disjBlock[i].x_bounds,
                disjBlock[i].disaggregatedVars.x,
                m.d[i].indicator_var,
                1,
                8,
            )
            if i == 1:  # this disjunct has x, w, and no y
                self.check_bound_constraints_on_disjBlock(
                    disjBlock[i].w_bounds,
                    disjBlock[i].disaggregatedVars.w,
                    m.d[i].indicator_var,
                    2,
                    7,
                )
                self.check_bound_constraints_on_disjunctionBlock(
                    transBlock._boundsConstraints[0, 'lb'],
                    transBlock._boundsConstraints[0, 'ub'],
                    transBlock._disaggregatedVars[0],
                    m.d[0].indicator_var,
                    -10,
                    -3,
                )
            elif i == 0:  # this disjunct has x, y, and no w
                self.check_bound_constraints_on_disjBlock(
                    disjBlock[i].y_bounds,
                    disjBlock[i].disaggregatedVars.y,
                    m.d[i].indicator_var,
                    -10,
                    -3,
                )
                self.check_bound_constraints_on_disjunctionBlock(
                    transBlock._boundsConstraints[1, 'lb'],
                    transBlock._boundsConstraints[1, 'ub'],
                    transBlock._disaggregatedVars[1],
                    m.d[1].indicator_var,
                    2,
                    7,
                )

    def test_error_for_or(self):
        m = models.makeTwoTermDisj_Nonlinear()
        m.disjunction.xor = False

        self.assertRaisesRegex(
            GDP_Error,
            "Cannot do hull reformulation for Disjunction "
            "'disjunction' with OR constraint. Must be an XOR!*",
            TransformationFactory('gdp.hull').apply_to,
            m,
        )

    def check_disaggregation_constraint(self, cons, var, disvar1, disvar2):
        assertExpressionsEqual(self, cons.expr, var == disvar1 + disvar2)

    def test_disaggregation_constraint(self):
        m = models.makeTwoTermDisj_Nonlinear()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        transBlock = m._pyomo_gdp_hull_reformulation
        disjBlock = transBlock.relaxedDisjuncts

        self.check_disaggregation_constraint(
            hull.get_disaggregation_constraint(m.w, m.disjunction),
            m.w,
            transBlock._disaggregatedVars[1],
            disjBlock[1].disaggregatedVars.w,
        )
        self.check_disaggregation_constraint(
            hull.get_disaggregation_constraint(m.x, m.disjunction),
            m.x,
            disjBlock[0].disaggregatedVars.x,
            disjBlock[1].disaggregatedVars.x,
        )
        self.check_disaggregation_constraint(
            hull.get_disaggregation_constraint(m.y, m.disjunction),
            m.y,
            transBlock._disaggregatedVars[0],
            disjBlock[0].disaggregatedVars.y,
        )

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
        cons = hull.get_transformed_constraints(orig1)
        self.assertEqual(len(cons), 1)
        trans1 = cons[0]
        self.assertIs(trans1.parent_block(), disjBlock[0])
        self.assertIs(hull.get_src_constraint(trans1), orig1)

        # second disjunct

        # first constraint
        orig1 = m.d[1].c1
        cons = hull.get_transformed_constraints(orig1)
        self.assertEqual(len(cons), 1)
        trans1 = cons[0]
        self.assertIs(trans1.parent_block(), disjBlock[1])
        self.assertIs(hull.get_src_constraint(trans1), orig1)

        # second constraint
        orig2 = m.d[1].c2
        cons = hull.get_transformed_constraints(orig2)
        self.assertEqual(len(cons), 1)
        trans2 = cons[0]
        self.assertIs(trans1.parent_block(), disjBlock[1])
        self.assertIs(hull.get_src_constraint(trans2), orig2)

        # third constraint
        orig3 = m.d[1].c3
        cons = hull.get_transformed_constraints(orig3)
        self.assertEqual(len(cons), 2)
        trans3 = cons[0]
        self.assertIs(hull.get_src_constraint(trans3), orig3)
        self.assertIs(trans3.parent_block(), disjBlock[1])
        trans32 = cons[1]
        self.assertIs(hull.get_src_constraint(trans32), orig3)
        self.assertIs(trans32.parent_block(), disjBlock[1])

    def test_disaggregatedVar_mappings(self):
        m = models.makeTwoTermDisj_Nonlinear()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        transBlock = m._pyomo_gdp_hull_reformulation
        disjBlock = transBlock.relaxedDisjuncts

        for i in [0, 1]:
            mappings = ComponentMap()
            mappings[m.x] = disjBlock[i].disaggregatedVars.x
            if i == 1:  # this disjunct has x, w, and no y
                mappings[m.w] = disjBlock[i].disaggregatedVars.w
                mappings[m.y] = transBlock._disaggregatedVars[0]
            elif i == 0:  # this disjunct has x, y, and no w
                mappings[m.y] = disjBlock[i].disaggregatedVars.y
                mappings[m.w] = transBlock._disaggregatedVars[1]

            for orig, disagg in mappings.items():
                self.assertIs(hull.get_src_var(disagg), orig)
                self.assertIs(hull.get_disaggregated_var(orig, m.d[i]), disagg)

    def test_bigMConstraint_mappings(self):
        m = models.makeTwoTermDisj_Nonlinear()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        transBlock = m._pyomo_gdp_hull_reformulation
        disjBlock = transBlock.relaxedDisjuncts

        for i in [0, 1]:
            mappings = ComponentMap()
            mappings[disjBlock[i].disaggregatedVars.x] = disjBlock[i].x_bounds
            if i == 1:  # this disjunct has x, w, and no y
                mappings[disjBlock[i].disaggregatedVars.w] = disjBlock[i].w_bounds
                mappings[transBlock._disaggregatedVars[0]] = {
                    key: val
                    for key, val in transBlock._boundsConstraints.items()
                    if key[0] == 0
                }
            elif i == 0:  # this disjunct has x, y, and no w
                mappings[disjBlock[i].disaggregatedVars.y] = disjBlock[i].y_bounds
                mappings[transBlock._disaggregatedVars[1]] = {
                    key: val
                    for key, val in transBlock._boundsConstraints.items()
                    if key[0] == 1
                }
            for var, cons in mappings.items():
                returned_cons = hull.get_var_bounds_constraint(var)
                # This sometimes refers a reference to the right part of a
                # larger indexed constraint, so the indexed constraints
                # themselves might not be the same object. The ConstraintDatas
                # are though:
                for key, constraintData in cons.items():
                    if type(key) is tuple:
                        key = key[1]
                    self.assertIs(returned_cons[key], constraintData)

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
        y_disagg = m.disj2.transformation_block.disaggregatedVars.component("disj2.y")
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
        self.assertIs(
            disj1y, m.disj1.transformation_block.parent_block()._disaggregatedVars[0]
        )
        self.assertIs(
            disj2y, m.disj2.transformation_block.disaggregatedVars.component("disj2.y")
        )
        self.assertIs(hull.get_src_var(disj1y), m.disj2.y)
        self.assertIs(hull.get_src_var(disj2y), m.disj2.y)

    def test_global_vars_local_to_a_disjunction_disaggregated(self):
        # The point of this is that where a variable is declared has absolutely
        # nothing to do with whether or not it should be disaggregated. With the
        # only exception being that we can tell disaggregated variables and we
        # know they are really and truly local to only one disjunct (EVER, in
        # the whole model) because we declared them.

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
        # disj1 has both x and y
        disj = m.disj1
        transBlock = disj.transformation_block
        varBlock = transBlock.disaggregatedVars
        self.assertEqual(len([v for v in varBlock.component_data_objects(Var)]), 2)
        x = varBlock.component("disj1.x")
        y = varBlock.component("disj1.y")
        self.assertIsInstance(x, Var)
        self.assertIsInstance(y, Var)
        self.assertIs(hull.get_disaggregated_var(m.disj1.x, disj), x)
        self.assertIs(hull.get_src_var(x), m.disj1.x)
        self.assertIs(hull.get_disaggregated_var(m.disj1.y, disj), y)
        self.assertIs(hull.get_src_var(y), m.disj1.y)
        # disj2 and disj4 have just y
        for disj in [m.disj2, m.disj4]:
            transBlock = disj.transformation_block
            varBlock = transBlock.disaggregatedVars
            self.assertEqual(len([v for v in varBlock.component_data_objects(Var)]), 1)
            y = varBlock.component("disj1.y")
            self.assertIsInstance(y, Var)
            self.assertIs(hull.get_disaggregated_var(m.disj1.y, disj), y)
            self.assertIs(hull.get_src_var(y), m.disj1.y)
        # disj3 has just x
        disj = m.disj3
        transBlock = disj.transformation_block
        varBlock = transBlock.disaggregatedVars
        self.assertEqual(len([v for v in varBlock.component_data_objects(Var)]), 1)
        x = varBlock.component("disj1.x")
        self.assertIsInstance(x, Var)
        self.assertIs(hull.get_disaggregated_var(m.disj1.x, disj), x)
        self.assertIs(hull.get_src_var(x), m.disj1.x)

        # there is a spare x on disjunction1's block
        x2 = m.disjunction1.algebraic_constraint.parent_block()._disaggregatedVars[0]
        self.assertIs(hull.get_disaggregated_var(m.disj1.x, m.disj2), x2)
        self.assertIs(hull.get_src_var(x2), m.disj1.x)
        # What really matters is that the above matches this:
        agg_cons = hull.get_disaggregation_constraint(m.disj1.x, m.disjunction1)
        assertExpressionsEqual(
            self,
            agg_cons.expr,
            m.disj1.x == x2 + hull.get_disaggregated_var(m.disj1.x, m.disj1),
        )

        # and both a spare x and y on disjunction2's block
        x2 = m.disjunction2.algebraic_constraint.parent_block()._disaggregatedVars[1]
        y1 = m.disjunction2.algebraic_constraint.parent_block()._disaggregatedVars[2]
        self.assertIs(hull.get_disaggregated_var(m.disj1.x, m.disj4), x2)
        self.assertIs(hull.get_src_var(x2), m.disj1.x)
        self.assertIs(hull.get_disaggregated_var(m.disj1.y, m.disj3), y1)
        self.assertIs(hull.get_src_var(y1), m.disj1.y)
        # and again what really matters is that these align with the
        # disaggregation constraints:
        agg_cons = hull.get_disaggregation_constraint(m.disj1.x, m.disjunction2)
        assertExpressionsEqual(
            self,
            agg_cons.expr,
            m.disj1.x == x2 + hull.get_disaggregated_var(m.disj1.x, m.disj3),
        )
        agg_cons = hull.get_disaggregation_constraint(m.disj1.y, m.disjunction2)
        assertExpressionsEqual(
            self,
            agg_cons.expr,
            m.disj1.y == y1 + hull.get_disaggregated_var(m.disj1.y, m.disj4),
        )

    def check_name_collision_disaggregated_vars(self, m, disj):
        hull = TransformationFactory('gdp.hull')
        transBlock = disj.transformation_block
        varBlock = transBlock.disaggregatedVars
        self.assertEqual(len([v for v in varBlock.component_data_objects(Var)]), 2)
        # ESJ: This is not what I expected. *Can* we still get name collisions,
        # if we're using a fully qualified name here?
        x2 = varBlock.component("'disj1.x'")
        x = varBlock.component("disj1.x")
        x_orig = m.component("disj1.x")
        self.assertIsInstance(x, Var)
        self.assertIsInstance(x2, Var)
        self.assertIs(hull.get_disaggregated_var(m.disj1.x, disj), x)
        self.assertIs(hull.get_src_var(x), m.disj1.x)
        self.assertIs(hull.get_disaggregated_var(x_orig, disj), x2)
        self.assertIs(hull.get_src_var(x2), x_orig)

    def test_disaggregated_var_name_collision(self):
        # same model as the test above, but now I am putting what was disj1.y as
        # m.'disj1.x', just to invite disaster, and adding constraints that
        # involve all the variables so they will all be disaggregated on the
        # Disjunct
        m = ConcreteModel()
        x = Var(bounds=(2, 11))
        m.add_component("disj1.x", x)
        m.disj1 = Disjunct()
        m.disj1.x = Var(bounds=(1, 10))
        m.disj1.cons1 = Constraint(expr=m.disj1.x + x <= 5)
        m.disj2 = Disjunct()
        m.disj2.cons = Constraint(expr=x >= 8)
        m.disj2.cons1 = Constraint(expr=m.disj1.x == 3)
        m.disjunction1 = Disjunction(expr=[m.disj1, m.disj2])

        m.disj3 = Disjunct()
        m.disj3.cons = Constraint(expr=m.disj1.x >= 7)
        m.disj3.cons1 = Constraint(expr=x >= 10)
        m.disj4 = Disjunct()
        m.disj4.cons = Constraint(expr=x == 3)
        m.disj4.cons1 = Constraint(expr=m.disj1.x == 4)
        m.disjunction2 = Disjunction(expr=[m.disj3, m.disj4])

        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        for disj in (m.disj1, m.disj2, m.disj3, m.disj4):
            self.check_name_collision_disaggregated_vars(m, disj)

    def test_do_not_transform_user_deactivated_disjuncts(self):
        ct.check_user_deactivated_disjuncts(self, 'hull')

    def test_improperly_deactivated_disjuncts(self):
        ct.check_improperly_deactivated_disjuncts(self, 'hull')

    def test_do_not_transform_userDeactivated_IndexedDisjunction(self):
        ct.check_do_not_transform_userDeactivated_indexedDisjunction(self, 'hull')

    def test_disjunction_deactivated(self):
        ct.check_disjunction_deactivated(self, 'hull')

    def test_disjunctDatas_deactivated(self):
        ct.check_disjunctDatas_deactivated(self, 'hull')

    def test_deactivated_constraints(self):
        ct.check_deactivated_constraints(self, 'hull')

    def check_no_double_transformation(self):
        ct.check_do_not_transform_twice_if_disjunction_reactivated(self, 'hull')

    def test_indicator_vars(self):
        ct.check_indicator_vars(self, 'hull')

    def test_xor_constraints(self):
        ct.check_xor_constraint(self, 'hull')

    def test_unbounded_var_error(self):
        m = models.makeTwoTermDisj_Nonlinear()
        # no bounds
        m.w.setlb(None)
        m.w.setub(None)
        self.assertRaisesRegex(
            GDP_Error,
            "Variables that appear in disjuncts must be "
            "bounded in order to use the hull "
            "transformation! Missing bound for w.*",
            TransformationFactory('gdp.hull').apply_to,
            m,
        )

    def check_threeTermDisj_IndexedConstraints(self, m, lb):
        transBlock = m._pyomo_gdp_hull_reformulation
        hull = TransformationFactory('gdp.hull')

        # 2 blocks: the original Disjunct and the transformation block
        self.assertEqual(len(list(m.component_objects(Block, descend_into=False))), 1)
        self.assertEqual(len(list(m.component_objects(Disjunct))), 1)

        # Each relaxed disjunct should have i disaggregated vars and i "d[i].c"
        # Constraints
        for i in [1, 2, 3]:
            relaxed = transBlock.relaxedDisjuncts[i - 1]
            self.assertEqual(
                len(list(relaxed.disaggregatedVars.component_objects(Var))), i
            )
            self.assertEqual(
                len(list(relaxed.disaggregatedVars.component_data_objects(Var))), i
            )
            # we always have the x[1] bounds constraint, then however many
            # original constraints were on the Disjunct
            self.assertEqual(len(list(relaxed.component_objects(Constraint))), 1 + i)
            if lb == 0:
                # i bounds constraints and i transformed constraints
                self.assertEqual(
                    len(list(relaxed.component_data_objects(Constraint))), i + i
                )
            else:
                # 2*i bounds constraints and i transformed constraints
                self.assertEqual(
                    len(list(relaxed.component_data_objects(Constraint))), 2 * i + i
                )

            # Check that there are i transformed constraints on relaxed:
            for j in range(1, i + 1):
                cons = hull.get_transformed_constraints(m.d[i].c[j])
                self.assertEqual(len(cons), 1)
                self.assertIs(cons[0].parent_block(), relaxed)

        # the remaining disaggregated variables are on the disjunction
        # transformation block
        self.assertEqual(
            len(list(transBlock.component_objects(Var, descend_into=False))), 1
        )
        self.assertEqual(
            len(list(transBlock.component_data_objects(Var, descend_into=False))), 2
        )
        # as are the XOR, reaggregation and their bounds constraints
        self.assertEqual(
            len(list(transBlock.component_objects(Constraint, descend_into=False))), 3
        )

        if lb == 0:
            # 3 reaggregation + 2 bounds + 1 xor (because one bounds constraint
            # is on the parent transformation block, and we don't need lb
            # constraints if lb = 0)
            self.assertEqual(
                len(
                    list(
                        transBlock.component_data_objects(
                            Constraint, descend_into=False
                        )
                    )
                ),
                6,
            )
        else:
            # 3 reaggregation + 4 bounds + 1 xor
            self.assertEqual(
                len(
                    list(
                        transBlock.component_data_objects(
                            Constraint, descend_into=False
                        )
                    )
                ),
                8,
            )

    def test_indexed_constraints_in_disjunct(self):
        m = models.makeThreeTermDisj_IndexedConstraints()

        TransformationFactory('gdp.hull').apply_to(m)

        self.check_threeTermDisj_IndexedConstraints(m, lb=0)

    def test_virtual_indexed_constraints_in_disjunct(self):
        m = ConcreteModel()
        m.I = [1, 2, 3]
        m.x = Var(m.I, bounds=(-1, 10))

        def d_rule(d, j):
            m = d.model()
            d.c = Constraint(Any)
            for k in range(j):
                d.c[k + 1] = m.x[k + 1] >= k + 1

        m.d = Disjunct(m.I, rule=d_rule)
        m.disjunction = Disjunction(expr=[m.d[i] for i in m.I])

        TransformationFactory('gdp.hull').apply_to(m)

        self.check_threeTermDisj_IndexedConstraints(m, lb=-1)

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
        with self.assertRaisesRegex(
            GDP_Error, r"Constraint 'b.simpledisj1.c\[1\]' has not been transformed."
        ):
            hull.get_transformed_constraints(m.b.simpledisj1.c[1])

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
        self.assertIs(
            transformed[0].parent_block(), m.b.simpledisj2.transformation_block
        )

        transformed = hull.get_transformed_constraints(m.b.simpledisj2.c[2])
        # simpledisj2.c[2] is a <= constraint
        self.assertEqual(len(transformed), 1)
        self.assertIs(
            transformed[0].parent_block(), m.b.simpledisj2.transformation_block
        )


class MultiTermDisj(unittest.TestCase, CommonTests):
    def test_xor_constraint(self):
        ct.check_three_term_xor_constraint(self, 'hull')

    def test_create_using(self):
        m = models.makeThreeTermIndexedDisj()
        self.diff_apply_to_and_create_using(m)

    def test_do_not_disaggregate_more_than_necessary(self):
        m = models.makeThreeTermDisjunctionWithOneVarInOneDisjunct()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        # check that there are only two disaggregated copies of x
        x1 = hull.get_disaggregated_var(m.x, m.d1)
        self.assertEqual(x1.lb, -2)
        self.assertEqual(x1.ub, 8)
        self.assertIs(hull.get_src_var(x1), m.x)

        x2 = m.disjunction.algebraic_constraint.parent_block()._disaggregatedVars[0]
        self.assertIs(hull.get_src_var(x2), m.x)
        self.assertIs(hull.get_disaggregated_var(m.x, m.d2), x2)
        self.assertIs(hull.get_disaggregated_var(m.x, m.d3), x2)

        # check the bounds constraints for the second copy of x
        bounds = hull.get_var_bounds_constraint(x2)
        self.assertEqual(len(bounds), 2)
        # -2(1 - d1.indicator_var) <= x2
        self.assertIsNone(bounds['lb'].lower)
        self.assertEqual(bounds['lb'].upper, 0)
        repn = generate_standard_repn(bounds['lb'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertIs(repn.linear_vars[1], x2)
        self.assertIs(repn.linear_vars[0], m.d1.indicator_var.get_associated_binary())
        self.assertEqual(repn.linear_coefs[0], 2)
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertEqual(repn.constant, -2)
        # x2 <= 8(1 - d1.indicator_var)
        self.assertIsNone(bounds['ub'].lower)
        self.assertEqual(bounds['ub'].upper, 0)
        repn = generate_standard_repn(bounds['ub'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertIs(repn.linear_vars[0], x2)
        self.assertIs(repn.linear_vars[1], m.d1.indicator_var.get_associated_binary())
        self.assertEqual(repn.linear_coefs[1], 8)
        self.assertEqual(repn.linear_coefs[0], 1)
        self.assertEqual(repn.constant, -8)

        # check the disaggregation constraint
        c = hull.get_disaggregation_constraint(m.x, m.disjunction)
        self.assertEqual(c.lower, 0)
        self.assertEqual(c.upper, 0)
        repn = generate_standard_repn(c.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 3)
        self.assertIs(repn.linear_vars[0], m.x)
        self.assertIs(repn.linear_vars[1], x2)
        self.assertIs(repn.linear_vars[2], x1)
        self.assertEqual(repn.linear_coefs[0], 1)
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertEqual(repn.linear_coefs[2], -1)
        self.assertEqual(repn.constant, 0)


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
            1: [
                hull.get_disaggregated_var(m.x[1], m.disjunct[1, 'a']),
                hull.get_disaggregated_var(m.x[1], m.disjunct[1, 'b']),
            ],
            2: [
                hull.get_disaggregated_var(m.x[2], m.disjunct[2, 'a']),
                hull.get_disaggregated_var(m.x[2], m.disjunct[2, 'b']),
            ],
            3: [
                hull.get_disaggregated_var(m.x[3], m.disjunct[3, 'a']),
                hull.get_disaggregated_var(m.x[3], m.disjunct[3, 'b']),
            ],
        }

        for i, disVars in disaggregatedVars.items():
            cons = hull.get_disaggregation_constraint(m.x[i], m.disjunction[i])
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
            (1, 'A'): [
                hull.get_disaggregated_var(m.a[1, 'A'], m.disjunct[0, 1, 'A']),
                hull.get_disaggregated_var(m.a[1, 'A'], m.disjunct[1, 1, 'A']),
            ],
            (1, 'B'): [
                hull.get_disaggregated_var(m.a[1, 'B'], m.disjunct[0, 1, 'B']),
                hull.get_disaggregated_var(m.a[1, 'B'], m.disjunct[1, 1, 'B']),
            ],
            (2, 'A'): [
                hull.get_disaggregated_var(m.a[2, 'A'], m.disjunct[0, 2, 'A']),
                hull.get_disaggregated_var(m.a[2, 'A'], m.disjunct[1, 2, 'A']),
            ],
            (2, 'B'): [
                hull.get_disaggregated_var(m.a[2, 'B'], m.disjunct[0, 2, 'B']),
                hull.get_disaggregated_var(m.a[2, 'B'], m.disjunct[1, 2, 'B']),
            ],
        }

        for i, disVars in disaggregatedVars.items():
            cons = hull.get_disaggregation_constraint(m.a[i], m.disjunction[i])
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

    def test_cannot_call_transformation_on_disjunction(self):
        ct.check_cannot_call_transformation_on_disjunction(self, 'hull')

    def check_trans_block_disjunctions_of_disjunct_datas(self, m):
        transBlock1 = m.component("_pyomo_gdp_hull_reformulation")
        self.assertIsInstance(transBlock1, Block)
        self.assertIsInstance(transBlock1.component("relaxedDisjuncts"), Block)
        # All of the transformed Disjuncts are here
        self.assertEqual(len(transBlock1.relaxedDisjuncts), 4)

        hull = TransformationFactory('gdp.hull')
        firstTerm2 = transBlock1.relaxedDisjuncts[2]
        self.assertIs(firstTerm2, m.firstTerm[2].transformation_block)
        self.assertIsInstance(firstTerm2.disaggregatedVars.component("x"), Var)
        constraints = hull.get_transformed_constraints(m.firstTerm[2].cons)
        self.assertEqual(len(constraints), 1)  # one equality constraint
        cons = constraints[0]
        self.assertIs(cons.parent_block(), firstTerm2)
        # also check for the bounds constraints for x
        dis_x = hull.get_disaggregated_var(m.x, m.firstTerm[2])
        cons = hull.get_var_bounds_constraint(dis_x)
        self.assertIsInstance(cons, Constraint)
        self.assertIs(cons.parent_block(), firstTerm2)
        self.assertEqual(len(cons), 2)

        secondTerm2 = transBlock1.relaxedDisjuncts[3]
        self.assertIs(secondTerm2, m.secondTerm[2].transformation_block)
        self.assertIsInstance(secondTerm2.disaggregatedVars.component("x"), Var)
        constraints = hull.get_transformed_constraints(m.secondTerm[2].cons)
        self.assertEqual(len(constraints), 1)
        cons = constraints[0]
        self.assertIs(cons.parent_block(), secondTerm2)
        # also check for the bounds constraints for x
        dis_x = hull.get_disaggregated_var(m.x, m.secondTerm[2])
        cons = hull.get_var_bounds_constraint(dis_x)
        self.assertIsInstance(cons, Constraint)
        self.assertIs(cons.parent_block(), secondTerm2)
        self.assertEqual(len(cons), 2)

        firstTerm1 = transBlock1.relaxedDisjuncts[0]
        self.assertIs(firstTerm1, m.firstTerm[1].transformation_block)
        self.assertIsInstance(firstTerm1.disaggregatedVars.component("x"), Var)
        self.assertTrue(firstTerm1.disaggregatedVars.x.is_fixed())
        self.assertEqual(value(firstTerm1.disaggregatedVars.x), 0)
        constraints = hull.get_transformed_constraints(m.firstTerm[1].cons)
        self.assertEqual(len(constraints), 1)
        cons = constraints[0]
        # It's just fixed to 0--so it's on the disaggregatedVar block, which is
        # fine.
        self.assertIs(cons.parent_block(), firstTerm1.disaggregatedVars)
        # also check for the bounds constraints for x
        dis_x = hull.get_disaggregated_var(m.x, m.firstTerm[1])
        cons = hull.get_var_bounds_constraint(dis_x)
        self.assertIsInstance(cons, Constraint)
        self.assertIs(cons.parent_block(), firstTerm1)
        self.assertEqual(len(cons), 2)

        secondTerm1 = transBlock1.relaxedDisjuncts[1]
        self.assertIs(secondTerm1, m.secondTerm[1].transformation_block)
        self.assertIsInstance(secondTerm1.disaggregatedVars.component("x"), Var)
        constraints = hull.get_transformed_constraints(m.secondTerm[1].cons)
        self.assertEqual(len(constraints), 1)
        cons = constraints[0]
        self.assertIs(cons.parent_block(), secondTerm1)
        # also check for the bounds constraints for x
        dis_x = hull.get_disaggregated_var(m.x, m.secondTerm[1])
        cons = hull.get_var_bounds_constraint(dis_x)
        self.assertIsInstance(cons, Constraint)
        self.assertIs(cons.parent_block(), secondTerm1)
        self.assertEqual(len(cons), 2)

    def test_simple_disjunction_of_disjunct_datas(self):
        ct.check_simple_disjunction_of_disjunct_datas(self, 'hull')

    def test_any_indexed_disjunction_of_disjunct_datas(self):
        m = models.makeAnyIndexedDisjunctionOfDisjunctDatas()
        TransformationFactory('gdp.hull').apply_to(m)

        self.check_trans_block_disjunctions_of_disjunct_datas(m)

        transBlock = m.component("_pyomo_gdp_hull_reformulation")
        self.assertIsInstance(transBlock.component("disjunction_xor"), Constraint)
        self.assertEqual(len(transBlock.component("disjunction_xor")), 2)

    def check_first_iteration(self, model):
        transBlock = model.component("_pyomo_gdp_hull_reformulation")
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(transBlock.component("disjunctionList_xor"), Constraint)
        self.assertEqual(len(transBlock.disjunctionList_xor), 1)
        self.assertFalse(model.disjunctionList[0].active)
        hull = TransformationFactory('gdp.hull')

        if model.component('firstTerm') is None:
            firstTerm_cons = hull.get_transformed_constraints(
                model.component("firstTerm[0]").cons
            )
            secondTerm_cons = hull.get_transformed_constraints(
                model.component("secondTerm[0]").cons
            )

        else:
            firstTerm_cons = hull.get_transformed_constraints(model.firstTerm[0].cons)
            secondTerm_cons = hull.get_transformed_constraints(model.secondTerm[0].cons)

        self.assertIsInstance(transBlock.relaxedDisjuncts, Block)
        self.assertEqual(len(transBlock.relaxedDisjuncts), 2)

        self.assertIsInstance(transBlock.relaxedDisjuncts[0].disaggregatedVars.x, Var)
        self.assertTrue(transBlock.relaxedDisjuncts[0].disaggregatedVars.x.is_fixed())
        self.assertEqual(value(transBlock.relaxedDisjuncts[0].disaggregatedVars.x), 0)
        self.assertEqual(len(firstTerm_cons), 1)
        self.assertIs(
            firstTerm_cons[0].parent_block(),
            # It fixes a var to 0
            transBlock.relaxedDisjuncts[0].disaggregatedVars,
        )
        self.assertIsInstance(transBlock.relaxedDisjuncts[0].x_bounds, Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[0].x_bounds), 2)

        self.assertIsInstance(transBlock.relaxedDisjuncts[1].disaggregatedVars.x, Var)
        self.assertFalse(transBlock.relaxedDisjuncts[1].disaggregatedVars.x.is_fixed())

        self.assertEqual(len(secondTerm_cons), 1)
        self.assertIs(secondTerm_cons[0].parent_block(), transBlock.relaxedDisjuncts[1])
        self.assertIsInstance(transBlock.relaxedDisjuncts[1].x_bounds, Constraint)
        self.assertEqual(len(transBlock.relaxedDisjuncts[1].x_bounds), 2)

    def check_second_iteration(self, model):
        transBlock = model.component("_pyomo_gdp_hull_reformulation_4")
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(transBlock.component("relaxedDisjuncts"), Block)
        self.assertEqual(len(transBlock.relaxedDisjuncts), 2)
        hull = TransformationFactory('gdp.hull')

        if model.component('firstTerm') is None:
            firstTerm_cons = hull.get_transformed_constraints(
                model.component("firstTerm[1]").cons
            )
            secondTerm_cons = hull.get_transformed_constraints(
                model.component("secondTerm[1]").cons
            )

        else:
            firstTerm_cons = hull.get_transformed_constraints(model.firstTerm[1].cons)
            secondTerm_cons = hull.get_transformed_constraints(model.secondTerm[1].cons)

        self.assertEqual(len(firstTerm_cons), 1)
        self.assertIs(firstTerm_cons[0].parent_block(), transBlock.relaxedDisjuncts[0])
        self.assertEqual(len(secondTerm_cons), 1)
        self.assertIs(secondTerm_cons[0].parent_block(), transBlock.relaxedDisjuncts[1])

        orig = model.component("_pyomo_gdp_hull_reformulation")
        self.assertIsInstance(
            model.disjunctionList[1].algebraic_constraint, constraint.ConstraintData
        )
        self.assertIsInstance(
            model.disjunctionList[0].algebraic_constraint, constraint.ConstraintData
        )
        self.assertFalse(model.disjunctionList[1].active)
        self.assertFalse(model.disjunctionList[0].active)

    def test_disjunction_and_disjuncts_indexed_by_any(self):
        ct.check_disjunction_and_disjuncts_indexed_by_any(self, 'hull')

    def test_iteratively_adding_disjunctions_transform_container(self):
        ct.check_iteratively_adding_disjunctions_transform_container(self, 'hull')

    def test_iteratively_adding_disjunctions_transform_model(self):
        ct.check_iteratively_adding_disjunctions_transform_model(self, 'hull')

    def test_iteratively_adding_to_indexed_disjunction_on_block(self):
        ct.check_iteratively_adding_to_indexed_disjunction_on_block(self, 'hull')


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
                d.cons_model = Constraint(expr=m.component("b.x") == 0)
            else:
                d.cons_model = Constraint(expr=m.component("b.x") <= -5)

        m.disjunct = Disjunct([0, 1], rule=disjunct_rule)
        m.disjunction = Disjunction(expr=[m.disjunct[0], m.disjunct[1]])

        return m

    def test_disaggregation_constraints(self):
        m = self.makeModel()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        disaggregationConstraints = (
            m._pyomo_gdp_hull_reformulation.disaggregationConstraints
        )
        consmap = [
            (m.component("b.x"), disaggregationConstraints[0]),
            (m.b.x, disaggregationConstraints[1]),
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
        ct.check_deactivated_disjunct_leaves_nested_disjunct_active(self, 'hull')

    def test_mappings_between_disjunctions_and_xors(self):
        # Tests that the XOR constraints are put on the parent block of the
        # disjunction, and checks the mappings.
        m = models.makeNestedDisjunctions()
        transform = TransformationFactory('gdp.hull')
        transform.apply_to(m)

        transBlock = m.component("_pyomo_gdp_hull_reformulation")

        disjunctionPairs = [
            (m.disjunction, transBlock.disjunction_xor),
            (
                m.disjunct[1].innerdisjunction[0],
                m.disjunct[1]
                .innerdisjunction[0]
                .algebraic_constraint.parent_block()
                .innerdisjunction_xor[0],
            ),
            (
                m.simpledisjunct.innerdisjunction,
                m.simpledisjunct.innerdisjunction.algebraic_constraint.parent_block().innerdisjunction_xor,
            ),
        ]

        # check disjunction mappings
        for disjunction, xor in disjunctionPairs:
            self.assertIs(disjunction.algebraic_constraint, xor)
            self.assertIs(transform.get_src_disjunction(xor), disjunction)

    def test_unique_reference_to_nested_indicator_var(self):
        ct.check_unique_reference_to_nested_indicator_var(self, 'hull')

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

    def test_nested_disjunction_target(self):
        ct.check_nested_disjunction_target(self, 'hull')

    def test_target_appears_twice(self):
        ct.check_target_appears_twice(self, 'hull')

    @unittest.skipIf(not linear_solvers, "No linear solver available")
    def test_relaxation_feasibility(self):
        m = models.makeNestedDisjunctions_FlatDisjuncts()
        TransformationFactory('gdp.hull').apply_to(m)

        solver = SolverFactory(linear_solvers[0])

        cases = [
            (True, True, True, True, None),
            (False, False, False, False, None),
            (True, False, False, False, None),
            (False, True, False, False, 1.1),
            (False, False, True, False, None),
            (False, False, False, True, None),
            (True, True, False, False, None),
            (True, False, True, False, 1.2),
            (True, False, False, True, 1.3),
            (True, False, True, True, None),
        ]
        for case in cases:
            m.d1.indicator_var.fix(case[0])
            m.d2.indicator_var.fix(case[1])
            m.d3.indicator_var.fix(case[2])
            m.d4.indicator_var.fix(case[3])
            results = solver.solve(m)
            if case[4] is None:
                self.assertEqual(
                    results.solver.termination_condition,
                    TerminationCondition.infeasible,
                )
            else:
                self.assertEqual(
                    results.solver.termination_condition, TerminationCondition.optimal
                )
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
            (True, True, True, True, None),
            (False, False, False, False, None),
            (True, False, False, False, None),
            (False, True, False, False, 1.1),
            (False, False, True, False, None),
            (False, False, False, True, None),
            (True, True, False, False, None),
            (True, False, True, False, 1.2),
            (True, False, False, True, 1.3),
            (True, False, True, True, None),
        ]
        for case in cases:
            m.d1.indicator_var.fix(case[0])
            m.d2.indicator_var.fix(case[1])
            m.d3.indicator_var.fix(case[2])
            m.d4.indicator_var.fix(case[3])
            results = solver.solve(m)
            if case[4] is None:
                self.assertEqual(
                    results.solver.termination_condition,
                    TerminationCondition.infeasible,
                )
            else:
                self.assertEqual(
                    results.solver.termination_condition, TerminationCondition.optimal
                )
                self.assertEqual(value(m.obj), case[4])

    def test_create_using(self):
        m = models.makeNestedDisjunctions_FlatDisjuncts()
        self.diff_apply_to_and_create_using(m)

    def check_outer_disaggregation_constraint(self, cons, var, disj1, disj2, rhs=None):
        if rhs is None:
            rhs = var
        hull = TransformationFactory('gdp.hull')
        self.assertTrue(cons.active)
        self.assertEqual(cons.lower, 0)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        ct.check_linear_coef(self, repn, rhs, 1)
        ct.check_linear_coef(self, repn, hull.get_disaggregated_var(var, disj1), -1)
        ct.check_linear_coef(self, repn, hull.get_disaggregated_var(var, disj2), -1)

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

    def check_transformed_constraint(self, cons, dis, lb, ind_var):
        hull = TransformationFactory('gdp.hull')
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertTrue(cons.active)
        self.assertIsNone(cons.lower)
        self.assertEqual(value(cons.upper), 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, dis, -1)
        ct.check_linear_coef(self, repn, ind_var, lb)

        orig = ind_var.parent_block().c
        self.assertIs(hull.get_src_constraint(cons), orig)

    def test_transformed_model_nestedDisjuncts(self):
        # This test tests *everything* for a simple nested disjunction case.
        m = models.makeNestedDisjunctions_NestedDisjuncts()
        m.LocalVars = Suffix(direction=Suffix.LOCAL)
        m.LocalVars[m.d1] = [
            m.d1.binary_indicator_var,
            m.d1.d3.binary_indicator_var,
            m.d1.d4.binary_indicator_var,
        ]

        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        self.check_transformed_model_nestedDisjuncts(
            m, m.d1.d3.binary_indicator_var, m.d1.d4.binary_indicator_var
        )

        # Last, check that there aren't things we weren't expecting
        all_cons = list(
            m.component_data_objects(Constraint, active=True, descend_into=Block)
        )
        # 2 disaggregation constraints for x 0,3
        # + 6 bounds constraints for x 6,8,9,13,14,16
        # + 2 bounds constraints for inner indicator vars 11, 12
        # + 2 exactly-one constraints 1,4
        # + 4 transformed constraints 2,5,7,15
        self.assertEqual(len(all_cons), 16)

    def check_transformed_model_nestedDisjuncts(self, m, d3, d4):
        # This function checks all of the 16 constraint expressions from
        # transforming models.makeNestedDisjunction_NestedDisjuncts when
        # declaring the inner indicator vars (d3 and d4) as local. Note that it
        # also is a correct test for the case where the inner indicator vars are
        # *not* declared as local, but not a complete one, since there are
        # additional constraints in that case (see
        # check_transformation_blocks_nestedDisjunctions in common_tests.py).
        hull = TransformationFactory('gdp.hull')
        transBlock = m._pyomo_gdp_hull_reformulation
        self.assertTrue(transBlock.active)

        # check outer xor
        xor = transBlock.disj_xor
        self.assertIsInstance(xor, Constraint)
        ct.check_obj_in_active_tree(self, xor)
        assertExpressionsEqual(
            self, xor.expr, m.d1.binary_indicator_var + m.d2.binary_indicator_var == 1
        )
        self.assertIs(xor, m.disj.algebraic_constraint)
        self.assertIs(m.disj, hull.get_src_disjunction(xor))

        # check inner xor
        xor = m.d1.disj2.algebraic_constraint
        self.assertIs(m.d1.disj2, hull.get_src_disjunction(xor))
        xor = hull.get_transformed_constraints(xor)
        self.assertEqual(len(xor), 1)
        xor = xor[0]
        ct.check_obj_in_active_tree(self, xor)
        xor_expr = self.simplify_cons(xor)
        assertExpressionsEqual(
            self, xor_expr, d3 + d4 - m.d1.binary_indicator_var == 0.0
        )

        # check disaggregation constraints
        x_d3 = hull.get_disaggregated_var(m.x, m.d1.d3)
        x_d4 = hull.get_disaggregated_var(m.x, m.d1.d4)
        x_d1 = hull.get_disaggregated_var(m.x, m.d1)
        x_d2 = hull.get_disaggregated_var(m.x, m.d2)
        for x in [x_d1, x_d2, x_d3, x_d4]:
            self.assertEqual(x.lb, 0)
            self.assertEqual(x.ub, 2)
        # Inner disjunction
        cons = hull.get_disaggregation_constraint(m.x, m.d1.disj2)
        ct.check_obj_in_active_tree(self, cons)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, x_d1 - x_d3 - x_d4 == 0.0)
        # Outer disjunction
        cons = hull.get_disaggregation_constraint(m.x, m.disj)
        ct.check_obj_in_active_tree(self, cons)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, m.x - x_d1 - x_d2 == 0.0)

        ## Transformed constraints
        cons = hull.get_transformed_constraints(m.d1.d3.c)
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        ct.check_obj_in_active_tree(self, cons)
        cons_expr = self.simplify_leq_cons(cons)
        assertExpressionsEqual(self, cons_expr, 1.2 * d3 - x_d3 <= 0.0)

        cons = hull.get_transformed_constraints(m.d1.d4.c)
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        ct.check_obj_in_active_tree(self, cons)
        cons_expr = self.simplify_leq_cons(cons)
        assertExpressionsEqual(self, cons_expr, 1.3 * d4 - x_d4 <= 0.0)

        cons = hull.get_transformed_constraints(m.d1.c)
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        ct.check_obj_in_active_tree(self, cons)
        cons_expr = self.simplify_leq_cons(cons)
        assertExpressionsEqual(
            self, cons_expr, 1.0 * m.d1.binary_indicator_var - x_d1 <= 0.0
        )

        cons = hull.get_transformed_constraints(m.d2.c)
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        ct.check_obj_in_active_tree(self, cons)
        cons_expr = self.simplify_leq_cons(cons)
        assertExpressionsEqual(
            self, cons_expr, 1.1 * m.d2.binary_indicator_var - x_d2 <= 0.0
        )

        ## Bounds constraints
        cons = hull.get_var_bounds_constraint(x_d1)
        # the lb is trivial in this case, so we just have 1
        self.assertEqual(len(cons), 1)
        ct.check_obj_in_active_tree(self, cons['ub'])
        cons_expr = self.simplify_leq_cons(cons['ub'])
        assertExpressionsEqual(
            self, cons_expr, x_d1 - 2 * m.d1.binary_indicator_var <= 0.0
        )
        cons = hull.get_var_bounds_constraint(x_d2)
        # the lb is trivial in this case, so we just have 1
        self.assertEqual(len(cons), 1)
        ct.check_obj_in_active_tree(self, cons['ub'])
        cons_expr = self.simplify_leq_cons(cons['ub'])
        assertExpressionsEqual(
            self, cons_expr, x_d2 - 2 * m.d2.binary_indicator_var <= 0.0
        )
        cons = hull.get_var_bounds_constraint(x_d3, m.d1.d3)
        # the lb is trivial in this case, so we just have 1
        self.assertEqual(len(cons), 1)
        # And we know it has actually been transformed again, so get that one
        cons = hull.get_transformed_constraints(cons['ub'])
        self.assertEqual(len(cons), 1)
        ub = cons[0]
        ct.check_obj_in_active_tree(self, ub)
        cons_expr = self.simplify_leq_cons(ub)
        assertExpressionsEqual(self, cons_expr, x_d3 - 2 * d3 <= 0.0)
        cons = hull.get_var_bounds_constraint(x_d4, m.d1.d4)
        # the lb is trivial in this case, so we just have 1
        self.assertEqual(len(cons), 1)
        # And we know it has actually been transformed again, so get that one
        cons = hull.get_transformed_constraints(cons['ub'])
        self.assertEqual(len(cons), 1)
        ub = cons[0]
        ct.check_obj_in_active_tree(self, ub)
        cons_expr = self.simplify_leq_cons(ub)
        assertExpressionsEqual(self, cons_expr, x_d4 - 2 * d4 <= 0.0)
        cons = hull.get_var_bounds_constraint(x_d3, m.d1)
        self.assertEqual(len(cons), 1)
        ub = cons['ub']
        ct.check_obj_in_active_tree(self, ub)
        cons_expr = self.simplify_leq_cons(ub)
        assertExpressionsEqual(
            self, cons_expr, x_d3 - 2 * m.d1.binary_indicator_var <= 0.0
        )
        cons = hull.get_var_bounds_constraint(x_d4, m.d1)
        self.assertEqual(len(cons), 1)
        ub = cons['ub']
        ct.check_obj_in_active_tree(self, ub)
        cons_expr = self.simplify_leq_cons(ub)
        assertExpressionsEqual(
            self, cons_expr, x_d4 - 2 * m.d1.binary_indicator_var <= 0.0
        )

        # Bounds constraints for local vars
        cons = hull.get_var_bounds_constraint(d3)
        ct.check_obj_in_active_tree(self, cons['ub'])
        assertExpressionsEqual(self, cons['ub'].expr, d3 <= m.d1.binary_indicator_var)
        cons = hull.get_var_bounds_constraint(d4)
        ct.check_obj_in_active_tree(self, cons['ub'])
        assertExpressionsEqual(self, cons['ub'].expr, d4 <= m.d1.binary_indicator_var)

    @unittest.skipIf(not linear_solvers, "No linear solver available")
    def test_solve_nested_model(self):
        # This is really a test that our variable references have all been moved
        # up correctly.
        m = models.makeNestedDisjunctions_NestedDisjuncts()
        m.LocalVars = Suffix(direction=Suffix.LOCAL)
        m.LocalVars[m.d1] = [
            m.d1.binary_indicator_var,
            m.d1.d3.binary_indicator_var,
            m.d1.d4.binary_indicator_var,
        ]
        hull = TransformationFactory('gdp.hull')
        m_hull = hull.create_using(m)

        SolverFactory(linear_solvers[0]).solve(m_hull)

        # check solution
        self.assertEqual(value(m_hull.d1.binary_indicator_var), 0)
        self.assertEqual(value(m_hull.d2.binary_indicator_var), 1)
        self.assertEqual(value(m_hull.x), 1.1)

        # transform inner problem with bigm, outer with hull and make sure it
        # still works
        TransformationFactory('gdp.bigm').apply_to(m, targets=(m.d1.disj2))
        hull.apply_to(m)

        SolverFactory(linear_solvers[0]).solve(m)

        # check solution
        self.assertEqual(value(m.d1.binary_indicator_var), 0)
        self.assertEqual(value(m.d2.binary_indicator_var), 1)
        self.assertEqual(value(m.x), 1.1)

    @unittest.skipIf(not linear_solvers, "No linear solver available")
    def test_disaggregated_vars_are_set_to_0_correctly(self):
        m = models.makeNestedDisjunctions_FlatDisjuncts()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        # this should be a feasible integer solution
        m.d1.indicator_var.fix(False)
        m.d2.indicator_var.fix(True)
        m.d3.indicator_var.fix(False)
        m.d4.indicator_var.fix(False)

        results = SolverFactory(linear_solvers[0]).solve(m)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(value(m.x), 1.1)

        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d1)), 0)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d2)), 1.1)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d3)), 0)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d4)), 0)

        # and what if one of the inner disjuncts is true?
        m.d1.indicator_var.fix(True)
        m.d2.indicator_var.fix(False)
        m.d3.indicator_var.fix(True)
        m.d4.indicator_var.fix(False)

        results = SolverFactory(linear_solvers[0]).solve(m)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(value(m.x), 1.2)

        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d1)), 1.2)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d2)), 0)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d3)), 1.2)
        self.assertEqual(value(hull.get_disaggregated_var(m.x, m.d4)), 0)

    def test_nested_with_local_vars(self):
        m = ConcreteModel()

        m.x = Var(bounds=(0, 10))
        m.S = RangeSet(2)

        @m.Disjunct()
        def d_l(d):
            d.lambdas = Var(m.S, bounds=(0, 1))
            d.LocalVars = Suffix(direction=Suffix.LOCAL)
            d.LocalVars[d] = list(d.lambdas.values())
            d.c1 = Constraint(expr=d.lambdas[1] + d.lambdas[2] == 1)
            d.c2 = Constraint(expr=m.x == 2 * d.lambdas[1] + 3 * d.lambdas[2])

        @m.Disjunct()
        def d_r(d):
            @d.Disjunct()
            def d_l(e):
                e.lambdas = Var(m.S, bounds=(0, 1))
                e.LocalVars = Suffix(direction=Suffix.LOCAL)
                e.LocalVars[e] = list(e.lambdas.values())
                e.c1 = Constraint(expr=e.lambdas[1] + e.lambdas[2] == 1)
                e.c2 = Constraint(expr=m.x == 2 * e.lambdas[1] + 3 * e.lambdas[2])

            @d.Disjunct()
            def d_r(e):
                e.lambdas = Var(m.S, bounds=(0, 1))
                e.LocalVars = Suffix(direction=Suffix.LOCAL)
                e.LocalVars[e] = list(e.lambdas.values())
                e.c1 = Constraint(expr=e.lambdas[1] + e.lambdas[2] == 1)
                e.c2 = Constraint(expr=m.x == 2 * e.lambdas[1] + 3 * e.lambdas[2])

            d.LocalVars = Suffix(direction=Suffix.LOCAL)
            d.LocalVars[d] = [
                d.d_l.indicator_var.get_associated_binary(),
                d.d_r.indicator_var.get_associated_binary(),
            ]
            d.inner_disj = Disjunction(expr=[d.d_l, d.d_r])

        m.disj = Disjunction(expr=[m.d_l, m.d_r])
        m.obj = Objective(expr=m.x)

        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        x1 = hull.get_disaggregated_var(m.x, m.d_l)
        x2 = hull.get_disaggregated_var(m.x, m.d_r)
        x3 = hull.get_disaggregated_var(m.x, m.d_r.d_l)
        x4 = hull.get_disaggregated_var(m.x, m.d_r.d_r)

        for d, x in [(m.d_l, x1), (m.d_r.d_l, x3), (m.d_r.d_r, x4)]:
            lambda1 = hull.get_disaggregated_var(d.lambdas[1], d)
            self.assertIs(lambda1, d.lambdas[1])
            lambda2 = hull.get_disaggregated_var(d.lambdas[2], d)
            self.assertIs(lambda2, d.lambdas[2])

            cons = hull.get_transformed_constraints(d.c1)
            self.assertEqual(len(cons), 1)
            convex_combo = cons[0]
            convex_combo_expr = self.simplify_cons(convex_combo)
            assertExpressionsEqual(
                self,
                convex_combo_expr,
                lambda1 + lambda2 - d.indicator_var.get_associated_binary() == 0.0,
            )
            cons = hull.get_transformed_constraints(d.c2)
            self.assertEqual(len(cons), 1)
            get_x = cons[0]
            get_x_expr = self.simplify_cons(get_x)
            assertExpressionsEqual(
                self, get_x_expr, x - 2 * lambda1 - 3 * lambda2 == 0.0
            )

        cons = hull.get_disaggregation_constraint(m.x, m.disj)
        assertExpressionsEqual(self, cons.expr, m.x == x1 + x2)
        cons = hull.get_disaggregation_constraint(m.x, m.d_r.inner_disj)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, x2 - x3 - x4 == 0.0)

    def test_nested_with_var_that_does_not_appear_in_every_disjunct(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.y = Var(bounds=(-4, 5))
        m.parent1 = Disjunct()
        m.parent2 = Disjunct()
        m.parent2.c = Constraint(expr=m.x == 0)
        m.parent_disjunction = Disjunction(expr=[m.parent1, m.parent2])
        m.child1 = Disjunct()
        m.child1.c = Constraint(expr=m.x <= 8)
        m.child2 = Disjunct()
        m.child2.c = Constraint(expr=m.x + m.y <= 3)
        m.child3 = Disjunct()
        m.child3.c = Constraint(expr=m.x <= 7)
        m.parent1.disjunction = Disjunction(expr=[m.child1, m.child2, m.child3])

        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        y_c2 = hull.get_disaggregated_var(m.y, m.child2)
        self.assertEqual(y_c2.bounds, (-4, 5))
        other_y = hull.get_disaggregated_var(m.y, m.child1)
        self.assertEqual(other_y.bounds, (-4, 5))
        other_other_y = hull.get_disaggregated_var(m.y, m.child3)
        self.assertIs(other_y, other_other_y)
        y_p1 = hull.get_disaggregated_var(m.y, m.parent1)
        self.assertEqual(y_p1.bounds, (-4, 5))
        y_p2 = hull.get_disaggregated_var(m.y, m.parent2)
        self.assertEqual(y_p2.bounds, (-4, 5))

        y_cons = hull.get_disaggregation_constraint(m.y, m.parent1.disjunction)
        # check that the disaggregated ys in the nested just sum to the original
        y_cons_expr = self.simplify_cons(y_cons)
        assertExpressionsEqual(self, y_cons_expr, y_p1 - other_y - y_c2 == 0.0)
        y_cons = hull.get_disaggregation_constraint(m.y, m.parent_disjunction)
        y_cons_expr = self.simplify_cons(y_cons)
        assertExpressionsEqual(self, y_cons_expr, m.y - y_p2 - y_p1 == 0.0)

        x_c1 = hull.get_disaggregated_var(m.x, m.child1)
        x_c2 = hull.get_disaggregated_var(m.x, m.child2)
        x_c3 = hull.get_disaggregated_var(m.x, m.child3)
        x_p1 = hull.get_disaggregated_var(m.x, m.parent1)
        x_p2 = hull.get_disaggregated_var(m.x, m.parent2)
        x_cons_parent = hull.get_disaggregation_constraint(m.x, m.parent_disjunction)
        assertExpressionsEqual(self, x_cons_parent.expr, m.x == x_p1 + x_p2)
        x_cons_child = hull.get_disaggregation_constraint(m.x, m.parent1.disjunction)
        x_cons_child_expr = self.simplify_cons(x_cons_child)
        assertExpressionsEqual(
            self, x_cons_child_expr, x_p1 - x_c1 - x_c2 - x_c3 == 0.0
        )

    def simplify_cons(self, cons):
        visitor = LinearRepnVisitor({}, {}, {}, None)
        lb = cons.lower
        ub = cons.upper
        self.assertEqual(cons.lb, cons.ub)
        repn = visitor.walk_expression(cons.body)
        self.assertIsNone(repn.nonlinear)
        return repn.to_expression(visitor) == lb

    def simplify_leq_cons(self, cons):
        visitor = LinearRepnVisitor({}, {}, {}, None)
        self.assertIsNone(cons.lower)
        ub = cons.upper
        repn = visitor.walk_expression(cons.body)
        self.assertIsNone(repn.nonlinear)
        return repn.to_expression(visitor) <= ub

    def test_nested_with_var_that_skips_a_level(self):
        m = ConcreteModel()

        m.x = Var(bounds=(-2, 9))
        m.y = Var(bounds=(-3, 8))

        m.y1 = Disjunct()
        m.y1.c1 = Constraint(expr=m.x >= 4)
        m.y1.z1 = Disjunct()
        m.y1.z1.c1 = Constraint(expr=m.y == 2)
        m.y1.z1.w1 = Disjunct()
        m.y1.z1.w1.c1 = Constraint(expr=m.x == 3)
        m.y1.z1.w2 = Disjunct()
        m.y1.z1.w2.c1 = Constraint(expr=m.x >= 1)
        m.y1.z1.disjunction = Disjunction(expr=[m.y1.z1.w1, m.y1.z1.w2])
        m.y1.z2 = Disjunct()
        m.y1.z2.c1 = Constraint(expr=m.y == 1)
        m.y1.disjunction = Disjunction(expr=[m.y1.z1, m.y1.z2])
        m.y2 = Disjunct()
        m.y2.c1 = Constraint(expr=m.x == 4)
        m.disjunction = Disjunction(expr=[m.y1, m.y2])

        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        x_y1 = hull.get_disaggregated_var(m.x, m.y1)
        x_y2 = hull.get_disaggregated_var(m.x, m.y2)
        x_z1 = hull.get_disaggregated_var(m.x, m.y1.z1)
        x_z2 = hull.get_disaggregated_var(m.x, m.y1.z2)
        x_w1 = hull.get_disaggregated_var(m.x, m.y1.z1.w1)
        x_w2 = hull.get_disaggregated_var(m.x, m.y1.z1.w2)

        y_z1 = hull.get_disaggregated_var(m.y, m.y1.z1)
        y_z2 = hull.get_disaggregated_var(m.y, m.y1.z2)
        y_y1 = hull.get_disaggregated_var(m.y, m.y1)
        y_y2 = hull.get_disaggregated_var(m.y, m.y2)

        cons = hull.get_disaggregation_constraint(m.x, m.y1.z1.disjunction)
        self.assertTrue(cons.active)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, x_z1 - x_w1 - x_w2 == 0.0)
        cons = hull.get_disaggregation_constraint(m.x, m.y1.disjunction)
        self.assertTrue(cons.active)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, x_y1 - x_z2 - x_z1 == 0.0)
        cons = hull.get_disaggregation_constraint(m.x, m.disjunction)
        self.assertTrue(cons.active)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, m.x - x_y1 - x_y2 == 0.0)
        cons = hull.get_disaggregation_constraint(
            m.y, m.y1.z1.disjunction, raise_exception=False
        )
        self.assertIsNone(cons)
        cons = hull.get_disaggregation_constraint(m.y, m.y1.disjunction)
        self.assertTrue(cons.active)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, y_y1 - y_z1 - y_z2 == 0.0)
        cons = hull.get_disaggregation_constraint(m.y, m.disjunction)
        self.assertTrue(cons.active)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(self, cons_expr, m.y - y_y2 - y_y1 == 0.0)

    @unittest.skipUnless(gurobi_available, "Gurobi is not available")
    def test_do_not_assume_nested_indicators_local(self):
        ct.check_do_not_assume_nested_indicators_local(self, 'gdp.hull')


class TestSpecialCases(unittest.TestCase):
    def test_local_vars(self):
        """checks that if nothing is marked as local, we assume it is all
        global. We disaggregate everything to be safe."""
        m = ConcreteModel()
        m.x = Var(bounds=(5, 100))
        m.y = Var(bounds=(0, 100))
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.y >= m.x)
        m.d2 = Disjunct()
        m.d2.z = Var()
        m.d2.c = Constraint(expr=m.y >= m.d2.z)
        m.disj = Disjunction(expr=[m.d1, m.d2])

        self.assertRaisesRegex(
            GDP_Error,
            ".*Missing bound for d2.z.*",
            TransformationFactory('gdp.hull').create_using,
            m,
        )
        m.d2.z.setlb(7)
        self.assertRaisesRegex(
            GDP_Error,
            ".*Missing bound for d2.z.*",
            TransformationFactory('gdp.hull').create_using,
            m,
        )
        m.d2.z.setub(9)

        i = TransformationFactory('gdp.hull').create_using(m)
        rd = i._pyomo_gdp_hull_reformulation.relaxedDisjuncts[1]
        varBlock = rd.disaggregatedVars
        # z should be disaggregated because we can't be sure it's not somewhere
        # else on the model. (Note however that the copy of x corresponding to
        # this disjunct is on the disjunction block)
        self.assertEqual(sorted(varBlock.component_map(Var)), ['d2.z', 'y'])
        # constraint on the disjunction block
        self.assertEqual(len(rd.component_map(Constraint)), 3)
        # bounds haven't changed on original
        self.assertEqual(i.d2.z.bounds, (7, 9))
        # check disaggregated variable
        z = varBlock.component('d2.z')
        self.assertIsInstance(z, Var)
        self.assertEqual(z.bounds, (0, 9))
        z_bounds = rd.component("d2.z_bounds")
        self.assertEqual(len(z_bounds), 2)
        self.assertEqual(z_bounds['lb'].lower, None)
        self.assertEqual(z_bounds['lb'].upper, 0)
        self.assertEqual(z_bounds['ub'].lower, None)
        self.assertEqual(z_bounds['ub'].upper, 0)
        i.d2.indicator_var = True
        z.set_value(2)
        self.assertEqual(z_bounds['lb'].body(), 5)
        self.assertEqual(z_bounds['ub'].body(), -7)

        m.d2.z.setlb(-9)
        m.d2.z.setub(-7)
        i = TransformationFactory('gdp.hull').create_using(m)
        rd = i._pyomo_gdp_hull_reformulation.relaxedDisjuncts[1]
        varBlock = rd.disaggregatedVars
        self.assertEqual(sorted(varBlock.component_map(Var)), ['d2.z', 'y'])
        self.assertEqual(len(rd.component_map(Constraint)), 3)
        # original bounds unchanged
        self.assertEqual(i.d2.z.bounds, (-9, -7))
        # check disaggregated variable
        z = varBlock.component("d2.z")
        self.assertIsInstance(z, Var)
        self.assertEqual(z.bounds, (-9, 0))
        z_bounds = rd.component("d2.z_bounds")
        self.assertEqual(len(z_bounds), 2)
        self.assertEqual(z_bounds['lb'].lower, None)
        self.assertEqual(z_bounds['lb'].upper, 0)
        self.assertEqual(z_bounds['ub'].lower, None)
        self.assertEqual(z_bounds['ub'].upper, 0)
        i.d2.indicator_var = True
        z.set_value(2)
        self.assertEqual(z_bounds['lb'].body(), -11)
        self.assertEqual(z_bounds['ub'].body(), 9)

    def test_local_var_suffix(self):
        hull = TransformationFactory('gdp.hull')

        model = ConcreteModel()
        model.x = Var(bounds=(5, 100))
        model.y = Var(bounds=(0, 100))
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
        z_disaggregated = m.d2.transformation_block.disaggregatedVars.component("d2.z")
        self.assertIsInstance(z_disaggregated, Var)
        self.assertIs(z_disaggregated, hull.get_disaggregated_var(m.d2.z, m.d2))

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
        self.assertIsNone(m.d2.transformation_block.disaggregatedVars.component("z"))


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
            self, 'hull'
        )

    def test_silly_target(self):
        ct.check_silly_target(self, 'hull')

    def test_retrieving_nondisjunctive_components(self):
        ct.check_retrieving_nondisjunctive_components(self, 'hull')

    def test_transform_empty_disjunction(self):
        ct.check_transform_empty_disjunction(self, 'hull')

    def test_deactivated_disjunct_nonzero_indicator_var(self):
        ct.check_deactivated_disjunct_nonzero_indicator_var(self, 'hull')

    def test_deactivated_disjunct_unfixed_indicator_var(self):
        ct.check_deactivated_disjunct_unfixed_indicator_var(self, 'hull')

    def test_infeasible_xor_because_all_disjuncts_deactivated(self):
        m = ct.setup_infeasible_xor_because_all_disjuncts_deactivated(self, 'hull')
        hull = TransformationFactory('gdp.hull')
        transBlock = m.component("_pyomo_gdp_hull_reformulation")
        self.assertIsInstance(transBlock, Block)
        self.assertEqual(len(transBlock.relaxedDisjuncts), 2)
        self.assertIsInstance(transBlock.component("disjunction_xor"), Constraint)
        disjunct1 = transBlock.relaxedDisjuncts[0]
        # we disaggregated the (deactivated) indicator variables
        d3_ind = (
            m.disjunction_disjuncts[0]
            .nestedDisjunction_disjuncts[0]
            .binary_indicator_var
        )
        d4_ind = (
            m.disjunction_disjuncts[0]
            .nestedDisjunction_disjuncts[1]
            .binary_indicator_var
        )
        d3_ind_dis = disjunct1.disaggregatedVars.component(
            "disjunction_disjuncts[0].nestedDisjunction_"
            "disjuncts[0].binary_indicator_var"
        )
        self.assertIs(
            hull.get_disaggregated_var(d3_ind, m.disjunction_disjuncts[0]), d3_ind_dis
        )
        self.assertIs(hull.get_src_var(d3_ind_dis), d3_ind)
        d4_ind_dis = disjunct1.disaggregatedVars.component(
            "disjunction_disjuncts[0].nestedDisjunction_"
            "disjuncts[1].binary_indicator_var"
        )
        self.assertIs(
            hull.get_disaggregated_var(d4_ind, m.disjunction_disjuncts[0]), d4_ind_dis
        )
        self.assertIs(hull.get_src_var(d4_ind_dis), d4_ind)

        relaxed_xor = hull.get_transformed_constraints(
            m.disjunction_disjuncts[0].nestedDisjunction.algebraic_constraint
        )
        self.assertEqual(len(relaxed_xor), 1)
        relaxed_xor = relaxed_xor[0]
        repn = generate_standard_repn(relaxed_xor.body)
        self.assertEqual(value(relaxed_xor.lower), 0)
        self.assertEqual(value(relaxed_xor.upper), 0)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 3)
        # constraint says that the disaggregated indicator variables of the
        # nested disjuncts sum to the indicator variable of the outer disjunct.
        ct.check_linear_coef(self, repn, m.disjunction.disjuncts[0].indicator_var, -1)
        ct.check_linear_coef(self, repn, d3_ind_dis, 1)
        ct.check_linear_coef(self, repn, d4_ind_dis, 1)
        self.assertEqual(repn.constant, 0)

        # but the disaggregation constraints are going to force them to 0 (which
        # will in turn force the outer disjunct indicator variable to 0, which
        # is what we want)
        d3_ind_dis_cons = transBlock.disaggregationConstraints[1]
        self.assertEqual(d3_ind_dis_cons.lower, 0)
        self.assertEqual(d3_ind_dis_cons.upper, 0)
        repn = generate_standard_repn(d3_ind_dis_cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, 0)
        ct.check_linear_coef(self, repn, d3_ind_dis, -1)
        ct.check_linear_coef(self, repn, transBlock._disaggregatedVars[0], -1)
        d4_ind_dis_cons = transBlock.disaggregationConstraints[2]
        self.assertEqual(d4_ind_dis_cons.lower, 0)
        self.assertEqual(d4_ind_dis_cons.upper, 0)
        repn = generate_standard_repn(d4_ind_dis_cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, 0)
        ct.check_linear_coef(self, repn, d4_ind_dis, -1)
        ct.check_linear_coef(self, repn, transBlock._disaggregatedVars[1], -1)

    def test_mapping_method_errors(self):
        m = models.makeTwoTermDisj_Nonlinear()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        with self.assertRaisesRegex(
            GDP_Error,
            ".*Either 'w' is not a disaggregated variable, "
            "or the disjunction that disaggregates it has "
            "not been properly transformed.",
        ):
            hull.get_var_bounds_constraint(m.w)

        log = StringIO()
        with LoggingIntercept(log, 'pyomo.gdp.hull', logging.ERROR):
            self.assertRaisesRegex(
                KeyError,
                r".*disjunction",
                hull.get_disaggregation_constraint,
                m.d[1].transformation_block.disaggregatedVars.w,
                m.disjunction,
            )
        self.assertRegex(
            log.getvalue(),
            ".*It doesn't appear that "
            r"'_pyomo_gdp_hull_reformulation."
            r"relaxedDisjuncts\[1\].disaggregatedVars.w' "
            r"is a variable that was disaggregated by "
            r"Disjunction 'disjunction'",
        )

        with self.assertRaisesRegex(
            GDP_Error, ".*'w' does not appear to be a disaggregated variable"
        ):
            hull.get_src_var(m.w)

        with self.assertRaisesRegex(
            GDP_Error,
            r".*It does not appear "
            r"'_pyomo_gdp_hull_reformulation."
            r"relaxedDisjuncts\[1\].disaggregatedVars.w' "
            r"is a variable that appears in disjunct "
            r"'d\[1\]'",
        ):
            hull.get_disaggregated_var(
                m.d[1].transformation_block.disaggregatedVars.w, m.d[1]
            )

        m.random_disjunction = Disjunction(expr=[m.w == 2, m.w >= 7])
        self.assertRaisesRegex(
            GDP_Error,
            "Disjunction 'random_disjunction' has not been properly "
            "transformed: None of its disjuncts are transformed.",
            hull.get_disaggregation_constraint,
            m.w,
            m.random_disjunction,
        )

        self.assertRaisesRegex(
            GDP_Error,
            r"Disjunct 'random_disjunction_disjuncts\[0\]' has not been "
            r"transformed",
            hull.get_disaggregated_var,
            m.w,
            m.random_disjunction.disjuncts[0],
        )

    def test_untransformed_arcs(self):
        ct.check_untransformed_network_raises_GDPError(self, 'hull')


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

        transBlock = m.disj1.transformation_block
        # Just make sure exactly the expected number of constraints are here and
        # that they are mapped to the correct original components.
        self.assertEqual(len(transBlock.component_map(Constraint)), 3)
        self.assertIs(
            hull.get_transformed_constraints(m.disj1.b.any_index['local'])[
                0
            ].parent_block(),
            transBlock,
        )
        self.assertIs(
            hull.get_transformed_constraints(m.disj1.b.any_index['nonlin-ub'])[
                0
            ].parent_block(),
            transBlock,
        )
        self.assertIs(
            hull.get_transformed_constraints(m.disj1.component('b.any_index'))[
                0
            ].parent_block(),
            transBlock,
        )

    def test_local_var_handled_correctly(self):
        m = self.makeModel()

        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        # test the local variable was handled correctly.
        self.assertIs(hull.get_disaggregated_var(m.x, m.disj1), m.x)
        self.assertEqual(m.x.lb, 0)
        self.assertEqual(m.x.ub, 5)
        self.assertIsNone(m.disj1.transformation_block.disaggregatedVars.component("x"))
        self.assertIsInstance(
            m.disj1.transformation_block.disaggregatedVars.component("y"), Var
        )

    # this doesn't require the block, I'm just coopting this test to make sure
    # of some nonlinear expressions.
    def test_transformed_constraints(self):
        m = self.makeModel()

        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        # test the transformed nonlinear constraints
        nonlin_ub_list = hull.get_transformed_constraints(
            m.disj1.b.any_index['nonlin-ub']
        )
        self.assertEqual(len(nonlin_ub_list), 1)
        cons = nonlin_ub_list[0]
        self.assertIs(cons.ctype, Constraint)
        self.assertIsNone(cons.lower)
        self.assertEqual(value(cons.upper), 0)
        repn = generate_standard_repn(cons.body)
        self.assertEqual(
            str(repn.nonlinear_expr),
            "(0.9999*disj1.binary_indicator_var + 0.0001)*"
            "(_pyomo_gdp_hull_reformulation.relaxedDisjuncts[0]."
            "disaggregatedVars.y/"
            "(0.9999*disj1.binary_indicator_var + 0.0001))**2",
        )
        self.assertEqual(len(repn.nonlinear_vars), 2)
        self.assertIs(repn.nonlinear_vars[0], m.disj1.binary_indicator_var)
        self.assertIs(repn.nonlinear_vars[1], hull.get_disaggregated_var(m.y, m.disj1))
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], m.disj1.binary_indicator_var)
        self.assertEqual(repn.linear_coefs[0], -4)

        nonlin_lb_list = hull.get_transformed_constraints(m.disj2.non_lin_lb)
        self.assertEqual(len(nonlin_lb_list), 1)
        cons = nonlin_lb_list[0]
        self.assertIs(cons.ctype, Constraint)
        self.assertIsNone(cons.lower)
        self.assertEqual(value(cons.upper), 0)
        repn = generate_standard_repn(cons.body)
        self.assertEqual(
            str(repn.nonlinear_expr),
            "- ((0.9999*disj2.binary_indicator_var + 0.0001)*"
            "log(1 + "
            "_pyomo_gdp_hull_reformulation.relaxedDisjuncts[1]."
            "disaggregatedVars.y/"
            "(0.9999*disj2.binary_indicator_var + 0.0001)))",
        )
        self.assertEqual(len(repn.nonlinear_vars), 2)
        self.assertIs(repn.nonlinear_vars[0], m.disj2.binary_indicator_var)
        self.assertIs(repn.nonlinear_vars[1], hull.get_disaggregated_var(m.y, m.disj2))
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], m.disj2.binary_indicator_var)
        self.assertEqual(repn.linear_coefs[0], 1)


class DisaggregatingFixedVars(unittest.TestCase):
    def test_disaggregate_fixed_variables(self):
        m = models.makeTwoTermDisj()
        m.x.fix(6)
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        # check that we did indeed disaggregate x
        transBlock = m.d[1].transformation_block
        self.assertIsInstance(transBlock.disaggregatedVars.component("x"), Var)
        self.assertIs(
            hull.get_disaggregated_var(m.x, m.d[1]), transBlock.disaggregatedVars.x
        )
        self.assertIs(hull.get_src_var(transBlock.disaggregatedVars.x), m.x)

    def test_do_not_disaggregate_fixed_variables(self):
        m = models.makeTwoTermDisj()
        m.x.fix(6)
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m, assume_fixed_vars_permanent=True)
        # check that we didn't disaggregate x
        transBlock = m.d[1].transformation_block
        self.assertIsNone(transBlock.disaggregatedVars.component("x"))


class NameDeprecationTest(unittest.TestCase):
    def test_name_deprecated(self):
        m = models.makeTwoTermDisj()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.gdp', logging.WARNING):
            TransformationFactory('gdp.chull').apply_to(m)
        self.assertIn(
            "DEPRECATED: The 'gdp.chull' name is deprecated. "
            "Please use the more apt 'gdp.hull' instead.",
            output.getvalue().replace('\n', ' '),
        )

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
    @unittest.skipIf('gurobi' not in linear_solvers, "Gurobi solver not available")
    def test_optimal_soln_feasible(self):
        m = ConcreteModel()
        m.Points = RangeSet(3)
        m.Centroids = RangeSet(2)

        m.X = Param(m.Points, initialize={1: 0.3672, 2: 0.8043, 3: 0.3059})

        m.cluster_center = Var(m.Centroids, bounds=(0, 2))
        m.distance = Var(m.Points, bounds=(0, 2))
        m.t = Var(m.Points, m.Centroids, bounds=(0, 2))

        @m.Disjunct(m.Points, m.Centroids)
        def AssignPoint(d, i, k):
            m = d.model()
            d.LocalVars = Suffix(direction=Suffix.LOCAL)
            d.LocalVars[d] = [m.t[i, k]]

            def distance1(d):
                return m.t[i, k] >= m.X[i] - m.cluster_center[k]

            def distance2(d):
                return m.t[i, k] >= -(m.X[i] - m.cluster_center[k])

            d.dist1 = Constraint(rule=distance1)
            d.dist2 = Constraint(rule=distance2)
            d.define_distance = Constraint(expr=m.distance[i] == m.t[i, k])

        @m.Disjunction(m.Points)
        def OneCentroidPerPt(m, i):
            return [m.AssignPoint[i, k] for k in m.Centroids]

        m.obj = Objective(expr=sum(m.distance[i] for i in m.Points))

        TransformationFactory('gdp.hull').apply_to(m)

        # fix an optimal solution
        m.AssignPoint[1, 1].indicator_var.fix(True)
        m.AssignPoint[1, 2].indicator_var.fix(False)
        m.AssignPoint[2, 1].indicator_var.fix(False)
        m.AssignPoint[2, 2].indicator_var.fix(True)
        m.AssignPoint[3, 1].indicator_var.fix(True)
        m.AssignPoint[3, 2].indicator_var.fix(False)

        m.cluster_center[1].fix(0.3059)
        m.cluster_center[2].fix(0.8043)

        m.distance[1].fix(0.0613)
        m.distance[2].fix(0)
        m.distance[3].fix(0)

        m.t[1, 1].fix(0.0613)
        m.t[1, 2].fix(0)
        m.t[2, 1].fix(0)
        m.t[2, 2].fix(0)
        m.t[3, 1].fix(0)
        m.t[3, 2].fix(0)

        results = SolverFactory('gurobi').solve(m)

        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )

        TOL = 1e-8
        for c in m.component_data_objects(Constraint, active=True):
            if c.lower is not None:
                self.assertGreaterEqual(value(c.body) + TOL, value(c.lower))
            if c.upper is not None:
                self.assertLessEqual(value(c.body) - TOL, value(c.upper))


class NetworkDisjuncts(unittest.TestCase, CommonTests):
    @unittest.skipIf(not ct.linear_solvers, "No linear solver available")
    def test_solution_maximize(self):
        ct.check_network_disjuncts(self, minimize=False, transformation='hull')

    @unittest.skipIf(not ct.linear_solvers, "No linear solver available")
    def test_solution_minimize(self):
        ct.check_network_disjuncts(self, minimize=True, transformation='hull')


class LogicalConstraintsOnDisjuncts(unittest.TestCase):
    def test_logical_constraints_transformed(self):
        m = models.makeLogicalConstraintsOnDisjuncts()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)

        y1 = m.Y[1].get_associated_binary()
        y2 = m.Y[2].get_associated_binary()

        # check the bigm transformation of the logical things on the disjuncts

        # first d[1]:
        cons = hull.get_transformed_constraints(
            m.d[1]._logical_to_disjunctive.transformed_constraints[1]
        )
        dis_z1 = hull.get_disaggregated_var(
            m.d[1]._logical_to_disjunctive.auxiliary_vars[1], m.d[1]
        )
        dis_y1 = hull.get_disaggregated_var(y1, m.d[1])

        self.assertEqual(len(cons), 1)
        # this simplifies because the dissaggregated variable is *always* 0
        c = cons[0]
        # hull transformation of z1 = 1 - y1:
        # dis_z1 + dis_y1 = d[1].ind_var
        self.assertEqual(c.lower, 0)
        self.assertEqual(c.upper, 0)
        repn = generate_standard_repn(c.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self, simplified, dis_z1 + dis_y1 - m.d[1].binary_indicator_var
        )

        cons = hull.get_transformed_constraints(
            m.d[1]._logical_to_disjunctive.transformed_constraints[2]
        )
        self.assertEqual(len(cons), 1)
        c = cons[0]
        # hull transformation of z1 >= 1
        assertExpressionsStructurallyEqual(
            self,
            c.expr,
            dis_z1 - (1 - m.d[1].binary_indicator_var) * 0
            >= m.d[1].binary_indicator_var,
        )

        # then d[4]:
        y1d = hull.get_disaggregated_var(y1, m.d[4])
        y2d = hull.get_disaggregated_var(y2, m.d[4])
        z1d = hull.get_disaggregated_var(
            m.d[4]._logical_to_disjunctive.auxiliary_vars[1], m.d[4]
        )
        z2d = hull.get_disaggregated_var(
            m.d[4]._logical_to_disjunctive.auxiliary_vars[2], m.d[4]
        )
        z3d = hull.get_disaggregated_var(
            m.d[4]._logical_to_disjunctive.auxiliary_vars[3], m.d[4]
        )

        # hull transformation of (1 - z1) + (1 - y1) + y2 >= 1:
        # dz1 + dy1 - dy2 <= m.d[4].ind_var
        cons = hull.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[1]
        )
        # these also are simple because it's really an equality, and since both
        # disaggregated variables will be 0 when the disjunct isn't selected, it
        # doesn't even need big-Ming.
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self, simplified, -m.d[4].binary_indicator_var + z1d + y1d - y2d
        )

        # hull transformation of z1 + 1 - (1 - y1) >= 1
        # -y1d - z1d <= -d[4].ind_var
        cons = hull.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[2]
        )
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self, simplified, m.d[4].binary_indicator_var - y1d - z1d
        )

        # hull transformation of z1 + (1 - y2) >= 1
        # y2d - z1d <= 0
        cons = hull.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[3]
        )
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(self, simplified, y2d - z1d)

        # hull transformation of (1 - z2) + y1 + (1 - y2) >= 1
        # z2d - y1d + y2d <= m.d[4].ind_var
        cons = hull.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[4]
        )
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self, simplified, -m.d[4].binary_indicator_var + z2d + y2d - y1d
        )

        # hull transformation of z2 + (1 - y1) >= 1
        # y1d - z2d <= 0
        cons = hull.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[5]
        )
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(self, simplified, y1d - z2d)

        # hull transformation of z2 + 1 - (1 - y2) >= 1
        # -y2d - z2d <= -d[4].ind_var
        cons = hull.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[6]
        )
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self, simplified, m.d[4].binary_indicator_var - y2d - z2d
        )

        # hull transformation of z3 <= z1
        # z3d - z1d <= 0
        cons = hull.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[7]
        )
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(self, simplified, z3d - z1d)

        # hull transformation of z3 <= z2
        # z3d - z2d <= 0
        cons = hull.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[8]
        )
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(self, simplified, z3d - z2d)

        # hull transformation of 1 - z3 <= 2 - (z1 + z2)
        cons = hull.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[9]
        )
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        assertExpressionsStructurallyEqual(
            self,
            cons.expr,
            1 - z3d - (2 - (z1d + z2d)) - (1 - m.d[4].binary_indicator_var) * (-1)
            <= 0 * m.d[4].binary_indicator_var,
        )

        # hull transformation of z3 >= 1
        cons = hull.get_transformed_constraints(
            m.d[4]._logical_to_disjunctive.transformed_constraints[10]
        )
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        assertExpressionsStructurallyEqual(
            self,
            cons.expr,
            z3d - (1 - m.d[4].binary_indicator_var) * 0 >= m.d[4].binary_indicator_var,
        )

        self.assertFalse(m.bwahaha.active)
        self.assertFalse(m.p.active)

    @unittest.skipIf(not ct.linear_solvers, "No linear solver available")
    def test_solution_obeys_logical_constraints(self):
        m = models.makeLogicalConstraintsOnDisjuncts()
        ct.check_solution_obeys_logical_constraints(self, 'hull', m)

    @unittest.skipIf(not ct.linear_solvers, "No linear solver available")
    def test_boolean_vars_on_disjunct(self):
        # Just to make sure we do everything in the correct order, make sure
        # that we can solve a model where some BooleanVars were declared on one
        # of the Disjuncts
        m = models.makeBooleanVarsOnDisjuncts()
        ct.check_solution_obeys_logical_constraints(self, 'hull', m)

    def test_pickle(self):
        ct.check_transformed_model_pickles(self, 'hull')

    @unittest.skipIf(not dill_available, "Dill is not available")
    def test_dill_pickle(self):
        try:
            # As of Nov 2024, this test needs a larger recursion limit
            # due to the various references among the modeling objects
            # 1385 is sufficient locally, but not always on GHA.
            rl = sys.getrecursionlimit()
            sys.setrecursionlimit(max(1500, rl))
            ct.check_transformed_model_pickles_with_dill(self, 'hull')
        finally:
            sys.setrecursionlimit(rl)


@unittest.skipUnless(gurobi_available, "Gurobi is not available")
class NestedDisjunctsInFlatGDP(unittest.TestCase):
    """
    This class tests the fix for #2702
    """

    def test_declare_disjuncts_in_disjunction_rule(self):
        ct.check_nested_disjuncts_in_flat_gdp(self, 'hull')
