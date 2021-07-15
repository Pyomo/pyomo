#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.environ import (TransformationFactory, Constraint, ConcreteModel,
                           Var, RangeSet, Objective, maximize, SolverFactory)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core import Block, value
import pyomo.gdp.tests.common_tests as ct
from pyomo.repn import generate_standard_repn
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.opt import check_available_solvers

from nose.tools import set_trace

solvers = check_available_solvers('gurobi')

class CommonTests:
    def diff_apply_to_and_create_using(self, model):
        ct.diff_apply_to_and_create_using(self, model, 'gdp.between_steps')

class PaperTwoCircleExample(unittest.TestCase, CommonTests):
    def makeModel(self):
        m = ConcreteModel()
        m.I = RangeSet(1,4)
        m.x = Var(m.I, bounds=(-2,6))

        m.disjunction = Disjunction(expr=[[sum(m.x[i]**2 for i in m.I) <= 1],
                                          [sum((3 - m.x[i])**2 for i in m.I) <=
                                           1]])

        m.obj = Objective(expr=m.x[2] - m.x[1], sense=maximize)

        return m

    def check_transformation_block(self, m, aux1lb, aux1ub, aux2lb, aux2ub):
        b = m.component("_pyomo_gdp_between_steps_reformulation")
        self.assertIsInstance(b, Block)

        # check we declared the right things
        self.assertTrue(len(b.component_map(Disjunction)), 1)
        self.assertTrue(len(b.component_map(Disjunct)), 2)
        self.assertTrue(len(b.component_map(Constraint)), 2) # global constraints
        
        disjunction = b.disjunction
        self.assertTrue(len(disjunction.disjuncts), 2)
        # each Disjunct has one constraint
        disj1 = disjunction.disjuncts[0]
        disj2 = disjunction.disjuncts[1]
        self.assertTrue(len(disj1.component_map(Constraint)), 1)
        self.assertTrue(len(disj2.component_map(Constraint)), 1)
        # each Disjunct has two variables declared on it (aux vars and indicator
        # var)
        self.assertTrue(len(disj1.component_map(Var)), 2)
        self.assertTrue(len(disj2.component_map(Var)), 2)


        aux_vars1 = disj1.component(
            "disjunction_disjuncts[0].constraint[1]_aux_vars")
        self.assertEqual(len(aux_vars1), 2)
        self.assertEqual(aux_vars1[0].lb, aux1lb)
        self.assertEqual(aux_vars1[0].ub, aux1ub)
        self.assertEqual(aux_vars1[1].lb, aux1lb)
        self.assertEqual(aux_vars1[1].ub, aux1ub)
        aux_vars2 = disj2.component(
            "disjunction_disjuncts[1].constraint[1]_aux_vars")
        self.assertEqual(len(aux_vars2), 2)
        self.assertEqual(aux_vars2[0].lb, aux2lb)
        self.assertEqual(aux_vars2[0].ub, aux2ub)
        self.assertEqual(aux_vars2[1].lb, aux2lb)
        self.assertEqual(aux_vars2[1].ub, aux2ub)

        # check the constraints on the disjuncts
        c = disj1.component("disjunction_disjuncts[0].constraint[1]")
        self.assertEqual(len(c), 1)
        c1 = c[0]
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 1)
        repn = generate_standard_repn(c1.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, 0)
        self.assertIs(repn.linear_vars[0], aux_vars1[0])
        self.assertIs(repn.linear_vars[1], aux_vars1[1])
        self.assertEqual(repn.linear_coefs[0], 1)        
        self.assertEqual(repn.linear_coefs[1], 1)

        c = disj2.component("disjunction_disjuncts[1].constraint[1]")
        self.assertEqual(len(c), 1)
        c2 = c[0]
        self.assertIsNone(c2.lower)
        self.assertEqual(value(c2.upper), -35)
        repn = generate_standard_repn(c2.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, 0)
        self.assertIs(repn.linear_vars[0], aux_vars2[0])
        self.assertIs(repn.linear_vars[1], aux_vars2[1])
        self.assertEqual(repn.linear_coefs[0], 1)        
        self.assertEqual(repn.linear_coefs[1], 1)

        # check the global constraints
        c = b.component(
            "disjunction_disjuncts[0].constraint[1]_split_constraints")
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 0)
        repn = generate_standard_repn(c1.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(len(repn.quadratic_vars), 2)
        self.assertEqual(repn.linear_coefs[0], -1)
        self.assertIs(repn.linear_vars[0], aux_vars1[0])
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertEqual(repn.quadratic_coefs[1], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[1])
        self.assertIs(repn.quadratic_vars[0][1], m.x[1])
        self.assertIs(repn.quadratic_vars[1][0], m.x[2])
        self.assertIs(repn.quadratic_vars[1][1], m.x[2])
        self.assertIsNone(repn.nonlinear_expr)

        c2 = c[1]
        self.assertIsNone(c2.lower)
        self.assertEqual(c2.upper, 0)
        repn = generate_standard_repn(c2.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(len(repn.quadratic_vars), 2)
        self.assertIs(repn.linear_vars[0], aux_vars1[1])
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertEqual(repn.quadratic_coefs[1], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[3])
        self.assertIs(repn.quadratic_vars[0][1], m.x[3])
        self.assertIs(repn.quadratic_vars[1][0], m.x[4])
        self.assertIs(repn.quadratic_vars[1][1], m.x[4])
        self.assertIsNone(repn.nonlinear_expr)

    def test_transformation_block_fbbt_bounds(self):
        m = self.makeModel()

        TransformationFactory('gdp.between_steps').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            compute_bounds_method=compute_bounds_on_expr)

        self.check_transformation_block(m, 0, 72, -72, 96)

    @unittest.skipIf('gurobi' not in solvers, 'Gurobi solver not available')
    def test_transformation_block_optimized_bounds(self):
        m = self.makeModel()

        # I'm using Gurobi because I'm assuming exact equality is going to work
        # out. And it definitely won't with ipopt.
        TransformationFactory('gdp.between_steps').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            subproblem_solver=SolverFactory('gurobi'))
        
        self.check_transformation_block(m, 0, 72, -18, 32)

    # def test_transformation_block_specified_bounds(self):
    #     m = self.makeModel()

    #     TransformationFactory('gdp.between_steps').apply_to(
    #         m,
    #         variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
    #         # ESJ: Shoot, I don't actually know how to do this... Because it
    #         # matters where the constant ends up. Because what I would write
    #         # down is the below. But this is wrong since the expression for the
    #         # constraint in the second Disjunct gets expanded and the constant
    #         # moved to the RHS.
    #         bounds={m.disjunction.disjuncts[0].constraint[1]: [(0,32), (0,32)],
    #                 m.disjunction.disjuncts[1].constraint[1]: [(0,32), (0,32)]})

    def test_transformation_block_arbitrary_partition(self):
        pass

class NonQuadraticNonlinear(unittest.TestCase, CommonTests):
    def makeModel(self):
        m = ConcreteModel()
        m.I = RangeSet(1,4)
        m.I1 = RangeSet(1,2)
        m.I2 = RangeSet(3,4)
        m.x = Var(m.I, bounds=(-2,6))

        # sum of 4-norms...
        m.disjunction = Disjunction(
            expr=[[sum(m.x[i]**4 for i in m.I1)**(1/4) + \
                   sum(m.x[i]**4 for i in m.I2)**(1/4) <= 1],
                  [sum((3 - m.x[i])**4 for i in m.I1)**(1/4) +
                   sum((3 - m.x[i])**4 for i in m.I2)**(1/4) <= 1]])

        m.obj = Objective(expr=m.x[2] - m.x[1], sense=maximize)

        return m

    def test_transformation_block_fbbt_bounds(self):
        m = self.makeModel()

        TransformationFactory('gdp.between_steps').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            compute_bounds_method=compute_bounds_on_expr)

        
