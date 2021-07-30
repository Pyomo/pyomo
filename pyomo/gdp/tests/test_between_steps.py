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
from pyomo.gdp.util import GDP_Error
from pyomo.gdp.plugins.between_steps import arbitrary_partition
from pyomo.core import Block, value
from pyomo.core.expr import current as EXPR
import pyomo.gdp.tests.common_tests as ct
from pyomo.repn import generate_standard_repn
from pyomo.opt import check_available_solvers

from nose.tools import set_trace

solvers = check_available_solvers('gurobi_direct')

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

    def check_disj_constraint(self, c1, upper, auxVar1, auxVar2):
        self.assertIsNone(c1.lower)
        c1.pprint()
        self.assertEqual(value(c1.upper), upper)
        repn = generate_standard_repn(c1.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, 0)
        self.assertIs(repn.linear_vars[0], auxVar1)
        self.assertIs(repn.linear_vars[1], auxVar2)
        self.assertEqual(repn.linear_coefs[0], 1)        
        self.assertEqual(repn.linear_coefs[1], 1)

    def check_global_constraint_disj1(self, c1, auxVar, var1, var2):
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 0)
        repn = generate_standard_repn(c1.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(len(repn.quadratic_vars), 2)
        self.assertEqual(repn.linear_coefs[0], -1)
        self.assertIs(repn.linear_vars[0], auxVar)
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertEqual(repn.quadratic_coefs[1], 1)
        self.assertIs(repn.quadratic_vars[0][0], var1)
        self.assertIs(repn.quadratic_vars[0][1], var1)
        self.assertIs(repn.quadratic_vars[1][0], var2)
        self.assertIs(repn.quadratic_vars[1][1], var2)
        self.assertIsNone(repn.nonlinear_expr)

    def check_global_constraint_disj2(self, c1, auxVar, var1, var2):
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 0)
        repn = generate_standard_repn(c1.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 3)
        self.assertEqual(len(repn.quadratic_vars), 2)
        self.assertEqual(repn.linear_coefs[0], -6)
        self.assertEqual(repn.linear_coefs[1], -6)
        self.assertEqual(repn.linear_coefs[2], -1)
        self.assertIs(repn.linear_vars[0], var1)
        self.assertIs(repn.linear_vars[1], var2)
        self.assertIs(repn.linear_vars[2], auxVar)
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertEqual(repn.quadratic_coefs[1], 1)
        self.assertIs(repn.quadratic_vars[0][0], var1)
        self.assertIs(repn.quadratic_vars[0][1], var1)
        self.assertIs(repn.quadratic_vars[1][0], var2)
        self.assertIs(repn.quadratic_vars[1][1], var2)
        self.assertIsNone(repn.nonlinear_expr)

    def check_transformation_block_structure(self, m, aux11lb, aux11ub, aux12lb,
                                             aux12ub, aux21lb, aux21ub, aux22lb,
                                             aux22ub):
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
        # TODO: gurobi default constraint tolerance is 1e-6, so let's say that's
        # our goal too. Have to tighten Gurobi's tolerance to even get here
        # though... And are we okay with LBs being too high rather than too low,
        # and vice versa for UB?
        self.assertAlmostEqual(aux_vars1[0].lb, aux11lb, places=6)
        self.assertAlmostEqual(aux_vars1[0].ub, aux11ub, places=6)
        self.assertAlmostEqual(aux_vars1[1].lb, aux12lb, places=6)
        self.assertAlmostEqual(aux_vars1[1].ub, aux12ub, places=6)
        aux_vars2 = disj2.component(
            "disjunction_disjuncts[1].constraint[1]_aux_vars")
        self.assertAlmostEqual(len(aux_vars2), 2)
        self.assertAlmostEqual(aux_vars2[0].lb, aux21lb, places=6)
        self.assertAlmostEqual(aux_vars2[0].ub, aux21ub, places=6)
        self.assertAlmostEqual(aux_vars2[1].lb, aux22lb, places=6)
        self.assertAlmostEqual(aux_vars2[1].ub, aux22ub, places=6)

        return b, disj1, disj2, aux_vars1, aux_vars2

    def check_disjunct_constraints(self, disj1, disj2, aux_vars1, aux_vars2):
        c = disj1.component("disjunction_disjuncts[0].constraint[1]")
        self.assertEqual(len(c), 1)
        c1 = c[0]
        self.check_disj_constraint(c1, 1, aux_vars1[0], aux_vars1[1])
        c = disj2.component("disjunction_disjuncts[1].constraint[1]")
        self.assertEqual(len(c), 1)
        c2 = c[0]
        self.check_disj_constraint(c2, -35, aux_vars2[0], aux_vars2[1])

    def check_transformation_block(self, m, aux11lb, aux11ub, aux12lb, aux12ub,
                                   aux21lb, aux21ub, aux22lb, aux22ub):
        (b, disj1, disj2, 
         aux_vars1, 
         aux_vars2) = self.check_transformation_block_structure(m, aux11lb,
                                                                aux11ub,
                                                                aux12lb,
                                                                aux12ub,
                                                                aux21lb,
                                                                aux21ub,
                                                                aux22lb,
                                                                aux22ub)

        self.check_disjunct_constraints(disj1, disj2, aux_vars1, aux_vars2)

        # check the global constraints
        c = b.component(
            "disjunction_disjuncts[0].constraint[1]_split_constraints")
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj1(c1, aux_vars1[0], m.x[1], m.x[2])
        c2 = c[1]
        self.check_global_constraint_disj1(c2, aux_vars1[1], m.x[3], m.x[4])

        c = b.component(
            "disjunction_disjuncts[1].constraint[1]_split_constraints")
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj2(c1, aux_vars2[0], m.x[1], m.x[2])
        c2 = c[1]
        self.check_global_constraint_disj2(c2, aux_vars2[1], m.x[3], m.x[4])

    def test_transformation_block_fbbt_bounds(self):
        m = self.makeModel()

        TransformationFactory('gdp.between_steps').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            compute_bounds_method='fbbt')

        self.check_transformation_block(m, 0, 72, 0, 72, -72, 96, -72, 96)

    @unittest.skipIf('gurobi_direct' not in solvers, 
                     'Gurobi direct solver not available')
    def test_transformation_block_optimized_bounds(self):
        m = self.makeModel()

        # I'm using Gurobi because I'm assuming exact equality is going to work
        # out. And it definitely won't with ipopt. (And Gurobi direct is way
        # faster than the LP interfeace to Gurobi for this... I assume because
        # writing nonlinear expressions is slow?)
        TransformationFactory('gdp.between_steps').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            subproblem_solver=SolverFactory('gurobi_direct'))
        
        self.check_transformation_block(m, 0, 72, 0, 72, -18, 32, -18, 32)

    def test_transformation_block_better_bounds_in_global_constraints(self):
        m = self.makeModel()
        m.c1 = Constraint(expr=m.x[1]**2 + m.x[2]**2 <= 32)
        m.c2 = Constraint(expr=m.x[3]**2 + m.x[4]**2 <= 32)
        m.c3 = Constraint(expr=(3 - m.x[1])**2 + (3 - m.x[2])**2 <= 32)
        m.c4 = Constraint(expr=(3 - m.x[3])**2 + (3 - m.x[4])**2 <= 32)
        opt = SolverFactory('gurobi_direct')
        opt.options['NonConvex'] = 2
        opt.options['FeasibilityTol'] = 1e-8

        TransformationFactory('gdp.between_steps').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            subproblem_solver=opt)

        self.check_transformation_block(m, 0, 32, 0, 32, -18, 14, -18, 14)

    @unittest.skipIf('gurobi_direct' not in solvers, 
                     'Gurobi direct solver not available')
    def test_transformation_block_arbitrary_even_partition(self):
        m = self.makeModel()
        
        # I'm using Gurobi because I'm assuming exact equality is going to work
        # out. And it definitely won't with ipopt. (And Gurobi direct is way
        # faster than the LP interface to Gurobi for this... I assume because
        # writing nonlinear expressions is slow?)
        TransformationFactory('gdp.between_steps').apply_to(
            m,
            P=2,
            subproblem_solver=SolverFactory('gurobi_direct'))
        
        self.check_transformation_block(m, 0, 72, 0, 72, -18, 32, -18, 32)

    @unittest.skipIf('gurobi_direct' not in solvers, 
                     'Gurobi direct solver not available')
    def test_assume_fixed_vars_permanent(self):
        m = self.makeModel()
        m.x[1].fix(0)

        # I'm using Gurobi because I'm assuming exact equality is going to work
        # out. And it definitely won't with ipopt. (And Gurobi direct is way
        # faster than the LP interface to Gurobi for this... I assume because
        # writing nonlinear expressions is slow?)
        TransformationFactory('gdp.between_steps').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            assume_fixed_vars_permanent=True,
            subproblem_solver=SolverFactory('gurobi_direct'))

        # This actually changes the structure of the model because fixed vars
        # move to the constants. I think this is fair, and we should allow it
        # because it will allow for a tighter relaxation.
        (b, disj1, disj2, 
         aux_vars1, 
         aux_vars2) = self.check_transformation_block_structure(m, 0, 36, 0, 72,
                                                                -9, 16, -18, 32)

        # check disjunct constraints
        self.check_disjunct_constraints(disj1, disj2, aux_vars1, aux_vars2)

        # now we can check the global constraints--these are what is different
        # because x[1] is gone.
        c = b.component(
            "disjunction_disjuncts[0].constraint[1]_split_constraints")
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 0)
        repn = generate_standard_repn(c1.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertEqual(repn.linear_coefs[0], -1)
        self.assertIs(repn.linear_vars[0], aux_vars1[0])
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[2])
        self.assertIs(repn.quadratic_vars[0][1], m.x[2])
        self.assertIsNone(repn.nonlinear_expr)
        c2 = c[1]
        self.check_global_constraint_disj1(c2, aux_vars1[1], m.x[3], m.x[4])

        c = b.component(
            "disjunction_disjuncts[1].constraint[1]_split_constraints")
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 0)
        repn = generate_standard_repn(c1.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertEqual(repn.linear_coefs[0], -6)
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertIs(repn.linear_vars[0], m.x[2])
        self.assertIs(repn.linear_vars[1], aux_vars2[0])
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[2])
        self.assertIs(repn.quadratic_vars[0][1], m.x[2])
        self.assertIsNone(repn.nonlinear_expr)
        c2 = c[1]
        self.check_global_constraint_disj2(c2, aux_vars2[1], m.x[3], m.x[4])

    @unittest.skipIf('gurobi_direct' not in solvers, 
                     'Gurobi direct solver not available')
    def test_transformation_block_arbitrary_odd_partition(self):
        m = self.makeModel()

        # I'm using Gurobi because I'm assuming exact equality is going to work
        # out. And it definitely won't with ipopt. (And Gurobi direct is way
        # faster than the LP interface to Gurobi for this... I assume because
        # writing nonlinear expressions is slow?)
        TransformationFactory('gdp.between_steps').apply_to(
            m,
            P=3,
            subproblem_solver=SolverFactory('gurobi_direct'))
        
        b = m.component("_pyomo_gdp_between_steps_reformulation")
        self.assertIsInstance(b, Block)

        # check we declared the right things
        self.assertTrue(len(b.component_map(Disjunction)), 1)
        self.assertTrue(len(b.component_map(Disjunct)), 2)
        self.assertTrue(len(b.component_map(Constraint)), 3) # global constraints
        
        disjunction = b.disjunction
        self.assertTrue(len(disjunction.disjuncts), 2)
        # each Disjunct has one constraint
        disj1 = disjunction.disjuncts[0]
        disj2 = disjunction.disjuncts[1]
        self.assertTrue(len(disj1.component_map(Constraint)), 1)
        self.assertTrue(len(disj2.component_map(Constraint)), 1)
        # each Disjunct has three variables declared on it (aux vars and
        # indicator var)
        self.assertTrue(len(disj1.component_map(Var)), 3)
        self.assertTrue(len(disj2.component_map(Var)), 3)

        aux_vars1 = disj1.component(
            "disjunction_disjuncts[0].constraint[1]_aux_vars")
        self.assertEqual(len(aux_vars1), 3)
        self.assertEqual(aux_vars1[0].lb, 0)
        self.assertEqual(aux_vars1[0].ub, 72)
        self.assertEqual(aux_vars1[1].lb, 0)
        self.assertEqual(aux_vars1[1].ub, 36)
        self.assertEqual(aux_vars1[2].lb, 0)
        self.assertEqual(aux_vars1[2].ub, 36)
        aux_vars2 = disj2.component(
            "disjunction_disjuncts[1].constraint[1]_aux_vars")
        self.assertEqual(len(aux_vars2), 3)
        # min and max of x1^2 - 6x1 + x2^2 - 6x2
        self.assertEqual(aux_vars2[0].lb, -18)
        self.assertEqual(aux_vars2[0].ub, 32)
        # min and max of x2^2 - 6x2
        self.assertEqual(aux_vars2[1].lb, -9)
        self.assertEqual(aux_vars2[1].ub, 16)
        self.assertEqual(aux_vars2[2].lb, -9)
        self.assertEqual(aux_vars2[2].ub, 16)

        # check the constraints on the disjuncts
        c = disj1.component("disjunction_disjuncts[0].constraint[1]")
        self.assertEqual(len(c), 1)
        c1 = c[0]
        self.assertIsNone(c1.lower)
        self.assertEqual(value(c1.upper), 1)
        repn = generate_standard_repn(c1.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 3)
        self.assertEqual(repn.constant, 0)
        self.assertIs(repn.linear_vars[0], aux_vars1[0])
        self.assertIs(repn.linear_vars[1], aux_vars1[1])
        self.assertIs(repn.linear_vars[2], aux_vars1[2])
        self.assertEqual(repn.linear_coefs[0], 1)        
        self.assertEqual(repn.linear_coefs[1], 1)
        self.assertEqual(repn.linear_coefs[2], 1)

        c = disj2.component("disjunction_disjuncts[1].constraint[1]")
        self.assertEqual(len(c), 1)
        c2 = c[0]
        self.assertIsNone(c2.lower)
        self.assertEqual(value(c2.upper), -35)
        repn = generate_standard_repn(c2.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 3)
        self.assertEqual(repn.constant, 0)
        self.assertIs(repn.linear_vars[0], aux_vars2[0])
        self.assertIs(repn.linear_vars[1], aux_vars2[1])
        self.assertIs(repn.linear_vars[2], aux_vars2[2])
        self.assertEqual(repn.linear_coefs[0], 1)        
        self.assertEqual(repn.linear_coefs[1], 1)
        self.assertEqual(repn.linear_coefs[2], 1)

        # check the global constraints
        c = b.component(
            "disjunction_disjuncts[0].constraint[1]_split_constraints")
        self.assertEqual(len(c), 3)
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
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.linear_vars[0], aux_vars1[1])
        self.assertEqual(repn.linear_coefs[0], -1)
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[3])
        self.assertIs(repn.quadratic_vars[0][1], m.x[3])

        c3 = c[2]
        self.assertIsNone(c3.lower)
        self.assertEqual(c3.upper, 0)
        repn = generate_standard_repn(c3.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.linear_vars[0], aux_vars1[2])
        self.assertEqual(repn.linear_coefs[0], -1)
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[4])
        self.assertIs(repn.quadratic_vars[0][1], m.x[4])
        self.assertIsNone(repn.nonlinear_expr)

        c = b.component(
            "disjunction_disjuncts[1].constraint[1]_split_constraints")
        self.assertEqual(len(c), 3)
        c1 = c[0]
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 0)
        repn = generate_standard_repn(c1.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 3)
        self.assertEqual(len(repn.quadratic_vars), 2)
        self.assertEqual(repn.linear_coefs[0], -6)
        self.assertEqual(repn.linear_coefs[1], -6)
        self.assertEqual(repn.linear_coefs[2], -1)
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertEqual(repn.quadratic_coefs[1], 1)
        self.assertIs(repn.linear_vars[2], aux_vars2[0])
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
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.linear_vars[0], m.x[3])
        self.assertIs(repn.linear_vars[1], aux_vars2[1])
        self.assertEqual(repn.linear_coefs[0], -6)
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[3])
        self.assertIs(repn.quadratic_vars[0][1], m.x[3])

        c3 = c[2]
        self.assertIsNone(c3.lower)
        self.assertEqual(c3.upper, 0)
        repn = generate_standard_repn(c3.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.linear_vars[0], m.x[4])
        self.assertIs(repn.linear_vars[1], aux_vars2[2])
        self.assertEqual(repn.linear_coefs[0], -6)
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[4])
        self.assertIs(repn.quadratic_vars[0][1], m.x[4])

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
        self.assertEqual(value(c2.upper), 1)
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
        self.assertEqual(len(repn.nonlinear_vars), 2)
        self.assertEqual(repn.linear_coefs[0], -1)
        self.assertIs(repn.linear_vars[0], aux_vars1[0])
        self.assertIs(repn.nonlinear_vars[0], m.x[1])
        self.assertIs(repn.nonlinear_vars[1], m.x[2])
        self.assertIsInstance(repn.nonlinear_expr, EXPR.PowExpression)
        self.assertEqual(repn.nonlinear_expr.args[1], 0.25)
        self.assertIsInstance(repn.nonlinear_expr.args[0], EXPR.SumExpression)
        self.assertEqual(len(repn.nonlinear_expr.args[0].args), 2)
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[0],
                              EXPR.PowExpression)
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[1],
                              EXPR.PowExpression)
        self.assertIs(repn.nonlinear_expr.args[0].args[0].args[0], m.x[1])
        self.assertEqual(repn.nonlinear_expr.args[0].args[0].args[1], 4)
        self.assertIs(repn.nonlinear_expr.args[0].args[1].args[0], m.x[2])
        self.assertEqual(repn.nonlinear_expr.args[0].args[1].args[1], 4)

        c2 = c[1]
        self.assertIsNone(c2.lower)
        self.assertEqual(c2.upper, 0)
        repn = generate_standard_repn(c2.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(len(repn.nonlinear_vars), 2)
        self.assertIs(repn.linear_vars[0], aux_vars1[1])
        self.assertIs(repn.nonlinear_vars[0], m.x[3])
        self.assertIs(repn.nonlinear_vars[1], m.x[4])
        self.assertIsInstance(repn.nonlinear_expr, EXPR.PowExpression)
        self.assertEqual(repn.nonlinear_expr.args[1], 0.25)
        self.assertIsInstance(repn.nonlinear_expr.args[0], EXPR.SumExpression)
        self.assertEqual(len(repn.nonlinear_expr.args[0].args), 2)
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[0],
                              EXPR.PowExpression)
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[1],
                              EXPR.PowExpression)
        self.assertIs(repn.nonlinear_expr.args[0].args[0].args[0], m.x[3])
        self.assertEqual(repn.nonlinear_expr.args[0].args[0].args[1], 4)
        self.assertIs(repn.nonlinear_expr.args[0].args[1].args[0], m.x[4])
        self.assertEqual(repn.nonlinear_expr.args[0].args[1].args[1], 4)

        c = b.component(
            "disjunction_disjuncts[1].constraint[1]_split_constraints")
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 0)
        repn = generate_standard_repn(c1.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(len(repn.nonlinear_vars), 2)
        self.assertEqual(repn.linear_coefs[0], -1)
        self.assertIs(repn.linear_vars[0], aux_vars2[0])
        self.assertIs(repn.nonlinear_vars[0], m.x[1])
        self.assertIs(repn.nonlinear_vars[1], m.x[2])
        self.assertIsInstance(repn.nonlinear_expr, EXPR.PowExpression)
        self.assertEqual(repn.nonlinear_expr.args[1], 0.25)
        self.assertIsInstance(repn.nonlinear_expr.args[0], EXPR.SumExpression)
        self.assertEqual(len(repn.nonlinear_expr.args[0].args), 2)
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[0],
                              EXPR.PowExpression)
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[1],
                              EXPR.PowExpression)
        sum_expr = repn.nonlinear_expr.args[0].args[0].args[0]
        self.assertIsInstance(sum_expr, EXPR.SumExpression)
        sum_repn = generate_standard_repn(sum_expr)
        self.assertEqual(sum_repn.constant, 3)
        self.assertTrue(sum_repn.is_linear())
        self.assertEqual(len(sum_repn.linear_vars), 1)
        self.assertEqual(sum_repn.linear_coefs[0], -1)
        self.assertIs(sum_repn.linear_vars[0], m.x[1])
        self.assertEqual(repn.nonlinear_expr.args[0].args[0].args[1], 4)
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[1],
                              EXPR.PowExpression)
        sum_expr = repn.nonlinear_expr.args[0].args[1].args[0]
        self.assertIsInstance(sum_expr, EXPR.SumExpression)
        sum_repn = generate_standard_repn(sum_expr)
        self.assertEqual(sum_repn.constant, 3)
        self.assertTrue(sum_repn.is_linear())
        self.assertEqual(len(sum_repn.linear_vars), 1)
        self.assertEqual(sum_repn.linear_coefs[0], -1)
        self.assertIs(sum_repn.linear_vars[0], m.x[2])
        self.assertEqual(repn.nonlinear_expr.args[0].args[1].args[1], 4)

        c2 = c[1]
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 0)
        repn = generate_standard_repn(c2.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(len(repn.nonlinear_vars), 2)
        self.assertEqual(repn.linear_coefs[0], -1)
        self.assertIs(repn.linear_vars[0], aux_vars2[1])
        self.assertIs(repn.nonlinear_vars[0], m.x[3])
        self.assertIs(repn.nonlinear_vars[1], m.x[4])
        self.assertIsInstance(repn.nonlinear_expr, EXPR.PowExpression)
        self.assertEqual(repn.nonlinear_expr.args[1], 0.25)
        self.assertIsInstance(repn.nonlinear_expr.args[0], EXPR.SumExpression)
        self.assertEqual(len(repn.nonlinear_expr.args[0].args), 2)
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[0],
                              EXPR.PowExpression)
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[1],
                              EXPR.PowExpression)
        sum_expr = repn.nonlinear_expr.args[0].args[0].args[0]
        self.assertIsInstance(sum_expr, EXPR.SumExpression)
        sum_repn = generate_standard_repn(sum_expr)
        self.assertEqual(sum_repn.constant, 3)
        self.assertTrue(sum_repn.is_linear())
        self.assertEqual(len(sum_repn.linear_vars), 1)
        self.assertEqual(sum_repn.linear_coefs[0], -1)
        self.assertIs(sum_repn.linear_vars[0], m.x[3])
        self.assertEqual(repn.nonlinear_expr.args[0].args[0].args[1], 4)
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[1],
                              EXPR.PowExpression)
        sum_expr = repn.nonlinear_expr.args[0].args[1].args[0]
        self.assertIsInstance(sum_expr, EXPR.SumExpression)
        sum_repn = generate_standard_repn(sum_expr)
        self.assertEqual(sum_repn.constant, 3)
        self.assertTrue(sum_repn.is_linear())
        self.assertEqual(len(sum_repn.linear_vars), 1)
        self.assertEqual(sum_repn.linear_coefs[0], -1)
        self.assertIs(sum_repn.linear_vars[0], m.x[4])
        self.assertEqual(repn.nonlinear_expr.args[0].args[1].args[1], 4)

    def test_transformation_block_fbbt_bounds(self):
        m = self.makeModel()

        TransformationFactory('gdp.between_steps').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            compute_bounds_method='fbbt')

        self.check_transformation_block(m, 0, (2*6**4)**0.25, 0, (2*5**4)**0.25)

    def test_invalid_partition_error(self):
        m = self.makeModel()

        self.assertRaisesRegex(
            GDP_Error,
            "Variables which appear in the expression "
            "\(x\[1\]\*\*4 \+ x\[2\]\*\*4\)\*\*0.25 "
            "are in different partitions, but this expression doesn't appear "
            "additively separable. Please expand it if it is additively "
            "separable or, more likely, ensure that all the constraints in "
            "the disjunction are additively separable with respect to the "
            "specified partition.",
            TransformationFactory('gdp.between_steps').apply_to,
            m,
            variable_partitions=[[m.x[3], m.x[2]], [m.x[1], m.x[4]]],
            compute_bounds_method='fbbt')
