#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

# Need solvers/writers registered.
import pyomo.environ as pyo

import pyutilib.th as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet
from pyomo.core import (Var, Constraint, Param, ConcreteModel, NonNegativeReals,
                        Binary, value, Block, Objective)
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.current import log
from pyomo.gdp import Disjunction, Disjunct
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.opt import SolverFactory, check_available_solvers
import pyomo.contrib.fme.fourier_motzkin_elimination

from six import StringIO
import logging
import random

solvers = check_available_solvers('glpk')

class TestFourierMotzkinElimination(unittest.TestCase):
    def setUp(self):
        # will need this so we know transformation block names in the test that
        # includes hull transformation
        random.seed(666)

    @staticmethod
    def makeModel():
        """
        This is a single-level reformulation of a bilevel model.
        We project out the dual variables to recover the reformulation in 
        the original space.
        """
        m = ConcreteModel()
        m.x = Var(bounds=(0,2))
        m.y = Var(domain=NonNegativeReals)
        m.lamb = Var([1, 2], domain=NonNegativeReals)
        m.M = Param([1, 2], mutable=True, default=100)
        m.u = Var([1, 2], domain=Binary)

        m.primal1 = Constraint(expr=m.x - 0.01*m.y <= 1)
        m.dual1 = Constraint(expr=1 - m.lamb[1] - 0.01*m.lamb[2] == 0)

        @m.Constraint([1, 2])
        def bound_lambdas(m, i):
            return m.lamb[i] <= m.u[i]*m.M[i]

        m.bound_y = Constraint(expr=m.y <= 1000*(1 - m.u[1]))
        m.dual2 = Constraint(expr=-m.x + 0.01*m.y + 1 <= (1 - m.u[2])*1000)

        return m

    def test_no_vars_specified(self):
        m = self.makeModel()
        self.assertRaisesRegexp(
            RuntimeError,
            "The Fourier-Motzkin Elimination transformation "
            "requires the argument vars_to_eliminate, a "
            "list of Vars to be projected out of the model.",
            TransformationFactory('contrib.fourier_motzkin_elimination').\
            apply_to,
            m)

    unfiltered_indices = [1, 2, 3, 6]
    filtered_indices = [1, 2, 3, 4]

    def check_projected_constraints(self, m, indices):
        constraints = m._pyomo_contrib_fme_transformation.projected_constraints

        # x - 0.01y <= 1
        cons = constraints[indices[0]]
        self.assertEqual(value(cons.lower), -1)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertTrue(body.is_linear())
        linear_vars = body.linear_vars
        coefs = body.linear_coefs
        self.assertEqual(len(linear_vars), 2)
        self.assertIs(linear_vars[0], m.x)
        self.assertEqual(coefs[0], -1)
        self.assertIs(linear_vars[1], m.y)
        self.assertEqual(coefs[1], 0.01)

        # y <= 1000*(1 - u_1)
        cons = constraints[indices[1]]
        self.assertEqual(value(cons.lower), -1000)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        linear_vars = body.linear_vars
        coefs = body.linear_coefs
        self.assertEqual(len(linear_vars), 2)
        self.assertIs(linear_vars[0], m.u[1])
        self.assertEqual(coefs[0], -1000)
        self.assertIs(linear_vars[1], m.y)
        self.assertEqual(coefs[1], -1)

        # -x + 0.01y + 1 <= 1000*(1 - u_2)
        cons = constraints[indices[2]]
        self.assertEqual(value(cons.lower), -999)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        linear_vars = body.linear_vars
        coefs = body.linear_coefs
        self.assertEqual(len(linear_vars), 3)
        self.assertIs(linear_vars[0], m.u[2])
        self.assertEqual(coefs[0], -1000)
        self.assertIs(linear_vars[1], m.x)
        self.assertEqual(coefs[1], 1)
        self.assertIs(linear_vars[2], m.y)
        self.assertEqual(coefs[2], -0.01)

        # u_2 + 100u_1 >= 1
        cons = constraints[indices[3]]
        self.assertEqual(value(cons.lower), 1)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        linear_vars = body.linear_vars
        coefs = body.linear_coefs
        self.assertEqual(len(linear_vars), 2)
        self.assertIs(linear_vars[1], m.u[2])
        self.assertEqual(coefs[1], 1)
        self.assertIs(linear_vars[0], m.u[1])
        self.assertEqual(coefs[0], 100)

    def test_transformed_constraints_indexed_var_arg(self):
        m = self.makeModel()
        TransformationFactory('contrib.fourier_motzkin_elimination').apply_to( 
            m,
            vars_to_eliminate = m.lamb,
            constraint_filtering_callback=None)
        # we get some trivial constraints too, but let's check that the ones
        # that should be there really are
        self.check_projected_constraints(m, self.unfiltered_indices)

    def test_transformed_constraints_varData_list_arg(self):
        m = self.makeModel()
        TransformationFactory('contrib.fourier_motzkin_elimination').apply_to( 
            m,
            vars_to_eliminate = [m.lamb[1], m.lamb[2]],
            constraint_filtering_callback=None)

        self.check_projected_constraints(m, self.unfiltered_indices)

    def test_transformed_constraints_indexedVar_list(self):
        m = self.makeModel()
        TransformationFactory('contrib.fourier_motzkin_elimination').apply_to( 
            m,
            vars_to_eliminate = [m.lamb],
            constraint_filtering_callback=None)

        self.check_projected_constraints(m, self.unfiltered_indices)

    def test_default_constraint_filtering(self):
        # We will filter constraints which are trivial based on variable bounds
        # during the transformation. This checks that we removed the constraints
        # we expect.
        m = self.makeModel()
        TransformationFactory('contrib.fourier_motzkin_elimination').apply_to( 
            m,
            vars_to_eliminate = m.lamb)

        # we still have all the right constraints
        self.check_projected_constraints(m, self.filtered_indices)
        # but now we *only* have the right constraints
        constraints = m._pyomo_contrib_fme_transformation.projected_constraints
        self.assertEqual(len(constraints), 4)

    def test_original_constraints_deactivated(self):
        m = self.makeModel()
        TransformationFactory('contrib.fourier_motzkin_elimination').apply_to( 
            m,
            vars_to_eliminate = m.lamb)
        
        self.assertFalse(m.primal1.active)
        self.assertFalse(m.dual1.active)
        self.assertFalse(m.dual2.active)
        self.assertFalse(m.bound_lambdas[1].active)
        self.assertFalse(m.bound_lambdas[2].active)
        self.assertFalse(m.bound_y.active)

    def test_infeasible_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.cons1 = Constraint(expr=m.x >= 6)
        m.cons2 = Constraint(expr=m.x <= 2)

        self.assertRaisesRegexp(
            RuntimeError,
            "Fourier-Motzkin found the model is infeasible!",
            TransformationFactory('contrib.fourier_motzkin_elimination').\
            apply_to,
            m, 
            vars_to_eliminate=m.x)

    def test_infeasible_model_no_var_bounds(self):
        m = ConcreteModel()
        m.x = Var()
        m.cons1 = Constraint(expr=m.x >= 6)
        m.cons2 = Constraint(expr=m.x <= 2)

        self.assertRaisesRegexp(
            RuntimeError,
            "Fourier-Motzkin found the model is infeasible!",
            TransformationFactory('contrib.fourier_motzkin_elimination').\
            apply_to,
            m, 
            vars_to_eliminate=m.x)
        
    def test_nonlinear_error(self):
        m = ConcreteModel()
        m.x = Var()
        m.cons = Constraint(expr=m.x**2 >= 2)
        m.cons2 = Constraint(expr=m.x<= 10)

        self.assertRaisesRegexp(
            RuntimeError,
            "Variable x appears in a nonlinear "
            "constraint. The Fourier-Motzkin "
            "Elimination transformation can only "
            "be used to eliminate variables "
            "which only appear linearly.",
            TransformationFactory('contrib.fourier_motzkin_elimination').\
            apply_to,
            m, 
            vars_to_eliminate=m.x)

    def test_components_we_do_not_understand_error(self):
        m = self.makeModel()
        m.disj = Disjunction(expr=[m.x == 0, m.y >= 2])

        self.assertRaisesRegexp(
            RuntimeError,
            "Found active component %s of type %s. The "
            "Fourier-Motzkin Elimination transformation can only "
            "handle purely algebraic models. That is, only "
            "Sets, Params, Vars, Constraints, Expressions, Blocks, "
            "and Objectives may be active on the model." % (m.disj.name, 
                                                            m.disj.type()),
            TransformationFactory('contrib.fourier_motzkin_elimination').\
            apply_to,
            m, 
            vars_to_eliminate=m.x)

    def test_bad_constraint_filtering_callback_error(self):
        m = self.makeModel()
        def not_a_callback(cons):
            raise RuntimeError("I don't know how to do my job.")
        fme = TransformationFactory('contrib.fourier_motzkin_elimination')
        log = StringIO()
        with LoggingIntercept(log, 'pyomo.contrib.fme', logging.ERROR):
            self.assertRaisesRegexp(
                RuntimeError,
                "I don't know how to do my job.",
                fme.apply_to,
                m,
                vars_to_eliminate=m.x,
                constraint_filtering_callback=not_a_callback)
        self.assertRegexpMatches(
            log.getvalue(),
            "Problem calling constraint filter callback "
            "on constraint with right-hand side -1.0 and body:*")

    def test_constraint_filtering_callback_not_callable_error(self):
        m = self.makeModel()
        fme = TransformationFactory('contrib.fourier_motzkin_elimination')
        log = StringIO()
        with LoggingIntercept(log, 'pyomo.contrib.fme', logging.ERROR):
            self.assertRaisesRegexp(
                TypeError,
                "'int' object is not callable",
                fme.apply_to,
                m,
                vars_to_eliminate=m.x,
                constraint_filtering_callback=5)
        self.assertRegexpMatches(
            log.getvalue(),
            "Problem calling constraint filter callback "
            "on constraint with right-hand side -1.0 and body:*")

    def test_combine_three_inequalities_and_flatten_blocks(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.b = Block()
        m.b.c = Constraint(expr=m.x >= 2)
        m.c = Constraint(expr=m.y <= m.x)
        m.b.b2 = Block()
        m.b.b2.c = Constraint(expr=m.y >= 4)
        TransformationFactory('contrib.fourier_motzkin_elimination').apply_to(
            m, vars_to_eliminate=m.y, do_integer_arithmetic=True)

        constraints = m._pyomo_contrib_fme_transformation.projected_constraints
        self.assertEqual(len(constraints), 2)
        cons = constraints[1]
        self.assertEqual(value(cons.lower), 2)
        self.assertIsNone(cons.upper)
        self.assertIs(cons.body, m.x)
        
        cons = constraints[2]
        self.assertEqual(value(cons.lower), 4)
        self.assertIsNone(cons.upper)
        self.assertIs(cons.body, m.x)

    def check_hull_projected_constraints(self, m, constraints, indices):
        # p[1] >= on.ind_var
        cons = constraints[indices[0]]
        self.assertEqual(cons.lower, 0)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 2)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.on.indicator_var)
        self.assertEqual(body.linear_coefs[0], -1)
        self.assertIs(body.linear_vars[1], m.p[1])
        self.assertEqual(body.linear_coefs[1], 1)

        # p[1] <= 10*on.ind_var + 10*off.ind_var
        cons = constraints[indices[1]]
        self.assertEqual(cons.lower, 0)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 3)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.off.indicator_var)
        self.assertEqual(body.linear_coefs[0], 10)
        self.assertIs(body.linear_vars[1], m.on.indicator_var)
        self.assertEqual(body.linear_coefs[1], 10)
        self.assertIs(body.linear_vars[2], m.p[1])
        self.assertEqual(body.linear_coefs[2], -1)

        # p[1] >= time1_disjuncts[0].ind_var
        cons = constraints[indices[2]]
        self.assertEqual(cons.lower, 0)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 2)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[1], m.time1_disjuncts[0].indicator_var)
        self.assertEqual(body.linear_coefs[1], -1)
        self.assertIs(body.linear_vars[0], m.p[1])
        self.assertEqual(body.linear_coefs[0], 1)

        # p[1] <= 10*time1_disjuncts[0].ind_var
        cons = constraints[indices[3]]
        self.assertEqual(cons.lower, 0)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 2)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.p[1])
        self.assertEqual(body.linear_coefs[0], -1)
        self.assertIs(body.linear_vars[1], m.time1_disjuncts[0].indicator_var)
        self.assertEqual(body.linear_coefs[1], 10)

        # p[2] - p[1] <= 3*on.ind_var + 2*startup.ind_var
        cons = constraints[indices[4]]
        self.assertEqual(value(cons.lower), 0)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 4)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.on.indicator_var)
        self.assertEqual(body.linear_coefs[0], 3)
        self.assertIs(body.linear_vars[1], m.p[1])
        self.assertEqual(body.linear_coefs[1], 1)
        self.assertIs(body.linear_vars[2], m.p[2])
        self.assertEqual(body.linear_coefs[2], -1)
        self.assertIs(body.linear_vars[3], m.startup.indicator_var)
        self.assertEqual(body.linear_coefs[3], 2)

        # p[2] >= on.ind_var + startup.ind_var
        cons = constraints[indices[5]]
        self.assertEqual(cons.lower, 0)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 3)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.on.indicator_var)
        self.assertEqual(body.linear_coefs[0], -1)
        self.assertIs(body.linear_vars[1], m.p[2])
        self.assertEqual(body.linear_coefs[1], 1)
        self.assertIs(body.linear_vars[2], m.startup.indicator_var)
        self.assertEqual(body.linear_coefs[2], -1)

        # p[2] <= 10*on.ind_var + 2*startup.ind_var
        cons = constraints[indices[6]]
        self.assertEqual(cons.lower, 0)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 3)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.on.indicator_var)
        self.assertEqual(body.linear_coefs[0], 10)
        self.assertIs(body.linear_vars[1], m.p[2])
        self.assertEqual(body.linear_coefs[1], -1)
        self.assertIs(body.linear_vars[2], m.startup.indicator_var)
        self.assertEqual(body.linear_coefs[2], 2)

        # 1 <= time1_disjuncts[0].ind_var + time_1.disjuncts[1].ind_var
        cons = constraints[indices[7]]
        self.assertEqual(cons.lower, 1)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 2)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.time1_disjuncts[0].indicator_var)
        self.assertEqual(body.linear_coefs[0], 1)
        self.assertIs(body.linear_vars[1], m.time1_disjuncts[1].indicator_var)
        self.assertEqual(body.linear_coefs[1], 1)

        # 1 >= time1_disjuncts[0].ind_var + time_1.disjuncts[1].ind_var
        cons = constraints[indices[8]]
        self.assertEqual(cons.lower, -1)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 2)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.time1_disjuncts[0].indicator_var)
        self.assertEqual(body.linear_coefs[0], -1)
        self.assertIs(body.linear_vars[1], m.time1_disjuncts[1].indicator_var)
        self.assertEqual(body.linear_coefs[1], -1)

        # 1 <= on.ind_var + startup.ind_var + off.ind_var
        cons = constraints[indices[9]]
        self.assertEqual(cons.lower, 1)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 3)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.off.indicator_var)
        self.assertEqual(body.linear_coefs[0], 1)
        self.assertIs(body.linear_vars[1], m.on.indicator_var)
        self.assertEqual(body.linear_coefs[1], 1)
        self.assertIs(body.linear_vars[2], m.startup.indicator_var)
        self.assertEqual(body.linear_coefs[2], 1)
        
        # 1 >= on.ind_var + startup.ind_var + off.ind_var
        cons = constraints[indices[10]]
        self.assertEqual(cons.lower, -1)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 3)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.off.indicator_var)
        self.assertEqual(body.linear_coefs[0], -1)
        self.assertIs(body.linear_vars[1], m.on.indicator_var)
        self.assertEqual(body.linear_coefs[1], -1)
        self.assertIs(body.linear_vars[2], m.startup.indicator_var)
        self.assertEqual(body.linear_coefs[2], -1)

    def create_hull_model(self):
        m = ConcreteModel()
        m.p = Var([1, 2], bounds=(0, 10))
        m.time1 = Disjunction(expr=[m.p[1] >= 1, m.p[1] == 0])

        m.on = Disjunct()
        m.on.above_min = Constraint(expr=m.p[2] >= 1)
        m.on.ramping = Constraint(expr=m.p[2] - m.p[1] <= 3)
        m.on.on_before = Constraint(expr=m.p[1] >= 1)

        m.startup = Disjunct()
        m.startup.startup_limit = Constraint(expr=(1, m.p[2], 2))
        m.startup.off_before = Constraint(expr=m.p[1] == 0)

        m.off = Disjunct()
        m.off.off = Constraint(expr=m.p[2] == 0)
        m.time2 = Disjunction(expr=[m.on, m.startup, m.off])

        m.obj = Objective(expr=m.p[1] + m.p[2])

        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        disaggregatedVars = ComponentSet(
            [hull.get_disaggregated_var(m.p[1], m.time1.disjuncts[0]),
             hull.get_disaggregated_var(m.p[1], m.time1.disjuncts[1]),
             hull.get_disaggregated_var(m.p[1], m.on),
             hull.get_disaggregated_var(m.p[2], m.on),
             hull.get_disaggregated_var(m.p[1], m.startup),
             hull.get_disaggregated_var(m.p[2], m.startup),
             hull.get_disaggregated_var(m.p[1], m.off),
             hull.get_disaggregated_var(m.p[2], m.off)
         ])
        
        return m, disaggregatedVars

    def test_project_disaggregated_vars(self):
        """This is a little bit more of an integration test with GDP, 
        but also an example of why FME is 'useful.' We will give a GDP, 
        take hull relaxation, and then project out the disaggregated 
        variables."""
        m, disaggregatedVars = self.create_hull_model()
        
        filtered = TransformationFactory('contrib.fourier_motzkin_elimination').\
                   create_using(m, vars_to_eliminate=disaggregatedVars)
        TransformationFactory('contrib.fourier_motzkin_elimination').apply_to(
            m, vars_to_eliminate=disaggregatedVars,
            constraint_filtering_callback=None, do_integer_arithmetic=True)

        constraints = m._pyomo_contrib_fme_transformation.projected_constraints
        # we of course get tremendous amounts of garbage, but we make sure that
        # what should be here is:
        self.check_hull_projected_constraints(m, constraints, [16, 11, 57, 59,
                                                               46, 48, 27, 1, 2,
                                                               4, 5])
        # and when we filter, it's still there.
        constraints = filtered._pyomo_contrib_fme_transformation.\
                      projected_constraints
        self.check_hull_projected_constraints(filtered, constraints, [6, 5, 16,
                                                                      17, 12,
                                                                      13, 8, 1,
                                                                      2, 3, 4])
    
    @unittest.skipIf(not 'glpk' in solvers, 'glpk not available')
    def test_post_processing(self):
        m, disaggregatedVars = self.create_hull_model()
        fme = TransformationFactory('contrib.fourier_motzkin_elimination')
        fme.apply_to(m, vars_to_eliminate=disaggregatedVars,
                     do_integer_arithmetic=True)
        # post-process
        fme.post_process_fme_constraints(m, SolverFactory('glpk'))

        constraints = m._pyomo_contrib_fme_transformation.projected_constraints
        self.assertEqual(len(constraints), 11)

        # They should be the same as the above, but now these are *all* the
        # constraints
        self.check_hull_projected_constraints(m, constraints, [6, 5, 16, 17, 12,
                                                               13, 8, 1, 2, 3,
                                                               4])

        # and check that we didn't change the model
        for disj in m.component_data_objects(Disjunct):
            self.assertIs(disj.indicator_var.domain, Binary)
        self.assertEqual(len([o for o in m.component_data_objects(Objective)]),
                         1)
        self.assertIsInstance(m.component("obj"), Objective)
        self.assertTrue(m.obj.active)
        
    @unittest.skipIf(not 'glpk' in solvers, 'glpk not available')
    def test_model_with_unrelated_nonlinear_expressions(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3], bounds=(0,3))
        m.y = Var()
        m.z = Var()

        @m.Constraint([1,2])
        def cons(m, i):
            return m.x[i] <= m.y**i

        m.cons2 = Constraint(expr=m.x[1] >= m.y)
        m.cons3 = Constraint(expr=m.x[2] >= m.z - 3)
        # This is vacuous, but I just want something that's not quadratic
        m.cons4 = Constraint(expr=m.x[3] <= log(m.y + 1))

        fme = TransformationFactory('contrib.fourier_motzkin_elimination')
        fme.apply_to(m, vars_to_eliminate=m.x,
                     projected_constraints_name='projected_constraints',
                     constraint_filtering_callback=None)
        constraints = m.projected_constraints

        # 0 <= y <= 3
        cons = constraints[5]
        self.assertEqual(value(cons.lower), 0)
        self.assertIs(cons.body, m.y)
        cons = constraints[6]
        self.assertEqual(value(cons.lower), -3)
        body = generate_standard_repn(cons.body)
        self.assertTrue(body.is_linear())
        self.assertEqual(len(body.linear_vars), 1)
        self.assertIs(body.linear_vars[0], m.y)
        self.assertEqual(body.linear_coefs[0], -1)

        # z <= y**2 + 3
        cons = constraints[2]
        self.assertEqual(value(cons.lower), -3)
        body = generate_standard_repn(cons.body)
        self.assertTrue(body.is_quadratic())
        self.assertEqual(len(body.linear_vars), 1)
        self.assertIs(body.linear_vars[0], m.z)
        self.assertEqual(body.linear_coefs[0], -1)
        self.assertEqual(len(body.quadratic_vars), 1)
        self.assertEqual(body.quadratic_coefs[0], 1)
        self.assertIs(body.quadratic_vars[0][0], m.y)
        self.assertIs(body.quadratic_vars[0][1], m.y)

        # z <= 6
        cons = constraints[4]
        self.assertEqual(cons.lower, -6)
        body = generate_standard_repn(cons.body)
        self.assertTrue(body.is_linear())
        self.assertEqual(len(body.linear_vars), 1)
        self.assertEqual(body.linear_coefs[0], -1)
        self.assertIs(body.linear_vars[0], m.z)

        # 0 <= ln(y+ 1)
        cons = constraints[1]
        self.assertEqual(value(cons.lower), 0)
        body = generate_standard_repn(cons.body)
        self.assertTrue(body.is_nonlinear())
        self.assertFalse(body.is_quadratic())
        self.assertEqual(len(body.linear_vars), 0)
        self.assertEqual(body.nonlinear_expr.name, 'log')
        self.assertEqual(len(body.nonlinear_expr.args[0].args), 2)
        self.assertIs(body.nonlinear_expr.args[0].args[0], m.y)
        self.assertEqual(body.nonlinear_expr.args[0].args[1], 1)

        # 0 <= y**2
        cons = constraints[3]
        self.assertEqual(value(cons.lower), 0)
        body = generate_standard_repn(cons.body)
        self.assertTrue(body.is_quadratic())
        self.assertEqual(len(body.quadratic_vars), 1)
        self.assertEqual(body.quadratic_coefs[0], 1)
        self.assertIs(body.quadratic_vars[0][0], m.y)
        self.assertIs(body.quadratic_vars[0][1], m.y)

        # check constraints valid for a selection of points (this is nonconvex,
        # but anyway...)
        pts = [#(sqrt(3), 6), Not numerically stable enough for this test
               (1, 4), (3, 6), (3, 0), (0, 0), (2,6)]
        for pt in pts:
            m.y.fix(pt[0])
            m.z.fix(pt[1])
            for i in constraints:
                self.assertLessEqual(value(constraints[i].lower),
                                     value(constraints[i].body))
        m.y.fixed = False
        m.z.fixed = False
        
        # check post process these are non-convex, so I don't want to deal with
        # it... (and this is a good test that I *don't* deal with it.)
        constraints[2].deactivate()
        constraints[3].deactivate()
        constraints[1].deactivate()
        # NOTE also that some of the suproblems in this test are unbounded: We
        # need to keep those constraints.
        fme.post_process_fme_constraints(
            m, SolverFactory('glpk'),
            projected_constraints=m.projected_constraints)
        # we needed all the constraints, so we kept them all
        self.assertEqual(len(constraints), 6)

        # last check that if someone activates something on the model in
        # between, we just use it. (I struggle to imagine why you would do this
        # because why withold the information *during* FME, but if there's some
        # reason, we may as well use all the information we've got.)
        m.some_new_cons = Constraint(expr=m.y <= 2)
        fme.post_process_fme_constraints(
            m, SolverFactory('glpk'),
            projected_constraints=m.projected_constraints)
        # now we should have lost one constraint
        self.assertEqual(len(constraints), 5)
        # and it should be the y <= 3 one...
        self.assertIsNone(dict(constraints).get(6))

    @unittest.skipIf(not 'glpk' in solvers, 'glpk not available')
    def test_noninteger_coefficients_of_vars_being_projected_error(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0,9))
        m.y = Var(bounds=(-5, 5))
        m.c1 = Constraint(expr=2*m.x + 0.5*m.y >= 2)
        m.c2 = Constraint(expr=0.25*m.y >= 0.5*m.x)

        fme = TransformationFactory('contrib.fourier_motzkin_elimination')
        self.assertRaisesRegexp(
            ValueError,
            "The do_integer_arithmetic flag was "
            "set to True, but the coefficient of "
            "x is non-integer within the specified tolerance, "
            "with value -0.5. \n"
            "Please set do_integer_arithmetic="
            "False, increase integer_tolerance, or make your data integer.",
            fme.apply_to,
            m, 
            vars_to_eliminate=m.x, 
            do_integer_arithmetic=True)

    @unittest.skipIf(not 'glpk' in solvers, 'glpk not available')
    def test_noninteger_coefficients_of_vars_not_being_projected_error(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0,9))
        m.y = Var(bounds=(-5, 5))
        m.c1 = Constraint(expr=2*m.x + 0.5*m.y >= 2)
        m.c2 = Constraint(expr=0.25*m.y >= 5*m.x)

        fme = TransformationFactory('contrib.fourier_motzkin_elimination')
        self.assertRaisesRegexp(
            ValueError,
            "The do_integer_arithmetic flag was "
            "set to True, but the coefficient of "
            "y is non-integer within the specified tolerance, "
            "with value 0.5. \n"
            "Please set do_integer_arithmetic="
            "False, increase integer_tolerance, or make your data integer.",
            fme.apply_to,
            m, 
            vars_to_eliminate=m.x, 
            do_integer_arithmetic=True)

    def test_integer_arithmetic_non1_coefficients(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0,9))
        m.y = Var(bounds=(-5, 5))
        m.c1 = Constraint(expr=4*m.x + m.y >= 4)
        m.c2 = Constraint(expr=m.y >= 2*m.x)

        fme = TransformationFactory('contrib.fourier_motzkin_elimination')
        
        fme.apply_to( m, vars_to_eliminate=m.x,
                      constraint_filtering_callback=None,
                      do_integer_arithmetic=True, verbose=True)

        constraints = m._pyomo_contrib_fme_transformation.projected_constraints

        self.assertEqual(len(constraints), 3)

        cons = constraints[3]
        self.assertEqual(value(cons.lower), -32)
        self.assertIs(cons.body, m.y)
        self.assertIsNone(cons.upper)

        cons = constraints[2]
        self.assertEqual(value(cons.lower), 0)
        self.assertIsNone(cons.upper)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_coefs), 1)
        self.assertIs(repn.linear_vars[0], m.y)
        self.assertEqual(repn.linear_coefs[0], 2)

        cons = constraints[1]
        self.assertEqual(value(cons.lower), 4)
        self.assertIsNone(cons.upper)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_coefs), 1)
        self.assertIs(repn.linear_vars[0], m.y)
        self.assertEqual(repn.linear_coefs[0], 3)

    def test_numerical_instability_almost_canceling(self):
        # It's possible that we get almost-but-not-quite zero on the variable
        # being eliminated when we are doing this with floating point
        # arithmetic. This can get ugly later becuase it might get muliplied by
        # a large number later and start to "reappear"
        m = ConcreteModel()
        m.x = Var()
        m.x0 = Var()
        m.y = Var()

        m.cons1 = Constraint(expr=(1.342 + 2.371e-8)*m.x0 <= m.x + 17*m.y)
        m.cons2 = Constraint(expr=(17.56 + 3.2e-7)*m.x0 >= m.y)
        
        fme = TransformationFactory('contrib.fourier_motzkin_elimination')
        
        fme.apply_to(m, vars_to_eliminate=[m.x0], verbose=True,
                     zero_tolerance=1e-9)

        constraints = m._pyomo_contrib_fme_transformation.projected_constraints

        # There's going to be numerical error here, and I can't really help
        # it. What I care about is that x0 really is gone.

        useful = constraints[1]
        repn = generate_standard_repn(useful.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_coefs), 2) # this is the real test
        self.assertEqual(useful.lower, 0)
        self.assertIs(repn.linear_vars[0], m.x)
        self.assertAlmostEqual(repn.linear_coefs[0], 0.7451564696962295)
        self.assertIs(repn.linear_vars[1], m.y)
        self.assertAlmostEqual(repn.linear_coefs[1], 12.610712377673217)
        self.assertEqual(repn.constant, 0)
        self.assertIsNone(useful.upper)

    def test_numerical_instability_early_elimination(self):
        # A more subtle numerical problem is that, in infinite precision, a
        # variable might be eliminated early. However, if this goes wrong, the
        # result can be unexpected (including getting no constraints when some
        # are expected.)
        m = ConcreteModel()
        m.x = Var()
        m.x0 = Var()
        m.y = Var()
        
        # we'll pretend that the 1.123e-9 is noise from previous calculations
        m.cons1 = Constraint(expr=0 <= (4.27 + 1.123e-9)*m.x + 13*m.y - m.x0)
        m.cons2 = Constraint(expr=m.x0 >= 12*m.y + 4.27*m.x)

        fme = TransformationFactory('contrib.fourier_motzkin_elimination')
        
        # doing my own clones because I want assertIs tests
        first = m.clone()
        second = m.clone()
        third = m.clone()

        fme.apply_to(first, vars_to_eliminate=[first.x0], zero_tolerance=1e-10)
        constraints = first._pyomo_contrib_fme_transformation.\
                      projected_constraints
        cons = constraints[1]
        self.assertEqual(cons.lower, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_coefs), 2) # x is still around
        self.assertIs(repn.linear_vars[0], first.x)
        self.assertAlmostEqual(repn.linear_coefs[0], 1.123e-9)
        self.assertIs(repn.linear_vars[1], first.y)
        self.assertEqual(repn.linear_coefs[1], 1)
        self.assertIsNone(cons.upper)

        # so just to drive home the point, this results in no constraints:
        # (Though also note that that only happens if x0 is the first to be
        # projected out)
        fme.apply_to(second, vars_to_eliminate=[second.x0, second.x],
                     zero_tolerance=1e-10)
        self.assertEqual(len(second._pyomo_contrib_fme_transformation.\
                             projected_constraints), 0)
        
        # but in this version, we assume that x is already gone...
        fme.apply_to(third, vars_to_eliminate=[third.x0], verbose=True,
                     zero_tolerance=1e-8)
        constraints = third._pyomo_contrib_fme_transformation.\
                      projected_constraints
        cons = constraints[1]
        self.assertEqual(cons.lower, 0)
        self.assertIs(cons.body, third.y)
        self.assertIsNone(cons.upper)

        # and this is exactly the same as the above:
        fme.apply_to(m, vars_to_eliminate=[m.x0, m.x], verbose=True,
                     zero_tolerance=1e-8)
        constraints = m._pyomo_contrib_fme_transformation.projected_constraints
        cons = constraints[1]
        self.assertEqual(cons.lower, 0)
        self.assertIs(cons.body, m.y)
        self.assertIsNone(cons.upper)

    def make_tiny_model_where_bounds_matter(self):
        m = ConcreteModel()
        m.b = Block()
        m.x = Var(bounds=(0, 15))
        m.y = Var(bounds=(3, 5))
        m.b.c = Constraint(expr=m.x + m.y <= 8)

        return m

    def check_tiny_model_constraints(self, constraints):
        m = constraints.model()
        self.assertEqual(len(constraints), 1)
        cons = constraints[1]
        self.assertEqual(value(cons.lower), -5)
        self.assertIsNone(cons.upper)
        repn = generate_standard_repn(cons.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], m.x)
        self.assertEqual(repn.linear_coefs[0], -1)

    def test_use_all_var_bounds(self):
        m = self.make_tiny_model_where_bounds_matter()

        fme = TransformationFactory('contrib.fourier_motzkin_elimination')
        fme.apply_to(m.b, vars_to_eliminate=[m.y])
        constraints = m.b.\
                      _pyomo_contrib_fme_transformation.projected_constraints

        # if we hadn't included y's bounds, then we wouldn't get any constraints
        # and y wouldn't be eliminated. If we do include y's bounds, we get new
        # information that x <= 5:
        self.check_tiny_model_constraints(constraints)

    def test_projected_constraints_named_correctly(self):
        m = self.make_tiny_model_where_bounds_matter()
        fme = TransformationFactory('contrib.fourier_motzkin_elimination')
        fme.apply_to(m.b, vars_to_eliminate=[m.y],
                     projected_constraints_name='fme_constraints')
        self.assertIsInstance(m.b.component("fme_constraints"), Constraint)
        self.check_tiny_model_constraints(m.b.fme_constraints)

        self.assertIsNone(m.b._pyomo_contrib_fme_transformation.component(
            "projected_constraints"))

    def test_non_unique_constraint_name_error(self):
        m = self.make_tiny_model_where_bounds_matter()
        fme = TransformationFactory('contrib.fourier_motzkin_elimination')
        self.assertRaisesRegexp(
            RuntimeError,
            "projected_constraints_name was specified "
            "as 'c', but this is already a component on "
            "the instance! Please specify a unique " 
            "name.",
            fme.apply_to,
            m.b, 
            vars_to_eliminate=[m.y],
            projected_constraints_name='c')

    def test_simple_hull_example(self):
        m = ConcreteModel()
        m.x0 = Var(bounds=(0,3))
        m.x1 = Var(bounds=(0,3))
        m.x = Var(bounds=(0,3))
        m.disaggregation = Constraint(expr=m.x == m.x0 + m.x1)
        m.y = Var(domain=Binary)
        m.cons = Constraint(expr=2*m.y <= m.x1)

        fme = TransformationFactory('contrib.fourier_motzkin_elimination')
        fme.apply_to(m, vars_to_eliminate=[m.x0, m.x1])

        constraints = m._pyomo_contrib_fme_transformation.projected_constraints

        self.assertEqual(len(constraints), 1)
        cons = constraints[1]
        self.assertIsNone(cons.upper)
        self.assertEqual(value(cons.lower), 0)
        repn = generate_standard_repn(cons.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertIs(repn.linear_vars[0], m.x)
        self.assertEqual(repn.linear_coefs[0], 1)
        self.assertIs(repn.linear_vars[1], m.y)
        self.assertEqual(repn.linear_coefs[1], -2)
        self.assertTrue(repn.is_linear())
