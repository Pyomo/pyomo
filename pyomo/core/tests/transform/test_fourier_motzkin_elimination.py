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

import pyutilib.th as unittest
from pyomo.core import (Var, Constraint, Param, ConcreteModel, NonNegativeReals,
                        Binary, value)
from pyomo.core.base import TransformationFactory
from pyomo.gdp import Disjunction
from pyomo.repn.standard_repn import generate_standard_repn

class TestFourierMotzkinElimination(unittest.TestCase):
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
            TransformationFactory('core.fourier_motzkin_elimination').apply_to,
            m)

    def check_projected_constraints(self, m):
        constraints = m._pyomo_core_fme_transformation.projected_constraints
        # x - 0.01y <= 1
        cons = constraints[4]
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
        cons = constraints[5]
        self.assertEqual(value(cons.lower), -1000)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        linear_vars = body.linear_vars
        coefs = body.linear_coefs
        self.assertEqual(len(linear_vars), 2)
        self.assertIs(linear_vars[0], m.y)
        self.assertEqual(coefs[0], -1)
        self.assertIs(linear_vars[1], m.u[1])
        self.assertEqual(coefs[1], -1000)

        # -x + 0.01y + 1 <= 1000*(1 - u_2)
        cons = constraints[6]
        self.assertEqual(value(cons.lower), -999)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        linear_vars = body.linear_vars
        coefs = body.linear_coefs
        self.assertEqual(len(linear_vars), 3)
        self.assertIs(linear_vars[0], m.x)
        self.assertEqual(coefs[0], 1)
        self.assertIs(linear_vars[1], m.y)
        self.assertEqual(coefs[1], -0.01)
        self.assertIs(linear_vars[2], m.u[2])
        self.assertEqual(coefs[2], -1000)

        # 100u_2 + 10000u_2 >= 100
        cons = constraints[2]
        self.assertEqual(value(cons.lower), 100)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        linear_vars = body.linear_vars
        coefs = body.linear_coefs
        self.assertEqual(len(linear_vars), 2)
        self.assertIs(linear_vars[0], m.u[2])
        self.assertEqual(coefs[0], 100)
        self.assertIs(linear_vars[1], m.u[1])
        self.assertEqual(coefs[1], 10000)

    def test_transformed_constraints_indexed_var_arg(self):
        m = self.makeModel()
        TransformationFactory('core.fourier_motzkin_elimination').apply_to( 
            m,
            vars_to_eliminate = m.lamb)

        # we get some trivial constraints too, but let's check that the ones
        # that should be there really are
        self.check_projected_constraints(m)

    def test_transformed_constraints_varData_list_arg(self):
        m = self.makeModel()
        TransformationFactory('core.fourier_motzkin_elimination').apply_to( 
            m,
            vars_to_eliminate = [m.lamb[1], m.lamb[2]])

        self.check_projected_constraints(m)

    def test_transformed_constraints_indexedVar_list(self):
        m = self.makeModel()
        TransformationFactory('core.fourier_motzkin_elimination').apply_to( 
            m,
            vars_to_eliminate = [m.lamb])

        self.check_projected_constraints(m)

    def test_original_constraints_deactivated(self):
        m = self.makeModel()
        TransformationFactory('core.fourier_motzkin_elimination').apply_to( 
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
            "Fourier-Motzkin found that model is infeasible!",
            TransformationFactory('core.fourier_motzkin_elimination').apply_to,
            m, 
            vars_to_eliminate=m.x)

    def test_infeasible_model_no_var_bounds(self):
        m = ConcreteModel()
        m.x = Var()
        m.cons1 = Constraint(expr=m.x >= 6)
        m.cons2 = Constraint(expr=m.x <= 2)

        self.assertRaisesRegexp(
            RuntimeError,
            "Fourier-Motzkin found that model is infeasible!",
            TransformationFactory('core.fourier_motzkin_elimination').apply_to,
            m, 
            vars_to_eliminate=m.x)
        
    def test_nonlinear_error(self):
        m = ConcreteModel()
        m.x = Var()
        m.cons = Constraint(expr=m.x**2 >= 2)
        m.cons2 = Constraint(expr=m.x<= 10)

        self.assertRaisesRegexp(
            RuntimeError,
            "Found nonlinear constraint %s. The "
            "Fourier-Motzkin Elimination transformation "
            "can only be applied to linear models!"
            % m.cons.name,
            TransformationFactory('core.fourier_motzkin_elimination').apply_to,
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
            TransformationFactory('core.fourier_motzkin_elimination').apply_to,
            m, 
            vars_to_eliminate=m.x)
