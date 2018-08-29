#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for model transformations
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.services

import pyomo.opt
from pyomo.environ import *


solvers = pyomo.opt.check_available_solvers('glpk')


class Test(unittest.TestCase):

    def setUp(self):
        self.model = AbstractModel()

    def tearDown(self):
        if os.path.exists("unknown.lp"):
            os.unlink("unknown.lp")
        pyutilib.services.TempfileManager.clear_tempfiles()
        if os.path.exists(os.path.join(currdir,'result.yml')):
            os.remove(os.path.join(currdir,'result.yml'))
        self.model = None

    @staticmethod
    def nonnegativeBounds(var):
        # Either the bounds or domain must enforce nonnegativity
        if var.lb is not None and var.lb >= 0:
            return True
        elif var.domain is not None and var.domain.bounds()[0] >= 0:
            return True
        else:
            return False

    def test_transform_dir(self):
        model = AbstractModel()
        self.assertTrue(set(TransformationFactory) >= set(['core.relax_integrality']))

    def test_relax_integrality1(self):
        # Coverage of the _clear_attribute method
        self.model.A = RangeSet(1,4)
        self.model.a = Var()
        self.model.b = Var(within=self.model.A)
        self.model.c = Var(within=NonNegativeIntegers)
        self.model.d = Var(within=Integers, bounds=(-2,3))
        self.model.e = Var(within=Boolean)
        self.model.f = Var(domain=Boolean)
        instance=self.model.create_instance()
        xfrm = TransformationFactory('core.relax_integrality')
        rinst = xfrm.create_using(instance)
        self.assertEqual(type(rinst.a.domain), RealSet)
        self.assertEqual(type(rinst.b.domain), RealSet)
        self.assertEqual(type(rinst.c.domain), RealSet)
        self.assertEqual(type(rinst.d.domain), RealSet)
        self.assertEqual(type(rinst.e.domain), RealSet)
        self.assertEqual(type(rinst.f.domain), RealSet)
        self.assertEqual(rinst.a.bounds, instance.a.bounds)
        self.assertEqual(rinst.b.bounds, instance.b.bounds)
        self.assertEqual(rinst.c.bounds, instance.c.bounds)
        self.assertEqual(rinst.d.bounds, instance.d.bounds)
        self.assertEqual(rinst.e.bounds, instance.e.bounds)
        self.assertEqual(rinst.f.bounds, instance.f.bounds)

    def test_relax_integrality2(self):
        # Coverage of the _clear_attribute method
        self.model.A = RangeSet(1,4)
        self.model.a = Var([1,2,3], dense=True)
        self.model.b = Var([1,2,3], within=self.model.A, dense=True)
        self.model.c = Var([1,2,3], within=NonNegativeIntegers, dense=True)
        self.model.d = Var([1,2,3], within=Integers, bounds=(-2,3), dense=True)
        self.model.e = Var([1,2,3], within=Boolean, dense=True)
        self.model.f = Var([1,2,3], domain=Boolean, dense=True)
        instance=self.model.create_instance()
        xfrm = TransformationFactory('core.relax_integrality')
        rinst = xfrm.create_using(instance)
        self.assertEqual(type(rinst.a[1].domain), RealSet)
        self.assertEqual(type(rinst.b[1].domain), RealSet)
        self.assertEqual(type(rinst.c[1].domain), RealSet)
        self.assertEqual(type(rinst.d[1].domain), RealSet)
        self.assertEqual(type(rinst.e[1].domain), RealSet)
        self.assertEqual(type(rinst.f[1].domain), RealSet)
        self.assertEqual(rinst.a[1].bounds, instance.a[1].bounds)
        self.assertEqual(rinst.b[1].bounds, instance.b[1].bounds)
        self.assertEqual(rinst.c[1].bounds, instance.c[1].bounds)
        self.assertEqual(rinst.d[1].bounds, instance.d[1].bounds)
        self.assertEqual(rinst.e[1].bounds, instance.e[1].bounds)
        self.assertEqual(rinst.f[1].bounds, instance.f[1].bounds)

    def test_nonnegativity_transformation_1(self):
        self.model.a = Var()
        self.model.b = Var(within=NonNegativeIntegers)
        self.model.c = Var(within=Integers, bounds=(-2,3))
        self.model.d = Var(within=Boolean)
        self.model.e = Var(domain=Boolean)

        instance=self.model.create_instance()
        xfrm = TransformationFactory('core.nonnegative_vars')
        transformed = xfrm.create_using(instance)

        # Check that all variables have nonnegative bounds or domains
        for c in ('a', 'b', 'c', 'd', 'e'):
            var = transformed.__getattribute__(c)
            for ndx in var:
                self.assertTrue(self.nonnegativeBounds(var[ndx]))

        # Check that discrete variables are still discrete, and continuous
        # continuous
        for ndx in transformed.a:
            self.assertTrue(isinstance(transformed.a[ndx].domain, RealSet))
        for ndx in transformed.b:
            self.assertTrue(isinstance(transformed.b[ndx].domain, IntegerSet))
        for ndx in transformed.c:
            self.assertTrue(isinstance(transformed.c[ndx].domain, IntegerSet))
        for ndx in transformed.d:
            self.assertTrue(isinstance(transformed.d[ndx].domain, BooleanSet))
        for ndx in transformed.e:
            self.assertTrue(isinstance(transformed.e[ndx].domain, BooleanSet))

    def test_nonnegativity_transformation_2(self):
        self.model.S = RangeSet(0,10)
        self.model.T = Set(initialize=["foo", "bar"])

        # Unindexed, singly indexed, and doubly indexed variables with
        # explicit bounds
        self.model.x1 = Var(bounds=(-3, 3))
        self.model.y1 = Var(self.model.S, bounds=(-3, 3))
        self.model.z1 = Var(self.model.S, self.model.T, bounds=(-3, 3))

        # Unindexed, singly indexed, and doubly indexed variables with
        # rule-defined bounds
        def boundsRule(*args):
            return (-4, 4)
        self.model.x2 = Var(bounds=boundsRule)
        self.model.y2 = Var(self.model.S, bounds=boundsRule)
        self.model.z2 = Var(self.model.S, self.model.T, bounds=boundsRule)


        # Unindexed, singly indexed, and doubly indexed variables with
        # explicit domains
        self.model.x3 = Var(domain=NegativeReals)
        self.model.y3 = Var(self.model.S, domain = NegativeIntegers)
        self.model.z3 = Var(self.model.S, self.model.T, domain = Reals)

        # Unindexed, singly indexed, and doubly indexed variables with
        # rule-defined domains
        def domainRule(*args):
            if len(args) == 1 or args[0] == 0:
                return NonNegativeReals
            elif args[0] == 1:
                return NonNegativeIntegers
            elif args[0] == 2:
                return NonPositiveReals
            elif args[0] == 3:
                return NonPositiveIntegers
            elif args[0] == 4:
                return NegativeReals
            elif args[0] == 5:
                return NegativeIntegers
            elif args[0] == 6:
                return PositiveReals
            elif args[0] == 7:
                return PositiveIntegers
            elif args[0] == 8:
                return Reals
            elif args[0] == 9:
                return Integers
            elif args[0] == 10:
                return Binary
            else:
                return NonNegativeReals

        self.model.x4 = Var(domain=domainRule)
        self.model.y4 = Var(self.model.S, domain=domainRule)
        self.model.z4 = Var(self.model.S, self.model.T, domain=domainRule)

        instance = self.model.create_instance()
        xfrm = TransformationFactory('core.nonnegative_vars')
        transformed = xfrm.create_using(instance)

        # Make sure everything is nonnegative
        for c in ('x', 'y', 'z'):
            for n in ('1', '2', '3', '4'):
                var = transformed.__getattribute__(c+n)
                for ndx in var._index:
                    self.assertTrue(self.nonnegativeBounds(var[ndx]))

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
    @unittest.expectedFailure
    def test_nonnegative_transform_3(self):
        self.model.S = RangeSet(0,10)
        self.model.T = Set(initialize=["foo", "bar"])

        # Unindexed, singly indexed, and doubly indexed variables with
        # explicit bounds
        self.model.x1 = Var(bounds=(-3, 3))
        self.model.y1 = Var(self.model.S, bounds=(-3, 3))
        self.model.z1 = Var(self.model.S, self.model.T, bounds=(-3, 3))

        # Unindexed, singly indexed, and doubly indexed variables with
        # rule-defined bounds
        def boundsRule(*args):
            return (-4, 4)
        self.model.x2 = Var(bounds=boundsRule)
        self.model.y2 = Var(self.model.S, bounds=boundsRule)
        self.model.z2 = Var(self.model.S, self.model.T, bounds=boundsRule)


        # Unindexed, singly indexed, and doubly indexed variables with
        # explicit domains
        self.model.x3 = Var(domain=NegativeReals, bounds=(-10, 10))
        self.model.y3 = Var(self.model.S, domain = NegativeIntegers, bounds=(-10, 10))
        self.model.z3 = Var(self.model.S, self.model.T, domain = Reals, bounds=(-10, 10))

        # Unindexed, singly indexed, and doubly indexed variables with
        # rule-defined domains
        def domainRule(*args):
            if len(args) == 1:
                arg = 0
            else:
                arg = args[1]

            if len(args) == 1 or arg == 0:
                return NonNegativeReals
            elif arg == 1:
                return NonNegativeIntegers
            elif arg == 2:
                return NonPositiveReals
            elif arg == 3:
                return NonPositiveIntegers
            elif arg == 4:
                return NegativeReals
            elif arg == 5:
                return NegativeIntegers
            elif arg == 6:
                return PositiveReals
            elif arg == 7:
                return PositiveIntegers
            elif arg == 8:
                return Reals
            elif arg == 9:
                return Integers
            elif arg == 10:
                return Binary
            else:
                return Reals

        self.model.x4 = Var(domain=domainRule, bounds=(-10, 10))
        self.model.y4 = Var(self.model.S, domain=domainRule, bounds=(-10, 10))
        self.model.z4 = Var(self.model.S, self.model.T, domain=domainRule, bounds=(-10, 10))

        def objRule(model):
            return sum(5*sum_product(model.__getattribute__(c+n)) \
                       for c in ('x', 'y', 'z') for n in ('1', '2', '3', '4'))

        self.model.obj = Objective(rule=objRule)

        transform = TransformationFactory('core.nonnegative_vars')
        instance=self.model.create_instance()
        transformed = transform.create_using(instance)

        opt = SolverFactory("glpk")

        instance_sol = opt.solve(instance)
        transformed_sol = opt.solve(transformed)

        self.assertEqual(
            instance_sol["Solution"][0]["Objective"]['obj']["value"],
            transformed_sol["Solution"][0]["Objective"]['obj']["value"]
            )

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
    @unittest.expectedFailure
    def test_nonnegative_transform_4(self):
        """ Same as #3, but adds constraints """
        self.model.S = RangeSet(0,10)
        self.model.T = Set(initialize=["foo", "bar"])

        # Unindexed, singly indexed, and doubly indexed variables with
        # explicit bounds
        self.model.x1 = Var(bounds=(-3, 3))
        self.model.y1 = Var(self.model.S, bounds=(-3, 3))
        self.model.z1 = Var(self.model.S, self.model.T, bounds=(-3, 3))

        # Unindexed, singly indexed, and doubly indexed variables with
        # rule-defined bounds
        def boundsRule(*args):
            return (-4, 4)
        self.model.x2 = Var(bounds=boundsRule)
        self.model.y2 = Var(self.model.S, bounds=boundsRule)
        self.model.z2 = Var(self.model.S, self.model.T, bounds=boundsRule)


        # Unindexed, singly indexed, and doubly indexed variables with
        # explicit domains
        self.model.x3 = Var(domain=NegativeReals, bounds=(-10, 10))
        self.model.y3 = Var(self.model.S, domain = NegativeIntegers, bounds=(-10, 10))
        self.model.z3 = Var(self.model.S, self.model.T, domain = Reals, bounds=(-10, 10))

        # Unindexed, singly indexed, and doubly indexed variables with
        # rule-defined domains
        def domainRule(*args):
            if len(args) == 1:
                arg = 0
            else:
                arg = args[1]

            if len(args) == 1 or arg == 0:
                return NonNegativeReals
            elif arg == 1:
                return NonNegativeIntegers
            elif arg == 2:
                return NonPositiveReals
            elif arg == 3:
                return NonPositiveIntegers
            elif arg == 4:
                return NegativeReals
            elif arg == 5:
                return NegativeIntegers
            elif arg == 6:
                return PositiveReals
            elif arg == 7:
                return PositiveIntegers
            elif arg == 8:
                return Reals
            elif arg == 9:
                return Integers
            elif arg == 10:
                return Binary
            else:
                return Reals

        self.model.x4 = Var(domain=domainRule, bounds=(-10, 10))
        self.model.y4 = Var(self.model.S, domain=domainRule, bounds=(-10, 10))
        self.model.z4 = Var(self.model.S, self.model.T, domain=domainRule, bounds=(-10, 10))

        # Add some constraints
        def makeXConRule(var):
            def xConRule(model, var):
                return (-1, var, 1)

        def makeYConRule(var):
            def yConRule(model, var, s):
                return (-1, var[s], 1)

        def makeZConRule(var):
            def zConRule(model, var, s, t):
                return (-1, var[s, t], 1)

        for n in ('1', '2', '3', '4'):
            self.model.__setattr__(
                "x" + n + "_constraint",
                Constraint(
                rule=makeXConRule(
                self.model.__getattribute__("x"+n))))

            self.model.__setattr__(
                "y" + n + "_constraint",
                Constraint(
                rule=makeYConRule(
                self.model.__getattribute__("y"+n))))

            self.model.__setattr__(
                "z" + n + "_constraint",
                Constraint(
                rule=makeZConRule(
                self.model.__getattribute__("z"+n))))

        def objRule(model):
            return sum(5*sum_product(model.__getattribute__(c+n)) \
                       for c in ('x', 'y', 'z') for n in ('1', '2', '3', '4'))

        self.model.obj = Objective(rule=objRule)

        transform = NonNegativeTransformation()
        instance=self.model.create_instance()
        transformed = transform(instance)

        opt = SolverFactory("glpk")

        instance_sol = opt.solve(instance)
        transformed_sol = opt.solve(transformed)

        self.assertEqual(
            instance_sol["Solution"][0]["Objective"]['obj']["value"],
            transformed_sol["Solution"][0]["Objective"]['obj']["value"]
            )

    @unittest.category('nightly', 'expensive')
    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
    @unittest.expectedFailure
    def test_standard_form_transform_1(self):
        self.model.S = RangeSet(0,10)
        self.model.T = Set(initialize=["foo", "bar"])

        # Unindexed, singly indexed, and doubly indexed variables with
        # explicit bounds
        self.model.x1 = Var(bounds=(-3, 3))
        self.model.y1 = Var(self.model.S, bounds=(-3, 3))
        self.model.z1 = Var(self.model.S, self.model.T, bounds=(-3, 3))

        # Unindexed, singly indexed, and doubly indexed variables with
        # rule-defined bounds
        def boundsRule(*args):
            return (-4, 4)
        self.model.x2 = Var(bounds=boundsRule)
        self.model.y2 = Var(self.model.S, bounds=boundsRule)
        self.model.z2 = Var(self.model.S, self.model.T, bounds=boundsRule)


        # Unindexed, singly indexed, and doubly indexed variables with
        # explicit domains
        self.model.x3 = Var(domain=NegativeReals, bounds=(-10, 10))
        self.model.y3 = Var(self.model.S, domain = NegativeIntegers, bounds=(-10, 10))
        self.model.z3 = Var(self.model.S, self.model.T, domain = Reals, bounds=(-10, 10))

        # Unindexed, singly indexed, and doubly indexed variables with
        # rule-defined domains
        def domainRule(*args):
            if len(args) == 1:
                arg = 0
            else:
                arg = args[1]

            if len(args) == 1 or arg == 0:
                return NonNegativeReals
            elif arg == 1:
                return NonNegativeIntegers
            elif arg == 2:
                return NonPositiveReals
            elif arg == 3:
                return NonPositiveIntegers
            elif arg == 4:
                return NegativeReals
            elif arg == 5:
                return NegativeIntegers
            elif arg == 6:
                return PositiveReals
            elif arg == 7:
                return PositiveIntegers
            elif arg == 8:
                return Reals
            elif arg == 9:
                return Integers
            elif arg == 10:
                return Binary
            else:
                return Reals

        self.model.x4 = Var(domain=domainRule, bounds=(-10, 10))
        self.model.y4 = Var(self.model.S, domain=domainRule, bounds=(-10, 10))
        self.model.z4 = Var(self.model.S, self.model.T, domain=domainRule, bounds=(-10, 10))

        def objRule(model):
            return sum(5*sum_product(model.__getattribute__(c+n)) \
                       for c in ('x', 'y', 'z') for n in ('1', '2', '3', '4'))

        self.model.obj = Objective(rule=objRule)

        transform = StandardForm()
        instance=self.model.create_instance()
        transformed = transform(instance)

        opt = SolverFactory("glpk")

        instance_sol = opt.solve(instance)
        transformed_sol = opt.solve(transformed)

        self.assertEqual(
            instance_sol["Solution"][0]["Objective"]['obj']["value"],
            transformed_sol["Solution"][0]["Objective"]['obj']["value"]
            )

    @unittest.category('nightly', 'expensive')
    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
    @unittest.expectedFailure
    def test_standard_form_transform_2(self):
        """ Same as #1, but adds constraints """
        self.model.S = RangeSet(0,10)
        self.model.T = Set(initialize=["foo", "bar"])

        # Unindexed, singly indexed, and doubly indexed variables with
        # explicit bounds
        self.model.x1 = Var(bounds=(-3, 3))
        self.model.y1 = Var(self.model.S, bounds=(-3, 3))
        self.model.z1 = Var(self.model.S, self.model.T, bounds=(-3, 3))

        # Unindexed, singly indexed, and doubly indexed variables with
        # rule-defined bounds
        def boundsRule(*args):
            return (-4, 4)
        self.model.x2 = Var(bounds=boundsRule)
        self.model.y2 = Var(self.model.S, bounds=boundsRule)
        self.model.z2 = Var(self.model.S, self.model.T, bounds=boundsRule)


        # Unindexed, singly indexed, and doubly indexed variables with
        # explicit domains
        self.model.x3 = Var(domain=NegativeReals, bounds=(-10, 10))
        self.model.y3 = Var(self.model.S, domain = NegativeIntegers, bounds=(-10, 10))
        self.model.z3 = Var(self.model.S, self.model.T, domain = Reals, bounds=(-10, 10))

        # Unindexed, singly indexed, and doubly indexed variables with
        # rule-defined domains
        def domainRule(*args):
            if len(args) == 1:
                arg = 0
            else:
                arg = args[1]

            if len(args) == 1 or arg == 0:
                return NonNegativeReals
            elif arg == 1:
                return NonNegativeIntegers
            elif arg == 2:
                return NonPositiveReals
            elif arg == 3:
                return NonPositiveIntegers
            elif arg == 4:
                return NegativeReals
            elif arg == 5:
                return NegativeIntegers
            elif arg == 6:
                return PositiveReals
            elif arg == 7:
                return PositiveIntegers
            elif arg == 8:
                return Reals
            elif arg == 9:
                return Integers
            elif arg == 10:
                return Binary
            else:
                return Reals

        self.model.x4 = Var(domain=domainRule, bounds=(-10, 10))
        self.model.y4 = Var(self.model.S, domain=domainRule, bounds=(-10, 10))
        self.model.z4 = Var(self.model.S, self.model.T, domain=domainRule, bounds=(-10, 10))

        # Add some constraints
        def makeXConRule(var):
            def xConRule(model, var):
                return (-1, var, 1)

        def makeYConRule(var):
            def yConRule(model, var, s):
                return (-1, var[s], 1)

        def makeZConRule(var):
            def zConRule(model, var, s, t):
                return (-1, var[s, t], 1)

        for n in ('1', '2', '3', '4'):
            self.model.__setattr__(
                "x" + n + "_constraint",
                Constraint(
                rule=makeXConRule(
                self.model.__getattribute__("x"+n))))

            self.model.__setattr__(
                "y" + n + "_constraint",
                Constraint(
                rule=makeYConRule(
                self.model.__getattribute__("y"+n))))

            self.model.__setattr__(
                "z" + n + "_constraint",
                Constraint(
                rule=makeZConRule(
                self.model.__getattribute__("z"+n))))

        def objRule(model):
            return sum(5*sum_product(model.__getattribute__(c+n)) \
                       for c in ('x', 'y', 'z') for n in ('1', '2', '3', '4'))

        self.model.obj = Objective(rule=objRule)

        transform = StandardForm()
        instance=self.model.create_instance()
        transformed = transform(instance)

        opt = SolverFactory("glpk")

        instance_sol = opt.solve(instance)
        transformed_sol = opt.solve(transformed)

        self.assertEqual(
            instance_sol["Solution"][0]["Objective"]['obj']["value"],
            transformed_sol["Solution"][0]["Objective"]['obj']["value"]
            )


if __name__ == "__main__":
    unittest.main()
