#
# Unit Tests for model transformations
#

import os
import sys
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.services
from pyomo.environ import *
from pyomo.opt import *
from pyomo.util.plugin import Plugin

solver = load_solvers('glpk')


class Test(unittest.TestCase):

    def setUp(self):
        self.model = AbstractModel()

    def tearDown(self):
        if os.path.exists("unknown.lp"):
            os.unlink("unknown.lp")
        pyutilib.services.TempfileManager.clear_tempfiles()

    @staticmethod
    def nonnegativeBounds(var):
        # Either the bounds or domain must enforce nonnegativity
        if var.lb is not None and var.lb >= 0:
            return True
        elif var.domain is not None and var.domain.bounds()[0] >= 0:
            return True
        else:
            return False

    @unittest.expectedFailure
    def test_relax_integrality1(self):
        """ Coverage of the _clear_attribute method """
        self.model.A = RangeSet(1,4)
        self.model.a = Var()
        self.model.b = Var(within=self.model.A)
        self.model.c = Var(within=NonNegativeIntegers)
        self.model.d = Var(within=Integers, bounds=(-2,3))
        self.model.e = Var(within=Boolean)
        self.model.f = Var(domain=Boolean)
        instance=self.model.create()
        rinst = apply_transformation('relax_integrality',instance)
        self.assertEqual(type(rinst.a.domain), type(Reals))
        self.assertEqual(type(rinst.b.domain), type(Reals))
        self.assertEqual(type(rinst.c.domain), type(Reals))
        self.assertEqual(type(rinst.d.domain), type(Reals))
        self.assertEqual(type(rinst.e.domain), type(Reals))
        self.assertEqual(type(rinst.f.domain), type(Reals))
        self.assertEqual(rinst.a.bounds, instance.a.bounds)
        self.assertEqual(rinst.b.bounds, instance.b.bounds)
        self.assertEqual(rinst.c.bounds, instance.c.bounds)
        self.assertEqual(rinst.d.bounds, instance.d.bounds)
        self.assertEqual(rinst.e.bounds, instance.e.bounds)
        self.assertEqual(rinst.f.bounds, instance.f.bounds)

    def test_apply_transformation1(self):
        self.assertTrue('base.relax_integrality' in apply_transformation())

    def test_apply_transformation2(self):
        self.assertEqual(apply_transformation('foo'),None)
        self.assertTrue(isinstance(apply_transformation('base.relax_integrality'),Plugin))
        self.assertTrue(isinstance(apply_transformation('base.relax_integrality'),Plugin))
        self.assertEqual(apply_transformation('foo', self.model),None)

    def test_nonnegativity_transformation_1(self):
        self.model.a = Var()
        self.model.b = Var(within=NonNegativeIntegers)
        self.model.c = Var(within=Integers, bounds=(-2,3))
        self.model.d = Var(within=Boolean)
        self.model.e = Var(domain=Boolean)

        instance=self.model.create()
        transformed = instance.transform('base.nonnegative_vars')

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

        instance = self.model.create()
        transformed = instance.transform('base.nonnegative_vars')

        # Make sure everything is nonnegative
        for c in ('x', 'y', 'z'):
            for n in ('1', '2', '3', '4'):
                var = transformed.__getattribute__(c+n)
                for ndx in var._index:
                    self.assertTrue(self.nonnegativeBounds(var[ndx]))

    @unittest.skipIf(solver['glpk'] is None, "glpk solver is not available")
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
            return sum(5*summation(model.__getattribute__(c+n)) \
                       for c in ('x', 'y', 'z') for n in ('1', '2', '3', '4'))

        self.model.obj = Objective(rule=objRule)

        transform = NonNegativeTransformation()
        instance=self.model.create()
        #transformed = apply_transformation("nonnegative_vars", instance)
        transformed = transform(instance)

        opt = solver["glpk"]

        instance_sol = opt.solve(instance)
        transformed_sol = opt.solve(transformed)

        self.assertEqual(
            instance_sol["Solution"][0]["Objective"]['obj']["value"],
            transformed_sol["Solution"][0]["Objective"]['obj']["value"]
            )

    @unittest.skipIf(solver['glpk'] is None, "glpk solver is not available")
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
            return sum(5*summation(model.__getattribute__(c+n)) \
                       for c in ('x', 'y', 'z') for n in ('1', '2', '3', '4'))

        self.model.obj = Objective(rule=objRule)

        transform = NonNegativeTransformation()
        instance=self.model.create()
        #transformed = apply_transformation("nonnegative_vars", instance)
        transformed = transform(instance)

        opt = solver["glpk"]

        instance_sol = opt.solve(instance)
        transformed_sol = opt.solve(transformed)

        self.assertEqual(
            instance_sol["Solution"][0]["Objective"]['obj']["value"],
            transformed_sol["Solution"][0]["Objective"]['obj']["value"]
            )

    @unittest.category('nightly')
    @unittest.skipIf(solver['glpk'] is None, "glpk solver is not available")
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
            return sum(5*summation(model.__getattribute__(c+n)) \
                       for c in ('x', 'y', 'z') for n in ('1', '2', '3', '4'))

        self.model.obj = Objective(rule=objRule)

        transform = StandardForm()
        instance=self.model.create()
        #transformed = apply_transformation("nonnegative_vars", instance)
        transformed = transform(instance)

        opt = solver["glpk"]

        instance_sol = opt.solve(instance)
        transformed_sol = opt.solve(transformed)

        self.assertEqual(
            instance_sol["Solution"][0]["Objective"]['obj']["value"],
            transformed_sol["Solution"][0]["Objective"]['obj']["value"]
            )

    @unittest.category('nightly')
    @unittest.skipIf(solver['glpk'] is None, "glpk solver is not available")
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
            return sum(5*summation(model.__getattribute__(c+n)) \
                       for c in ('x', 'y', 'z') for n in ('1', '2', '3', '4'))

        self.model.obj = Objective(rule=objRule)

        transform = StandardForm()
        instance=self.model.create()
        #transformed = apply_transformation("nonnegative_vars", instance)
        transformed = transform(instance)

        opt = solver["glpk"]

        instance_sol = opt.solve(instance)
        transformed_sol = opt.solve(transformed)

        self.assertEqual(
            instance_sol["Solution"][0]["Objective"]['obj']["value"],
            transformed_sol["Solution"][0]["Objective"]['obj']["value"]
            )


if __name__ == "__main__":
    unittest.main()
