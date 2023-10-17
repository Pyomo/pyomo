#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from io import StringIO
from os.path import join, normpath
import pickle

from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)

from pyomo.environ import (
    BooleanVar,
    ConcreteModel,
    Constraint,
    LogicalConstraint,
    NonNegativeIntegers,
    SolverFactory,
    Suffix,
    TransformationFactory,
    value,
    Var,
)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.tests.common_tests import (
    check_linear_coef,
    check_nested_disjuncts_in_flat_gdp,
    check_obj_in_active_tree,
    check_pprint_equal,
)
from pyomo.gdp.tests.models import make_indexed_equality_model
from pyomo.repn import generate_standard_repn

gurobi_available = (
    SolverFactory('gurobi').available(exception_flag=False)
    and SolverFactory('gurobi').license_is_valid()
)
exdir = normpath(join(PYOMO_ROOT_DIR, 'examples', 'gdp'))


class LinearModelDecisionTreeExample(unittest.TestCase):
    def make_model(self):
        m = ConcreteModel()
        m.x1 = Var(bounds=(-10, 10))
        m.x2 = Var(bounds=(-20, 20))
        m.d = Var(bounds=(-1000, 1000))

        m.d1 = Disjunct()
        m.d1.x1_bounds = Constraint(expr=(0.5, m.x1, 2))
        m.d1.x2_bounds = Constraint(expr=(0.75, m.x2, 3))
        m.d1.func = Constraint(expr=m.x1 + m.x2 == m.d)

        m.d2 = Disjunct()
        m.d2.x1_bounds = Constraint(expr=(0.65, m.x1, 3))
        m.d2.x2_bounds = Constraint(expr=(3, m.x2, 10))
        m.d2.func = Constraint(expr=2 * m.x1 + 4 * m.x2 + 7 == m.d)

        m.d3 = Disjunct()
        m.d3.x1_bounds = Constraint(expr=(2, m.x1, 10))
        m.d3.x2_bounds = Constraint(expr=(0.55, m.x2, 1))
        m.d3.func = Constraint(expr=m.x1 - 5 * m.x2 - 3 == m.d)

        m.disjunction = Disjunction(expr=[m.d1, m.d2, m.d3])

        return m

    def get_Ms(self, m):
        return {
            (m.d1.x1_bounds, m.d2): (0.15, 1),
            (m.d1.x2_bounds, m.d2): (2.25, 7),
            (m.d1.x1_bounds, m.d3): (1.5, 8),
            (m.d1.x2_bounds, m.d3): (-0.2, -2),
            (m.d2.x1_bounds, m.d1): (-0.15, -1),
            (m.d2.x2_bounds, m.d1): (-2.25, -7),
            (m.d2.x1_bounds, m.d3): (1.35, 7),
            (m.d2.x2_bounds, m.d3): (-2.45, -9),
            (m.d3.x1_bounds, m.d1): (-1.5, -8),
            (m.d3.x2_bounds, m.d1): (0.2, 2),
            (m.d3.x1_bounds, m.d2): (-1.35, -7),
            (m.d3.x2_bounds, m.d2): (2.45, 9),
            (m.d1.func, m.d2): (-40, -16.65),
            (m.d1.func, m.d3): (6.3, 9),
            (m.d2.func, m.d1): (9.75, 18),
            (m.d2.func, m.d3): (16.95, 29),
            (m.d3.func, m.d1): (-21, -7.5),
            (m.d3.func, m.d2): (-103, -37.65),
        }

    def check_untightened_bounds_constraint(
        self, cons, var, parent_disj, disjunction, Ms, lower=None, upper=None
    ):
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 3)
        self.assertIsNone(cons.lower)
        self.assertEqual(value(cons.upper), 0)
        if lower is not None:
            self.assertEqual(repn.constant, lower)
            check_linear_coef(self, repn, var, -1)
            for disj in disjunction.disjuncts:
                if disj is not parent_disj:
                    check_linear_coef(
                        self, repn, disj.binary_indicator_var, Ms[disj] - lower
                    )
        if upper is not None:
            self.assertEqual(repn.constant, -upper)
            check_linear_coef(self, repn, var, 1)
            for disj in disjunction.disjuncts:
                if disj is not parent_disj:
                    check_linear_coef(
                        self, repn, disj.binary_indicator_var, -Ms[disj] + upper
                    )

    def check_all_untightened_bounds_constraints(self, m, mbm):
        # d1.x1_bounds
        cons = mbm.get_transformed_constraints(m.d1.x1_bounds)
        self.assertEqual(len(cons), 2)
        lower = cons[0]
        check_obj_in_active_tree(self, lower)
        self.check_untightened_bounds_constraint(
            lower, m.x1, m.d1, m.disjunction, {m.d2: 0.65, m.d3: 2}, lower=0.5
        )
        upper = cons[1]
        check_obj_in_active_tree(self, upper)
        self.check_untightened_bounds_constraint(
            upper, m.x1, m.d1, m.disjunction, {m.d2: 3, m.d3: 10}, upper=2
        )

        # d1.x2_bounds
        cons = mbm.get_transformed_constraints(m.d1.x2_bounds)
        self.assertEqual(len(cons), 2)
        lower = cons[0]
        check_obj_in_active_tree(self, lower)
        self.check_untightened_bounds_constraint(
            lower, m.x2, m.d1, m.disjunction, {m.d2: 3, m.d3: 0.55}, lower=0.75
        )
        upper = cons[1]
        check_obj_in_active_tree(self, upper)
        self.check_untightened_bounds_constraint(
            upper, m.x2, m.d1, m.disjunction, {m.d2: 10, m.d3: 1}, upper=3
        )

        # d2.x1_bounds
        cons = mbm.get_transformed_constraints(m.d2.x1_bounds)
        self.assertEqual(len(cons), 2)
        lower = cons[0]
        check_obj_in_active_tree(self, lower)
        self.check_untightened_bounds_constraint(
            lower, m.x1, m.d2, m.disjunction, {m.d1: 0.5, m.d3: 2}, lower=0.65
        )
        upper = cons[1]
        check_obj_in_active_tree(self, upper)
        self.check_untightened_bounds_constraint(
            upper, m.x1, m.d2, m.disjunction, {m.d1: 2, m.d3: 10}, upper=3
        )

        # d2.x2_bounds
        cons = mbm.get_transformed_constraints(m.d2.x2_bounds)
        self.assertEqual(len(cons), 2)
        lower = cons[0]
        check_obj_in_active_tree(self, lower)
        self.check_untightened_bounds_constraint(
            lower, m.x2, m.d2, m.disjunction, {m.d1: 0.75, m.d3: 0.55}, lower=3
        )
        upper = cons[1]
        self.check_untightened_bounds_constraint(
            upper, m.x2, m.d2, m.disjunction, {m.d1: 3, m.d3: 1}, upper=10
        )

        # d3.x1_bounds
        cons = mbm.get_transformed_constraints(m.d3.x1_bounds)
        self.assertEqual(len(cons), 2)
        lower = cons[0]
        self.check_untightened_bounds_constraint(
            lower, m.x1, m.d3, m.disjunction, {m.d1: 0.5, m.d2: 0.65}, lower=2
        )
        upper = cons[1]
        check_obj_in_active_tree(self, upper)
        self.check_untightened_bounds_constraint(
            upper, m.x1, m.d3, m.disjunction, {m.d1: 2, m.d2: 3}, upper=10
        )

        # d3.x2_bounds
        cons = mbm.get_transformed_constraints(m.d3.x2_bounds)
        self.assertEqual(len(cons), 2)
        lower = cons[0]
        check_obj_in_active_tree(self, lower)
        self.check_untightened_bounds_constraint(
            lower, m.x2, m.d3, m.disjunction, {m.d1: 0.75, m.d2: 3}, lower=0.55
        )
        upper = cons[1]
        check_obj_in_active_tree(self, upper)
        self.check_untightened_bounds_constraint(
            upper, m.x2, m.d3, m.disjunction, {m.d1: 3, m.d2: 10}, upper=1
        )

    def check_linear_func_constraints(self, m, mbm, Ms=None):
        if Ms is None:
            Ms = self.get_Ms(m)

        # d1.func
        cons = mbm.get_transformed_constraints(m.d1.func)
        self.assertEqual(len(cons), 2)
        lower = cons[0]
        check_obj_in_active_tree(self, lower)
        self.assertEqual(value(lower.upper), 0)
        self.assertIsNone(lower.lower)
        repn = generate_standard_repn(lower.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 5)
        self.assertEqual(repn.constant, 0)
        check_linear_coef(self, repn, m.x1, -1)
        check_linear_coef(self, repn, m.x2, -1)
        check_linear_coef(self, repn, m.d, 1)
        check_linear_coef(self, repn, m.d2.binary_indicator_var, Ms[m.d1.func, m.d2][0])
        check_linear_coef(self, repn, m.d3.binary_indicator_var, Ms[m.d1.func, m.d3][0])
        upper = cons[1]
        check_obj_in_active_tree(self, upper)
        self.assertEqual(value(upper.upper), 0)
        self.assertIsNone(upper.lower)
        repn = generate_standard_repn(upper.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 5)
        self.assertEqual(repn.constant, 0)
        check_linear_coef(self, repn, m.x1, 1)
        check_linear_coef(self, repn, m.x2, 1)
        check_linear_coef(self, repn, m.d, -1)
        check_linear_coef(
            self, repn, m.d2.binary_indicator_var, -Ms[m.d1.func, m.d2][1]
        )
        check_linear_coef(
            self, repn, m.d3.binary_indicator_var, -Ms[m.d1.func, m.d3][1]
        )

        # d2.func
        cons = mbm.get_transformed_constraints(m.d2.func)
        self.assertEqual(len(cons), 2)
        lower = cons[0]
        check_obj_in_active_tree(self, lower)
        self.assertEqual(value(lower.upper), 0)
        self.assertIsNone(lower.lower)
        repn = generate_standard_repn(lower.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 5)
        self.assertEqual(repn.constant, -7)
        check_linear_coef(self, repn, m.x1, -2)
        check_linear_coef(self, repn, m.x2, -4)
        check_linear_coef(self, repn, m.d, 1)
        check_linear_coef(self, repn, m.d1.binary_indicator_var, Ms[m.d2.func, m.d1][0])
        check_linear_coef(self, repn, m.d3.binary_indicator_var, Ms[m.d2.func, m.d3][0])
        upper = cons[1]
        check_obj_in_active_tree(self, upper)
        self.assertEqual(value(upper.upper), 0)
        self.assertIsNone(upper.lower)
        repn = generate_standard_repn(upper.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 5)
        self.assertEqual(repn.constant, 7)
        check_linear_coef(self, repn, m.x1, 2)
        check_linear_coef(self, repn, m.x2, 4)
        check_linear_coef(self, repn, m.d, -1)
        check_linear_coef(
            self, repn, m.d1.binary_indicator_var, -Ms[m.d2.func, m.d1][1]
        )
        check_linear_coef(
            self, repn, m.d3.binary_indicator_var, -Ms[m.d2.func, m.d3][1]
        )

        # d3.func
        cons = mbm.get_transformed_constraints(m.d3.func)
        self.assertEqual(len(cons), 2)
        lower = cons[0]
        check_obj_in_active_tree(self, lower)
        self.assertEqual(value(lower.upper), 0)
        self.assertIsNone(lower.lower)
        repn = generate_standard_repn(lower.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 5)
        self.assertEqual(repn.constant, 3)
        check_linear_coef(self, repn, m.x1, -1)
        check_linear_coef(self, repn, m.x2, 5)
        check_linear_coef(self, repn, m.d, 1)
        check_linear_coef(self, repn, m.d1.binary_indicator_var, Ms[m.d3.func, m.d1][0])
        check_linear_coef(self, repn, m.d2.binary_indicator_var, Ms[m.d3.func, m.d2][0])
        upper = cons[1]
        check_obj_in_active_tree(self, upper)
        self.assertEqual(value(upper.upper), 0)
        self.assertIsNone(upper.lower)
        repn = generate_standard_repn(upper.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 5)
        self.assertEqual(repn.constant, -3)
        check_linear_coef(self, repn, m.x1, 1)
        check_linear_coef(self, repn, m.x2, -5)
        check_linear_coef(self, repn, m.d, -1)
        check_linear_coef(
            self, repn, m.d1.binary_indicator_var, -Ms[m.d3.func, m.d1][1]
        )
        check_linear_coef(
            self, repn, m.d2.binary_indicator_var, -Ms[m.d3.func, m.d2][1]
        )

    @unittest.skipUnless(gurobi_available, "Gurobi is not available")
    def test_calculated_Ms_correct(self):
        # Calculating all the Ms is expensive, so we just do it in this one test
        # and then specify them for the others
        m = self.make_model()
        mbm = TransformationFactory('gdp.mbigm')
        mbm.apply_to(m, reduce_bound_constraints=False)

        self.check_all_untightened_bounds_constraints(m, mbm)
        self.check_linear_func_constraints(m, mbm)

        self.assertStructuredAlmostEqual(mbm.get_all_M_values(m), self.get_Ms(m))

    def test_transformed_constraints_correct_Ms_specified(self):
        m = self.make_model()
        mbm = TransformationFactory('gdp.mbigm')
        mbm.apply_to(m, bigM=self.get_Ms(m), reduce_bound_constraints=False)

        self.check_all_untightened_bounds_constraints(m, mbm)
        self.check_linear_func_constraints(m, mbm)

    def test_pickle_transformed_model(self):
        m = self.make_model()
        TransformationFactory('gdp.mbigm').apply_to(m, bigM=self.get_Ms(m))

        # pickle and unpickle the transformed model
        unpickle = pickle.loads(pickle.dumps(m))

        check_pprint_equal(self, m, unpickle)

    def test_mappings_between_original_and_transformed_components(self):
        m = self.make_model()
        mbm = TransformationFactory('gdp.mbigm')
        mbm.apply_to(m, bigM=self.get_Ms(m), reduce_bound_constraints=False)

        d1_block = m.d1.transformation_block
        self.assertIs(mbm.get_src_disjunct(d1_block), m.d1)
        d2_block = m.d2.transformation_block
        self.assertIs(mbm.get_src_disjunct(d2_block), m.d2)
        d3_block = m.d3.transformation_block
        self.assertIs(mbm.get_src_disjunct(d3_block), m.d3)

        for disj in [m.d1, m.d2, m.d3]:
            for comp in ['x1_bounds', 'x2_bounds', 'func']:
                original_cons = disj.component(comp)
                transformed = mbm.get_transformed_constraints(original_cons)
                for cons in transformed:
                    self.assertIn(original_cons, mbm.get_src_constraints(cons))

    def test_algebraic_constraints(self):
        m = self.make_model()
        mbm = TransformationFactory('gdp.mbigm')
        mbm.apply_to(m, bigM=self.get_Ms(m), reduce_bound_constraints=False)

        self.assertIsNotNone(m.disjunction.algebraic_constraint)
        xor = m.disjunction.algebraic_constraint
        self.assertIs(mbm.get_src_disjunction(xor), m.disjunction)

        self.assertEqual(value(xor.lower), 1)
        self.assertEqual(value(xor.upper), 1)
        repn = generate_standard_repn(xor.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(value(repn.constant), 0)
        self.assertEqual(len(repn.linear_vars), 3)
        check_linear_coef(self, repn, m.d1.binary_indicator_var, 1)
        check_linear_coef(self, repn, m.d2.binary_indicator_var, 1)
        check_linear_coef(self, repn, m.d3.binary_indicator_var, 1)
        check_obj_in_active_tree(self, xor)

    def check_pretty_bound_constraints(self, cons, var, bounds, lb):
        self.assertEqual(value(cons.upper), 0)
        self.assertIsNone(cons.lower)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), len(bounds) + 1)
        self.assertEqual(repn.constant, 0)
        if lb:
            check_linear_coef(self, repn, var, -1)
            for disj, bnd in bounds.items():
                check_linear_coef(self, repn, disj.binary_indicator_var, bnd)
        else:
            check_linear_coef(self, repn, var, 1)
            for disj, bnd in bounds.items():
                check_linear_coef(self, repn, disj.binary_indicator_var, -bnd)

    def test_bounds_constraints_correct(self):
        m = self.make_model()

        mbm = TransformationFactory('gdp.mbigm')
        mbm.apply_to(m, bigM=self.get_Ms(m), reduce_bound_constraints=True)

        # Check that all the constraints are mapped to the same transformed
        # constraints.
        cons = mbm.get_transformed_constraints(m.d1.x1_bounds)
        self.assertEqual(len(cons), 2)
        same = mbm.get_transformed_constraints(m.d2.x1_bounds)
        self.assertEqual(len(same), 2)
        self.assertIs(same[0], cons[0])
        self.assertIs(same[1], cons[1])
        sameagain = mbm.get_transformed_constraints(m.d3.x1_bounds)
        self.assertEqual(len(sameagain), 2)
        self.assertIs(sameagain[0], cons[0])
        self.assertIs(sameagain[1], cons[1])

        self.check_pretty_bound_constraints(
            cons[0], m.x1, {m.d1: 0.5, m.d2: 0.65, m.d3: 2}, lb=True
        )
        self.check_pretty_bound_constraints(
            cons[1], m.x1, {m.d1: 2, m.d2: 3, m.d3: 10}, lb=False
        )

        cons = mbm.get_transformed_constraints(m.d1.x2_bounds)
        self.assertEqual(len(cons), 2)
        same = mbm.get_transformed_constraints(m.d2.x2_bounds)
        self.assertEqual(len(same), 2)
        self.assertIs(same[0], cons[0])
        self.assertIs(same[1], cons[1])
        sameagain = mbm.get_transformed_constraints(m.d3.x2_bounds)
        self.assertEqual(len(sameagain), 2)
        self.assertIs(sameagain[0], cons[0])
        self.assertIs(sameagain[1], cons[1])

        self.check_pretty_bound_constraints(
            cons[0], m.x2, {m.d1: 0.75, m.d2: 3, m.d3: 0.55}, lb=True
        )
        self.check_pretty_bound_constraints(
            cons[1], m.x2, {m.d1: 3, m.d2: 10, m.d3: 1}, lb=False
        )

    def test_bound_constraints_correct_with_redundant_constraints(self):
        m = self.make_model()

        m.d1.bogus_x1_bounds = Constraint(expr=(0.25, m.x1, 2.25))
        mbm = TransformationFactory('gdp.mbigm')
        mbm.apply_to(m, reduce_bound_constraints=True, bigM=self.get_Ms(m))

        cons = mbm.get_transformed_constraints(m.d1.x1_bounds)
        self.assertEqual(len(cons), 2)
        self.check_pretty_bound_constraints(
            cons[0], m.x1, {m.d1: 0.5, m.d2: 0.65, m.d3: 2}, lb=True
        )
        self.check_pretty_bound_constraints(
            cons[1], m.x1, {m.d1: 2, m.d2: 3, m.d3: 10}, lb=False
        )

        cons = mbm.get_transformed_constraints(m.d1.x2_bounds)
        self.assertEqual(len(cons), 2)
        self.check_pretty_bound_constraints(
            cons[0], m.x2, {m.d1: 0.75, m.d2: 3, m.d3: 0.55}, lb=True
        )
        self.check_pretty_bound_constraints(
            cons[1], m.x2, {m.d1: 3, m.d2: 10, m.d3: 1}, lb=False
        )

    def test_Ms_specified_as_args_honored(self):
        m = self.make_model()

        Ms = self.get_Ms(m)
        # now modify some of them--these might not be valid and I don't care
        Ms[m.d2.x2_bounds, m.d3] = (-100, 100)
        Ms[m.d3.func, m.d1] = [10, 20]

        mbigm = TransformationFactory('gdp.mbigm')
        mbigm.apply_to(m, bigM=Ms, reduce_bound_constraints=False)

        self.assertStructuredAlmostEqual(mbigm.get_all_M_values(m), Ms)
        self.check_linear_func_constraints(m, mbigm, Ms)

        # Just check the constraint we should have changed
        cons = mbigm.get_transformed_constraints(m.d2.x2_bounds)
        self.assertEqual(len(cons), 2)
        # This is a little backwards because I structured these so we give the
        # bound not the value of M. So the logic here is that if we want M to
        # turn out to be -100, we need b - 3 = -100, so we pretend the bound was
        # b=-97. The same logic holds for the next one too.
        self.check_untightened_bounds_constraint(
            cons[0], m.x2, m.d2, m.disjunction, {m.d1: 0.75, m.d3: -97}, lower=3
        )
        self.check_untightened_bounds_constraint(
            cons[1], m.x2, m.d2, m.disjunction, {m.d1: 3, m.d3: 110}, upper=10
        )

    # TODO: If Suffixes allow tuple keys then we can support them and it will
    # look something like this:
    # def test_Ms_specified_as_suffixes_honored(self):
    #     m = self.make_model()
    #     m.BigM = Suffix(direction=Suffix.LOCAL)
    #     m.BigM[(m.d2.x2_bounds, m.d3)] = (-100, 100)
    #     m.d3.BigM = Suffix(direction=Suffix.LOCAL)
    #     m.d3.BigM[(m.d3.func, m.d1)] = [10, 20]

    #     arg_Ms = self.get_Ms(m)
    #     # delete the keys we replaced above
    #     del arg_Ms[m.d2.x2_bounds, m.d3]
    #     del arg_Ms[m.d3.func, m.d1]

    #     mbigm = TransformationFactory('gdp.mbigm')
    #     mbigm.apply_to(m, bigM=arg_Ms)

    #     Ms = self.get_Ms(m)
    #     self.assertStructuredAlmostEqual(mbigm.get_all_M_values(m), Ms)
    #     self.check_linear_func_constraints(m, mbigm, Ms)

    #     # Just check the constraint we should have changed
    #     cons = mbigm.get_transformed_constraints(m.d2.x2_bounds)
    #     self.assertEqual(len(cons), 2)
    #     # This is a little backwards because I structured these so we give the
    #     # bound not the value of M. So the logic here is that if we want M to
    #     # turn out to be -100, we need b - 3 = -100, so we pretend the bound was
    #     # b=-97. The same logic holds for the next one too.
    #     self.check_untightened_bounds_constraint(cons[0], m.x2, m.d2,
    #                                              m.disjunction, {m.d1: 0.75,
    #                                                              m.d3: -97},
    #                                              lower=3)
    #     self.check_untightened_bounds_constraint(cons[1], m.x2, m.d2,
    #                                              m.disjunction, {m.d1: 3,
    #                                                              m.d3: 110},
    #                                              upper=10)

    def add_fourth_disjunct(self, m):
        m.disjunction.deactivate()

        # Add a disjunct
        m.d4 = Disjunct()
        m.d4.x1_ub = Constraint(expr=m.x1 <= 8)
        m.d4.x2_lb = Constraint(expr=m.x2 >= -5)

        # Make a four-term disjunction
        m.disjunction2 = Disjunction(expr=[m.d1, m.d2, m.d3, m.d4])

    def test_deactivated_disjunct(self):
        m = self.make_model()
        # Add a new thing and deactivate it
        self.add_fourth_disjunct(m)
        m.d4.deactivate()

        mbm = TransformationFactory('gdp.mbigm')
        mbm.apply_to(m, bigM=self.get_Ms(m), reduce_bound_constraints=False)

        # we don't transform d1
        self.assertIsNone(m.d4.transformation_block)
        # and everything else is the same
        self.check_linear_func_constraints(m, mbm)
        self.check_all_untightened_bounds_constraints(m, mbm)

    @unittest.skipUnless(gurobi_available, "Gurobi is not available")
    def test_var_bounds_substituted_for_missing_bound_constraints(self):
        m = self.make_model()
        # Add a new thing with constraints that don't give both bounds on x1 and
        # x2
        self.add_fourth_disjunct(m)

        mbm = TransformationFactory('gdp.mbigm')
        # We will ignore the specified M values for the bounds constraints, but
        # issue a warning about what was unnecessary.
        out = StringIO()
        with LoggingIntercept(out, 'pyomo.gdp.mbigm'):
            mbm.apply_to(m, bigM=self.get_Ms(m), reduce_bound_constraints=True)

        warnings = out.getvalue()
        self.assertIn(
            "Unused arguments in the bigM map! "
            "These arguments were not used by the "
            "transformation:",
            warnings,
        )
        for cons, disj in [
            (m.d1.x1_bounds, m.d2),
            (m.d1.x2_bounds, m.d2),
            (m.d1.x1_bounds, m.d3),
            (m.d1.x2_bounds, m.d3),
            (m.d2.x1_bounds, m.d1),
            (m.d2.x2_bounds, m.d1),
            (m.d2.x1_bounds, m.d3),
            (m.d2.x2_bounds, m.d3),
            (m.d3.x1_bounds, m.d1),
            (m.d3.x2_bounds, m.d1),
            (m.d3.x1_bounds, m.d2),
            (m.d3.x2_bounds, m.d2),
        ]:
            self.assertIn("(%s, %s)" % (cons.name, disj.name), warnings)

        # check that the bounds constraints are right
        # for x1:
        cons = mbm.get_transformed_constraints(m.d1.x1_bounds)
        self.assertEqual(len(cons), 2)
        sameish = mbm.get_transformed_constraints(m.d4.x1_ub)
        self.assertEqual(len(sameish), 1)
        self.assertIs(sameish[0], cons[1])

        self.check_pretty_bound_constraints(
            cons[1], m.x1, {m.d1: 2, m.d2: 3, m.d3: 10, m.d4: 8}, lb=False
        )
        self.check_pretty_bound_constraints(
            cons[0], m.x1, {m.d1: 0.5, m.d2: 0.65, m.d3: 2, m.d4: -10}, lb=True
        )

        # and for x2:
        cons = mbm.get_transformed_constraints(m.d1.x2_bounds)
        self.assertEqual(len(cons), 2)
        sameish = mbm.get_transformed_constraints(m.d4.x2_lb)
        self.assertEqual(len(sameish), 1)
        self.assertIs(sameish[0], cons[0])

        self.check_pretty_bound_constraints(
            cons[1], m.x2, {m.d1: 3, m.d2: 10, m.d3: 1, m.d4: 20}, lb=False
        )
        self.check_pretty_bound_constraints(
            cons[0], m.x2, {m.d1: 0.75, m.d2: 3, m.d3: 0.55, m.d4: -5}, lb=True
        )

    def test_nested_gdp_error(self):
        m = self.make_model()
        m.d1.disjunction = Disjunction(expr=[m.x1 >= 5, m.x1 <= 4])
        with self.assertRaisesRegex(
            GDP_Error,
            "Found nested Disjunction 'd1.disjunction'. The multiple bigm "
            "transformation does not support nested GDPs. "
            "Please flatten the model before calling the "
            "transformation",
        ):
            TransformationFactory('gdp.mbigm').apply_to(m)

    @unittest.skipUnless(gurobi_available, "Gurobi is not available")
    def test_logical_constraints_on_disjuncts(self):
        m = self.make_model()
        m.d1.Y = BooleanVar()
        m.d1.Z = BooleanVar()
        m.d1.logical = LogicalConstraint(expr=m.d1.Y.implies(m.d1.Z))

        mbm = TransformationFactory('gdp.mbigm')
        mbm.apply_to(m, bigM=self.get_Ms(m), reduce_bound_constraints=False)

        y = m.d1.Y.get_associated_binary()
        z = m.d1.Z.get_associated_binary()
        z1 = m.d1._logical_to_disjunctive.auxiliary_vars[3]

        # MbigM transformation of: (1 - z1) + (1 - y) + z >= 1
        # (1 - z1) + (1 - y) + z >= 1 - d2.ind_var - d3.ind_var
        transformed = mbm.get_transformed_constraints(
            m.d1._logical_to_disjunctive.transformed_constraints[1]
        )
        self.assertEqual(len(transformed), 1)
        c = transformed[0]
        check_obj_in_active_tree(self, c)
        self.assertIsNone(c.lower)
        self.assertEqual(value(c.upper), 0)
        repn = generate_standard_repn(c.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self,
            simplified,
            -m.d2.binary_indicator_var - m.d3.binary_indicator_var + z1 + y - z - 1,
        )

        # MbigM transformation of: z1 + 1 - (1 - y) >= 1
        # z1 + y >= 1 - d2.ind_var - d3.ind_var
        transformed = mbm.get_transformed_constraints(
            m.d1._logical_to_disjunctive.transformed_constraints[2]
        )
        self.assertEqual(len(transformed), 1)
        c = transformed[0]
        check_obj_in_active_tree(self, c)
        self.assertIsNone(c.lower)
        self.assertEqual(value(c.upper), 0)
        repn = generate_standard_repn(c.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self,
            simplified,
            -m.d2.binary_indicator_var - m.d3.binary_indicator_var - y - z1 + 1,
        )

        # MbigM transformation of: z1 + 1 - z >= 1
        # z1 + 1 - z >= 1 - d2.ind_var - d3.ind_var
        transformed = mbm.get_transformed_constraints(
            m.d1._logical_to_disjunctive.transformed_constraints[3]
        )
        self.assertEqual(len(transformed), 1)
        c = transformed[0]
        check_obj_in_active_tree(self, c)
        self.assertIsNone(c.lower)
        self.assertEqual(value(c.upper), 0)
        repn = generate_standard_repn(c.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsStructurallyEqual(
            self,
            simplified,
            -m.d2.binary_indicator_var - m.d3.binary_indicator_var + z - z1,
        )

    def check_traditionally_bigmed_constraints(self, m, mbm, Ms):
        cons = mbm.get_transformed_constraints(m.d1.func)
        self.assertEqual(len(cons), 2)
        lb = cons[0]
        ub = cons[1]
        assertExpressionsEqual(
            self,
            lb.expr,
            0.0 <= m.x1 + m.x2 - m.d - Ms[m.d1][0] * (1 - m.d1.binary_indicator_var),
        )
        # [ESJ 11/23/22]: It's really hard to use assertExpressionsEqual on the
        # ub constraints because SumExpressions are sharing args, I think. So
        # when they get constructed in the transformation (because they come
        # after the lb constraints), there are nested SumExpressions. Instead of
        # trying to reproduce them I am just building a "flat" SumExpression
        # with generate_standard_repn and comparing that.
        self.assertIsNone(ub.lower)
        self.assertEqual(ub.upper, 0)
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsEqual(
            self,
            simplified,
            m.x1 + m.x2 - m.d + Ms[m.d1][1] * m.d1.binary_indicator_var - Ms[m.d1][1],
        )

        cons = mbm.get_transformed_constraints(m.d2.func)
        self.assertEqual(len(cons), 2)
        lb = cons[0]
        ub = cons[1]
        assertExpressionsEqual(
            self,
            lb.expr,
            0.0
            <= 2 * m.x1
            + 4 * m.x2
            + 7
            - m.d
            - Ms[m.d2][0] * (1 - m.d2.binary_indicator_var),
        )
        self.assertIsNone(ub.lower)
        self.assertEqual(ub.upper, 0)
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsEqual(
            self,
            simplified,
            2 * m.x1
            + 4 * m.x2
            - m.d
            + Ms[m.d2][1] * m.d2.binary_indicator_var
            - (Ms[m.d2][1] - 7),
        )

        cons = mbm.get_transformed_constraints(m.d3.func)
        self.assertEqual(len(cons), 2)
        lb = cons[0]
        ub = cons[1]
        assertExpressionsEqual(
            self,
            lb.expr,
            0.0
            <= m.x1
            - 5 * m.x2
            - 3
            - m.d
            - Ms[m.d3][0] * (1 - m.d3.binary_indicator_var),
        )
        self.assertIsNone(ub.lower)
        self.assertEqual(ub.upper, 0)
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        simplified = repn.constant + sum(
            repn.linear_coefs[i] * repn.linear_vars[i]
            for i in range(len(repn.linear_vars))
        )
        assertExpressionsEqual(
            self,
            simplified,
            m.x1
            - 5 * m.x2
            - m.d
            + Ms[m.d3][1] * m.d3.binary_indicator_var
            - (Ms[m.d3][1] + 3),
        )

    def test_only_multiple_bigm_bound_constraints(self):
        m = self.make_model()
        mbm = TransformationFactory('gdp.mbigm')
        mbm.apply_to(m, only_mbigm_bound_constraints=True)

        cons = mbm.get_transformed_constraints(m.d1.x1_bounds)
        self.assertEqual(len(cons), 2)
        self.check_pretty_bound_constraints(
            cons[0], m.x1, {m.d1: 0.5, m.d2: 0.65, m.d3: 2}, lb=True
        )
        self.check_pretty_bound_constraints(
            cons[1], m.x1, {m.d1: 2, m.d2: 3, m.d3: 10}, lb=False
        )

        cons = mbm.get_transformed_constraints(m.d1.x2_bounds)
        self.assertEqual(len(cons), 2)
        self.check_pretty_bound_constraints(
            cons[0], m.x2, {m.d1: 0.75, m.d2: 3, m.d3: 0.55}, lb=True
        )
        self.check_pretty_bound_constraints(
            cons[1], m.x2, {m.d1: 3, m.d2: 10, m.d3: 1}, lb=False
        )

        self.check_traditionally_bigmed_constraints(
            m,
            mbm,
            {m.d1: (-1030.0, 1030.0), m.d2: (-1093.0, 1107.0), m.d3: (-1113.0, 1107.0)},
        )

    def test_only_multiple_bigm_bound_constraints_arg_Ms(self):
        m = self.make_model()
        mbm = TransformationFactory('gdp.mbigm')
        Ms = {m.d1: 1050, m.d2.func: (-2000, 1200), None: 4000}
        mbm.apply_to(m, only_mbigm_bound_constraints=True, bigM=Ms)

        cons = mbm.get_transformed_constraints(m.d1.x1_bounds)
        self.assertEqual(len(cons), 2)
        self.check_pretty_bound_constraints(
            cons[0], m.x1, {m.d1: 0.5, m.d2: 0.65, m.d3: 2}, lb=True
        )
        self.check_pretty_bound_constraints(
            cons[1], m.x1, {m.d1: 2, m.d2: 3, m.d3: 10}, lb=False
        )

        cons = mbm.get_transformed_constraints(m.d1.x2_bounds)
        self.assertEqual(len(cons), 2)
        self.check_pretty_bound_constraints(
            cons[0], m.x2, {m.d1: 0.75, m.d2: 3, m.d3: 0.55}, lb=True
        )
        self.check_pretty_bound_constraints(
            cons[1], m.x2, {m.d1: 3, m.d2: 10, m.d3: 1}, lb=False
        )

        self.check_traditionally_bigmed_constraints(
            m, mbm, {m.d1: (-1050, 1050), m.d2: (-2000, 1200), m.d3: (-4000, 4000)}
        )


@unittest.skipUnless(gurobi_available, "Gurobi is not available")
class NestedDisjunctsInFlatGDP(unittest.TestCase):
    """
    This class tests the fix for #2702
    """

    def test_declare_disjuncts_in_disjunction_rule(self):
        check_nested_disjuncts_in_flat_gdp(self, 'bigm')


@unittest.skipUnless(gurobi_available, "Gurobi is not available")
class IndexedDisjunction(unittest.TestCase):
    def test_two_term_indexed_disjunction(self):
        """
        This test checks that we don't do anything silly with transformation Blocks in
        the case that the Disjunction is indexed.
        """
        m = make_indexed_equality_model()

        mbm = TransformationFactory('gdp.mbigm')
        mbm.apply_to(m)

        cons = mbm.get_transformed_constraints(m.d[1].disjuncts[0].constraint[1])
        self.assertEqual(len(cons), 2)
        assertExpressionsEqual(
            self,
            cons[0].expr,
            m.x[1]
            >= m.d[1].disjuncts[0].binary_indicator_var
            + 2.0 * m.d[1].disjuncts[1].binary_indicator_var,
        )
        assertExpressionsEqual(
            self,
            cons[1].expr,
            m.x[1]
            <= m.d[1].disjuncts[0].binary_indicator_var
            + 2.0 * m.d[1].disjuncts[1].binary_indicator_var,
        )
        cons_again = mbm.get_transformed_constraints(m.d[1].disjuncts[1].constraint[1])
        self.assertEqual(len(cons_again), 2)
        self.assertIs(cons_again[0], cons[0])
        self.assertIs(cons_again[1], cons[1])

        cons = mbm.get_transformed_constraints(m.d[2].disjuncts[0].constraint[1])
        self.assertEqual(len(cons), 2)
        assertExpressionsEqual(
            self,
            cons[0].expr,
            m.x[2]
            >= m.d[2].disjuncts[0].binary_indicator_var
            + 2.0 * m.d[2].disjuncts[1].binary_indicator_var,
        )
        assertExpressionsEqual(
            self,
            cons[1].expr,
            m.x[2]
            <= m.d[2].disjuncts[0].binary_indicator_var
            + 2.0 * m.d[2].disjuncts[1].binary_indicator_var,
        )
        cons_again = mbm.get_transformed_constraints(m.d[2].disjuncts[1].constraint[1])
        self.assertEqual(len(cons_again), 2)
        self.assertIs(cons_again[0], cons[0])
        self.assertIs(cons_again[1], cons[1])
