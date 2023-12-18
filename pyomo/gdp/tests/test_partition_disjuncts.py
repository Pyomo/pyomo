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

import pyomo.common.unittest as unittest
from pyomo.environ import (
    TransformationFactory,
    Constraint,
    ConcreteModel,
    Var,
    RangeSet,
    Objective,
    maximize,
    SolverFactory,
    Any,
    Reference,
    LogicalConstraint,
)
from pyomo.core.expr.logical_expr import (
    EquivalenceExpression,
    NotExpression,
    AndExpression,
    ExactlyExpression,
)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error, check_model_algebraic
from pyomo.gdp.plugins.partition_disjuncts import (
    arbitrary_partition,
    compute_optimal_bounds,
    compute_fbbt_bounds,
)
from pyomo.core import Block, value
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.common_tests as ct
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.opt import check_available_solvers


solvers = check_available_solvers('gurobi_direct')


class CommonTests:
    def diff_apply_to_and_create_using(self, model, **kwargs):
        ct.diff_apply_to_and_create_using(
            self, model, 'gdp.partition_disjuncts', **kwargs
        )


class PaperTwoCircleExample(unittest.TestCase, CommonTests):
    def check_disj_constraint(self, c1, upper, auxVar1, auxVar2):
        self.assertIsNone(c1.lower)
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

    def check_aux_var_bounds(
        self,
        aux_vars1,
        aux_vars2,
        aux11lb,
        aux11ub,
        aux12lb,
        aux12ub,
        aux21lb,
        aux21ub,
        aux22lb,
        aux22ub,
    ):
        self.assertEqual(len(aux_vars1), 2)
        # Gurobi default constraint tolerance is 1e-6, so let's say that's
        # our goal too. Have to tighten Gurobi's tolerance to even get here
        # though...
        self.assertAlmostEqual(aux_vars1[0].lb, aux11lb, places=6)
        self.assertAlmostEqual(aux_vars1[0].ub, aux11ub, places=6)
        self.assertAlmostEqual(aux_vars1[1].lb, aux12lb, places=6)
        self.assertAlmostEqual(aux_vars1[1].ub, aux12ub, places=6)

        self.assertAlmostEqual(len(aux_vars2), 2)
        self.assertAlmostEqual(aux_vars2[0].lb, aux21lb, places=6)
        self.assertAlmostEqual(aux_vars2[0].ub, aux21ub, places=6)
        self.assertAlmostEqual(aux_vars2[1].lb, aux22lb, places=6)
        self.assertAlmostEqual(aux_vars2[1].ub, aux22ub, places=6)

    def check_transformation_block_disjuncts_and_constraints(
        self, m, original_disjunction, disjunction_name=None
    ):
        b = m.component("_pyomo_gdp_partition_disjuncts_reformulation")
        self.assertIsInstance(b, Block)

        # check we declared the right things
        self.assertEqual(len(b.component_map(Disjunction)), 1)
        self.assertEqual(len(b.component_map(Disjunct)), 2)
        # global constraints:
        self.assertEqual(len(b.component_map(Constraint)), 2)
        # equivalence constraints between old and new Disjunct indicator_vars
        self.assertEqual(len(b.component_map(LogicalConstraint)), 1)

        if disjunction_name is None:
            disjunction = b.disjunction
        else:
            disjunction = b.component(disjunction_name)
        self.assertEqual(len(disjunction.disjuncts), 2)
        # each Disjunct has one constraint
        disj1 = disjunction.disjuncts[0]
        disj2 = disjunction.disjuncts[1]
        self.assertEqual(len(disj1.component_map(Constraint)), 1)
        self.assertEqual(len(disj2.component_map(Constraint)), 1)

        # check the logical equivalence constraints
        equivalence = b.component("indicator_var_equalities")
        self.assertIsInstance(equivalence, LogicalConstraint)
        self.assertEqual(len(equivalence), 2)
        for i, variables in enumerate(
            [
                (original_disjunction.disjuncts[0].indicator_var, disj1.indicator_var),
                (original_disjunction.disjuncts[1].indicator_var, disj2.indicator_var),
            ]
        ):
            cons = equivalence[i]
            self.assertIsInstance(cons.body, EquivalenceExpression)
            self.assertIs(cons.body.args[0], variables[0])
            self.assertIs(cons.body.args[1], variables[1])

        return b, disj1, disj2

    def check_transformation_block_structure(
        self, m, aux11lb, aux11ub, aux12lb, aux12ub, aux21lb, aux21ub, aux22lb, aux22ub
    ):
        (b, disj1, disj2) = self.check_transformation_block_disjuncts_and_constraints(
            m, m.disjunction
        )

        # each Disjunct has two variables declared on it (aux vars and indicator
        # var), plus a reference to the indicator_var from the original Disjunct
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)

        aux_vars1 = disj1.component("disjunction_disjuncts[0].constraint[1]_aux_vars")
        aux_vars2 = disj2.component("disjunction_disjuncts[1].constraint[1]_aux_vars")
        self.check_aux_var_bounds(
            aux_vars1,
            aux_vars2,
            aux11lb,
            aux11ub,
            aux12lb,
            aux12ub,
            aux21lb,
            aux21ub,
            aux22lb,
            aux22ub,
        )

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

    def check_transformation_block(
        self,
        m,
        aux11lb,
        aux11ub,
        aux12lb,
        aux12ub,
        aux21lb,
        aux21ub,
        aux22lb,
        aux22ub,
        partitions,
    ):
        (
            b,
            disj1,
            disj2,
            aux_vars1,
            aux_vars2,
        ) = self.check_transformation_block_structure(
            m, aux11lb, aux11ub, aux12lb, aux12ub, aux21lb, aux21ub, aux22lb, aux22ub
        )

        self.check_disjunct_constraints(disj1, disj2, aux_vars1, aux_vars2)

        # check the global constraints
        c = b.component("disjunction_disjuncts[0].constraint[1]_split_constraints")
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj1(
            c1, aux_vars1[0], partitions[0][0], partitions[0][1]
        )
        c2 = c[1]
        self.check_global_constraint_disj1(
            c2, aux_vars1[1], partitions[1][0], partitions[1][1]
        )

        c = b.component("disjunction_disjuncts[1].constraint[1]_split_constraints")
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj2(
            c1, aux_vars2[0], partitions[0][0], partitions[0][1]
        )
        c2 = c[1]
        self.check_global_constraint_disj2(
            c2, aux_vars2[1], partitions[1][0], partitions[1][1]
        )

    def test_transformation_block_fbbt_bounds(self):
        m = models.makeBetweenStepsPaperExample()

        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            compute_bounds_method=compute_fbbt_bounds,
        )

        self.check_transformation_block(
            m,
            0,
            72,
            0,
            72,
            -72,
            96,
            -72,
            96,
            partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
        )

    def check_transformation_block_indexed_var_on_disjunct(
        self, m, original_disjunction
    ):
        (b, disj1, disj2) = self.check_transformation_block_disjuncts_and_constraints(
            m, original_disjunction
        )

        # Has it's own indicator var, a Reference to the original Disjunct's
        # indicator var, the aux vars, and the Reference to x
        self.assertEqual(len(disj1.component_map(Var)), 4)
        # Same as above minus the Reference to x
        self.assertEqual(len(disj2.component_map(Var)), 3)

        aux_vars1 = disj1.component("disj1.c_aux_vars")
        aux_vars2 = disj2.component("disj2.c_aux_vars")
        self.check_aux_var_bounds(aux_vars1, aux_vars2, 0, 72, 0, 72, -72, 96, -72, 96)

        # check the transformed constraints on the disjuncts
        c = disj1.component("disj1.c")
        self.assertEqual(len(c), 1)
        c1 = c[0]
        self.check_disj_constraint(c1, 1, aux_vars1[0], aux_vars1[1])
        c = disj2.component("disj2.c")
        self.assertEqual(len(c), 1)
        c2 = c[0]
        self.check_disj_constraint(c2, -35, aux_vars2[0], aux_vars2[1])

        # check the global constraints
        c = b.component("disj1.c_split_constraints")
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj1(c1, aux_vars1[0], m.disj1.x[1], m.disj1.x[2])
        c2 = c[1]
        self.check_global_constraint_disj1(c2, aux_vars1[1], m.disj1.x[3], m.disj1.x[4])

        c = b.component("disj2.c_split_constraints")
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj2(c1, aux_vars2[0], m.disj1.x[1], m.disj1.x[2])
        c2 = c[1]
        self.check_global_constraint_disj2(c2, aux_vars2[1], m.disj1.x[3], m.disj1.x[4])

        return b, disj1, disj2

    def test_transformation_block_indexed_var_on_disjunct(self):
        m = models.makeBetweenStepsPaperExample_DeclareVarOnDisjunct()

        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions=[
                [m.disj1.x[1], m.disj1.x[2]],
                [m.disj1.x[3], m.disj1.x[4]],
            ],
            compute_bounds_method=compute_fbbt_bounds,
        )

        self.check_transformation_block_indexed_var_on_disjunct(m, m.disjunction)

    def check_transformation_block_nested_disjunction(
        self, m, disj2, x, disjunction_block=None
    ):
        if disjunction_block is None:
            block_prefix = ""
            disjunction_parent = m
        else:
            block_prefix = disjunction_block + "."
            disjunction_parent = m.component(disjunction_block)
        (
            inner_b,
            inner_disj1,
            inner_disj2,
        ) = self.check_transformation_block_disjuncts_and_constraints(
            disj2,
            disjunction_parent.disj2.disjunction,
            "%sdisj2.disjunction" % block_prefix,
        )

        # Has it's own indicator var, the aux vars, and the Reference to the
        # original indicator_var
        self.assertEqual(len(inner_disj1.component_map(Var)), 3)
        self.assertEqual(len(inner_disj2.component_map(Var)), 3)

        aux_vars1 = inner_disj1.component(
            "%sdisj2.disjunction_disjuncts[0].constraint[1]_aux_vars" % block_prefix
        )
        aux_vars2 = inner_disj2.component(
            "%sdisj2.disjunction_disjuncts[1].constraint[1]_aux_vars" % block_prefix
        )
        self.check_aux_var_bounds(aux_vars1, aux_vars2, 0, 72, 0, 72, -72, 96, -72, 96)

        # check the transformed constraints on the disjuncts
        c = inner_disj1.component(
            "%sdisj2.disjunction_disjuncts[0].constraint[1]" % block_prefix
        )
        self.assertEqual(len(c), 1)
        c1 = c[0]
        self.check_disj_constraint(c1, 1, aux_vars1[0], aux_vars1[1])
        c = inner_disj2.component(
            "%sdisj2.disjunction_disjuncts[1].constraint[1]" % block_prefix
        )
        self.assertEqual(len(c), 1)
        c2 = c[0]
        self.check_disj_constraint(c2, -35, aux_vars2[0], aux_vars2[1])

        # check the global constraints
        c = inner_b.component(
            "%sdisj2.disjunction_disjuncts[0].constraint[1]"
            "_split_constraints" % block_prefix
        )
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj1(c1, aux_vars1[0], x[1], x[2])
        c2 = c[1]
        self.check_global_constraint_disj1(c2, aux_vars1[1], x[3], x[4])

        c = inner_b.component(
            "%sdisj2.disjunction_disjuncts[1].constraint[1]"
            "_split_constraints" % block_prefix
        )
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj2(c1, aux_vars2[0], x[1], x[2])
        c2 = c[1]
        self.check_global_constraint_disj2(c2, aux_vars2[1], x[3], x[4])

    def test_transformation_block_nested_disjunction(self):
        m = models.makeBetweenStepsPaperExample_Nested()

        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions=[
                [m.disj1.x[1], m.disj1.x[2]],
                [m.disj1.x[3], m.disj1.x[4]],
            ],
            compute_bounds_method=compute_fbbt_bounds,
        )

        # everything for the outer disjunction should look exactly the same as
        # the test above:
        (b, disj1, disj2) = self.check_transformation_block_indexed_var_on_disjunct(
            m, m.disjunction
        )

        # AND, we should have a transformed inner disjunction on disj2:
        self.check_transformation_block_nested_disjunction(m, disj2, m.disj1.x)

    def test_transformation_block_nested_disjunction_outer_disjunction_target(self):
        """We should get identical behavior to the previous test if we
        specify the outer disjunction as the target"""
        m = models.makeBetweenStepsPaperExample_Nested()

        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            targets=m.disjunction,
            variable_partitions=[
                [m.disj1.x[1], m.disj1.x[2]],
                [m.disj1.x[3], m.disj1.x[4]],
            ],
            compute_bounds_method=compute_fbbt_bounds,
        )

        # everything for the outer disjunction should look exactly the same as
        # the test above:
        (b, disj1, disj2) = self.check_transformation_block_indexed_var_on_disjunct(
            m, m.disjunction
        )

        # AND, we should have a transformed inner disjunction on disj2:
        self.check_transformation_block_nested_disjunction(m, disj2, m.disj1.x)

    def test_transformation_block_nested_disjunction_badly_ordered_targets(self):
        """This tests that we preprocess targets correctly because we don't
        want to double transform the inner disjunct, which is what would happen
        if we did things in the order given."""
        m = models.makeBetweenStepsPaperExample_Nested()

        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            targets=[m.disj2, m.disjunction],
            variable_partitions=[
                [m.disj1.x[1], m.disj1.x[2]],
                [m.disj1.x[3], m.disj1.x[4]],
            ],
            compute_bounds_method=compute_fbbt_bounds,
        )

        # everything for the outer disjunction should look exactly the same as
        # the test above:
        (b, disj1, disj2) = self.check_transformation_block_indexed_var_on_disjunct(
            m, m.disjunction
        )

        # AND, we should have a transformed inner disjunction on disj2:
        self.check_transformation_block_nested_disjunction(m, disj2, m.disj1.x)

    def check_hierarchical_nested_model(self, m):
        (b, disj1, disj2) = self.check_transformation_block_disjuncts_and_constraints(
            m.disjunction_block,
            m.disjunction_block.disjunction,
            "disjunction_block.disjunction",
        )
        # each Disjunct has two variables declared on it (aux vars and indicator
        # var), plus a reference to the indicator_var from the original Disjunct
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)

        aux_vars1 = disj1.component("disj1.c_aux_vars")
        aux_vars2 = disj2.component("disjunct_block.disj2.c_aux_vars")
        self.check_aux_var_bounds(aux_vars1, aux_vars2, 0, 72, 0, 72, -72, 96, -72, 96)
        # check the transformed constraints on the disjuncts
        c = disj1.component("disj1.c")
        self.assertEqual(len(c), 1)
        c1 = c[0]
        self.check_disj_constraint(c1, 1, aux_vars1[0], aux_vars1[1])
        c = disj2.component("disjunct_block.disj2.c")
        self.assertEqual(len(c), 1)
        c2 = c[0]
        self.check_disj_constraint(c2, -35, aux_vars2[0], aux_vars2[1])

        # check the global constraints
        c = b.component("disj1.c_split_constraints")
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj1(c1, aux_vars1[0], m.x[1], m.x[2])
        c2 = c[1]
        self.check_global_constraint_disj1(c2, aux_vars1[1], m.x[3], m.x[4])

        c = b.component("disjunct_block.disj2.c_split_constraints")
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj2(c1, aux_vars2[0], m.x[1], m.x[2])
        c2 = c[1]
        self.check_global_constraint_disj2(c2, aux_vars2[1], m.x[3], m.x[4])

        # check the inner disjunction
        self.check_transformation_block_nested_disjunction(
            m, disj2, m.x, "disjunct_block"
        )

    def test_hierarchical_nested_badly_ordered_targets(self):
        m = models.makeHierarchicalNested_DeclOrderMatchesInstantiationOrder()

        # If we don't preprocess targets by actually finding who is nested in
        # who, this would force the Disjunct to be transformed before its
        # Disjunction because they are hidden on blocks. Then this would fail
        # because the partition doesn't specify what to do with the auxiliary
        # variables created by the inner disjunction. If we correctly descend
        # into Blocks and order according to the nesting structure, all will be
        # well.
        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            targets=[m.disjunction_block, m.disjunct_block.disj2],
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            compute_bounds_method=compute_fbbt_bounds,
        )

        self.check_hierarchical_nested_model(m)

    def test_hierarchical_nested_decl_order_opposite_instantiation_order(self):
        m = models.makeHierarchicalNested_DeclOrderOppositeInstantiationOrder()
        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            compute_bounds_method=compute_fbbt_bounds,
        )

        self.check_hierarchical_nested_model(m)

    def test_transformation_block_nested_disjunction_target(self):
        m = models.makeBetweenStepsPaperExample_Nested()

        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            targets=m.disj2.disjunction,
            variable_partitions=[
                [m.disj1.x[1], m.disj1.x[2]],
                [m.disj1.x[3], m.disj1.x[4]],
            ],
            compute_bounds_method=compute_fbbt_bounds,
        )

        self.check_transformation_block_nested_disjunction(m, m.disj2, m.disj1.x)
        # NOTE: If you then transformed the whole model (or the outer
        # disjunction), you would double-transform in the sense that you would
        # again transform the Disjunction this creates. But I think it serves
        # you right, and this is still the correct behavior. There's nothing we
        # can do about you manually transforming from in to out--there's no way
        # for us to know. It is confusing though since bigm and hull need to go
        # from the leaves up and this is opposite.

    @unittest.skipIf(
        'gurobi_direct' not in solvers, 'Gurobi direct solver not available'
    )
    def test_transformation_block_optimized_bounds(self):
        m = models.makeBetweenStepsPaperExample()

        # I'm using Gurobi because I'm assuming exact equality is going to work
        # out. And it definitely won't with ipopt. (And Gurobi direct is way
        # faster than the LP interfeace to Gurobi for this... I assume because
        # writing nonlinear expressions is slow?)
        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            compute_bounds_solver=SolverFactory('gurobi_direct'),
            compute_bounds_method=compute_optimal_bounds,
        )

        self.check_transformation_block(
            m,
            0,
            72,
            0,
            72,
            -18,
            32,
            -18,
            32,
            partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
        )

    def test_no_solver_error(self):
        m = models.makeBetweenStepsPaperExample()

        with self.assertRaisesRegex(
            GDP_Error,
            "No solver was specified to optimize the "
            "subproblems for computing expression "
            "bounds! "
            "Please specify a configured solver in the "
            "'compute_bounds_solver' argument if using "
            "'compute_optimal_bounds.'",
        ):
            TransformationFactory('gdp.partition_disjuncts').apply_to(
                m,
                variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
                compute_bounds_method=compute_optimal_bounds,
            )

    @unittest.skipIf(
        'gurobi_direct' not in solvers, 'Gurobi direct solver not available'
    )
    def test_transformation_block_better_bounds_in_global_constraints(self):
        m = models.makeBetweenStepsPaperExample()
        m.c1 = Constraint(expr=m.x[1] ** 2 + m.x[2] ** 2 <= 32)
        m.c2 = Constraint(expr=m.x[3] ** 2 + m.x[4] ** 2 <= 32)
        m.c3 = Constraint(expr=(3 - m.x[1]) ** 2 + (3 - m.x[2]) ** 2 <= 32)
        m.c4 = Constraint(expr=(3 - m.x[3]) ** 2 + (3 - m.x[4]) ** 2 <= 32)
        opt = SolverFactory('gurobi_direct')
        opt.options['NonConvex'] = 2
        opt.options['FeasibilityTol'] = 1e-8

        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            compute_bounds_solver=opt,
            compute_bounds_method=compute_optimal_bounds,
        )

        self.check_transformation_block(
            m,
            0,
            32,
            0,
            32,
            -18,
            14,
            -18,
            14,
            partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
        )

    @unittest.skipIf(
        'gurobi_direct' not in solvers, 'Gurobi direct solver not available'
    )
    def test_transformation_block_arbitrary_even_partition(self):
        m = models.makeBetweenStepsPaperExample()

        # I'm using Gurobi because I'm assuming exact equality is going to work
        # out. And it definitely won't with ipopt. (And Gurobi direct is way
        # faster than the LP interface to Gurobi for this... I assume because
        # writing nonlinear expressions is slow?)
        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            num_partitions=2,
            compute_bounds_solver=SolverFactory('gurobi_direct'),
            compute_bounds_method=compute_optimal_bounds,
        )
        # The above will partition as [[x[1], x[3]], [x[2], x[4]]]
        self.check_transformation_block(
            m,
            0,
            72,
            0,
            72,
            -18,
            32,
            -18,
            32,
            partitions=[[m.x[1], m.x[3]], [m.x[2], m.x[4]]],
        )

    @unittest.skipIf(
        'gurobi_direct' not in solvers, 'Gurobi direct solver not available'
    )
    def test_assume_fixed_vars_not_permanent(self):
        m = models.makeBetweenStepsPaperExample()
        m.x[1].fix(0)
        m.disjunction.disjuncts[0].indicator_var.fix(True)

        # I'm using Gurobi because I'm assuming exact equality is going to work
        # out. And it definitely won't with ipopt. (And Gurobi direct is way
        # faster than the LP interface to Gurobi for this... I assume because
        # writing nonlinear expressions is slow?)
        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            assume_fixed_vars_permanent=False,
            compute_bounds_solver=SolverFactory('gurobi_direct'),
            compute_bounds_method=compute_optimal_bounds,
        )

        self.assertTrue(m.x[1].fixed)
        self.assertEqual(value(m.x[1]), 0)
        self.assertTrue(m.disjunction_disjuncts[0].indicator_var.fixed)
        self.assertTrue(value(m.disjunction.disjuncts[0].indicator_var))

        m.x[1].fixed = False
        # should be identical to the case where x[1] was not fixed
        self.check_transformation_block(
            m,
            0,
            72,
            0,
            72,
            -18,
            32,
            -18,
            32,
            partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
        )

    @unittest.skipIf(
        'gurobi_direct' not in solvers, 'Gurobi direct solver not available'
    )
    def test_assume_fixed_vars_permanent(self):
        m = models.makeBetweenStepsPaperExample()
        m.x[1].fix(0)
        m.disjunction.disjuncts[0].indicator_var.fix(True)

        # I'm using Gurobi because I'm assuming exact equality is going to work
        # out. And it definitely won't with ipopt. (And Gurobi direct is way
        # faster than the LP interface to Gurobi for this... I assume because
        # writing nonlinear expressions is slow?)
        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            assume_fixed_vars_permanent=True,
            compute_bounds_solver=SolverFactory('gurobi_direct'),
            compute_bounds_method=compute_optimal_bounds,
        )

        # Fixing BooleanVars is the same either way. We just check that it was
        # maintained through the transformation.
        self.assertTrue(m.disjunction_disjuncts[0].indicator_var.fixed)
        self.assertTrue(value(m.disjunction.disjuncts[0].indicator_var))

        # This actually changes the structure of the model because fixed vars
        # move to the constants. I think this is fair, and we should allow it
        # because it will allow for a tighter relaxation.
        (
            b,
            disj1,
            disj2,
            aux_vars1,
            aux_vars2,
        ) = self.check_transformation_block_structure(m, 0, 36, 0, 72, -9, 16, -18, 32)

        # check disjunct constraints
        self.check_disjunct_constraints(disj1, disj2, aux_vars1, aux_vars2)

        # now we can check the global constraints--these are what is different
        # because x[1] is gone.
        c = b.component("disjunction_disjuncts[0].constraint[1]_split_constraints")
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

        c = b.component("disjunction_disjuncts[1].constraint[1]_split_constraints")
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

    @unittest.skipIf(
        'gurobi_direct' not in solvers, 'Gurobi direct solver not available'
    )
    def test_transformation_block_arbitrary_odd_partition(self):
        m = models.makeBetweenStepsPaperExample()

        # I'm using Gurobi because I'm assuming exact equality is going to work
        # out. And it definitely won't with ipopt. (And Gurobi direct is way
        # faster than the LP interface to Gurobi for this... I assume because
        # writing nonlinear expressions is slow?)
        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            num_partitions=3,
            compute_bounds_solver=SolverFactory('gurobi_direct'),
            compute_bounds_method=compute_optimal_bounds,
        )

        (b, disj1, disj2) = self.check_transformation_block_disjuncts_and_constraints(
            m, m.disjunction
        )

        # each Disjunct has three variables declared on it (aux vars and
        # indicator var), plus a reference to the indicator_var of the original
        # Disjunct
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)

        aux_vars1 = disj1.component("disjunction_disjuncts[0].constraint[1]_aux_vars")
        self.assertEqual(len(aux_vars1), 3)
        self.assertEqual(aux_vars1[0].lb, 0)
        self.assertEqual(aux_vars1[0].ub, 72)
        self.assertEqual(aux_vars1[1].lb, 0)
        self.assertEqual(aux_vars1[1].ub, 36)
        self.assertEqual(aux_vars1[2].lb, 0)
        self.assertEqual(aux_vars1[2].ub, 36)
        aux_vars2 = disj2.component("disjunction_disjuncts[1].constraint[1]_aux_vars")
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
        c = b.component("disjunction_disjuncts[0].constraint[1]_split_constraints")
        self.assertEqual(len(c), 3)
        c.pprint()
        c1 = c[0]
        self.check_global_constraint_disj1(c1, aux_vars1[0], m.x[1], m.x[4])

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
        self.assertIs(repn.quadratic_vars[0][0], m.x[2])
        self.assertIs(repn.quadratic_vars[0][1], m.x[2])

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
        self.assertIs(repn.quadratic_vars[0][0], m.x[3])
        self.assertIs(repn.quadratic_vars[0][1], m.x[3])
        self.assertIsNone(repn.nonlinear_expr)

        c = b.component("disjunction_disjuncts[1].constraint[1]_split_constraints")
        self.assertEqual(len(c), 3)
        c1 = c[0]
        self.check_global_constraint_disj2(c1, aux_vars2[0], m.x[1], m.x[4])

        c2 = c[1]
        self.assertIsNone(c2.lower)
        self.assertEqual(c2.upper, 0)
        repn = generate_standard_repn(c2.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.linear_vars[0], m.x[2])
        self.assertIs(repn.linear_vars[1], aux_vars2[1])
        self.assertEqual(repn.linear_coefs[0], -6)
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[2])
        self.assertIs(repn.quadratic_vars[0][1], m.x[2])

        c3 = c[2]
        self.assertIsNone(c3.lower)
        self.assertEqual(c3.upper, 0)
        repn = generate_standard_repn(c3.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.linear_vars[0], m.x[3])
        self.assertIs(repn.linear_vars[1], aux_vars2[2])
        self.assertEqual(repn.linear_coefs[0], -6)
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[3])
        self.assertIs(repn.quadratic_vars[0][1], m.x[3])

    def test_transformed_disjuncts_mapped_correctly(self):
        # we map disjuncts to disjuncts because this is a GDP -> GDP
        # transformation
        m = models.makeBetweenStepsPaperExample()

        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            compute_bounds_method=compute_fbbt_bounds,
        )

        b = m.component("_pyomo_gdp_partition_disjuncts_reformulation")
        self.assertIs(
            m.disjunction.disjuncts[0].transformation_block, b.disjunction.disjuncts[0]
        )
        self.assertIs(
            m.disjunction.disjuncts[1].transformation_block, b.disjunction.disjuncts[1]
        )

    def test_transformed_disjunctions_mapped_correctly(self):
        # we map disjunctions to disjunctions because this is a GDP -> GDP
        # transformation
        m = models.makeBetweenStepsPaperExample()

        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            compute_bounds_method=compute_fbbt_bounds,
        )

        b = m.component("_pyomo_gdp_partition_disjuncts_reformulation")
        self.assertIs(m.disjunction.algebraic_constraint, b.disjunction)

    def add_disjunction(self, b):
        m = b.model()
        b.another_disjunction = Disjunction(
            expr=[
                [(m.x[1] - 1) ** 2 + m.x[2] ** 2 <= 1],
                # writing this constraint backwards to test the flipping logic
                [-((m.x[1] - 2) ** 2) - (m.x[2] - 3) ** 2 >= -1],
            ]
        )

    def make_model_with_added_disjunction_on_block(self):
        m = models.makeBetweenStepsPaperExample()

        m.b = Block()
        self.add_disjunction(m.b)

        return m

    def check_second_disjunction_aux_vars(self, aux_vars1, aux_vars2):
        self.assertEqual(len(aux_vars1), 2)
        # min and max of of x_1**2 - 2x_1
        self.assertEqual(aux_vars1[0].lb, -1)
        self.assertEqual(aux_vars1[0].ub, 24)
        # min and max of x_2**2
        self.assertEqual(aux_vars1[1].lb, 0)
        self.assertEqual(aux_vars1[1].ub, 36)

        self.assertEqual(len(aux_vars2), 2)
        # min and max of of x_1**2 - 4x_1
        self.assertEqual(aux_vars2[0].lb, -4)
        self.assertEqual(aux_vars2[0].ub, 12)
        # min and max of x_2**2 - 3x_2
        self.assertEqual(aux_vars2[1].lb, -9)
        self.assertEqual(aux_vars2[1].ub, 16)

    def check_second_disjunction_global_constraint_disj1(self, c, aux_vars1):
        m = c.model()
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 0)
        repn = generate_standard_repn(c1.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.linear_coefs[0], -2)
        self.assertIs(repn.linear_vars[0], m.x[1])
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertIs(repn.linear_vars[1], aux_vars1[0])
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[1])
        self.assertIs(repn.quadratic_vars[0][1], m.x[1])
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIsNone(repn.nonlinear_expr)

        c2 = c[1]
        self.assertIsNone(c2.lower)
        self.assertEqual(c2.upper, 0)
        repn = generate_standard_repn(c2.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(repn.linear_coefs[0], -1)
        self.assertIs(repn.linear_vars[0], aux_vars1[1])
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[2])
        self.assertIs(repn.quadratic_vars[0][1], m.x[2])
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIsNone(repn.nonlinear_expr)

    def check_second_disjunction_global_constraint_disj2(self, c, aux_vars2):
        m = c.model()
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.assertIsNone(c1.lower)
        self.assertEqual(c1.upper, 0)
        repn = generate_standard_repn(c1.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.linear_coefs[0], -4)
        self.assertIs(repn.linear_vars[0], m.x[1])
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertIs(repn.linear_vars[1], aux_vars2[0])
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[1])
        self.assertIs(repn.quadratic_vars[0][1], m.x[1])
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIsNone(repn.nonlinear_expr)

        c2 = c[1]
        self.assertIsNone(c2.lower)
        self.assertEqual(c2.upper, 0)
        repn = generate_standard_repn(c2.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.linear_coefs[0], -6)
        self.assertIs(repn.linear_vars[0], m.x[2])
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertIs(repn.linear_vars[1], aux_vars2[1])
        self.assertEqual(len(repn.quadratic_vars), 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x[2])
        self.assertIs(repn.quadratic_vars[0][1], m.x[2])
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertIsNone(repn.nonlinear_expr)

    def test_disjunction_target(self):
        m = self.make_model_with_added_disjunction_on_block()

        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            compute_bounds_method=compute_fbbt_bounds,
            targets=[m.disjunction],
        )

        # should be the same as before
        self.check_transformation_block(
            m,
            0,
            72,
            0,
            72,
            -72,
            96,
            -72,
            96,
            partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
        )

        # and another_disjunction should be untransformed
        self.assertIsNone(m.b.another_disjunction.algebraic_constraint)
        self.assertIsNone(m.b.another_disjunction.disjuncts[0].transformation_block)
        self.assertIsNone(m.b.another_disjunction.disjuncts[1].transformation_block)

    @unittest.skipIf(
        'gurobi_direct' not in solvers, 'Gurobi direct solver not available'
    )
    def test_block_target(self):
        m = self.make_model_with_added_disjunction_on_block()

        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions=[[m.x[1]], [m.x[2]]],
            compute_bounds_solver=SolverFactory('gurobi_direct'),
            compute_bounds_method=compute_optimal_bounds,
            targets=[m.b],
        )

        # we didn't transform the disjunction not on b
        self.assertIsNone(m.disjunction.algebraic_constraint)
        self.assertIsNone(m.disjunction.disjuncts[0].transformation_block)
        self.assertIsNone(m.disjunction.disjuncts[1].transformation_block)

        b = m.b.component("_pyomo_gdp_partition_disjuncts_reformulation")
        self.assertIsInstance(b, Block)

        # check we declared the right things
        self.assertEqual(len(b.component_map(Disjunction)), 1)
        self.assertEqual(len(b.component_map(Disjunct)), 2)
        self.assertEqual(len(b.component_map(Constraint)), 2)  # global
        # constraints

        disjunction = b.component("b.another_disjunction")
        self.assertIsInstance(disjunction, Disjunction)
        self.assertEqual(len(disjunction.disjuncts), 2)
        # each Disjunct has one constraint
        disj1 = disjunction.disjuncts[0]
        disj2 = disjunction.disjuncts[1]
        self.assertEqual(len(disj1.component_map(Constraint)), 1)
        self.assertEqual(len(disj2.component_map(Constraint)), 1)
        # each Disjunct has three variables declared on it (indexed aux vars and
        # indicator var), plus a Reference to the indicator_var on the original
        # Disjunct
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)

        aux_vars1 = disj1.component(
            "b.another_disjunction_disjuncts[0].constraint[1]_aux_vars"
        )

        aux_vars2 = disj2.component(
            "b.another_disjunction_disjuncts[1].constraint[1]_aux_vars"
        )
        self.check_second_disjunction_aux_vars(aux_vars1, aux_vars2)

        # check constraints on disjuncts
        c1 = disj1.component("b.another_disjunction_disjuncts[0].constraint[1]")
        self.assertEqual(len(c1), 1)
        self.check_disj_constraint(c1[0], 0, aux_vars1[0], aux_vars1[1])

        c2 = disj2.component("b.another_disjunction_disjuncts[1].constraint[1]")
        self.assertEqual(len(c2), 1)
        self.check_disj_constraint(c2[0], -12, aux_vars2[0], aux_vars2[1])

        # check global constraints
        c = b.component(
            "b.another_disjunction_disjuncts[0].constraint[1]_split_constraints"
        )
        self.check_second_disjunction_global_constraint_disj1(c, aux_vars1)

        c = b.component(
            "b.another_disjunction_disjuncts[1].constraint[1]_split_constraints"
        )
        self.check_second_disjunction_global_constraint_disj2(c, aux_vars2)

    @unittest.skipIf(
        'gurobi_direct' not in solvers, 'Gurobi direct solver not available'
    )
    def test_indexed_block_target(self):
        m = ConcreteModel()
        m.b = Block(Any)
        m.b[0].transfer_attributes_from(models.makeBetweenStepsPaperExample())
        m.x = Reference(m.b[0].x)
        self.add_disjunction(m.b[1])

        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions={
                m.b[1].another_disjunction: [[m.x[1]], [m.x[2]]],
                m.b[0].disjunction: [[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            },
            compute_bounds_solver=SolverFactory('gurobi_direct'),
            compute_bounds_method=compute_optimal_bounds,
            targets=[m.b],
        )

        b0 = m.b[0].component("_pyomo_gdp_partition_disjuncts_reformulation")
        self.assertIsInstance(b0, Block)

        # check we declared the right things
        self.assertEqual(len(b0.component_map(Disjunction)), 1)
        self.assertEqual(len(b0.component_map(Disjunct)), 2)
        self.assertEqual(len(b0.component_map(Constraint)), 2)  # global
        # constraints
        b1 = m.b[1].component("_pyomo_gdp_partition_disjuncts_reformulation")
        self.assertIsInstance(b1, Block)

        # check we declared the right things
        self.assertEqual(len(b1.component_map(Disjunction)), 1)
        self.assertEqual(len(b1.component_map(Disjunct)), 2)
        self.assertEqual(len(b1.component_map(Constraint)), 2)  # global
        # constraints

        ############################
        # Check the added disjunction
        #############################
        disjunction = b1.component("b[1].another_disjunction")
        self.assertIsInstance(disjunction, Disjunction)
        self.assertEqual(len(disjunction.disjuncts), 2)
        # each Disjunct has one constraint
        disj1 = disjunction.disjuncts[0]
        disj2 = disjunction.disjuncts[1]
        self.assertEqual(len(disj1.component_map(Constraint)), 1)
        self.assertEqual(len(disj2.component_map(Constraint)), 1)
        # each Disjunct has two variables declared on it (indexed aux vars and
        # indicator var), plus a reference to the indicator_var on the original
        # Disjunct
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)

        aux_vars1 = disj1.component(
            "b[1].another_disjunction_disjuncts[0].constraint[1]_aux_vars"
        )
        aux_vars2 = disj2.component(
            "b[1].another_disjunction_disjuncts[1].constraint[1]_aux_vars"
        )
        self.check_second_disjunction_aux_vars(aux_vars1, aux_vars2)

        # check constraints on disjuncts
        c1 = disj1.component("b[1].another_disjunction_disjuncts[0].constraint[1]")
        self.assertEqual(len(c1), 1)
        self.check_disj_constraint(c1[0], 0, aux_vars1[0], aux_vars1[1])

        c2 = disj2.component("b[1].another_disjunction_disjuncts[1].constraint[1]")
        self.assertEqual(len(c2), 1)
        self.check_disj_constraint(c2[0], -12, aux_vars2[0], aux_vars2[1])

        # check global constraints
        c = b1.component(
            "b[1].another_disjunction_disjuncts[0].constraint[1]_split_constraints"
        )
        self.check_second_disjunction_global_constraint_disj1(c, aux_vars1)

        c = b1.component(
            "b[1].another_disjunction_disjuncts[1].constraint[1]_split_constraints"
        )
        self.check_second_disjunction_global_constraint_disj2(c, aux_vars2)

        ############################
        # Check the original disjunction
        #############################
        disjunction = b0.component("b[0].disjunction")
        self.assertIsInstance(disjunction, Disjunction)
        self.assertEqual(len(disjunction.disjuncts), 2)
        # each Disjunct has one constraint
        disj1 = disjunction.disjuncts[0]
        disj2 = disjunction.disjuncts[1]
        self.assertEqual(len(disj1.component_map(Constraint)), 1)
        self.assertEqual(len(disj2.component_map(Constraint)), 1)
        # each Disjunct has two variables declared on it (indexed aux vars and
        # indicator var), plus a reference to the indicator_var on the original
        # Disjunct
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)

        aux_vars1 = disj1.component(
            "b[0].disjunction_disjuncts[0].constraint[1]_aux_vars"
        )
        aux_vars2 = disj2.component(
            "b[0].disjunction_disjuncts[1].constraint[1]_aux_vars"
        )
        self.check_aux_var_bounds(aux_vars1, aux_vars2, 0, 72, 0, 72, -18, 32, -18, 32)

        # check constraints on disjuncts
        c1 = disj1.component("b[0].disjunction_disjuncts[0].constraint[1]")
        self.assertEqual(len(c1), 1)
        self.check_disj_constraint(c1[0], 1, aux_vars1[0], aux_vars1[1])

        c2 = disj2.component("b[0].disjunction_disjuncts[1].constraint[1]")
        self.assertEqual(len(c2), 1)
        self.check_disj_constraint(c2[0], -35, aux_vars2[0], aux_vars2[1])

        # check global constraints
        c = b0.component(
            "b[0].disjunction_disjuncts[0].constraint[1]_split_constraints"
        )
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj1(c1, aux_vars1[0], m.x[1], m.x[2])
        c2 = c[1]
        self.check_global_constraint_disj1(c2, aux_vars1[1], m.x[3], m.x[4])

        c = b0.component(
            "b[0].disjunction_disjuncts[1].constraint[1]_split_constraints"
        )
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj2(c1, aux_vars2[0], m.x[1], m.x[2])
        c2 = c[1]
        self.check_global_constraint_disj2(c2, aux_vars2[1], m.x[3], m.x[4])

    @unittest.skipIf(
        'gurobi_direct' not in solvers, 'Gurobi direct solver not available'
    )
    def test_indexed_disjunction_target(self):
        m = ConcreteModel()
        m.I = RangeSet(1, 4)
        m.x = Var(m.I, bounds=(-2, 6))
        m.indexed = Disjunction(Any)
        m.indexed[1] = [
            [sum(m.x[i] ** 2 for i in m.I) <= 1],
            [sum((3 - m.x[i]) ** 2 for i in m.I) <= 1],
        ]
        m.indexed[0] = [
            [(m.x[1] - 1) ** 2 + m.x[2] ** 2 <= 1],
            [-((m.x[1] - 2) ** 2) - (m.x[2] - 3) ** 2 >= -1],
        ]
        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions={
                m.indexed[0]: [[m.x[1]], [m.x[2]]],
                m.indexed[1]: [[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            },
            compute_bounds_solver=SolverFactory('gurobi_direct'),
            compute_bounds_method=compute_optimal_bounds,
            targets=[m.indexed],
        )

        b = m.component("_pyomo_gdp_partition_disjuncts_reformulation")
        self.assertIsInstance(b, Block)

        # check we declared the right things
        self.assertEqual(len(b.component_map(Disjunction)), 2)
        self.assertEqual(len(b.component_map(Disjunct)), 4)
        self.assertEqual(len(b.component_map(Constraint)), 4)  # global
        # constraints
        ############################
        # Check the added disjunction
        #############################
        disjunction = b.component("indexed[0]")
        self.assertIsInstance(disjunction, Disjunction)
        self.assertEqual(len(disjunction.disjuncts), 2)
        # each Disjunct has one constraint
        disj1 = disjunction.disjuncts[0]
        disj2 = disjunction.disjuncts[1]
        self.assertEqual(len(disj1.component_map(Constraint)), 1)
        self.assertEqual(len(disj2.component_map(Constraint)), 1)
        # each Disjunct has two variables declared on it (indexed aux vars and
        # indicator var), plus a reference to the indicator_var on the original
        # Disjunct
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)
        aux_vars1 = disj1.component("indexed_disjuncts[2].constraint[1]_aux_vars")
        aux_vars2 = disj2.component("indexed_disjuncts[3].constraint[1]_aux_vars")
        self.check_second_disjunction_aux_vars(aux_vars1, aux_vars2)

        # check constraints on disjuncts
        c1 = disj1.component("indexed_disjuncts[2].constraint[1]")
        self.assertEqual(len(c1), 1)
        self.check_disj_constraint(c1[0], 0, aux_vars1[0], aux_vars1[1])

        c2 = disj2.component("indexed_disjuncts[3].constraint[1]")
        self.assertEqual(len(c2), 1)
        self.check_disj_constraint(c2[0], -12, aux_vars2[0], aux_vars2[1])

        # check global constraints
        c = b.component("indexed_disjuncts[2].constraint[1]_split_constraints")
        self.check_second_disjunction_global_constraint_disj1(c, aux_vars1)

        c = b.component("indexed_disjuncts[3].constraint[1]_split_constraints")
        self.check_second_disjunction_global_constraint_disj2(c, aux_vars2)

        ############################
        # Check the original disjunction
        #############################
        disjunction = b.component("indexed[1]")
        self.assertIsInstance(disjunction, Disjunction)
        self.assertEqual(len(disjunction.disjuncts), 2)
        # each Disjunct has one constraint
        disj1 = disjunction.disjuncts[0]
        disj2 = disjunction.disjuncts[1]
        self.assertEqual(len(disj1.component_map(Constraint)), 1)
        self.assertEqual(len(disj2.component_map(Constraint)), 1)
        # each Disjunct has two variables declared on it (indexed aux vars and
        # indicator var), plus a reference to the original Disjunct's
        # indicator_var
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)

        aux_vars1 = disj1.component("indexed_disjuncts[0].constraint[1]_aux_vars")
        aux_vars2 = disj2.component("indexed_disjuncts[1].constraint[1]_aux_vars")
        self.check_aux_var_bounds(aux_vars1, aux_vars2, 0, 72, 0, 72, -18, 32, -18, 32)

        # check constraints on disjuncts
        c1 = disj1.component("indexed_disjuncts[0].constraint[1]")
        self.assertEqual(len(c1), 1)
        self.check_disj_constraint(c1[0], 1, aux_vars1[0], aux_vars1[1])

        c2 = disj2.component("indexed_disjuncts[1].constraint[1]")
        self.assertEqual(len(c2), 1)
        self.check_disj_constraint(c2[0], -35, aux_vars2[0], aux_vars2[1])

        # check global constraints
        c = b.component("indexed_disjuncts[0].constraint[1]_split_constraints")
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj1(c1, aux_vars1[0], m.x[1], m.x[2])
        c2 = c[1]
        self.check_global_constraint_disj1(c2, aux_vars1[1], m.x[3], m.x[4])

        c = b.component("indexed_disjuncts[1].constraint[1]_split_constraints")
        self.assertEqual(len(c), 2)
        c1 = c[0]
        self.check_global_constraint_disj2(c1, aux_vars2[0], m.x[1], m.x[2])
        c2 = c[1]
        self.check_global_constraint_disj2(c2, aux_vars2[1], m.x[3], m.x[4])

    def test_incomplete_partition_error(self):
        m = models.makeBetweenStepsPaperExample()
        self.assertRaisesRegex(
            GDP_Error,
            "Partition specified for disjunction "
            r"containing Disjunct 'disjunction_disjuncts\[0\]' does not "
            "include all the variables that appear "
            "in the disjunction. The following variables "
            "are not assigned to any part of the "
            r"partition: 'x\[3\]', 'x\[4\]'",
            TransformationFactory('gdp.partition_disjuncts').apply_to,
            m,
            variable_partitions=[[m.x[1]], [m.x[2]]],
            compute_bounds_method=compute_fbbt_bounds,
        )

    def test_unbounded_expression_error(self):
        m = models.makeBetweenStepsPaperExample()
        for i in m.x:
            m.x[i].setub(None)

        self.assertRaisesRegex(
            GDP_Error,
            r"Expression x\[1\]\*x\[1\] from constraint "
            r"'disjunction_disjuncts\[0\].constraint\[1\]' is unbounded! "
            "Please ensure all variables that appear in the constraint are "
            "bounded or specify compute_bounds_method=compute_optimal_bounds "
            "if the expression is bounded by the global constraints.",
            TransformationFactory('gdp.partition_disjuncts').apply_to,
            m,
            variable_partitions=[[m.x[1]], [m.x[2]], [m.x[3], m.x[4]]],
            compute_bounds_method=compute_fbbt_bounds,
        )

    def test_no_value_for_P_error(self):
        m = models.makeBetweenStepsPaperExample()
        with self.assertRaisesRegex(
            GDP_Error,
            "No value for P was given for disjunction "
            "disjunction! Please specify a value of P "
            r"\(number of partitions\), if you do not specify the "
            "partitions directly.",
        ):
            TransformationFactory('gdp.partition_disjuncts').apply_to(m)

    def test_create_using(self):
        m = models.makeBetweenStepsPaperExample()
        self.diff_apply_to_and_create_using(m, num_partitions=2)


class NonQuadraticNonlinear(unittest.TestCase, CommonTests):
    def check_transformation_block(self, m, aux1lb, aux1ub, aux2lb, aux2ub):
        b = m.component("_pyomo_gdp_partition_disjuncts_reformulation")
        self.assertIsInstance(b, Block)

        # check we declared the right things
        self.assertEqual(len(b.component_map(Disjunction)), 1)
        self.assertEqual(len(b.component_map(Disjunct)), 2)
        # global constraints:
        self.assertEqual(len(b.component_map(Constraint)), 2)
        # logical constraints linking old Disjuncts' indicator variables to
        # transformed Disjuncts' indicator variables
        self.assertEqual(len(b.component_map(LogicalConstraint)), 1)

        disjunction = b.disjunction
        self.assertEqual(len(disjunction.disjuncts), 2)
        # each Disjunct has one constraint
        disj1 = disjunction.disjuncts[0]
        disj2 = disjunction.disjuncts[1]
        self.assertEqual(len(disj1.component_map(Constraint)), 1)
        self.assertEqual(len(disj2.component_map(Constraint)), 1)
        # each Disjunct has two variables declared on it (aux vars and indicator
        # var), plus a reference to the indicator_var on the original Disjunct
        self.assertEqual(len(disj1.component_map(Var)), 3)
        self.assertEqual(len(disj2.component_map(Var)), 3)

        equivalence = b.component("indicator_var_equalities")
        self.assertIsInstance(equivalence, LogicalConstraint)
        self.assertEqual(len(equivalence), 2)
        for i, variables in enumerate(
            [
                (m.disjunction.disjuncts[0].indicator_var, disj1.indicator_var),
                (m.disjunction.disjuncts[1].indicator_var, disj2.indicator_var),
            ]
        ):
            cons = equivalence[i]
            self.assertIsInstance(cons.body, EquivalenceExpression)
            self.assertEqual(cons.body.args, variables)

        aux_vars1 = disj1.component("disjunction_disjuncts[0].constraint[1]_aux_vars")
        self.assertEqual(len(aux_vars1), 2)
        self.assertEqual(aux_vars1[0].lb, aux1lb)
        self.assertEqual(aux_vars1[0].ub, aux1ub)
        self.assertEqual(aux_vars1[1].lb, aux1lb)
        self.assertEqual(aux_vars1[1].ub, aux1ub)
        aux_vars2 = disj2.component("disjunction_disjuncts[1].constraint[1]_aux_vars")
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
        c = b.component("disjunction_disjuncts[0].constraint[1]_split_constraints")
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
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[0], EXPR.PowExpression)
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[1], EXPR.PowExpression)
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
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[0], EXPR.PowExpression)
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[1], EXPR.PowExpression)
        self.assertIs(repn.nonlinear_expr.args[0].args[0].args[0], m.x[3])
        self.assertEqual(repn.nonlinear_expr.args[0].args[0].args[1], 4)
        self.assertIs(repn.nonlinear_expr.args[0].args[1].args[0], m.x[4])
        self.assertEqual(repn.nonlinear_expr.args[0].args[1].args[1], 4)

        c = b.component("disjunction_disjuncts[1].constraint[1]_split_constraints")
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
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[0], EXPR.PowExpression)
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[1], EXPR.PowExpression)
        sum_expr = repn.nonlinear_expr.args[0].args[0].args[0]
        self.assertIsInstance(sum_expr, EXPR.SumExpression)
        sum_repn = generate_standard_repn(sum_expr)
        self.assertEqual(sum_repn.constant, 3)
        self.assertTrue(sum_repn.is_linear())
        self.assertEqual(len(sum_repn.linear_vars), 1)
        self.assertEqual(sum_repn.linear_coefs[0], -1)
        self.assertIs(sum_repn.linear_vars[0], m.x[1])
        self.assertEqual(repn.nonlinear_expr.args[0].args[0].args[1], 4)
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[1], EXPR.PowExpression)
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
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[0], EXPR.PowExpression)
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[1], EXPR.PowExpression)
        sum_expr = repn.nonlinear_expr.args[0].args[0].args[0]
        self.assertIsInstance(sum_expr, EXPR.SumExpression)
        sum_repn = generate_standard_repn(sum_expr)
        self.assertEqual(sum_repn.constant, 3)
        self.assertTrue(sum_repn.is_linear())
        self.assertEqual(len(sum_repn.linear_vars), 1)
        self.assertEqual(sum_repn.linear_coefs[0], -1)
        self.assertIs(sum_repn.linear_vars[0], m.x[3])
        self.assertEqual(repn.nonlinear_expr.args[0].args[0].args[1], 4)
        self.assertIsInstance(repn.nonlinear_expr.args[0].args[1], EXPR.PowExpression)
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
        m = models.makeNonQuadraticNonlinearGDP()

        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            compute_bounds_method=compute_fbbt_bounds,
        )

        self.check_transformation_block(
            m, 0, (2 * 6**4) ** 0.25, 0, (2 * 5**4) ** 0.25
        )

    def test_invalid_partition_error(self):
        m = models.makeNonQuadraticNonlinearGDP()

        self.assertRaisesRegex(
            GDP_Error,
            "Variables which appear in the expression "
            r"\(x\[1\]\*\*4 \+ x\[2\]\*\*4\)\*\*0.25 "
            "are in different partitions, but this expression doesn't appear "
            "additively separable. Please expand it if it is additively "
            "separable or, more likely, ensure that all the constraints in "
            "the disjunction are additively separable with respect to the "
            "specified partition. If you did not specify a partition, only "
            "a value of P, note that to automatically partition the "
            "variables, we assume all the expressions are additively "
            "separable.",
            TransformationFactory('gdp.partition_disjuncts').apply_to,
            m,
            variable_partitions=[[m.x[3], m.x[2]], [m.x[1], m.x[4]]],
            compute_bounds_method=compute_fbbt_bounds,
        )

    def test_invalid_partition_error_multiply_vars_in_different_partition(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-10, 10))
        m.y = Var(bounds=(-60, 56))
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x**2 + m.x * m.y + m.y**2 <= 32)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x**2 + m.y**2 <= 3)
        m.disjunction = Disjunction(expr=[m.d1, m.d2])
        with self.assertRaisesRegex(
            GDP_Error,
            "Variables 'x' and 'y' are "
            "multiplied in Constraint 'd1.c', "
            "but they are in different "
            "partitions! Please ensure that "
            "all the constraints in the "
            "disjunction are "
            "additively separable with "
            "respect to the specified "
            "partition.",
        ):
            TransformationFactory('gdp.partition_disjuncts').apply_to(
                m,
                variable_partitions=[[m.x], [m.y]],
                compute_bounds_method=compute_fbbt_bounds,
            )

    def test_non_additively_separable_expression(self):
        m = models.makeNonQuadraticNonlinearGDP()
        # I'm adding a dumb constraint, but I just want to make sure that a
        # not-additively-separable but legal-according-to-the-partition
        # constraint gets through as expected. As an added bonus, this checks
        # how things work when part of the expression is empty for one part in
        # the partition.
        m.disjunction.disjuncts[0].another_constraint = Constraint(
            expr=m.x[1] ** 3 <= 0.5
        )

        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]],
            compute_bounds_method=compute_fbbt_bounds,
        )

        # we just need to check the first Disjunct's transformation
        b = m.component("_pyomo_gdp_partition_disjuncts_reformulation")
        disj1 = b.disjunction.disjuncts[0]

        self.assertEqual(len(disj1.component_map(Constraint)), 2)
        # has indicator_var and two sets of auxiliary variables, plus a reference
        # to the indicator_var on the original Disjunct
        self.assertEqual(len(disj1.component_map(Var)), 4)
        self.assertEqual(len(disj1.component_map(Constraint)), 2)

        aux_vars1 = disj1.component("disjunction_disjuncts[0].constraint[1]_aux_vars")
        # we check these in test_transformation_block_fbbt_bounds

        aux_vars2 = disj1.component(
            "disjunction_disjuncts[0].another_constraint_aux_vars"
        )
        self.assertEqual(len(aux_vars2), 1)
        self.assertEqual(aux_vars2[0].lb, -8)
        self.assertEqual(aux_vars2[0].ub, 216)

        # check the constraint
        cons = disj1.component("disjunction_disjuncts[0].another_constraint")
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0.5)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(repn.linear_coefs[0], 1)
        self.assertIs(repn.linear_vars[0], aux_vars2[0])

        # now check the global constraint
        cons = b.component(
            "disjunction_disjuncts[0].another_constraint_split_constraints"
        )
        self.assertEqual(len(cons), 1)
        cons = cons[0]
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertEqual(repn.linear_coefs[0], -1)
        self.assertIs(repn.linear_vars[0], aux_vars2[0])
        self.assertEqual(len(repn.nonlinear_vars), 1)
        self.assertIs(repn.nonlinear_vars[0], m.x[1])
        nonlinear = repn.nonlinear_expr
        self.assertIsInstance(nonlinear, EXPR.PowExpression)
        self.assertIs(nonlinear.args[0], m.x[1])
        self.assertEqual(nonlinear.args[1], 3)

    def test_create_using(self):
        m = models.makeNonQuadraticNonlinearGDP()
        self.diff_apply_to_and_create_using(
            m, variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]]
        )

    def test_infeasible_value_of_P(self):
        m = models.makeNonQuadraticNonlinearGDP()

        self.assertRaisesRegex(
            GDP_Error,
            "Variables which appear in the "
            r"expression \(x\[1\]\*\*4 \+ x\[2\]\*\*4\)\*\*0.25 are in "
            "different "
            "partitions, but this "
            "expression doesn't appear "
            "additively separable. Please "
            "expand it if it is additively "
            "separable or, more likely, "
            "ensure that all the "
            "constraints in the disjunction "
            "are additively separable with "
            "respect to the specified "
            "partition. If you did not "
            "specify a partition, only "
            "a value of P, note that to "
            "automatically partition the "
            "variables, we assume all the "
            "expressions are additively "
            "separable.",
            TransformationFactory('gdp.partition_disjuncts').apply_to,
            m,
            num_partitions=3,
        )


# This is just a pile of tests that are structural that we use for bigm and
# hull, so might as well for this too.
class CommonModels(unittest.TestCase, CommonTests):
    def test_user_deactivated_disjuncts(self):
        ct.check_user_deactivated_disjuncts(
            self, 'partition_disjuncts', check_trans_block=False, num_partitions=2
        )

    def test_improperly_deactivated_disjuncts(self):
        ct.check_improperly_deactivated_disjuncts(
            self, 'partition_disjuncts', num_partitions=2
        )

    def test_do_not_transform_userDeactivated_indexedDisjunction(self):
        ct.check_do_not_transform_userDeactivated_indexedDisjunction(
            self, 'partition_disjuncts', num_partitions=2
        )

    def test_disjunction_deactivated(self):
        ct.check_disjunction_deactivated(self, 'partition_disjuncts', num_partitions=2)

    def test_disjunctDatas_deactivated(self):
        ct.check_disjunctDatas_deactivated(
            self, 'partition_disjuncts', num_partitions=2
        )

    def test_deactivated_constraints(self):
        ct.check_deactivated_constraints(self, 'partition_disjuncts', num_partitions=2)

    def test_deactivated_disjuncts(self):
        ct.check_deactivated_disjuncts(self, 'partition_disjuncts', num_partitions=2)

    def test_deactivated_disjunctions(self):
        ct.check_deactivated_disjunctions(self, 'partition_disjuncts', num_partitions=2)

    def test_constraints_deactivated_indexedDisjunction(self):
        ct.check_constraints_deactivated_indexedDisjunction(
            self, 'partition_disjuncts', num_partitions=2
        )

    # targets

    def test_only_targets_inactive(self):
        ct.check_only_targets_inactive(self, 'partition_disjuncts', num_partitions=2)

    def test_target_not_a_component_error(self):
        ct.check_target_not_a_component_error(
            self, 'partition_disjuncts', num_partitions=2
        )

    def test_indexedDisj_targets_inactive(self):
        ct.check_indexedDisj_targets_inactive(
            self, 'partition_disjuncts', num_partitions=2
        )

    def test_warn_for_untransformed(self):
        ct.check_warn_for_untransformed(self, 'partition_disjuncts', num_partitions=2)

    def test_disjData_targets_inactive(self):
        ct.check_disjData_targets_inactive(
            self, 'partition_disjuncts', num_partitions=2
        )

    def test_indexedBlock_targets_inactive(self):
        ct.check_indexedBlock_targets_inactive(
            self, 'partition_disjuncts', num_partitions=2
        )

    def test_blockData_targets_inactive(self):
        ct.check_blockData_targets_inactive(
            self, 'partition_disjuncts', num_partitions=2
        )

    # transforming blocks

    def test_transformation_simple_block(self):
        ct.check_transformation_simple_block(
            self, 'partition_disjuncts', num_partitions=2
        )

    def test_transform_block_data(self):
        ct.check_transform_block_data(self, 'partition_disjuncts', num_partitions=2)

    def test_simple_block_target(self):
        ct.check_simple_block_target(self, 'partition_disjuncts', num_partitions=2)

    def test_block_data_target(self):
        ct.check_block_data_target(self, 'partition_disjuncts', num_partitions=2)

    def test_indexed_block_target(self):
        ct.check_indexed_block_target(self, 'partition_disjuncts', num_partitions=2)

    def test_block_targets_inactive(self):
        ct.check_block_targets_inactive(self, 'partition_disjuncts', num_partitions=2)

    # common error messages

    def test_transform_empty_disjunction(self):
        ct.check_transform_empty_disjunction(
            self, 'partition_disjuncts', num_partitions=2
        )

    def test_deactivated_disjunct_nonzero_indicator_var(self):
        ct.check_deactivated_disjunct_nonzero_indicator_var(
            self, 'partition_disjuncts', num_partitions=2
        )

    def test_deactivated_disjunct_unfixed_indicator_var(self):
        ct.check_deactivated_disjunct_unfixed_indicator_var(
            self, 'partition_disjuncts', num_partitions=2
        )

    def test_silly_target(self):
        ct.check_silly_target(self, 'partition_disjuncts', num_partitions=2)

    def test_error_for_same_disjunct_in_multiple_disjunctions(self):
        ct.check_error_for_same_disjunct_in_multiple_disjunctions(
            self, 'partition_disjuncts', num_partitions=2
        )

    def test_cannot_call_transformation_on_disjunction(self):
        ct.check_cannot_call_transformation_on_disjunction(
            self, 'partition_disjuncts', num_partitions=2
        )

    def test_disjunction_target_err(self):
        ct.check_disjunction_target_err(self, 'partition_disjuncts', num_partitions=2)

    # nested disjunctions (only checking that everything is transformed)

    def test_disjuncts_inactive_nested(self):
        ct.check_disjuncts_inactive_nested(
            self, 'partition_disjuncts', num_partitions=2
        )

    def test_deactivated_disjunct_leaves_nested_disjunct_active(self):
        ct.check_deactivated_disjunct_leaves_nested_disjunct_active(
            self, 'partition_disjuncts', num_partitions=2
        )

    def test_disjunct_targets_inactive(self):
        ct.check_disjunct_targets_inactive(
            self, 'partition_disjuncts', num_partitions=2
        )

    def test_disjunctData_targets_inactive(self):
        ct.check_disjunctData_targets_inactive(
            self, 'partition_disjuncts', num_partitions=2
        )

    # check handling for benign types

    def test_RangeSet(self):
        ct.check_RangeSet(self, 'partition_disjuncts', num_partitions=2)

    def test_Expression(self):
        ct.check_Expression(self, 'partition_disjuncts', num_partitions=2)

    def test_untransformed_network_raises_GDPError(self):
        ct.check_untransformed_network_raises_GDPError(
            self, 'partition_disjuncts', num_partitions=2
        )

    @unittest.skipUnless(ct.linear_solvers, "Could not find a linear solver")
    def test_network_disjuncts(self):
        ct.check_network_disjuncts(self, True, 'between_steps', num_partitions=2)
        ct.check_network_disjuncts(self, False, 'between_steps', num_partitions=2)


class LogicalExpressions(unittest.TestCase, CommonTests):
    def test_logical_constraints_on_disjunct_copied(self):
        m = models.makeLogicalConstraintsOnDisjuncts_NonlinearConvex()
        TransformationFactory('gdp.partition_disjuncts').apply_to(
            m,
            variable_partitions=[[m.x], [m.y]],
            compute_bounds_method=compute_fbbt_bounds,
        )
        d1 = m.d[1].transformation_block
        self.assertEqual(len(d1.component_map(LogicalConstraint)), 1)
        c = d1.component("logical_constraints")
        self.assertIsInstance(c, LogicalConstraint)
        self.assertEqual(len(c), 1)
        self.assertIsInstance(c[1].expr, NotExpression)
        self.assertIs(c[1].expr.args[0], m.Y[1])

        d2 = m.d[2].transformation_block
        self.assertEqual(len(d2.component_map(LogicalConstraint)), 1)
        c = d2.component("logical_constraints")
        self.assertIsInstance(c, LogicalConstraint)
        self.assertEqual(len(c), 1)
        self.assertIsInstance(c[1].expr, AndExpression)
        self.assertEqual(len(c[1].expr.args), 2)
        self.assertIs(c[1].expr.args[0], m.Y[1])
        self.assertIs(c[1].expr.args[1], m.Y[2])

        d3 = m.d[3].transformation_block
        self.assertEqual(len(d3.component_map(LogicalConstraint)), 1)
        c = d3.component("logical_constraints")
        self.assertEqual(len(c), 0)

        d4 = m.d[4].transformation_block
        self.assertEqual(len(d4.component_map(LogicalConstraint)), 1)
        c = d4.component("logical_constraints")
        self.assertIsInstance(c, LogicalConstraint)
        self.assertEqual(len(c), 2)
        self.assertIsInstance(c[1].expr, ExactlyExpression)
        self.assertEqual(len(c[1].expr.args), 2)
        self.assertEqual(c[1].expr.args[0], 1)
        self.assertIs(c[1].expr.args[1], m.Y[1])
        self.assertIsInstance(c[2].expr, NotExpression)
        self.assertIs(c[2].expr.args[0], m.Y[2])

    # [ESJ 11/30/21]: This will be a good test when #1032 is implemented for the
    # writers. In the meantime, it doesn't work unless you manually map the
    # BooleanVars on the Disjuncts to binary variables declared somewhere that
    # will still be in the active tree after the call to partition_disjuncts.

    # @unittest.skipIf('gurobi_direct' not in solvers,
    #                  'Gurobi direct solver not available')
    # def test_solve_model_with_boolean_vars_on_disjuncts(self):
    #     # Make sure that we are making references to everything we need to so
    #     # that transformed models are solvable. This is testing both that the
    #     # LogicalConstraints are copied over correctly and that we know how to
    #     # handle when BooleanVars are declared on Disjuncts.
    #     m = models.makeBooleanVarsOnDisjuncts()
    #     # This is actually useless because there is only one variable, but
    #     # that's fine--we just want to make sure the transformed model is
    #     # solvable.
    #     TransformationFactory('gdp.between_steps').apply_to(
    #         m, variable_partitions=[[m.x]],
    #         compute_bounds_method=compute_fbbt_bounds)

    #     self.assertTrue(check_model_algebraic(m))

    #     SolverFactory('gurobi_direct').solve(m)
    #     self.assertAlmostEqual(value(m.x), 8)
    #     self.assertFalse(value(m.d[1].indicator_var))
    #     self.assertTrue(value(m.d[2].indicator_var))
    #     self.assertTrue(value(m.d[3].indicator_var))
    #     self.assertFalse(value(m.d[4].indicator_var))

    @unittest.skipIf(
        'gurobi_direct' not in solvers, 'Gurobi direct solver not available'
    )
    def test_original_indicator_vars_in_logical_constraints(self):
        m = models.makeLogicalConstraintsOnDisjuncts()
        TransformationFactory('gdp.between_steps').apply_to(
            m, variable_partitions=[[m.x]], compute_bounds_method=compute_fbbt_bounds
        )

        self.assertTrue(check_model_algebraic(m))

        SolverFactory('gurobi_direct').solve(m)
        self.assertAlmostEqual(value(m.x), 8)
        self.assertFalse(value(m.d[1].indicator_var))
        self.assertTrue(value(m.d[2].indicator_var))
        self.assertTrue(value(m.d[3].indicator_var))
        self.assertFalse(value(m.d[4].indicator_var))
