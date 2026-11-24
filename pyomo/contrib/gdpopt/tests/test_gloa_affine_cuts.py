# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Tests for constant GLOA affine cuts."""

import pyomo.common.unittest as unittest
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available
from pyomo.core.expr.numeric_expr import enable_expression_optimizations
from pyomo.core.expr.numvalue import is_potentially_variable
from pyomo.environ import ConcreteModel, Constraint, Objective, SolverFactory, Var
from pyomo.gdp import Disjunction


class CutsInspected(Exception):
    """Stop after inspecting cuts, before solving the next discrete problem."""


@unittest.skipUnless(mcpp_available(), "MC++ is not available")
@unittest.skipUnless(SolverFactory('glpk').available(), "GLPK is not available")
@unittest.skipUnless(SolverFactory('ipopt').available(), "Ipopt is not available")
class TestGLOAConstantAffineCuts(unittest.TestCase):
    def solve_with_constant_cut(self, side, violation):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1), initialize=0)
        m.y = Var(bounds=(0, 1), initialize=0)
        m.c = Constraint(
            expr=m.x**2 <= -violation if side == 'upper' else -m.x**2 >= violation
        )
        m.d = Disjunction(expr=[[m.y == 0], [m.y == 1]])
        m.d.disjuncts[0].indicator_var.set_value(True)
        m.d.disjuncts[1].indicator_var.set_value(False)
        m.obj = Objective(expr=m.y)

        zero_optimization = not is_potentially_variable(0 * m.x)
        self.addCleanup(enable_expression_optimizations, zero=zero_optimization)
        enable_expression_optimizations(zero=True)

        def set_linearization_point(solver, subproblem, util_block):
            subproblem.x.set_value(0)

        def check_cuts(solver, discrete, util_block):
            if solver.initialization_iteration == 0:
                return
            # One estimator is constant; the other must still generate a cut.
            cuts = discrete.GDPopt_aff.GDPopt_aff_cons
            self.assertEqual(len(cuts), 1)
            self.assertTrue(is_potentially_variable(cuts[0].body))
            raise CutsInspected()

        # Ipopt accepts the small violation at its looser feasibility tolerance.
        # GDPopt must independently compare the constant cut to its own tolerance.
        SolverFactory('gdpopt.gloa').solve(
            m,
            mip_solver='glpk',
            nlp_solver='ipopt',
            init_algorithm='fix_disjuncts',
            mip_presolve=False,
            subproblem_presolve=False,
            constraint_tolerance=1e-6,
            nlp_solver_args={
                'options': {
                    'tol': 1e-3,
                    'constr_viol_tol': 1e-3,
                    'bound_relax_factor': 1e-3,
                }
            },
            call_after_subproblem_solve=set_linearization_point,
            call_before_discrete_problem_solve=check_cuts,
            iterlim=1,
        )

    def test_constant_cuts_within_tolerance(self):
        for side in ('lower', 'upper'):
            for violation in (5e-7, 1e-6):
                with self.subTest(side=side, violation=violation):
                    with self.assertRaises(CutsInspected):
                        self.solve_with_constant_cut(side, violation)

    def test_constant_cut_violation_diagnostics(self):
        for side, estimator in (('lower', 'concave'), ('upper', 'convex')):
            with self.subTest(side=side):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Constant %s affine cut for constraint 'c'.*"
                    "value 0.0.*%s bound .* by 2e-06.*"
                    "constraint_tolerance 1e-06.*numerical error" % (estimator, side),
                ):
                    self.solve_with_constant_cut(side, 2e-6)
