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

import pyomo.common.unittest as unittest

from pyomo.opt import TerminationCondition, SolutionStatus, check_available_solvers
import pyomo.environ as pyo
import pyomo.kernel as pmo
import sys

diff_tol = 1e-3

mosek_available = check_available_solvers('mosek_direct')


@unittest.skipIf(not mosek_available, "MOSEK's python bindings are not available")
class MOSEKDirectTests(unittest.TestCase):
    def setUp(self):
        self.stderr = sys.stderr
        sys.stderr = None

    def tearDown(self):
        sys.stderr = self.stderr

    def test_interface_call(self):
        interface_instance = type(pyo.SolverFactory('mosek_direct'))
        alt_1 = pyo.SolverFactory('mosek')
        alt_2 = pyo.SolverFactory('mosek', solver_io='python')
        alt_3 = pyo.SolverFactory('mosek', solver_io='direct')
        self.assertIsInstance(alt_1, interface_instance)
        self.assertIsInstance(alt_2, interface_instance)
        self.assertIsInstance(alt_3, interface_instance)

    def test_infeasible_lp(self):
        model = pyo.ConcreteModel()
        model.X = pyo.Var(within=pyo.NonNegativeReals)
        model.C1 = pyo.Constraint(expr=model.X == 1)
        model.C2 = pyo.Constraint(expr=model.X == 2)
        model.O = pyo.Objective(expr=model.X)

        opt = pyo.SolverFactory("mosek_direct")
        results = opt.solve(model)

        self.assertIn(
            results.solver.termination_condition,
            (
                TerminationCondition.infeasible,
                TerminationCondition.infeasibleOrUnbounded,
            ),
        )

    def test_unbounded_lp(self):
        model = pyo.ConcreteModel()
        model.X = pyo.Var()
        model.O = pyo.Objective(expr=model.X)

        opt = pyo.SolverFactory("mosek_direct")
        results = opt.solve(model)

        self.assertIn(
            results.solver.termination_condition,
            (
                TerminationCondition.unbounded,
                TerminationCondition.infeasibleOrUnbounded,
            ),
        )

    def test_optimal_lp(self):
        model = pyo.ConcreteModel()
        model.X = pyo.Var(within=pyo.NonNegativeReals)
        model.O = pyo.Objective(expr=model.X)

        opt = pyo.SolverFactory("mosek_direct")
        results = opt.solve(model, load_solutions=False)

        self.assertEqual(results.solution.status, SolutionStatus.optimal)

    def test_get_duals_lp(self):
        model = pyo.ConcreteModel()
        model.X = pyo.Var(within=pyo.NonNegativeReals)
        model.Y = pyo.Var(within=pyo.NonNegativeReals)

        model.C1 = pyo.Constraint(expr=2 * model.X + model.Y >= 8)
        model.C2 = pyo.Constraint(expr=model.X + 3 * model.Y >= 6)

        model.O = pyo.Objective(expr=model.X + model.Y)

        opt = pyo.SolverFactory("mosek_direct")
        results = opt.solve(model, suffixes=['dual'], load_solutions=False)

        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        model.solutions.load_from(results)

        self.assertAlmostEqual(model.dual[model.C1], 0.4, 4)
        self.assertAlmostEqual(model.dual[model.C2], 0.2, 4)

    def test_infeasible_mip(self):
        model = pyo.ConcreteModel()
        model.X = pyo.Var(within=pyo.NonNegativeIntegers)
        model.C1 = pyo.Constraint(expr=model.X == 1)
        model.C2 = pyo.Constraint(expr=model.X == 2)
        model.O = pyo.Objective(expr=model.X)

        opt = pyo.SolverFactory("mosek_direct")
        results = opt.solve(model)

        self.assertIn(
            results.solver.termination_condition,
            (
                TerminationCondition.infeasibleOrUnbounded,
                TerminationCondition.infeasible,
            ),
        )

    def test_unbounded_mip(self):
        model = pyo.AbstractModel()
        model.X = pyo.Var(within=pyo.Integers)
        model.O = pyo.Objective(expr=model.X)

        instance = model.create_instance()
        opt = pyo.SolverFactory("mosek_direct")
        results = opt.solve(instance)

        self.assertIn(
            results.solver.termination_condition,
            (
                TerminationCondition.unbounded,
                TerminationCondition.infeasibleOrUnbounded,
            ),
        )

    def test_optimal_mip(self):
        model = pyo.ConcreteModel()
        model.X = pyo.Var(within=pyo.NonNegativeIntegers)
        model.O = pyo.Objective(expr=model.X)

        opt = pyo.SolverFactory("mosek_direct")
        results = opt.solve(model, load_solutions=False)

        self.assertEqual(results.solution.status, SolutionStatus.optimal)

    def test_qcqo(self):
        model = pmo.block()
        model.x = pmo.variable_list()
        for i in range(3):
            model.x.append(pmo.variable(lb=0.0))

        model.cons = pmo.constraint(
            expr=model.x[0]
            + model.x[1]
            + model.x[2]
            - model.x[0] ** 2
            - model.x[1] ** 2
            - 0.1 * model.x[2] ** 2
            + 0.2 * model.x[0] * model.x[2]
            >= 1.0
        )

        model.o = pmo.objective(
            expr=model.x[0] ** 2
            + 0.1 * model.x[1] ** 2
            + model.x[2] ** 2
            - model.x[0] * model.x[2]
            - model.x[1],
            sense=pmo.minimize,
        )

        opt = pmo.SolverFactory("mosek_direct")
        results = opt.solve(model)

        self.assertAlmostEqual(results.problem.upper_bound, -4.9176e-01, 4)
        self.assertAlmostEqual(results.problem.lower_bound, -4.9180e-01, 4)

        del model

    def test_conic(self):
        model = pmo.block()
        model.o = pmo.objective(0.0)
        model.c = pmo.constraint(body=0.0, rhs=1)

        b = model.quadratic = pmo.block()
        b.x = pmo.variable_tuple((pmo.variable(), pmo.variable()))
        b.r = pmo.variable(lb=0)
        b.c = pmo.conic.quadratic(x=b.x, r=b.r)
        model.o.expr += b.r
        model.c.body += b.r
        del b

        b = model.rotated_quadratic = pmo.block()
        b.x = pmo.variable_tuple((pmo.variable(), pmo.variable()))
        b.r1 = pmo.variable(lb=0)
        b.r2 = pmo.variable(lb=0)
        b.c = pmo.conic.rotated_quadratic(x=b.x, r1=b.r1, r2=b.r2)
        model.o.expr += b.r1 + b.r2
        model.c.body += b.r1 + b.r2
        del b

        import mosek

        if mosek.Env().getversion() >= (9, 0, 0):
            b = model.primal_exponential = pmo.block()
            b.x1 = pmo.variable(lb=0)
            b.x2 = pmo.variable()
            b.r = pmo.variable(lb=0)
            b.c = pmo.conic.primal_exponential(x1=b.x1, x2=b.x2, r=b.r)
            model.o.expr += b.r
            model.c.body += b.r
            del b

            b = model.primal_power = pmo.block()
            b.x = pmo.variable_tuple((pmo.variable(), pmo.variable()))
            b.r1 = pmo.variable(lb=0)
            b.r2 = pmo.variable(lb=0)
            b.c = pmo.conic.primal_power(x=b.x, r1=b.r1, r2=b.r2, alpha=0.6)
            model.o.expr += b.r1 + b.r2
            model.c.body += b.r1 + b.r2
            del b

            b = model.dual_exponential = pmo.block()
            b.x1 = pmo.variable()
            b.x2 = pmo.variable(ub=0)
            b.r = pmo.variable(lb=0)
            b.c = pmo.conic.dual_exponential(x1=b.x1, x2=b.x2, r=b.r)
            model.o.expr += b.r
            model.c.body += b.r
            del b

            b = model.dual_power = pmo.block()
            b.x = pmo.variable_tuple((pmo.variable(), pmo.variable()))
            b.r1 = pmo.variable(lb=0)
            b.r2 = pmo.variable(lb=0)
            b.c = pmo.conic.dual_power(x=b.x, r1=b.r1, r2=b.r2, alpha=0.4)
            model.o.expr += b.r1 + b.r2
            model.c.body += b.r1 + b.r2

        if mosek.Env().getversion() >= (10, 0, 0):
            b = model.primal_geomean = pmo.block()
            b.r = pmo.variable_tuple((pmo.variable(), pmo.variable()))
            b.x = pmo.variable()
            b.c = pmo.conic.primal_geomean(r=b.r, x=b.x)
            model.o.expr += b.r[0] + b.r[1]
            model.c.body += b.r[0] + b.r[1]
            del b

            b = model.dual_geomean = pmo.block()
            b.r = pmo.variable_tuple((pmo.variable(), pmo.variable()))
            b.x = pmo.variable()
            b.c = pmo.conic.dual_geomean(r=b.r, x=b.x)
            model.o.expr += b.r[0] + b.r[1]
            model.c.body += b.r[0] + b.r[1]
            del b

            b = model.svec_psdcone = pmo.block()
            b.x = pmo.variable_tuple((pmo.variable(), pmo.variable(), pmo.variable()))
            b.c = pmo.conic.svec_psdcone(x=b.x)
            model.o.expr += b.x[0] + 2 * b.x[1] + b.x[2]
            model.c.body += b.x[0] + 2 * b.x[1] + b.x[2]
            del b

        opt = pmo.SolverFactory("mosek_direct")
        results = opt.solve(model)

        self.assertEqual(results.solution.status, SolutionStatus.optimal)

    def _test_model(self):
        model = pmo.block()
        model.x0, model.x1, model.x2 = [pmo.variable() for i in range(3)]
        model.obj = pmo.objective(2 * model.x0 + 3 * model.x1 - model.x2, sense=-1)

        model.con1 = pmo.constraint(model.x0 + model.x1 + model.x2 == 1)
        model.quad = pmo.conic.quadratic.as_domain(
            r=0.03,
            x=[
                pmo.expression(1.5 * model.x0 + 0.1 * model.x1),
                pmo.expression(0.3 * model.x0 + 2.1 * model.x2 + 0.1),
            ],
        )
        return model

    def test_conic_duals(self):
        check = [-1.94296808, -0.303030303, -1.91919191]
        # load_duals (without args)
        with pmo.SolverFactory('mosek_direct') as solver:
            model = self._test_model()
            results = solver.solve(model)
            model.dual = pmo.suffix(direction=pmo.suffix.IMPORT)
            solver.load_duals()
            for i in range(3):
                self.assertAlmostEqual(model.dual[model.quad.q][i], check[i], 5)
        # load_duals  (with args)
        with pmo.SolverFactory('mosek_direct') as solver:
            model = self._test_model()
            results = solver.solve(model)
            model.dual = pmo.suffix(direction=pmo.suffix.IMPORT)
            solver.load_duals([model.quad.q])
            for i in range(3):
                self.assertAlmostEqual(model.dual[model.quad.q][i], check[i], 5)
        # save_results=True (deprecated)
        with pmo.SolverFactory('mosek_direct') as solver:
            model = self._test_model()
            model.dual = pmo.suffix(direction=pmo.suffix.IMPORT)
            results = solver.solve(model, save_results=True)
            for i in range(3):
                self.assertAlmostEqual(
                    results.Solution.constraint['x11']['Dual'][i], check[i], 5
                )

    def test_solver_parameters(self):
        import mosek

        solver = pyo.SolverFactory('mosek_direct')
        model = self._test_model()
        solver.solve(
            model,
            options={
                'dparam.optimizer_max_time': 1.0,
                'iparam.intpnt_solve_form': mosek.solveform.dual,
                'mosek.iparam.intpnt_max_iterations': 10,
                'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': '1.0e-7',
                'MSK_IPAR_PRESOLVE_USE': '0',
                'sparam.param_comment_sign': '##',
            },
        )
        # Check if iparams were set correctly
        self.assertEqual(
            solver._solver_model.getintparam(mosek.iparam.intpnt_solve_form),
            mosek.solveform.dual,
        )
        self.assertEqual(
            solver._solver_model.getintparam(mosek.iparam.intpnt_max_iterations), 10
        )
        self.assertEqual(
            solver._solver_model.getintparam(mosek.iparam.presolve_use),
            mosek.presolvemode.off,
        )
        # Check if dparams were set correctly
        self.assertEqual(
            solver._solver_model.getdouparam(mosek.dparam.optimizer_max_time), 1.0
        )
        self.assertEqual(
            solver._solver_model.getdouparam(mosek.dparam.intpnt_co_tol_rel_gap), 1.0e-7
        )
        # Check if sparam is set correctly
        self.assertEqual(
            solver._solver_model.getstrparam(mosek.sparam.param_comment_sign)[1], '##'
        )
        # Check for TypeErrors
        with self.assertRaises(TypeError) as typeCheck:
            solver.solve(model, options={'mosek.dparam.intpnt_co_tol_rel_gap': '1.4'})
        with self.assertRaises(TypeError) as typeCheck:
            solver.solve(model, options={'iparam.log': 1.2})
        # Check for AttributeError
        with self.assertRaises(AttributeError) as assertCheck:
            solver.solve(model, options={'wrong.name': '1'})
        with self.assertRaises(AttributeError) as typeCheck:
            solver.solve(model, options={'mosek.iparam.log': 'mosek.wrong.input'})
        # Check for wrong parameter name (but valid MOSEK attribute)
        with self.assertRaises(ValueError) as typeCheck:
            solver.solve(model, options={'mosek.mark.up': 'wrong.val'})
        # Check for parameter names with wrong length
        with self.assertRaises(AssertionError) as typeCheck:
            solver.solve(model, options={'mosek.iparam.log.level': 10})


if __name__ == "__main__":
    unittest.main()
