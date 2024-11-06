#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
Tests for the PyROS solver.
"""

import logging
import math
import time

import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import Bunch
from pyomo.common.errors import InvalidValueError
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.repn.plugins import nl_writer as pyomo_nl_writer
import pyomo.repn.ampl as pyomo_ampl_repn
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.errors import ApplicationError, InfeasibleConstraintException
from pyomo.environ import maximize as pyo_max, units as u
from pyomo.opt import (
    SolverResults,
    SolverStatus,
    SolutionStatus,
    TerminationCondition,
    Solution,
)
from pyomo.environ import (
    Reals,
    Set,
    Block,
    ConcreteModel,
    Constraint,
    Objective,
    Param,
    SolverFactory,
    Var,
    exp,
    log,
    sqrt,
    value,
    maximize,
    minimize,
)

from pyomo.contrib.pyros.solve_data import ROSolveResults
from pyomo.contrib.pyros.uncertainty_sets import (
    BoxSet,
    AxisAlignedEllipsoidalSet,
    FactorModelSet,
    IntersectionSet,
    DiscreteScenarioSet,
)
from pyomo.contrib.pyros.util import (
    IterationLogRecord,
    ObjectiveType,
    pyrosTerminationCondition,
)

logger = logging.getLogger(__name__)


if not (numpy_available and scipy_available):
    raise unittest.SkipTest('PyROS unit tests require parameterized, numpy, and scipy')

# === Config args for testing
nlp_solver = 'ipopt'
global_solver = 'baron'
global_solver_args = dict()
nlp_solver_args = dict()

_baron = SolverFactory('baron')
baron_available = _baron.available(exception_flag=False)
if baron_available:
    baron_license_is_valid = _baron.license_is_valid()
    baron_version = _baron.version()
else:
    baron_license_is_valid = False
    baron_version = (0, 0, 0)

_scip = SolverFactory('scip')
scip_available = _scip.available(exception_flag=False)
if scip_available:
    scip_license_is_valid = _scip.license_is_valid()
    scip_version = _scip.version()
else:
    scip_license_is_valid = False
    scip_version = (0, 0, 0)

_ipopt = SolverFactory("ipopt")
ipopt_available = _ipopt.available(exception_flag=False)


# @SolverFactory.register("time_delay_solver")
class TimeDelaySolver(object):
    """
    Solver which puts program to sleep for a specified
    duration after having been invoked a specified number
    of times.
    """

    def __init__(self, calls_to_sleep, max_time, sub_solver):
        self.max_time = max_time
        self.calls_to_sleep = calls_to_sleep
        self.sub_solver = sub_solver

        self.num_calls = 0
        self.options = Bunch()

    def available(self, exception_flag=True):
        return True

    def license_is_valid(self):
        return True

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        pass

    def solve(self, model, **kwargs):
        """
        'Solve' a model.

        Parameters
        ----------
        model : ConcreteModel
            Model of interest.

        Returns
        -------
        results : SolverResults
            Solver results.
        """

        # ensure only one active objective
        active_objs = [
            obj for obj in model.component_data_objects(Objective, active=True)
        ]
        assert len(active_objs) == 1

        if self.num_calls < self.calls_to_sleep:
            # invoke subsolver
            results = self.sub_solver.solve(model, **kwargs)
            self.num_calls += 1
        else:
            # trigger time delay
            time.sleep(self.max_time)
            results = SolverResults()

            # reset number of calls
            self.num_calls = 0

            # generate solution (current model variable values)
            sol = Solution()
            sol.variable = {
                var.name: {"Value": value(var)}
                for var in model.component_data_objects(Var, active=True)
            }
            sol._cuid = False
            sol.status = SolutionStatus.stoppedByLimit
            results.solution.insert(sol)

            # set up results.solver
            results.solver.time = self.max_time
            results.solver.termination_condition = TerminationCondition.maxTimeLimit
            results.solver.status = SolverStatus.warning

        return results


def build_leyffer():
    """
    Build original Leyffer two-variable test problem.
    """
    m = ConcreteModel()

    m.u = Param(initialize=1.125, mutable=True)

    m.x1 = Var(initialize=0, bounds=(0, None))
    m.x2 = Var(initialize=0, bounds=(0, None))

    m.con = Constraint(expr=m.x1 * sqrt(m.u) - m.u * m.x2 <= 2)
    m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

    return m


def build_leyffer_two_cons():
    """
    Build extended Leyffer problem with single uncertain parameter.
    """
    m = ConcreteModel()

    m.u = Param(initialize=1.125, mutable=True)

    m.x1 = Var(initialize=0, bounds=(0, None))
    m.x2 = Var(initialize=0, bounds=(0, None))
    m.x3 = Var(initialize=0, bounds=(None, None))

    m.con1 = Constraint(expr=m.x1 * sqrt(m.u) - m.x2 * m.u <= 2)
    m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u == m.x3)

    m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

    return m


def build_leyffer_two_cons_two_params():
    """
    Build extended Leyffer problem with two uncertain parameters.
    """
    m = ConcreteModel()

    m.u1 = Param(initialize=1.125, mutable=True)
    m.u2 = Param(initialize=1, mutable=True)

    m.x1 = Var(initialize=0, bounds=(0, None))
    m.x2 = Var(initialize=0, bounds=(0, None))
    m.x3 = Var(initialize=0, bounds=(None, None))

    m.con1 = Constraint(expr=m.x1 * sqrt(m.u1) - m.x2 * m.u1 <= 2)
    m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u1 == m.x3)

    m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - m.u2) ** 2)

    return m


class TestPyROSSolveFactorModelSet(unittest.TestCase):
    """
    Test PyROS successfully solves model with factor model uncertainty.
    """

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_two_stg_mod_with_factor_model_set(self):
        """
        Test two-stage model with `FactorModelSet`
        as the uncertainty set.
        """
        m = build_leyffer_two_cons_two_params()

        # Define the uncertainty set
        # we take the parameter `u2` to be 'fixed'
        fset = FactorModelSet(
            origin=[1.125, 1], beta=1, number_of_factors=1, psi_mat=[[0.5], [0.5]]
        )

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1, m.x2],
            second_stage_variables=[],
            uncertain_params=[m.u1, m.u2],
            uncertainty_set=fset,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            options={
                "objective_focus": ObjectiveType.worst_case,
                "solve_master_globally": True,
            },
        )

        # check successful termination
        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_optimal,
            msg="Did not identify robust optimal solution to problem instance.",
        )


class TestPyROSSolveAxisAlignedEllipsoidalSet(unittest.TestCase):
    """
    Unit tests for the AxisAlignedEllipsoidalSet.
    """

    @unittest.skipUnless(
        scip_available and scip_license_is_valid, "SCIP is not available and licensed"
    )
    def test_two_stg_mod_with_axis_aligned_set(self):
        """
        Test two-stage model with `AxisAlignedEllipsoidalSet`
        as the uncertainty set.
        """
        # define model
        m = build_leyffer_two_cons_two_params()

        # Define the uncertainty set
        # we take the parameter `u2` to be 'fixed'
        ellipsoid = AxisAlignedEllipsoidalSet(center=[1.125, 1], half_lengths=[1, 0])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory("scip")
        global_subsolver = SolverFactory("scip")

        # Call the PyROS solver
        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1, m.x2],
            second_stage_variables=[],
            uncertain_params=[m.u1, m.u2],
            uncertainty_set=ellipsoid,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            options={
                "objective_focus": ObjectiveType.worst_case,
                "solve_master_globally": True,
            },
        )

        # check successful termination
        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_optimal,
            msg="Did not identify robust optimal solution to problem instance.",
        )
        self.assertGreater(
            results.iterations,
            0,
            msg="Robust infeasible model terminated in 0 iterations (nominal case).",
        )


class TestPyROSSolveDiscreteSet(unittest.TestCase):
    """
    Test PyROS solves models with discrete uncertainty sets.
    """

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_two_stg_model_discrete_set_single_scenario(self):
        """
        Test two-stage model under discrete uncertainty with
        a single scenario.
        """
        m = build_leyffer_two_cons_two_params()

        # uncertainty set
        discrete_set = DiscreteScenarioSet(scenarios=[(1.125, 1)])

        # Instantiate PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1],
            second_stage_variables=[m.x2],
            uncertain_params=[m.u1, m.u2],
            uncertainty_set=discrete_set,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            options={
                "objective_focus": ObjectiveType.worst_case,
                "solve_master_globally": True,
            },
        )

        # check successful termination
        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_optimal,
            msg="Did not identify robust optimal solution to problem instance.",
        )

        # only one iteration required
        self.assertEqual(
            results.iterations,
            1,
            msg=(
                "PyROS was unable to solve a singleton discrete set instance "
                " successfully within a single iteration."
            ),
        )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_two_stg_model_discrete_set(self):
        """
        Test PyROS successfully solves two-stage model with
        multiple scenarios.
        """
        m = build_leyffer()

        discrete_set = DiscreteScenarioSet(scenarios=[[0.25], [1.125], [2]])

        global_solver = SolverFactory("baron")
        pyros_solver = SolverFactory("pyros")

        res = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1],
            second_stage_variables=[m.x2],
            uncertain_params=[m.u],
            uncertainty_set=discrete_set,
            local_solver=global_solver,
            global_solver=global_solver,
            decision_rule_order=0,
            solve_master_globally=True,
            objective_focus=ObjectiveType.worst_case,
        )

        self.assertEqual(
            res.pyros_termination_condition,
            pyrosTerminationCondition.robust_optimal,
            msg=(
                "Failed to solve discrete set multiple scenarios instance to "
                "robust optimality"
            ),
        )


class TestPyROSRobustInfeasible(unittest.TestCase):
    @unittest.skipUnless(baron_available, "BARON is not available and licensed")
    def test_pyros_robust_infeasible(self):
        """
        Test PyROS behavior when robust infeasibility detected
        from a master problem.
        """
        m = ConcreteModel()
        m.q = Param(initialize=0.5, mutable=True)
        m.x = Var(bounds=(m.q, 1))
        # makes model infeasible since 2 is outside bounds
        m.con1 = Constraint(expr=m.x == 2)
        m.obj = Objective(expr=m.x)
        baron = SolverFactory("baron")
        pyros = SolverFactory("pyros")
        results = pyros.solve(
            model=m,
            first_stage_variables=[m.x],
            second_stage_variables=[],
            uncertain_params=m.q,
            uncertainty_set=BoxSet([[0, 1]]),
            local_solver=baron,
            global_solver=baron,
            solve_master_globally=True,
        )

        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_infeasible,
        )
        self.assertEqual(results.iterations, 1)
        # since x was not initialized
        self.assertEqual(results.final_objective_value, None)


global_solver = "baron"


# === regression test for the solver
@unittest.skipUnless(baron_available, "Global NLP solver is not available.")
class RegressionTest(unittest.TestCase):
    """
    Collection of regression tests.
    """

    def build_regression_test_model(self):
        """
        Create model used for regression tests.
        """
        m = ConcreteModel()
        m.name = "s381"

        m.set_params = Set(initialize=list(range(4)))
        m.p = Param(m.set_params, initialize=2, mutable=True)

        m.x1 = Var(within=Reals, bounds=(0, None), initialize=0.1)
        m.x2 = Var(within=Reals, bounds=(0, None), initialize=0.1)
        m.x3 = Var(within=Reals, bounds=(0, None), initialize=0.1)

        m.con1 = Constraint(expr=m.p[1] * m.x1 + m.x2 + m.x3 <= 2)

        m.obj = Objective(expr=(m.x1 - 1) * 2, sense=minimize)

        m.decision_vars = [m.x1, m.x2, m.x3]

        m.uncertain_params = [m.p]

        return m

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_regression_constant_drs(self):
        m = self.build_regression_test_model()

        box_set = BoxSet(bounds=[(1.8, 2.2)])
        solver = SolverFactory("baron")
        pyros = SolverFactory("pyros")
        results = pyros.solve(
            model=m,
            first_stage_variables=m.decision_vars,
            second_stage_variables=[],
            uncertain_params=[m.p[1]],
            uncertainty_set=box_set,
            local_solver=solver,
            global_solver=solver,
            options={"objective_focus": ObjectiveType.nominal},
        )
        self.assertTrue(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_feasible,
        )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_regression_affine_drs(self):
        m = self.build_regression_test_model()

        box_set = BoxSet(bounds=[(1.8, 2.2)])
        solver = SolverFactory("baron")
        pyros = SolverFactory("pyros")
        results = pyros.solve(
            model=m,
            first_stage_variables=m.decision_vars,
            second_stage_variables=[],
            uncertain_params=[m.p[1]],
            uncertainty_set=box_set,
            local_solver=solver,
            global_solver=solver,
            options={
                "objective_focus": ObjectiveType.nominal,
                "decision_rule_order": 1,
            },
        )
        self.assertTrue(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_feasible,
        )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_regression_quadratic_drs(self):
        m = self.build_regression_test_model()

        box_set = BoxSet(bounds=[(1.8, 2.2)])
        solver = SolverFactory("baron")
        pyros = SolverFactory("pyros")
        results = pyros.solve(
            model=m,
            first_stage_variables=m.decision_vars,
            second_stage_variables=[],
            uncertain_params=[m.p[1]],
            uncertainty_set=box_set,
            local_solver=solver,
            global_solver=solver,
            options={
                "objective_focus": ObjectiveType.nominal,
                "decision_rule_order": 2,
            },
        )
        self.assertTrue(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_feasible,
        )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_identifying_violating_param_realization(self):
        m = build_leyffer_two_cons()

        # Define the uncertainty set
        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1, m.x2],
            second_stage_variables=[],
            uncertain_params=[m.u],
            uncertainty_set=interval,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            options={
                "objective_focus": ObjectiveType.worst_case,
                "solve_master_globally": True,
            },
        )

        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_optimal,
            msg="Did not identify robust optimal solution to problem instance.",
        )
        self.assertGreater(
            results.iterations,
            0,
            msg="Robust infeasible model terminated in 0 iterations (nominal case).",
        )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    @unittest.skipUnless(
        baron_version < (23, 1, 5) or baron_version >= (23, 6, 23),
        "Test known to fail for BARON 23.1.5 and versions preceding 23.6.23",
    )
    def test_terminate_with_max_iter(self):
        m = build_leyffer_two_cons()

        # Define the uncertainty set
        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1],
            second_stage_variables=[m.x2],
            uncertain_params=[m.u],
            uncertainty_set=interval,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            options={
                "objective_focus": ObjectiveType.worst_case,
                "solve_master_globally": True,
                "max_iter": 1,
                "decision_rule_order": 2,
            },
        )

        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.max_iter,
            msg="Returned termination condition is not return max_iter.",
        )

        self.assertEqual(
            results.iterations,
            1,
            msg=(
                f"Number of iterations in results object is {results.iterations}, "
                f"but expected value 1."
            ),
        )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_terminate_with_time_limit(self):
        m = build_leyffer_two_cons()

        # Define the uncertainty set
        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1, m.x2],
            second_stage_variables=[],
            uncertain_params=[m.u],
            uncertainty_set=interval,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            objective_focus=ObjectiveType.worst_case,
            solve_master_globally=True,
            time_limit=0.001,
        )

        # validate termination condition
        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.time_out,
            msg="Returned termination condition is not return time_out.",
        )

        # verify subsolver options are unchanged
        subsolvers = [local_subsolver, global_subsolver]
        for slvr, desc in zip(subsolvers, ["Local", "Global"]):
            self.assertEqual(
                len(list(slvr.options.keys())),
                0,
                msg=f"{desc} subsolver options were changed by PyROS",
            )
            self.assertIs(
                getattr(slvr.options, "MaxTime", None),
                None,
                msg=(
                    f"{desc} subsolver (BARON) MaxTime setting was added "
                    "by PyROS, but not reverted"
                ),
            )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_pyros_backup_solvers(self):
        m = ConcreteModel()
        m.name = "s381"

        class BadSolver:
            def __init__(self, max_num_calls):
                self.max_num_calls = max_num_calls
                self.num_calls = 0

            def available(self, exception_flag=True):
                return True

            def solve(self, *args, **kwargs):
                if self.num_calls < self.max_num_calls:
                    self.num_calls += 1
                    return SolverFactory("baron").solve(*args, **kwargs)
                res = SolverResults()
                res.solver.termination_condition = TerminationCondition.maxIterations
                res.solver.status = SolverStatus.warning
                return res

        m.x1 = Var(within=Reals, bounds=(0, None), initialize=0.1)
        m.x2 = Var(within=Reals, bounds=(0, None), initialize=0.1)
        m.x3 = Var(within=Reals, bounds=(0, None), initialize=0.1)

        # === State Vars = [x13]
        # === Decision Vars ===
        m.decision_vars = [m.x1, m.x2, m.x3]

        # === Uncertain Params ===
        m.set_params = Set(initialize=list(range(4)))
        m.p = Param(m.set_params, initialize=2, mutable=True)
        m.uncertain_params = [m.p]

        m.obj = Objective(expr=(m.x1 - 1) * 2, sense=minimize)
        m.con1 = Constraint(expr=m.p[1] * m.x1 + m.x2 + m.x3 <= 2)

        box_set = BoxSet(bounds=[(1.8, 2.2)])
        pyros = SolverFactory("pyros")
        results = pyros.solve(
            model=m,
            first_stage_variables=m.decision_vars,
            second_stage_variables=[],
            uncertain_params=[m.p[1]],
            uncertainty_set=box_set,
            # note: allow 4 calls to work normally
            #       to permit successful solution of uncertainty
            #       bounding problems
            local_solver=BadSolver(4),
            global_solver=BadSolver(4),
            backup_local_solvers=[SolverFactory("baron")],
            backup_global_solvers=[SolverFactory("baron")],
            options={"objective_focus": ObjectiveType.nominal},
            solve_master_globally=True,
        )
        self.assertTrue(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_feasible,
        )

    @unittest.skipUnless(
        SolverFactory('baron').license_is_valid(),
        "Global NLP solver is not available and licensed.",
    )
    def test_separation_terminate_time_limit(self):
        """
        Test PyROS time limit status returned in event
        separation problem times out.
        """
        m = build_leyffer_two_cons()

        # Define the uncertainty set
        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = TimeDelaySolver(
            calls_to_sleep=0, sub_solver=SolverFactory("baron"), max_time=1
        )
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1, m.x2],
            second_stage_variables=[],
            uncertain_params=[m.u],
            uncertainty_set=interval,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            objective_focus=ObjectiveType.worst_case,
            solve_master_globally=True,
            time_limit=1,
        )

        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.time_out,
            msg="Returned termination condition is not return time_out.",
        )

    @unittest.skipUnless(
        ipopt_available
        and SolverFactory('gams').license_is_valid()
        and SolverFactory('baron').license_is_valid()
        and SolverFactory("scip").license_is_valid(),
        "IPOPT not available or one of GAMS/BARON/SCIP not licensed",
    )
    def test_pyros_subsolver_time_limit_adjustment(self):
        """
        Check that PyROS does not ultimately alter state of
        subordinate solver options due to time limit adjustments.
        """
        m = build_leyffer_two_cons()

        # Define the uncertainty set
        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # subordinate solvers to test.
        # for testing, we pass each as the 'local' solver,
        # and the BARON solver without custom options
        # as the 'global' solver
        baron_no_options = SolverFactory("baron")
        local_subsolvers = [
            SolverFactory("gams:conopt"),
            SolverFactory("gams:conopt"),
            SolverFactory("ipopt"),
            SolverFactory("ipopt", options={"max_cpu_time": 300}),
            SolverFactory("scip"),
            SolverFactory("scip", options={"limits/time": 300}),
            baron_no_options,
            SolverFactory("baron", options={"MaxTime": 300}),
        ]
        local_subsolvers[0].options["add_options"] = ["option reslim=100;"]

        # Call the PyROS solver
        for idx, opt in enumerate(local_subsolvers):
            original_solver_options = opt.options.copy()
            results = pyros_solver.solve(
                model=m,
                first_stage_variables=[m.x1, m.x2],
                second_stage_variables=[],
                uncertain_params=[m.u],
                uncertainty_set=interval,
                local_solver=opt,
                global_solver=baron_no_options,
                objective_focus=ObjectiveType.worst_case,
                solve_master_globally=True,
                time_limit=100,
            )
            self.assertEqual(
                results.pyros_termination_condition,
                pyrosTerminationCondition.robust_optimal,
                msg=(
                    "Returned termination condition with local "
                    f"subsolver {idx + 1} of 2 is not robust_optimal."
                ),
            )
            self.assertEqual(
                opt.options,
                original_solver_options,
                msg=(
                    f"Options for subordinate solver {opt} were changed "
                    "by PyROS, and the changes wee not properly reverted."
                ),
            )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_terminate_with_application_error(self):
        """
        Check that PyROS correctly raises ApplicationError
        in event of abnormal IPOPT termination.
        """
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=1.5)
        m.x1 = Var(initialize=-1)
        m.obj = Objective(expr=log(m.x1) * m.p)
        m.con = Constraint(expr=m.x1 * m.p >= -2)

        solver = SolverFactory("ipopt")
        solver.options["halt_on_ampl_error"] = "yes"
        baron = SolverFactory("baron")

        box_set = BoxSet(bounds=[(1, 2)])
        pyros_solver = SolverFactory("pyros")
        with self.assertRaisesRegex(
            ApplicationError, r"Solver \(ipopt\) did not exit normally"
        ):
            pyros_solver.solve(
                model=m,
                first_stage_variables=[m.x1],
                second_stage_variables=[],
                uncertain_params=[m.p],
                uncertainty_set=box_set,
                local_solver=solver,
                global_solver=baron,
                objective_focus=ObjectiveType.nominal,
                time_limit=1000,
            )

        # check solver settings are unchanged
        self.assertEqual(
            len(list(solver.options.keys())),
            1,
            msg=(f"Local subsolver {solver} options were changed by PyROS"),
        )
        self.assertEqual(
            solver.options["halt_on_ampl_error"],
            "yes",
            msg=(
                f"Local subsolver {solver} option "
                "'halt_on_ampl_error' was changed by PyROS"
            ),
        )
        self.assertEqual(
            len(list(baron.options.keys())),
            0,
            msg=(f"Global subsolver {baron} options were changed by PyROS"),
        )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_master_subsolver_error(self):
        """
        Test PyROS on a two-stage problem with a subsolver error
        termination in the initial master problem.
        """
        m = ConcreteModel()

        m.q = Param(initialize=1, mutable=True)

        m.x1 = Var(initialize=1, bounds=(0, 1))

        # source of subsolver error: can't converge to log(0)
        # in separation problem (make x2 second-stage var)
        m.x2 = Var(initialize=2, bounds=(0, m.q))

        m.obj = Objective(expr=log(m.x1) + m.x2)

        box_set = BoxSet(bounds=[(0, 1)])

        local_solver = SolverFactory("ipopt")
        global_solver = SolverFactory("baron")
        pyros_solver = SolverFactory("pyros")

        res = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1],
            second_stage_variables=[m.x2],
            uncertain_params=[m.q],
            uncertainty_set=box_set,
            local_solver=local_solver,
            global_solver=global_solver,
            decision_rule_order=1,
            tee=True,
        )
        self.assertEqual(
            res.pyros_termination_condition,
            pyrosTerminationCondition.subsolver_error,
            msg=(
                f"Returned termination condition for separation error"
                "test is not {pyrosTerminationCondition.subsolver_error}.",
            ),
        )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_separation_subsolver_error(self):
        """
        Test PyROS on a two-stage problem with a subsolver error
        termination in separation.
        """
        m = ConcreteModel()

        m.q = Param(initialize=1, mutable=True)

        m.x1 = Var(initialize=1, bounds=(0, 1))

        # source of subsolver error: can't converge to log(0)
        # in separation problem (make x2 second-stage var)
        m.x2 = Var(initialize=2, bounds=(0, log(m.q)))

        m.obj = Objective(expr=m.x1 + m.x2)

        box_set = BoxSet(bounds=[(0, 1)])

        local_solver = SolverFactory("ipopt")
        global_solver = SolverFactory("baron")
        pyros_solver = SolverFactory("pyros")

        res = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1],
            second_stage_variables=[m.x2],
            uncertain_params=[m.q],
            uncertainty_set=box_set,
            local_solver=local_solver,
            global_solver=global_solver,
            decision_rule_order=1,
            tee=True,
        )
        self.assertEqual(
            res.pyros_termination_condition,
            pyrosTerminationCondition.subsolver_error,
            msg=(
                "Returned termination condition for separation error"
                f"test is not {pyrosTerminationCondition.subsolver_error}."
            ),
        )

    @unittest.skipUnless(ipopt_available, "IPOPT is not available.")
    @unittest.skipUnless(baron_license_is_valid, "BARON is not available and licensed.")
    def test_discrete_separation_subsolver_error(self):
        """
        Test PyROS for two-stage problem with discrete type set,
        subsolver error status.
        """

        class BadSeparationSolver:
            def __init__(self, solver):
                self.solver = solver

            def available(self, exception_flag=False):
                return self.solver.available(exception_flag=exception_flag)

            def solve(self, model, *args, **kwargs):
                is_separation = hasattr(model, "uncertainty")
                if is_separation:
                    res = SolverResults()
                    res.solver.termination_condition = TerminationCondition.unknown
                else:
                    res = self.solver.solve(model, *args, **kwargs)
                return res

        m = ConcreteModel()

        m.q = Param(initialize=1, mutable=True)
        m.x1 = Var(initialize=1, bounds=(0, 1))
        m.x2 = Var(initialize=2, bounds=(0, m.q))
        m.obj = Objective(expr=m.x1 + m.x2, sense=maximize)

        discrete_set = DiscreteScenarioSet(scenarios=[(1,), (0,)])

        local_solver = SolverFactory("ipopt")
        global_solver = SolverFactory("baron")
        pyros_solver = SolverFactory("pyros")

        with LoggingIntercept(level=logging.WARNING) as LOG:
            res = pyros_solver.solve(
                model=m,
                first_stage_variables=[m.x1],
                second_stage_variables=[m.x2],
                uncertain_params=[m.q],
                uncertainty_set=discrete_set,
                local_solver=BadSeparationSolver(local_solver),
                global_solver=BadSeparationSolver(global_solver),
                decision_rule_order=1,
                tee=True,
            )

        self.assertRegex(LOG.getvalue(), "Could not.*separation.*iteration 0.*")
        self.assertEqual(
            res.pyros_termination_condition, pyrosTerminationCondition.subsolver_error
        )
        self.assertEqual(res.iterations, 1)

    @unittest.skipUnless(ipopt_available, "IPOPT is not available.")
    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_discrete_separation_invalid_value_error(self):
        """
        Test PyROS properly handles InvalidValueError.
        """
        m = ConcreteModel()

        m.q = Param(initialize=1, mutable=True)
        m.x1 = Var(initialize=1, bounds=(0, 1))

        # upper bound induces invalid value error: separation
        # max(x2 - log(m.q)) will force subsolver to q = 0
        m.x2 = Var(initialize=2, bounds=(None, log(m.q)))

        m.obj = Objective(expr=m.x1 + m.x2, sense=maximize)

        discrete_set = DiscreteScenarioSet(scenarios=[(1,), (0,)])

        local_solver = SolverFactory("ipopt")
        global_solver = SolverFactory("baron")
        pyros_solver = SolverFactory("pyros")

        with LoggingIntercept(level=logging.ERROR) as LOG:
            with self.assertRaises(InvalidValueError):
                pyros_solver.solve(
                    model=m,
                    first_stage_variables=[m.x1],
                    second_stage_variables=[m.x2],
                    uncertain_params=[m.q],
                    uncertainty_set=discrete_set,
                    local_solver=local_solver,
                    global_solver=global_solver,
                    decision_rule_order=1,
                    tee=True,
                )

        err_str = LOG.getvalue()
        self.assertRegex(
            err_str, "Optimizer.*exception.*separation problem.*iteration 0"
        )

    @unittest.skipUnless(ipopt_available, "IPOPT is not available.")
    def test_pyros_nl_and_ampl_writer_tol(self):
        """
        Test PyROS subsolver call routine behavior
        with respect to the NL and AMPL writer tolerances is as
        expected.
        """
        m = ConcreteModel()
        m.q = Param(initialize=1, mutable=True)
        m.x1 = Var(initialize=1, bounds=(0, 1))
        m.x2 = Var(initialize=2, bounds=(0, m.q))
        m.obj = Objective(expr=m.x1 + m.x2)

        # fixed just inside the PyROS-specified NL writer tolerance.
        m.x1.fix(m.x1.upper + 9.9e-5)

        current_nl_writer_tol = pyomo_nl_writer.TOL, pyomo_ampl_repn.TOL
        ipopt_solver = SolverFactory("ipopt")
        pyros_solver = SolverFactory("pyros")

        pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1],
            second_stage_variables=[m.x2],
            uncertain_params=[m.q],
            uncertainty_set=BoxSet([[0, 1]]),
            local_solver=ipopt_solver,
            global_solver=ipopt_solver,
            decision_rule_order=0,
            solve_master_globally=False,
            bypass_global_separation=True,
        )

        self.assertEqual(
            (pyomo_nl_writer.TOL, pyomo_ampl_repn.TOL),
            current_nl_writer_tol,
            msg="Pyomo writer tolerances not restored as expected.",
        )

        # fixed just outside the PyROS-specified writer tolerances.
        # this should be exceptional.
        m.x1.fix(m.x1.upper + 1.01e-4)

        err_msg = (
            "model contains a trivially infeasible variable.*x1"
            ".*fixed.*outside bounds"
        )
        with self.assertRaisesRegex(InfeasibleConstraintException, err_msg):
            pyros_solver.solve(
                model=m,
                first_stage_variables=[m.x1],
                second_stage_variables=[m.x2],
                uncertain_params=[m.q],
                uncertainty_set=BoxSet([[0, 1]]),
                local_solver=ipopt_solver,
                global_solver=ipopt_solver,
                decision_rule_order=0,
                solve_master_globally=False,
                bypass_global_separation=True,
            )

        self.assertEqual(
            (pyomo_nl_writer.TOL, pyomo_ampl_repn.TOL),
            current_nl_writer_tol,
            msg=(
                "Pyomo writer tolerances not restored as expected "
                "after exceptional test."
            ),
        )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_pyros_math_domain_error(self):
        """
        Test PyROS on a two-stage problem, discrete
        set type with a math domain error evaluating
        second-stage inequality constraint expressions in separation.
        """
        m = ConcreteModel()
        m.q = Param(initialize=1, mutable=True)
        m.x1 = Var(initialize=1, bounds=(0, 1))
        m.x2 = Var(initialize=2, bounds=(-m.q, log(m.q)))
        m.obj = Objective(expr=m.x1 + m.x2)

        box_set = BoxSet(bounds=[[0, 1]])

        local_solver = SolverFactory("baron")
        global_solver = SolverFactory("baron")
        pyros_solver = SolverFactory("pyros")

        with self.assertRaisesRegex(
            expected_exception=ArithmeticError,
            expected_regex=(
                "Evaluation of second-stage inequality constraint.*math domain error.*"
            ),
            msg="ValueError arising from math domain error not raised",
        ):
            # should raise math domain error:
            # (1) lower bounding constraint on x2 solved first
            #     in separation, q = 0 in worst case
            # (2) now tries to evaluate log(q), but q = 0
            pyros_solver.solve(
                model=m,
                first_stage_variables=[m.x1],
                second_stage_variables=[m.x2],
                uncertain_params=[m.q],
                uncertainty_set=box_set,
                local_solver=local_solver,
                global_solver=global_solver,
                decision_rule_order=1,
                tee=True,
            )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_pyros_no_perf_cons(self):
        """
        Ensure PyROS properly accommodates models with no
        second-stage inequality constraints
        (such as effectively deterministic models).
        """
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))
        m.q = Param(mutable=True, initialize=1)

        m.obj = Objective(expr=m.x * m.q)

        pyros_solver = SolverFactory("pyros")
        res = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x],
            second_stage_variables=[],
            uncertain_params=[m.q],
            uncertainty_set=BoxSet(bounds=[[0, 1]]),
            local_solver=SolverFactory("ipopt"),
            global_solver=SolverFactory("ipopt"),
            solve_master_globally=True,
        )
        self.assertEqual(
            res.pyros_termination_condition,
            pyrosTerminationCondition.robust_feasible,
            msg=(
                f"Returned termination condition for separation error"
                "test is not {pyrosTerminationCondition.subsolver_error}.",
            ),
        )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_nominal_focus_robust_feasible(self):
        """
        Test problem under nominal objective focus terminates
        successfully.
        """
        m = build_leyffer_two_cons()

        # singleton set, guaranteed robust feasibility
        discrete_scenarios = DiscreteScenarioSet(scenarios=[[1.125]])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1, m.x2],
            second_stage_variables=[],
            uncertain_params=[m.u],
            uncertainty_set=discrete_scenarios,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            solve_master_globally=False,
            bypass_local_separation=True,
            options={
                "objective_focus": ObjectiveType.nominal,
                "solve_master_globally": True,
            },
        )
        # check for robust feasible termination
        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_feasible,
            msg="Returned termination condition is not return robust_optimal.",
        )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_discrete_separation(self):
        m = build_leyffer_two_cons()

        # Define the uncertainty set
        discrete_scenarios = DiscreteScenarioSet(scenarios=[[0.25], [2.0], [1.125]])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1, m.x2],
            second_stage_variables=[],
            uncertain_params=[m.u],
            uncertainty_set=discrete_scenarios,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            options={
                "objective_focus": ObjectiveType.worst_case,
                "solve_master_globally": True,
            },
        )

        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_optimal,
            msg="Returned termination condition is not return robust_optimal.",
        )

    @unittest.skipUnless(
        scip_available and scip_license_is_valid, "SCIP is not available and licensed."
    )
    def test_higher_order_decision_rules(self):
        m = build_leyffer_two_cons()

        # Define the uncertainty set
        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory("scip")
        global_subsolver = SolverFactory("scip")

        # Call the PyROS solver
        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1],
            second_stage_variables=[m.x2],
            uncertain_params=[m.u],
            uncertainty_set=interval,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            options={
                "objective_focus": ObjectiveType.worst_case,
                "solve_master_globally": True,
                "decision_rule_order": 2,
            },
        )

        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_optimal,
            msg="Returned termination condition is not return robust_optimal.",
        )

    @unittest.skipUnless(scip_available, "Global NLP solver is not available.")
    def test_coefficient_matching_solve(self):
        # Write the deterministic Pyomo model
        m = build_leyffer()
        m.eq_con = Constraint(
            expr=m.u**2 * (m.x2 - 1)
            + m.u * (m.x1**3 + 0.5)
            - 5 * m.u * m.x1 * m.x2
            + m.u * (m.x1 + 2)
            == 0
        )

        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('scip')
        global_subsolver = SolverFactory("scip")

        # Call the PyROS solver
        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1, m.x2],
            second_stage_variables=[],
            uncertain_params=[m.u],
            uncertainty_set=interval,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            options={
                "objective_focus": ObjectiveType.worst_case,
                "solve_master_globally": True,
            },
        )

        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_optimal,
            msg=(
                "Non-optimal termination condition from robust"
                "feasible coefficient matching problem."
            ),
        )
        self.assertAlmostEqual(
            results.final_objective_value,
            6.0394,
            2,
            msg="Incorrect objective function value.",
        )

    def build_mitsos_4_3(self):
        """
        Create instance of Problem 4_3 from Mitsos (2011)'s
        Test Set of semi-infinite programs.
        """
        # construct the deterministic model
        m = ConcreteModel()
        m.u = Param(initialize=0.5, mutable=True)
        m.x1 = Var(bounds=[-1000, 1000])
        m.x2 = Var(bounds=[-1000, 1000])
        m.x3 = Var(bounds=[-1000, 1000])
        m.con = Constraint(expr=exp(m.u - 1) - m.x1 - m.x2 * m.u - m.x3 * m.u**2 <= 0)
        m.eq_con = Constraint(
            expr=(
                m.u**2 * (m.x2 - 1)
                + m.u * (m.x1**3 + 0.5)
                - 5 * m.u * m.x1 * m.x2
                + m.u * (m.x1 + 2)
                == 0
            )
        )
        m.obj = Objective(expr=m.x1 + m.x2 / 2 + m.x3 / 3)

        return m

    @unittest.skipUnless(
        baron_license_is_valid and scip_available and scip_license_is_valid,
        "Global solvers BARON and SCIP not both available and licensed",
    )
    @unittest.skipIf(
        (24, 1, 5) <= baron_version and baron_version <= (24, 5, 8),
        f"Test expected to fail for BARON version {baron_version}",
    )
    def test_coeff_matching_solver_insensitive(self):
        """
        Check that result for instance with constraint subject to
        coefficient matching is insensitive to subsolver settings. Based
        on Mitsos (2011) semi-infinite programming instance 4_3.
        """
        m = self.build_mitsos_4_3()

        # instantiate BARON subsolver and PyROS solver
        baron = SolverFactory("baron")
        scip = SolverFactory("scip")
        pyros_solver = SolverFactory("pyros")

        # solve with PyROS
        solver_names = {"baron": baron, "scip": scip}
        for name, solver in solver_names.items():
            res = pyros_solver.solve(
                model=m,
                first_stage_variables=[],
                second_stage_variables=[m.x1, m.x2, m.x3],
                uncertain_params=[m.u],
                uncertainty_set=BoxSet(bounds=[[0, 1]]),
                local_solver=solver,
                global_solver=solver,
                objective_focus=ObjectiveType.worst_case,
                solve_master_globally=True,
                bypass_local_separation=True,
                robust_feasibility_tolerance=1e-4,
            )
            self.assertEqual(
                first=res.iterations,
                second=2,
                msg=(
                    "Iterations for Watson 43 instance solved with "
                    f"subsolver {name} not as expected"
                ),
            )
            np.testing.assert_allclose(
                actual=res.final_objective_value,
                # this value can be hand-calculated by analyzing the
                # initial master problem
                desired=0.9781633,
                rtol=0,
                atol=5e-3,
                err_msg=(
                    "Final objective for Watson 43 instance solved with "
                    f"subsolver {name} not as expected"
                ),
            )

    @unittest.skipUnless(
        scip_available and scip_license_is_valid, "SCIP is not available and licensed."
    )
    def test_coefficient_matching_partitioning_insensitive(self):
        """
        Check that result for instance with constraint subject to
        coefficient matching is insensitive to DOF partitioning. Model
        is based on Mitsos (2011) semi-infinite programming instance
        4_3.
        """
        m = self.build_mitsos_4_3()

        global_solver = SolverFactory("scip")
        pyros_solver = SolverFactory("pyros")

        # solve with PyROS
        partitionings = [
            {"fsv": [m.x1, m.x2, m.x3], "ssv": []},
            {"fsv": [], "ssv": [m.x1, m.x2, m.x3]},
        ]
        for partitioning in partitionings:
            res = pyros_solver.solve(
                model=m,
                first_stage_variables=partitioning["fsv"],
                second_stage_variables=partitioning["ssv"],
                uncertain_params=[m.u],
                uncertainty_set=BoxSet(bounds=[[0, 1]]),
                local_solver=global_solver,
                global_solver=global_solver,
                objective_focus=ObjectiveType.worst_case,
                solve_master_globally=True,
                bypass_local_separation=True,
                robust_feasibility_tolerance=1e-4,
            )
            self.assertEqual(
                first=res.iterations,
                second=2,
                msg=(
                    "Iterations for Watson 43 instance solved with "
                    f"first-stage vars {[fsv.name for fsv in partitioning['fsv']]} "
                    f"second-stage vars {[ssv.name for ssv in partitioning['ssv']]} "
                    "not as expected"
                ),
            )
            np.testing.assert_allclose(
                actual=res.final_objective_value,
                desired=0.9781633,
                rtol=0,
                atol=5e-3,
                err_msg=(
                    "Final objective for Watson 43 instance solved with "
                    f"first-stage vars {[fsv.name for fsv in partitioning['fsv']]} "
                    f"second-stage vars {[ssv.name for ssv in partitioning['ssv']]} "
                    "not as expected"
                ),
            )

    @unittest.skipUnless(baron_available, "BARON is not available.")
    def test_coefficient_matching_robust_infeasible_proof_in_pyros(self):
        # Write the deterministic Pyomo model
        m = build_leyffer()
        m.eq_con = Constraint(
            expr=m.u * (m.x1**3 + 0.5)
            - 5 * m.u * m.x1 * m.x2
            + m.u * (m.x1 + 2)
            + m.u**2
            == 0
        )

        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory("baron")
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver

        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1, m.x2],
            second_stage_variables=[],
            uncertain_params=[m.u],
            uncertainty_set=interval,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            options={
                "objective_focus": ObjectiveType.worst_case,
                "solve_master_globally": True,
            },
        )

        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_infeasible,
            msg="Robust infeasible problem not identified via coefficient matching.",
        )

    @unittest.skipUnless(ipopt_available, "IPOPT not available.")
    def test_coefficient_matching_nonlinear_expr(self):
        """
        Test behavior of PyROS solver for model with
        equality constraint that cannot be reformulated via
        coefficient matching due to nonlinearity.
        """
        m = build_leyffer()
        m.eq_con = Constraint(expr=m.u**2 * (m.x2 - 1) == 0)

        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory("ipopt")
        global_subsolver = SolverFactory("ipopt")

        # Call the PyROS solver
        with LoggingIntercept(module="pyomo.contrib.pyros", level=logging.DEBUG) as LOG:
            results = pyros_solver.solve(
                model=m,
                first_stage_variables=[m.x1],
                second_stage_variables=[m.x2],
                uncertain_params=[m.u],
                uncertainty_set=interval,
                local_solver=local_subsolver,
                global_solver=global_subsolver,
                options={
                    "objective_focus": ObjectiveType.worst_case,
                    "solve_master_globally": False,
                    "bypass_global_separation": True,
                    "decision_rule_order": 1,
                },
            )

        pyros_log = LOG.getvalue()
        self.assertRegex(
            pyros_log, r".*Equality constraint '.*eq_con.*'.*cannot be written.*"
        )

        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_feasible,
        )


@unittest.skipUnless(scip_available, "Global NLP solver is not available.")
class testBypassingSeparation(unittest.TestCase):
    @unittest.skipUnless(scip_available, "SCIP is not available.")
    @unittest.skipUnless(ipopt_available, "IPOPT is not available.")
    def test_bypass_global_separation(self):
        """Test bypassing of global separation solve calls."""
        m = build_leyffer_two_cons()

        # Define the uncertainty set
        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('ipopt')
        global_subsolver = SolverFactory("scip")

        # Call the PyROS solver
        with LoggingIntercept(level=logging.WARNING) as LOG:
            results = pyros_solver.solve(
                model=m,
                first_stage_variables=[m.x1],
                second_stage_variables=[m.x2],
                uncertain_params=[m.u],
                uncertainty_set=interval,
                local_solver=local_subsolver,
                global_solver=global_subsolver,
                options={
                    "objective_focus": ObjectiveType.worst_case,
                    "solve_master_globally": True,
                    "decision_rule_order": 0,
                    "bypass_global_separation": True,
                },
            )

        # check termination robust optimal
        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_optimal,
            msg="Returned termination condition is not return robust_optimal.",
        )

        # since robust optimal, we also expect warning-level logger
        # message about bypassing of global separation subproblems
        warning_msgs = LOG.getvalue()
        self.assertRegex(
            warning_msgs,
            (
                r".*Option to bypass global separation was chosen\. "
                r"Robust feasibility and optimality of the reported "
                r"solution are not guaranteed\."
            ),
        )


@unittest.skipUnless(
    baron_available and baron_license_is_valid,
    "Global NLP solver is not available and licensed.",
)
class testUninitializedVars(unittest.TestCase):
    def test_uninitialized_vars(self):
        """
        Test a simple PyROS model instance with uninitialized
        first-stage and second-stage variables.
        """
        m = ConcreteModel()

        # parameters
        m.ell0 = Param(initialize=1)
        m.u0 = Param(initialize=3)
        m.ell = Param(initialize=1)
        m.u = Param(initialize=5)
        m.p = Param(initialize=m.u0, mutable=True)
        m.r = Param(initialize=0.1)

        # variables
        m.x = Var(bounds=(m.ell0, m.u0))
        m.z = Var(bounds=(m.ell0, m.p))
        m.t = Var(initialize=1, bounds=(0, m.r))
        m.w = Var(bounds=(0, 1))

        # objectives
        m.obj = Objective(expr=-m.x**2 + m.z**2)

        # auxiliary constraints
        m.t_lb_con = Constraint(expr=m.x - m.z <= m.t)
        m.t_ub_con = Constraint(expr=-m.t <= m.x - m.z)

        # other constraints
        m.con1 = Constraint(expr=m.x - m.z >= 0.1)
        m.eq_con = Constraint(expr=m.w == 0.5 * m.t)

        box_set = BoxSet(bounds=((value(m.ell), value(m.u)),))

        # solvers
        local_solver = SolverFactory("ipopt")
        global_solver = SolverFactory("baron")

        # pyros setup
        pyros_solver = SolverFactory("pyros")

        # solve for different decision rule orders
        for dr_order in [0, 1, 2]:
            model = m.clone()

            # degree of freedom partitioning
            fsv = [model.x]
            ssv = [model.z, model.t]
            uncertain_params = [model.p]

            res = pyros_solver.solve(
                model=model,
                first_stage_variables=fsv,
                second_stage_variables=ssv,
                uncertain_params=uncertain_params,
                uncertainty_set=box_set,
                local_solver=local_solver,
                global_solver=global_solver,
                objective_focus=ObjectiveType.worst_case,
                decision_rule_order=2,
                solve_master_globally=True,
            )

            self.assertEqual(
                res.pyros_termination_condition,
                pyrosTerminationCondition.robust_optimal,
                msg=(
                    "Returned termination condition for solve with"
                    f"decision rule order {dr_order} is not return "
                    "robust_optimal."
                ),
            )


@unittest.skipUnless(scip_available, "Global NLP solver is not available.")
class testModelMultipleObjectives(unittest.TestCase):
    """
    This class contains tests for models with multiple
    Objective attributes.
    """

    def test_multiple_objs(self):
        """Test bypassing of global separation solve calls."""
        m = build_leyffer_two_cons()
        m.obj2 = Objective(expr=m.obj.expr / 2)

        # add block, with another objective
        m.b = Block()
        m.b.obj = Objective(expr=m.obj.expr / 2)

        # Define the uncertainty set
        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('ipopt')
        global_subsolver = SolverFactory("scip")

        solve_kwargs = dict(
            model=m,
            first_stage_variables=[m.x1],
            second_stage_variables=[m.x2],
            uncertain_params=[m.u],
            uncertainty_set=interval,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            options={
                "objective_focus": ObjectiveType.worst_case,
                "solve_master_globally": True,
                "decision_rule_order": 0,
            },
        )

        # check validation error raised due to multiple objectives
        with self.assertRaisesRegex(
            ValueError, r"Expected model with exactly 1 active objective.*has 3"
        ):
            pyros_solver.solve(**solve_kwargs)

        # check validation error raised due to multiple objectives
        m.b.obj.deactivate()
        with self.assertRaisesRegex(
            ValueError, r"Expected model with exactly 1 active objective.*has 2"
        ):
            pyros_solver.solve(**solve_kwargs)

        # now solve with only one active obj,
        # check successful termination
        m.obj2.deactivate()
        res = pyros_solver.solve(**solve_kwargs)
        self.assertIs(
            res.pyros_termination_condition, pyrosTerminationCondition.robust_optimal
        )

        # check active objectives
        self.assertEqual(len(list(m.component_data_objects(Objective, active=True))), 1)
        self.assertTrue(m.obj.active)

        # swap to maximization objective.
        # and solve again
        m.obj_max = Objective(expr=-m.obj.expr, sense=pyo_max)
        m.obj.deactivate()
        max_obj_res = pyros_solver.solve(**solve_kwargs)

        # check active objectives
        self.assertEqual(len(list(m.component_data_objects(Objective, active=True))), 1)
        self.assertTrue(m.obj_max.active)

        self.assertTrue(
            math.isclose(
                res.final_objective_value,
                -max_obj_res.final_objective_value,
                abs_tol=2e-4,  # 2x the default robust feasibility tolerance
            ),
            msg=(
                f"Robust optimal objective value {res.final_objective_value} "
                "for problem with minimization objective not close to "
                f"negative of value {max_obj_res.final_objective_value} "
                "of equivalent maximization objective."
            ),
        )


class TestMasterFeasibilityUnitConsistency(unittest.TestCase):
    """
    Test cases for models with unit-laden model components.
    """

    @unittest.skipUnless(
        scip_available and scip_license_is_valid, "SCIP is not available and licensed."
    )
    def test_two_stg_mod_with_axis_aligned_set(self):
        """
        Test two-stage model with `AxisAlignedEllipsoidalSet`
        as the uncertainty set.
        """
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None), units=u.m)
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u1 = Param(initialize=1.125, mutable=True, units=u.s)
        m.u2 = Param(initialize=1, mutable=True, units=u.m**2)

        m.con1 = Constraint(expr=m.x1 * m.u1 ** (0.5) - m.x2 * m.u1 <= 2)
        m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u1 == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - m.u2) ** 2)

        # Define the uncertainty set
        # we take the parameter `u2` to be 'fixed'
        ellipsoid = AxisAlignedEllipsoidalSet(center=[1.125, 1], half_lengths=[1, 0])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory("scip")
        global_subsolver = SolverFactory("scip")

        # Call the PyROS solver
        # note: second-stage variable and uncertain params have units
        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1],
            second_stage_variables=[m.x2],
            uncertain_params=[m.u1, m.u2],
            uncertainty_set=ellipsoid,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            options={
                "objective_focus": ObjectiveType.worst_case,
                "solve_master_globally": True,
            },
        )

        # check successful termination
        # and that more than one iteration required
        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_optimal,
            msg="Did not identify robust optimal solution to problem instance.",
        )
        self.assertGreater(
            results.iterations,
            1,
            msg=(
                "PyROS requires no more than one iteration to solve the model."
                " Hence master feasibility problem construction not tested."
                " Consider implementing a more challenging model for this"
                " test case."
            ),
        )


class TestSubsolverTiming(unittest.TestCase):
    """
    Tests to confirm that the PyROS subsolver timing routines
    work appropriately.
    """

    def simple_nlp_model(self):
        """
        Create simple NLP for the unit tests defined
        within this class
        """
        return build_leyffer_two_cons_two_params()

    @unittest.skipUnless(
        SolverFactory('appsi_ipopt').available(exception_flag=False),
        "Local NLP solver is not available.",
    )
    def test_pyros_appsi_ipopt(self):
        """
        Test PyROS usage with solver appsi ipopt
        works without exceptions.
        """
        m = self.simple_nlp_model()

        # Define the uncertainty set
        # we take the parameter `u2` to be 'fixed'
        ellipsoid = AxisAlignedEllipsoidalSet(center=[1.125, 1], half_lengths=[1, 0])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('appsi_ipopt')
        global_subsolver = SolverFactory("appsi_ipopt")

        # Call the PyROS solver
        # note: second-stage variable and uncertain params have units
        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1],
            second_stage_variables=[m.x2],
            uncertain_params=[m.u1, m.u2],
            uncertainty_set=ellipsoid,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            objective_focus=ObjectiveType.worst_case,
            solve_master_globally=False,
            bypass_global_separation=True,
        )
        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_feasible,
            msg="Did not identify robust optimal solution to problem instance.",
        )
        self.assertFalse(
            math.isnan(results.time),
            msg=(
                "PyROS solve time is nan (expected otherwise since subsolver"
                "time estimates are made using TicTocTimer"
            ),
        )

    @unittest.skipUnless(
        SolverFactory('gams:ipopt').available(exception_flag=False),
        "Local NLP solver GAMS/IPOPT is not available.",
    )
    def test_pyros_gams_ipopt(self):
        """
        Test PyROS usage with solver GAMS ipopt
        works without exceptions.
        """
        m = self.simple_nlp_model()

        # Define the uncertainty set
        # we take the parameter `u2` to be 'fixed'
        ellipsoid = AxisAlignedEllipsoidalSet(center=[1.125, 1], half_lengths=[1, 0])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('gams:ipopt')
        global_subsolver = SolverFactory("gams:ipopt")

        # Call the PyROS solver
        # note: second-stage variable and uncertain params have units
        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1],
            second_stage_variables=[m.x2],
            uncertain_params=[m.u1, m.u2],
            uncertainty_set=ellipsoid,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            objective_focus=ObjectiveType.worst_case,
            solve_master_globally=False,
            bypass_global_separation=True,
        )
        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_feasible,
            msg="Did not identify robust optimal solution to problem instance.",
        )
        self.assertFalse(
            math.isnan(results.time),
            msg=(
                "PyROS solve time is nan (expected otherwise since subsolver"
                "time estimates are made using TicTocTimer"
            ),
        )

    @unittest.skipUnless(
        scip_available and scip_license_is_valid, "SCIP is not available and licensed."
    )
    def test_two_stg_mod_with_intersection_set(self):
        """
        Test two-stage model with `AxisAlignedEllipsoidalSet`
        as the uncertainty set.
        """
        m = self.simple_nlp_model()

        # construct the IntersectionSet
        ellipsoid = AxisAlignedEllipsoidalSet(center=[1.125, 1], half_lengths=[1, 0])
        bset = BoxSet(bounds=[[1, 2], [0.5, 1.5]])
        iset = IntersectionSet(ellipsoid=ellipsoid, bset=bset)

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory("scip")
        global_subsolver = SolverFactory("scip")

        # Call the PyROS solver
        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1, m.x2],
            second_stage_variables=[],
            uncertain_params=[m.u1, m.u2],
            uncertainty_set=iset,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            options={
                "objective_focus": ObjectiveType.worst_case,
                "solve_master_globally": True,
            },
        )

        # check successful termination
        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_optimal,
            msg="Did not identify robust optimal solution to problem instance.",
        )
        self.assertGreater(
            results.iterations,
            0,
            msg="Robust infeasible model terminated in 0 iterations (nominal case).",
        )


class TestIterationLogRecord(unittest.TestCase):
    """
    Test the PyROS `IterationLogRecord` class.
    """

    def test_log_header(self):
        """Test method for logging iteration log table header."""
        ans = (
            "------------------------------------------------------------------------------\n"
            "Itn  Objective    1-Stg Shift  2-Stg Shift  #CViol  Max Viol     Wall Time (s)\n"
            "------------------------------------------------------------------------------\n"
        )
        with LoggingIntercept(level=logging.INFO) as LOG:
            IterationLogRecord.log_header(logger.info)

        self.assertEqual(
            LOG.getvalue(),
            ans,
            msg="Messages logged for iteration table header do not match expected result",
        )

    def test_log_standard_iter_record(self):
        """Test logging function for PyROS IterationLogRecord."""

        # for some fields, we choose floats with more than four
        # four decimal points to ensure rounding also matches
        iter_record = IterationLogRecord(
            iteration=4,
            objective=1.234567,
            first_stage_var_shift=2.3456789e-8,
            second_stage_var_shift=3.456789e-7,
            dr_var_shift=1.234567e-7,
            num_violated_cons=10,
            max_violation=7.654321e-3,
            elapsed_time=21.2,
            dr_polishing_success=True,
            all_sep_problems_solved=True,
            global_separation=False,
        )

        # now check record logged as expected
        ans = (
            "4     1.2346e+00  2.3457e-08   3.4568e-07   10      7.6543e-03   "
            "21.200       \n"
        )
        with LoggingIntercept(level=logging.INFO) as LOG:
            iter_record.log(logger.info)
        result = LOG.getvalue()

        self.assertEqual(
            ans,
            result,
            msg="Iteration log record message does not match expected result",
        )

    def test_log_iter_record_polishing_failed(self):
        """Test iteration log record in event of polishing failure."""
        # for some fields, we choose floats with more than four
        # four decimal points to ensure rounding also matches
        iter_record = IterationLogRecord(
            iteration=4,
            objective=1.234567,
            first_stage_var_shift=2.3456789e-8,
            second_stage_var_shift=3.456789e-7,
            dr_var_shift=1.234567e-7,
            num_violated_cons=10,
            max_violation=7.654321e-3,
            elapsed_time=21.2,
            dr_polishing_success=False,
            all_sep_problems_solved=True,
            global_separation=False,
        )

        # now check record logged as expected
        ans = (
            "4     1.2346e+00  2.3457e-08   3.4568e-07*  10      7.6543e-03   "
            "21.200       \n"
        )
        with LoggingIntercept(level=logging.INFO) as LOG:
            iter_record.log(logger.info)
        result = LOG.getvalue()

        self.assertEqual(
            ans,
            result,
            msg="Iteration log record message does not match expected result",
        )

    def test_log_iter_record_global_separation(self):
        """
        Test iteration log record in event global separation performed.
        In this case, a 'g' should be appended to the max violation
        reported. Useful in the event neither local nor global separation
        was bypassed.
        """
        # for some fields, we choose floats with more than four
        # four decimal points to ensure rounding also matches
        iter_record = IterationLogRecord(
            iteration=4,
            objective=1.234567,
            first_stage_var_shift=2.3456789e-8,
            second_stage_var_shift=3.456789e-7,
            dr_var_shift=1.234567e-7,
            num_violated_cons=10,
            max_violation=7.654321e-3,
            elapsed_time=21.2,
            dr_polishing_success=True,
            all_sep_problems_solved=True,
            global_separation=True,
        )

        # now check record logged as expected
        ans = (
            "4     1.2346e+00  2.3457e-08   3.4568e-07   10      7.6543e-03g  "
            "21.200       \n"
        )
        with LoggingIntercept(level=logging.INFO) as LOG:
            iter_record.log(logger.info)
        result = LOG.getvalue()

        self.assertEqual(
            ans,
            result,
            msg="Iteration log record message does not match expected result",
        )

    def test_log_iter_record_not_all_sep_solved(self):
        """
        Test iteration log record in event not all separation problems
        were solved successfully. This may have occurred if the PyROS
        solver time limit was reached, or the user-provides subordinate
        optimizer(s) were unable to solve a separation subproblem
        to an acceptable level.
        A '+' should be appended to the number of second-stage
        inequality constraints found to be violated.
        """
        # for some fields, we choose floats with more than four
        # four decimal points to ensure rounding also matches
        iter_record = IterationLogRecord(
            iteration=4,
            objective=1.234567,
            first_stage_var_shift=2.3456789e-8,
            second_stage_var_shift=3.456789e-7,
            dr_var_shift=1.234567e-7,
            num_violated_cons=10,
            max_violation=7.654321e-3,
            elapsed_time=21.2,
            dr_polishing_success=True,
            all_sep_problems_solved=False,
            global_separation=False,
        )

        # now check record logged as expected
        ans = (
            "4     1.2346e+00  2.3457e-08   3.4568e-07   10+     7.6543e-03   "
            "21.200       \n"
        )
        with LoggingIntercept(level=logging.INFO) as LOG:
            iter_record.log(logger.info)
        result = LOG.getvalue()

        self.assertEqual(
            ans,
            result,
            msg="Iteration log record message does not match expected result",
        )

    def test_log_iter_record_all_special(self):
        """
        Test iteration log record in event DR polishing and global
        separation failed.
        """
        # for some fields, we choose floats with more than four
        # four decimal points to ensure rounding also matches
        iter_record = IterationLogRecord(
            iteration=4,
            objective=1.234567,
            first_stage_var_shift=2.3456789e-8,
            second_stage_var_shift=3.456789e-7,
            dr_var_shift=1.234567e-7,
            num_violated_cons=10,
            max_violation=7.654321e-3,
            elapsed_time=21.2,
            dr_polishing_success=False,
            all_sep_problems_solved=False,
            global_separation=True,
        )

        # now check record logged as expected
        ans = (
            "4     1.2346e+00  2.3457e-08   3.4568e-07*  10+     7.6543e-03g  "
            "21.200       \n"
        )
        with LoggingIntercept(level=logging.INFO) as LOG:
            iter_record.log(logger.info)
        result = LOG.getvalue()

        self.assertEqual(
            ans,
            result,
            msg="Iteration log record message does not match expected result",
        )

    def test_log_iter_record_attrs_none(self):
        """
        Test logging of iteration record in event some
        attributes are of value `None`. In this case, a '-'
        should be printed in lieu of a numerical value.
        Example where this occurs: the first iteration,
        in which there is no first-stage shift or DR shift.
        """
        # for some fields, we choose floats with more than four
        # four decimal points to ensure rounding also matches
        iter_record = IterationLogRecord(
            iteration=0,
            objective=-1.234567,
            first_stage_var_shift=None,
            second_stage_var_shift=None,
            dr_var_shift=None,
            num_violated_cons=10,
            max_violation=7.654321e-3,
            elapsed_time=21.2,
            dr_polishing_success=True,
            all_sep_problems_solved=False,
            global_separation=True,
        )

        # now check record logged as expected
        ans = (
            "0    -1.2346e+00  -            -            10+     7.6543e-03g  "
            "21.200       \n"
        )
        with LoggingIntercept(level=logging.INFO) as LOG:
            iter_record.log(logger.info)
        result = LOG.getvalue()

        self.assertEqual(
            ans,
            result,
            msg="Iteration log record message does not match expected result",
        )


class TestROSolveResults(unittest.TestCase):
    """
    Test PyROS solver results object.
    """

    def test_ro_solve_results_str(self):
        """
        Test string representation of RO solve results object.
        """
        res = ROSolveResults(
            config=SolverFactory("pyros").CONFIG(),
            iterations=4,
            final_objective_value=123.456789,
            time=300.34567,
            pyros_termination_condition=pyrosTerminationCondition.robust_optimal,
        )
        ans = (
            "Termination stats:\n"
            " Iterations            : 4\n"
            " Solve time (wall s)   : 300.346\n"
            " Final objective value : 1.2346e+02\n"
            " Termination condition : pyrosTerminationCondition.robust_optimal"
        )
        self.assertEqual(
            str(res),
            ans,
            msg=(
                "String representation of PyROS results object does not "
                "match expected value"
            ),
        )

    def test_ro_solve_results_str_attrs_none(self):
        """
        Test string representation of PyROS solve results in event
        one of the printed attributes is of value `None`.
        This may occur at instantiation or, for example,
        whenever the PyROS solver confirms robust infeasibility through
        coefficient matching.
        """
        res = ROSolveResults(
            config=SolverFactory("pyros").CONFIG(),
            iterations=0,
            final_objective_value=None,
            time=300.34567,
            pyros_termination_condition=pyrosTerminationCondition.robust_optimal,
        )
        ans = (
            "Termination stats:\n"
            " Iterations            : 0\n"
            " Solve time (wall s)   : 300.346\n"
            " Final objective value : None\n"
            " Termination condition : pyrosTerminationCondition.robust_optimal"
        )
        self.assertEqual(
            str(res),
            ans,
            msg=(
                "String representation of PyROS results object does not "
                "match expected value"
            ),
        )


class TestPyROSSolverLogIntros(unittest.TestCase):
    """
    Test logging of introductory information by PyROS solver.
    """

    def test_log_config(self):
        """
        Test method for logging PyROS solver config dict.
        """
        pyros_solver = SolverFactory("pyros")
        config = pyros_solver.CONFIG(dict(nominal_uncertain_param_vals=[0.5]))
        with LoggingIntercept(level=logging.INFO) as LOG:
            pyros_solver._log_config(logger=logger, config=config, level=logging.INFO)

        ans = (
            "Solver options:\n"
            " time_limit=None\n"
            " keepfiles=False\n"
            " tee=False\n"
            " load_solution=True\n"
            " symbolic_solver_labels=False\n"
            " objective_focus=<ObjectiveType.nominal: 2>\n"
            " nominal_uncertain_param_vals=[0.5]\n"
            " decision_rule_order=0\n"
            " solve_master_globally=False\n"
            " max_iter=-1\n"
            " robust_feasibility_tolerance=0.0001\n"
            " separation_priority_order={}\n"
            " progress_logger=<PreformattedLogger pyomo.contrib.pyros (INFO)>\n"
            " backup_local_solvers=[]\n"
            " backup_global_solvers=[]\n"
            " subproblem_file_directory=None\n"
            " bypass_local_separation=False\n"
            " bypass_global_separation=False\n"
            " p_robustness={}\n" + "-" * 78 + "\n"
        )

        logged_str = LOG.getvalue()
        self.assertEqual(
            logged_str,
            ans,
            msg=(
                "Logger output for PyROS solver config (default case) "
                "does not match expected result."
            ),
        )

    def test_log_intro(self):
        """
        Test logging of PyROS solver introductory messages.
        """
        pyros_solver = SolverFactory("pyros")
        with LoggingIntercept(level=logging.INFO) as LOG:
            pyros_solver._log_intro(logger=logger, level=logging.INFO)

        intro_msgs = LOG.getvalue()

        # last character should be newline; disregard it
        intro_msg_lines = intro_msgs.split("\n")[:-1]

        # check number of lines is as expected
        self.assertEqual(
            len(intro_msg_lines),
            14,
            msg=(
                "PyROS solver introductory message does not contain"
                "the expected number of lines."
            ),
        )

        # first and last lines of the introductory section
        self.assertEqual(intro_msg_lines[0], "=" * 78)
        self.assertEqual(intro_msg_lines[-1], "=" * 78)

        # check regex main text
        self.assertRegex(
            " ".join(intro_msg_lines[1:-1]),
            r"PyROS: The Pyomo Robust Optimization Solver, v.* \(IDAES\)\.",
        )

    def test_log_disclaimer(self):
        """
        Test logging of PyROS solver disclaimer messages.
        """
        pyros_solver = SolverFactory("pyros")
        with LoggingIntercept(level=logging.INFO) as LOG:
            pyros_solver._log_disclaimer(logger=logger, level=logging.INFO)

        disclaimer_msgs = LOG.getvalue()

        # last character should be newline; disregard it
        disclaimer_msg_lines = disclaimer_msgs.split("\n")[:-1]

        # check number of lines is as expected
        self.assertEqual(
            len(disclaimer_msg_lines),
            5,
            msg=(
                "PyROS solver disclaimer message does not contain"
                "the expected number of lines."
            ),
        )

        # regex first line of disclaimer section
        self.assertRegex(disclaimer_msg_lines[0], r"=.* DISCLAIMER .*=")
        # check last line of disclaimer section
        self.assertEqual(disclaimer_msg_lines[-1], "=" * 78)

        # check regex main text
        self.assertRegex(
            " ".join(disclaimer_msg_lines[1:-1]),
            r"PyROS is still under development.*ticket at.*",
        )


class UnavailableSolver:
    def available(self, exception_flag=True):
        if exception_flag:
            raise ApplicationError(f"Solver {self.__class__} not available")
        return False

    def solve(self, model, *args, **kwargs):
        return SolverResults()


class TestPyROSUnavailableSubsolvers(unittest.TestCase):
    """
    Check that appropriate exceptionsa are raised if
    PyROS is invoked with unavailable subsolvers.
    """

    def test_pyros_unavailable_subsolver(self):
        """
        Test PyROS raises expected error message when
        unavailable subsolver is passed.
        """
        m = ConcreteModel()
        m.p = Param(range(3), initialize=0, mutable=True)
        m.z = Var([0, 1], initialize=0)
        m.con = Constraint(expr=m.z[0] + m.z[1] >= m.p[0])
        m.obj = Objective(expr=m.z[0] + m.z[1])

        pyros_solver = SolverFactory("pyros")

        exc_str = r".*Solver.*UnavailableSolver.*not available"
        with self.assertRaisesRegex(ValueError, exc_str):
            # note: ConfigDict interface raises ValueError
            #       once any exception is triggered,
            #       so we check for that instead of ApplicationError
            with LoggingIntercept(level=logging.ERROR) as LOG:
                pyros_solver.solve(
                    model=m,
                    first_stage_variables=[m.z[0]],
                    second_stage_variables=[m.z[1]],
                    uncertain_params=[m.p[0]],
                    uncertainty_set=BoxSet([[0, 1]]),
                    local_solver=SimpleTestSolver(),
                    global_solver=UnavailableSolver(),
                )

        error_msgs = LOG.getvalue()[:-1]
        self.assertRegex(
            error_msgs, r"Output of `available\(\)` method.*global solver.*"
        )

    @unittest.skipUnless(ipopt_available, "IPOPT is not available.")
    def test_pyros_unavailable_backup_subsolver(self):
        """
        Test PyROS raises expected error message when
        unavailable backup subsolver is passed.
        """
        m = ConcreteModel()
        m.p = Param(range(3), initialize=0, mutable=True)
        m.z = Var([0, 1], initialize=0)
        m.con = Constraint(expr=m.z[0] + m.z[1] >= m.p[0])
        m.obj = Objective(expr=m.z[0] + m.z[1])

        pyros_solver = SolverFactory("pyros")

        # note: ConfigDict interface raises ValueError
        #       once any exception is triggered,
        #       so we check for that instead of ApplicationError
        with LoggingIntercept(level=logging.WARNING) as LOG:
            pyros_solver.solve(
                model=m,
                first_stage_variables=[m.z[0]],
                second_stage_variables=[m.z[1]],
                uncertain_params=[m.p[0]],
                uncertainty_set=BoxSet([[0, 1]]),
                local_solver=SolverFactory("ipopt"),
                global_solver=SolverFactory("ipopt"),
                backup_global_solvers=[UnavailableSolver()],
                bypass_global_separation=True,
            )

        error_msgs = LOG.getvalue()[:-1]
        self.assertRegex(
            error_msgs,
            r"Output of `available\(\)` method.*backup global solver.*"
            r"Removing from list.*",
        )


class TestPyROSResolveKwargs(unittest.TestCase):
    """
    Test PyROS resolves kwargs as expected.
    """

    @unittest.skipUnless(ipopt_available, "IPOPT is not available.")
    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_pyros_kwargs_with_overlap(self):
        """
        Test PyROS works as expected when there is overlap between
        keyword arguments passed explicitly and implicitly
        through `options`.
        """
        m = build_leyffer_two_cons_two_params()

        # Define the uncertainty set
        # we take the parameter `u2` to be 'fixed'
        ellipsoid = AxisAlignedEllipsoidalSet(center=[1.125, 1], half_lengths=[1, 0])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('ipopt')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        results = pyros_solver.solve(
            model=m,
            first_stage_variables=[m.x1, m.x2],
            second_stage_variables=[],
            uncertain_params=[m.u1, m.u2],
            uncertainty_set=ellipsoid,
            local_solver=local_subsolver,
            global_solver=global_subsolver,
            bypass_local_separation=True,
            solve_master_globally=True,
            options={
                "objective_focus": ObjectiveType.worst_case,
                "solve_master_globally": False,
                "max_iter": 1,
                "time_limit": 1000,
            },
        )

        # check termination status as expected
        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.max_iter,
            msg="Termination condition not as expected",
        )
        self.assertEqual(
            results.iterations, 1, msg="Number of iterations not as expected"
        )

        # check config resolved as expected
        config = results.config
        self.assertEqual(
            config.bypass_local_separation,
            True,
            msg="Resolved value of kwarg `bypass_local_separation` not as expected.",
        )
        self.assertEqual(
            config.solve_master_globally,
            True,
            msg="Resolved value of kwarg `solve_master_globally` not as expected.",
        )
        self.assertEqual(
            config.max_iter,
            1,
            msg="Resolved value of kwarg `max_iter` not as expected.",
        )
        self.assertEqual(
            config.objective_focus,
            ObjectiveType.worst_case,
            msg="Resolved value of kwarg `objective_focus` not as expected.",
        )
        self.assertEqual(
            config.time_limit,
            1e3,
            msg="Resolved value of kwarg `time_limit` not as expected.",
        )


class SimpleTestSolver:
    """
    Simple test solver class with no actual solve()
    functionality. Written to test unrelated aspects
    of PyROS functionality.
    """

    def available(self, exception_flag=False):
        """
        Check solver available.
        """
        return True

    def solve(self, model, **kwds):
        """
        Return SolverResults object with 'unknown' termination
        condition. Model remains unchanged.
        """
        res = SolverResults()
        res.solver.termination_condition = TerminationCondition.unknown

        return res


class TestPyROSSolverAdvancedValidation(unittest.TestCase):
    """
    Test PyROS solver returns expected exception messages
    when arguments are invalid.
    """

    def build_simple_test_model(self):
        """
        Build simple valid test model.
        """
        return build_leyffer()

    def test_pyros_invalid_model_type(self):
        """
        Test PyROS fails if model is not of correct class.
        """
        mdl = self.build_simple_test_model()

        local_solver = SimpleTestSolver()
        global_solver = SimpleTestSolver()

        pyros = SolverFactory("pyros")

        exc_str = "Model should be of type.*but is of type.*"
        with self.assertRaisesRegex(TypeError, exc_str):
            pyros.solve(
                model=2,
                first_stage_variables=[mdl.x1],
                second_stage_variables=[mdl.x2],
                uncertain_params=[mdl.u],
                uncertainty_set=BoxSet([[1 / 4, 2]]),
                local_solver=local_solver,
                global_solver=global_solver,
            )

    def test_pyros_multiple_objectives(self):
        """
        Test PyROS raises exception if input model has multiple
        objectives.
        """
        mdl = self.build_simple_test_model()
        mdl.obj2 = Objective(expr=(mdl.x1 + mdl.x2))

        local_solver = SimpleTestSolver()
        global_solver = SimpleTestSolver()

        pyros = SolverFactory("pyros")

        exc_str = "Expected model with exactly 1 active.*but.*has 2"
        with self.assertRaisesRegex(ValueError, exc_str):
            pyros.solve(
                model=mdl,
                first_stage_variables=[mdl.x1],
                second_stage_variables=[mdl.x2],
                uncertain_params=[mdl.u],
                uncertainty_set=BoxSet([[1 / 4, 2]]),
                local_solver=local_solver,
                global_solver=global_solver,
            )

    def test_pyros_empty_dof_vars(self):
        """
        Test PyROS solver raises exception raised if there are no
        first-stage variables or second-stage variables.
        """
        # build model
        mdl = self.build_simple_test_model()

        # prepare solvers
        pyros = SolverFactory("pyros")
        local_solver = SimpleTestSolver()
        global_solver = SimpleTestSolver()

        # perform checks
        exc_str = (
            "Arguments `first_stage_variables` and "
            "`second_stage_variables` are both empty lists."
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            pyros.solve(
                model=mdl,
                first_stage_variables=[],
                second_stage_variables=[],
                uncertain_params=[mdl.u],
                uncertainty_set=BoxSet([[1 / 4, 2]]),
                local_solver=local_solver,
                global_solver=global_solver,
            )

    def test_pyros_overlap_dof_vars(self):
        """
        Test PyROS solver raises exception raised if there are Vars
        passed as both first-stage and second-stage.
        """
        # build model
        mdl = self.build_simple_test_model()

        # prepare solvers
        pyros = SolverFactory("pyros")
        local_solver = SimpleTestSolver()
        global_solver = SimpleTestSolver()

        # perform checks
        exc_str = (
            "Arguments `first_stage_variables` and `second_stage_variables` "
            "contain at least one common Var object."
        )
        with LoggingIntercept(level=logging.ERROR) as LOG:
            with self.assertRaisesRegex(ValueError, exc_str):
                pyros.solve(
                    model=mdl,
                    first_stage_variables=[mdl.x1],
                    second_stage_variables=[mdl.x1, mdl.x2],
                    uncertain_params=[mdl.u],
                    uncertainty_set=BoxSet([[1 / 4, 2]]),
                    local_solver=local_solver,
                    global_solver=global_solver,
                )

        # check logger output is as expected
        log_msgs = LOG.getvalue().split("\n")[:-1]
        self.assertEqual(
            len(log_msgs), 3, "Error message does not contain expected number of lines."
        )
        self.assertRegex(
            text=log_msgs[0],
            expected_regex=(
                "The following Vars were found in both `first_stage_variables`"
                "and `second_stage_variables`.*"
            ),
        )
        self.assertRegex(text=log_msgs[1], expected_regex=" 'x1'")
        self.assertRegex(
            text=log_msgs[2],
            expected_regex="Ensure no Vars are included in both arguments.",
        )

    def test_pyros_vars_not_in_model(self):
        """
        Test PyROS appropriately raises exception if there are
        variables not included in active model objective
        or constraints which are not descended from model.
        """
        # set up model
        mdl = self.build_simple_test_model()
        mdl.name = "model1"
        mdl2 = self.build_simple_test_model()
        mdl2.name = "model2"

        # set up solvers
        local_solver = SimpleTestSolver()
        global_solver = SimpleTestSolver()
        pyros = SolverFactory("pyros")

        mdl.bad_con = Constraint(expr=mdl.x1 + mdl2.x2 >= 1)
        mdl2.x3 = Var(initialize=1)

        # now perform checks
        with LoggingIntercept(level=logging.ERROR) as LOG:
            exc_str = "Found Vars.*active.*" "not descended from.*model.*"
            with self.assertRaisesRegex(ValueError, exc_str):
                pyros.solve(
                    model=mdl,
                    first_stage_variables=[mdl.x1, mdl.x2],
                    second_stage_variables=[mdl2.x3],
                    uncertain_params=[mdl.u],
                    uncertainty_set=BoxSet([[1 / 4, 2]]),
                    local_solver=local_solver,
                    global_solver=global_solver,
                )

        log_msgs = LOG.getvalue().split("\n")
        invalid_vars_strs_list = log_msgs[1:-1]
        self.assertEqual(
            len(invalid_vars_strs_list),
            1,
            msg="Number of lines referencing name of invalid Vars not as expected.",
        )
        self.assertRegex(
            text=invalid_vars_strs_list[0], expected_regex=f"{mdl2.x2.name!r}"
        )

    def test_pyros_non_continuous_vars(self):
        """
        Test PyROS raises exception if model contains
        non-continuous variables.
        """
        # build model; make one variable discrete
        mdl = self.build_simple_test_model()
        mdl.x2.domain = NonNegativeIntegers
        mdl.name = "test_model"

        # prepare solvers
        pyros = SolverFactory("pyros")
        local_solver = SimpleTestSolver()
        global_solver = SimpleTestSolver()

        # perform checks
        exc_str = "Model with name 'test_model' contains non-continuous Vars."
        with LoggingIntercept(level=logging.ERROR) as LOG:
            with self.assertRaisesRegex(ValueError, exc_str):
                pyros.solve(
                    model=mdl,
                    first_stage_variables=[mdl.x1],
                    second_stage_variables=[mdl.x2],
                    uncertain_params=[mdl.u],
                    uncertainty_set=BoxSet([[1 / 4, 2]]),
                    local_solver=local_solver,
                    global_solver=global_solver,
                )

        # check logger output is as expected
        log_msgs = LOG.getvalue().split("\n")[:-1]
        self.assertEqual(
            len(log_msgs), 3, "Error message does not contain expected number of lines."
        )
        self.assertRegex(
            text=log_msgs[0],
            expected_regex=(
                "The following Vars of model with name 'test_model' "
                "are non-continuous:"
            ),
        )
        self.assertRegex(text=log_msgs[1], expected_regex=" 'x2'")
        self.assertRegex(
            text=log_msgs[2],
            expected_regex=(
                "Ensure all model variables passed to " "PyROS solver are continuous."
            ),
        )

    def test_pyros_uncertainty_dimension_mismatch(self):
        """
        Test PyROS solver raises exception if uncertainty
        set dimension does not match the number
        of uncertain parameters.
        """
        # build model
        mdl = self.build_simple_test_model()

        # prepare solvers
        pyros = SolverFactory("pyros")
        local_solver = SimpleTestSolver()
        global_solver = SimpleTestSolver()

        # perform checks
        exc_str = (
            r"Length of argument `uncertain_params` does not match dimension "
            r"of argument `uncertainty_set` \(1 != 2\)."
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            pyros.solve(
                model=mdl,
                first_stage_variables=[mdl.x1],
                second_stage_variables=[mdl.x2],
                uncertain_params=[mdl.u],
                uncertainty_set=BoxSet([[1 / 4, 2], [0, 1]]),
                local_solver=local_solver,
                global_solver=global_solver,
            )

    @unittest.skipUnless(ipopt_available, "IPOPT is not available.")
    def test_pyros_nominal_point_not_in_set(self):
        """
        Test PyROS raises exception if nominal point is not in the
        uncertainty set.

        NOTE: need executable solvers to solve set bounding problems
              for validity checks.
        """
        # build model
        mdl = self.build_simple_test_model()

        # prepare solvers
        pyros = SolverFactory("pyros")
        local_solver = SolverFactory("ipopt")
        global_solver = SolverFactory("ipopt")

        # perform checks
        exc_str = (
            r"Nominal uncertain parameter realization \[0\] "
            "is not a point in the uncertainty set.*"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            pyros.solve(
                model=mdl,
                first_stage_variables=[mdl.x1],
                second_stage_variables=[mdl.x2],
                uncertain_params=[mdl.u],
                uncertainty_set=BoxSet([[1 / 4, 2]]),
                local_solver=local_solver,
                global_solver=global_solver,
                nominal_uncertain_param_vals=[0],
            )

    @unittest.skipUnless(ipopt_available, "IPOPT is not available.")
    def test_pyros_nominal_point_len_mismatch(self):
        """
        Test PyROS raises exception if there is mismatch between length
        of nominal uncertain parameter specification and number
        of uncertain parameters.
        """
        # build model
        mdl = self.build_simple_test_model()

        # prepare solvers
        pyros = SolverFactory("pyros")
        local_solver = SolverFactory("ipopt")
        global_solver = SolverFactory("ipopt")

        # perform checks
        exc_str = (
            r"Lengths of arguments `uncertain_params` "
            r"and `nominal_uncertain_param_vals` "
            r"do not match \(1 != 2\)."
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            pyros.solve(
                model=mdl,
                first_stage_variables=[mdl.x1],
                second_stage_variables=[mdl.x2],
                uncertain_params=[mdl.u],
                uncertainty_set=BoxSet([[1 / 4, 2]]),
                local_solver=local_solver,
                global_solver=global_solver,
                nominal_uncertain_param_vals=[0, 1],
            )

    @unittest.skipUnless(ipopt_available, "IPOPT is not available.")
    def test_pyros_invalid_bypass_separation(self):
        """
        Test PyROS raises exception if both local and
        global separation are set to be bypassed.
        """
        # build model
        mdl = self.build_simple_test_model()

        # prepare solvers
        pyros = SolverFactory("pyros")
        local_solver = SolverFactory("ipopt")
        global_solver = SolverFactory("ipopt")

        # perform checks
        exc_str = (
            r"Arguments `bypass_local_separation` and `bypass_global_separation` "
            r"cannot both be True."
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            pyros.solve(
                model=mdl,
                first_stage_variables=[mdl.x1],
                second_stage_variables=[mdl.x2],
                uncertain_params=[mdl.u],
                uncertainty_set=BoxSet([[1 / 4, 2]]),
                local_solver=local_solver,
                global_solver=global_solver,
                bypass_local_separation=True,
                bypass_global_separation=True,
            )


if __name__ == "__main__":
    unittest.main()
