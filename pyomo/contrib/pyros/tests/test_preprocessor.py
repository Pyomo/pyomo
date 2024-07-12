"""
Tests for the PyROS preprocessor.
"""


import logging
import textwrap
import unittest

from pyomo.common.collections import Bunch, ComponentSet, ComponentMap
from pyomo.common.dependencies import numpy as numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
from pyomo.core.base import (
    Var,
    Constraint,
    Objective,
    ConcreteModel,
    Param,
    RangeSet,
    maximize,
    Block,
)
from pyomo.core.base.set_types import (
    NonNegativeReals,
    NonPositiveReals,
    Reals,
)
from pyomo.core.expr import (
    identify_mutable_parameters,
    identify_variables,
    log,
    sin,
    exp,
    RangedExpression,
)
from pyomo.core.expr.compare import assertExpressionsEqual

from pyomo.contrib.pyros.util import (
    ObjectiveType,
    get_effective_var_partitioning,
    get_var_certain_uncertain_bounds,
    get_var_bound_pairs,
    turn_nonadjustable_var_bounds_to_constraints,
    turn_adjustable_var_bounds_to_constraints,
    standardize_inequality_constraints,
    standardize_equality_constraints,
    standardize_active_objective,
    declare_objective_expressions,
    add_decision_rule_constraints,
    add_decision_rule_variables,
    perform_coefficient_matching,
    setup_working_model,
    VariablePartitioning,
    preprocess_model_data,
    log_model_statistics,
)
parameterized, param_available = attempt_import('parameterized')

if not (numpy_available and scipy_available and param_available):
    raise unittest.SkipTest(
        'PyROS preprocessor unit tests require parameterized, numpy, and scipy'
    )
parameterized = parameterized.parameterized


logger = logging.getLogger(__name__)


class TestEffectiveVarPartitioning(unittest.TestCase):
    """
    Test method(s) for identification of nonadjustable variables
    which are not necessarily in the user-provided sequence of
    first-stage variables.
    """

    def build_simple_test_model_data(self):
        """
        Build simple model for effective variable partitioning tests.
        """
        m = ConcreteModel()
        m.x1 = Var(bounds=(2, 2))
        m.x2 = Var()
        m.z = Var()
        m.y = Var(range(1, 5))
        m.q = Param(mutable=True, initialize=1)

        m.c0 = Constraint(expr=m.q + m.x1 + m.z == 0)
        m.c1 = Constraint(expr=(0, m.x1 - m.z, 0))
        m.c2 = Constraint(expr=m.x1 ** 2 - m.z + m.y[1] == 0)
        m.c2_dupl = Constraint(expr=m.x1 ** 2 - m.z + m.y[1] == 0)
        m.c3 = Constraint(expr=m.x1 ** 3 + m.y[1] + 2 * m.y[2] == 0)
        m.c4 = Constraint(
            expr=m.x2 ** 2 + m.y[1] + m.y[2] + m.y[3] + m.y[4] == 0
        )
        m.c5 = Constraint(
            expr=m.x2 + 2 * m.y[2] + m.y[3] + 2 * m.y[4] == 0
        )

        model_data = Bunch()
        model_data.working_model = ConcreteModel()
        model_data.working_model.user_model = mdl = m.clone()
        model_data.working_model.uncertain_params = [mdl.q]

        user_var_partitioning = model_data.working_model.user_var_partitioning = Bunch()
        user_var_partitioning.first_stage_variables = [mdl.x1, mdl.x2]
        user_var_partitioning.second_stage_variables = [mdl.z]
        user_var_partitioning.state_variables = list(mdl.y.values())

        return model_data

    def test_effective_partitioning_system(self):
        """
        Test effective partitioning on an example system of
        constraints.
        """
        model_data = self.build_simple_test_model_data()
        m = model_data.working_model.user_model

        config = Bunch()
        config.decision_rule_order = 0
        config.progress_logger = logger

        expected_partitioning = {
            "first_stage_variables": [m.x1, m.x2, m.z, m.y[1], m.y[2]],
            "second_stage_variables": [],
            "state_variables": [m.y[3], m.y[4]],
        }
        for dr_order in [0, 1, 2]:
            config.decision_rule_order = dr_order
            actual_partitioning = get_effective_var_partitioning(
                model_data=model_data,
                config=config,
            )
            for vartype, expected_vars in expected_partitioning.items():
                actual_vars = getattr(actual_partitioning, vartype)
                self.assertEqual(
                    ComponentSet(expected_vars),
                    ComponentSet(actual_vars),
                    msg=(
                        f"Effective {vartype!r} are not as expected "
                        f"for decision rule order {config.decision_rule_order}. "
                        "\n"
                        f"Expected: {[var.name for var in expected_vars]}"
                        "\n"
                        f"Actual: {[var.name for var in actual_vars]}"
                    ),
                )

        # linear coefficient below tolerance;
        # that should prevent pretriangularization
        m.c2.set_value(m.x1 ** 2 + m.z + 1e-10 * m.y[1] == 0)
        m.c2_dupl.set_value(m.x1 ** 2 + m.z + 1e-10 * m.y[1] == 0)
        expected_partitioning = {
            "first_stage_variables": [m.x1, m.x2, m.z],
            "second_stage_variables": [],
            "state_variables": list(m.y.values()),
        }
        for dr_order in [0, 1, 2]:
            config.decision_rule_order = dr_order
            actual_partitioning = get_effective_var_partitioning(
                model_data=model_data,
                config=config,
            )
            for vartype, expected_vars in expected_partitioning.items():
                actual_vars = getattr(actual_partitioning, vartype)
                self.assertEqual(
                    ComponentSet(expected_vars),
                    ComponentSet(actual_vars),
                    msg=(
                        f"Effective {vartype!r} are not as expected "
                        f"for decision rule order {config.decision_rule_order}. "
                        "\n"
                        f"Expected: {[var.name for var in expected_vars]}"
                        "\n"
                        f"Actual: {[var.name for var in actual_vars]}"
                    ),
                )

        # put linear coefs above tolerance again:
        # original behavior expected
        m.c2.set_value(1e-6 * m.y[1] + m.x1 ** 2 + m.z + 1e-10 * m.y[1] == 0)
        m.c2_dupl.set_value(1e-6 * m.y[1] + m.x1 ** 2 + m.z + 1e-10 * m.y[1] == 0)
        expected_partitioning = {
            "first_stage_variables": [m.x1, m.x2, m.z, m.y[1], m.y[2]],
            "second_stage_variables": [],
            "state_variables": [m.y[3], m.y[4]],
        }
        for dr_order in [0, 1, 2]:
            config.decision_rule_order = dr_order
            actual_partitioning = get_effective_var_partitioning(
                model_data=model_data,
                config=config,
            )
            for vartype, expected_vars in expected_partitioning.items():
                actual_vars = getattr(actual_partitioning, vartype)
                self.assertEqual(
                    ComponentSet(expected_vars),
                    ComponentSet(actual_vars),
                    msg=(
                        f"Effective {vartype!r} are not as expected "
                        f"for decision rule order {config.decision_rule_order}. "
                        "\n"
                        f"Expected: {[var.name for var in expected_vars]}"
                        "\n"
                        f"Actual: {[var.name for var in actual_vars]}"
                    ),
                )

        # introducing this simple nonlinearity prevents
        # y[2] from being identified as pretriangular
        expected_partitioning = {
            "first_stage_variables": [m.x1, m.x2, m.z, m.y[1]],
            "second_stage_variables": [],
            "state_variables": [m.y[2], m.y[3], m.y[4]],
        }
        m.c3.set_value(m.x1 ** 3 + m.y[1] + 2 * m.y[1] * m.y[2] == 0)
        for dr_order in [0, 1, 2]:
            config.decision_rule_order = dr_order
            actual_partitioning = get_effective_var_partitioning(
                model_data=model_data,
                config=config,
            )
            for vartype, expected_vars in expected_partitioning.items():
                actual_vars = getattr(actual_partitioning, vartype)
                self.assertEqual(
                    ComponentSet(expected_vars),
                    ComponentSet(actual_vars),
                    msg=(
                        f"Effective {vartype!r} are not as expected "
                        f"for decision rule order {config.decision_rule_order}. "
                        "\n"
                        f"Expected: {[var.name for var in expected_vars]}"
                        "\n"
                        f"Actual: {[var.name for var in actual_vars]}"
                    ),
                )

        # fixing y[2] should make y[2] nonadjustable regardless
        m.y[2].fix(10)
        expected_partitioning = {
            "first_stage_variables": [m.x1, m.x2, m.z, m.y[1], m.y[2]],
            "second_stage_variables": [],
            "state_variables": [m.y[3], m.y[4]],
        }
        for dr_order in [0, 1, 2]:
            config.decision_rule_order = dr_order
            actual_partitioning = get_effective_var_partitioning(
                model_data=model_data,
                config=config,
            )
            for vartype, expected_vars in expected_partitioning.items():
                actual_vars = getattr(actual_partitioning, vartype)
                self.assertEqual(
                    ComponentSet(expected_vars),
                    ComponentSet(actual_vars),
                    msg=(
                        f"Effective {vartype!r} are not as expected "
                        f"for decision rule order {config.decision_rule_order}. "
                        "\n"
                        f"Expected: {[var.name for var in expected_vars]}"
                        "\n"
                        f"Actual: {[var.name for var in actual_vars]}"
                    ),
                )

    def test_effective_partitioning_modified_linear_system(self):
        """
        Test effective partitioning on modified system of equations.
        """
        model_data = self.build_simple_test_model_data()
        m = model_data.working_model.user_model

        # now the second-stage variable can't be determined uniquely;
        # can't pretriangularize this unless z already known to be
        # nonadjustable
        m.c1.set_value((0, m.x1 + m.z ** 2, 0))

        config = Bunch()
        config.decision_rule_order = 0
        config.progress_logger = logger

        expected_partitioning_static_dr = {
            "first_stage_variables": [m.x1, m.x2, m.z, m.y[1], m.y[2]],
            "second_stage_variables": [],
            "state_variables": [m.y[3], m.y[4]],
        }
        actual_partitioning_static_dr = get_effective_var_partitioning(
            model_data=model_data,
            config=config,
        )
        for vartype, expected_vars in expected_partitioning_static_dr.items():
            actual_vars = getattr(actual_partitioning_static_dr, vartype)
            self.assertEqual(
                ComponentSet(expected_vars),
                ComponentSet(actual_vars),
                msg=(
                    f"Effective {vartype!r} are not as expected "
                    f"for decision rule order {config.decision_rule_order}. "
                    "\n"
                    f"Expected: {[var.name for var in expected_vars]}"
                    "\n"
                    f"Actual: {[var.name for var in actual_vars]}"
                ),
            )

        config.decision_rule_order = 1
        expected_partitioning_nonstatic_dr = {
            "first_stage_variables": [m.x1, m.x2],
            "second_stage_variables": [m.z],
            "state_variables": list(m.y.values()),
        }
        for dr_order in [1, 2]:
            actual_partitioning_nonstatic_dr = get_effective_var_partitioning(
                model_data=model_data,
                config=config,
            )
            for vartype, expected_vars in expected_partitioning_nonstatic_dr.items():
                actual_vars = getattr(actual_partitioning_nonstatic_dr, vartype)
                self.assertEqual(
                    ComponentSet(expected_vars),
                    ComponentSet(actual_vars),
                    msg=(
                        f"Effective {vartype!r} are not as expected "
                        f"for decision rule order {config.decision_rule_order}. "
                        "\n"
                        f"Expected: {[var.name for var in expected_vars]}"
                        "\n"
                        f"Actual: {[var.name for var in actual_vars]}"
                    ),
                )


class TestSetupModelData(unittest.TestCase):
    """
    Test method for setting up the working model works as expected.
    """
    def build_test_model_data(self):
        """
        Build model data object for the preprocessor.
        """
        model_data = Bunch()
        model_data.original_model = m = ConcreteModel()

        # PARAMS: one uncertain, one certain
        m.p = Param(initialize=2, mutable=True)
        m.q = Param(initialize=4.5, mutable=True)

        # first-stage variables
        m.x1 = Var(bounds=(0, m.q), initialize=1)
        m.x2 = Var(domain=NonNegativeReals, bounds=[m.p, m.p], initialize=m.p)

        # second-stage variables
        m.z1 = Var(domain=RangeSet(2, 4, 0), bounds=[-m.p, m.q], initialize=2)
        m.z2 = Var(bounds=(-2 * m.q ** 2, None), initialize=1)
        m.z3 = Var(bounds=(-m.q, 0), initialize=0)
        m.z4 = Var(initialize=5)
        m.z5 = Var(domain=NonNegativeReals, bounds=(m.q, m.q))

        # state variables
        m.y1 = Var(domain=NonNegativeReals, initialize=0)
        m.y2 = Var(initialize=10)
        # note: y3 out-of-scope, as it will not appear in the active
        #       Objective and Constraint objects
        m.y3 = Var(domain=RangeSet(0, 1, 0), bounds=(0.2, 0.5))

        # fix some variables
        m.z4.fix()
        m.y2.fix()

        # EQUALITY CONSTRAINTS
        m.eq1 = Constraint(expr=m.q * (m.z3 + m.x2) == 0)
        m.eq2 = Constraint(expr=m.x1 - m.z1 == 0)
        m.eq3 = Constraint(expr=m.x1 ** 2 + m.x2 + m.p * m.z2 == m.p)
        m.eq4 = Constraint(expr=m.z3 + m.y1 == m.q)

        # INEQUALITY CONSTRAINTS
        m.ineq1 = Constraint(expr=(-m.p, m.x1 + m.z1, exp(m.q)))
        m.ineq2 = Constraint(expr=(0, m.x1 + m.x2, 10))
        m.ineq3 = Constraint(expr=(2 * m.q, 2 * (m.z3 + m.y1), 2 * m.q))
        m.ineq4 = Constraint(expr=-m.q <= m.y2 ** 2 + log(m.y2))

        # out of scope: deactivated
        m.ineq5 = Constraint(expr=m.y3 <= m.q)
        m.ineq5.deactivate()

        # OBJECTIVE
        # contains a rich combination of first-stage and second-stage terms
        m.obj = Objective(
            expr=(
                m.p ** 2
                + 2 * m.p * m.q
                + log(m.x1)
                + 2 * m.p * m.x1
                + m.q ** 2 * m.x1
                + m.p ** 3 * (m.z1 + m.z2 + m.y1)
                + m.z4
                + m.z5
            ),
        )

        # set up the var partitioning
        user_var_partitioning = VariablePartitioning(
            first_stage_variables=[m.x1, m.x2],
            second_stage_variables=[m.z1, m.z2, m.z3, m.z4, m.z5],
            # note: y3 out of scope, so excluded
            state_variables=[m.y1, m.y2],
        )

        return model_data, user_var_partitioning

    def test_setup_working_model(self):
        """
        Test method for setting up the working model is as expected.
        """
        model_data, user_var_partitioning = self.build_test_model_data()
        om = model_data.original_model
        config = Bunch(uncertain_params=[om.q])

        setup_working_model(model_data, config, user_var_partitioning)
        working_model = model_data.working_model

        # active constraints
        m = model_data.working_model.user_model
        self.assertEqual(
            ComponentSet(working_model.original_active_equality_cons),
            ComponentSet([m.eq1, m.eq2, m.eq3, m.eq4]),
        )
        self.assertEqual(
            ComponentSet(working_model.original_active_inequality_cons),
            ComponentSet([m.ineq1, m.ineq2, m.ineq3, m.ineq4]),
        )

        # active objective
        self.assertTrue(m.obj.active)

        # user var partitioning
        up = working_model.user_var_partitioning
        self.assertEqual(
            ComponentSet(up.first_stage_variables),
            ComponentSet([m.x1, m.x2]),
        )
        self.assertEqual(
            ComponentSet(up.second_stage_variables),
            ComponentSet([m.z1, m.z2, m.z3, m.z4, m.z5]),
        )
        self.assertEqual(
            ComponentSet(up.state_variables),
            ComponentSet([m.y1, m.y2]),
        )

        # uncertain params
        self.assertEqual(
            ComponentSet(working_model.uncertain_params),
            ComponentSet([m.q]),
        )

        # ensure original model unchanged
        self.assertFalse(
            hasattr(om, "util"),
            msg="Original model still has temporary util block",
        )

        # constraint partitioning initialization
        self.assertEqual(working_model.effective_first_stage_inequality_cons, [])
        self.assertEqual(working_model.effective_performance_inequality_cons, [])
        self.assertEqual(working_model.effective_first_stage_equality_cons, [])
        self.assertEqual(working_model.effective_performance_equality_cons, [])


class TestResolveVarBounds(unittest.TestCase):
    """
    Tests for resolution of variable bounds.
    """
    def test_resolve_var_bounds(self):
        """
        Test resolve variable bounds.
        """
        m = ConcreteModel()
        m.q1 = Param(initialize=1, mutable=True)
        m.q2 = Param(initialize=1, mutable=True)
        m.p1 = Param(initialize=5, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.z1 = Var(bounds=(0, 1))
        m.z2 = Var(bounds=(1, 1))
        m.z3 = Var(domain=NonNegativeReals, bounds=(2, 4))
        m.z4 = Var(domain=NonNegativeReals, bounds=(m.q1, 0))
        m.z5 = Var(domain=RangeSet(2, 4, 0), bounds=(4, 6))
        m.z6 = Var(domain=NonNegativeReals, bounds=(m.q1, m.q1))
        m.z7 = Var(domain=NonNegativeReals, bounds=(m.q1, 1 * m.q1))
        m.z8 = Var(domain=RangeSet(0, 5, 0), bounds=[m.q1, m.q2])
        m.z9 = Var(domain=RangeSet(0, 5, 0), bounds=[m.q1, m.p1])
        m.z10 = Var(domain=RangeSet(0, 5, 0), bounds=[m.q1, m.p2])

        # useful for checking domains later
        original_var_domains = ComponentMap((
            (var, var.domain) for var in
            (m.z1, m.z2, m.z3, m.z4, m.z5, m.z6, m.z7, m.z8, m.z9, m.z10)
        ))

        expected_bounds = (
            (m.z1, (0, None, 1), (None, None, None)),
            (m.z2, (None, 1, None), (None, None, None)),
            (m.z3, (2, None, 4), (None, None, None)),
            (m.z4, (None, 0, None), (m.q1, None, None)),
            (m.z5, (None, 4, None), (None, None, None)),
            (m.z6, (0, None, None), (None, m.q1, None)),
            # the 1 * q expression is simplified to just q
            # when variable bounds are specified
            (m.z7, (0, None, None), (None, m.q1, None)),
            (m.z8, (0, None, 5), (m.q1, None, m.q2)),
            (m.z9, (0, None, m.p1), (m.q1, None, None)),
            (m.z10, (0, None, m.p2), (m.q1, None, None)),
        )
        for var, exp_cert_bounds, exp_uncert_bounds in expected_bounds:
            actual_cert_bounds, actual_uncert_bounds = get_var_certain_uncertain_bounds(
                var, [m.q1, m.q2]
            )
            for btype, exp_bound in zip(("lower", "eq", "upper"), exp_cert_bounds):
                actual_bound = getattr(actual_cert_bounds, btype)
                self.assertIs(
                    exp_bound,
                    actual_bound,
                    msg=(
                        f"Resolved certain {btype} bound for variable "
                        f"{var.name!r} is not as expected. "
                        "\n Expected certain bounds: "
                        f"lower={str(exp_cert_bounds[0])}, "
                        f"eq={str(exp_cert_bounds[1])}, "
                        f"upper={str(exp_cert_bounds[2])} "
                        "\n Actual certain bounds: "
                        f"lower={str(actual_cert_bounds.lower)}, "
                        f"eq={str(actual_cert_bounds.eq)}, "
                        f"upper={str(actual_cert_bounds.upper)} "
                    ),
                )

            for btype, exp_bound in zip(("lower", "eq", "upper"), exp_uncert_bounds):
                actual_bound = getattr(actual_uncert_bounds, btype)
                self.assertIs(
                    exp_bound,
                    actual_bound,
                    msg=(
                        f"Resolved uncertain {btype} bound for variable "
                        f"{var.name!r} is not as expected. "
                        "\n Expected uncertain bounds: "
                        f"lower={str(exp_uncert_bounds[0])}, "
                        f"eq={str(exp_uncert_bounds[1])}, "
                        f"upper={str(exp_uncert_bounds[2])} "
                        "\n Actual uncertain bounds: "
                        f"lower={str(actual_uncert_bounds.lower)}, "
                        f"eq={str(actual_uncert_bounds.eq)}, "
                        f"upper={str(actual_uncert_bounds.upper)} "
                    ),
                )

        # the bounds resolution method should leave domains unaltered
        for var, orig_domain in original_var_domains.items():
            self.assertIs(
                var.domain,
                orig_domain,
                msg=(
                    f"Domain for var {var.name!r} appears to have been changed "
                    f"from {orig_domain} to {var.domain} "
                    "by the bounds resolution method "
                    f"{get_var_certain_uncertain_bounds.__name__!r}."
                )
            )


class TestTurnVarBoundsToConstraints(unittest.TestCase):
    """
    Tests for reformulating variable bounds to explicit
    inequality/equality constraints.
    """

    def build_simple_test_model_data(self):
        """
        Build simple model data object for turning bounds
        to constraints.
        """
        model_data = Bunch()

        model_data.working_model = ConcreteModel()
        model_data.working_model.user_model = m = ConcreteModel()

        m.q1 = Param(initialize=1, mutable=True)
        m.q2 = Param(initialize=1, mutable=True)
        m.p1 = Param(initialize=5, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)

        m.z1 = Var(bounds=(None, None))
        m.z2 = Var(bounds=(1, 1))
        m.z3 = Var(domain=NonNegativeReals, bounds=(2, m.p1))
        m.z4 = Var(domain=NonNegativeReals, bounds=(m.q1, 0))
        m.z5 = Var(domain=RangeSet(2, 4, 0), bounds=(4, m.q2))
        m.z6 = Var(domain=NonNegativeReals, bounds=(m.q1, m.q1))
        m.z7 = Var(domain=NonPositiveReals, bounds=(m.q1, 1 * m.q1))
        m.z8 = Var(domain=RangeSet(0, 5, 0), bounds=[m.q1, m.q2])
        m.z9 = Var(domain=RangeSet(0, 5, 0), bounds=[m.q1, m.p1])
        m.z10 = Var(domain=RangeSet(0, 5, 0), bounds=[m.q1, m.p2])

        model_data.working_model.uncertain_params = [m.q1, m.q2]
        model_data.working_model.effective_performance_equality_cons = []
        model_data.working_model.effective_performance_inequality_cons = []

        return model_data

    def test_turn_nonadjustable_bounds_to_constraints(self):
        """
        Test subroutine for reformulating bounds on nonadjustable
        variables to constraints.

        This subroutine should reformulate only the uncertain
        declared bounds for the nonadjustable variables.
        All other variable bounds should be left unchanged.
        All variable domains should remain unchanged.
        """
        model_data = self.build_simple_test_model_data()

        working_model = model_data.working_model
        m = model_data.working_model.user_model

        # mock effective partitioning for testing
        ep = model_data.working_model.effective_var_partitioning = Bunch()
        ep.first_stage_variables = [
            m.z1, m.z2, m.z3, m.z4, m.z5, m.z6, m.z7, m.z8
        ]
        ep.second_stage_variables = [m.z9]
        ep.state_variables = [m.z10]
        effective_first_stage_var_set = ComponentSet(ep.first_stage_variables)

        original_var_domains_and_bounds = ComponentMap(
            (var, (var.domain, get_var_bound_pairs(var)[1]))
            for var in model_data.working_model.user_model.component_data_objects(Var)
        )

        # expected final bounds and bound constraint types
        expected_final_nonadj_var_bounds = ComponentMap((
            (m.z1, (get_var_bound_pairs(m.z1)[1], [])),
            (m.z2, (get_var_bound_pairs(m.z2)[1], [])),
            (m.z3, (get_var_bound_pairs(m.z3)[1], [])),
            (m.z4, ((None, 0), ["lower"])),
            (m.z5, ((4, None), ["upper"])),
            (m.z6, ((None, None), ["eq"])),
            (m.z7, ((None, None), ["eq"])),
            (m.z8, ((None, None), ["lower", "upper"])),
        ))

        turn_nonadjustable_var_bounds_to_constraints(model_data)

        for var, (orig_domain, orig_bounds) in original_var_domains_and_bounds.items():
            # all var domains should remain unchanged
            self.assertIs(
                var.domain,
                orig_domain,
                msg=(
                    f"Domain of variable {var.name!r} was changed from "
                    f"{orig_domain} to {var.domain} by "
                    f"{turn_nonadjustable_var_bounds_to_constraints.__name__!r}. "
                ),
            )
            _, (final_lb, final_ub) = get_var_bound_pairs(var)

            if var not in effective_first_stage_var_set:
                # these are the adjustable variables.
                # bounds should not have been changed
                self.assertIs(
                    orig_bounds[0],
                    final_lb,
                    msg=(
                        f"Lower bound for adjustable variable {var.name!r} appears to "
                        f"have been changed from {orig_bounds[0]} to {final_lb}."
                    ),
                )
                self.assertIs(
                    orig_bounds[1],
                    final_ub,
                    msg=(
                        f"Upper bound for adjustable variable {var.name!r} appears to "
                        f"have been changed from {orig_bounds[1]} to {final_ub}."
                    ),
                )
            else:
                # these are the nonadjustable variables.
                # only the uncertain bounds should have been
                # changed, and accompanying constraints added

                expected_bounds, con_bound_types = expected_final_nonadj_var_bounds[var]
                expected_lb, expected_ub = expected_bounds

                self.assertIs(
                    expected_lb,
                    final_lb,
                    msg=(
                        f"Lower bound for nonadjustable variable {var.name!r} "
                        f"should be {expected_lb}, but was "
                        f"found to be {final_lb}."
                    ),
                )
                self.assertIs(
                    expected_ub,
                    final_ub,
                    msg=(
                        f"Upper bound for nonadjustable variable {var.name!r} "
                        f"should be {expected_ub}, but was "
                        f"found to be {final_ub}."
                    ),
                )

                for cbtype in con_bound_types:
                    # verify the bound constraints were added
                    # and are as expected
                    varname = var.getname(
                        relative_to=m, fully_qualified=True
                    )
                    bound_con = model_data.working_model.user_model.find_component(
                        f"var_{varname}_uncertain_{cbtype}_bound_con"
                    )
                    self.assertIsNotNone(
                        bound_con,
                        msg=f"Bound constraint for variable {var.name!r} not found."
                    )
                    if cbtype == "eq":
                        self.assertIn(
                            bound_con,
                            working_model.effective_performance_equality_cons,
                            msg=(
                                "Bound constraint "
                                f"{bound_con.name!r} "
                                "not in first-stage equality constraint set."
                            ),
                        )
                    else:
                        self.assertIn(
                            bound_con,
                            working_model.effective_performance_inequality_cons,
                            msg=(
                                "Bound constraint "
                                f"{bound_con.name!r} "
                                "not in first-stage inequality constraint set."
                            ),
                        )

        # verify bound constraint expressions
        assertExpressionsEqual(
            self,
            m.var_z4_uncertain_lower_bound_con.expr, -m.z4 <= -m.q1
        )
        assertExpressionsEqual(
            self,
            m.var_z5_uncertain_upper_bound_con.expr, m.z5 <= m.q2
        )
        assertExpressionsEqual(
            self, m.var_z6_uncertain_eq_bound_con.expr, m.z6 == m.q1
        )
        assertExpressionsEqual(
            self, m.var_z7_uncertain_eq_bound_con.expr, m.z7 == m.q1
        )
        assertExpressionsEqual(
            self, m.var_z8_uncertain_lower_bound_con.expr, -m.z8 <= -m.q1,
        )
        assertExpressionsEqual(
            self, m.var_z8_uncertain_upper_bound_con.expr, m.z8 <= m.q2,
        )

        # check constraint partitioning
        self.assertEqual(
            len(working_model.effective_performance_inequality_cons),
            4,
            msg="Number of performance inequalities not as expected.,"
        )
        self.assertEqual(
            len(working_model.effective_performance_equality_cons),
            2,
            msg="Number of performance equalities not as expected.,"
        )

    def test_turn_adjustable_bounds_to_constraints(self):
        """
        Test subroutine for reformulating domains and bounds
        on adjustable variables to constraints.

        This subroutine should reformulate the domain and
        declared bounds for every adjustable
        (i.e. effective second-stage and effective state)
        variable.
        The domains and bounds for all other variables
        should be left unchanged.
        """
        model_data = self.build_simple_test_model_data()

        m = model_data.working_model.user_model
        uncertain_params_set = ComponentSet(model_data.working_model.uncertain_params)

        # simple mock partitioning for the test
        ep = model_data.working_model.effective_var_partitioning = Bunch()
        ep.first_stage_variables = [m.z9, m.z10]
        ep.second_stage_variables = [m.z1, m.z2, m.z3, m.z4, m.z5, m.z6]
        ep.state_variables = [m.z7, m.z8]
        effective_first_stage_var_set = ComponentSet(ep.first_stage_variables)

        original_var_domains_and_bounds = ComponentMap(
            (var, (var.domain, get_var_bound_pairs(var)[1]))
            for var in model_data.working_model.user_model.component_data_objects(Var)
        )

        # for checking the correct bound constraints were
        # added.
        # - first list: types of certain bound constraints
        #               that should have been added
        # - second list: types of uncertain bound constraints
        #               that should have been added
        expected_cert_uncert_bound_con_types = ComponentMap((
            (m.z1, ([], [])),
            (m.z2, (["eq"], [])),
            (m.z3, (["lower", "upper"], [])),
            (m.z4, (["eq"], ["lower"])),
            (m.z5, (["eq"], ["upper"])),
            (m.z6, (["lower"], ["eq"])),
            (m.z7, (["upper"], ["eq"])),
            (m.z8, (["lower", "upper"], ["lower", "upper"])),
        ))

        turn_adjustable_var_bounds_to_constraints(model_data)
        for var, (orig_domain, orig_bounds) in original_var_domains_and_bounds.items():
            _, (final_lb, final_ub) = get_var_bound_pairs(var)
            if var not in effective_first_stage_var_set:
                # these are the adjustable variables.
                # domains should have been removed,
                # i.e. changed to reals.
                # bounds should also have been removed
                self.assertIs(
                    var.domain,
                    Reals,
                    msg=(
                        f"Domain of adjustable variable {var.name!r}  "
                        "should now be Reals, but was instead found to be "
                        f"{var.domain}"
                    ),
                )
                self.assertIsNone(
                    final_lb,
                    msg=(
                        f"Declared lower bound for adjustable variable {var.name!r} "
                        "should now be None, as all adjustable variable bounds "
                        "should have been removed, but was instead found to be"
                        f"{final_lb}."
                    ),
                )
                self.assertIsNone(
                    final_ub,
                    msg=(
                        f"Declared upper bound for adjustable variable {var.name!r} "
                        "should now be None, as all adjustable variable bounds "
                        "should have been removed, but was instead found to be"
                        f"{final_ub}."
                    ),
                )

                # check the constraints added are as expected:
                # they are present, involve only the variable
                # of interest, and where applicable, the
                # uncertain parameters
                varname = var.getname(
                    relative_to=m, fully_qualified=True
                )
                cert_bound_con_types, uncert_bound_con_types = (
                    expected_cert_uncert_bound_con_types[var]
                )
                for ccbtype in cert_bound_con_types:
                    cert_bound_con_name = f"var_{varname}_certain_{ccbtype}_bound_con"

                    cert_bound_con = model_data.working_model.user_model.find_component(
                        cert_bound_con_name
                    )
                    self.assertIsNotNone(
                        cert_bound_con,
                        msg=(
                            f"Expected working model to contain a certain {ccbtype} "
                            f"bound constraint with name {cert_bound_con_name!r}, "
                            f"for the variable {var.name!r}, "
                            "but no such constraint was not found."
                        )
                    )
                    vars_in_bound_con = ComponentSet(
                        identify_variables(cert_bound_con.body)
                    )
                    self.assertEqual(
                        vars_in_bound_con,
                        ComponentSet((var,)),
                        msg=(
                            f"Bound constraint {cert_bound_con.name} should involve "
                            f"only the variable with name {var.name!r}, but involves "
                            f"the variables {vars_in_bound_con}."
                        ),
                    )

                    uncertain_params_in_bound_con = ComponentSet(
                        identify_mutable_parameters(cert_bound_con.body)
                        & uncertain_params_set
                    )
                    self.assertFalse(
                        uncertain_params_in_bound_con,
                        msg=(
                            f"Uncertain parameters were found in the expression "
                            "of the bound constraint with name"
                            f"{cert_bound_con.name!r}; expression is "
                            f"{cert_bound_con.expr}"
                        ),
                    )

                for ucbtype in uncert_bound_con_types:
                    unc_bound_con_name = f"var_{varname}_uncertain_{ucbtype}_bound_con"
                    unc_bound_con = model_data.working_model.user_model.find_component(
                        unc_bound_con_name
                    )

                    self.assertIsNotNone(
                        unc_bound_con,
                        msg=(
                            f"Expected working model to contain an uncertain {ucbtype} "
                            f"bound constraint with name {unc_bound_con_name!r}, "
                            f"for the variable {var.name!r}, "
                            "but no such constraint was not found."
                        ),
                    )

                    vars_in_bound_con = ComponentSet(
                        identify_variables(unc_bound_con.body)
                    )
                    self.assertEqual(
                        vars_in_bound_con,
                        ComponentSet((var,)),
                        msg=(
                            f"Bound constraint {unc_bound_con.name} should involve "
                            f"only the variable with name {var.name!r}, but involves "
                            f"the variables {vars_in_bound_con}."
                        ),
                    )

                    # we want to ensure that uncertain params,
                    # rather than their values,
                    # are used to create the bound constraints
                    uncertain_params_in_bound_con = ComponentSet(
                        identify_mutable_parameters(unc_bound_con.expr)
                        & uncertain_params_set
                    )
                    self.assertTrue(
                        uncertain_params_in_bound_con,
                        msg=(
                            f"No uncertain parameters were found in the bound "
                            f"constraint with name {unc_bound_con.name!r}."
                        ),
                    )
            else:
                # these are the nonadjustable variables.
                # domains and bounds should be left unchanged
                self.assertIs(
                    var.domain,
                    orig_domain,
                    msg=(
                        f"Domain of adjustable variable {var.name!r}  "
                        "should now be Reals, but was instead found to be "
                        f"{var.domain}"
                    ),
                )
                self.assertIs(
                    orig_bounds[0],
                    final_lb,
                    msg=(
                        f"Lower bound for nonadjustable variable {var.name!r} "
                        "appears to "
                        f"have been changed from {orig_bounds[0]} to {final_lb}."
                    ),
                )
                self.assertIs(
                    orig_bounds[1],
                    final_ub,
                    msg=(
                        f"Upper bound for nonadjustable variable {var.name!r} "
                        "appears to "
                        f"have been changed from {orig_bounds[1]} to {final_ub}."
                    ),
                )

        # verify bound constraint expressions
        assertExpressionsEqual(
            self,
            m.var_z2_certain_eq_bound_con.expr,
            m.z2 == 1,
        )
        assertExpressionsEqual(
            self,
            m.var_z3_certain_lower_bound_con.expr,
            -m.z3 <= -2,
        )
        assertExpressionsEqual(
            self,
            m.var_z3_certain_upper_bound_con.expr,
            m.z3 <= m.p1,
        )
        assertExpressionsEqual(
            self,
            m.var_z4_certain_eq_bound_con.expr,
            m.z4 == 0,
        )
        assertExpressionsEqual(
            self,
            m.var_z4_uncertain_lower_bound_con.expr,
            - m.z4 <= -m.q1,
        )
        assertExpressionsEqual(
            self,
            m.var_z5_certain_eq_bound_con.expr,
            m.z5 == 4,
        )
        assertExpressionsEqual(
            self,
            m.var_z5_uncertain_upper_bound_con.expr,
            m.z5 <= m.q2,
        )
        assertExpressionsEqual(
            self,
            m.var_z6_certain_lower_bound_con.expr,
            -m.z6 <= 0,
        )
        assertExpressionsEqual(
            self,
            m.var_z6_uncertain_eq_bound_con.expr,
            m.z6 == m.q1,
        )
        assertExpressionsEqual(
            self,
            m.var_z7_certain_upper_bound_con.expr,
            m.z7 <= 0,
        )
        assertExpressionsEqual(
            self,
            m.var_z7_uncertain_eq_bound_con.expr,
            m.z7 == m.q1,
        )
        assertExpressionsEqual(
            self,
            m.var_z8_certain_lower_bound_con.expr,
            -m.z8 <= 0,
        )
        assertExpressionsEqual(
            self,
            m.var_z8_certain_upper_bound_con.expr,
            m.z8 <= 5,
        )
        assertExpressionsEqual(
            self,
            m.var_z8_uncertain_lower_bound_con.expr,
            - m.z8 <= -m.q1,
        )
        assertExpressionsEqual(
            self,
            m.var_z8_uncertain_upper_bound_con.expr,
            m.z8 <= m.q2,
        )

        working_model = model_data.working_model
        self.assertEqual(
            len(working_model.effective_performance_inequality_cons),
            10,
            msg="Number of performance inequalty constraints not as expected.",
        )
        self.assertEqual(
            len(working_model.effective_performance_equality_cons),
            5,
            msg="Number of performance equalty constraints not as expected.",
        )


class TestStandardizeInequalityConstraints(unittest.TestCase):
    """
    Test standardization of inequality constraints.
    """

    def build_simple_test_model_data(self):
        """
        Build model data object for testing constraint standardization
        routines.
        """
        model_data = Bunch()
        model_data.working_model = ConcreteModel()
        model_data.working_model.user_model = m = Block()

        m.x1 = Var()
        m.x2 = Var()
        m.z1 = Var()
        m.z2 = Var()
        m.y1 = Var()

        m.p = Param(initialize=2, mutable=True)
        m.q = Param(mutable=True, initialize=1)

        m.c1 = Constraint(expr=m.x1 <= 1)
        m.c2 = Constraint(expr=(1, m.x1, 2))
        m.c3 = Constraint(expr=m.q <= m.x1)
        m.c4 = Constraint(expr=(log(m.p), m.x2, m.q))
        m.c5 = Constraint(expr=(m.q, m.x2, 2 * m.q))
        m.c6 = Constraint(expr=m.z1 <= 1)
        m.c7 = Constraint(expr=(0, m.z2, 1))
        m.c8 = Constraint(expr=(m.p ** 0.5, m.y1, m.p))
        m.c9 = Constraint(expr=m.y1 - m.q <= 0)
        m.c10 = Constraint(expr=m.y1 <= m.q ** 2)
        m.c11 = Constraint(expr=m.z2 <= m.q)
        m.c12 = Constraint(expr=(m.q ** 2, m.x1, sin(m.p)))

        m.c11.deactivate()

        model_data.working_model.uncertain_params = [m.q]

        model_data.working_model.effective_first_stage_inequality_cons = []
        model_data.working_model.effective_performance_inequality_cons = []

        model_data.working_model.original_active_inequality_cons = [
            m.c1, m.c2, m.c3, m.c4, m.c5, m.c6, m.c7, m.c8, m.c9, m.c10, m.c12,
        ]

        ep = model_data.working_model.effective_var_partitioning = Bunch()
        ep.first_stage_variables = [m.x1, m.x2]
        ep.second_stage_variables = [m.z1, m.z2]
        ep.state_variables = [m.y1]

        return model_data

    def test_standardize_inequality_constraints(self):
        """
        Test inequality constraint standardization routine.
        """
        model_data = self.build_simple_test_model_data()
        working_model = model_data.working_model
        m = working_model.user_model

        standardize_inequality_constraints(model_data)
        m = working_model.user_model

        self.assertEqual(len(working_model.effective_first_stage_inequality_cons), 4)
        self.assertEqual(len(working_model.effective_performance_inequality_cons), 12)

        self.assertTrue(m.c1.active)
        self.assertIn(m.c1, working_model.effective_first_stage_inequality_cons)
        assertExpressionsEqual(self, m.c1.expr, m.x1 <= 1)

        # 1 <= m.x1 <= 2; first-stage constraint. no modification
        self.assertTrue(m.c2.active)
        self.assertIn(m.c2, working_model.effective_first_stage_inequality_cons)
        assertExpressionsEqual(
            self, m.c2.expr, RangedExpression((1, m.x1, 2), False)
        )

        # m.q <= m.x1; single performance inequality. modify in place
        self.assertTrue(m.c3.active)
        self.assertIn(m.c3, working_model.effective_performance_inequality_cons)
        assertExpressionsEqual(self, m.c3.expr, - m.x1 <= -m.q)

        # log(m.p) <= m.x2 <= m.q
        # log(m.p) <= m.x2 stays in place as first-stage inequality
        # m.x2 - m.q <= 0 added as performance inequality
        self.assertTrue(m.c4.active)
        c4_upper_bound_con = m.find_component("con_c4_upper_bound_con")
        self.assertIn(m.c4, working_model.effective_first_stage_inequality_cons)
        self.assertIn(
            c4_upper_bound_con,
            working_model.effective_performance_inequality_cons,
        )
        assertExpressionsEqual(self, m.c4.expr, m.c4.expr, log(m.p) <= m.x2)
        assertExpressionsEqual(self, c4_upper_bound_con.expr, m.x2 <= m.q)

        # m.q <= m.x2 <= 2 * m.q
        # two constraints, one for each bound. deactivate the original
        self.assertFalse(m.c5.active)
        c5_lower_bound_con = m.find_component("con_c5_lower_bound_con")
        c5_upper_bound_con = m.find_component("con_c5_upper_bound_con")
        self.assertIn(
            c5_lower_bound_con,
            working_model.effective_performance_inequality_cons,
        )
        self.assertIn(
            c5_upper_bound_con,
            working_model.effective_performance_inequality_cons,
        )
        assertExpressionsEqual(self, c5_lower_bound_con.expr, - m.x2 <= -m.q)
        assertExpressionsEqual(self, c5_upper_bound_con.expr, m.x2 <= 2 * m.q)

        # single performance inequality m.z1 - 1.0 <= 0
        self.assertTrue(m.c6.active)
        self.assertIn(m.c6, working_model.effective_performance_inequality_cons)
        assertExpressionsEqual(self, m.c6.expr, m.z1 <= 1.0)

        # two new performance inequalities:
        # 0 - m.z2 <= 0 and m.z2 - 1 <= 0
        # the original should be deactivated
        self.assertFalse(m.c7.active)
        c7_lower_bound_con = m.find_component("con_c7_lower_bound_con")
        c7_upper_bound_con = m.find_component("con_c7_upper_bound_con")
        self.assertIn(
            c7_lower_bound_con, working_model.effective_performance_inequality_cons,
        )
        self.assertIn(
            c7_upper_bound_con, working_model.effective_performance_inequality_cons,
        )
        assertExpressionsEqual(self, c7_lower_bound_con.expr, -m.z2 <= 0.0)
        assertExpressionsEqual(self, c7_upper_bound_con.expr, m.z2 <= 1.0)

        # m.p ** 0.5 <= m.y1 <= m.p
        # two performance inequalities; deactivate the original
        self.assertFalse(m.c8.active)
        c8_lower_bound_con = m.find_component("con_c8_lower_bound_con")
        c8_upper_bound_con = m.find_component("con_c8_upper_bound_con")
        self.assertIn(
            c8_lower_bound_con, working_model.effective_performance_inequality_cons,
        )
        self.assertIn(
            c8_upper_bound_con, working_model.effective_performance_inequality_cons,
        )
        assertExpressionsEqual(self, c8_lower_bound_con.expr, - m.y1 <= -m.p ** 0.5)
        assertExpressionsEqual(self, c8_upper_bound_con.expr, m.y1 <= m.p)

        # m.y1 - m.q <= 0
        # single performance inequality
        self.assertTrue(m.c9.active)
        self.assertIn(m.c9, working_model.effective_performance_inequality_cons)
        assertExpressionsEqual(self, m.c9.expr, m.y1 - m.q <= 0.0)

        # m.y1 <= m.q ** 2
        # single performance inequality
        self.assertTrue(m.c10.active)
        self.assertIn(m.c10, working_model.effective_performance_inequality_cons)
        assertExpressionsEqual(self, m.c10.expr, m.y1 <= m.q ** 2)

        # originally deactivated;
        # no modification
        self.assertFalse(m.c11.active)
        assertExpressionsEqual(self, m.c11.expr, m.z2 <= m.q)

        # lower bound performance; upper bound first-stage
        self.assertTrue(m.c12.active)
        c12_lower_bound_con = m.find_component("con_c12_lower_bound_con")
        self.assertIn(
            c12_lower_bound_con, working_model.effective_performance_inequality_cons
        )
        self.assertIn(m.c12, working_model.effective_first_stage_inequality_cons)
        assertExpressionsEqual(self, m.c12.expr, m.x1 <= sin(m.p))
        assertExpressionsEqual(self, c12_lower_bound_con.expr, - m.x1 <= -m.q ** 2)

    def test_standardize_inequality_error(self):
        """
        Test exception raised by inequality constraint standardization
        method if equality-type expression detected.
        """
        model_data = self.build_simple_test_model_data()
        working_model = model_data.working_model
        m = working_model.user_model

        # change to equality constraint to trigger the exception
        m.c6.set_value(m.z1 == 1)

        exc_str = r"Found an equality bound.*1.0.*for the constraint.*c6'"
        with self.assertRaisesRegex(ValueError, exc_str):
            standardize_inequality_constraints(model_data)


class TestStandardizeEqualityConstraints(unittest.TestCase):
    """
    Test standardization of equality constraints.
    """

    def build_simple_test_model_data(self):
        """
        Build model data object for testing constraint standardization
        routines.
        """
        model_data = Bunch()
        model_data.working_model = ConcreteModel()
        model_data.working_model.user_model = m = Block()

        m.x1 = Var()
        m.x2 = Var()
        m.z1 = Var()
        m.z2 = Var()
        m.y1 = Var()

        m.p = Param(initialize=2, mutable=True)
        m.q = Param(mutable=True, initialize=1)

        # first-stage equalities
        m.eq1 = Constraint(expr=m.x1 + log(m.p) == 1)
        m.eq2 = Constraint(expr=(1, m.x2, 1))

        # performance equalities
        m.eq3 = Constraint(expr=m.x2 * m.q == 1)
        m.eq4 = Constraint(expr=m.x2 - m.z1 ** 2 == 0)
        m.eq5 = Constraint(expr=m.q == m.y1)
        m.eq6 = Constraint(expr=(m.q, m.y1, m.q))
        m.eq7 = Constraint(expr=m.z2 == 0)

        m.eq7.deactivate()

        model_data.working_model.uncertain_params = [m.q]

        model_data.working_model.effective_first_stage_equality_cons = []
        model_data.working_model.effective_performance_equality_cons = []

        model_data.working_model.original_active_equality_cons = [
            m.eq1, m.eq2, m.eq3, m.eq4, m.eq5, m.eq6,
        ]

        ep = model_data.working_model.effective_var_partitioning = Bunch()
        ep.first_stage_variables = [m.x1, m.x2]
        ep.second_stage_variables = [m.z1, m.z2]
        ep.state_variables = [m.y1]

        return model_data

    def test_standardize_equality_constraints(self):
        """
        Test inequality constraint standardization routine.
        """
        model_data = self.build_simple_test_model_data()
        working_model = model_data.working_model
        m = working_model.user_model

        standardize_equality_constraints(model_data)

        self.assertEqual(
            ComponentSet(working_model.effective_first_stage_equality_cons),
            ComponentSet([m.eq1, m.eq2]),
        )
        self.assertEqual(
            ComponentSet(working_model.effective_performance_equality_cons),
            ComponentSet([m.eq3, m.eq4, m.eq5, m.eq6]),
        )

        # should be first-stage
        self.assertTrue(m.eq1.active)
        assertExpressionsEqual(self, m.eq1.expr, m.x1 + log(m.p) == 1)

        self.assertTrue(m.eq2.active)
        assertExpressionsEqual(self, m.eq2.expr, RangedExpression((1, m.x2, 1), False))

        self.assertTrue(m.eq3.active)
        assertExpressionsEqual(self, m.eq3.expr, m.x2 * m.q == 1)

        self.assertTrue(m.eq4.active)
        assertExpressionsEqual(self, m.eq4.expr, m.x2 - m.z1 ** 2 == 0)

        self.assertTrue(m.eq5.active)
        assertExpressionsEqual(self, m.eq5.expr, m.q == m.y1)

        self.assertTrue(m.eq6.active)
        assertExpressionsEqual(
            self, m.eq6.expr, RangedExpression((m.q, m.y1, m.q), False),
        )

        # excluded from the list of active constraints;
        # state should remain unchanged
        self.assertFalse(m.eq7.active)
        assertExpressionsEqual(self, m.eq7.expr, m.z2 == 0)


class TestStandardizeActiveObjective(unittest.TestCase):
    """
    Test methods for standardization of the active objective.
    """

    def build_simple_test_model_data(self):
        """
        Build simple model for testing active objective
        standardization.
        """
        model_data = Bunch()
        model_data.working_model = ConcreteModel()
        model_data.working_model.user_model = m = Block()

        m.x = Var(initialize=1)
        m.z = Var(initialize=2)
        m.y = Var()

        m.p = Param(initialize=1, mutable=True)
        m.q = Param(initialize=1, mutable=True)

        m.obj1 = Objective(
            expr=(
                10 + m.p + m.q + m.p * m.x + m.z * m.p + m.y ** 2 * m.q + m.y + log(m.x)
            ),
        )
        m.obj2 = Objective(
            expr=m.p + m.x * m.z + m.z ** 2,
        )

        model_data.working_model.uncertain_params = [m.q]

        up = model_data.working_model.user_var_partitioning = Bunch()
        up.first_stage_variables = [m.x]
        up.second_stage_variables = [m.z]
        up.state_variables = [m.y]

        ep = model_data.working_model.effective_var_partitioning = Bunch()
        ep.first_stage_variables = [m.x, m.z]
        ep.second_stage_variables = []
        ep.state_variables = [m.y]

        model_data.working_model.effective_first_stage_inequality_cons = []
        model_data.working_model.effective_performance_inequality_cons = []

        return model_data

    def test_declare_objective_expressions(self):
        """
        Test method for identification/declaration
        of per-stage objective summands.
        """
        model_data = self.build_simple_test_model_data()
        working_model = model_data.working_model
        m = model_data.working_model.user_model

        declare_objective_expressions(working_model, m.obj1)
        assertExpressionsEqual(
            self,
            working_model.first_stage_objective.expr,
            10 + m.p + m.p * m.x + log(m.x),
        )
        assertExpressionsEqual(
            self,
            working_model.second_stage_objective.expr,
            m.q + m.z * m.p + m.y ** 2 * m.q + m.y,
        )
        assertExpressionsEqual(
            self,
            working_model.full_objective.expr,
            m.obj1.expr,
        )

    def test_declare_objective_expressions_maximization_obj(self):
        """
        Test per-stage objective summand expressions are constructed
        as expected when the objective is of a maximization sense.
        """
        model_data = self.build_simple_test_model_data()
        working_model = model_data.working_model
        m = model_data.working_model.user_model
        m.obj1.sense = maximize

        declare_objective_expressions(working_model, m.obj1)
        assertExpressionsEqual(
            self,
            working_model.first_stage_objective.expr,
            -10 - m.p - m.p * m.x - log(m.x),
        )
        assertExpressionsEqual(
            self,
            working_model.second_stage_objective.expr,
            -m.q - m.z * m.p - m.y ** 2 * m.q - m.y,
        )
        assertExpressionsEqual(
            self,
            working_model.full_objective.expr,
            -m.obj1.expr,
        )

    def test_standardize_active_obj_worst_case_focus(self):
        """
        Test preprocesing step for standardization
        of the active model objective.
        """
        model_data = self.build_simple_test_model_data()
        working_model = model_data.working_model
        m = model_data.working_model.user_model
        config = Bunch(objective_focus=ObjectiveType.worst_case)

        m.obj1.activate()
        m.obj2.deactivate()

        standardize_active_objective(model_data, config)

        self.assertFalse(
            m.obj1.active,
            msg=(
                f"Objective {m.obj1.name!r} should have been deactivated by "
                f"{standardize_active_objective}."
            ),
        )
        self.assertNotIn(
            working_model.epigraph_con,
            working_model.effective_first_stage_inequality_cons,
            msg=(
                f"Epigraph constraint {working_model.epigraph_con.name!r} "
                "should not be in the list of effective first-stage inequalities."
            ),
        )
        self.assertIn(
            working_model.epigraph_con,
            working_model.effective_performance_inequality_cons,
            msg=(
                f"Epigraph constraint {working_model.epigraph_con.name!r} "
                "should be in the list of effective performance inequalities."
            ),
        )
        assertExpressionsEqual(
            self,
            working_model.epigraph_con.expr,
            m.obj1.expr - working_model.epigraph_var <= 0,
        )

    def test_standardize_active_obj_nominal_focus(self):
        """
        Test standardization of active objective under nominal
        objective focus.
        """
        model_data = self.build_simple_test_model_data()
        working_model = model_data.working_model
        m = model_data.working_model.user_model
        config = Bunch(objective_focus=ObjectiveType.nominal)

        m.obj1.activate()
        m.obj2.deactivate()

        standardize_active_objective(model_data, config)

        self.assertFalse(
            m.obj1.active,
            msg=(
                f"Objective {m.obj1.name!r} should have been deactivated by "
                f"{standardize_active_objective}."
            ),
        )
        self.assertIn(
            working_model.epigraph_con,
            working_model.effective_first_stage_inequality_cons,
            msg=(
                f"Epigraph constraint {working_model.epigraph_con.name!r} "
                "should be in the list of effective first-stage inequalities."
            ),
        )
        self.assertNotIn(
            working_model.epigraph_con,
            working_model.effective_performance_inequality_cons,
            msg=(
                f"Epigraph constraint {working_model.epigraph_con.name!r} "
                "should not be in the list of effective performance inequalities."
            ),
        )
        assertExpressionsEqual(
            self,
            working_model.epigraph_con.expr,
            m.obj1.expr - working_model.epigraph_var <= 0,
        )

    def test_standardize_active_obj_unsupported_focus(self):
        """
        Test standardization of active objective under
        an objective focus currently not supported
        """
        model_data = self.build_simple_test_model_data()
        m = model_data.working_model.user_model
        config = Bunch(objective_focus="bad_focus")

        m.obj1.activate()
        m.obj2.deactivate()

        exc_str = r"Classification.*not implemented for objective focus 'bad_focus'"
        with self.assertRaisesRegex(ValueError, exc_str):
            standardize_active_objective(model_data, config)

    def test_standardize_active_obj_nonadjustable_max(self):
        """
        Test standardize active objective for case in which
        the objective is independent of the nonadjustable variables
        and of a maximization sense.
        """
        model_data = self.build_simple_test_model_data()
        working_model = model_data.working_model
        m = working_model.user_model
        config = Bunch(objective_focus=ObjectiveType.worst_case)

        # assume all variables nonadjustable
        ep = model_data.working_model.effective_var_partitioning
        ep.first_stage_variables = [m.x, m.z]
        ep.second_stage_variables = []
        ep.state_variables = [m.y]

        m.obj1.deactivate()
        m.obj2.activate()
        m.obj2.sense = maximize

        standardize_active_objective(model_data, config)

        self.assertFalse(
            m.obj2.active,
            msg=(
                f"Objective {m.obj2.name!r} should have been deactivated by "
                f"{standardize_active_objective}."
            ),
        )
        self.assertIn(
            working_model.epigraph_con,
            working_model.effective_first_stage_inequality_cons,
            msg=(
                f"Epigraph constraint {working_model.epigraph_con.name!r} "
                "should be in the list of effective first-stage inequalities."
            ),
        )
        self.assertNotIn(
            working_model.epigraph_con,
            working_model.effective_performance_inequality_cons,
            msg=(
                f"Epigraph constraint {working_model.epigraph_con.name!r} "
                "should not be in the list of effective performance inequalities."
            ),
        )

        assertExpressionsEqual(
            self,
            working_model.epigraph_con.expr,
            -m.obj2.expr - working_model.epigraph_var <= 0,
        )


class TestAddDecisionRuleVars(unittest.TestCase):
    """
    Test method for adding decision rule variables to working model.
    There should be one indexed decision rule variable for every
    effective second-stage variable.
    The number of decision rule variables per effective second-stage
    variable should depend on:

    - the number of uncertain parameters in the model
    - the decision rule order specified by the user.
    """

    def build_simple_test_model_data(self):
        """
        Make simple model data object for DR variable
        declaration testing.
        """
        model_data = Bunch()
        model_data.working_model = ConcreteModel()
        model_data.working_model.user_model = m = Block()

        # uncertain parameters
        m.q = Param(range(3), initialize=0, mutable=True)

        # second-stage variables
        m.x = Var()
        m.z1 = Var([0, 1], initialize=0)
        m.z2 = Var()
        m.y = Var()

        model_data.working_model.uncertain_params = list(m.q.values())

        up = model_data.working_model.user_var_partitioning = Bunch()
        up.first_stage_variables = [m.x]
        up.second_stage_variables = [m.z1, m.z2]
        up.state_variables = [m.y]

        ep = model_data.working_model.effective_var_partitioning = Bunch()
        ep.first_stage_variables = [m.x, m.z1]
        ep.second_stage_variables = [m.z2]
        ep.state_variables = [m.y]

        return model_data

    def test_correct_num_dr_vars_static(self):
        """
        Test DR variable setup routines declare the correct
        number of DR coefficient variables, static DR case.
        """
        model_data = self.build_simple_test_model_data()

        config = Bunch()
        config.decision_rule_order = 0

        add_decision_rule_variables(model_data=model_data, config=config)

        for indexed_dr_var in model_data.working_model.decision_rule_vars:
            self.assertEqual(
                len(indexed_dr_var),
                1,
                msg=(
                    "Number of decision rule coefficient variables "
                    f"in indexed Var object {indexed_dr_var.name!r}"
                    "does not match correct value."
                ),
            )

        effective_second_stage_vars = (
            model_data.working_model.effective_var_partitioning.second_stage_variables
        )
        self.assertEqual(
            len(ComponentSet(model_data.working_model.decision_rule_vars)),
            len(effective_second_stage_vars),
            msg=(
                "Number of unique indexed DR variable components should equal "
                "number of second-stage variables."
            ),
        )

        # check mapping is as expected
        ess_dr_var_zip = zip(
            effective_second_stage_vars,
            model_data.working_model.decision_rule_vars,
        )
        for ess_var, indexed_dr_var in ess_dr_var_zip:
            mapped_dr_var = model_data.working_model.eff_ss_var_to_dr_var_map[ess_var]
            self.assertIs(
                mapped_dr_var,
                indexed_dr_var,
                msg=(
                    f"Second-stage var {ess_var.name!r} "
                    f"is mapped to DR var {mapped_dr_var.name!r}, "
                    f"but expected mapping to DR var {indexed_dr_var.name!r}."
                )
            )

    def test_correct_num_dr_vars_affine(self):
        """
        Test DR variable setup routines declare the correct
        number of DR coefficient variables, affine DR case.
        """
        model_data = self.build_simple_test_model_data()

        config = Bunch()
        config.decision_rule_order = 1

        add_decision_rule_variables(model_data=model_data, config=config)

        for indexed_dr_var in model_data.working_model.decision_rule_vars:
            self.assertEqual(
                len(indexed_dr_var),
                1 + len(model_data.working_model.uncertain_params),
                msg=(
                    "Number of decision rule coefficient variables "
                    f"in indexed Var object {indexed_dr_var.name!r}"
                    "does not match correct value."
                ),
            )

        effective_second_stage_vars = (
            model_data.working_model.effective_var_partitioning.second_stage_variables
        )
        self.assertEqual(
            len(ComponentSet(model_data.working_model.decision_rule_vars)),
            len(effective_second_stage_vars),
            msg=(
                "Number of unique indexed DR variable components should equal "
                "number of second-stage variables."
            ),
        )

        # check mapping is as expected
        ess_dr_var_zip = zip(
            effective_second_stage_vars,
            model_data.working_model.decision_rule_vars,
        )
        for ess_var, indexed_dr_var in ess_dr_var_zip:
            mapped_dr_var = model_data.working_model.eff_ss_var_to_dr_var_map[ess_var]
            self.assertIs(
                mapped_dr_var,
                indexed_dr_var,
                msg=(
                    f"Second-stage var {ess_var.name!r} "
                    f"is mapped to DR var {mapped_dr_var.name!r}, "
                    f"but expected mapping to DR var {indexed_dr_var.name!r}."
                )
            )

    def test_correct_num_dr_vars_quadratic(self):
        """
        Test DR variable setup routines declare the correct
        number of DR coefficient variables, quadratic DR case.
        """
        model_data = self.build_simple_test_model_data()

        config = Bunch()
        config.decision_rule_order = 2

        add_decision_rule_variables(model_data=model_data, config=config)

        num_params = len(model_data.working_model.uncertain_params)

        for indexed_dr_var in model_data.working_model.decision_rule_vars:
            self.assertEqual(
                len(indexed_dr_var),
                1  # static term
                + num_params  # affine terms
                # quadratic terms
                + sp.special.comb(num_params, 2, repetition=True, exact=True),
                msg=(
                    "Number of decision rule coefficient variables "
                    f"in indexed Var object {indexed_dr_var.name!r}"
                    "does not match correct value."
                ),
            )

        effective_second_stage_vars = (
            model_data.working_model.effective_var_partitioning.second_stage_variables
        )
        self.assertEqual(
            len(ComponentSet(model_data.working_model.decision_rule_vars)),
            len(effective_second_stage_vars),
            msg=(
                "Number of unique indexed DR variable components should equal "
                "number of second-stage variables."
            ),
        )

        # check mapping is as expected
        ess_dr_var_zip = zip(
            effective_second_stage_vars,
            model_data.working_model.decision_rule_vars,
        )
        for ess_var, indexed_dr_var in ess_dr_var_zip:
            mapped_dr_var = model_data.working_model.eff_ss_var_to_dr_var_map[ess_var]
            self.assertIs(
                mapped_dr_var,
                indexed_dr_var,
                msg=(
                    f"Second-stage var {ess_var.name!r} "
                    f"is mapped to DR var {mapped_dr_var.name!r}, "
                    f"but expected mapping to DR var {indexed_dr_var.name!r}."
                )
            )


class TestAddDecisionRuleConstraints(unittest.TestCase):
    """
    Test method for adding decision rule equality constraints
    to the working model. There should be as many decision
    rule equality constraints as there are effective second-stage
    variables, and each constraint should relate an effective
    second-stage variable to the uncertain parameters and corresponding
    decision rule variables.
    """

    def build_simple_test_model_data(self):
        """
        Make simple test model for DR variable
        declaration testing.
        """
        model_data = Bunch()
        model_data.working_model = ConcreteModel()
        model_data.working_model.user_model = m = Block()

        # uncertain parameters
        m.q = Param(range(3), initialize=0, mutable=True)

        # second-stage variables
        m.x = Var()
        m.z1 = Var([0, 1], initialize=0)
        m.z2 = Var()
        m.y = Var()

        model_data.working_model.uncertain_params = list(m.q.values())

        up = model_data.working_model.user_var_partitioning = Bunch()
        up.first_stage_variables = [m.x]
        up.second_stage_variables = [m.z1, m.z2]
        up.state_variables = [m.y]

        ep = model_data.working_model.effective_var_partitioning = Bunch()
        ep.first_stage_variables = [m.x, m.z1]
        ep.second_stage_variables = [m.z2]
        ep.state_variables = [m.y]

        return model_data

    def test_num_dr_eqns_added_correct(self):
        """
        Check that number of DR equality constraints added
        by constraint declaration routines matches the number
        of second-stage variables in the model.
        """
        model_data = self.build_simple_test_model_data()

        # set up simple config-like object
        config = Bunch()
        config.decision_rule_order = 0

        add_decision_rule_variables(model_data, config)
        add_decision_rule_constraints(model_data, config)

        effective_second_stage_vars = (
            model_data.working_model.effective_var_partitioning.second_stage_variables
        )
        self.assertEqual(
            len(model_data.working_model.decision_rule_eqns),
            len(effective_second_stage_vars),
            msg=(
                "Number of decision rule equations should match number of "
                "effective second-stage variables."
            )
        )

        # check second-stage var to DR equation mapping is as expected
        ess_dr_var_zip = zip(
            effective_second_stage_vars,
            model_data.working_model.decision_rule_eqns,
        )
        for ess_var, indexed_dr_eqn in ess_dr_var_zip:
            mapped_dr_eqn = model_data.working_model.eff_ss_var_to_dr_eqn_map[ess_var]
            self.assertIs(
                mapped_dr_eqn,
                indexed_dr_eqn,
                msg=(
                    f"Second-stage var {ess_var.name!r} "
                    f"is mapped to DR equation {mapped_dr_eqn.name!r}, "
                    f"but expected mapping to DR equation {indexed_dr_eqn.name!r}."
                )
            )

    def test_dr_eqns_form_correct(self):
        """
        Check that form of decision rule equality constraints
        is as expected.

        Decision rule equations should be of the standard form:
            (sum of DR monomial terms) - (second-stage variable) == 0
        where each monomial term should be of form:
            (product of uncertain parameters) * (decision rule variable)

        This test checks that the equality constraints are of this
        standard form.
        """
        model_data = self.build_simple_test_model_data()
        working_model = model_data.working_model
        m = model_data.working_model.user_model

        # set up simple config-like object
        config = Bunch()
        config.decision_rule_order = 2

        # add DR variables and constraints
        add_decision_rule_variables(model_data, config)
        add_decision_rule_constraints(model_data, config)

        dr_zip = zip(
            model_data.working_model.effective_var_partitioning.second_stage_variables,
            model_data.working_model.decision_rule_vars,
            model_data.working_model.decision_rule_eqns,
        )
        for ss_var, indexed_dr_var, dr_eq in dr_zip:
            expected_dr_eq_expression = (
                indexed_dr_var[0]
                + indexed_dr_var[1] * m.q[0]
                + indexed_dr_var[2] * m.q[1]
                + indexed_dr_var[3] * m.q[2]
                + indexed_dr_var[4] * m.q[0] * m.q[0]
                + indexed_dr_var[5] * m.q[0] * m.q[1]
                + indexed_dr_var[6] * m.q[0] * m.q[2]
                + indexed_dr_var[7] * m.q[1] * m.q[1]
                + indexed_dr_var[8] * m.q[1] * m.q[2]
                + indexed_dr_var[9] * m.q[2] * m.q[2]
                - ss_var
                == 0
            )
            assertExpressionsEqual(self, dr_eq.expr, expected_dr_eq_expression)

            expected_dr_var_to_exponent_map = ComponentMap((
                (indexed_dr_var[0], 0),
                (indexed_dr_var[1], 1),
                (indexed_dr_var[2], 1),
                (indexed_dr_var[3], 1),
                (indexed_dr_var[4], 2),
                (indexed_dr_var[5], 2),
                (indexed_dr_var[6], 2),
                (indexed_dr_var[7], 2),
                (indexed_dr_var[8], 2),
                (indexed_dr_var[9], 2),
            ))
            self.assertEqual(
                working_model.dr_var_to_exponent_map,
                expected_dr_var_to_exponent_map,
                msg="DR variable to exponent map not as expected.",
            )


class TestCoefficientMatching(unittest.TestCase):
    """
    Unit tests for PyROS coefficient matching routine.
    """
    def setup_test_model_data(self):
        """
        Set up simple test model for coefficient matching
        tests.
        """
        model_data = Bunch()
        model_data.working_model = working_model = ConcreteModel()
        model_data.working_model.user_model = m = Block()

        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.u = Param(initialize=1.125, mutable=True)
        m.con = Constraint(expr=m.u ** (0.5) * m.x1 - m.u * m.x2 <= 2)
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)
        m.eq_con = Constraint(
            expr=m.u**2 * (m.x2 - 1)
            + m.u * (m.x1**3 + 0.5)
            - 5 * m.u * m.x1 * m.x2
            == - m.u * (m.x1 + 2)
        )

        # redundant, but makes the tests more rigorous
        # as we want to check that loops in the coefficient
        # matching routine are exited appropriately
        m.eq_con_2 = Constraint(expr=m.u * (m.x2 - 1) == 0)

        working_model.uncertain_params = [m.u]

        working_model.effective_first_stage_equality_cons = []
        working_model.effective_performance_equality_cons = [m.eq_con, m.eq_con_2]

        working_model.all_variables = [m.x1, m.x2]
        ep = working_model.effective_var_partitioning = Bunch()
        ep.first_stage_variables = [m.x1]
        ep.second_stage_variables = [m.x2]
        ep.state_variables = []

        return model_data

    def test_coefficient_matching_correct_constraints_added(self):
        """
        Test coefficient matching adds correct number of constraints
        in event of sucessful use.
        """
        model_data = self.setup_test_model_data()
        m = model_data.working_model.user_model

        # all vars first-stage
        ep = model_data.working_model.effective_var_partitioning
        ep.first_stage_variables = [m.x1, m.x2]
        ep.second_stage_variables = []

        config = Bunch()
        config.decision_rule_order = 1
        config.progress_logger = logger

        model_data.working_model.decision_rule_vars = []
        model_data.working_model.decision_rule_eqns = []
        model_data.working_model.all_nonadjustable_variables = list(
            ep.first_stage_variables
        )

        robust_infeasible = perform_coefficient_matching(model_data, config)

        self.assertFalse(
            robust_infeasible,
            msg=(
                "Coefficient matching unexpectedly detected"
                "a robust infeasible constraint"
            ),
        )
        self.assertEqual(
            len(model_data.working_model.coefficient_matching_conlist),
            3,
            msg="Number of coefficient matching constraints not as expected."
        )

        assertExpressionsEqual(
            self,
            model_data.working_model.coefficient_matching_conlist[1].expr,
            2.5 + m.x1 + (-5) * (m.x1 * m.x2) + m.x1 ** 3 == 0,
        )
        assertExpressionsEqual(
            self,
            model_data.working_model.coefficient_matching_conlist[2].expr,
            (-1) + m.x2 == 0,
        )
        assertExpressionsEqual(
            self,
            model_data.working_model.coefficient_matching_conlist[3].expr,
            (-1) + m.x2 == 0,
        )

        # check constraint partitioning updated as expected
        self.assertEqual(
            model_data.working_model.effective_performance_equality_cons,
            [],
        )
        self.assertEqual(
            model_data.working_model.effective_first_stage_equality_cons,
            list(model_data.working_model.coefficient_matching_conlist.values()),
        )

    def test_coefficient_matching_nonlinear(self):
        """
        Test coefficient matching raises exception in event
        of encountering unsupported nonlinearities.
        """
        model_data = self.setup_test_model_data()

        config = Bunch()
        config.decision_rule_order = 1
        config.progress_logger = logging.getLogger(
            self.test_coefficient_matching_nonlinear.__name__
        )
        config.progress_logger.setLevel(logging.DEBUG)

        add_decision_rule_variables(model_data=model_data, config=config)
        add_decision_rule_constraints(model_data=model_data, config=config)

        ep = model_data.working_model.effective_var_partitioning
        model_data.working_model.all_nonadjustable_variables = list(
            ep.first_stage_variables
            + list(model_data.working_model.decision_rule_var_0.values())
        )

        m = model_data.working_model.user_model

        # we want only one of the constraints to trigger the error
        # change eq_con_2 to give a valid matching constraint
        m.eq_con_2.set_value(m.u * (m.x1 - 1) == 0)

        with LoggingIntercept(level=logging.DEBUG) as LOG:
            robust_infeasible = perform_coefficient_matching(model_data, config)

        err_msg = LOG.getvalue()
        self.assertRegex(
            text=err_msg,
            expected_regex=(
                r".*Equality constraint 'user_model\.eq_con'.*cannot be written.*"
            ),
        )

        self.assertFalse(
            robust_infeasible,
            msg=(
                "Coefficient matching unexpectedly detected"
                "a robust infeasible constraint"
            ),
        )

        # check constraint partitioning updated as expected
        self.assertEqual(
            model_data.working_model.effective_performance_equality_cons,
            [model_data.working_model.user_model.eq_con],
        )
        self.assertEqual(
            model_data.working_model.effective_first_stage_equality_cons,
            [model_data.working_model.coefficient_matching_conlist[1]],
        )
        assertExpressionsEqual(
            self,
            model_data.working_model.coefficient_matching_conlist[1].expr,
            (-1) + m.x1 == 0,
        )

    def test_coefficient_matching_robust_infeasible_proof(self):
        """
        Test coefficient matching detects robust infeasibility
        as expected.
        """
        # Write the deterministic Pyomo model
        model_data = self.setup_test_model_data()
        m = model_data.working_model.user_model
        m.eq_con.set_value(
            expr=m.u * (m.x1**3 + 0.5)
            - 5 * m.u * m.x1 * m.x2
            + m.u * (m.x1 + 2)
            + m.u**2
            == 0
        )
        ep = model_data.working_model.effective_var_partitioning
        ep.first_stage_variables = [m.x1, m.x2]
        ep.second_stage_variables = []

        config = Bunch()
        config.decision_rule_order = 1
        config.progress_logger = logger

        model_data.working_model.all_nonadjustable_variables = list(
            ep.first_stage_variables
        )

        with LoggingIntercept(level=logging.INFO) as LOG:
            robust_infeasible = perform_coefficient_matching(model_data, config)

        self.assertTrue(
            robust_infeasible,
            msg="Coefficient matching should be proven robust infeasible.",
        )
        robust_infeasible_msg = LOG.getvalue()
        self.assertRegex(
            text=robust_infeasible_msg,
            expected_regex=(
                r"PyROS has determined that the model is robust infeasible\. "
                r"One reason for this.*equality constraint 'user_model\.eq_con'.*"
            )
        )


class TestPreprocessModelData(unittest.TestCase):
    """
    Test the PyROS preprocessor.
    """
    def build_test_model_data(self):
        """
        Build model data object for the preprocessor.
        """
        model_data = Bunch()
        model_data.original_model = m = ConcreteModel()

        # PARAMS: one uncertain, one certain
        m.p = Param(initialize=2, mutable=True)
        m.q = Param(initialize=4.5, mutable=True)

        # first-stage variables
        m.x1 = Var(bounds=(0, m.q), initialize=1)
        m.x2 = Var(domain=NonNegativeReals, bounds=[m.p, m.p], initialize=m.p)

        # second-stage variables
        m.z1 = Var(domain=RangeSet(2, 4, 0), bounds=[-m.p, m.q], initialize=2)
        m.z2 = Var(bounds=(-2 * m.q ** 2, None), initialize=1)
        m.z3 = Var(bounds=(-m.q, 0), initialize=0)
        m.z4 = Var(initialize=5)
        # the bounds produce an equality constraint
        # that then leads to coefficient matching.
        # problem is robust infeasible if DR static, else
        # matching constraints are added
        m.z5 = Var(domain=NonNegativeReals, bounds=(m.q, m.q))

        # state variables
        m.y1 = Var(domain=NonNegativeReals, initialize=0)
        m.y2 = Var(initialize=10)
        # note: y3 out-of-scope, as it will not appear in the active
        #       Objective and Constraint objects
        m.y3 = Var(domain=RangeSet(0, 1, 0), bounds=(0.2, 0.5))

        # fix some variables
        m.z4.fix()
        m.y2.fix()

        # EQUALITY CONSTRAINTS
        # this will be reformulated by coefficient matching
        m.eq1 = Constraint(expr=m.q * (m.z3 + m.x2) == 0)
        # ranged constraints with identical bounds are considered equalities
        # this makes z1 nonadjustable
        m.eq2 = Constraint(expr=m.x1 - m.z1 == 0)
        # pretriangular: makes z2 nonadjustable, so first-stage
        m.eq3 = Constraint(expr=m.x1 ** 2 + m.x2 + m.p * m.z2 == m.p)
        # performance equality
        m.eq4 = Constraint(expr=m.z3 + m.y1 == m.q)

        # INEQUALITY CONSTRAINTS
        # since x1, z1 nonadjustable, LB is first-stage. but UB is performance
        m.ineq1 = Constraint(expr=(-m.p, m.x1 + m.z1, exp(m.q)))
        # two first-stage inequalities
        m.ineq2 = Constraint(expr=(0, m.x1 + m.x2, 10))
        # though the bounds are structurally equal, they are not
        # identical objects, so this constitutes two performance inequalities
        # note: these inequalities redundant, as collectively these constraints
        # are mathematically identical to eq4
        m.ineq3 = Constraint(expr=(2 * m.q, 2 * (m.z3 + m.y1), 2 * m.q))
        # performance inequality. trivially satisfied/infeasible,
        # since y2 is fixed
        m.ineq4 = Constraint(expr=-m.q <= m.y2 ** 2 + log(m.y2))

        # out of scope: deactivated
        m.ineq5 = Constraint(expr=m.y3 <= m.q)
        m.ineq5.deactivate()

        # OBJECTIVE
        # contains a rich combination of first-stage and second-stage terms
        m.obj = Objective(
            expr=(
                m.p ** 2
                + 2 * m.p * m.q
                + log(m.x1)
                + 2 * m.p * m.x1
                + m.q ** 2 * m.x1
                + m.p ** 3 * (m.z1 + m.z2 + m.y1)
                + m.z4
                + m.z5
            ),
        )

        # set up the var partitioning
        user_var_partitioning = VariablePartitioning(
            first_stage_variables=[m.x1, m.x2],
            second_stage_variables=[m.z1, m.z2, m.z3, m.z4, m.z5],
            # note: y3 out of scope, so excluded
            state_variables=[m.y1, m.y2],
        )

        return model_data, user_var_partitioning

    def test_preprocessor_effective_var_partitioning_static_dr(self):
        """
        Test preprocessor repartitions the variables
        as expected.
        """
        # setup
        model_data, user_var_partitioning = self.build_test_model_data()
        om = model_data.original_model
        config = Bunch(
            uncertain_params=[om.q],
            objective_focus=ObjectiveType.worst_case,
            decision_rule_order=0,
            progress_logger=logger,
        )
        preprocess_model_data(
            model_data, config, user_var_partitioning,
        )
        ep = model_data.working_model.effective_var_partitioning
        ublk = model_data.working_model.user_model
        self.assertEqual(
            ComponentSet(ep.first_stage_variables),
            ComponentSet([
                # all second-stage variables are nonadjustable
                # due to the DR
                ublk.x1, ublk.x2, ublk.z1, ublk.z2,
                ublk.z3, ublk.z4, ublk.z5, ublk.y2,
            ]),
        )
        self.assertEqual(ep.second_stage_variables, [])
        self.assertEqual(ep.state_variables, [ublk.y1])

        working_model = model_data.working_model
        self.assertEqual(
            ComponentSet(working_model.all_nonadjustable_variables),
            ComponentSet(
                [
                    ublk.x1, ublk.x2, ublk.z1, ublk.z2,
                    ublk.z3, ublk.z4, ublk.z5, ublk.y2,
                ]
                + [working_model.epigraph_var]
            ),
        )
        self.assertEqual(
            ComponentSet(working_model.all_variables),
            ComponentSet(
                [
                    ublk.x1,
                    ublk.x2,
                    ublk.z1,
                    ublk.z2,
                    ublk.z3,
                    ublk.z4,
                    ublk.z5,
                    ublk.y1,
                    ublk.y2,
                ]
                + [working_model.epigraph_var]
            ),
        )

    @parameterized.expand([
        ["affine", 1],
        ["quadratic", 2],
    ])
    def test_preprocessor_effective_var_partitioning_nonstatic_dr(self, name, dr_order):
        """
        Test preprocessor repartitions the variables
        as expected.
        """
        model_data, user_var_partitioning = self.build_test_model_data()
        om = model_data.original_model
        config = Bunch(
            uncertain_params=[om.q],
            objective_focus=ObjectiveType.worst_case,
            decision_rule_order=dr_order,
            progress_logger=logger,
        )
        preprocess_model_data(
            model_data, config, user_var_partitioning,
        )
        ep = model_data.working_model.effective_var_partitioning
        ublk = model_data.working_model.user_model
        self.assertEqual(
            ComponentSet(ep.first_stage_variables),
            ComponentSet([ublk.x1, ublk.x2, ublk.z1, ublk.z2, ublk.z4, ublk.y2]),
        )
        self.assertEqual(
            ComponentSet(ep.second_stage_variables),
            ComponentSet([ublk.z3, ublk.z5]),
        )
        self.assertEqual(
            ComponentSet(ep.state_variables),
            ComponentSet([ublk.y1]),
        )
        working_model = model_data.working_model
        self.assertEqual(
            ComponentSet(working_model.all_nonadjustable_variables),
            ComponentSet(
                [ublk.x1, ublk.x2, ublk.z1, ublk.z2, ublk.z4, ublk.y2]
                + [working_model.epigraph_var]
                + list(working_model.decision_rule_var_0.values())
                + list(working_model.decision_rule_var_1.values())
            ),
        )
        self.assertEqual(
            ComponentSet(working_model.all_variables),
            ComponentSet(
                [
                    ublk.x1,
                    ublk.x2,
                    ublk.z1,
                    ublk.z2,
                    ublk.z3,
                    ublk.z4,
                    ublk.z5,
                    ublk.y1,
                    ublk.y2,
                ]
                + [working_model.epigraph_var]
                + list(working_model.decision_rule_var_0.values())
                + list(working_model.decision_rule_var_1.values())
            ),
        )

    @parameterized.expand([
        ["affine_nominal", 1, "nominal"],
        ["affine_worst_case", 1, "worst_case"],
        # eq1 doesn't get reformulated in coefficient matching
        #  as the polynomial degree is too high
        ["quadratic_nominal", 2, "nominal"],
        ["quadratic_worst_case", 2, "worst_case"],
    ])
    def test_preprocessor_constraint_partitioning_nonstatic_dr(
            self, name, dr_order, obj_focus,
            ):
        """
        Test preprocessor partitions constraints as expected
        for nonstatic DR.
        """
        model_data, user_var_partitioning = self.build_test_model_data()
        om = model_data.original_model
        config = Bunch(
            uncertain_params=[om.q],
            objective_focus=ObjectiveType[obj_focus],
            decision_rule_order=dr_order,
            progress_logger=logger,
        )
        preprocess_model_data(
            model_data, config, user_var_partitioning,
        )

        working_model = model_data.working_model
        ublk = working_model.user_model
        self.assertEqual(
            ComponentSet(working_model.effective_first_stage_inequality_cons),
            ComponentSet(
                [ublk.ineq1, ublk.ineq2]
                + ([working_model.epigraph_con] if obj_focus == "nominal" else [])
            ),
        )
        self.assertEqual(
            ComponentSet(working_model.effective_first_stage_equality_cons),
            ComponentSet(
                [
                    ublk.eq2,
                    ublk.eq3,
                    *working_model.coefficient_matching_conlist.values(),
                ]
            ),
        )
        self.assertEqual(
            ComponentSet(working_model.effective_performance_inequality_cons),
            ComponentSet(
                [
                    ublk.find_component("var_x1_uncertain_upper_bound_con"),
                    ublk.find_component("var_z1_uncertain_upper_bound_con"),
                    ublk.find_component("var_z2_uncertain_lower_bound_con"),
                    ublk.find_component("var_z3_certain_upper_bound_con"),
                    ublk.find_component("var_z3_uncertain_lower_bound_con"),
                    ublk.find_component("var_z5_certain_lower_bound_con"),
                    ublk.find_component("var_y1_certain_lower_bound_con"),
                    ublk.find_component("con_ineq1_upper_bound_con"),
                    ublk.find_component("con_ineq3_lower_bound_con"),
                    ublk.find_component("con_ineq3_upper_bound_con"),
                    ublk.ineq4,
                ]
                + ([working_model.epigraph_con] if obj_focus == "worst_case" else [])
            ),
        )
        self.assertEqual(
            ComponentSet(working_model.effective_performance_equality_cons),
            # eq1 doesn't get reformulated in coefficient matching
            # when DR order is 2 as the polynomial degree is too high
            ComponentSet([ublk.eq4] + ([ublk.eq1] if dr_order == 2 else [])),
        )

        # verify the constraints are active
        for fs_eq_con in working_model.effective_first_stage_equality_cons:
            self.assertTrue(fs_eq_con.active, msg=f"{fs_eq_con.name} inactive")
        for fs_ineq_con in working_model.effective_first_stage_inequality_cons:
            self.assertTrue(fs_ineq_con.active, msg=f"{fs_ineq_con.name} inactive")
        for perf_eq_con in working_model.effective_performance_equality_cons:
            self.assertTrue(perf_eq_con.active, msg=f"{perf_eq_con.name} inactive")
        for perf_ineq_con in working_model.effective_performance_inequality_cons:
            self.assertTrue(perf_ineq_con.active, msg=f"{perf_ineq_con.name} inactive")

        # verify the constraint expressions
        m = ublk
        assertExpressionsEqual(self, m.x1.lower, 0)
        assertExpressionsEqual(
            self,
            m.var_x1_uncertain_upper_bound_con.expr, m.x1 <= m.q,
        )

        assertExpressionsEqual(
            self,
            m.var_z1_uncertain_upper_bound_con.expr,
            m.z1 <= m.q,
        )
        assertExpressionsEqual(
            self,
            m.var_z2_uncertain_lower_bound_con.expr,
            -m.z2 <= -(-2 * m.q ** 2),
        )
        assertExpressionsEqual(
            self,
            m.var_z3_uncertain_lower_bound_con.expr,
            -m.z3 <= -(-m.q),
        )
        assertExpressionsEqual(
            self,
            m.var_z3_certain_upper_bound_con.expr,
            m.z3 <= 0,
        )
        assertExpressionsEqual(
            self,
            m.var_z5_certain_lower_bound_con.expr,
            -m.z5 <= 0,
        )
        assertExpressionsEqual(
            self,
            m.var_y1_certain_lower_bound_con.expr,
            -m.y1 <= 0,
        )
        assertExpressionsEqual(
            self,
            m.ineq1.expr,
            -m.p <= m.x1 + m.z1,
        )
        assertExpressionsEqual(
            self,
            m.con_ineq1_upper_bound_con.expr,
            m.x1 + m.z1 <= exp(m.q),
        )
        assertExpressionsEqual(
            self,
            m.ineq2.expr,
            RangedExpression((0, m.x1 + m.x2, 10), False),
        )
        assertExpressionsEqual(
            self,
            m.con_ineq3_lower_bound_con.expr,
            -(2 * (m.z3 + m.y1)) <= -(2 * m.q),
        )
        assertExpressionsEqual(
            self,
            m.con_ineq3_upper_bound_con.expr,
            2 * (m.z3 + m.y1) <= 2 * m.q,
        )
        assertExpressionsEqual(
            self,
            m.ineq3.upper,
            None,
        )
        self.assertFalse(m.ineq3.active)
        assertExpressionsEqual(
            self,
            m.ineq4.expr,
            -(m.y2 ** 2 + log(m.y2)) <= -(-m.q),
        )
        self.assertFalse(m.ineq5.active)

        assertExpressionsEqual(
            self,
            m.eq2.expr,
            m.x1 - m.z1 == 0,
        )
        assertExpressionsEqual(
            self,
            m.eq3.expr,
            m.x1 ** 2 + m.x2 + m.p * m.z2 == m.p,
        )
        if dr_order < 2:
            # due to coefficient matching
            self.assertFalse(m.eq1.active)

    @parameterized.expand([
        ["static", 0, True],
        ["affine", 1, False],
        ["quadratic", 2, False],
    ])
    def test_preprocessor_coefficient_matching(
            self, name, dr_order, expected_robust_infeas,
            ):
        """
        Check preprocessor robust infeasibility return status.
        """
        model_data, user_var_partitioning = self.build_test_model_data()
        om = model_data.original_model
        config = Bunch(
            uncertain_params=[om.q],
            objective_focus=ObjectiveType.worst_case,
            decision_rule_order=dr_order,
            progress_logger=logger,
        )

        # static DR, problem should be robust infeasible
        # due to the coefficient matching constraints derived
        # from bounds on z5
        robust_infeasible = preprocess_model_data(
            model_data, config, user_var_partitioning,
        )
        self.assertIsInstance(robust_infeasible, bool)
        self.assertEqual(robust_infeasible, expected_robust_infeas)

        # check the coefficient matching constraint expressions
        working_model = model_data.working_model
        m = model_data.working_model.user_model
        working_model.coefficient_matching_conlist.pprint()
        if config.decision_rule_order == 1:
            # check the constraint expressions of eq1 and z5 bound
            self.assertFalse(m.eq1.active)
            assertExpressionsEqual(
                self,
                working_model.coefficient_matching_conlist[1].expr,
                working_model.decision_rule_vars[1][0] == 0,
            )
            assertExpressionsEqual(
                self,
                working_model.coefficient_matching_conlist[2].expr,
                -1 + working_model.decision_rule_vars[1][1] == 0,
            )
            assertExpressionsEqual(
                self,
                working_model.coefficient_matching_conlist[3].expr,
                working_model.decision_rule_vars[0][0] + m.x2 == 0,
            )
            assertExpressionsEqual(
                self,
                working_model.coefficient_matching_conlist[4].expr,
                working_model.decision_rule_vars[0][1] == 0,
            )
        if config.decision_rule_order == 2:
            # check the constraint expressions of eq1 and eq4
            self.assertTrue(m.eq1.active)
            assertExpressionsEqual(
                self,
                m.eq1.expr,
                m.q * (m.z3 + m.x2) == 0,
            )
            assertExpressionsEqual(
                self,
                working_model.coefficient_matching_conlist[1].expr,
                working_model.decision_rule_vars[1][0] == 0,
            )
            assertExpressionsEqual(
                self,
                working_model.coefficient_matching_conlist[2].expr,
                -1 + working_model.decision_rule_vars[1][1] == 0,
            )
            assertExpressionsEqual(
                self,
                working_model.coefficient_matching_conlist[3].expr,
                working_model.decision_rule_vars[1][2] == 0,
            )

    @parameterized.expand([
        ["static", 0],
        ["affine", 1],
        ["quadratic", 2],
    ])
    def test_preprocessor_objective_standardization(self, name, dr_order):
        """
        Test preprocessor standardizes the active objective as
        expected.
        """
        model_data, user_var_partitioning = self.build_test_model_data()
        om = model_data.original_model
        config = Bunch(
            uncertain_params=[om.q],
            objective_focus=ObjectiveType.worst_case,
            decision_rule_order=dr_order,
            progress_logger=logger,
        )
        preprocess_model_data(
            model_data, config, user_var_partitioning,
        )

        ublk = model_data.working_model.user_model
        working_model = model_data.working_model
        assertExpressionsEqual(
            self,
            working_model.epigraph_con.expr,
            ublk.obj.expr - working_model.epigraph_var <= 0
        )
        assertExpressionsEqual(
            self,
            working_model.full_objective.expr,
            ublk.obj.expr,
        )

        # recall: objective summands are classified according
        # to dependence on uncertain parameters and variables
        # the *user* considers adjustable
        # so the summands should be independent of the DR order
        assertExpressionsEqual(
            self,
            working_model.first_stage_objective.expr,
            ublk.p ** 2 + log(ublk.x1) + 2 * ublk.p * ublk.x1,
        )
        assertExpressionsEqual(
            self,
            working_model.second_stage_objective.expr,
            (
                2 * ublk.p * ublk.q
                + ublk.q ** 2 * ublk.x1
                + ublk.p ** 3 * (ublk.z1 + ublk.z2 + ublk.y1)
                + ublk.z4
                + ublk.z5
            ),
        )

    @parameterized.expand([["nominal"], ["worst_case"]])
    def test_preprocessor_log_model_statistics_affine_dr(self, obj_focus):
        """
        Test statistics of the preprocessed working model are
        logged as expected.
        """
        model_data, user_var_partitioning = self.build_test_model_data()
        om = model_data.original_model
        config = Bunch(
            uncertain_params=[om.q],
            objective_focus=ObjectiveType[obj_focus],
            decision_rule_order=1,
            progress_logger=logger,
        )
        preprocess_model_data(
            model_data, config, user_var_partitioning,
        )

        # expected model stats worked out by hand
        expected_log_str = textwrap.dedent(
            f"""
            Model Statistics:
              Number of variables : 14
                Epigraph variable : 1
                First-stage variables : 2
                Second-stage variables : 5 (2 adj.)
                State variables : 2 (1 adj.)
                Decision rule variables : 4
              Number of uncertain parameters : 1
              Number of constraints : 23
                Equality constraints : 9
                  Coefficient matching constraints : 4
                  Other first-stage equations : 2
                  Performance equations : 1
                  Decision rule equations : 2
                Inequality constraints : 14
                  First-stage inequalities : {3 if obj_focus == 'nominal' else 2}
                  Performance inequalities : {11 if obj_focus == 'nominal' else 12}
            """
        )

        with LoggingIntercept(level=logging.INFO) as LOG:
            log_model_statistics(model_data, config)
        log_str = LOG.getvalue()

        log_lines = log_str.splitlines()[1:]
        expected_log_lines = expected_log_str.splitlines()[1:]

        self.assertEqual(len(log_lines), len(expected_log_lines))
        for line, expected_line in zip(log_lines, expected_log_lines):
            self.assertEqual(line, expected_line)

    @parameterized.expand([["nominal"], ["worst_case"]])
    def test_preprocessor_log_model_statistics_quadratic_dr(self, obj_focus):
        """
        Test statistics of the preprocessed working model are
        logged as expected.
        """
        model_data, user_var_partitioning = self.build_test_model_data()
        om = model_data.original_model
        config = Bunch(
            uncertain_params=[om.q],
            objective_focus=ObjectiveType[obj_focus],
            decision_rule_order=2,
            progress_logger=logger,
        )
        preprocess_model_data(
            model_data, config, user_var_partitioning,
        )

        # expected model stats worked out by hand
        expected_log_str = textwrap.dedent(
            f"""
            Model Statistics:
              Number of variables : 16
                Epigraph variable : 1
                First-stage variables : 2
                Second-stage variables : 5 (2 adj.)
                State variables : 2 (1 adj.)
                Decision rule variables : 6
              Number of uncertain parameters : 1
              Number of constraints : 23
                Equality constraints : 9
                  Coefficient matching constraints : 3
                  Other first-stage equations : 2
                  Performance equations : 2
                  Decision rule equations : 2
                Inequality constraints : 14
                  First-stage inequalities : {3 if obj_focus == 'nominal' else 2}
                  Performance inequalities : {11 if obj_focus == 'nominal' else 12}
            """
        )

        with LoggingIntercept(level=logging.INFO) as LOG:
            log_model_statistics(model_data, config)
        log_str = LOG.getvalue()

        log_lines = log_str.splitlines()[1:]
        expected_log_lines = expected_log_str.splitlines()[1:]

        self.assertEqual(len(log_lines), len(expected_log_lines))
        for line, expected_line in zip(log_lines, expected_log_lines):
            self.assertEqual(line, expected_line)


if __name__ == "__main__":
    unittest.main()
