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

"""
Tests for the PyROS preprocessor.
"""


import logging
import textwrap
import pyomo.common.unittest as unittest

from pyomo.common.collections import Bunch, ComponentSet, ComponentMap
from pyomo.common.dependencies import numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
from pyomo.core.base import (
    Any,
    Var,
    Constraint,
    Expression,
    Objective,
    ConcreteModel,
    Param,
    RangeSet,
    maximize,
    Block,
    Suffix,
)
from pyomo.core.base.set_types import NonNegativeReals, NonPositiveReals, Reals
from pyomo.core.expr import (
    LinearExpression,
    log,
    sin,
    exp,
    RangedExpression,
    SumExpression,
)
from pyomo.core.expr.compare import assertExpressionsEqual

from pyomo.contrib.pyros.uncertainty_sets import BoxSet, DiscreteScenarioSet
from pyomo.contrib.pyros.util import (
    ModelData,
    ObjectiveType,
    get_all_first_stage_eq_cons,
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
    reformulate_state_var_independent_eq_cons,
    setup_working_model,
    VariablePartitioning,
    preprocess_model_data,
    log_model_statistics,
    DEFAULT_SEPARATION_PRIORITY,
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
        m.q = Param(mutable=True, initialize=1)
        m.q2 = Param(mutable=True, initialize=2)
        m.x1 = Var(bounds=(m.q2, m.q2))
        m.x2 = Var()
        m.z = Var()
        m.y = Var(range(1, 5))

        m.c0 = Constraint(expr=m.q + m.x1 + m.z == 0)
        m.c1 = Constraint(expr=(0, m.x1 - m.z * (m.q2 - 1), 0))
        m.c2 = Constraint(expr=m.x1**2 - m.z + m.y[1] == 0)
        m.c2_dupl = Constraint(expr=m.x1**2 - m.z + m.y[1] == 0)
        m.c3 = Constraint(expr=m.x1**3 + m.y[1] + 2 * m.y[2] == 0)
        m.c4 = Constraint(expr=m.x2**2 + m.y[1] + m.y[2] + m.y[3] + m.y[4] == 0)
        m.c5 = Constraint(expr=m.x2 + 2 * m.y[2] + m.y[3] + 2 * m.y[4] == 0)

        model_data = ModelData(
            original_model=m,
            config=Bunch(separation_priority_order=dict()),
            timing=None,
        )
        model_data.working_model = ConcreteModel()
        model_data.working_model.user_model = mdl = m.clone()
        model_data.working_model.uncertain_params = [mdl.q, mdl.q2]
        model_data.working_model.effective_uncertain_params = [mdl.q]

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

        config = model_data.config
        config.decision_rule_order = 0
        config.progress_logger = logger

        expected_partitioning = {
            "first_stage_variables": [m.x1, m.x2, m.z, m.y[1], m.y[2]],
            "second_stage_variables": [],
            "state_variables": [m.y[3], m.y[4]],
        }
        for dr_order in [0, 1, 2]:
            config.decision_rule_order = dr_order
            actual_partitioning = get_effective_var_partitioning(model_data=model_data)
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
        m.c2.set_value(m.x1**2 + m.z + 1e-10 * m.y[1] == 0)
        m.c2_dupl.set_value(m.x1**2 + m.z + 1e-10 * m.y[1] == 0)
        expected_partitioning = {
            "first_stage_variables": [m.x1, m.x2, m.z],
            "second_stage_variables": [],
            "state_variables": list(m.y.values()),
        }
        for dr_order in [0, 1, 2]:
            config.decision_rule_order = dr_order
            actual_partitioning = get_effective_var_partitioning(model_data)
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
        m.c2.set_value(1e-6 * m.y[1] + m.x1**2 + m.z + 1e-10 * m.y[1] == 0)
        m.c2_dupl.set_value(1e-6 * m.y[1] + m.x1**2 + m.z + 1e-10 * m.y[1] == 0)
        expected_partitioning = {
            "first_stage_variables": [m.x1, m.x2, m.z, m.y[1], m.y[2]],
            "second_stage_variables": [],
            "state_variables": [m.y[3], m.y[4]],
        }
        for dr_order in [0, 1, 2]:
            config.decision_rule_order = dr_order
            actual_partitioning = get_effective_var_partitioning(model_data)
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
        m.c3.set_value(m.x1**3 + m.y[1] + 2 * m.y[1] * m.y[2] == 0)
        for dr_order in [0, 1, 2]:
            config.decision_rule_order = dr_order
            actual_partitioning = get_effective_var_partitioning(model_data)
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
            actual_partitioning = get_effective_var_partitioning(model_data)
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
        m.c1.set_value((0, m.x1 + m.z**2, 0))

        config = model_data.config
        config.decision_rule_order = 0
        config.progress_logger = logger

        expected_partitioning_static_dr = {
            "first_stage_variables": [m.x1, m.x2, m.z, m.y[1], m.y[2]],
            "second_stage_variables": [],
            "state_variables": [m.y[3], m.y[4]],
        }
        actual_partitioning_static_dr = get_effective_var_partitioning(model_data)
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
                model_data
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
        m = ConcreteModel()

        # PARAMS: one uncertain, one certain
        m.p = Param(initialize=2, mutable=True)
        m.q = Param(initialize=4.5, mutable=True)

        # first-stage variables
        m.x1 = Var(bounds=(0, m.q), initialize=1)
        m.x2 = Var(domain=NonNegativeReals, bounds=[m.p, m.p], initialize=m.p)

        # second-stage variables
        m.z1 = Var(domain=RangeSet(2, 4, 0), bounds=[-m.p, m.q], initialize=2)
        m.z2 = Var(bounds=(-2 * m.q**2, None), initialize=1)
        m.z3 = Var(bounds=(-m.q, 0), initialize=0)
        m.z4 = Var(initialize=5)
        m.z5 = Var(domain=NonNegativeReals, bounds=(m.q, m.q))

        # state variables
        m.y1 = Var(domain=NonNegativeReals, initialize=0)
        m.y2 = Var(initialize=10)
        # note: y3 out-of-scope, as it will not appear in the active
        #       Objective and Constraint objects
        m.y3 = Var(domain=RangeSet(0, 1, 0), bounds=(0.2, 0.5))

        # Var to represent an uncertain Param;
        # bounds will be ignored
        m.q2var = Var(bounds=(0, None), initialize=3.2)

        # fix some variables
        m.z4.fix()
        m.y2.fix()

        # NAMED EXPRESSIONS: mainly to test
        # Var -> Param substitution for uncertain params
        m.nexpr = Expression(expr=log(m.y2) + m.q2var)

        # EQUALITY CONSTRAINTS
        m.eq1 = Constraint(expr=m.q * (m.z3 + m.x2) == 0)
        m.eq2 = Constraint(expr=m.x1 - m.z1 == 0)
        m.eq3 = Constraint(expr=m.x1**2 + m.x2 + m.p * m.z2 == m.p)
        m.eq4 = Constraint(expr=m.z3 + m.y1 == m.q)

        # INEQUALITY CONSTRAINTS
        m.ineq1 = Constraint(expr=(-m.p, m.x1 + m.z1, exp(m.q)))
        m.ineq2 = Constraint(expr=(0, m.x1 + m.x2, 10))
        m.ineq3 = Constraint(expr=(2 * m.q, 2 * (m.z3 + m.y1), 2 * m.q))
        m.ineq4 = Constraint(expr=-m.q <= m.y2**2 + m.nexpr)

        # out of scope: deactivated
        m.ineq5 = Constraint(expr=m.y3 <= m.q)
        m.ineq5.deactivate()

        # OBJECTIVE
        # contains a rich combination of first-stage and second-stage terms
        m.obj = Objective(
            expr=(
                m.p**2
                + 2 * m.p * m.q
                + log(m.x1)
                + 2 * m.p * m.x1
                + m.q**2 * m.x1
                + m.p**3 * (m.z1 + m.z2 + m.y1)
                + m.z4
                + m.z5
            )
        )

        # inactive objective
        m.inactive_obj = Objective(expr=1 + m.q2var + m.x1)
        m.inactive_obj.deactivate()

        # set up the var partitioning
        user_var_partitioning = VariablePartitioning(
            first_stage_variables=[m.x1, m.x2],
            second_stage_variables=[m.z1, m.z2, m.z3, m.z4, m.z5],
            # note: y3 out of scope, so excluded
            state_variables=[m.y1, m.y2],
        )

        model_data = ModelData(original_model=m, config=Bunch(), timing=None)

        return model_data, user_var_partitioning

    def test_setup_working_model(self):
        """
        Test method for setting up the working model is as expected.
        """
        model_data, user_var_partitioning = self.build_test_model_data()
        om = model_data.original_model
        config = model_data.config
        config.uncertain_params = [om.q, om.q2var]
        config.progress_logger = logger
        config.nominal_uncertain_param_vals = [om.q.value, om.q2var.value]

        setup_working_model(model_data, user_var_partitioning)
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
        self.assertFalse(m.inactive_obj.active)

        # user var partitioning
        up = working_model.user_var_partitioning
        self.assertEqual(
            ComponentSet(up.first_stage_variables), ComponentSet([m.x1, m.x2])
        )
        self.assertEqual(
            ComponentSet(up.second_stage_variables),
            ComponentSet([m.z1, m.z2, m.z3, m.z4, m.z5]),
        )
        self.assertEqual(ComponentSet(up.state_variables), ComponentSet([m.y1, m.y2]))

        # uncertain params
        self.assertEqual(
            ComponentSet(working_model.orig_uncertain_params),
            ComponentSet([m.q, m.q2var]),
        )

        self.assertEqual(list(working_model.temp_uncertain_params.index_set()), [1])
        temp_uncertain_param = working_model.temp_uncertain_params[1]
        self.assertEqual(
            ComponentSet(working_model.uncertain_params),
            ComponentSet([m.q, temp_uncertain_param]),
        )

        # ensure original model unchanged
        self.assertFalse(
            hasattr(om, "util"), msg="Original model still has temporary util block"
        )

        # constraint partitioning initialization
        self.assertFalse(working_model.first_stage.inequality_cons)
        self.assertFalse(working_model.first_stage.dr_dependent_equality_cons)
        self.assertFalse(working_model.first_stage.dr_independent_equality_cons)
        self.assertFalse(working_model.second_stage.inequality_cons)
        self.assertFalse(working_model.second_stage.equality_cons)

        # ensure uncertain Param substitutions carried out properly
        ublk = model_data.working_model.user_model
        self.assertExpressionsEqual(
            ublk.nexpr.expr, log(ublk.y2) + temp_uncertain_param
        )
        self.assertExpressionsEqual(
            ublk.inactive_obj.expr, LinearExpression([1, temp_uncertain_param, m.x1])
        )
        self.assertExpressionsEqual(ublk.ineq4.expr, -ublk.q <= ublk.y2**2 + ublk.nexpr)

        # other component expressions should remain as declared
        self.assertExpressionsEqual(ublk.eq1.expr, ublk.q * (ublk.z3 + ublk.x2) == 0)
        self.assertExpressionsEqual(ublk.eq2.expr, ublk.x1 - ublk.z1 == 0)
        self.assertExpressionsEqual(
            ublk.eq3.expr, ublk.x1**2 + ublk.x2 + ublk.p * ublk.z2 == ublk.p
        )
        self.assertExpressionsEqual(ublk.eq4.expr, ublk.z3 + ublk.y1 == ublk.q)
        self.assertExpressionsEqual(
            ublk.ineq1.expr,
            RangedExpression((-ublk.p, ublk.x1 + ublk.z1, exp(ublk.q)), False),
        )
        self.assertExpressionsEqual(
            ublk.ineq2.expr, RangedExpression((0, ublk.x1 + ublk.x2, 10), False)
        )
        self.assertExpressionsEqual(
            ublk.ineq3.expr,
            RangedExpression((2 * ublk.q, 2 * (ublk.z3 + ublk.y1), 2 * ublk.q), False),
        )
        self.assertExpressionsEqual(ublk.ineq5.expr, ublk.y3 <= ublk.q)
        self.assertExpressionsEqual(
            ublk.obj.expr,
            (
                ublk.p**2
                + 2 * ublk.p * ublk.q
                + log(ublk.x1)
                + 2 * ublk.p * ublk.x1
                + ublk.q**2 * ublk.x1
                + ublk.p**3 * (ublk.z1 + ublk.z2 + ublk.y1)
                + ublk.z4
                + ublk.z5
            ),
        )


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
        original_var_domains = ComponentMap(
            (
                (var, var.domain)
                for var in (m.z1, m.z2, m.z3, m.z4, m.z5, m.z6, m.z7, m.z8, m.z9, m.z10)
            )
        )

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
                ),
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
        m = ConcreteModel()
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
        m.z11 = Var(bounds=[m.q1, m.q1])

        model_data = ModelData(
            original_model=None,
            config=Bunch(separation_priority_order=dict()),
            timing=None,
        )

        model_data.working_model = ConcreteModel()
        model_data.working_model.user_model = m
        model_data.working_model.uncertain_params = [m.q1, m.q2, m.p1]
        model_data.working_model.effective_uncertain_params = [m.q1, m.q2]

        model_data.working_model.second_stage = Block()
        model_data.working_model.second_stage.inequality_cons = Constraint(Any)
        model_data.working_model.second_stage.equality_cons = Constraint(Any)
        model_data.separation_priority_order = dict()

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
            m.z1,
            m.z2,
            m.z3,
            m.z4,
            m.z5,
            m.z6,
            m.z7,
            m.z8,
            m.z11,
        ]
        ep.second_stage_variables = [m.z9]
        ep.state_variables = [m.z10]
        effective_first_stage_var_set = ComponentSet(ep.first_stage_variables)

        # also want to test resolution of separation priorities
        model_data.config.separation_priority_order["z3"] = 10
        model_data.config.separation_priority_order["z8"] = 9
        model_data.config.separation_priority_order["z11"] = None
        m.pyros_separation_priority = Suffix()
        m.pyros_separation_priority[m.z4] = 1
        m.pyros_separation_priority[m.z6] = 2
        # note: this suffix entry, rather than the
        #       config specification, should determine the priority
        m.pyros_separation_priority[m.z8] = 4

        original_var_domains_and_bounds = ComponentMap(
            (var, (var.domain, get_var_bound_pairs(var)[1]))
            for var in model_data.working_model.user_model.component_data_objects(Var)
        )

        # expected final bounds and bound constraint types
        expected_final_nonadj_var_bounds = ComponentMap(
            (
                (m.z1, (get_var_bound_pairs(m.z1)[1], [])),
                (m.z2, (get_var_bound_pairs(m.z2)[1], [])),
                (m.z3, (get_var_bound_pairs(m.z3)[1], [])),
                (m.z4, ((None, 0), ["lower"])),
                (m.z5, ((4, None), ["upper"])),
                (m.z6, ((None, None), ["eq"])),
                (m.z7, ((None, None), ["eq"])),
                (m.z8, ((None, None), ["lower", "upper"])),
                (m.z11, ((m.q1, m.q1), [])),
            )
        )

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

        second_stage = working_model.second_stage

        # verify bound constraint expressions
        assertExpressionsEqual(
            self,
            second_stage.inequality_cons["var_z4_uncertain_lower_bound_con"].expr,
            -m.z4 <= -m.q1,
        )
        assertExpressionsEqual(
            self,
            second_stage.inequality_cons["var_z5_uncertain_upper_bound_con"].expr,
            m.z5 <= m.q2,
        )
        assertExpressionsEqual(
            self,
            second_stage.equality_cons["var_z6_uncertain_eq_bound_con"].expr,
            m.z6 == m.q1,
        )
        assertExpressionsEqual(
            self,
            second_stage.equality_cons["var_z7_uncertain_eq_bound_con"].expr,
            m.z7 == m.q1,
        )
        assertExpressionsEqual(
            self,
            second_stage.inequality_cons["var_z8_uncertain_lower_bound_con"].expr,
            -m.z8 <= -m.q1,
        )
        assertExpressionsEqual(
            self,
            second_stage.inequality_cons["var_z8_uncertain_upper_bound_con"].expr,
            m.z8 <= m.q2,
        )

        # check constraint partitioning
        self.assertEqual(
            len(working_model.second_stage.inequality_cons),
            4,
            msg="Number of second-stage inequalities not as expected.",
        )
        self.assertEqual(
            len(working_model.second_stage.equality_cons),
            2,
            msg="Number of second-stage equalities not as expected.",
        )

        # check separation priorities
        self.assertEqual(
            len(model_data.separation_priority_order),
            (len(second_stage.inequality_cons) + len(second_stage.equality_cons)),
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z4_uncertain_lower_bound_con"], 1
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z5_uncertain_upper_bound_con"],
            DEFAULT_SEPARATION_PRIORITY,
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z6_uncertain_eq_bound_con"], 2
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z7_uncertain_eq_bound_con"],
            DEFAULT_SEPARATION_PRIORITY,
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z8_uncertain_lower_bound_con"], 4
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z8_uncertain_upper_bound_con"], 4
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

        model_data.working_model.first_stage = Block()
        model_data.working_model.first_stage.inequality_cons = Constraint(Any)
        model_data.working_model.first_stage.dr_independent_equality_cons = Constraint(
            Any
        )

        m = model_data.working_model.user_model

        # simple mock partitioning for the test
        ep = model_data.working_model.effective_var_partitioning = Bunch()
        ep.first_stage_variables = [m.z9, m.z10]
        ep.second_stage_variables = [m.z1, m.z2, m.z3, m.z4, m.z5, m.z6]
        ep.state_variables = [m.z7, m.z8, m.z11]
        effective_first_stage_var_set = ComponentSet(ep.first_stage_variables)

        original_var_domains_and_bounds = ComponentMap(
            (var, (var.domain, get_var_bound_pairs(var)[1]))
            for var in model_data.working_model.user_model.component_data_objects(Var)
        )

        model_data.config.separation_priority_order["z3"] = 10
        model_data.config.separation_priority_order["z8"] = 9
        model_data.config.separation_priority_order["z11"] = None
        m.pyros_separation_priority = Suffix()
        m.pyros_separation_priority[m.z4] = 1
        m.pyros_separation_priority[m.z6] = 2
        # note: this suffix entry, rather than the
        #       config specification, should determine the priority
        m.pyros_separation_priority[m.z8] = 4

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

        first_stage = model_data.working_model.first_stage
        second_stage = model_data.working_model.second_stage

        self.assertEqual(len(first_stage.dr_independent_equality_cons), 1)
        self.assertEqual(len(first_stage.inequality_cons), 0)
        self.assertEqual(len(second_stage.inequality_cons), 10)
        self.assertEqual(len(second_stage.equality_cons), 5)

        # verify bound constraint expressions
        assertExpressionsEqual(
            self,
            second_stage.equality_cons["var_z2_certain_eq_bound_con"].expr,
            m.z2 == 1,
        )
        assertExpressionsEqual(
            self,
            second_stage.inequality_cons["var_z3_certain_lower_bound_con"].expr,
            -m.z3 <= -2,
        )
        assertExpressionsEqual(
            self,
            second_stage.inequality_cons["var_z3_certain_upper_bound_con"].expr,
            m.z3 <= m.p1,
        )
        assertExpressionsEqual(
            self,
            second_stage.equality_cons["var_z4_certain_eq_bound_con"].expr,
            m.z4 == 0,
        )
        assertExpressionsEqual(
            self,
            second_stage.inequality_cons["var_z4_uncertain_lower_bound_con"].expr,
            -m.z4 <= -m.q1,
        )
        assertExpressionsEqual(
            self,
            second_stage.equality_cons["var_z5_certain_eq_bound_con"].expr,
            m.z5 == 4,
        )
        assertExpressionsEqual(
            self,
            second_stage.inequality_cons["var_z5_uncertain_upper_bound_con"].expr,
            m.z5 <= m.q2,
        )
        assertExpressionsEqual(
            self,
            second_stage.inequality_cons["var_z6_certain_lower_bound_con"].expr,
            -m.z6 <= 0,
        )
        assertExpressionsEqual(
            self,
            second_stage.equality_cons["var_z6_uncertain_eq_bound_con"].expr,
            m.z6 == m.q1,
        )
        assertExpressionsEqual(
            self,
            second_stage.inequality_cons["var_z7_certain_upper_bound_con"].expr,
            m.z7 <= 0,
        )
        assertExpressionsEqual(
            self,
            second_stage.equality_cons["var_z7_uncertain_eq_bound_con"].expr,
            m.z7 == m.q1,
        )
        assertExpressionsEqual(
            self,
            second_stage.inequality_cons["var_z8_certain_lower_bound_con"].expr,
            -m.z8 <= 0,
        )
        assertExpressionsEqual(
            self,
            second_stage.inequality_cons["var_z8_certain_upper_bound_con"].expr,
            m.z8 <= 5,
        )
        assertExpressionsEqual(
            self,
            second_stage.inequality_cons["var_z8_uncertain_lower_bound_con"].expr,
            -m.z8 <= -m.q1,
        )
        assertExpressionsEqual(
            self,
            second_stage.inequality_cons["var_z8_uncertain_upper_bound_con"].expr,
            m.z8 <= m.q2,
        )
        fs_dr_indep_cons = first_stage.dr_independent_equality_cons
        assertExpressionsEqual(
            self, fs_dr_indep_cons["var_z11_uncertain_eq_bound_con"].expr, m.z11 == m.q1
        )

        # check separation priorities
        self.assertEqual(
            len(model_data.separation_priority_order),
            (len(second_stage.inequality_cons) + len(second_stage.equality_cons)),
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z2_certain_eq_bound_con"],
            DEFAULT_SEPARATION_PRIORITY,
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z3_certain_lower_bound_con"], 10
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z3_certain_upper_bound_con"], 10
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z4_certain_eq_bound_con"], 1
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z4_uncertain_lower_bound_con"], 1
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z5_certain_eq_bound_con"],
            DEFAULT_SEPARATION_PRIORITY,
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z5_uncertain_upper_bound_con"],
            DEFAULT_SEPARATION_PRIORITY,
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z6_certain_lower_bound_con"], 2
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z6_uncertain_eq_bound_con"], 2
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z7_certain_upper_bound_con"],
            DEFAULT_SEPARATION_PRIORITY,
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z7_uncertain_eq_bound_con"],
            DEFAULT_SEPARATION_PRIORITY,
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z8_certain_lower_bound_con"], 4
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z8_certain_upper_bound_con"], 4
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z8_uncertain_lower_bound_con"], 4
        )
        self.assertEqual(
            model_data.separation_priority_order["var_z8_uncertain_upper_bound_con"], 4
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
        model_data = ModelData(original_model=None, timing=None, config=Bunch())
        model_data.working_model = ConcreteModel()
        model_data.working_model.user_model = m = Block()

        m.x1 = Var()
        m.x2 = Var()
        m.z1 = Var()
        m.z2 = Var()
        m.y1 = Var()

        m.p = Param(initialize=2, mutable=True)
        m.q = Param(mutable=True, initialize=1)
        m.q_cert = Param(mutable=True, initialize=1)

        m.c1 = Constraint(expr=m.x1 <= 1)
        m.c2 = Constraint(expr=(1, m.x1, 2 * m.q_cert))
        m.c3 = Constraint(expr=m.q <= m.x1)
        m.c3_up = Constraint(expr=m.x1 - 2 * m.q <= 0)
        m.c4 = Constraint(expr=(log(m.p), m.x2, m.q))
        m.c5 = Constraint(expr=(m.q, m.x2, 2 * m.q))
        m.c6 = Constraint(expr=m.z1 <= 1)
        m.c7 = Constraint(expr=(0, m.z2, 1))
        m.c8 = Constraint(expr=(m.p**0.5, m.y1, m.p))
        m.c9 = Constraint(expr=m.y1 - m.q <= 0)
        m.c10 = Constraint(expr=m.y1 <= m.q**2)
        m.c11 = Constraint(expr=m.z2 <= m.q)
        m.c12 = Constraint(expr=(m.q**2, m.x1, sin(m.p) * m.q_cert))
        m.c13 = Constraint(expr=m.x1 <= m.q)

        m.c11.deactivate()

        model_data.working_model.uncertain_params = [m.q, m.q_cert]
        model_data.working_model.effective_uncertain_params = [m.q]

        model_data.working_model.first_stage = Block()
        model_data.working_model.first_stage.inequality_cons = Constraint(Any)
        model_data.working_model.second_stage = Block()
        model_data.working_model.second_stage.inequality_cons = Constraint(Any)

        model_data.working_model.original_active_inequality_cons = [
            m.c1,
            m.c2,
            m.c3,
            m.c3_up,
            m.c4,
            m.c5,
            m.c6,
            m.c7,
            m.c8,
            m.c9,
            m.c10,
            m.c12,
            m.c13,
        ]

        ep = model_data.working_model.effective_var_partitioning = Bunch()
        ep.first_stage_variables = [m.x1, m.x2]
        ep.second_stage_variables = [m.z1, m.z2]
        ep.state_variables = [m.y1]

        model_data.separation_priority_order = dict()

        return model_data

    def test_standardize_inequality_constraints(self):
        """
        Test inequality constraint standardization routine.
        """
        model_data = self.build_simple_test_model_data()
        working_model = model_data.working_model
        m = working_model.user_model

        model_data.config.separation_priority_order = dict(c3=1, c5=2)
        m.pyros_separation_priority = Suffix()
        m.pyros_separation_priority[m.c5] = 10
        m.pyros_separation_priority[m.c12] = 5
        m.pyros_separation_priority[m.c13] = None
        standardize_inequality_constraints(model_data)

        fs_ineq_cons = working_model.first_stage.inequality_cons
        ss_ineq_cons = working_model.second_stage.inequality_cons
        sep_priority_dict = model_data.separation_priority_order

        self.assertEqual(len(fs_ineq_cons), 5)
        self.assertEqual(len(ss_ineq_cons), 13)
        self.assertEqual(len(sep_priority_dict), 13)

        self.assertFalse(m.c1.active)
        new_c1_con = fs_ineq_cons["ineq_con_c1"]
        self.assertTrue(new_c1_con.active)
        assertExpressionsEqual(self, new_c1_con.expr, m.x1 <= 1)

        # 1 <= m.x1 <= 2; first-stage constraint. no modification
        self.assertFalse(m.c2.active)
        new_c2_con = fs_ineq_cons["ineq_con_c2"]
        self.assertTrue(new_c2_con.active)
        assertExpressionsEqual(
            self, new_c2_con.expr, RangedExpression((1, m.x1, 2 * m.q_cert), False)
        )

        # m.q <= m.x1; single second-stage inequality. modify in place
        self.assertFalse(m.c3.active)
        new_c3_con = ss_ineq_cons["ineq_con_c3_lower_bound_con"]
        self.assertTrue(new_c3_con.active)
        assertExpressionsEqual(self, new_c3_con.expr, -m.x1 <= -m.q)
        self.assertEqual(sep_priority_dict[new_c3_con.index()], 1)

        # m.x1 - 2 * m.q <= 0;
        # single second-stage inequality. modify in place
        # test case where uncertain param is in body,
        # rather than bound, and rest of expression is first-stage
        self.assertFalse(m.c3_up.active)
        new_c3_up_con = ss_ineq_cons["ineq_con_c3_up_upper_bound_con"]
        self.assertTrue(new_c3_up_con.active)
        assertExpressionsEqual(self, new_c3_up_con.expr, m.x1 - 2 * m.q <= 0.0)

        # log(m.p) <= m.x2 <= m.q
        # lower bound is first-stage, upper bound second-stage
        self.assertFalse(m.c4.active)
        new_c4_lower_bound_con = fs_ineq_cons["ineq_con_c4_lower_bound_con"]
        new_c4_upper_bound_con = ss_ineq_cons["ineq_con_c4_upper_bound_con"]
        self.assertTrue(new_c4_lower_bound_con.active)
        self.assertTrue(new_c4_upper_bound_con.active)
        assertExpressionsEqual(self, new_c4_lower_bound_con.expr, log(m.p) <= m.x2)
        assertExpressionsEqual(self, new_c4_upper_bound_con.expr, m.x2 <= m.q)

        # m.q <= m.x2 <= 2 * m.q
        # two second-stage constraints, one for each bound
        self.assertFalse(m.c5.active)
        new_c5_lower_bound_con = ss_ineq_cons["ineq_con_c5_lower_bound_con"]
        new_c5_upper_bound_con = ss_ineq_cons["ineq_con_c5_upper_bound_con"]
        self.assertTrue(new_c5_lower_bound_con.active)
        self.assertTrue(new_c5_lower_bound_con.active)
        assertExpressionsEqual(self, new_c5_lower_bound_con.expr, -m.x2 <= -m.q)
        assertExpressionsEqual(self, new_c5_upper_bound_con.expr, m.x2 <= 2 * m.q)
        self.assertEqual(sep_priority_dict[new_c5_lower_bound_con.index()], 10)
        self.assertEqual(sep_priority_dict[new_c5_upper_bound_con.index()], 10)

        # single second-stage inequality
        self.assertFalse(m.c6.active)
        new_c6_upper_bound_con = ss_ineq_cons["ineq_con_c6_upper_bound_con"]
        self.assertTrue(new_c6_upper_bound_con.active)
        assertExpressionsEqual(self, new_c6_upper_bound_con.expr, m.z1 <= 1.0)

        # two new second-stage inequalities
        self.assertFalse(m.c7.active)
        new_c7_lower_bound_con = ss_ineq_cons["ineq_con_c7_lower_bound_con"]
        new_c7_upper_bound_con = ss_ineq_cons["ineq_con_c7_upper_bound_con"]
        self.assertTrue(new_c7_lower_bound_con.active)
        self.assertTrue(new_c7_upper_bound_con.active)
        assertExpressionsEqual(self, new_c7_lower_bound_con.expr, -m.z2 <= 0.0)
        assertExpressionsEqual(self, new_c7_upper_bound_con.expr, m.z2 <= 1.0)

        # m.p ** 0.5 <= m.y1 <= m.p
        # two second-stage inequalities
        self.assertFalse(m.c8.active)
        new_c8_lower_bound_con = ss_ineq_cons["ineq_con_c8_lower_bound_con"]
        new_c8_upper_bound_con = ss_ineq_cons["ineq_con_c8_upper_bound_con"]
        self.assertTrue(new_c8_lower_bound_con.active)
        self.assertTrue(new_c8_upper_bound_con.active)
        assertExpressionsEqual(self, new_c8_lower_bound_con.expr, -m.y1 <= -m.p**0.5)
        assertExpressionsEqual(self, new_c8_upper_bound_con.expr, m.y1 <= m.p)

        # m.y1 - m.q <= 0
        # one second-stage inequality
        self.assertFalse(m.c9.active)
        new_c9_upper_bound_con = ss_ineq_cons["ineq_con_c9_upper_bound_con"]
        self.assertTrue(new_c9_upper_bound_con.active)
        assertExpressionsEqual(self, new_c9_upper_bound_con.expr, m.y1 - m.q <= 0.0)

        # m.y1 <= m.q ** 2
        # single second-stage inequality
        self.assertFalse(m.c10.active)
        new_c10_upper_bound_con = ss_ineq_cons["ineq_con_c10_upper_bound_con"]
        self.assertTrue(new_c10_upper_bound_con.active)
        assertExpressionsEqual(self, new_c10_upper_bound_con.expr, m.y1 <= m.q**2)

        # originally deactivated;
        # no modification
        self.assertFalse(m.c11.active)
        assertExpressionsEqual(self, m.c11.expr, m.z2 <= m.q)

        # lower bound second-stage; upper bound first-stage
        self.assertFalse(m.c12.active)
        new_c12_lower_bound_con = ss_ineq_cons["ineq_con_c12_lower_bound_con"]
        new_c12_upper_bound_con = fs_ineq_cons["ineq_con_c12_upper_bound_con"]
        self.assertTrue(new_c12_lower_bound_con.active)
        self.assertTrue(new_c12_upper_bound_con.active)
        assertExpressionsEqual(self, new_c12_lower_bound_con.expr, -m.x1 <= -m.q**2)
        assertExpressionsEqual(
            self, new_c12_upper_bound_con.expr, m.x1 <= sin(m.p) * m.q_cert
        )
        self.assertEqual(sep_priority_dict[new_c12_lower_bound_con.index()], 5)

        self.assertFalse(m.c13.active)
        new_c13_upper_bound_con = fs_ineq_cons["ineq_con_c13"]
        self.assertTrue(new_c13_upper_bound_con.active)
        assertExpressionsEqual(self, new_c13_upper_bound_con.expr, m.x1 <= m.q)

        # check separation priorities
        for con_name in ss_ineq_cons:
            should_have_default_priority = (
                all(cname not in con_name for cname in ["c3", "c5", "c12"])
                or "c3_up" in con_name
            )
            if should_have_default_priority:
                self.assertEqual(
                    model_data.separation_priority_order[con_name],
                    DEFAULT_SEPARATION_PRIORITY,
                    msg=(
                        f"Separation priority for entry {con_name!r} of second-stage "
                        "inequalities not as expected."
                    ),
                )

    def test_standardize_inequality_error(self):
        """
        Test exception raised by inequality constraint standardization
        method if equality-type expression detected.
        """
        model_data = self.build_simple_test_model_data()
        model_data.config.separation_priority_order = dict()
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
        model_data = ModelData(
            original_model=None,
            timing=None,
            config=Bunch(separation_priority_order=dict()),
        )
        model_data.working_model = ConcreteModel()
        model_data.working_model.user_model = m = Block()

        m.x1 = Var()
        m.x2 = Var()
        m.z1 = Var()
        m.z2 = Var()
        m.y1 = Var()

        m.p = Param(initialize=2, mutable=True)
        m.q = Param(mutable=True, initialize=1)
        m.q_cert = Param(mutable=True, initialize=1)

        # first-stage equalities
        m.eq1 = Constraint(expr=m.x1 + log(m.p) == m.q_cert + 1)
        m.eq2 = Constraint(expr=(1, m.x2, 1))
        m.eq2_unc = Constraint(expr=(m.q, m.x2, m.q))

        # second-stage equalities
        m.eq3 = Constraint(expr=m.x2 * m.q == 1)
        m.eq4 = Constraint(expr=m.x2 - m.z1**2 == 0)
        m.eq5 = Constraint(expr=m.q == m.y1)
        m.eq6 = Constraint(expr=(m.q, m.y1, m.q))
        m.eq7 = Constraint(expr=m.z2 == 0)

        # make eq7 out of scope
        m.eq7.deactivate()

        model_data.working_model.uncertain_params = [m.q, m.q_cert]
        model_data.working_model.effective_uncertain_params = [m.q]

        model_data.working_model.first_stage = Block()
        model_data.working_model.first_stage.dr_dependent_equality_cons = Constraint(
            Any
        )
        model_data.working_model.first_stage.dr_independent_equality_cons = Constraint(
            Any
        )
        model_data.working_model.second_stage = Block()
        model_data.working_model.second_stage.equality_cons = Constraint(Any)

        model_data.working_model.original_active_equality_cons = [
            m.eq1,
            m.eq2,
            m.eq2_unc,
            m.eq3,
            m.eq4,
            m.eq5,
            m.eq6,
        ]

        ep = model_data.working_model.effective_var_partitioning = Bunch()
        ep.second_stage_variables = [m.x1, m.x2]
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

        m.pyros_separation_priority = Suffix()
        model_data.config.separation_priority_order[m.eq3.local_name] = 2
        model_data.config.separation_priority_order[m.eq5.local_name] = 3
        m.pyros_separation_priority[m.eq3] = 1
        m.pyros_separation_priority[m.eq4] = 10
        m.pyros_separation_priority[m.eq2_unc] = None

        standardize_equality_constraints(model_data)

        first_stage_eq_cons = working_model.first_stage.dr_independent_equality_cons
        second_stage_eq_cons = working_model.second_stage.equality_cons

        self.assertEqual(len(first_stage_eq_cons), 3)
        self.assertEqual(len(second_stage_eq_cons), 4)

        self.assertFalse(m.eq1.active)
        new_eq1_con = first_stage_eq_cons["eq_con_eq1"]
        self.assertTrue(new_eq1_con.active)
        assertExpressionsEqual(self, new_eq1_con.expr, m.x1 + log(m.p) == m.q_cert + 1)

        self.assertFalse(m.eq2.active)
        new_eq2_con = first_stage_eq_cons["eq_con_eq2"]
        self.assertTrue(new_eq2_con.active)
        assertExpressionsEqual(
            self, new_eq2_con.expr, RangedExpression((1, m.x2, 1), False)
        )

        self.assertFalse(m.eq2_unc.active)
        new_eq2_unc_con = first_stage_eq_cons["eq_con_eq2_unc"]
        self.assertTrue(new_eq2_unc_con.active)
        assertExpressionsEqual(
            self, new_eq2_unc_con.expr, RangedExpression((m.q, m.x2, m.q), False)
        )

        self.assertFalse(m.eq3.active)
        new_eq3_con = second_stage_eq_cons["eq_con_eq3"]
        self.assertTrue(new_eq3_con.active)
        assertExpressionsEqual(self, new_eq3_con.expr, m.x2 * m.q == 1)

        self.assertFalse(m.eq4.active)
        new_eq4_con = second_stage_eq_cons["eq_con_eq4"]
        self.assertTrue(new_eq4_con)
        assertExpressionsEqual(self, new_eq4_con.expr, m.x2 - m.z1**2 == 0)

        self.assertFalse(m.eq5.active)
        new_eq5_con = second_stage_eq_cons["eq_con_eq5"]
        self.assertTrue(new_eq5_con)
        assertExpressionsEqual(self, new_eq5_con.expr, m.q == m.y1)

        self.assertFalse(m.eq6.active)
        new_eq6_con = second_stage_eq_cons["eq_con_eq6"]
        self.assertTrue(new_eq6_con.active)
        assertExpressionsEqual(
            self, new_eq6_con.expr, RangedExpression((m.q, m.y1, m.q), False)
        )

        # excluded from the list of active constraints;
        # state should remain unchanged
        self.assertFalse(m.eq7.active)
        assertExpressionsEqual(self, m.eq7.expr, m.z2 == 0)

        final_priority_dict = model_data.separation_priority_order
        self.assertEqual(len(final_priority_dict), 4)
        self.assertEqual(final_priority_dict["eq_con_eq3"], 1)
        self.assertEqual(final_priority_dict["eq_con_eq4"], 10)
        self.assertEqual(final_priority_dict["eq_con_eq5"], 3)
        self.assertEqual(final_priority_dict["eq_con_eq6"], DEFAULT_SEPARATION_PRIORITY)


class TestStandardizeActiveObjective(unittest.TestCase):
    """
    Test methods for standardization of the active objective.
    """

    def build_simple_test_model_data(self):
        """
        Build simple model for testing active objective
        standardization.
        """
        model_data = ModelData(original_model=None, timing=None, config=Bunch())
        model_data.working_model = ConcreteModel()
        model_data.working_model.user_model = m = Block()

        m.x = Var(initialize=1)
        m.z = Var(initialize=2)
        m.y = Var()

        m.p = Param(initialize=1, mutable=True)
        m.q = Param(initialize=1, mutable=True)
        m.q_cert = Param(initialize=1, mutable=True)

        m.obj1 = Objective(
            expr=(
                10
                + m.p
                + m.q
                + m.p * m.q_cert * m.x
                + m.z * m.p
                + m.y**2 * m.q
                + m.y
                + log(m.x)
            )
        )
        m.obj2 = Objective(expr=m.p + m.x * m.z + m.z**2)

        model_data.working_model.uncertain_params = [m.q, m.q_cert]
        model_data.working_model.effective_uncertain_params = [m.q]

        up = model_data.working_model.user_var_partitioning = Bunch()
        up.first_stage_variables = [m.x]
        up.second_stage_variables = [m.z]
        up.state_variables = [m.y]

        ep = model_data.working_model.effective_var_partitioning = Bunch()
        ep.first_stage_variables = [m.x, m.z]
        ep.second_stage_variables = []
        ep.state_variables = [m.y]

        model_data.working_model.first_stage = Block()
        model_data.working_model.first_stage.inequality_cons = Constraint(Any)
        model_data.working_model.second_stage = Block()
        model_data.working_model.second_stage.inequality_cons = Constraint(Any)

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
            10 + m.p + m.p * m.q_cert * m.x + log(m.x),
        )
        assertExpressionsEqual(
            self,
            working_model.second_stage_objective.expr,
            m.q + m.z * m.p + m.y**2 * m.q + m.y,
        )
        assertExpressionsEqual(self, working_model.full_objective.expr, m.obj1.expr)

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
            -10 - m.p - m.p * m.q_cert * m.x - log(m.x),
        )
        assertExpressionsEqual(
            self,
            working_model.second_stage_objective.expr,
            -m.q - m.z * m.p - m.y**2 * m.q - m.y,
        )
        assertExpressionsEqual(self, working_model.full_objective.expr, -m.obj1.expr)

    def test_standardize_active_obj_worst_case_focus(self):
        """
        Test preprocessing step for standardization
        of the active model objective.
        """
        model_data = self.build_simple_test_model_data()
        working_model = model_data.working_model
        m = model_data.working_model.user_model
        model_data.config.objective_focus = ObjectiveType.worst_case

        m.obj1.activate()
        m.obj2.deactivate()

        standardize_active_objective(model_data)

        self.assertFalse(
            m.obj1.active,
            msg=(
                f"Objective {m.obj1.name!r} should have been deactivated by "
                f"{standardize_active_objective}."
            ),
        )
        assertExpressionsEqual(
            self,
            working_model.second_stage.inequality_cons["epigraph_con"].expr,
            m.obj1.expr - working_model.first_stage.epigraph_var <= 0,
        )
        self.assertEqual(model_data.separation_priority_order["epigraph_con"], 0)

    def test_standardize_active_obj_nominal_focus(self):
        """
        Test standardization of active objective under nominal
        objective focus.
        """
        model_data = self.build_simple_test_model_data()
        working_model = model_data.working_model
        m = model_data.working_model.user_model
        model_data.config.objective_focus = ObjectiveType.nominal

        m.obj1.activate()
        m.obj2.deactivate()

        standardize_active_objective(model_data)

        self.assertFalse(
            m.obj1.active,
            msg=(
                f"Objective {m.obj1.name!r} should have been deactivated by "
                f"{standardize_active_objective}."
            ),
        )
        assertExpressionsEqual(
            self,
            working_model.first_stage.inequality_cons["epigraph_con"].expr,
            m.obj1.expr - working_model.first_stage.epigraph_var <= 0,
        )
        self.assertNotIn("epigraph_con", model_data.separation_priority_order)

    def test_standardize_active_obj_unsupported_focus(self):
        """
        Test standardization of active objective under
        an objective focus currently not supported
        """
        model_data = self.build_simple_test_model_data()
        m = model_data.working_model.user_model
        model_data.config.objective_focus = "bad_focus"

        m.obj1.activate()
        m.obj2.deactivate()

        exc_str = r"Classification.*not implemented for objective focus 'bad_focus'"
        with self.assertRaisesRegex(ValueError, exc_str):
            standardize_active_objective(model_data)

    def test_standardize_active_obj_nonadjustable_max(self):
        """
        Test standardize active objective for case in which
        the objective is independent of the nonadjustable variables
        and of a maximization sense.
        """
        model_data = self.build_simple_test_model_data()
        working_model = model_data.working_model
        m = working_model.user_model
        model_data.config.objective_focus = ObjectiveType.worst_case

        # assume all variables nonadjustable
        ep = model_data.working_model.effective_var_partitioning
        ep.first_stage_variables = [m.x, m.z]
        ep.second_stage_variables = []
        ep.state_variables = [m.y]

        m.obj1.deactivate()
        m.obj2.activate()
        m.obj2.sense = maximize

        standardize_active_objective(model_data)

        self.assertFalse(
            m.obj2.active,
            msg=(
                f"Objective {m.obj2.name!r} should have been deactivated by "
                f"{standardize_active_objective}."
            ),
        )

        assertExpressionsEqual(
            self,
            working_model.first_stage.inequality_cons["epigraph_con"].expr,
            -m.obj2.expr - working_model.first_stage.epigraph_var <= 0,
        )
        self.assertNotIn("epigraph_con", model_data.separation_priority_order)


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
        model_data.config = Bunch()
        model_data.working_model = ConcreteModel()
        model_data.working_model.user_model = m = Block()

        # uncertain parameters
        m.q = Param(range(4), initialize=0, mutable=True)

        # second-stage variables
        m.x = Var()
        m.z1 = Var([0, 1], initialize=0)
        m.z2 = Var()
        m.y = Var()

        model_data.working_model.uncertain_params = list(m.q.values())
        model_data.working_model.effective_uncertain_params = [m.q[0], m.q[1], m.q[2]]

        up = model_data.working_model.user_var_partitioning = Bunch()
        up.first_stage_variables = [m.x]
        up.second_stage_variables = [m.z1, m.z2]
        up.state_variables = [m.y]

        ep = model_data.working_model.effective_var_partitioning = Bunch()
        ep.first_stage_variables = [m.x, m.z1]
        ep.second_stage_variables = [m.z2]
        ep.state_variables = [m.y]

        model_data.working_model.first_stage = Block()

        return model_data

    def test_correct_num_dr_vars_static(self):
        """
        Test DR variable setup routines declare the correct
        number of DR coefficient variables, static DR case.
        """
        model_data = self.build_simple_test_model_data()
        model_data.config.decision_rule_order = 0

        add_decision_rule_variables(model_data)

        for indexed_dr_var in model_data.working_model.first_stage.decision_rule_vars:
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
            len(ComponentSet(model_data.working_model.first_stage.decision_rule_vars)),
            len(effective_second_stage_vars),
            msg=(
                "Number of unique indexed DR variable components should equal "
                "number of second-stage variables."
            ),
        )

        # check mapping is as expected
        ess_dr_var_zip = zip(
            effective_second_stage_vars,
            model_data.working_model.first_stage.decision_rule_vars,
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
                ),
            )

    def test_correct_num_dr_vars_affine(self):
        """
        Test DR variable setup routines declare the correct
        number of DR coefficient variables, affine DR case.
        """
        model_data = self.build_simple_test_model_data()
        model_data.config.decision_rule_order = 1

        add_decision_rule_variables(model_data)

        for indexed_dr_var in model_data.working_model.first_stage.decision_rule_vars:
            self.assertEqual(
                len(indexed_dr_var),
                1 + len(model_data.working_model.effective_uncertain_params),
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
            len(ComponentSet(model_data.working_model.first_stage.decision_rule_vars)),
            len(effective_second_stage_vars),
            msg=(
                "Number of unique indexed DR variable components should equal "
                "number of second-stage variables."
            ),
        )

        # check mapping is as expected
        ess_dr_var_zip = zip(
            effective_second_stage_vars,
            model_data.working_model.first_stage.decision_rule_vars,
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
                ),
            )

    def test_correct_num_dr_vars_quadratic(self):
        """
        Test DR variable setup routines declare the correct
        number of DR coefficient variables, quadratic DR case.
        """
        model_data = self.build_simple_test_model_data()
        model_data.config.decision_rule_order = 2

        add_decision_rule_variables(model_data)

        num_params = len(model_data.working_model.effective_uncertain_params)

        for indexed_dr_var in model_data.working_model.first_stage.decision_rule_vars:
            self.assertEqual(
                len(indexed_dr_var),
                1 + num_params  # static term  # affine terms
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
            len(ComponentSet(model_data.working_model.first_stage.decision_rule_vars)),
            len(effective_second_stage_vars),
            msg=(
                "Number of unique indexed DR variable components should equal "
                "number of second-stage variables."
            ),
        )

        # check mapping is as expected
        ess_dr_var_zip = zip(
            effective_second_stage_vars,
            model_data.working_model.first_stage.decision_rule_vars,
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
                ),
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
        model_data.config = Bunch()
        model_data.working_model = ConcreteModel()
        model_data.working_model.user_model = m = Block()

        # uncertain parameters
        m.q = Param(range(4), initialize=0, mutable=True)

        # second-stage variables
        m.x = Var()
        m.z1 = Var([0, 1], initialize=0)
        m.z2 = Var()
        m.y = Var()

        model_data.working_model.uncertain_params = list(m.q.values())
        model_data.working_model.effective_uncertain_params = [m.q[0], m.q[1], m.q[2]]

        up = model_data.working_model.user_var_partitioning = Bunch()
        up.first_stage_variables = [m.x]
        up.second_stage_variables = [m.z1, m.z2]
        up.state_variables = [m.y]

        ep = model_data.working_model.effective_var_partitioning = Bunch()
        ep.first_stage_variables = [m.x, m.z1]
        ep.second_stage_variables = [m.z2]
        ep.state_variables = [m.y]

        model_data.working_model.first_stage = Block()
        model_data.working_model.second_stage = Block()

        return model_data

    def test_num_dr_eqns_added_correct(self):
        """
        Check that number of DR equality constraints added
        by constraint declaration routines matches the number
        of second-stage variables in the model.
        """
        model_data = self.build_simple_test_model_data()
        model_data.config.decision_rule_order = 2

        add_decision_rule_variables(model_data)
        add_decision_rule_constraints(model_data)

        effective_second_stage_vars = (
            model_data.working_model.effective_var_partitioning.second_stage_variables
        )
        self.assertEqual(
            len(model_data.working_model.second_stage.decision_rule_eqns),
            len(effective_second_stage_vars),
            msg=(
                "Number of decision rule equations should match number of "
                "effective second-stage variables."
            ),
        )

        # check second-stage var to DR equation mapping is as expected
        ess_dr_var_zip = zip(
            effective_second_stage_vars,
            model_data.working_model.second_stage.decision_rule_eqns.values(),
        )
        for ess_var, dr_eqn in ess_dr_var_zip:
            mapped_dr_eqn = model_data.working_model.eff_ss_var_to_dr_eqn_map[ess_var]
            self.assertIs(
                mapped_dr_eqn,
                dr_eqn,
                msg=(
                    f"Second-stage var {ess_var.name!r} "
                    f"is mapped to DR equation {mapped_dr_eqn.name!r}, "
                    f"but expected mapping to DR equation {dr_eqn.name!r}."
                ),
            )
            self.assertTrue(mapped_dr_eqn.active)

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
        model_data.config.decision_rule_order = 2

        # add DR variables and constraints
        add_decision_rule_variables(model_data)
        add_decision_rule_constraints(model_data)

        dr_zip = zip(
            model_data.working_model.effective_var_partitioning.second_stage_variables,
            model_data.working_model.first_stage.decision_rule_vars,
            model_data.working_model.second_stage.decision_rule_eqns.values(),
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

            expected_dr_var_to_exponent_map = ComponentMap(
                (
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
                )
            )
            self.assertEqual(
                working_model.dr_var_to_exponent_map,
                expected_dr_var_to_exponent_map,
                msg="DR variable to exponent map not as expected.",
            )


class TestReformulateStateVarIndependentEqCons(unittest.TestCase):
    """
    Unit tests for routine that reformulates
    state variable-independent second-stage equality constraints.
    """

    def setup_test_model_data(self, uncertainty_set=None):
        """
        Set up simple test model for testing the reformulation
        routine.
        """
        model_data = ModelData(
            config=Bunch(
                uncertainty_set=uncertainty_set or BoxSet([[0, 1]]),
                separation_priority_order=dict(),
            ),
            original_model=None,
            timing=None,
        )
        model_data.working_model = working_model = ConcreteModel()
        model_data.working_model.user_model = m = Block()

        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.u = Param(initialize=1.125, mutable=True)
        m.u_cert = Param(initialize=1, mutable=True)
        m.con = Constraint(expr=m.u ** (0.5) * m.x1 - m.u * m.x2 <= 2)
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)
        m.eq_con = Constraint(
            expr=(
                m.u**2 * (m.x2 - 1)
                + m.u * (m.x1**3 + 0.5)
                - 5 * m.u * m.u_cert * m.x1 * m.x2
                == -m.u * (m.x1 + 2)
            )
        )

        # mathematically redundant, but makes the tests more rigorous
        # as we want to check that loops in the coefficient
        # matching routine are exited appropriately
        m.eq_con_2 = Constraint(expr=m.u * (m.x2 - 1) == 0)

        working_model.uncertain_params = [m.u, m.u_cert]
        working_model.effective_uncertain_params = [m.u]

        working_model.first_stage = Block()
        working_model.first_stage.dr_dependent_equality_cons = Constraint(Any)
        working_model.first_stage.dr_independent_equality_cons = Constraint(Any)
        working_model.second_stage = Block()
        working_model.second_stage.equality_cons = Constraint(Any)
        working_model.second_stage.inequality_cons = Constraint(Any)

        working_model.second_stage.equality_cons["eq_con"] = m.eq_con.expr
        working_model.second_stage.equality_cons["eq_con_2"] = m.eq_con_2.expr
        working_model.second_stage.inequality_cons["con"] = m.con.expr

        # mock separation priorities added during equality
        #  constraint standardization
        model_data.separation_priority_order["eq_con"] = DEFAULT_SEPARATION_PRIORITY
        model_data.separation_priority_order["eq_con_2"] = DEFAULT_SEPARATION_PRIORITY

        # deactivate constraints on user model, as these are not
        # what the reformulation routine actually processes
        m.eq_con.deactivate()
        m.eq_con_2.deactivate()
        m.con.deactivate()

        working_model.all_variables = [m.x1, m.x2]
        ep = working_model.effective_var_partitioning = Bunch()
        ep.first_stage_variables = [m.x1]
        ep.second_stage_variables = [m.x2]
        ep.state_variables = []

        return model_data

    def test_coefficient_matching_correct_constraints_added(self):
        """
        Test coefficient matching adds correct number of constraints
        in event of successful use.
        """
        model_data = self.setup_test_model_data()
        m = model_data.working_model.user_model

        # all vars first-stage
        ep = model_data.working_model.effective_var_partitioning
        ep.first_stage_variables = [m.x1, m.x2]
        ep.second_stage_variables = []

        model_data.config.decision_rule_order = 1
        model_data.config.progress_logger = logger

        model_data.working_model.first_stage.decision_rule_vars = []
        model_data.working_model.second_stage.decision_rule_eqns = []
        model_data.working_model.all_nonadjustable_variables = list(
            ep.first_stage_variables
        )

        robust_infeasible = reformulate_state_var_independent_eq_cons(model_data)

        self.assertFalse(
            robust_infeasible,
            msg=(
                "Coefficient matching unexpectedly detected"
                "a robust infeasible constraint"
            ),
        )

        fs_blk = model_data.working_model.first_stage
        dr_dependent_fs_eq_cons = fs_blk.dr_dependent_equality_cons
        dr_independent_fs_eq_cons = fs_blk.dr_independent_equality_cons
        self.assertFalse(dr_dependent_fs_eq_cons)
        self.assertEqual(
            len(dr_independent_fs_eq_cons),
            3,
            msg="Number of coefficient matching constraints not as expected.",
        )
        self.assertEqual(len(model_data.working_model.second_stage.equality_cons), 0)
        # we originally declared an inequality constraint on the model
        self.assertEqual(len(model_data.working_model.second_stage.inequality_cons), 1)

        assertExpressionsEqual(
            self,
            dr_independent_fs_eq_cons["coeff_matching_eq_con_coeff_1"].expr,
            m.x1**3 + 0.5 + 5 * m.x1 * m.x2 * (-1) + (-1) * (m.x1 + 2) * (-1) == 0,
        )
        assertExpressionsEqual(
            self,
            dr_independent_fs_eq_cons["coeff_matching_eq_con_coeff_2"].expr,
            m.x2 - 1 == 0,
        )
        assertExpressionsEqual(
            self,
            dr_independent_fs_eq_cons["coeff_matching_eq_con_2_coeff_1"].expr,
            m.x2 - 1 == 0,
        )

    def test_reformulate_nonlinear_state_var_independent_eq_con(self):
        """
        Test routine appropriately performs coefficient matching
        of polynomial-like constraints,
        and recasting of nonlinear constraints to opposing equalities.
        """
        model_data = self.setup_test_model_data()

        model_data.config.decision_rule_order = 1
        model_data.config.progress_logger = logging.getLogger(
            self.test_reformulate_nonlinear_state_var_independent_eq_con.__name__
        )
        model_data.config.progress_logger.setLevel(logging.DEBUG)

        add_decision_rule_variables(model_data)
        add_decision_rule_constraints(model_data)

        ep = model_data.working_model.effective_var_partitioning
        model_data.working_model.all_nonadjustable_variables = list(
            ep.first_stage_variables
            + list(model_data.working_model.first_stage.decision_rule_var_0.values())
        )

        wm = model_data.working_model
        m = model_data.working_model.user_model

        # we want only one of the constraints to be 'nonlinear'
        # change eq_con_2 to give a valid matching constraint
        wm.second_stage.equality_cons["eq_con_2"].set_value(m.u * (m.x1 - 1) == 0)

        with LoggingIntercept(level=logging.DEBUG) as LOG:
            robust_infeasible = reformulate_state_var_independent_eq_cons(model_data)

        err_msg = LOG.getvalue()
        self.assertRegex(
            text=err_msg,
            expected_regex=(r".*Equality constraint '.*eq_con.*'.*cannot be written.*"),
        )

        self.assertFalse(
            robust_infeasible,
            msg=(
                "Coefficient matching unexpectedly detected"
                "a robust infeasible constraint"
            ),
        )

        fs_blk = model_data.working_model.first_stage
        dr_independent_fs_eq_cons = fs_blk.dr_independent_equality_cons
        # check constraint partitioning updated as expected
        self.assertFalse(wm.second_stage.equality_cons)
        self.assertEqual(len(wm.second_stage.inequality_cons), 3)
        self.assertEqual(len(wm.first_stage.dr_independent_equality_cons), 1)
        self.assertFalse(wm.first_stage.dr_dependent_equality_cons)

        second_stage_ineq_cons = wm.second_stage.inequality_cons
        self.assertTrue(second_stage_ineq_cons["reform_lower_bound_from_eq_con"].active)
        self.assertTrue(second_stage_ineq_cons["reform_upper_bound_from_eq_con"].active)
        self.assertTrue(
            dr_independent_fs_eq_cons["coeff_matching_eq_con_2_coeff_1"].active
        )

        # expressions for the new opposing inequalities
        # and coefficient matching constraint
        assertExpressionsEqual(
            self,
            second_stage_ineq_cons["reform_lower_bound_from_eq_con"].expr,
            -(
                m.u**2 * (m.x2 - 1)
                + m.u * (m.x1**3 + 0.5)
                - ((5 * m.u * m.u_cert * m.x1) * m.x2)
                - (-m.u) * (m.x1 + 2)
            )
            <= 0.0,
        )
        assertExpressionsEqual(
            self,
            second_stage_ineq_cons["reform_upper_bound_from_eq_con"].expr,
            (
                m.u**2 * (m.x2 - 1)
                + m.u * (m.x1**3 + 0.5)
                - ((5 * m.u * m.u_cert * m.x1) * m.x2)
                - (-m.u) * (m.x1 + 2)
                <= 0.0
            ),
        )
        assertExpressionsEqual(
            self,
            dr_independent_fs_eq_cons["coeff_matching_eq_con_2_coeff_1"].expr,
            m.x1 - 1 == 0,
        )

        # separation priorities were also updated
        self.assertEqual(
            model_data.separation_priority_order["reform_lower_bound_from_eq_con"], 0
        )
        self.assertEqual(
            model_data.separation_priority_order["reform_upper_bound_from_eq_con"], 0
        )

    def test_reformulate_equality_cons_discrete_set(self):
        """
        Test routine for reformulating state-variable-independent
        second-stage equality constraints under scenario-based
        uncertainty works as expected.
        """
        model_data = self.setup_test_model_data(
            uncertainty_set=DiscreteScenarioSet([[0], [0.7]])
        )

        model_data.config.decision_rule_order = 1
        model_data.config.progress_logger = logging.getLogger(
            self.test_reformulate_nonlinear_state_var_independent_eq_con.__name__
        )
        model_data.config.progress_logger.setLevel(logging.DEBUG)

        add_decision_rule_variables(model_data)
        add_decision_rule_constraints(model_data)

        ep = model_data.working_model.effective_var_partitioning
        model_data.working_model.all_nonadjustable_variables = list(
            ep.first_stage_variables
            + list(model_data.working_model.first_stage.decision_rule_var_0.values())
        )

        wm = model_data.working_model
        m = model_data.working_model.user_model
        wm.second_stage.equality_cons["eq_con_2"].set_value(m.u * (m.x1 - 1) == 0)

        robust_infeasible = reformulate_state_var_independent_eq_cons(model_data)

        # check constraint partitioning updated as expected
        dr_dependent_equality_cons = wm.first_stage.dr_dependent_equality_cons
        dr_independent_equality_cons = wm.first_stage.dr_independent_equality_cons
        self.assertFalse(robust_infeasible)
        self.assertFalse(wm.second_stage.equality_cons)
        self.assertEqual(len(wm.second_stage.inequality_cons), 1)
        self.assertEqual(len(wm.first_stage.dr_dependent_equality_cons), 2)
        self.assertEqual(len(wm.first_stage.dr_independent_equality_cons), 2)

        self.assertTrue(dr_dependent_equality_cons["scenario_0_eq_con"].active)
        self.assertTrue(dr_dependent_equality_cons["scenario_1_eq_con"].active)
        self.assertTrue(dr_independent_equality_cons["scenario_0_eq_con_2"].active)
        self.assertTrue(dr_independent_equality_cons["scenario_1_eq_con_2"].active)

        # expressions for the new opposing inequalities
        # and coefficient matching constraint
        dr_vars = list(wm.first_stage.decision_rule_vars[0].values())
        assertExpressionsEqual(
            self,
            wm.first_stage.dr_dependent_equality_cons["scenario_0_eq_con"].expr,
            (
                0 * SumExpression([dr_vars[0] + 0 * dr_vars[1], -1])
                + 0 * (m.x1**3 + 0.5)
                - ((0 * m.u_cert * m.x1) * (dr_vars[0] + 0 * dr_vars[1]))
                == (0 * (m.x1 + 2))
            ),
        )
        assertExpressionsEqual(
            self,
            wm.first_stage.dr_dependent_equality_cons["scenario_1_eq_con"].expr,
            (
                (0.7**2) * SumExpression([dr_vars[0] + 0.7 * dr_vars[1], -1])
                + 0.7 * (m.x1**3 + 0.5)
                - ((5 * 0.7 * m.u_cert * m.x1) * (dr_vars[0] + 0.7 * dr_vars[1]))
                == (-0.7 * (m.x1 + 2))
            ),
        )
        assertExpressionsEqual(
            self,
            wm.first_stage.dr_independent_equality_cons["scenario_0_eq_con_2"].expr,
            0 * (m.x1 - 1) == 0,
        )
        assertExpressionsEqual(
            self,
            wm.first_stage.dr_independent_equality_cons["scenario_1_eq_con_2"].expr,
            0.7 * (m.x1 - 1) == 0,
        )

    def test_coefficient_matching_robust_infeasible_proof(self):
        """
        Test coefficient matching detects robust infeasibility
        as expected.
        """
        # Write the deterministic Pyomo model
        model_data = self.setup_test_model_data()
        m = model_data.working_model.user_model
        model_data.working_model.first_stage.decision_rule_vars = []
        model_data.working_model.second_stage.equality_cons["eq_con"].set_value(
            expr=m.u * (m.x1**3 + 0.5)
            - 5 * m.u * m.x1 * m.x2
            + m.u * (m.x1 + 2)
            + m.u**2
            == 0
        )
        ep = model_data.working_model.effective_var_partitioning
        ep.first_stage_variables = [m.x1, m.x2]
        ep.second_stage_variables = []

        model_data.config.decision_rule_order = 1
        model_data.config.progress_logger = logger

        model_data.working_model.all_nonadjustable_variables = list(
            ep.first_stage_variables
        )

        with LoggingIntercept(level=logging.INFO) as LOG:
            robust_infeasible = reformulate_state_var_independent_eq_cons(model_data)

        self.assertTrue(
            robust_infeasible,
            msg="Coefficient matching should be proven robust infeasible.",
        )
        robust_infeasible_msg = LOG.getvalue()
        self.assertRegex(
            text=robust_infeasible_msg,
            expected_regex=(
                r"PyROS has determined that the model is robust infeasible\. "
                r"One reason for this.*equality constraint '.*eq_con.*'.*"
            ),
        )


class TestPreprocessModelData(unittest.TestCase):
    """
    Test the PyROS preprocessor.
    """

    def build_test_model_data(self):
        """
        Build model data object for the preprocessor.
        """
        m = ConcreteModel()

        # PARAMS: p uncertain, q certain
        m.p = Param(initialize=2, mutable=True)
        m.q = Param(initialize=4.5, mutable=True)
        m.q_cert = Param(initialize=1, mutable=True)

        # first-stage variables
        m.x1 = Var(bounds=(0, m.q), initialize=1)
        m.x2 = Var(domain=NonNegativeReals, bounds=[m.p, m.p], initialize=m.p)

        # second-stage variables
        m.z1 = Var(domain=RangeSet(2, 4, 0), bounds=[-m.p, m.q], initialize=2)
        m.z2 = Var(bounds=(-2 * m.q**2, None), initialize=1)
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

        # Var representing uncertain parameter
        m.q2var = Var(initialize=3.2)

        # named Expression in terms of uncertain parameter
        # represented by a Var
        m.q2expr = Expression(expr=m.q2var * 10)

        # EQUALITY CONSTRAINTS
        # this will be reformulated by coefficient matching
        m.eq1 = Constraint(expr=m.q * (m.z3 + m.x2 * m.q_cert) == 0)
        # ranged constraints with identical bounds are considered equalities
        # this makes z1 nonadjustable
        m.eq2 = Constraint(expr=m.x1 - m.z1 == 0)
        # if q_cert is not effectively uncertain, then pretriangular:
        # makes z2 nonadjustable, so first-stage
        m.eq3 = Constraint(expr=m.x1**2 + m.x2 * m.q_cert + m.p * m.z2 == m.p)
        # second-stage equality
        m.eq4 = Constraint(expr=m.z3 + m.y1 + 5 * m.q2var == m.q)

        # duplicate of eq4: we will enforce this only nominally
        # through separation priorities
        m.eq5 = Constraint(expr=m.z3 + m.y1 + 5 * m.q2var == m.q)

        # INEQUALITY CONSTRAINTS
        # since x1, z1 nonadjustable, LB is first-stage,
        # but UB second-stage due to uncertain param q
        m.ineq1 = Constraint(expr=(-m.p, m.x1 + m.z1, exp(m.q)))
        # two first-stage inequalities
        m.ineq2 = Constraint(expr=(0, m.x1 + m.x2, 10 * m.q_cert))
        # though the bounds are structurally equal, they are not
        # identical objects, so this constitutes
        # two second-stage inequalities
        # note: these inequalities redundant,
        # as collectively these constraints
        # are mathematically identical to eq4
        m.ineq3 = Constraint(expr=(2 * m.q, 2 * (m.z3 + m.y1), 2 * m.q))
        # second-stage inequality. trivially satisfied/infeasible,
        # since y2 is fixed
        m.ineq4 = Constraint(expr=-m.q <= m.y2**2 + log(m.y2))

        # out of scope: deactivated
        m.ineq5 = Constraint(expr=m.y3 <= m.q)
        m.ineq5.deactivate()

        # ineq constraint in which the only uncertain parameter
        # is represented by a Var. will be second-stage due
        # to the presence of the uncertain parameter
        m.ineq6 = Constraint(expr=-m.q2var <= m.x1)

        # will be a nominal-only constraint
        m.ineq7 = Constraint(expr=m.y3 <= 2 * m.q)

        # OBJECTIVE
        # contains a rich combination of first-stage and second-stage terms
        m.obj = Objective(
            expr=(
                m.p**2
                + 2 * m.p * m.q
                + log(m.x1) * m.q_cert
                + 2 * m.p * m.x1
                + m.q**2 * m.x1
                + m.p**3 * (m.z1 + m.z2 + m.y1)
                + m.z4
                + m.z5
                + m.q2expr
            )
        )

        model_data = ModelData(
            original_model=m,
            timing=None,
            config=Bunch(
                uncertainty_set=BoxSet([[4, 5], [3, 4], [1, 1]]),
                uncertain_params=[m.q, m.q2var, m.q_cert],
                nominal_uncertain_param_vals=[m.q.value, m.q2var.value, m.q_cert.value],
                separation_priority_order=dict(),
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
        config = model_data.config
        config.update(
            dict(
                objective_focus=ObjectiveType.worst_case,
                decision_rule_order=0,
                progress_logger=logger,
                separation_priority_order=dict(),
            )
        )
        preprocess_model_data(model_data, user_var_partitioning)
        ep = model_data.working_model.effective_var_partitioning
        ublk = model_data.working_model.user_model
        self.assertEqual(
            ComponentSet(ep.first_stage_variables),
            ComponentSet(
                [
                    # all second-stage variables are nonadjustable
                    # due to the DR
                    ublk.x1,
                    ublk.x2,
                    ublk.z1,
                    ublk.z2,
                    ublk.z3,
                    ublk.z4,
                    ublk.z5,
                    ublk.y2,
                ]
            ),
        )
        self.assertEqual(ep.second_stage_variables, [])
        self.assertEqual(ep.state_variables, [ublk.y1])

        working_model = model_data.working_model
        self.assertEqual(
            ComponentSet(working_model.all_nonadjustable_variables),
            ComponentSet(
                [ublk.x1, ublk.x2, ublk.z1, ublk.z2, ublk.z3, ublk.z4, ublk.z5, ublk.y2]
                + [working_model.first_stage.epigraph_var]
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
                + [working_model.first_stage.epigraph_var]
            ),
        )

    @parameterized.expand([["affine", 1], ["quadratic", 2]])
    def test_preprocessor_effective_var_partitioning_nonstatic_dr(self, name, dr_order):
        """
        Test preprocessor repartitions the variables
        as expected.
        """
        model_data, user_var_partitioning = self.build_test_model_data()
        config = model_data.config
        config.update(
            dict(
                objective_focus=ObjectiveType.worst_case,
                decision_rule_order=dr_order,
                progress_logger=logger,
                separation_priority_order=dict(),
            )
        )
        preprocess_model_data(model_data, user_var_partitioning)
        ep = model_data.working_model.effective_var_partitioning
        ublk = model_data.working_model.user_model
        self.assertEqual(
            ComponentSet(ep.first_stage_variables),
            ComponentSet([ublk.x1, ublk.x2, ublk.z1, ublk.z2, ublk.z4, ublk.y2]),
        )
        self.assertEqual(
            ComponentSet(ep.second_stage_variables), ComponentSet([ublk.z3, ublk.z5])
        )
        self.assertEqual(ComponentSet(ep.state_variables), ComponentSet([ublk.y1]))
        working_model = model_data.working_model
        self.assertEqual(
            ComponentSet(working_model.all_nonadjustable_variables),
            ComponentSet(
                [ublk.x1, ublk.x2, ublk.z1, ublk.z2, ublk.z4, ublk.y2]
                + [working_model.first_stage.epigraph_var]
                + list(working_model.first_stage.decision_rule_var_0.values())
                + list(working_model.first_stage.decision_rule_var_1.values())
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
                + [working_model.first_stage.epigraph_var]
                + list(working_model.first_stage.decision_rule_var_0.values())
                + list(working_model.first_stage.decision_rule_var_1.values())
            ),
        )

    @parameterized.expand(
        [
            ["affine_nominal", 1, "nominal"],
            ["affine_worst_case", 1, "worst_case"],
            # eq1 doesn't get reformulated in coefficient matching
            #  as the polynomial degree is too high
            ["quadratic_nominal", 2, "nominal"],
            ["quadratic_worst_case", 2, "worst_case"],
        ]
    )
    def test_preprocessor_constraint_partitioning_nonstatic_dr(
        self, name, dr_order, obj_focus
    ):
        """
        Test preprocessor partitions constraints as expected
        for nonstatic DR.
        """
        model_data, user_var_partitioning = self.build_test_model_data()
        model_data.config.update(
            dict(
                objective_focus=ObjectiveType[obj_focus],
                decision_rule_order=dr_order,
                progress_logger=logger,
                separation_priority_order=dict(ineq3=2, ineq4=3),
            )
        )
        om = model_data.original_model
        om.pyros_separation_priority = Suffix()
        om.pyros_separation_priority[om.eq5] = None
        om.pyros_separation_priority[om.ineq4] = 5
        om.pyros_separation_priority[om.ineq7] = None

        preprocess_model_data(model_data, user_var_partitioning)

        working_model = model_data.working_model
        ublk = working_model.user_model

        # list of expected coefficient matching constraint names
        # equality bound constraint for z5 and/or eq1 are subject
        # to reformulation
        if dr_order == 1:
            coeff_matching_con_names = [
                "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_0",
                "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_1",
                "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_2",
                'coeff_matching_eq_con_eq1_coeff_1',
                'coeff_matching_eq_con_eq1_coeff_2',
                'coeff_matching_eq_con_eq1_coeff_3',
            ]
        else:
            coeff_matching_con_names = [
                "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_0",
                "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_1",
                "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_2",
                "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_3",
                "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_4",
                "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_5",
            ]

        self.assertEqual(
            list(working_model.first_stage.inequality_cons),
            (
                ["ineq_con_ineq1_lower_bound_con", "ineq_con_ineq2", "ineq_con_ineq7"]
                + (["epigraph_con"] if obj_focus == "nominal" else [])
            ),
        )
        self.assertEqual(
            list(working_model.first_stage.dr_independent_equality_cons),
            ["eq_con_eq2", "eq_con_eq3", "eq_con_eq5"],
        )
        self.assertEqual(
            list(working_model.first_stage.dr_dependent_equality_cons),
            coeff_matching_con_names,
        )
        self.assertEqual(
            list(working_model.second_stage.inequality_cons),
            (
                [
                    "var_x1_uncertain_upper_bound_con",
                    "var_z1_uncertain_upper_bound_con",
                    "var_z2_uncertain_lower_bound_con",
                    "var_z3_certain_upper_bound_con",
                    "var_z3_uncertain_lower_bound_con",
                    "var_z5_certain_lower_bound_con",
                    "var_y1_certain_lower_bound_con",
                    "ineq_con_ineq1_upper_bound_con",
                    "ineq_con_ineq3_lower_bound_con",
                    "ineq_con_ineq3_upper_bound_con",
                    "ineq_con_ineq4_lower_bound_con",
                    "ineq_con_ineq6_lower_bound_con",
                ]
                + (["epigraph_con"] if obj_focus == "worst_case" else [])
                + (
                    # for quadratic DR,
                    # eq1 gets reformulated to two inequality constraints
                    # since it is state variable independent and
                    # too nonlinear for coefficient matching
                    [
                        "reform_lower_bound_from_eq_con_eq1",
                        "reform_upper_bound_from_eq_con_eq1",
                    ]
                    if dr_order == 2
                    else []
                )
            ),
        )
        self.assertEqual(
            list(working_model.second_stage.equality_cons),
            # eq1 doesn't get reformulated in coefficient matching
            # when DR order is 2 as the polynomial degree is too high
            ["eq_con_eq4"],
        )

        # verify the constraints are active
        for fs_eq_con in get_all_first_stage_eq_cons(working_model):
            self.assertTrue(fs_eq_con.active, msg=f"{fs_eq_con.name} inactive")
        for fs_ineq_con in working_model.first_stage.inequality_cons.values():
            self.assertTrue(fs_ineq_con.active, msg=f"{fs_ineq_con.name} inactive")
        for perf_eq_con in working_model.second_stage.equality_cons.values():
            self.assertTrue(perf_eq_con.active, msg=f"{perf_eq_con.name} inactive")
        for perf_ineq_con in working_model.second_stage.inequality_cons.values():
            self.assertTrue(perf_ineq_con.active, msg=f"{perf_ineq_con.name} inactive")

        # verify the constraint expressions
        m = ublk
        fs = working_model.first_stage
        ss = working_model.second_stage
        assertExpressionsEqual(self, m.x1.lower, 0)
        assertExpressionsEqual(
            self,
            ss.inequality_cons["var_x1_uncertain_upper_bound_con"].expr,
            m.x1 <= m.q,
        )

        assertExpressionsEqual(
            self,
            ss.inequality_cons["var_z1_uncertain_upper_bound_con"].expr,
            m.z1 <= m.q,
        )
        assertExpressionsEqual(
            self,
            ss.inequality_cons["var_z2_uncertain_lower_bound_con"].expr,
            -m.z2 <= -(-2 * m.q**2),
        )
        assertExpressionsEqual(
            self,
            ss.inequality_cons["var_z3_uncertain_lower_bound_con"].expr,
            -m.z3 <= -(-m.q),
        )
        assertExpressionsEqual(
            self, ss.inequality_cons["var_z3_certain_upper_bound_con"].expr, m.z3 <= 0
        )
        assertExpressionsEqual(
            self, ss.inequality_cons["var_z5_certain_lower_bound_con"].expr, -m.z5 <= 0
        )
        assertExpressionsEqual(
            self, ss.inequality_cons["var_y1_certain_lower_bound_con"].expr, -m.y1 <= 0
        )
        assertExpressionsEqual(
            self,
            fs.inequality_cons["ineq_con_ineq1_lower_bound_con"].expr,
            -m.p <= m.x1 + m.z1,
        )
        assertExpressionsEqual(
            self,
            ss.inequality_cons["ineq_con_ineq1_upper_bound_con"].expr,
            m.x1 + m.z1 <= exp(m.q),
        )
        assertExpressionsEqual(
            self,
            fs.inequality_cons["ineq_con_ineq2"].expr,
            RangedExpression((0, m.x1 + m.x2, 10 * m.q_cert), False),
        )
        assertExpressionsEqual(
            self,
            ss.inequality_cons["ineq_con_ineq3_lower_bound_con"].expr,
            -(2 * (m.z3 + m.y1)) <= -(2 * m.q),
        )
        assertExpressionsEqual(
            self,
            ss.inequality_cons["ineq_con_ineq3_upper_bound_con"].expr,
            2 * (m.z3 + m.y1) <= 2 * m.q,
        )
        assertExpressionsEqual(
            self,
            ss.inequality_cons["ineq_con_ineq4_lower_bound_con"].expr,
            -(m.y2**2 + log(m.y2)) <= -(-m.q),
        )
        self.assertFalse(m.ineq5.active)
        assertExpressionsEqual(
            self,
            ss.inequality_cons["ineq_con_ineq6_lower_bound_con"].expr,
            -m.x1 <= -(-1 * working_model.temp_uncertain_params[1]),
        )
        assertExpressionsEqual(
            self, fs.inequality_cons["ineq_con_ineq7"].expr, m.y3 <= 2 * m.q
        )

        assertExpressionsEqual(
            self, fs.dr_independent_equality_cons["eq_con_eq2"].expr, m.x1 - m.z1 == 0
        )
        assertExpressionsEqual(
            self,
            fs.dr_independent_equality_cons["eq_con_eq3"].expr,
            m.x1**2 + m.x2 * m.q_cert + m.p * m.z2 == m.p,
        )
        if dr_order < 2:
            # due to coefficient matching, this should have been deleted
            self.assertNotIn("eq_con_eq1", ss.equality_cons)
        assertExpressionsEqual(
            self,
            fs.dr_independent_equality_cons["eq_con_eq5"].expr,
            m.z3 + m.y1 + 5 * working_model.temp_uncertain_params[1] == m.q,
        )

        # user model block should have no active constraints
        self.assertFalse(list(m.component_data_objects(Constraint, active=True)))

        # check separation priorities are as expected
        self.assertEqual(
            list(model_data.separation_priority_order.keys()),
            list(ss.inequality_cons.keys()),
        )
        final_priority_dict = model_data.separation_priority_order
        self.assertEqual(final_priority_dict["ineq_con_ineq3_lower_bound_con"], 2)
        self.assertEqual(final_priority_dict["ineq_con_ineq3_upper_bound_con"], 2)
        self.assertEqual(final_priority_dict["ineq_con_ineq4_lower_bound_con"], 5)
        for con_name, order in model_data.separation_priority_order.items():
            if "ineq3" not in con_name and "ineq4" not in con_name:
                self.assertEqual(
                    order,
                    DEFAULT_SEPARATION_PRIORITY,
                    msg=(
                        "Separation priority order for second-stage inequality "
                        f"{con_name!r} not as expected."
                    ),
                )
        self.assertFalse(ublk.pyros_separation_priority.active)

    @parameterized.expand(
        [["static", 0, True], ["affine", 1, False], ["quadratic", 2, False]]
    )
    def test_preprocessor_coefficient_matching(
        self, name, dr_order, expected_robust_infeas
    ):
        """
        Check preprocessor robust infeasibility return status.
        """
        model_data, user_var_partitioning = self.build_test_model_data()
        config = model_data.config
        config.update(
            dict(
                objective_focus=ObjectiveType.worst_case,
                decision_rule_order=dr_order,
                progress_logger=logger,
                separation_priority_order=dict(eq1=1),
            )
        )

        # for static DR, problem should be robust infeasible
        # due to the coefficient matching constraints derived
        # from bounds on z5
        robust_infeasible = preprocess_model_data(model_data, user_var_partitioning)
        # check the coefficient matching constraint expressions
        working_model = model_data.working_model
        m = model_data.working_model.user_model
        fs = working_model.first_stage
        dr_dependent_fs_eqs = working_model.first_stage.dr_dependent_equality_cons
        ss_ineqs = working_model.second_stage.inequality_cons

        self.assertIsInstance(robust_infeasible, bool)
        self.assertEqual(robust_infeasible, expected_robust_infeas)
        if not expected_robust_infeas:
            # all equality constraints were processed,
            # so only inequality constraint names should appear in the
            # priority dict
            self.assertEqual(
                list(model_data.separation_priority_order.keys()), list(ss_ineqs.keys())
            )

        if config.decision_rule_order == 1:
            # check the constraint expressions of eq1 and z5 bound
            assertExpressionsEqual(
                self,
                dr_dependent_fs_eqs[
                    "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_0"
                ].expr,
                fs.decision_rule_vars[1][0] == 0,
            )
            assertExpressionsEqual(
                self,
                dr_dependent_fs_eqs[
                    "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_1"
                ].expr,
                fs.decision_rule_vars[1][1] - 1 == 0,
            )
            assertExpressionsEqual(
                self,
                dr_dependent_fs_eqs[
                    "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_2"
                ].expr,
                fs.decision_rule_vars[1][2] == 0,
            )
            assertExpressionsEqual(
                self,
                dr_dependent_fs_eqs["coeff_matching_eq_con_eq1_coeff_1"].expr,
                # note: the certain parameter was eliminated by
                # the expression parser
                fs.decision_rule_vars[0][0] + m.x2 == 0,
            )
            assertExpressionsEqual(
                self,
                dr_dependent_fs_eqs["coeff_matching_eq_con_eq1_coeff_2"].expr,
                fs.decision_rule_vars[0][1] == 0,
            )
            assertExpressionsEqual(
                self,
                dr_dependent_fs_eqs["coeff_matching_eq_con_eq1_coeff_3"].expr,
                fs.decision_rule_vars[0][2] == 0,
            )
        if config.decision_rule_order == 2:
            # eq1 should be deactivated and refomulated to 2 inequalities
            assertExpressionsEqual(
                self,
                ss_ineqs["reform_lower_bound_from_eq_con_eq1"].expr,
                -(m.q * (m.z3 + m.x2 * m.q_cert)) <= 0.0,
            )
            assertExpressionsEqual(
                self,
                ss_ineqs["reform_upper_bound_from_eq_con_eq1"].expr,
                m.q * (m.z3 + m.x2 * m.q_cert) <= 0.0,
            )

            # check separation priority properly accounted for
            sep_priority_dict = model_data.separation_priority_order
            self.assertEqual(sep_priority_dict["reform_upper_bound_from_eq_con_eq1"], 1)
            self.assertEqual(sep_priority_dict["reform_lower_bound_from_eq_con_eq1"], 1)

            # check coefficient matching constraint expressions
            assertExpressionsEqual(
                self,
                dr_dependent_fs_eqs[
                    "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_0"
                ].expr,
                fs.decision_rule_vars[1][0] == 0,
            )
            assertExpressionsEqual(
                self,
                dr_dependent_fs_eqs[
                    "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_1"
                ].expr,
                fs.decision_rule_vars[1][1] - 1 == 0,
            )
            assertExpressionsEqual(
                self,
                dr_dependent_fs_eqs[
                    "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_2"
                ].expr,
                fs.decision_rule_vars[1][2] == 0,
            )
            assertExpressionsEqual(
                self,
                dr_dependent_fs_eqs[
                    "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_3"
                ].expr,
                fs.decision_rule_vars[1][3] == 0,
            )
            assertExpressionsEqual(
                self,
                dr_dependent_fs_eqs[
                    "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_4"
                ].expr,
                fs.decision_rule_vars[1][4] == 0,
            )
            assertExpressionsEqual(
                self,
                dr_dependent_fs_eqs[
                    "coeff_matching_var_z5_uncertain_eq_bound_con_coeff_5"
                ].expr,
                fs.decision_rule_vars[1][5] == 0,
            )

    @parameterized.expand([["static", 0], ["affine", 1], ["quadratic", 2]])
    def test_preprocessor_objective_standardization(self, name, dr_order):
        """
        Test preprocessor standardizes the active objective as
        expected.
        """
        model_data, user_var_partitioning = self.build_test_model_data()
        config = model_data.config
        config.update(
            dict(
                objective_focus=ObjectiveType.worst_case,
                decision_rule_order=dr_order,
                progress_logger=logger,
                separation_priority_order=dict(),
            )
        )
        preprocess_model_data(model_data, user_var_partitioning)

        ublk = model_data.working_model.user_model
        working_model = model_data.working_model

        assertExpressionsEqual(
            self,
            working_model.second_stage.inequality_cons["epigraph_con"].expr,
            ublk.obj.expr - working_model.first_stage.epigraph_var <= 0,
        )
        assertExpressionsEqual(self, working_model.full_objective.expr, ublk.obj.expr)

        # recall: objective summands are classified according
        # to dependence on uncertain parameters and variables
        # the *user* considers adjustable,
        # so the summands should be independent of the DR order
        # (which itself affects the effective var partitioning)
        assertExpressionsEqual(
            self,
            working_model.first_stage_objective.expr,
            ublk.p**2 + log(ublk.x1) * ublk.q_cert + 2 * ublk.p * ublk.x1,
        )
        assertExpressionsEqual(
            self,
            working_model.second_stage_objective.expr,
            (
                2 * ublk.p * ublk.q
                + ublk.q**2 * ublk.x1
                + ublk.p**3 * (ublk.z1 + ublk.z2 + ublk.y1)
                + ublk.z4
                + ublk.z5
                + ublk.q2expr
            ),
        )

    @parameterized.expand([["nominal"], ["worst_case"]])
    def test_preprocessor_sep_priorities_suffix_finder_none(self, obj_focus):
        """
        Test preprocessor resolves separation priorities as expected
        when an active separation priority Suffix component
        has a custom value mapped to `None`.
        """
        model_data, user_var_partitioning = self.build_test_model_data()
        config = model_data.config
        config.update(
            dict(
                objective_focus=ObjectiveType[obj_focus],
                decision_rule_order=1,
                progress_logger=logger,
                separation_priority_order=dict(),
            )
        )
        model_data.original_model.pyros_separation_priority = Suffix()
        model_data.original_model.pyros_separation_priority[None] = 10
        preprocess_model_data(model_data, user_var_partitioning)
        ss_ineq_cons = model_data.working_model.second_stage.inequality_cons
        self.assertEqual(
            list(model_data.separation_priority_order.keys()), list(ss_ineq_cons.keys())
        )
        for con_idx in ss_ineq_cons.keys():
            if con_idx != "epigraph_con":
                self.assertEqual(model_data.separation_priority_order[con_idx], 10)
            else:
                # custom prioritization of epigraph constraint is ignored
                self.assertEqual(
                    model_data.separation_priority_order[con_idx],
                    DEFAULT_SEPARATION_PRIORITY,
                )

    @parameterized.expand([["nominal"], ["worst_case"]])
    def test_preprocessor_log_model_statistics_affine_dr(self, obj_focus):
        """
        Test statistics of the preprocessed working model are
        logged as expected.
        """
        model_data, user_var_partitioning = self.build_test_model_data()
        config = model_data.config
        config.update(
            dict(
                objective_focus=ObjectiveType[obj_focus],
                decision_rule_order=1,
                progress_logger=logger,
                separation_priority_order=dict(eq5=None, ineq7=None),
            )
        )
        preprocess_model_data(model_data, user_var_partitioning)

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
              Number of uncertain parameters : 3 (2 eff.)
              Number of constraints : 28
                Equality constraints : 12
                  Coefficient matching constraints : 6
                  Other first-stage equations : 3
                  Second-stage equations : 1
                  Decision rule equations : 2
                Inequality constraints : 16
                  First-stage inequalities : {4 if obj_focus == 'nominal' else 3}
                  Second-stage inequalities : {12 if obj_focus == 'nominal' else 13}
            """
        )

        with LoggingIntercept(level=logging.INFO) as LOG:
            log_model_statistics(model_data)
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
        config = model_data.config
        config.update(
            dict(
                objective_focus=ObjectiveType[obj_focus],
                decision_rule_order=2,
                progress_logger=logger,
                separation_priority_order=dict(eq5=None, ineq7=None),
            )
        )
        preprocess_model_data(model_data, user_var_partitioning)

        # expected model stats worked out by hand
        expected_log_str = textwrap.dedent(
            f"""
            Model Statistics:
              Number of variables : 22
                Epigraph variable : 1
                First-stage variables : 2
                Second-stage variables : 5 (2 adj.)
                State variables : 2 (1 adj.)
                Decision rule variables : 12
              Number of uncertain parameters : 3 (2 eff.)
              Number of constraints : 30
                Equality constraints : 12
                  Coefficient matching constraints : 6
                  Other first-stage equations : 3
                  Second-stage equations : 1
                  Decision rule equations : 2
                Inequality constraints : 18
                  First-stage inequalities : {4 if obj_focus == 'nominal' else 3}
                  Second-stage inequalities : {14 if obj_focus == 'nominal' else 15}
            """
        )

        with LoggingIntercept(level=logging.INFO) as LOG:
            log_model_statistics(model_data)
        log_str = LOG.getvalue()

        log_lines = log_str.splitlines()[1:]
        expected_log_lines = expected_log_str.splitlines()[1:]

        self.assertEqual(len(log_lines), len(expected_log_lines))
        for line, expected_line in zip(log_lines, expected_log_lines):
            self.assertEqual(line, expected_line)


if __name__ == "__main__":
    unittest.main()
