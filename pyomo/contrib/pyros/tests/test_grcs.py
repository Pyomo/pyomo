'''
Unit tests for the grcs API
One class per function being tested, minimum one test per class
'''

import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.core.expr import identify_variables, identify_mutable_parameters
from pyomo.contrib.pyros.util import (
    selective_clone,
    add_decision_rule_variables,
    add_decision_rule_constraints,
    model_is_valid,
    turn_bounds_to_constraints,
    transform_to_standard_form,
    ObjectiveType,
    pyrosTerminationCondition,
    coefficient_matching,
    TimingData,
    IterationLogRecord,
)
from pyomo.contrib.pyros.util import replace_uncertain_bounds_with_constraints
from pyomo.contrib.pyros.util import get_vars_from_component
from pyomo.contrib.pyros.util import identify_objective_functions
from pyomo.common.collections import Bunch
import time
from pyomo.contrib.pyros.util import time_code
from pyomo.contrib.pyros.uncertainty_sets import *
from pyomo.contrib.pyros.master_problem_methods import (
    add_scenario_to_master,
    initial_construct_master,
    solve_master,
    minimize_dr_vars,
)
from pyomo.contrib.pyros.solve_data import MasterProblemData, ROSolveResults
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.environ import maximize as pyo_max
from pyomo.common.errors import ApplicationError
from pyomo.opt import (
    SolverResults,
    SolverStatus,
    SolutionStatus,
    TerminationCondition,
    Solution,
)
from pyomo.environ import (
    Constraint,
    Expression,
    Objective,
    Param,
    SolverFactory,
    Var,
    cos,
    exp,
    log,
    sin,
    sqrt,
    value,
)
import logging


logger = logging.getLogger(__name__)


if not (numpy_available and scipy_available):
    raise unittest.SkipTest('PyROS unit tests require numpy and scipy')

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

    def available(self):
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


# === util.py
class testSelectiveClone(unittest.TestCase):
    '''
    Testing for the selective_clone function. This function takes as input a Pyomo model object
    and a list of variables objects "first_stage_vars" in that Pyomo model which should *not* be cloned.
    It returns a clone of the original Pyomo model object wherein the "first_stage_vars" members are unchanged,
    i.e. all cloned model expressions still reference the "first_stage_vars" of the original model object.
    '''

    def test_cloning_negative_case(self):
        '''
        Testing correct behavior if incorrect first_stage_vars list object is passed to selective_clone
        '''
        m = ConcreteModel()
        m.x = Var(initialize=2)
        m.y = Var(initialize=2)
        m.p = Param(initialize=1)
        m.con = Constraint(expr=m.x * m.p + m.y <= 0)

        n = ConcreteModel()
        n.x = Var()
        m.first_stage_vars = [n.x]

        cloned_model = selective_clone(block=m, first_stage_vars=m.first_stage_vars)

        self.assertNotEqual(
            id(m.first_stage_vars),
            id(cloned_model.first_stage_vars),
            msg="First stage variables should not be equal.",
        )

    def test_cloning_positive_case(self):
        '''
        Testing if selective_clone works correctly for correct first_stage_var object definition.
        '''
        m = ConcreteModel()
        m.x = Var(initialize=2)
        m.y = Var(initialize=2)
        m.p = Param(initialize=1)
        m.con = Constraint(expr=m.x * m.p + m.y <= 0)
        m.first_stage_vars = [m.x]

        cloned_model = selective_clone(block=m, first_stage_vars=m.first_stage_vars)

        self.assertEqual(
            id(m.x), id(cloned_model.x), msg="First stage variables should be equal."
        )
        self.assertNotEqual(
            id(m.y),
            id(cloned_model.y),
            msg="Non-first-stage variables should not be equal.",
        )
        self.assertNotEqual(
            id(m.p), id(cloned_model.p), msg="Params should not be equal."
        )
        self.assertNotEqual(
            id(m.con),
            id(cloned_model.con),
            msg="Constraint objects should not be equal.",
        )


class testAddDecisionRuleVars(unittest.TestCase):
    '''
    Testing the method to add decision rule variables to a Pyomo model. This function should add decision rule
    variables to the list of first_stage_variables in a model object. The number of decision rule variables added
    depends on the number of control variables in the model and the number of uncertain parameters in the model.
    '''

    @unittest.skipIf(not scipy_available, 'Scipy is not available.')
    def test_add_decision_rule_vars_positive_case(self):
        '''
        Testing whether the correct number of decision rule variables is created in each DR type case
        '''
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.z1 = Var(initialize=0)
        m.z2 = Var(initialize=0)

        m.working_model = ConcreteModel()
        m.working_model.util = Block()

        m.working_model.util.second_stage_variables = [m.z1, m.z2]
        m.working_model.util.uncertain_params = [m.p1, m.p2]
        m.working_model.util.first_stage_variables = []

        m.working_model.util.first_stage_variables = []
        config = Block()
        config.decision_rule_order = 0

        add_decision_rule_variables(model_data=m, config=config)

        self.assertEqual(
            len(m.working_model.util.first_stage_variables),
            len(m.working_model.util.second_stage_variables),
            msg="For static approximation decision rule the number of decision rule variables"
            "added to the list of design variables should equal the number of control variables.",
        )

        m.working_model.util.first_stage_variables = []

        m.working_model.del_component(m.working_model.decision_rule_var_0)
        m.working_model.del_component(m.working_model.decision_rule_var_1)

        config.decision_rule_order = 1

        add_decision_rule_variables(m, config=config)

        self.assertEqual(
            len(m.working_model.util.first_stage_variables),
            len(m.working_model.util.second_stage_variables)
            * (1 + len(m.working_model.util.uncertain_params)),
            msg="For affine decision rule the number of decision rule variables add to the "
            "list of design variables should equal the number of control variables"
            "multiplied by the number of uncertain parameters plus 1.",
        )

        m.working_model.util.first_stage_variables = []

        m.working_model.del_component(m.working_model.decision_rule_var_0)
        m.working_model.del_component(m.working_model.decision_rule_var_1)
        m.working_model.del_component(m.working_model.decision_rule_var_0_index)
        m.working_model.del_component(m.working_model.decision_rule_var_1_index)

        config.decision_rule_order = 2

        add_decision_rule_variables(m, config=config)

        self.assertEqual(
            len(m.working_model.util.first_stage_variables),
            len(m.working_model.util.second_stage_variables)
            * int(
                2 * len(m.working_model.util.uncertain_params)
                + sp.special.comb(N=len(m.working_model.util.uncertain_params), k=2)
                + 1
            ),
            msg="For quadratic decision rule the number of decision rule variables add to the "
            "list of design variables should equal the number of control variables"
            "multiplied by 2 time the number of uncertain parameters plus all 2-combinations"
            "of uncertain parameters plus 1.",
        )


class testAddDecisionRuleConstraints(unittest.TestCase):
    '''
    Testing the addition of decision rule constraints functionally relating second-stage (control) variables to
    uncertain parameters and decision rule variables. This method should add constraints to the model object equal
    to the number of control variables. These constraints should reference the uncertain parameters and unique
    decision rule variables per control variable.
    '''

    def test_correct_number_of_decision_rule_constraints(self):
        '''
        Number of decision rule constraints added to the model should equal number of control variables in
        list "second_stage_variables".
        '''
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.z1 = Var(initialize=0)
        m.z2 = Var(initialize=0)

        m.working_model = ConcreteModel()
        m.working_model.util = Block()

        # === Decision rule vars have been added
        m.working_model.decision_rule_var_0 = Var(initialize=0)
        m.working_model.decision_rule_var_1 = Var(initialize=0)

        m.working_model.util.second_stage_variables = [m.z1, m.z2]
        m.working_model.util.uncertain_params = [m.p1, m.p2]

        decision_rule_cons = []
        config = Block()
        config.decision_rule_order = 0

        add_decision_rule_constraints(model_data=m, config=config)

        for c in m.working_model.component_data_objects(Constraint, descend_into=True):
            if "decision_rule_eqn_" in c.name:
                decision_rule_cons.append(c)
                m.working_model.del_component(c)

        self.assertEqual(
            len(decision_rule_cons),
            len(m.working_model.util.second_stage_variables),
            msg="The number of decision rule constraints added to model should equal"
            "the number of control variables in the model.",
        )

        decision_rule_cons = []
        config.decision_rule_order = 1

        # === Decision rule vars have been added
        m.working_model.del_component(m.working_model.decision_rule_var_0)
        m.working_model.del_component(m.working_model.decision_rule_var_1)

        m.working_model.decision_rule_var_0 = Var([0, 1, 2], initialize=0)
        m.working_model.decision_rule_var_1 = Var([0, 1, 2], initialize=0)

        add_decision_rule_constraints(model_data=m, config=config)

        for c in m.working_model.component_data_objects(Constraint, descend_into=True):
            if "decision_rule_eqn_" in c.name:
                decision_rule_cons.append(c)
                m.working_model.del_component(c)

        self.assertEqual(
            len(decision_rule_cons),
            len(m.working_model.util.second_stage_variables),
            msg="The number of decision rule constraints added to model should equal"
            "the number of control variables in the model.",
        )

        decision_rule_cons = []
        config.decision_rule_order = 2

        # === Decision rule vars have been added
        m.working_model.del_component(m.working_model.decision_rule_var_0)
        m.working_model.del_component(m.working_model.decision_rule_var_1)
        m.working_model.del_component(m.working_model.decision_rule_var_0_index)
        m.working_model.del_component(m.working_model.decision_rule_var_1_index)

        m.working_model.decision_rule_var_0 = Var([0, 1, 2, 3, 4, 5], initialize=0)
        m.working_model.decision_rule_var_1 = Var([0, 1, 2, 3, 4, 5], initialize=0)

        add_decision_rule_constraints(model_data=m, config=config)

        for c in m.working_model.component_data_objects(Constraint, descend_into=True):
            if "decision_rule_eqn_" in c.name:
                decision_rule_cons.append(c)
                m.working_model.del_component(c)

        self.assertEqual(
            len(decision_rule_cons),
            len(m.working_model.util.second_stage_variables),
            msg="The number of decision rule constraints added to model should equal"
            "the number of control variables in the model.",
        )


class testModelIsValid(unittest.TestCase):
    def test_model_is_valid_via_possible_inputs(self):
        m = ConcreteModel()
        m.x = Var()
        m.obj1 = Objective(expr=m.x**2)
        self.assertTrue(model_is_valid(m))
        m.obj2 = Objective(expr=m.x)
        self.assertFalse(model_is_valid(m))
        m.obj2.deactivate()
        self.assertTrue(model_is_valid(m))
        m.del_component("obj1")
        m.del_component("obj2")
        self.assertFalse(model_is_valid(m))


class testTurnBoundsToConstraints(unittest.TestCase):
    def test_bounds_to_constraints(self):
        m = ConcreteModel()
        m.x = Var(initialize=1, bounds=(0, 1))
        m.y = Var(initialize=0, bounds=(None, 1))
        m.w = Var(initialize=0, bounds=(1, None))
        m.z = Var(initialize=0, bounds=(None, None))
        turn_bounds_to_constraints(m.z, m)
        self.assertEqual(
            len(list(m.component_data_objects(Constraint))),
            0,
            msg="Inequality constraints were written for bounds on a variable with no bounds.",
        )
        turn_bounds_to_constraints(m.y, m)
        self.assertEqual(
            len(list(m.component_data_objects(Constraint))),
            1,
            msg="Inequality constraints were not "
            "written correctly for a variable with an upper bound and no lower bound.",
        )
        turn_bounds_to_constraints(m.w, m)
        self.assertEqual(
            len(list(m.component_data_objects(Constraint))),
            2,
            msg="Inequality constraints were not "
            "written correctly for a variable with a lower bound and no upper bound.",
        )
        turn_bounds_to_constraints(m.x, m)
        self.assertEqual(
            len(list(m.component_data_objects(Constraint))),
            4,
            msg="Inequality constraints were not "
            "written correctly for a variable with both lower and upper bound.",
        )

    def test_uncertain_bounds_to_constraints(self):
        # test model
        m = ConcreteModel()
        # parameters
        m.p = Param(initialize=8, mutable=True)
        m.r = Param(initialize=-5, mutable=True)
        m.q = Param(initialize=1, mutable=False)
        m.s = Param(initialize=1, mutable=True)
        m.n = Param(initialize=1, mutable=True)

        # variables, with bounds contingent on params
        m.u = Var(initialize=0, bounds=(0, m.p))
        m.v = Var(initialize=1, bounds=(m.r, m.p))
        m.w = Var(initialize=1, bounds=(None, None))
        m.x = Var(initialize=1, bounds=(0, exp(-1 * m.p / 8) * m.q * m.s))
        m.y = Var(initialize=-1, bounds=(m.r * m.p, 0))
        m.z = Var(initialize=1, bounds=(0, m.s))
        m.t = Var(initialize=1, bounds=(0, m.p**2))

        # objective
        m.obj = Objective(sense=maximize, expr=m.x**2 - m.y + m.t**2 + m.v)

        # clone model
        mod = m.clone()
        uncertain_params = [mod.n, mod.p, mod.r]

        # check variable replacement without any active objective
        # or active performance constraints
        mod.obj.deactivate()
        replace_uncertain_bounds_with_constraints(mod, uncertain_params)
        self.assertTrue(
            hasattr(mod, 'uncertain_var_bound_cons'),
            msg='Uncertain variable bounds erroneously added. '
            'Check only variables participating in active '
            'objective and constraints are added.',
        )
        self.assertFalse(mod.uncertain_var_bound_cons)
        mod.obj.activate()

        # add performance constraints
        constraints_m = ConstraintList()
        m.add_component('perf_constraints', constraints_m)
        constraints_m.add(m.w == 2 * m.x + m.y)
        constraints_m.add(m.v + m.x + m.y >= 0)
        constraints_m.add(m.y**2 + m.z >= 0)
        constraints_m.add(m.x**2 + m.u <= 1)
        constraints_m[4].deactivate()

        # clone model with constraints added
        mod_2 = m.clone()

        # manually replace uncertain parameter bounds with explicit constraints
        uncertain_cons = ConstraintList()
        m.add_component('uncertain_var_bound_cons', uncertain_cons)
        uncertain_cons.add(m.x - m.x.upper <= 0)
        uncertain_cons.add(m.y.lower - m.y <= 0)
        uncertain_cons.add(m.v - m.v._ub <= 0)
        uncertain_cons.add(m.v.lower - m.v <= 0)
        uncertain_cons.add(m.t - m.t.upper <= 0)

        # remove corresponding variable bounds
        m.x.setub(None)
        m.y.setlb(None)
        m.v.setlb(None)
        m.v.setub(None)
        m.t.setub(None)

        # check that vars participating in
        # active objective and activated constraints correctly determined
        svars_con = ComponentSet(get_vars_from_component(mod_2, Constraint))
        svars_obj = ComponentSet(get_vars_from_component(mod_2, Objective))
        vars_in_active_cons = ComponentSet(
            [mod_2.z, mod_2.w, mod_2.y, mod_2.x, mod_2.v]
        )
        vars_in_active_obj = ComponentSet([mod_2.x, mod_2.y, mod_2.t, mod_2.v])
        self.assertEqual(
            svars_con,
            vars_in_active_cons,
            msg='Mismatch of variables participating in activated constraints.',
        )
        self.assertEqual(
            svars_obj,
            vars_in_active_obj,
            msg='Mismatch of variables participating in activated objectives.',
        )

        # replace bounds in model with performance constraints
        uncertain_params = [mod_2.p, mod_2.r]
        replace_uncertain_bounds_with_constraints(mod_2, uncertain_params)

        # check that same number of constraints added to model
        self.assertEqual(
            len(list(m.component_data_objects(Constraint))),
            len(list(mod_2.component_data_objects(Constraint))),
            msg='Mismatch between number of explicit variable '
            'bound inequality constraints added '
            'automatically and added manually.',
        )

        # check that explicit constraints contain correct vars and params
        vars_in_cons = ComponentSet()
        params_in_cons = ComponentSet()

        # get variables, mutable params in the explicit constraints
        cons = mod_2.uncertain_var_bound_cons
        for idx in cons:
            for p in identify_mutable_parameters(cons[idx].expr):
                params_in_cons.add(p)
            for v in identify_variables(cons[idx].expr):
                vars_in_cons.add(v)
        # reduce only to uncertain mutable params found
        params_in_cons = params_in_cons & uncertain_params

        # expected participating variables
        vars_with_bounds_removed = ComponentSet([mod_2.x, mod_2.y, mod_2.v, mod_2.t])
        # complete the check
        self.assertEqual(
            params_in_cons,
            ComponentSet([mod_2.p, mod_2.r]),
            msg='Mismatch of parameters added to explicit inequality constraints.',
        )
        self.assertEqual(
            vars_in_cons,
            vars_with_bounds_removed,
            msg='Mismatch of variables added to explicit inequality constraints.',
        )


class testTransformToStandardForm(unittest.TestCase):
    def test_transform_to_std_form(self):
        """Check that `pyros.util.transform_to_standard_form` works
        correctly for an example model. That is:
        - all Constraints with a finite `upper` or `lower` attribute
          are either equality constraints, or inequalities
          of the standard form `expression(vars) <= upper`;
        - every inequality Constraint for which the `upper` and `lower`
          attribute are identical is converted to an equality constraint;
        - every inequality Constraint with distinct finite `upper` and
          `lower` attributes is split into two standard form inequality
          Constraints.
        """

        m = ConcreteModel()

        m.p = Param(initialize=1, mutable=True)

        m.x = Var(initialize=0)
        m.y = Var(initialize=1)
        m.z = Var(initialize=1)

        # example constraints
        m.c1 = Constraint(expr=m.x >= 1)
        m.c2 = Constraint(expr=-m.y <= 0)
        m.c3 = Constraint(rule=(None, m.x + m.y, None))
        m.c4 = Constraint(rule=(1, m.x + m.y, 2))
        m.c5 = Constraint(rule=(m.p, m.x, m.p))
        m.c6 = Constraint(rule=(1.0000, m.z, 1.0))

        # example ConstraintList
        clist = ConstraintList()
        m.add_component('clist', clist)
        clist.add(m.y <= 0)
        clist.add(m.x >= 1)
        clist.add((0, m.x, 1))

        num_orig_cons = len(
            [
                con
                for con in m.component_data_objects(
                    Constraint, active=True, descend_into=True
                )
            ]
        )
        # constraints with finite, distinct lower & upper bounds
        num_lbub_cons = len(
            [
                con
                for con in m.component_data_objects(
                    Constraint, active=True, descend_into=True
                )
                if con.lower is not None
                and con.upper is not None
                and con.lower is not con.upper
            ]
        )

        # count constraints with no bounds
        num_nobound_cons = len(
            [
                con
                for con in m.component_data_objects(
                    Constraint, active=True, descend_into=True
                )
                if con.lower is None and con.upper is None
            ]
        )

        transform_to_standard_form(m)
        cons = [
            con
            for con in m.component_data_objects(
                Constraint, active=True, descend_into=True
            )
        ]
        for con in cons:
            has_lb_or_ub = not (con.lower is None and con.upper is None)
            if has_lb_or_ub and not con.equality:
                self.assertTrue(
                    con.lower is None,
                    msg="Constraint %s not in standard form" % con.name,
                )
                lb_is_ub = con.lower is con.upper
                self.assertFalse(
                    lb_is_ub,
                    msg="Constraint %s should be converted to equality" % con.name,
                )
            if con is not m.c3:
                self.assertTrue(
                    has_lb_or_ub,
                    msg="Constraint %s should have"
                    " a lower or upper bound" % con.name,
                )

        self.assertEqual(
            len(
                [
                    con
                    for con in m.component_data_objects(
                        Constraint, active=True, descend_into=True
                    )
                ]
            ),
            num_orig_cons + num_lbub_cons - num_nobound_cons,
            msg="Expected number of constraints after\n "
            "standardizing constraints not matched. "
            "Number of constraints after\n "
            "transformation"
            " should be (number constraints in original "
            "model) \n + (number of constraints with "
            "distinct finite lower and upper bounds).",
        )

    def test_transform_does_not_alter_num_of_constraints(self):
        """
        Check that if model does not contain any constraints
        for which both the `lower` and `upper` attributes are
        distinct and not None, then number of constraints remains the same
        after constraint standardization.
        Standard form for the purpose of PyROS is all inequality constraints
        as `g(.)<=0`.
        """
        m = ConcreteModel()
        m.x = Var(initialize=1, bounds=(0, 1))
        m.y = Var(initialize=0, bounds=(None, 1))
        m.con1 = Constraint(expr=m.x >= 1 + m.y)
        m.con2 = Constraint(expr=m.x**2 + m.y**2 >= 9)
        original_num_constraints = len(list(m.component_data_objects(Constraint)))
        transform_to_standard_form(m)
        final_num_constraints = len(list(m.component_data_objects(Constraint)))
        self.assertEqual(
            original_num_constraints,
            final_num_constraints,
            msg="Transform to standard form function led to a "
            "different number of constraints than in the original model.",
        )
        number_of_non_standard_form_inequalities = len(
            list(
                c for c in list(m.component_data_objects(Constraint)) if c.lower != None
            )
        )
        self.assertEqual(
            number_of_non_standard_form_inequalities,
            0,
            msg="All inequality constraints were not transformed to standard form.",
        )


# === UncertaintySets.py
# Mock abstract class
class myUncertaintySet(UncertaintySet):
    '''
    returns single Constraint representing the uncertainty set which is
    simply a linear combination of uncertain_params
    '''

    def set_as_constraint(self, uncertain_params, **kwargs):
        return Constraint(expr=sum(v for v in uncertain_params) <= 0)

    def point_in_set(self, uncertain_params, **kwargs):
        return True

    def geometry(self):
        self.geometry = Geometry.LINEAR

    def dim(self):
        self.dim = 1

    def parameter_bounds(self):
        return [(0, 1)]


class testAbstractUncertaintySetClass(unittest.TestCase):
    '''
    The UncertaintySet class has an abstract base class implementing set_as_constraint method, as well as a couple
    basic uncertainty sets (ellipsoidal, polyhedral). The set_as_constraint method must return a Constraint object
    which references the Param objects from the uncertain_params list in the original model object.
    '''

    def test_uncertainty_set_with_correct_params(self):
        '''
        Case in which the UncertaintySet is constructed using the uncertain_param objects from the model to
        which the uncertainty set constraint is being added.
        '''
        m = ConcreteModel()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = m.uncertain_params

        _set = myUncertaintySet()
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars
        )
        uncertain_params_in_expr = list(
            v
            for v in m.uncertain_param_vars
            if v in ComponentSet(identify_variables(expr=m.uncertainty_set_contr.expr))
        )

        self.assertEqual(
            [id(u) for u in uncertain_params_in_expr],
            [id(u) for u in m.uncertain_param_vars],
            msg="Uncertain param Var objects used to construct uncertainty set constraint must"
            "be the same uncertain param Var objects in the original model.",
        )

    def test_uncertainty_set_with_incorrect_params(self):
        '''
        Case in which the UncertaintySet is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        '''
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.uncertain_params = [m.p1, m.p2]

        _set = myUncertaintySet()
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_params
        )
        variables_in_constr = list(
            v
            for v in m.uncertain_params
            if v in ComponentSet(identify_variables(expr=m.uncertainty_set_contr.expr))
        )

        self.assertEqual(
            len(variables_in_constr),
            0,
            msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
            "variable expression.",
        )


class testEllipsoidalUncertaintySetClass(unittest.TestCase):
    """
    Unit tests for the EllipsoidalSet
    """

    def test_normal_construction_and_update(self):
        """
        Test EllipsoidalSet constructor and setter
        work normally when arguments are appropriate.
        """
        center = [0, 0]
        shape_matrix = [[1, 0], [0, 2]]
        scale = 2
        eset = EllipsoidalSet(center, shape_matrix, scale)
        np.testing.assert_allclose(
            center, eset.center, err_msg="EllipsoidalSet center not as expected"
        )
        np.testing.assert_allclose(
            shape_matrix,
            eset.shape_matrix,
            err_msg="EllipsoidalSet shape matrix not as expected",
        )
        np.testing.assert_allclose(
            scale, eset.scale, err_msg="EllipsoidalSet scale not as expected"
        )

        # check attributes update
        new_center = [-1, -3]
        new_shape_matrix = [[2, 1], [1, 3]]
        new_scale = 1

        eset.center = new_center
        eset.shape_matrix = new_shape_matrix
        eset.scale = new_scale

        np.testing.assert_allclose(
            new_center,
            eset.center,
            err_msg="EllipsoidalSet center update not as expected",
        )
        np.testing.assert_allclose(
            new_shape_matrix,
            eset.shape_matrix,
            err_msg="EllipsoidalSet shape matrix update not as expected",
        )
        np.testing.assert_allclose(
            new_scale, eset.scale, err_msg="EllipsoidalSet scale update not as expected"
        )

    def test_error_on_ellipsoidal_dim_change(self):
        """
        EllipsoidalSet dimension is considered immutable.
        Test ValueError raised when center size is not equal
        to set dimension.
        """
        invalid_center = [0, 0]
        shape_matrix = [[1, 0], [0, 1]]
        scale = 2

        eset = EllipsoidalSet([0, 0], shape_matrix, scale)

        exc_str = r"Attempting to set.*dimension 2 to value of dimension 3"

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            eset.center = [0, 0, 0]

    def test_error_on_neg_scale(self):
        """
        Test ValueError raised if scale attribute set to negative
        value.
        """
        center = [0, 0]
        shape_matrix = [[1, 0], [0, 2]]
        neg_scale = -1

        exc_str = r".*must be a non-negative real \(provided.*-1\)"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            EllipsoidalSet(center, shape_matrix, neg_scale)

        # construct a valid EllipsoidalSet
        eset = EllipsoidalSet(center, shape_matrix, scale=2)

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            eset.scale = neg_scale

    def test_error_on_shape_matrix_with_wrong_size(self):
        """
        Test error in event EllipsoidalSet shape matrix
        is not in accordance with set dimension.
        """
        center = [0, 0]
        invalid_shape_matrix = [[1, 0]]
        scale = 1

        exc_str = r".*must be a square matrix of size 2.*\(provided.*shape \(1, 2\)\)"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            EllipsoidalSet(center, invalid_shape_matrix, scale)

        # construct a valid EllipsoidalSet
        eset = EllipsoidalSet(center, [[1, 0], [0, 1]], scale)

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            eset.shape_matrix = invalid_shape_matrix

    def test_error_on_invalid_shape_matrix(self):
        """
        Test exceptional cases of invalid square shape matrix
        arguments
        """
        center = [0, 0]
        scale = 3

        # assert error on construction
        with self.assertRaisesRegex(
            ValueError,
            r"Shape matrix must be symmetric",
            msg="Asymmetric shape matrix test failed",
        ):
            EllipsoidalSet(center, [[1, 1], [0, 1]], scale)
        with self.assertRaises(
            np.linalg.LinAlgError, msg="Singular shape matrix test failed"
        ):
            EllipsoidalSet(center, [[0, 0], [0, 0]], scale)
        with self.assertRaisesRegex(
            ValueError,
            r"Non positive-definite.*",
            msg="Indefinite shape matrix test failed",
        ):
            EllipsoidalSet(center, [[1, 0], [0, -2]], scale)

        # construct a valid EllipsoidalSet
        eset = EllipsoidalSet(center, [[1, 0], [0, 2]], scale)

        # assert error on update
        with self.assertRaisesRegex(
            ValueError,
            r"Shape matrix must be symmetric",
            msg="Asymmetric shape matrix test failed",
        ):
            eset.shape_matrix = [[1, 1], [0, 1]]
        with self.assertRaises(
            np.linalg.LinAlgError, msg="Singular shape matrix test failed"
        ):
            eset.shape_matrix = [[0, 0], [0, 0]]
        with self.assertRaisesRegex(
            ValueError,
            r"Non positive-definite.*",
            msg="Indefinite shape matrix test failed",
        ):
            eset.shape_matrix = [[1, 0], [0, -2]]

    def test_uncertainty_set_with_correct_params(self):
        '''
        Case in which the UncertaintySet is constructed using the uncertain_param objects from the model to
        which the uncertainty set constraint is being added.
        '''
        m = ConcreteModel()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        cov = [[1, 0], [0, 1]]
        s = 1

        _set = EllipsoidalSet(center=[0, 0], shape_matrix=cov, scale=s)
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars
        )
        uncertain_params_in_expr = list(
            v
            for v in m.uncertain_param_vars.values()
            if v
            in ComponentSet(identify_variables(expr=m.uncertainty_set_contr[1].expr))
        )

        self.assertEqual(
            [id(u) for u in uncertain_params_in_expr],
            [id(u) for u in m.uncertain_param_vars.values()],
            msg="Uncertain param Var objects used to construct uncertainty set constraint must"
            " be the same uncertain param Var objects in the original model.",
        )

    def test_uncertainty_set_with_incorrect_params(self):
        '''
        Case in which the EllipsoidalSet is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        '''
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(
            range(len(m.uncertain_params)), initialize=0, mutable=True
        )
        cov = [[1, 0], [0, 1]]
        s = 1

        _set = EllipsoidalSet(center=[0, 0], shape_matrix=cov, scale=s)
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars
        )
        variables_in_constr = list(
            v
            for v in m.uncertain_params
            if v
            in ComponentSet(identify_variables(expr=m.uncertainty_set_contr[1].expr))
        )

        self.assertEqual(
            len(variables_in_constr),
            0,
            msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
            " variable expression.",
        )

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        cov = [[1, 0], [0, 1]]
        s = 1

        _set = EllipsoidalSet(center=[0, 0], shape_matrix=cov, scale=s)
        self.assertTrue(
            _set.point_in_set([0, 0]), msg="Point is not in the EllipsoidalSet."
        )

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0.5)
        cov = [[1, 0], [0, 1]]
        s = 1

        _set = EllipsoidalSet(center=[0, 0], shape_matrix=cov, scale=s)
        config = Block()
        config.uncertainty_set = _set

        EllipsoidalSet.add_bounds_on_uncertain_parameters(model=m, config=config)

        self.assertNotEqual(
            m.util.uncertain_param_vars[0].lb,
            None,
            "Bounds not added correctly for EllipsoidalSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[0].ub,
            None,
            "Bounds not added correctly for EllipsoidalSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[1].lb,
            None,
            "Bounds not added correctly for EllipsoidalSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[1].ub,
            None,
            "Bounds not added correctly for EllipsoidalSet",
        )

    def test_ellipsoidal_set_bounds(self):
        """Check `EllipsoidalSet` parameter bounds method correct."""
        cov = [[2, 1], [1, 2]]
        scales = [0.5, 2]
        mean = [1, 1]

        for scale in scales:
            ell = EllipsoidalSet(center=mean, shape_matrix=cov, scale=scale)
            bounds = ell.parameter_bounds
            actual_bounds = list()
            for idx, val in enumerate(mean):
                diff = (cov[idx][idx] * scale) ** 0.5
                actual_bounds.append((val - diff, val + diff))
            self.assertTrue(
                np.allclose(np.array(bounds), np.array(actual_bounds)),
                msg=(
                    f"EllipsoidalSet bounds {bounds} do not match their actual"
                    f" values {actual_bounds} (for scale {scale}"
                    f" and shape matrix {cov})."
                    " Check the `parameter_bounds`"
                    " method for the EllipsoidalSet."
                ),
            )


class testAxisAlignedEllipsoidalUncertaintySetClass(unittest.TestCase):
    """
    Unit tests for the AxisAlignedEllipsoidalSet.
    """

    def test_normal_construction_and_update(self):
        """
        Test AxisAlignedEllipsoidalSet constructor and setter
        work normally when bounds are appropriate.
        """
        center = [0, 0]
        half_lengths = [1, 3]
        aset = AxisAlignedEllipsoidalSet(center, half_lengths)
        np.testing.assert_allclose(
            center,
            aset.center,
            err_msg="AxisAlignedEllipsoidalSet center not as expected",
        )
        np.testing.assert_allclose(
            half_lengths,
            aset.half_lengths,
            err_msg="AxisAlignedEllipsoidalSet half-lengths not as expected",
        )

        # check attributes update
        new_center = [-1, -3]
        new_half_lengths = [0, 1]
        aset.center = new_center
        aset.half_lengths = new_half_lengths

        np.testing.assert_allclose(
            new_center,
            aset.center,
            err_msg="AxisAlignedEllipsoidalSet center update not as expected",
        )
        np.testing.assert_allclose(
            new_half_lengths,
            aset.half_lengths,
            err_msg=("AxisAlignedEllipsoidalSet half lengths update not as expected"),
        )

    def test_error_on_axis_aligned_dim_change(self):
        """
        AxisAlignedEllipsoidalSet dimension is considered immutable.
        Test ValueError raised when attempting to alter the
        box set dimension (i.e. number of rows of `bounds`).
        """
        center = [0, 0]
        half_lengths = [1, 3]
        aset = AxisAlignedEllipsoidalSet(center, half_lengths)

        exc_str = r"Attempting to set.*dimension 2 to value of dimension 3"
        with self.assertRaisesRegex(ValueError, exc_str):
            aset.center = [0, 0, 1]

        with self.assertRaisesRegex(ValueError, exc_str):
            aset.half_lengths = [0, 0, 1]

    def test_error_on_negative_axis_aligned_half_lengths(self):
        """
        Test ValueError if half lengths for AxisAlignedEllipsoidalSet
        contains a negative value.
        """
        center = [1, 1]
        invalid_half_lengths = [1, -1]
        exc_str = r"Entry -1 of.*'half_lengths' is negative.*"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            AxisAlignedEllipsoidalSet(center, invalid_half_lengths)

        # construct a valid axis-aligned ellipsoidal set
        aset = AxisAlignedEllipsoidalSet(center, [1, 0])

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            aset.half_lengths = invalid_half_lengths

    def test_uncertainty_set_with_correct_params(self):
        '''
        Case in which the UncertaintySet is constructed using the uncertain_param objects from the model to
        which the uncertainty set constraint is being added.
        '''
        m = ConcreteModel()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        _set = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars
        )
        uncertain_params_in_expr = list(
            v
            for v in m.uncertain_param_vars.values()
            if v
            in ComponentSet(identify_variables(expr=m.uncertainty_set_contr[1].expr))
        )

        self.assertEqual(
            [id(u) for u in uncertain_params_in_expr],
            [id(u) for u in m.uncertain_param_vars.values()],
            msg="Uncertain param Var objects used to construct uncertainty set constraint must"
            " be the same uncertain param Var objects in the original model.",
        )

    def test_uncertainty_set_with_incorrect_params(self):
        '''
        Case in which the set is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        '''
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(
            range(len(m.uncertain_params)), initialize=0, mutable=True
        )
        _set = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars
        )
        variables_in_constr = list(
            v
            for v in m.uncertain_params
            if v
            in ComponentSet(identify_variables(expr=m.uncertainty_set_contr[1].expr))
        )

        self.assertEqual(
            len(variables_in_constr),
            0,
            msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
            " variable expression.",
        )

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        _set = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
        self.assertTrue(
            _set.point_in_set([0, 0]),
            msg="Point is not in the AxisAlignedEllipsoidalSet.",
        )

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0.5)

        _set = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
        config = Block()
        config.uncertainty_set = _set

        AxisAlignedEllipsoidalSet.add_bounds_on_uncertain_parameters(
            model=m, config=config
        )

        self.assertNotEqual(
            m.util.uncertain_param_vars[0].lb,
            None,
            "Bounds not added correctly for AxisAlignedEllipsoidalSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[0].ub,
            None,
            "Bounds not added correctly for AxisAlignedEllipsoidalSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[1].lb,
            None,
            "Bounds not added correctly for AxisAlignedEllipsoidalSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[1].ub,
            None,
            "Bounds not added correctly for AxisAlignedEllipsoidalSet",
        )

    def test_set_with_zero_half_lengths(self):
        # construct ellipsoid
        half_lengths = [1, 0, 2, 0]
        center = [1, 1, 1, 1]
        ell = AxisAlignedEllipsoidalSet(center, half_lengths)

        # construct model
        m = ConcreteModel()
        m.v1 = Var()
        m.v2 = Var([1, 2])
        m.v3 = Var()

        # test constraints
        conlist = ell.set_as_constraint([m.v1, m.v2, m.v3])
        eq_cons = [con for con in conlist.values() if con.equality]

        self.assertEqual(
            len(conlist),
            3,
            msg=(
                "Constraint list for this `AxisAlignedEllipsoidalSet` should"
                f" be of length 3, but is of length {len(conlist)}"
            ),
        )
        self.assertEqual(
            len(eq_cons),
            2,
            msg=(
                "Number of equality constraints for this"
                "`AxisAlignedEllipsoidalSet` should be 2,"
                f" there are {len(eq_cons)} such constraints"
            ),
        )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_two_stg_mod_with_axis_aligned_set(self):
        """
        Test two-stage model with `AxisAlignedEllipsoidalSet`
        as the uncertainty set.
        """
        # define model
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u1 = Param(initialize=1.125, mutable=True)
        m.u2 = Param(initialize=1, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u1 ** (0.5) - m.x2 * m.u1 <= 2)
        m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u1 == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - m.u2) ** 2)

        # Define the uncertainty set
        # we take the parameter `u2` to be 'fixed'
        ellipsoid = AxisAlignedEllipsoidalSet(center=[1.125, 1], half_lengths=[1, 0])

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


class testPolyhedralUncertaintySetClass(unittest.TestCase):
    """
    Unit tests for the Polyhedral set.
    """

    def test_normal_construction_and_update(self):
        """
        Test PolyhedralSet constructor and attribute setters work
        appropriately.
        """
        lhs_coefficients_mat = [[1, 2, 3], [4, 5, 6]]
        rhs_vec = [1, 3]

        pset = PolyhedralSet(lhs_coefficients_mat, rhs_vec)

        # check attributes are as expected
        np.testing.assert_allclose(lhs_coefficients_mat, pset.coefficients_mat)
        np.testing.assert_allclose(rhs_vec, pset.rhs_vec)

        # update the set
        pset.coefficients_mat = [[1, 0, 1], [1, 1, 1.5]]
        pset.rhs_vec = [3, 4]

        # check updates work
        np.testing.assert_allclose([[1, 0, 1], [1, 1, 1.5]], pset.coefficients_mat)
        np.testing.assert_allclose([3, 4], pset.rhs_vec)

    def test_error_on_polyhedral_set_dim_change(self):
        """
        PolyhedralSet dimension (number columns of 'coefficients_mat')
        is considered immutable.
        Test ValueError raised if attempt made to change dimension.
        """
        # construct valid set
        pset = PolyhedralSet([[1, 2, 3], [4, 5, 6]], [1, 3])

        exc_str = (
            r".*must have 3 columns to match set dimension \(provided.*2 columns\)"
        )

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            pset.coefficients_mat = [[1, 2], [3, 4]]

    def test_error_on_inconsistent_rows(self):
        """
        Number of rows of budget membership mat is immutable.
        Similarly, size of rhs_vec is immutable.
        Check ValueError raised in event of attempted change.
        """
        coeffs_mat_exc_str = (
            r".*must have 2 rows to match shape of attribute 'rhs_vec' "
            r"\(provided.*3 rows\)"
        )
        rhs_vec_exc_str = (
            r".*must have 2 entries to match shape of attribute "
            r"'coefficients_mat' \(provided.*3 entries\)"
        )
        # assert error on construction
        with self.assertRaisesRegex(ValueError, rhs_vec_exc_str):
            PolyhedralSet([[1, 2], [3, 4]], rhs_vec=[1, 3, 3])

        # construct a valid polyhedral set
        # (2 x 2 coefficients, 2-vector for RHS)
        pset = PolyhedralSet([[1, 2], [3, 4]], rhs_vec=[1, 3])

        # assert error on update
        with self.assertRaisesRegex(ValueError, coeffs_mat_exc_str):
            # 3 x 2 matrix row mismatch
            pset.coefficients_mat = [[1, 2], [3, 4], [5, 6]]
        with self.assertRaisesRegex(ValueError, rhs_vec_exc_str):
            # 3-vector mismatches 2 rows
            pset.rhs_vec = [1, 3, 2]

    def test_error_on_empty_set(self):
        """
        Check ValueError raised if nonemptiness check performed
        at construction returns a negative result.
        """
        exc_str = r"PolyhedralSet.*is empty.*"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            PolyhedralSet([[1], [-1]], rhs_vec=[1, -3])

    def test_error_on_polyhedral_mat_all_zero_columns(self):
        """
        Test ValueError raised if budget membership mat
        has a column with all zeros.
        """
        invalid_col_mat = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
        rhs_vec = [1, 1, 2]

        exc_str = r".*all entries zero in columns at indexes: 0, 1.*"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            PolyhedralSet(invalid_col_mat, rhs_vec)

        # construct a valid budget set
        pset = PolyhedralSet([[1, 0, 1], [1, 1, 0], [1, 1, 1]], rhs_vec)

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            pset.coefficients_mat = invalid_col_mat

    def test_uncertainty_set_with_correct_params(self):
        '''
        Case in which the UncertaintySet is constructed using the uncertain_param objects from the model to
        which the uncertainty set constraint is being added.
        '''
        m = ConcreteModel()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        A = [[0, 1], [1, 0]]
        b = [0, 0]

        _set = PolyhedralSet(lhs_coefficients_mat=A, rhs_vec=b)
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars
        )
        uncertain_params_in_expr = ComponentSet()
        for con in m.uncertainty_set_contr.values():
            con_vars = ComponentSet(identify_variables(expr=con.expr))
            for v in m.uncertain_param_vars.values():
                if v in con_vars:
                    uncertain_params_in_expr.add(v)

        self.assertEqual(
            uncertain_params_in_expr,
            ComponentSet(m.uncertain_param_vars.values()),
            msg="Uncertain param Var objects used to construct uncertainty set constraint must"
            " be the same uncertain param Var objects in the original model.",
        )

    def test_uncertainty_set_with_incorrect_params(self):
        '''
        Case in which the PolyHedral is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        '''
        m = ConcreteModel()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(
            range(len(m.uncertain_params)), initialize=0, mutable=True
        )
        A = [[0, 1], [1, 0]]
        b = [0, 0]

        _set = PolyhedralSet(lhs_coefficients_mat=A, rhs_vec=b)
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars
        )
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            vars_in_expr.extend(
                v
                for v in m.uncertain_param_vars
                if v in ComponentSet(identify_variables(expr=con.expr))
            )

        self.assertEqual(
            len(vars_in_expr),
            0,
            msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
            " variable expression.",
        )

    def test_polyhedral_set_as_constraint(self):
        '''
        The set_as_constraint method must return an indexed uncertainty_set_constr
        which has as many elements at their are dimensions in A.
        '''

        A = [[1, 0], [0, 1]]
        b = [0, 0]

        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)

        polyhedral_set = PolyhedralSet(lhs_coefficients_mat=A, rhs_vec=b)
        m.uncertainty_set_constr = polyhedral_set.set_as_constraint(
            uncertain_params=[m.p1, m.p2]
        )

        self.assertEqual(
            len(A),
            len(m.uncertainty_set_constr.index_set()),
            msg="Polyhedral uncertainty set constraints must be as many as the"
            "number of rows in the matrix A.",
        )

    def test_point_in_set(self):
        A = [[1, 0], [0, 1]]
        b = [0, 0]

        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        polyhedral_set = PolyhedralSet(lhs_coefficients_mat=A, rhs_vec=b)
        self.assertTrue(
            polyhedral_set.point_in_set([0, 0]),
            msg="Point is not in the PolyhedralSet.",
        )

    @unittest.skipUnless(baron_available, "Global NLP solver is not available.")
    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0.5)

        A = [[1, 0], [0, 1]]
        b = [0, 0]

        polyhedral_set = PolyhedralSet(lhs_coefficients_mat=A, rhs_vec=b)
        config = Block()
        config.uncertainty_set = polyhedral_set
        config.global_solver = SolverFactory("baron")

        PolyhedralSet.add_bounds_on_uncertain_parameters(model=m, config=config)

        self.assertNotEqual(
            m.util.uncertain_param_vars[0].lb,
            None,
            "Bounds not added correctly for PolyhedralSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[0].ub,
            None,
            "Bounds not added correctly for PolyhedralSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[1].lb,
            None,
            "Bounds not added correctly for PolyhedralSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[1].ub,
            None,
            "Bounds not added correctly for PolyhedralSet",
        )


class testBudgetUncertaintySetClass(unittest.TestCase):
    '''
    Budget uncertainty sets.
    Required inputs are matrix budget_membership_mat, rhs_vec.
    '''

    def test_normal_budget_construction_and_update(self):
        """
        Test BudgetSet constructor and attribute setters work
        appropriately.
        """
        budget_mat = [[1, 0, 1], [0, 1, 0]]
        budget_rhs_vec = [1, 3]

        # check attributes are as expected
        buset = BudgetSet(budget_mat, budget_rhs_vec)

        np.testing.assert_allclose(budget_mat, buset.budget_membership_mat)
        np.testing.assert_allclose(budget_rhs_vec, buset.budget_rhs_vec)
        np.testing.assert_allclose(
            [[1, 0, 1], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            buset.coefficients_mat,
        )
        np.testing.assert_allclose([1, 3, 0, 0, 0], buset.rhs_vec)
        np.testing.assert_allclose(np.zeros(3), buset.origin)

        # update the set
        buset.budget_membership_mat = [[1, 1, 0], [0, 0, 1]]
        buset.budget_rhs_vec = [3, 4]

        # check updates work
        np.testing.assert_allclose([[1, 1, 0], [0, 0, 1]], buset.budget_membership_mat)
        np.testing.assert_allclose([3, 4], buset.budget_rhs_vec)
        np.testing.assert_allclose(
            [[1, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            buset.coefficients_mat,
        )
        np.testing.assert_allclose([3, 4, 0, 0, 0], buset.rhs_vec)

        # update origin
        buset.origin = [1, 0, -1.5]
        np.testing.assert_allclose([1, 0, -1.5], buset.origin)

    def test_error_on_budget_set_dim_change(self):
        """
        BudgetSet dimension is considered immutable.
        Test ValueError raised when attempting to alter the
        budget set dimension.
        """
        budget_mat = [[1, 0, 1], [0, 1, 0]]
        budget_rhs_vec = [1, 3]
        bu_set = BudgetSet(budget_mat, budget_rhs_vec)

        # error on budget incidence matrix update
        exc_str = (
            r".*must have 3 columns to match set dimension \(provided.*1 columns\)"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            bu_set.budget_membership_mat = [[1], [1]]

        # error on origin update
        exc_str = (
            r".*must have 3 entries to match set dimension \(provided.*4 entries\)"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            bu_set.origin = [1, 2, 1, 0]

    def test_error_on_budget_member_mat_row_change(self):
        """
        Number of rows of budget membership mat is immutable.
        Hence, size of budget_rhs_vec is also immutable.
        """
        budget_mat = [[1, 0, 1], [0, 1, 0]]
        budget_rhs_vec = [1, 3]
        bu_set = BudgetSet(budget_mat, budget_rhs_vec)

        exc_str = (
            r".*must have 2 rows to match shape of attribute 'budget_rhs_vec' "
            r"\(provided.*1 rows\)"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            bu_set.budget_membership_mat = [[1, 0, 1]]

        exc_str = (
            r".*must have 2 entries to match shape of attribute "
            r"'budget_membership_mat' \(provided.*1 entries\)"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            bu_set.budget_rhs_vec = [1]

    def test_error_on_neg_budget_rhs_vec_entry(self):
        """
        Test ValueError raised if budget RHS vec has entry
        with negative value entry.
        """
        budget_mat = [[1, 0, 1], [1, 1, 0]]
        neg_val_rhs_vec = [1, -1]

        exc_str = r"Entry -1 of.*'budget_rhs_vec' is negative*"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            BudgetSet(budget_mat, neg_val_rhs_vec)

        # construct a valid budget set
        buset = BudgetSet(budget_mat, [1, 1])

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            buset.budget_rhs_vec = neg_val_rhs_vec

    def test_error_on_non_bool_budget_mat_entry(self):
        """
        Test ValueError raised if budget membership mat has
        entry which is not a 0-1 value.
        """
        invalid_budget_mat = [[1, 0, 1], [1, 1, 0.1]]
        budget_rhs_vec = [1, 1]

        exc_str = r"Attempting.*entries.*not 0-1 values \(example: 0.1\).*"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            BudgetSet(invalid_budget_mat, budget_rhs_vec)

        # construct a valid budget set
        buset = BudgetSet([[1, 0, 1], [1, 1, 0]], budget_rhs_vec)

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            buset.budget_membership_mat = invalid_budget_mat

    def test_error_on_budget_mat_all_zero_rows(self):
        """
        Test ValueError raised if budget membership mat
        has a row with all zeros.
        """
        invalid_row_mat = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        budget_rhs_vec = [1, 1, 2]

        exc_str = r".*all entries zero in rows at indexes: 0, 2.*"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            BudgetSet(invalid_row_mat, budget_rhs_vec)

        # construct a valid budget set
        buset = BudgetSet([[1, 0, 1], [1, 1, 0], [1, 1, 1]], budget_rhs_vec)

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            buset.budget_membership_mat = invalid_row_mat

    def test_error_on_budget_mat_all_zero_columns(self):
        """
        Test ValueError raised if budget membership mat
        has a column with all zeros.
        """
        invalid_col_mat = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
        budget_rhs_vec = [1, 1, 2]

        exc_str = r".*all entries zero in columns at indexes: 0, 1.*"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            BudgetSet(invalid_col_mat, budget_rhs_vec)

        # construct a valid budget set
        buset = BudgetSet([[1, 0, 1], [1, 1, 0], [1, 1, 1]], budget_rhs_vec)

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            buset.budget_membership_mat = invalid_col_mat

    @unittest.skipUnless(
        SolverFactory("cbc").available(exception_flag=False),
        "LP solver CBC not available",
    )
    def test_budget_set_parameter_bounds_correct(self):
        """
        If LP solver is available, test parameter bounds method
        for factor model set is correct (check against
        results from an LP solver).
        """
        solver = SolverFactory("cbc")

        # construct budget set instances
        buset1 = BudgetSet(
            budget_membership_mat=[[1, 1], [0, 1]], rhs_vec=[2, 3], origin=None
        )
        buset2 = BudgetSet(
            budget_membership_mat=[[1, 0], [1, 1]], rhs_vec=[3, 2], origin=[1, 1]
        )

        # check parameter bounds matches LP results
        # exactly for each case
        for buset in [buset1, buset2]:
            param_bounds = buset.parameter_bounds
            lp_param_bounds = eval_parameter_bounds(buset, solver)

            self.assertTrue(
                np.allclose(param_bounds, lp_param_bounds),
                msg=(
                    "Parameter bounds not consistent with LP values for "
                    "BudgetSet with parameterization:\n"
                    f"budget_membership_mat={buset.budget_membership_mat},\n"
                    f"budget_rhs_vec={buset.budget_rhs_vec},\n"
                    f"origin={buset.origin}.\n"
                    f"({param_bounds} does not match {lp_param_bounds})"
                ),
            )

    def test_uncertainty_set_with_correct_params(self):
        '''
        Case in which the UncertaintySet is constructed using the uncertain_param objects from the model to
        which the uncertainty set constraint is being added.
        '''
        m = ConcreteModel()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        # Single budget
        budget_membership_mat = [[1 for i in range(len(m.uncertain_param_vars))]]
        rhs_vec = [
            0.1 * len(m.uncertain_param_vars)
            + sum(p.value for p in m.uncertain_param_vars.values())
        ]

        _set = BudgetSet(budget_membership_mat=budget_membership_mat, rhs_vec=rhs_vec)
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars
        )
        uncertain_params_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if v in ComponentSet(identify_variables(expr=con.expr)):
                    if id(v) not in list(id(u) for u in uncertain_params_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        uncertain_params_in_expr.append(v)

        self.assertEqual(
            [id(u) for u in uncertain_params_in_expr],
            [id(u) for u in m.uncertain_param_vars.values()],
            msg="Uncertain param Var objects used to construct uncertainty set constraint must"
            " be the same uncertain param Var objects in the original model.",
        )

    def test_uncertainty_set_with_incorrect_params(self):
        '''
        Case in which the BudgetSet is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        '''
        m = ConcreteModel()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(
            range(len(m.uncertain_params)), initialize=0, mutable=True
        )
        # Single budget
        budget_membership_mat = [[1 for i in range(len(m.uncertain_param_vars))]]
        rhs_vec = [
            0.1 * len(m.uncertain_param_vars)
            + sum(p.value for p in m.uncertain_param_vars.values())
        ]

        _set = BudgetSet(budget_membership_mat=budget_membership_mat, rhs_vec=rhs_vec)
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars
        )
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            vars_in_expr.extend(
                v
                for v in m.uncertain_param_vars.values()
                if v in ComponentSet(identify_variables(expr=con.expr))
            )

        self.assertEqual(
            len(vars_in_expr),
            0,
            msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
            " variable expression.",
        )

    def test_budget_set_as_constraint(self):
        '''
        The set_as_constraint method must return an indexed uncertainty_set_constr
        which has as many elements at their are dimensions in A.
        '''

        m = ConcreteModel()
        m.p1 = Var(initialize=1)
        m.p2 = Var(initialize=1)
        m.uncertain_params = [m.p1, m.p2]

        # Single budget
        budget_membership_mat = [[1 for i in range(len(m.uncertain_params))]]
        rhs_vec = [
            0.1 * len(m.uncertain_params) + sum(p.value for p in m.uncertain_params)
        ]

        budget_set = BudgetSet(
            budget_membership_mat=budget_membership_mat, rhs_vec=rhs_vec
        )
        m.uncertainty_set_constr = budget_set.set_as_constraint(
            uncertain_params=m.uncertain_params
        )

        self.assertEqual(
            len(budget_set.coefficients_mat),
            len(m.uncertainty_set_constr.index_set()),
            msg=(
                "Number of budget set constraints should be equal to the "
                "number of rows in the 'coefficients_mat' attribute"
            ),
        )

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)

        budget_membership_mat = [[1 for i in range(len(m.uncertain_params))]]
        rhs_vec = [
            0.1 * len(m.uncertain_params) + sum(p.value for p in m.uncertain_params)
        ]

        budget_set = BudgetSet(
            budget_membership_mat=budget_membership_mat, rhs_vec=rhs_vec
        )
        self.assertTrue(
            budget_set.point_in_set([0, 0]), msg="Point is not in the BudgetSet."
        )

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0.5)

        budget_membership_mat = [[1 for i in range(len(m.util.uncertain_param_vars))]]
        rhs_vec = [
            0.1 * len(m.util.uncertain_param_vars)
            + sum(value(p) for p in m.util.uncertain_param_vars.values())
        ]

        budget_set = BudgetSet(
            budget_membership_mat=budget_membership_mat, rhs_vec=rhs_vec
        )
        config = Block()
        config.uncertainty_set = budget_set

        BudgetSet.add_bounds_on_uncertain_parameters(model=m, config=config)

        self.assertNotEqual(
            m.util.uncertain_param_vars[0].lb,
            None,
            "Bounds not added correctly for BudgetSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[0].ub,
            None,
            "Bounds not added correctly for BudgetSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[1].lb,
            None,
            "Bounds not added correctly for BudgetSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[1].ub,
            None,
            "Bounds not added correctly for BudgetSet",
        )


class testCardinalityUncertaintySetClass(unittest.TestCase):
    '''
    Cardinality uncertainty sets. Required inputs are origin, positive_deviation, gamma.
    Because Cardinality adds cassi vars to model, must pass model to set_as_constraint()
    '''

    def test_normal_cardinality_construction_and_update(self):
        """
        Test CardinalitySet constructor and setter work normally
        when bounds are appropriate.
        """
        # valid inputs
        cset = CardinalitySet(origin=[0, 0], positive_deviation=[1, 3], gamma=2)

        # check attributes are as expected
        np.testing.assert_allclose(cset.origin, [0, 0])
        np.testing.assert_allclose(cset.positive_deviation, [1, 3])
        np.testing.assert_allclose(cset.gamma, 2)
        self.assertEqual(cset.dim, 2)

        # update the set
        cset.origin = [1, 2]
        cset.positive_deviation = [3, 0]
        cset.gamma = 0.5

        # check updates work
        np.testing.assert_allclose(cset.origin, [1, 2])
        np.testing.assert_allclose(cset.positive_deviation, [3, 0])
        np.testing.assert_allclose(cset.gamma, 0.5)

    def test_error_on_neg_positive_deviation(self):
        """
        Cardinality set positive deviation attribute should
        contain nonnegative numerical entries.

        Check ValueError raised if any negative entries provided.
        """
        origin = [0, 0]
        positive_deviation = [1, -2]  # invalid
        gamma = 2

        exc_str = r"Entry -2 of attribute 'positive_deviation' is negative value"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            cset = CardinalitySet(origin, positive_deviation, gamma)

        # construct a valid cardinality set
        cset = CardinalitySet(origin, [1, 1], gamma)

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            cset.positive_deviation = positive_deviation

    def test_error_on_invalid_gamma(self):
        """
        Cardinality set gamma attribute should be a float-like
        between 0 and the set dimension.

        Check ValueError raised if gamma attribute is set
        to an invalid value.
        """
        origin = [0, 0]
        positive_deviation = [1, 1]
        gamma = 3  # should be invalid

        exc_str = (
            r".*attribute 'gamma' must be a real number "
            r"between 0 and dimension 2 \(provided value 3\)"
        )

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            CardinalitySet(origin, positive_deviation, gamma)

        # construct a valid cardinality set
        cset = CardinalitySet(origin, positive_deviation, gamma=2)

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            cset.gamma = gamma

    def test_error_on_cardinality_set_dim_change(self):
        """
        Dimension is considered immutable.
        Test ValueError raised when attempting to alter the
        set dimension (i.e. number of entries of `origin`).
        """
        # construct a valid cardinality set
        cset = CardinalitySet(origin=[0, 0], positive_deviation=[1, 1], gamma=2)

        exc_str = r"Attempting to set.*dimension 2 to value of dimension 3"

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            cset.origin = [0, 0, 0]
        with self.assertRaisesRegex(ValueError, exc_str):
            cset.positive_deviation = [1, 1, 1]

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_uncertainty_set_with_correct_params(self):
        '''
        Case in which the UncertaintySet is constructed using the uncertain_param objects from the model to
        which the uncertainty set constraint is being added.
        '''
        m = ConcreteModel()
        m.util = Block()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)

        center = list(p.value for p in m.uncertain_param_vars.values())
        positive_deviation = list(0.3 for j in range(len(center)))
        gamma = np.ceil(len(m.uncertain_param_vars) / 2)

        _set = CardinalitySet(
            origin=center, positive_deviation=positive_deviation, gamma=gamma
        )
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars, model=m
        )
        uncertain_params_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if v in ComponentSet(identify_variables(expr=con.expr)):
                    if id(v) not in list(id(u) for u in uncertain_params_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        uncertain_params_in_expr.append(v)

        self.assertEqual(
            [id(u) for u in uncertain_params_in_expr],
            [id(u) for u in m.uncertain_param_vars.values()],
            msg="Uncertain param Var objects used to construct uncertainty set constraint must"
            " be the same uncertain param Var objects in the original model.",
        )

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_uncertainty_set_with_incorrect_params(self):
        '''
        Case in which the CardinalitySet is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        '''
        m = ConcreteModel()
        m.util = Block()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(
            range(len(m.uncertain_params)), initialize=0, mutable=True
        )

        center = list(p.value for p in m.uncertain_param_vars.values())
        positive_deviation = list(0.3 for j in range(len(center)))
        gamma = np.ceil(len(m.uncertain_param_vars) / 2)

        _set = CardinalitySet(
            origin=center, positive_deviation=positive_deviation, gamma=gamma
        )
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars, model=m
        )
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if id(v) in [id(u) for u in list(identify_variables(expr=con.expr))]:
                    if id(v) not in list(id(u) for u in vars_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        vars_in_expr.append(v)

        self.assertEqual(
            len(vars_in_expr),
            0,
            msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
            " variable expression.",
        )

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)

        center = list(p.value for p in m.uncertain_param_vars.values())
        positive_deviation = list(0.3 for j in range(len(center)))
        gamma = np.ceil(len(m.uncertain_param_vars) / 2)

        _set = CardinalitySet(
            origin=center, positive_deviation=positive_deviation, gamma=gamma
        )

        self.assertTrue(
            _set.point_in_set([0, 0]), msg="Point is not in the CardinalitySet."
        )

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0.5)

        center = list(p.value for p in m.util.uncertain_param_vars.values())
        positive_deviation = list(0.3 for j in range(len(center)))
        gamma = np.ceil(len(center) / 2)

        cardinality_set = CardinalitySet(
            origin=center, positive_deviation=positive_deviation, gamma=gamma
        )
        config = Block()
        config.uncertainty_set = cardinality_set

        CardinalitySet.add_bounds_on_uncertain_parameters(model=m, config=config)

        self.assertNotEqual(
            m.util.uncertain_param_vars[0].lb,
            None,
            "Bounds not added correctly for CardinalitySet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[0].ub,
            None,
            "Bounds not added correctly for CardinalitySet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[1].lb,
            None,
            "Bounds not added correctly for CardinalitySet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[1].ub,
            None,
            "Bounds not added correctly for CardinalitySet",
        )


def eval_parameter_bounds(uncertainty_set, solver):
    """
    Evaluate parameter bounds of uncertainty set by solving
    bounding problems (as opposed to via the `parameter_bounds`
    method).
    """
    bounding_mdl = uncertainty_set.bounding_model()

    param_bounds = []
    for idx, obj in bounding_mdl.param_var_objectives.items():
        # activate objective for corresponding dimension
        obj.activate()
        bounds = []

        # solve for lower bound, then upper bound
        # solve should be successful
        for sense in (minimize, maximize):
            obj.sense = sense
            solver.solve(bounding_mdl)
            bounds.append(value(obj))

        # add parameter bounds for current dimension
        param_bounds.append(tuple(bounds))

        # ensure sense is minimize when done, deactivate
        obj.sense = minimize
        obj.deactivate()

    return param_bounds


class testBoxUncertaintySetClass(unittest.TestCase):
    """
    Unit tests for the box uncertainty set (BoxSet).
    """

    def test_normal_construction_and_update(self):
        """
        Test BoxSet constructor and setter work normally
        when bounds are appropriate.
        """
        bounds = [[1, 2], [3, 4]]
        bset = BoxSet(bounds=bounds)
        np.testing.assert_allclose(
            bounds, bset.bounds, err_msg="BoxSet bounds not as expected"
        )

        # check bounds update
        new_bounds = [[3, 4], [5, 6]]
        bset.bounds = new_bounds
        np.testing.assert_allclose(
            new_bounds, bset.bounds, err_msg="BoxSet bounds not as expected"
        )

    def test_error_on_box_set_dim_change(self):
        """
        BoxSet dimension is considered immutable.
        Test ValueError raised when attempting to alter the
        box set dimension (i.e. number of rows of `bounds`).
        """
        bounds = [[1, 2], [3, 4]]
        bset = BoxSet(bounds=bounds)  # 2-dimensional set

        exc_str = r"Attempting to set.*dimension 2 to a value of dimension 3"
        with self.assertRaisesRegex(ValueError, exc_str):
            bset.bounds = [[1, 2], [3, 4], [5, 6]]

    def test_error_on_lb_exceeds_ub(self):
        """
        Test exception raised when an LB exceeds a UB.
        """
        bad_bounds = [[1, 2], [4, 3]]

        exc_str = r"Lower bound 4 exceeds upper bound 3"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            BoxSet(bad_bounds)

        # construct a valid box set
        bset = BoxSet([[1, 2], [3, 4]])

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            bset.bounds = bad_bounds

    def test_error_on_ragged_bounds_array(self):
        """
        Test ValueError raised on attempting to set BoxSet bounds
        to a ragged array.

        This test also validates `uncertainty_sets.is_ragged` for all
        pre-defined array-like attributes of all set-types, as the
        `is_ragged` method is used throughout.
        """
        # example ragged arrays
        ragged_arrays = (
            [[1, 2], 3],  # list and int in same sequence
            [[1, 2], [3, [4, 5]]],  # 2nd row ragged (list and int)
            [[1, 2], [3]],  # variable row lengths
        )

        # construct valid box set
        bset = BoxSet(bounds=[[1, 2], [3, 4]])

        # exception message should match this regex
        exc_str = r"Argument `bounds` should not be a ragged array-like.*"
        for ragged_arr in ragged_arrays:
            # assert error on construction
            with self.assertRaisesRegex(ValueError, exc_str):
                BoxSet(bounds=ragged_arr)

            # assert error on update
            with self.assertRaisesRegex(ValueError, exc_str):
                bset.bounds = ragged_arr

    def test_error_on_invalid_bounds_shape(self):
        """
        Test ValueError raised when attempting to set
        Box set bounds to array of incorrect shape
        (should be a 2-D array with 2 columns).
        """
        # 3d array
        three_d_arr = [[[1, 2], [3, 4], [5, 6]]]
        exc_str = (
            r"Argument `bounds` must be a 2-dimensional.*"
            r"\(detected 3 dimensions.*\)"
        )

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            BoxSet(three_d_arr)

        # construct valid box set
        bset = BoxSet([[1, 2], [3, 4], [5, 6]])

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            bset.bounds = three_d_arr

    def test_error_on_wrong_number_columns(self):
        """
        BoxSet bounds should be a 2D array-like with 2 columns.
        ValueError raised if number columns wrong
        """
        three_col_arr = [[1, 2, 3], [4, 5, 6]]
        exc_str = (
            r"Attribute 'bounds' should be of shape \(\.{3},2\), "
            r"but detected shape \(\.{3},3\)"
        )

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            BoxSet(three_col_arr)

        # construct a valid box set
        bset = BoxSet([[1, 2], [3, 4]])

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            bset.bounds = three_col_arr

    def test_error_on_empty_last_dimension(self):
        """
        Check ValueError raised when last dimension of BoxSet bounds is
        empty.
        """
        empty_2d_arr = [[], [], []]
        exc_str = (
            r"Last dimension of argument `bounds` must be non-empty "
            r"\(detected shape \(3, 0\)\)"
        )

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            BoxSet(bounds=empty_2d_arr)

        # create a valid box set
        bset = BoxSet([[1, 2], [3, 4]])

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            bset.bounds = empty_2d_arr

    def test_error_on_non_numeric_bounds(self):
        """
        Test that ValueError is raised if box set bounds
        are set to array-like with entries of a non-numeric
        type (such as int, float).
        """
        # invalid bounds (contains an entry type str)
        new_bounds = [[1, "test"], [3, 2]]

        exc_str = (
            r"Entry 'test' of the argument `bounds` "
            r"is not a valid numeric type \(provided type 'str'\)"
        )

        # assert error on construction
        with self.assertRaisesRegex(TypeError, exc_str):
            BoxSet(new_bounds)

        # construct a valid box set
        bset = BoxSet(bounds=[[1, 2], [3, 4]])

        # assert error on update
        with self.assertRaisesRegex(TypeError, exc_str):
            bset.bounds = new_bounds

    def test_error_on_bounds_with_nan_or_inf(self):
        """
        Box set bounds set to array-like with inf or nan.
        """
        # construct a valid box set
        bset = BoxSet(bounds=[[1, 2], [3, 4]])

        for val_str in ["inf", "nan"]:
            bad_bounds = [[1, float(val_str)], [2, 3]]
            exc_str = (
                fr"Entry '{val_str}' of the argument `bounds` "
                fr"is not a finite numeric value"
            )
            # assert error on construction
            with self.assertRaisesRegex(ValueError, exc_str):
                BoxSet(bad_bounds)

            # assert error on update
            with self.assertRaisesRegex(ValueError, exc_str):
                bset.bounds = bad_bounds

    def test_uncertainty_set_with_correct_params(self):
        '''
        Case in which the UncertaintySet is constructed using the uncertain_param objects from the model to
        which the uncertainty set constraint is being added.
        '''
        m = ConcreteModel()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        bounds = [(-1, 1), (-1, 1)]
        _set = BoxSet(bounds=bounds)
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars
        )
        uncertain_params_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if v in ComponentSet(identify_variables(expr=con.expr)):
                    if id(v) not in list(id(u) for u in uncertain_params_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        uncertain_params_in_expr.append(v)

        self.assertEqual(
            [id(u) for u in uncertain_params_in_expr],
            [id(u) for u in m.uncertain_param_vars.values()],
            msg="Uncertain param Var objects used to construct uncertainty set constraint must"
            " be the same uncertain param Var objects in the original model.",
        )

    def test_uncertainty_set_with_incorrect_params(self):
        '''
        Case in which the set is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        '''
        m = ConcreteModel()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(
            range(len(m.uncertain_params)), initialize=0, mutable=True
        )
        bounds = [(-1, 1), (-1, 1)]
        _set = BoxSet(bounds=bounds)
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars
        )
        vars_in_expr = []
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if id(v) in [id(u) for u in list(identify_variables(expr=con.expr))]:
                    if id(v) not in list(id(u) for u in vars_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        vars_in_expr.append(v)

        self.assertEqual(
            len(vars_in_expr),
            0,
            msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
            " variable expression.",
        )

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)

        bounds = [(-1, 1), (-1, 1)]
        _set = BoxSet(bounds=bounds)
        self.assertTrue(_set.point_in_set([0, 0]), msg="Point is not in the BoxSet.")

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0)

        bounds = [(-1, 1), (-1, 1)]
        box_set = BoxSet(bounds=bounds)
        config = Block()
        config.uncertainty_set = box_set

        BoxSet.add_bounds_on_uncertain_parameters(model=m, config=config)

        self.assertEqual(
            m.util.uncertain_param_vars[0].lb,
            -1,
            "Bounds not added correctly for BoxSet",
        )
        self.assertEqual(
            m.util.uncertain_param_vars[0].ub,
            1,
            "Bounds not added correctly for BoxSet",
        )
        self.assertEqual(
            m.util.uncertain_param_vars[1].lb,
            -1,
            "Bounds not added correctly for BoxSet",
        )
        self.assertEqual(
            m.util.uncertain_param_vars[1].ub,
            1,
            "Bounds not added correctly for BoxSet",
        )


class testDiscreteUncertaintySetClass(unittest.TestCase):
    '''
    Discrete uncertainty sets. Required inputis a scenarios list.
    '''

    def test_normal_discrete_set_construction_and_update(self):
        """
        Test DiscreteScenarioSet constructor and setter work normally
        when scenarios are appropriate.
        """
        scenarios = [[0, 0, 0], [1, 2, 3]]

        # normal construction should work
        dset = DiscreteScenarioSet(scenarios)

        # check scenarios added appropriately
        np.testing.assert_allclose(
            scenarios, dset.scenarios, err_msg="BoxSet bounds not as expected"
        )

        # check scenarios updated appropriately
        new_scenarios = [[0, 1, 2], [1, 2, 0], [3, 5, 4]]
        dset.scenarios = new_scenarios
        np.testing.assert_allclose(
            new_scenarios, dset.scenarios, err_msg="BoxSet bounds not as expected"
        )

    def test_error_on_discrete_set_dim_change(self):
        """
        Test ValueError raised when attempting to update
        DiscreteScenarioSet dimension.
        """
        scenarios = [[1, 2], [3, 4]]
        dset = DiscreteScenarioSet(scenarios)  # 2-dimensional set

        exc_str = (
            r".*must have 2 columns.* to match set dimension "
            r"\(provided.*with 3 columns\)"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            dset.scenarios = [[1, 2, 3], [4, 5, 6]]

    def test_uncertainty_set_with_correct_params(self):
        '''
        Case in which the UncertaintySet is constructed using the uncertain_param objects from the model to
        which the uncertainty set constraint is being added.
        '''
        m = ConcreteModel()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        scenarios = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]
        _set = DiscreteScenarioSet(scenarios=scenarios)
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars
        )
        uncertain_params_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if v in ComponentSet(identify_variables(expr=con.expr)):
                    if id(v) not in list(id(u) for u in uncertain_params_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        uncertain_params_in_expr.append(v)

        self.assertEqual(
            [id(u) for u in uncertain_params_in_expr],
            [id(u) for u in m.uncertain_param_vars.values()],
            msg="Uncertain param Var objects used to construct uncertainty set constraint must"
            " be the same uncertain param Var objects in the original model.",
        )

    def test_uncertainty_set_with_incorrect_params(self):
        '''
        Case in which the set is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        '''
        m = ConcreteModel()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(
            range(len(m.uncertain_params)), initialize=0, mutable=True
        )
        scenarios = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]
        _set = DiscreteScenarioSet(scenarios=scenarios)
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars
        )
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if id(v) in [id(u) for u in list(identify_variables(expr=con.expr))]:
                    if id(v) not in list(id(u) for u in vars_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        vars_in_expr.append(v)

        self.assertEqual(
            len(vars_in_expr),
            0,
            msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
            " variable expression.",
        )

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)

        scenarios = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]
        _set = DiscreteScenarioSet(scenarios=scenarios)
        self.assertTrue(
            _set.point_in_set([0, 0]), msg="Point is not in the DiscreteScenarioSet."
        )

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0)

        scenarios = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]
        _set = DiscreteScenarioSet(scenarios=scenarios)
        config = Block()
        config.uncertainty_set = _set

        DiscreteScenarioSet.add_bounds_on_uncertain_parameters(model=m, config=config)

        self.assertNotEqual(
            m.util.uncertain_param_vars[0].lb,
            None,
            "Bounds not added correctly for DiscreteScenarioSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[0].ub,
            None,
            "Bounds not added correctly for DiscreteScenarioSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[1].lb,
            None,
            "Bounds not added correctly for DiscreteScenarioSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[1].ub,
            None,
            "Bounds not added correctly for DiscreteScenarioSet",
        )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_two_stg_model_discrete_set_single_scenario(self):
        """
        Test two-stage model under discrete uncertainty with
        a single scenario.
        """
        m = ConcreteModel()

        # model params
        m.u1 = Param(initialize=1.125, mutable=True)
        m.u2 = Param(initialize=1, mutable=True)

        # model vars
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))

        # model constraints
        m.con1 = Constraint(expr=m.x1 * m.u1 ** (0.5) - m.x2 * m.u1 <= 2)
        m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u1 == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - m.u2) ** 2)

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
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 10))
        m.x2 = Var(bounds=(0, 10))
        m.u = Param(mutable=True, initialize=1.125)
        m.con = Constraint(expr=sqrt(m.u) * m.x1 - m.u * m.x2 <= 2)
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - m.u) ** 2)

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


class testFactorModelUncertaintySetClass(unittest.TestCase):
    '''
    FactorModelSet uncertainty sets. Required inputs are psi_matrix, number_of_factors, origin and beta.
    '''

    def test_normal_factor_model_construction_and_update(self):
        """
        Test FactorModelSet constructor and setter work normally
        when attribute values are appropriate.
        """
        # valid inputs
        fset = FactorModelSet(
            origin=[0, 0, 1],
            number_of_factors=2,
            psi_mat=[[1, 2], [0, 1], [1, 0]],
            beta=0.1,
        )

        # check attributes are as expected
        np.testing.assert_allclose(fset.origin, [0, 0, 1])
        np.testing.assert_allclose(fset.psi_mat, [[1, 2], [0, 1], [1, 0]])
        np.testing.assert_allclose(fset.number_of_factors, 2)
        np.testing.assert_allclose(fset.beta, 0.1)
        self.assertEqual(fset.dim, 3)

        # update the set
        fset.origin = [1, 1, 0]
        fset.psi_mat = [[1, 0], [0, 1], [1, 1]]
        fset.beta = 0.5

        # check updates work
        np.testing.assert_allclose(fset.origin, [1, 1, 0])
        np.testing.assert_allclose(fset.psi_mat, [[1, 0], [0, 1], [1, 1]])
        np.testing.assert_allclose(fset.beta, 0.5)

    def test_error_on_factor_model_set_dim_change(self):
        """
        Test ValueError raised when attempting to change FactorModelSet
        dimension (by changing number of entries in origin
        or number of rows of psi_mat).
        """
        origin = [0, 0, 0]
        number_of_factors = 2
        psi_mat = [[1, 0], [0, 1], [1, 1]]
        beta = 0.5

        # construct factor model set
        fset = FactorModelSet(origin, number_of_factors, psi_mat, beta)

        # assert error on psi mat update
        exc_str = (
            r"should be of shape \(3, 2\) to match.*dimensions "
            r"\(provided shape \(2, 2\)\)"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            fset.psi_mat = [[1, 0], [1, 2]]

        # assert error on origin update
        exc_str = r"Attempting.*factor model set of dimension 3 to value of dimension 2"
        with self.assertRaisesRegex(ValueError, exc_str):
            fset.origin = [1, 3]

    def test_error_on_invalid_number_of_factors(self):
        """
        Test ValueError raised if number of factors
        is negative int, or AttributeError
        if attempting to update (should be immutable).
        """
        exc_str = r".*'number_of_factors' must be a positive int \(provided value -1\)"
        with self.assertRaisesRegex(ValueError, exc_str):
            FactorModelSet(origin=[0], number_of_factors=-1, psi_mat=[[1, 1]], beta=0.1)

        fset = FactorModelSet(
            origin=[0], number_of_factors=2, psi_mat=[[1, 1]], beta=0.1
        )

        exc_str = r".*'number_of_factors' is immutable"
        with self.assertRaisesRegex(AttributeError, exc_str):
            fset.number_of_factors = 3

    def test_error_on_invalid_beta(self):
        """
        Test ValueError raised if beta is invalid (exceeds 1 or
        is negative)
        """
        origin = [0, 0, 0]
        number_of_factors = 2
        psi_mat = [[1, 0], [0, 1], [1, 1]]
        neg_beta = -0.5
        big_beta = 1.5

        # assert error on construction
        neg_exc_str = (
            r".*must be a real number between 0 and 1.*\(provided value -0.5\)"
        )
        big_exc_str = r".*must be a real number between 0 and 1.*\(provided value 1.5\)"
        with self.assertRaisesRegex(ValueError, neg_exc_str):
            FactorModelSet(origin, number_of_factors, psi_mat, neg_beta)
        with self.assertRaisesRegex(ValueError, big_exc_str):
            FactorModelSet(origin, number_of_factors, psi_mat, big_beta)

        # create a valid factor model set
        fset = FactorModelSet(origin, number_of_factors, psi_mat, 1)

        # assert error on update
        with self.assertRaisesRegex(ValueError, neg_exc_str):
            fset.beta = neg_beta
        with self.assertRaisesRegex(ValueError, big_exc_str):
            fset.beta = big_beta

    @unittest.skipUnless(
        SolverFactory("cbc").available(exception_flag=False),
        "LP solver CBC not available",
    )
    def test_factor_model_parameter_bounds_correct(self):
        """
        If LP solver is available, test parameter bounds method
        for factor model set is correct (check against
        results from an LP solver).
        """
        solver = SolverFactory("cbc")

        # four cases where prior parameter bounds
        # approximations were probably too tight
        fset1 = FactorModelSet(
            origin=[0, 0],
            number_of_factors=3,
            psi_mat=[[1, -1, 1], [1, 0.1, 1]],
            beta=1 / 6,
        )
        fset2 = FactorModelSet(
            origin=[0], number_of_factors=3, psi_mat=[[1, 6, 8]], beta=1 / 2
        )
        fset3 = FactorModelSet(
            origin=[1], number_of_factors=2, psi_mat=[[1, 2]], beta=1 / 4
        )
        fset4 = FactorModelSet(
            origin=[1], number_of_factors=3, psi_mat=[[-1, -6, -8]], beta=1 / 2
        )

        # check parameter bounds matches LP results
        # exactly for each case
        for fset in [fset1, fset2, fset3, fset4]:
            param_bounds = fset.parameter_bounds
            lp_param_bounds = eval_parameter_bounds(fset, solver)

            self.assertTrue(
                np.allclose(param_bounds, lp_param_bounds),
                msg=(
                    "Parameter bounds not consistent with LP values for "
                    "FactorModelSet with parameterization:\n"
                    f"F={fset.number_of_factors},\n"
                    f"beta={fset.beta},\n"
                    f"psi_mat={fset.psi_mat},\n"
                    f"origin={fset.origin}."
                ),
            )

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_uncertainty_set_with_correct_params(self):
        '''
        Case in which the UncertaintySet is constructed using the uncertain_param objects from the model to
        which the uncertainty set constraint is being added.
        '''
        m = ConcreteModel()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.util = Block()
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        F = 1
        psi_mat = np.zeros(shape=(len(m.uncertain_params), F))
        for i in range(len(psi_mat)):
            random_row_entries = list(np.random.uniform(low=0, high=0.2, size=F))
            for j in range(len(psi_mat[i])):
                psi_mat[i][j] = random_row_entries[j]
        _set = FactorModelSet(
            origin=[0, 0], psi_mat=psi_mat, number_of_factors=F, beta=1
        )
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars, model=m
        )
        uncertain_params_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if v in ComponentSet(identify_variables(expr=con.expr)):
                    if id(v) not in list(id(u) for u in uncertain_params_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        uncertain_params_in_expr.append(v)

        self.assertEqual(
            [id(u) for u in uncertain_params_in_expr],
            [id(u) for u in m.uncertain_param_vars.values()],
            msg="Uncertain param Var objects used to construct uncertainty set constraint must"
            " be the same uncertain param Var objects in the original model.",
        )

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_uncertainty_set_with_incorrect_params(self):
        '''
        Case in which the set is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        '''
        m = ConcreteModel()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.util = Block()
        m.uncertain_param_vars = Param(
            range(len(m.uncertain_params)), initialize=0, mutable=True
        )
        F = 1
        psi_mat = np.zeros(shape=(len(m.uncertain_params), F))
        for i in range(len(psi_mat)):
            random_row_entries = list(np.random.uniform(low=0, high=0.2, size=F))
            for j in range(len(psi_mat[i])):
                psi_mat[i][j] = random_row_entries[j]
        _set = FactorModelSet(
            origin=[0, 0], psi_mat=psi_mat, number_of_factors=F, beta=1
        )
        m.uncertainty_set_contr = _set.set_as_constraint(
            uncertain_params=m.uncertain_param_vars, model=m
        )
        vars_in_expr = []
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if id(v) in [id(u) for u in list(identify_variables(expr=con.expr))]:
                    if id(v) not in list(id(u) for u in vars_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        vars_in_expr.append(v)

        self.assertEqual(
            len(vars_in_expr),
            0,
            msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
            " variable expression.",
        )

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)

        F = 1
        psi_mat = np.zeros(shape=(len(m.uncertain_params), F))
        for i in range(len(psi_mat)):
            random_row_entries = list(np.random.uniform(low=0, high=0.2, size=F))
            for j in range(len(psi_mat[i])):
                psi_mat[i][j] = random_row_entries[j]
        _set = FactorModelSet(
            origin=[0, 0], psi_mat=psi_mat, number_of_factors=F, beta=1
        )
        self.assertTrue(
            _set.point_in_set([0, 0]), msg="Point is not in the FactorModelSet."
        )

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0)

        F = 1
        psi_mat = np.zeros(shape=(len(list(m.util.uncertain_param_vars.values())), F))
        for i in range(len(psi_mat)):
            random_row_entries = list(np.random.uniform(low=0, high=0.2, size=F))
            for j in range(len(psi_mat[i])):
                psi_mat[i][j] = random_row_entries[j]
        _set = FactorModelSet(
            origin=[0, 0], psi_mat=psi_mat, number_of_factors=F, beta=1
        )
        config = Block()
        config.uncertainty_set = _set

        FactorModelSet.add_bounds_on_uncertain_parameters(model=m, config=config)

        self.assertNotEqual(
            m.util.uncertain_param_vars[0].lb,
            None,
            "Bounds not added correctly for FactorModelSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[0].ub,
            None,
            "Bounds not added correctly for FactorModelSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[1].lb,
            None,
            "Bounds not added correctly for FactorModelSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[1].ub,
            None,
            "Bounds not added correctly for FactorModelSet",
        )


class testIntersectionSetClass(unittest.TestCase):
    """
    Unit tests for the IntersectionSet class.
    Required input is set objects to intersect,
    and set_as_constraint requires
    an NLP solver to confirm the intersection is not empty.
    """

    def test_normal_construction_and_update(self):
        """
        Test IntersectionSet constructor and setter
        work normally when arguments are appropriate.
        """
        bset = BoxSet(bounds=[[-1, 1], [-1, 1], [-1, 1]])
        aset = AxisAlignedEllipsoidalSet([0, 0, 0], [1, 1, 1])

        iset = IntersectionSet(box_set=bset, axis_aligned_set=aset)
        self.assertIn(
            bset,
            iset.all_sets,
            msg=(
                "IntersectionSet 'all_sets' attribute does not"
                "contain expected BoxSet"
            ),
        )
        self.assertIn(
            aset,
            iset.all_sets,
            msg=(
                "IntersectionSet 'all_sets' attribute does not"
                "contain expected AxisAlignedEllipsoidalSet"
            ),
        )

    def test_error_on_intersecting_wrong_dims(self):
        """
        Test ValueError raised if IntersectionSet sets
        are not of same dimension.
        """
        bset = BoxSet(bounds=[[-1, 1], [-1, 1]])
        aset = AxisAlignedEllipsoidalSet([0, 0], [2, 2])
        wrong_aset = AxisAlignedEllipsoidalSet([0, 0, 0], [1, 1, 1])

        exc_str = r".*of dimension 2, but attempting to add set of dimension 3"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            IntersectionSet(box_set=bset, axis_set=aset, wrong_set=wrong_aset)

        # construct a valid intersection set
        iset = IntersectionSet(box_set=bset, axis_set=aset)
        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            iset.all_sets.append(wrong_aset)

    def test_type_error_on_invalid_arg(self):
        """
        Test TypeError raised if an argument not of type
        UncertaintySet is passed to the IntersectionSet
        constructor or appended to 'all_sets'.
        """
        bset = BoxSet(bounds=[[-1, 1], [-1, 1]])
        aset = AxisAlignedEllipsoidalSet([0, 0], [2, 2])

        exc_str = (
            r"Entry '1' of the argument `all_sets` is not An `UncertaintySet` "
            r"object.*\(provided type 'int'\)"
        )

        # assert error on construction
        with self.assertRaisesRegex(TypeError, exc_str):
            IntersectionSet(box_set=bset, axis_set=aset, invalid_arg=1)

        # construct a valid intersection set
        iset = IntersectionSet(box_set=bset, axis_set=aset)

        # assert error on update
        with self.assertRaisesRegex(TypeError, exc_str):
            iset.all_sets.append(1)

    def test_error_on_intersection_dim_change(self):
        """
        IntersectionSet dimension is considered immutable.
        Test ValueError raised when attempting to set the
        constituent sets to a different dimension.
        """
        bset = BoxSet(bounds=[[-1, 1], [-1, 1]])
        aset = AxisAlignedEllipsoidalSet([0, 0], [2, 2])

        # construct the set
        iset = IntersectionSet(box_set=bset, axis_set=aset)

        exc_str = r"Attempting to set.*dimension 2 to a sequence.* of dimension 1"

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            # attempt to set to 1-dimensional sets
            iset.all_sets = [BoxSet([[1, 1]]), AxisAlignedEllipsoidalSet([0], [1])]

    def test_error_on_too_few_sets(self):
        """
        Check ValueError raised if too few sets are passed
        to the intersection set.
        """
        exc_str = r"Attempting.*minimum required length 2.*iterable of length 1"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            IntersectionSet(bset=BoxSet([[1, 2]]))

        # construct a valid intersection set
        iset = IntersectionSet(
            box_set=BoxSet([[1, 2]]), axis_set=AxisAlignedEllipsoidalSet([0], [1])
        )

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            # attempt to set to 1-dimensional sets
            iset.all_sets = [BoxSet([[1, 1]])]

    def test_intersection_uncertainty_set_list_behavior(self):
        """
        Test the 'all_sets' attribute of the IntersectionSet
        class behaves like a regular Python list.
        """
        iset = IntersectionSet(
            bset=BoxSet([[0, 2]]), aset=AxisAlignedEllipsoidalSet([0], [1])
        )

        # an UncertaintySetList of length 2.
        # should behave like a list of length 2
        all_sets = iset.all_sets

        # test append
        all_sets.append(BoxSet([[1, 2]]))
        del all_sets[2:]

        # test extend
        all_sets.extend([BoxSet([[1, 2]]), EllipsoidalSet([0], [[1]], 2)])
        del all_sets[2:]

        # index in range. Allow slicing as well
        # none of these should result in exception
        all_sets[0]
        all_sets[1]
        all_sets[100:]
        all_sets[0:2:20]
        all_sets[0:2:1]
        all_sets[-20:-1:2]

        # index out of range
        self.assertRaises(IndexError, lambda: all_sets[2])
        self.assertRaises(IndexError, lambda: all_sets[-3])

        # assert min length ValueError if attempting to clear
        # list to length less than 2
        with self.assertRaisesRegex(ValueError, r"Length.* must be at least 2"):
            all_sets[:] = all_sets[0]
        with self.assertRaisesRegex(ValueError, r"Length.* must be at least 2"):
            del all_sets[1]
        with self.assertRaisesRegex(ValueError, r"Length.* must be at least 2"):
            del all_sets[1:]
        with self.assertRaisesRegex(ValueError, r"Length.* must be at least 2"):
            del all_sets[:]
        with self.assertRaisesRegex(ValueError, r"Length.* must be at least 2"):
            all_sets.clear()
        with self.assertRaisesRegex(ValueError, r"Length.* must be at least 2"):
            all_sets[0:] = []

        # assignment out of range
        with self.assertRaisesRegex(IndexError, r"assignment index out of range"):
            all_sets[-3] = BoxSet([[1, 1.5]])
        with self.assertRaisesRegex(IndexError, r"assignment index out of range"):
            all_sets[2] = BoxSet([[1, 1.5]])

        # assigning to slices should work fine
        all_sets[3:] = [BoxSet([[1, 1.5]]), BoxSet([[1, 3]])]

    @unittest.skipUnless(
        SolverFactory('ipopt').available(exception_flag=False),
        "Local NLP solver is not available.",
    )
    def test_uncertainty_set_with_correct_params(self):
        '''
        Case in which the UncertaintySet is constructed using the uncertain_param objects from the model to
        which the uncertainty set constraint is being added.
        '''
        m = ConcreteModel()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        bounds = [(-1, 1), (-1, 1)]
        Q1 = BoxSet(bounds=bounds)
        Q2 = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
        Q = IntersectionSet(Q1=Q1, Q2=Q2)

        config = ConfigBlock()
        solver = SolverFactory("ipopt")
        config.declare("global_solver", ConfigValue(default=solver))

        m.uncertainty_set_contr = Q.set_as_constraint(
            uncertain_params=m.uncertain_param_vars, config=config
        )
        uncertain_params_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if v in ComponentSet(identify_variables(expr=con.expr)):
                    if id(v) not in list(id(u) for u in uncertain_params_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        uncertain_params_in_expr.append(v)

        self.assertEqual(
            [id(u) for u in uncertain_params_in_expr],
            [id(u) for u in m.uncertain_param_vars.values()],
            msg="Uncertain param Var objects used to construct uncertainty set constraint must"
            " be the same uncertain param Var objects in the original model.",
        )

    @unittest.skipUnless(
        SolverFactory('ipopt').available(exception_flag=False),
        "Local NLP solver is not available.",
    )
    def test_uncertainty_set_with_incorrect_params(self):
        '''
        Case in which the set is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        '''
        m = ConcreteModel()
        # At this stage, the separation problem has uncertain_params which are now Var objects
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(
            range(len(m.uncertain_params)), initialize=0, mutable=True
        )
        bounds = [(-1, 1), (-1, 1)]

        Q1 = BoxSet(bounds=bounds)
        Q2 = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
        Q = IntersectionSet(Q1=Q1, Q2=Q2)

        solver = SolverFactory("ipopt")
        config = ConfigBlock()
        config.declare("global_solver", ConfigValue(default=solver))

        m.uncertainty_set_contr = Q.set_as_constraint(
            uncertain_params=m.uncertain_param_vars, config=config
        )
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if id(v) in [id(u) for u in list(identify_variables(expr=con.expr))]:
                    if id(v) not in list(id(u) for u in vars_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        vars_in_expr.append(v)

        self.assertEqual(
            len(vars_in_expr),
            0,
            msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
            " variable expression.",
        )

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)

        bounds = [(-1, 1), (-1, 1)]
        Q1 = BoxSet(bounds=bounds)
        Q2 = BoxSet(bounds=[(-2, 1), (-1, 2)])
        Q = IntersectionSet(Q1=Q1, Q2=Q2)
        self.assertTrue(
            Q.point_in_set([0, 0]), msg="Point is not in the IntersectionSet."
        )

    @unittest.skipUnless(baron_available, "Global NLP solver is not available.")
    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0.5)

        bounds = [(-1, 1), (-1, 1)]
        Q1 = BoxSet(bounds=bounds)
        Q2 = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[5, 5])
        Q = IntersectionSet(Q1=Q1, Q2=Q2)
        config = Block()
        config.uncertainty_set = Q
        config.global_solver = SolverFactory("baron")

        IntersectionSet.add_bounds_on_uncertain_parameters(m, config)

        self.assertNotEqual(
            m.util.uncertain_param_vars[0].lb,
            None,
            "Bounds not added correctly for IntersectionSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[0].ub,
            None,
            "Bounds not added correctly for IntersectionSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[1].lb,
            None,
            "Bounds not added correctly for IntersectionSet",
        )
        self.assertNotEqual(
            m.util.uncertain_param_vars[1].ub,
            None,
            "Bounds not added correctly for IntersectionSet",
        )


# === master_problem_methods.py
class testInitialConstructMaster(unittest.TestCase):
    def test_initial_construct_master(self):
        model_data = MasterProblemData()
        model_data.timing = None
        model_data.working_model = ConcreteModel()
        master_data = initial_construct_master(model_data)
        self.assertTrue(
            hasattr(master_data, "master_model"),
            msg="Initial construction of master problem "
            "did not create a master problem ConcreteModel object.",
        )


class testAddScenarioToMaster(unittest.TestCase):
    def test_add_scenario_to_master(self):
        working_model = ConcreteModel()
        working_model.p = Param([1, 2], initialize=0, mutable=True)
        working_model.x = Var()
        model_data = MasterProblemData()
        model_data.working_model = working_model
        model_data.timing = None
        master_data = initial_construct_master(model_data)
        master_data.master_model.scenarios[0, 0].transfer_attributes_from(
            working_model.clone()
        )
        master_data.master_model.scenarios[0, 0].util = Block()
        master_data.master_model.scenarios[0, 0].util.first_stage_variables = [
            master_data.master_model.scenarios[0, 0].x
        ]
        master_data.master_model.scenarios[0, 0].util.uncertain_params = [
            master_data.master_model.scenarios[0, 0].p[1],
            master_data.master_model.scenarios[0, 0].p[2],
        ]
        add_scenario_to_master(master_data, violations=[1, 1])

        self.assertEqual(
            len(master_data.master_model.scenarios),
            2,
            msg="Scenario not added to master correctly. Expected 2 scenarios.",
        )


global_solver = "baron"


class testSolveMaster(unittest.TestCase):
    @unittest.skipUnless(baron_available, "Global NLP solver is not available.")
    def test_solve_master(self):
        working_model = m = ConcreteModel()
        m.x = Var(initialize=0.5, bounds=(0, 10))
        m.y = Var(initialize=1.0, bounds=(0, 5))
        m.z = Var(initialize=0, bounds=(None, None))
        m.p = Param(initialize=1, mutable=True)
        m.obj = Objective(expr=m.x)
        m.con = Constraint(expr=m.x + m.y + m.z <= 3)
        model_data = MasterProblemData()
        model_data.working_model = working_model
        model_data.timing = None
        model_data.iteration = 0
        master_data = initial_construct_master(model_data)
        master_data.master_model.scenarios[0, 0].transfer_attributes_from(
            working_model.clone()
        )
        master_data.master_model.scenarios[0, 0].util = Block()
        master_data.master_model.scenarios[0, 0].util.first_stage_variables = [
            master_data.master_model.scenarios[0, 0].x
        ]
        master_data.master_model.scenarios[0, 0].util.decision_rule_vars = []
        master_data.master_model.scenarios[0, 0].util.second_stage_variables = []
        master_data.master_model.scenarios[0, 0].util.uncertain_params = [
            master_data.master_model.scenarios[0, 0].p
        ]
        master_data.master_model.scenarios[0, 0].first_stage_objective = 0
        master_data.master_model.scenarios[0, 0].second_stage_objective = Expression(
            expr=master_data.master_model.scenarios[0, 0].x
        )
        master_data.iteration = 0
        master_data.timing = TimingData()

        box_set = BoxSet(bounds=[(0, 2)])
        solver = SolverFactory(global_solver)
        config = ConfigBlock()
        config.declare("backup_global_solvers", ConfigValue(default=[]))
        config.declare("backup_local_solvers", ConfigValue(default=[]))
        config.declare("solve_master_globally", ConfigValue(default=True))
        config.declare("global_solver", ConfigValue(default=solver))
        config.declare("tee", ConfigValue(default=False))
        config.declare("decision_rule_order", ConfigValue(default=1))
        config.declare("objective_focus", ConfigValue(default=ObjectiveType.worst_case))
        config.declare(
            "second_stage_variables",
            ConfigValue(
                default=master_data.master_model.scenarios[
                    0, 0
                ].util.second_stage_variables
            ),
        )
        config.declare("subproblem_file_directory", ConfigValue(default=None))
        config.declare("time_limit", ConfigValue(default=None))
        config.declare(
            "progress_logger", ConfigValue(default=logging.getLogger(__name__))
        )

        with time_code(master_data.timing, "main", is_main_timer=True):
            master_soln = solve_master(master_data, config)
            self.assertEqual(
                master_soln.termination_condition,
                TerminationCondition.optimal,
                msg=(
                    "Could not solve simple master problem with solve_master "
                    "function."
                ),
            )


# === regression test for the solver
class coefficientMatchingTests(unittest.TestCase):
    def test_coefficient_matching_correct_num_constraints_added(self):
        # Write the deterministic Pyomo model
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con = Constraint(expr=m.u ** (0.5) * m.x1 - m.u * m.x2 <= 2)
        m.eq_con = Constraint(
            expr=m.u**2 * (m.x2 - 1)
            + m.u * (m.x1**3 + 0.5)
            - 5 * m.u * m.x1 * m.x2
            + m.u * (m.x1 + 2)
            == 0
        )
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

        config = Block()
        config.uncertainty_set = Block()
        config.uncertainty_set.parameter_bounds = [(0.25, 2)]

        m.util = Block()
        m.util.first_stage_variables = [m.x1, m.x2]
        m.util.second_stage_variables = []
        m.util.uncertain_params = [m.u]

        config.decision_rule_order = 0

        m.util.h_x_q_constraints = ComponentSet()

        coeff_matching_success, robust_infeasible = coefficient_matching(
            m, m.eq_con, [m.u], config
        )

        self.assertEqual(
            coeff_matching_success, True, msg="Coefficient matching was unsuccessful."
        )
        self.assertEqual(
            robust_infeasible,
            False,
            msg="Coefficient matching detected a robust infeasible constraint (1 == 0).",
        )
        self.assertEqual(
            len(m.coefficient_matching_constraints),
            2,
            msg="Coefficient matching produced incorrect number of h(x,q)=0 constraints.",
        )

        config.decision_rule_order = 1
        model_data = Block()
        model_data.working_model = m

        m.util.first_stage_variables = [m.x1]
        m.util.second_stage_variables = [m.x2]

        add_decision_rule_variables(model_data=model_data, config=config)
        add_decision_rule_constraints(model_data=model_data, config=config)

        coeff_matching_success, robust_infeasible = coefficient_matching(
            m, m.eq_con, [m.u], config
        )
        self.assertEqual(
            coeff_matching_success,
            False,
            msg="Coefficient matching should have been "
            "unsuccessful for higher order polynomial expressions.",
        )
        self.assertEqual(
            robust_infeasible,
            False,
            msg="Coefficient matching is not successful, "
            "but should not be proven robust infeasible.",
        )

    def test_coefficient_matching_robust_infeasible_proof(self):
        # Write the deterministic Pyomo model
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con = Constraint(expr=m.u ** (0.5) * m.x1 - m.u * m.x2 <= 2)
        m.eq_con = Constraint(
            expr=m.u * (m.x1**3 + 0.5)
            - 5 * m.u * m.x1 * m.x2
            + m.u * (m.x1 + 2)
            + m.u**2
            == 0
        )
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

        config = Block()
        config.uncertainty_set = Block()
        config.uncertainty_set.parameter_bounds = [(0.25, 2)]

        m.util = Block()
        m.util.first_stage_variables = [m.x1, m.x2]
        m.util.second_stage_variables = []
        m.util.uncertain_params = [m.u]

        config.decision_rule_order = 0

        m.util.h_x_q_constraints = ComponentSet()

        coeff_matching_success, robust_infeasible = coefficient_matching(
            m, m.eq_con, [m.u], config
        )

        self.assertEqual(
            coeff_matching_success,
            False,
            msg="Coefficient matching should have been unsuccessful.",
        )
        self.assertEqual(
            robust_infeasible,
            True,
            msg="Coefficient matching should be proven robust infeasible.",
        )


# === regression test for the solver
@unittest.skipUnless(baron_available, "Global NLP solver is not available.")
class RegressionTest(unittest.TestCase):
    def regression_test_constant_drs(self):
        model = m = ConcreteModel()
        m.name = "s381"

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

    def regression_test_affine_drs(self):
        model = m = ConcreteModel()
        m.name = "s381"

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

    def regression_test_quad_drs(self):
        model = m = ConcreteModel()
        m.name = "s381"

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
    def test_minimize_dr_norm(self):
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.z1 = Var(initialize=0, bounds=(0, 1))
        m.z2 = Var(initialize=0, bounds=(0, 1))

        m.working_model = ConcreteModel()
        m.working_model.util = Block()

        m.working_model.util.second_stage_variables = [m.z1, m.z2]
        m.working_model.util.uncertain_params = [m.p1, m.p2]
        m.working_model.util.first_stage_variables = []
        m.working_model.util.state_vars = []

        m.working_model.util.first_stage_variables = []
        config = Bunch()
        config.decision_rule_order = 1
        config.objective_focus = ObjectiveType.nominal
        config.global_solver = SolverFactory('baron')
        config.uncertain_params = m.working_model.util.uncertain_params
        config.tee = False
        config.solve_master_globally = True
        config.time_limit = None
        config.progress_logger = logging.getLogger(__name__)

        add_decision_rule_variables(model_data=m, config=config)
        add_decision_rule_constraints(model_data=m, config=config)

        # === Make master_type model
        master = ConcreteModel()
        master.scenarios = Block(NonNegativeIntegers, NonNegativeIntegers)
        master.scenarios[0, 0].transfer_attributes_from(m.working_model.clone())
        master.scenarios[0, 0].first_stage_objective = 0
        master.scenarios[0, 0].second_stage_objective = Expression(
            expr=(master.scenarios[0, 0].util.second_stage_variables[0] - 1) ** 2
            + (master.scenarios[0, 0].util.second_stage_variables[1] - 1) ** 2
        )
        master.obj = Objective(expr=master.scenarios[0, 0].second_stage_objective)
        master_data = MasterProblemData()
        master_data.master_model = master
        master_data.master_model.const_efficiency_applied = False
        master_data.master_model.linear_efficiency_applied = False
        master_data.iteration = 0

        master_data.timing = TimingData()
        with time_code(master_data.timing, "main", is_main_timer=True):
            results, success = minimize_dr_vars(model_data=master_data, config=config)
            self.assertEqual(
                results.solver.termination_condition,
                TerminationCondition.optimal,
                msg="Minimize dr norm did not solve to optimality.",
            )
            self.assertTrue(
                success, msg=f"DR polishing success {success}, expected True."
            )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_identifying_violating_param_realization(self):
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u ** (0.5) - m.x2 * m.u <= 2)
        m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

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
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u ** (0.5) - m.x2 * m.u <= 2)
        m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

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
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u ** (0.5) - m.x2 * m.u <= 2)
        m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

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
        SolverFactory('baron').license_is_valid(),
        "Global NLP solver is not available and licensed.",
    )
    def test_separation_terminate_time_limit(self):
        """
        Test PyROS time limit status returned in event
        separation problem times out.
        """
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u ** (0.5) - m.x2 * m.u <= 2)
        m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

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
        SolverFactory('gams').license_is_valid()
        and SolverFactory('baron').license_is_valid(),
        "Global NLP solver is not available and licensed.",
    )
    def test_gams_successful_time_limit(self):
        """
        Test PyROS time limit status returned in event
        separation problem times out.
        """
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u ** (0.5) - m.x2 * m.u <= 2)
        m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

        # Define the uncertainty set
        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        # two GAMS solvers, one of which has reslim set
        # (overridden when invoked in PyROS)
        local_subsolvers = [
            SolverFactory("gams:conopt"),
            SolverFactory("gams:conopt"),
            SolverFactory("ipopt"),
        ]
        local_subsolvers[0].options["add_options"] = ["option reslim=100;"]
        global_subsolver = SolverFactory("baron")
        global_subsolver.options["MaxTime"] = 300

        # Call the PyROS solver
        for idx, opt in enumerate(local_subsolvers):
            results = pyros_solver.solve(
                model=m,
                first_stage_variables=[m.x1, m.x2],
                second_stage_variables=[],
                uncertain_params=[m.u],
                uncertainty_set=interval,
                local_solver=opt,
                global_solver=global_subsolver,
                objective_focus=ObjectiveType.worst_case,
                solve_master_globally=True,
                time_limit=100,
            )

            self.assertEqual(
                results.pyros_termination_condition,
                pyrosTerminationCondition.robust_optimal,
                msg=(
                    f"Returned termination condition with local "
                    "subsolver {idx + 1} of 2 is not robust_optimal."
                ),
            )

        # check first local subsolver settings
        # remain unchanged after PyROS exit
        self.assertEqual(
            len(list(local_subsolvers[0].options["add_options"])),
            1,
            msg=(
                f"Local subsolver {local_subsolvers[0]} options 'add_options'"
                "were changed by PyROS"
            ),
        )
        self.assertEqual(
            local_subsolvers[0].options["add_options"][0],
            "option reslim=100;",
            msg=(
                f"Local subsolver {local_subsolvers[0]} setting "
                "'add_options' was modified "
                "by PyROS, but changes were not properly undone"
            ),
        )

        # check global subsolver settings unchanged
        self.assertEqual(
            len(list(global_subsolver.options.keys())),
            1,
            msg=(f"Global subsolver {global_subsolver} options were changed by PyROS"),
        )
        self.assertEqual(
            global_subsolver.options["MaxTime"],
            300,
            msg=(
                f"Global subsolver {global_subsolver} setting "
                "'MaxTime' was modified "
                "by PyROS, but changes were not properly undone"
            ),
        )

        # check other local subsolvers remain unchanged
        for slvr, key in zip(local_subsolvers[1:], ["add_options", "max_cpu_time"]):
            # no custom options were added to the `options`
            # attribute of the optimizer, so any attribute
            # of `options` should be `None`
            self.assertIs(
                getattr(slvr.options, key, None),
                None,
                msg=(
                    f"Local subsolver {slvr} setting '{key}' was added "
                    "by PyROS, but not reverted"
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
        d_set = DiscreteScenarioSet(scenarios=[(1,), (0,)])

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

    # FIXME: This test is expected to fail now, as writing out invalid
    # models generates an exception in the problem writer (and is never
    # actually sent to the solver)
    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    @unittest.expectedFailure
    def test_discrete_separation_subsolver_error(self):
        """
        Test PyROS for two-stage problem with discrete type set,
        subsolver error status.
        """
        m = ConcreteModel()

        m.q = Param(initialize=1, mutable=True)
        m.x1 = Var(initialize=1, bounds=(0, 1))

        # upper bound induces subsolver error: separation
        # max(x2 - log(m.q)) will force subsolver to q = 0
        m.x2 = Var(initialize=2, bounds=(None, log(m.q)))

        m.obj = Objective(expr=m.x1 + m.x2, sense=maximize)

        discrete_set = DiscreteScenarioSet(scenarios=[(1,), (0,)])

        local_solver = SolverFactory("ipopt")
        global_solver = SolverFactory("baron")
        pyros_solver = SolverFactory("pyros")

        res = pyros_solver.solve(
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
        self.assertEqual(
            res.pyros_termination_condition,
            pyrosTerminationCondition.subsolver_error,
            msg=(
                "Returned termination condition for separation error"
                f"test is not {pyrosTerminationCondition.subsolver_error}."
            ),
        )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_pyros_math_domain_error(self):
        """
        Test PyROS on a two-stage problem, discrete
        set type with a math domain error evaluating
        performance constraint expressions in separation.
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
                "Evaluation of performance constraint.*math domain error.*"
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
        performance constraints (such as effectively deterministic
        models).
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
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u ** (0.5) - m.x2 * m.u <= 2)
        m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u == m.x3)
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

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
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u ** (0.5) - m.x2 * m.u <= 2)
        m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

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
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    @unittest.skipUnless(
        baron_version == (23, 1, 5), "Test runs >90 minutes with Baron 22.9.30"
    )
    def test_higher_order_decision_rules(self):
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u ** (0.5) - m.x2 * m.u <= 2)
        m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

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
                "decision_rule_order": 2,
            },
        )

        self.assertEqual(
            results.pyros_termination_condition,
            pyrosTerminationCondition.robust_optimal,
            msg="Returned termination condition is not return robust_optimal.",
        )

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_coefficient_matching_solve(self):
        # Write the deterministic Pyomo model
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con = Constraint(expr=m.u ** (0.5) * m.x1 - m.u * m.x2 <= 2)
        m.eq_con = Constraint(
            expr=m.u**2 * (m.x2 - 1)
            + m.u * (m.x1**3 + 0.5)
            - 5 * m.u * m.x1 * m.x2
            + m.u * (m.x1 + 2)
            == 0
        )
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

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
            msg="Non-optimal termination condition from robust feasible coefficient matching problem.",
        )
        self.assertAlmostEqual(
            results.final_objective_value,
            6.0394,
            2,
            msg="Incorrect objective function value.",
        )

    def create_mitsos_4_3(self):
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
    def test_coeff_matching_solver_insensitive(self):
        """
        Check that result for instance with constraint subject to
        coefficient matching is insensitive to subsolver settings. Based
        on Mitsos (2011) semi-infinite programming instance 4_3.
        """
        m = self.create_mitsos_4_3()

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
                desired=0.9781633,
                rtol=0,
                atol=5e-3,
                err_msg=(
                    "Final objective for Watson 43 instance solved with "
                    f"subsolver {name} not as expected"
                ),
            )

    @unittest.skipUnless(
        baron_license_is_valid and baron_version >= (23, 2, 27),
        "BARON licensing and version requirements not met",
    )
    def test_coefficient_matching_partitioning_insensitive(self):
        """
        Check that result for instance with constraint subject to
        coefficient matching is insensitive to DOF partitioning. Model
        is based on Mitsos (2011) semi-infinite programming instance
        4_3.
        """
        m = self.create_mitsos_4_3()

        # instantiate BARON subsolver and PyROS solver
        baron = SolverFactory("baron")
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
                local_solver=baron,
                global_solver=baron,
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

    def test_coefficient_matching_raises_error_4_3(self):
        """
        Check that result for instance with constraint subject to
        coefficient matching results in exception certifying robustness
        cannot be certified where expected. Model
        is based on Mitsos (2011) semi-infinite programming instance
        4_3.
        """
        m = self.create_mitsos_4_3()

        # instantiate BARON subsolver and PyROS solver
        baron = SolverFactory("baron")
        pyros_solver = SolverFactory("pyros")

        # solve with PyROS
        dr_orders = [1, 2]
        for dr_order in dr_orders:
            regex_assert_mgr = self.assertRaisesRegex(
                ValueError,
                expected_regex=(
                    "Coefficient matching unsuccessful. See the solver logs."
                ),
            )
            logging_intercept_mgr = LoggingIntercept(level=logging.ERROR)

            with regex_assert_mgr, logging_intercept_mgr as LOG:
                pyros_solver.solve(
                    model=m,
                    first_stage_variables=[],
                    second_stage_variables=[m.x1, m.x2, m.x3],
                    uncertain_params=[m.u],
                    uncertainty_set=BoxSet(bounds=[[0, 1]]),
                    local_solver=baron,
                    global_solver=baron,
                    objective_focus=ObjectiveType.worst_case,
                    decision_rule_order=dr_order,
                    solve_master_globally=True,
                    bypass_local_separation=True,
                    robust_feasibility_tolerance=1e-4,
                )

            detailed_error_msg = LOG.getvalue()
            self.assertRegex(
                detailed_error_msg[:-1],
                (
                    r"Equality constraint.*cannot be guaranteed to "
                    r"be robustly feasible.*"
                    r"Consider editing this constraint.*"
                ),
            )

    def test_coefficient_matching_robust_infeasible_proof_in_pyros(self):
        # Write the deterministic Pyomo model
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con = Constraint(expr=m.u ** (0.5) * m.x1 - m.u * m.x2 <= 2)
        m.eq_con = Constraint(
            expr=m.u * (m.x1**3 + 0.5)
            - 5 * m.u * m.x1 * m.x2
            + m.u * (m.x1 + 2)
            + m.u**2
            == 0
        )
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

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
            pyrosTerminationCondition.robust_infeasible,
            msg="Robust infeasible problem not identified via coefficient matching.",
        )

    def test_coefficient_matching_nonlinear_expr(self):
        # Write the deterministic Pyomo model
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con = Constraint(expr=m.u ** (0.5) * m.x1 - m.u * m.x2 <= 2)
        m.eq_con = Constraint(
            expr=m.u**2 * (m.x2 - 1)
            + m.u * (m.x1**3 + 0.5)
            - 5 * m.u * m.x1 * m.x2
            + m.u * (m.x1 + 2)
            == 0
        )
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        with self.assertRaises(
            ValueError,
            msg="ValueError should be raised for general "
            "nonlinear expressions in h(x,z,q)=0 constraints.",
        ):
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
                    "decision_rule_order": 1,
                },
            )


@unittest.skipUnless(
    baron_available and baron_license_is_valid,
    "Global NLP solver is not available and licensed.",
)
class testBypassingSeparation(unittest.TestCase):
    def test_bypass_global_separation(self):
        """Test bypassing of global separation solve calls."""
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u ** (0.5) - m.x2 * m.u <= 2)
        m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

        # Define the uncertainty set
        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('ipopt')
        global_subsolver = SolverFactory("baron")

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


@unittest.skipUnless(
    baron_available and baron_license_is_valid,
    "Global NLP solver is not available and licensed.",
)
class testModelMultipleObjectives(unittest.TestCase):
    """
    This class contains tests for models with multiple
    Objective attributes.
    """

    def test_multiple_objs(self):
        """Test bypassing of global separation solve calls."""
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u ** (0.5) - m.x2 * m.u <= 2)
        m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

        # add another objective
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
        global_subsolver = SolverFactory("baron")

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
            AttributeError,
            "This model structure is not currently handled by the ROSolver.",
        ):
            pyros_solver.solve(**solve_kwargs)

        # check validation error raised due to multiple objectives
        m.b.obj.deactivate()
        with self.assertRaisesRegex(
            AttributeError,
            "This model structure is not currently handled by the ROSolver.",
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


class testModelIdentifyObjectives(unittest.TestCase):
    """
    This class contains tests for validating routines used to
    determine the first-stage and second-stage portions of a
    two-stage expression.
    """

    def test_identify_objectives(self):
        """
        Test first and second-stage objective identification
        for a simple two-stage model.
        """
        # model
        m = ConcreteModel()

        # parameters
        m.p = Param(range(4), initialize=1, mutable=True)
        m.q = Param(initialize=1)

        # variables
        m.x = Var(range(4))
        m.z = Var()
        m.y = Var(initialize=2)

        # objective
        m.obj = Objective(
            expr=(
                (m.x[0] + m.y)
                * (
                    sum(m.x[idx] * m.p[idx] for idx in range(3))
                    + m.q * m.z
                    + m.x[0] * m.q
                )
                + sin(m.x[0] + m.q)
                + cos(m.x[2] + m.z)
            )
        )

        # util block for specifying DOF and uncertainty
        m.util = Block()
        m.util.first_stage_variables = list(m.x.values())
        m.util.second_stage_variables = [m.z]
        m.util.uncertain_params = [m.p[0], m.p[1]]

        identify_objective_functions(m, m.obj)

        fsv_set = ComponentSet(m.util.first_stage_variables)
        uncertain_param_set = ComponentSet(m.util.uncertain_params)

        # determine vars and uncertain params participating in
        # objective
        fsv_in_obj = ComponentSet(
            var for var in identify_variables(m.obj) if var in fsv_set
        )
        ssv_in_obj = ComponentSet(
            var for var in identify_variables(m.obj) if var not in fsv_set
        )
        uncertain_params_in_obj = ComponentSet(
            param
            for param in identify_mutable_parameters(m.obj)
            if param in uncertain_param_set
        )

        # determine vars and uncertain params participating in
        # first-stage objective
        fsv_in_first_stg_cost = ComponentSet(
            var for var in identify_variables(m.first_stage_objective) if var in fsv_set
        )
        ssv_in_first_stg_cost = ComponentSet(
            var
            for var in identify_variables(m.first_stage_objective)
            if var not in fsv_set
        )
        uncertain_params_in_first_stg_cost = ComponentSet(
            param
            for param in identify_mutable_parameters(m.first_stage_objective)
            if param in uncertain_param_set
        )

        # determine vars and uncertain params participating in
        # second-stage objective
        fsv_in_second_stg_cost = ComponentSet(
            var
            for var in identify_variables(m.second_stage_objective)
            if var in fsv_set
        )
        ssv_in_second_stg_cost = ComponentSet(
            var
            for var in identify_variables(m.second_stage_objective)
            if var not in fsv_set
        )
        uncertain_params_in_second_stg_cost = ComponentSet(
            param
            for param in identify_mutable_parameters(m.second_stage_objective)
            if param in uncertain_param_set
        )

        # now perform checks
        self.assertTrue(
            fsv_in_first_stg_cost | fsv_in_second_stg_cost == fsv_in_obj,
            f"{{var.name for var in fsv_in_first_stg_cost | fsv_in_second_stg_cost}} "
            f"is not {{var.name for var in fsv_in_obj}}",
        )
        self.assertFalse(
            ssv_in_first_stg_cost,
            f"First-stage expression {str(m.first_stage_objective.expr)}"
            f" consists of non first-stage variables "
            f"{{var.name for var in fsv_in_second_stg_cost}}",
        )
        self.assertTrue(
            ssv_in_second_stg_cost == ssv_in_obj,
            f"{[var.name for var in ssv_in_second_stg_cost]} is not"
            f"{{var.name for var in ssv_in_obj}}",
        )
        self.assertFalse(
            uncertain_params_in_first_stg_cost,
            f"First-stage expression {str(m.first_stage_objective.expr)}"
            " consists of uncertain params"
            f" {{p.name for p in uncertain_params_in_first_stg_cost}}",
        )
        self.assertTrue(
            uncertain_params_in_second_stg_cost == uncertain_params_in_obj,
            f"{{p.name for p in uncertain_params_in_second_stg_cost}} is not "
            f"{{p.name for p in uncertain_params_in_obj}}",
        )

    def test_identify_objectives_var_expr(self):
        """
        Test first and second-stage objective identification
        for an objective expression consisting only of a Var.
        """
        # model
        m = ConcreteModel()

        # parameters
        m.p = Param(range(4), initialize=1, mutable=True)
        m.q = Param(initialize=1)

        # variables
        m.x = Var(range(4))

        # objective
        m.obj = Objective(expr=m.x[1])

        # util block for specifying DOF and uncertainty
        m.util = Block()
        m.util.first_stage_variables = list(m.x.values())
        m.util.second_stage_variables = list()
        m.util.uncertain_params = list()

        identify_objective_functions(m, m.obj)
        fsv_in_second_stg_obj = list(
            v.name for v in identify_variables(m.second_stage_objective)
        )

        # perform checks
        self.assertTrue(list(identify_variables(m.first_stage_objective)) == [m.x[1]])
        self.assertFalse(
            fsv_in_second_stg_obj,
            "Second stage objective contains variable(s) " f"{fsv_in_second_stg_obj}",
        )


class testMasterFeasibilityUnitConsistency(unittest.TestCase):
    """
    Test cases for models with unit-laden model components.
    """

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    @unittest.skipUnless(
        baron_version < (23, 1, 5), "Test known to fail beginning with Baron 23.1.5"
    )
    def test_two_stg_mod_with_axis_aligned_set(self):
        """
        Test two-stage model with `AxisAlignedEllipsoidalSet`
        as the uncertainty set.
        """
        from pyomo.environ import units as u

        # define model
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
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

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
        # define model
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u1 = Param(initialize=1.125, mutable=True)
        m.u2 = Param(initialize=1, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u1 ** (0.5) - m.x2 * m.u1 <= 2)
        m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u1 == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - m.u2) ** 2)

        return m

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
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_two_stg_mod_with_intersection_set(self):
        """
        Test two-stage model with `AxisAlignedEllipsoidalSet`
        as the uncertainty set.
        """
        # define model
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u1 = Param(initialize=1.125, mutable=True)
        m.u2 = Param(initialize=1, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u1 ** (0.5) - m.x2 * m.u1 <= 2)
        m.con2 = Constraint(expr=m.x1**2 - m.x2**2 * m.u1 == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - m.u2) ** 2)

        # construct the IntersectionSet
        ellipsoid = AxisAlignedEllipsoidalSet(center=[1.125, 1], half_lengths=[1, 0])
        bset = BoxSet(bounds=[[1, 2], [0.5, 1.5]])
        iset = IntersectionSet(ellipsoid=ellipsoid, bset=bset)

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
        A '+' should be appended to the number of performance constraints
        found to be violated.
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


if __name__ == "__main__":
    unittest.main()
