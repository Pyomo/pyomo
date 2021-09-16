'''
Unit tests for the grcs API
One class per function being tested, minimum one test per class
'''

import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.environ import *
from pyomo.core.expr.current import identify_variables, identify_mutable_parameters
from pyomo.contrib.pyros.util import selective_clone, add_decision_rule_variables, add_decision_rule_constraints, \
    model_is_valid, turn_bounds_to_constraints, transform_to_standard_form, ObjectiveType, pyrosTerminationCondition, \
    coefficient_matching
from pyomo.contrib.pyros.uncertainty_sets import *
from pyomo.contrib.pyros.master_problem_methods import add_scenario_to_master, initial_construct_master, solve_master, \
    minimize_dr_vars
from pyomo.contrib.pyros.solve_data import MasterProblemData
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available

if not (numpy_available and scipy_available):
    raise unittest.SkipTest('PyROS unit tests require numpy and scipy')

# === Config args for testing
nlp_solver = 'ipopt'
global_solver = 'baron'
global_solver_args = dict()
nlp_solver_args = dict()

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
        m.con = Constraint(expr= m.x * m.p + m.y <= 0)

        n = ConcreteModel()
        n.x = Var()
        m.first_stage_vars = [n.x]

        cloned_model = selective_clone(block=m, first_stage_vars=m.first_stage_vars)

        self.assertNotEqual(id(m.first_stage_vars), id(cloned_model.first_stage_vars), msg="First stage variables should"
                                                                                           "not be equal.")
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

        self.assertEqual(id(m.x), id(cloned_model.x),
                            msg="First stage variables should"
                                "be equal.")
        self.assertNotEqual(id(m.y), id(cloned_model.y), msg="Non-first-stage variables should not be equal.")
        self.assertNotEqual(id(m.p), id(cloned_model.p), msg="Params should not be equal.")
        self.assertNotEqual(id(m.con), id(cloned_model.con), msg="Constraint objects should not be equal.")

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

        self.assertEqual(len(m.working_model.util.first_stage_variables), len(m.working_model.util.second_stage_variables),
                         msg="For static approximation decision rule the number of decision rule variables"
                             "added to the list of design variables should equal the number of control variables.")

        m.working_model.util.first_stage_variables = []

        m.working_model.del_component(m.working_model.decision_rule_var_0)
        m.working_model.del_component(m.working_model.decision_rule_var_1)

        config.decision_rule_order = 1

        add_decision_rule_variables(m, config=config)

        self.assertEqual(len(m.working_model.util.first_stage_variables),
                         len(m.working_model.util.second_stage_variables)*(1 + len(m.working_model.util.uncertain_params)),
                         msg="For affine decision rule the number of decision rule variables add to the "
                             "list of design variables should equal the number of control variables"
                             "multiplied by the number of uncertain parameters plus 1.")

        m.working_model.util.first_stage_variables = []

        m.working_model.del_component(m.working_model.decision_rule_var_0)
        m.working_model.del_component(m.working_model.decision_rule_var_1)
        m.working_model.del_component(m.working_model.decision_rule_var_0_index)
        m.working_model.del_component(m.working_model.decision_rule_var_1_index)

        config.decision_rule_order = 2

        add_decision_rule_variables(m, config=config)

        self.assertEqual(len(m.working_model.util.first_stage_variables),
                         len(m.working_model.util.second_stage_variables)*
                         int(2 * len(m.working_model.util.uncertain_params) +
                             sp.special.comb(N=len(m.working_model.util.uncertain_params), k=2) + 1),
                         msg="For quadratic decision rule the number of decision rule variables add to the "
                             "list of design variables should equal the number of control variables"
                             "multiplied by 2 time the number of uncertain parameters plus all 2-combinations"
                             "of uncertain parameters plus 1.")

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

        add_decision_rule_constraints(model_data=m,config=config)

        for c in m.working_model.component_data_objects(Constraint, descend_into=True):
            if "decision_rule_eqn_" in c.name:
                decision_rule_cons.append(c)
                m.working_model.del_component(c)

        self.assertEqual(len(decision_rule_cons), len(m.working_model.util.second_stage_variables),
                         msg="The number of decision rule constraints added to model should equal"
                             "the number of control variables in the model.")

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

        self.assertEqual(len(decision_rule_cons), len(m.working_model.util.second_stage_variables),
                         msg="The number of decision rule constraints added to model should equal"
                             "the number of control variables in the model.")

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

        self.assertEqual(len(decision_rule_cons), len(m.working_model.util.second_stage_variables),
                         msg="The number of decision rule constraints added to model should equal"
                             "the number of control variables in the model.")

class testModelIsValid(unittest.TestCase):

    def test_model_is_valid_via_possible_inputs(self):
        m = ConcreteModel()
        m.x = Var()
        m.obj1 = Objective(expr = m.x**2)
        self.assertTrue(model_is_valid(m))
        m.obj2 = Objective(expr = m.x)
        self.assertFalse(model_is_valid(m))
        m.del_component("obj1")
        m.del_component("obj2")
        self.assertFalse(model_is_valid(m))

class testTurnBoundsToConstraints(unittest.TestCase):

    def test_bounds_to_constraints(self):
        m = ConcreteModel()
        m.x = Var(initialize=1, bounds=(0,1))
        m.y = Var(initialize=0, bounds=(None,1))
        m.w = Var(initialize=0, bounds=(1, None))
        m.z = Var(initialize=0, bounds=(None,None))
        turn_bounds_to_constraints(m.z, m)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 0,
                         msg="Inequality constraints were written for bounds on a variable with no bounds.")
        turn_bounds_to_constraints(m.y, m)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 1,
                         msg="Inequality constraints were not "
                             "written correctly for a variable with an upper bound and no lower bound.")
        turn_bounds_to_constraints(m.w, m)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 2,
                         msg="Inequality constraints were not "
                             "written correctly for a variable with a lower bound and no upper bound.")
        turn_bounds_to_constraints(m.x, m)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 4,
                         msg="Inequality constraints were not "
                             "written correctly for a variable with both lower and upper bound.")

class testTransformToStandardForm(unittest.TestCase):

    def test_transform_does_not_alter_num_of_constraints(self):
        '''
        Standard form for the purpose of PyROS is all inequality constraints as g(.)<=0
        :return:
        '''
        m = ConcreteModel()
        m.x = Var(initialize=1, bounds=(0, 1))
        m.y = Var(initialize=0, bounds=(None, 1))
        m.con1 = Constraint(expr=m.x >= 1 + m.y)
        m.con2 = Constraint(expr=m.x**2 + m.y**2>= 9)
        original_num_constraints = len(list(m.component_data_objects(Constraint)))
        transform_to_standard_form(m)
        final_num_constraints = len(list(m.component_data_objects(Constraint)))
        self.assertEqual(original_num_constraints, final_num_constraints,
                         msg="Transform to standard form function led to a "
                             "different number of constraints than in the original model.")
        number_of_non_standard_form_inequalities = \
            len(list(c for c in list(m.component_data_objects(Constraint)) if c.lower != None))
        self.assertEqual(number_of_non_standard_form_inequalities, 0,
                         msg="All inequality constraints were not transformed to standard form.")

# === UncertaintySets.py
# Mock abstract class
class myUncertaintySet(UncertaintySet):
    '''
    returns single Constraint representing the uncertainty set which is
    simply a linear combination of uncertain_params
    '''

    def set_as_constraint(self, uncertain_params, **kwargs):

        return Constraint(expr= sum(v for v in uncertain_params) <= 0)

    def point_in_set(self, uncertain_params, **kwargs):

        return True

    def geometry(self):
        self.geometry = Geometry.LINEAR

    def dim(self):
        self.dim = 1

    def parameter_bounds(self):
        return [(0,1)]

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
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        uncertain_params_in_expr = list(
            v for v in m.uncertain_param_vars if
            v in ComponentSet(identify_variables(expr=m.uncertainty_set_contr.expr))
        )

        self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars],
                          msg="Uncertain param Var objects used to construct uncertainty set constraint must"
                              "be the same uncertain param Var objects in the original model.")

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
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_params)
        variables_in_constr = list(
            v for v in m.uncertain_params if
            v in ComponentSet(identify_variables(expr=m.uncertainty_set_contr.expr))
        )

        self.assertEqual(len(variables_in_constr), 0,
                         msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
                             "variable expression.")

class testEllipsoidalUncertaintySetClass(unittest.TestCase):
    '''
    Ellipsoidal uncertainty sets. Required inputs are covariance matrix covar, scale, mean, and list
    of uncertain params.
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
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        cov = [[1,0], [0,1]]
        s = 1

        _set = EllipsoidalSet(center=[0, 0], shape_matrix=cov, scale=s)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        uncertain_params_in_expr = list(
            v for v in m.uncertain_param_vars.values() if
            v in ComponentSet(identify_variables(expr=m.uncertainty_set_contr[1].expr))
        )

        self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars.values()],
                          msg="Uncertain param Var objects used to construct uncertainty set constraint must"
                              " be the same uncertain param Var objects in the original model.")

    def test_uncertainty_set_with_incorrect_params(self):
        '''
        Case in which the EllipsoidalSet is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        '''
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
        cov = [[1,0],[0,1]]
        s = 1

        _set = EllipsoidalSet(center=[0, 0], shape_matrix=cov, scale=s)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        variables_in_constr = list(
            v for v in m.uncertain_params if
            v in ComponentSet(identify_variables(expr=m.uncertainty_set_contr[1].expr))
        )

        self.assertEqual(len(variables_in_constr), 0,
                         msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
                             " variable expression.")

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        cov = [[1, 0], [0, 1]]
        s = 1

        _set = EllipsoidalSet(center=[0, 0], shape_matrix=cov, scale=s)
        self.assertTrue(_set.point_in_set([0,0]), msg="Point is not in the EllipsoidalSet.")

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

        self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None,
                            "Bounds not added correctly for EllipsoidalSet")
        self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None,
                            "Bounds not added correctly for EllipsoidalSet")
        self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None,
                            "Bounds not added correctly for EllipsoidalSet")
        self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None,
                            "Bounds not added correctly for EllipsoidalSet")

class testAxisAlignedEllipsoidalUncertaintySetClass(unittest.TestCase):
    '''
    Axis aligned ellipsoidal uncertainty sets. Required inputs are half-lengths, nominal point, and right-hand side.
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
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        _set = AxisAlignedEllipsoidalSet(center=[0,0], half_lengths=[2,1])
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        uncertain_params_in_expr = list(
            v for v in m.uncertain_param_vars.values() if
            v in ComponentSet(identify_variables(expr=m.uncertainty_set_contr[1].expr))
        )

        self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars.values()],
                          msg="Uncertain param Var objects used to construct uncertainty set constraint must"
                              " be the same uncertain param Var objects in the original model.")

    def test_uncertainty_set_with_incorrect_params(self):
        '''
        Case in which the set is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        '''
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
        _set = AxisAlignedEllipsoidalSet(center=[0,0], half_lengths=[2,1])
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        variables_in_constr = list(
            v for v in m.uncertain_params if
            v in ComponentSet(identify_variables(expr=m.uncertainty_set_contr[1].expr))
        )

        self.assertEqual(len(variables_in_constr), 0,
                         msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
                             " variable expression.")

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        _set = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
        self.assertTrue(_set.point_in_set([0, 0]),
                        msg="Point is not in the AxisAlignedEllipsoidalSet.")
        
    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0,1], initialize=0.5)

        _set = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
        config = Block()
        config.uncertainty_set = _set

        AxisAlignedEllipsoidalSet.add_bounds_on_uncertain_parameters(model=m, config=config)

        self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None, "Bounds not added correctly for AxisAlignedEllipsoidalSet")
        self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None, "Bounds not added correctly for AxisAlignedEllipsoidalSet")
        self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None, "Bounds not added correctly for AxisAlignedEllipsoidalSet")
        self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None, "Bounds not added correctly for AxisAlignedEllipsoidalSet")

class testPolyhedralUncertaintySetClass(unittest.TestCase):
    '''
    Polyhedral uncertainty sets. Required inputs are matrix A, right-hand-side b, and list of uncertain params.
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
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        A = [[0, 1], [1, 0]]
        b = [0, 0]

        _set = PolyhedralSet(lhs_coefficients_mat=A, rhs_vec=b, )
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        uncertain_params_in_expr = ComponentSet()
        for con in m.uncertainty_set_contr.values():
            con_vars = ComponentSet(identify_variables(expr=con.expr))
            for v in m.uncertain_param_vars.values():
                if v in con_vars:
                    uncertain_params_in_expr.add(v)

        self.assertEqual(uncertain_params_in_expr,
                         ComponentSet(m.uncertain_param_vars.values()),
                         msg="Uncertain param Var objects used to construct uncertainty set constraint must"
                              " be the same uncertain param Var objects in the original model.")

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
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
        A = [[0, 1], [1, 0]]
        b = [0, 0]

        _set = PolyhedralSet(lhs_coefficients_mat=A, rhs_vec=b)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            vars_in_expr.extend(
                v for v in m.uncertain_param_vars if
                v in ComponentSet(identify_variables(expr=con.expr))
            )

        self.assertEqual(len(vars_in_expr), 0,
                             msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
                                 " variable expression.")

    def test_polyhedral_set_as_constraint(self):
        '''
        The set_as_constraint method must return an indexed uncertainty_set_constr
        which has as many elements at their are dimensions in A.
        '''

        A = [[1, 0],[0, 1]]
        b = [0, 0]

        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)

        polyhedral_set = PolyhedralSet(lhs_coefficients_mat=A, rhs_vec=b)
        m.uncertainty_set_constr = polyhedral_set.set_as_constraint(uncertain_params=[m.p1, m.p2])

        self.assertEqual(len(A), len(m.uncertainty_set_constr.index_set()),
                         msg="Polyhedral uncertainty set constraints must be as many as the"
                             "number of rows in the matrix A.")

    def test_point_in_set(self):
        A = [[1, 0], [0, 1]]
        b = [0, 0]

        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        polyhedral_set = PolyhedralSet(lhs_coefficients_mat=A, rhs_vec=b)
        self.assertTrue(polyhedral_set.point_in_set([0, 0]),
                        msg="Point is not in the PolyhedralSet.")
    
    @unittest.skipUnless(SolverFactory('baron').available(exception_flag=False), "Global NLP solver is not available.")
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

        self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None, "Bounds not added correctly for PolyhedralSet")
        self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None, "Bounds not added correctly for PolyhedralSet")
        self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None, "Bounds not added correctly for PolyhedralSet")
        self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None, "Bounds not added correctly for PolyhedralSet")

class testBudgetUncertaintySetClass(unittest.TestCase):
    '''
    Budget uncertainty sets. Required inputs are matrix budget_membership_mat, rhs_vec.
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
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        # Single budget
        budget_membership_mat = [[1 for i in range(len(m.uncertain_param_vars))]]
        rhs_vec = [0.1 * len(m.uncertain_param_vars) + sum(p.value for p in m.uncertain_param_vars.values())]

        _set = BudgetSet(budget_membership_mat=budget_membership_mat,
                        rhs_vec=rhs_vec)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        uncertain_params_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if v in ComponentSet(identify_variables(expr=con.expr)):
                    if id(v) not in list(id(u) for u in uncertain_params_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        uncertain_params_in_expr.append(v)


        self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars.values()],
                          msg="Uncertain param Var objects used to construct uncertainty set constraint must"
                              " be the same uncertain param Var objects in the original model.")

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
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
        # Single budget
        budget_membership_mat = [[1 for i in range(len(m.uncertain_param_vars))]]
        rhs_vec = [0.1 * len(m.uncertain_param_vars) + sum(p.value for p in m.uncertain_param_vars.values())]

        _set = BudgetSet(budget_membership_mat=budget_membership_mat,
                        rhs_vec=rhs_vec)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            vars_in_expr.extend(
                v for v in m.uncertain_param_vars.values() if
                v in ComponentSet(identify_variables(expr=con.expr))
            )

        self.assertEqual(len(vars_in_expr), 0,
                             msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
                                 " variable expression.")

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
        rhs_vec = [0.1 * len(m.uncertain_params) + sum(p.value for p in m.uncertain_params)]

        budget_set = BudgetSet(budget_membership_mat=budget_membership_mat,
                               rhs_vec=rhs_vec)
        m.uncertainty_set_constr = budget_set.set_as_constraint(uncertain_params=m.uncertain_params)

        self.assertEqual(len(budget_membership_mat), len(m.uncertainty_set_constr.index_set()),
                         msg="Budget uncertainty set constraints must be as many as the"
                             "number of rows in the matrix A.")

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)

        budget_membership_mat = [[1 for i in range(len(m.uncertain_params))]]
        rhs_vec = [0.1 * len(m.uncertain_params) + sum(p.value for p in m.uncertain_params)]

        budget_set = BudgetSet(budget_membership_mat=budget_membership_mat,
                               rhs_vec=rhs_vec)
        self.assertTrue(budget_set.point_in_set([0, 0]),
                        msg="Point is not in the BudgetSet.")

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0,1], initialize=0.5)

        budget_membership_mat = [[1 for i in range(len(m.util.uncertain_param_vars))]]
        rhs_vec = [0.1 * len(m.util.uncertain_param_vars) + sum(value(p) for p in m.util.uncertain_param_vars.values())]

        budget_set = BudgetSet(budget_membership_mat=budget_membership_mat,
                               rhs_vec=rhs_vec)
        config = Block()
        config.uncertainty_set = budget_set

        BudgetSet.add_bounds_on_uncertain_parameters(model=m, config=config)

        self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None, "Bounds not added correctly for BudgetSet")
        self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None, "Bounds not added correctly for BudgetSet")
        self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None, "Bounds not added correctly for BudgetSet")
        self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None, "Bounds not added correctly for BudgetSet")

class testCardinalityUncertaintySetClass(unittest.TestCase):
    '''
    Cardinality uncertainty sets. Required inputs are origin, positive_deviation, gamma.
    Because Cardinality adds cassi vars to model, must pass model to set_as_constraint()
    '''

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

        _set = CardinalitySet(origin=center,
                        positive_deviation=positive_deviation, gamma=gamma)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars, model=m)
        uncertain_params_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if v in ComponentSet(identify_variables(expr=con.expr)):
                    if id(v) not in list(id(u) for u in uncertain_params_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        uncertain_params_in_expr.append(v)


        self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars.values()],
                          msg="Uncertain param Var objects used to construct uncertainty set constraint must"
                              " be the same uncertain param Var objects in the original model.")


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
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)

        center = list(p.value for p in m.uncertain_param_vars.values())
        positive_deviation = list(0.3 for j in range(len(center)))
        gamma = np.ceil(len(m.uncertain_param_vars) / 2)

        _set = CardinalitySet(origin=center,
                             positive_deviation=positive_deviation, gamma=gamma)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars, model=m)
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if id(v) in [id(u) for u in list(identify_variables(expr=con.expr))]:
                    if id(v) not in list(id(u) for u in vars_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        vars_in_expr.append(v)

        self.assertEqual(len(vars_in_expr), 0,
                             msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
                                 " variable expression.")


    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)

        center = list(p.value for p in m.uncertain_param_vars.values())
        positive_deviation = list(0.3 for j in range(len(center)))
        gamma = np.ceil(len(m.uncertain_param_vars) / 2)

        _set = CardinalitySet(origin=center,
                             positive_deviation=positive_deviation, gamma=gamma)

        self.assertTrue(_set.point_in_set([0, 0]),
                        msg="Point is not in the CardinalitySet.")

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0,1], initialize=0.5)

        center = list(p.value for p in m.util.uncertain_param_vars.values())
        positive_deviation = list(0.3 for j in range(len(center)))
        gamma = np.ceil(len(center) / 2)

        cardinality_set = CardinalitySet(origin=center,
                             positive_deviation=positive_deviation, gamma=gamma)
        config = Block()
        config.uncertainty_set = cardinality_set

        CardinalitySet.add_bounds_on_uncertain_parameters(model=m, config=config)

        self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None, "Bounds not added correctly for CardinalitySet")
        self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None, "Bounds not added correctly for CardinalitySet")
        self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None, "Bounds not added correctly for CardinalitySet")
        self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None, "Bounds not added correctly for CardinalitySet")

class testBoxUncertaintySetClass(unittest.TestCase):
    '''
    Box uncertainty sets. Required input is bounds list.
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
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        bounds = [(-1,1), (-1,1)]
        _set = BoxSet(bounds=bounds)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        uncertain_params_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if v in ComponentSet(identify_variables(expr=con.expr)):
                    if id(v) not in list(id(u) for u in uncertain_params_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        uncertain_params_in_expr.append(v)

        self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars.values()],
                          msg="Uncertain param Var objects used to construct uncertainty set constraint must"
                              " be the same uncertain param Var objects in the original model.")

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
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
        bounds = [(-1, 1), (-1, 1)]
        _set = BoxSet(bounds=bounds)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        vars_in_expr = []
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if id(v) in [id(u) for u in list(identify_variables(expr=con.expr))]:
                    if id(v) not in list(id(u) for u in vars_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        vars_in_expr.append(v)

        self.assertEqual(len(vars_in_expr), 0,
                             msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
                                 " variable expression.")

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)

        bounds = [(-1, 1), (-1, 1)]
        _set = BoxSet(bounds=bounds)
        self.assertTrue(_set.point_in_set([0, 0]),
                        msg="Point is not in the BoxSet.")

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0,1], initialize=0)

        bounds = [(-1, 1), (-1, 1)]
        box_set = BoxSet(bounds=bounds)
        config = Block()
        config.uncertainty_set = box_set

        BoxSet.add_bounds_on_uncertain_parameters(model=m, config=config)

        self.assertEqual(m.util.uncertain_param_vars[0].lb, -1, "Bounds not added correctly for BoxSet")
        self.assertEqual(m.util.uncertain_param_vars[0].ub, 1, "Bounds not added correctly for BoxSet")
        self.assertEqual(m.util.uncertain_param_vars[1].lb, -1, "Bounds not added correctly for BoxSet")
        self.assertEqual(m.util.uncertain_param_vars[1].ub, 1, "Bounds not added correctly for BoxSet")

class testDiscreteUncertaintySetClass(unittest.TestCase):
    '''
    Discrete uncertainty sets. Required inputis a scenarios list.
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
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        scenarios = [(0,0), (1,0), (0,1), (1,1), (2,0)]
        _set = DiscreteScenarioSet(scenarios=scenarios)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        uncertain_params_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if v in ComponentSet(identify_variables(expr=con.expr)):
                    if id(v) not in list(id(u) for u in uncertain_params_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        uncertain_params_in_expr.append(v)


        self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars.values()],
                          msg="Uncertain param Var objects used to construct uncertainty set constraint must"
                              " be the same uncertain param Var objects in the original model.")

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
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
        scenarios = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]
        _set = DiscreteScenarioSet(scenarios=scenarios)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if id(v) in [id(u) for u in list(identify_variables(expr=con.expr))]:
                    if id(v) not in list(id(u) for u in vars_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        vars_in_expr.append(v)

        self.assertEqual(len(vars_in_expr), 0,
                             msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
                                 " variable expression.")

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)

        scenarios = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]
        _set = DiscreteScenarioSet(scenarios=scenarios)
        self.assertTrue(_set.point_in_set([0, 0]),
                        msg="Point is not in the DiscreteScenarioSet.")

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0,1], initialize=0)

        scenarios = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]
        _set = DiscreteScenarioSet(scenarios=scenarios)
        config = Block()
        config.uncertainty_set = _set

        DiscreteScenarioSet.add_bounds_on_uncertain_parameters(model=m, config=config)

        self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None, "Bounds not added correctly for DiscreteScenarioSet")
        self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None, "Bounds not added correctly for DiscreteScenarioSet")
        self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None, "Bounds not added correctly for DiscreteScenarioSet")
        self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None, "Bounds not added correctly for DiscreteScenarioSet")

class testFactorModelUncertaintySetClass(unittest.TestCase):
    '''
    FactorModelSet uncertainty sets. Required inputs are psi_matrix, number_of_factors, origin and beta.
    '''

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
        F=1
        psi_mat = np.zeros(shape=(len(m.uncertain_params), F))
        for i in range(len(psi_mat)):
            random_row_entries = list(np.random.uniform(low=0, high=0.2, size=F))
            for j in range(len(psi_mat[i])):
                psi_mat[i][j] = random_row_entries[j]
        _set = FactorModelSet(origin=[0,0], psi_mat=psi_mat, number_of_factors=F, beta=1)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars, model=m)
        uncertain_params_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if v in ComponentSet(identify_variables(expr=con.expr)):
                    if id(v) not in list(id(u) for u in uncertain_params_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        uncertain_params_in_expr.append(v)


        self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars.values()],
                          msg="Uncertain param Var objects used to construct uncertainty set constraint must"
                              " be the same uncertain param Var objects in the original model.")

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
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
        F = 1
        psi_mat = np.zeros(shape=(len(m.uncertain_params), F))
        for i in range(len(psi_mat)):
            random_row_entries = list(np.random.uniform(low=0, high=0.2, size=F))
            for j in range(len(psi_mat[i])):
                psi_mat[i][j] = random_row_entries[j]
        _set = FactorModelSet(origin=[0, 0], psi_mat=psi_mat, number_of_factors=F, beta=1)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars, model=m)
        vars_in_expr = []
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if id(v) in [id(u) for u in list(identify_variables(expr=con.expr))]:
                    if id(v) not in list(id(u) for u in vars_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        vars_in_expr.append(v)

        self.assertEqual(len(vars_in_expr), 0,
                             msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
                                 " variable expression.")

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
        _set = FactorModelSet(origin=[0, 0], psi_mat=psi_mat, number_of_factors=F, beta=1)
        self.assertTrue(_set.point_in_set([0, 0]),
                        msg="Point is not in the FactorModelSet.")

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0,1], initialize=0)

        F = 1
        psi_mat = np.zeros(shape=(len(list(m.util.uncertain_param_vars.values())), F))
        for i in range(len(psi_mat)):
            random_row_entries = list(np.random.uniform(low=0, high=0.2, size=F))
            for j in range(len(psi_mat[i])):
                psi_mat[i][j] = random_row_entries[j]
        _set = FactorModelSet(origin=[0, 0], psi_mat=psi_mat, number_of_factors=F, beta=1)
        config = Block()
        config.uncertainty_set = _set

        FactorModelSet.add_bounds_on_uncertain_parameters(model=m, config=config)

        self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None, "Bounds not added correctly for FactorModelSet")
        self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None, "Bounds not added correctly for FactorModelSet")
        self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None, "Bounds not added correctly for FactorModelSet")
        self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None, "Bounds not added correctly for FactorModelSet")

class testIntersectionSetClass(unittest.TestCase):
    '''
    Intersection uncertainty sets. Required input is set objects to intersect, and set_as_constraint requires
    a NLP solver to confirm the intersection is not empty.
    '''

    @unittest.skipUnless(SolverFactory('ipopt').available(exception_flag=False), "Local NLP solver is not available.")
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
        bounds = [(-1,1), (-1,1)]
        Q1 = BoxSet(bounds=bounds)
        Q2 = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
        Q = IntersectionSet(Q1=Q1, Q2=Q2)

        config = ConfigBlock()
        solver = SolverFactory("ipopt")
        config.declare("global_solver", ConfigValue(default=solver))

        m.uncertainty_set_contr = Q.set_as_constraint(uncertain_params=m.uncertain_param_vars, config=config)
        uncertain_params_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if v in ComponentSet(identify_variables(expr=con.expr)):
                    if id(v) not in list(id(u) for u in uncertain_params_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        uncertain_params_in_expr.append(v)

        self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars.values()],
                          msg="Uncertain param Var objects used to construct uncertainty set constraint must"
                              " be the same uncertain param Var objects in the original model.")

    @unittest.skipUnless(SolverFactory('ipopt').available(exception_flag=False), "Local NLP solver is not available.")
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
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
        bounds = [(-1, 1), (-1, 1)]

        Q1 = BoxSet(bounds=bounds)
        Q2 = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
        Q = IntersectionSet(Q1=Q1, Q2=Q2)

        solver = SolverFactory("ipopt")
        config = ConfigBlock()
        config.declare("global_solver", ConfigValue(default=solver))

        m.uncertainty_set_contr = Q.set_as_constraint(uncertain_params=m.uncertain_param_vars, config=config)
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if id(v) in [id(u) for u in list(identify_variables(expr=con.expr))]:
                    if id(v) not in list(id(u) for u in vars_in_expr):
                        # Not using ID here leads to it thinking both are in the list already when they aren't
                        vars_in_expr.append(v)

        self.assertEqual(len(vars_in_expr), 0,
                             msg="Uncertainty set constraint contains no Var objects, consists of a not potentially"
                                 " variable expression.")

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
        self.assertTrue(Q.point_in_set([0, 0]),
                        msg="Point is not in the IntersectionSet.")

    @unittest.skipUnless(SolverFactory('baron').available(exception_flag=False), "Global NLP solver is not available.")
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

        self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None, "Bounds not added correctly for IntersectionSet")
        self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None, "Bounds not added correctly for IntersectionSet")
        self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None, "Bounds not added correctly for IntersectionSet")
        self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None, "Bounds not added correctly for IntersectionSet")

# === master_problem_methods.py
class testInitialConstructMaster(unittest.TestCase):
    '''

    '''

    def test_initial_construct_master(self):
        model_data = MasterProblemData()
        model_data.timing = None
        model_data.working_model = ConcreteModel()
        master_data = initial_construct_master(model_data)
        self.assertTrue(hasattr(master_data, "master_model"),
                        msg="Initial construction of master problem "
                            "did not create a master problem ConcreteModel object.")

class testAddScenarioToMaster(unittest.TestCase):
    '''

    '''

    def test_add_scenario_to_master(self):
        working_model = ConcreteModel()
        working_model.p = Param([1,2],initialize=0,mutable=True)
        working_model.x = Var()
        model_data = MasterProblemData()
        model_data.working_model = working_model
        model_data.timing = None
        master_data = initial_construct_master(model_data)
        master_data.master_model.scenarios[0,0].transfer_attributes_from(working_model.clone())
        master_data.master_model.scenarios[0, 0].util = Block()
        master_data.master_model.scenarios[0, 0].util.first_stage_variables = \
            [master_data.master_model.scenarios[0,0].x]
        master_data.master_model.scenarios[0,0].util.uncertain_params = [master_data.master_model.scenarios[0,0].p[1],
                                                                        master_data.master_model.scenarios[0,0].p[2]]
        add_scenario_to_master(master_data, violations=[1,1])

        self.assertEqual(len(master_data.master_model.scenarios), 2, msg="Scenario not added to master correctly. "
                                                                         "Expected 2 scenarios.")

global_solver = "baron"
class testSolveMaster(unittest.TestCase):

    @unittest.skipUnless(SolverFactory('baron').available(exception_flag=False), "Global NLP solver is not available.")
    def test_solve_master(self):
        working_model = m = ConcreteModel()
        m.x = Var(initialize=0.5, bounds=(0,10))
        m.y = Var(initialize=1.0, bounds=(0,5))
        m.z = Var(initialize=0, bounds=(None, None))
        m.p = Param(initialize=1, mutable=True)
        m.obj = Objective(expr=m.x)
        m.con = Constraint(expr = m.x + m.y + m.z <= 3)
        model_data = MasterProblemData()
        model_data.working_model = working_model
        model_data.timing = None
        model_data.iteration = 0
        master_data = initial_construct_master(model_data)
        master_data.master_model.scenarios[0, 0].transfer_attributes_from(working_model.clone())
        master_data.master_model.scenarios[0, 0].util = Block()
        master_data.master_model.scenarios[0, 0].util.first_stage_variables = \
            [master_data.master_model.scenarios[0, 0].x]
        master_data.master_model.scenarios[0, 0].util.decision_rule_vars = []
        master_data.master_model.scenarios[0, 0].util.second_stage_variables = []
        master_data.master_model.scenarios[0, 0].util.uncertain_params = [master_data.master_model.scenarios[0, 0].p]
        master_data.master_model.scenarios[0, 0].first_stage_objective = 0
        master_data.master_model.scenarios[0, 0].second_stage_objective = \
            Expression(expr=master_data.master_model.scenarios[0, 0].x)
        master_data.iteration = 0
        box_set = BoxSet(bounds=[(0,2)])
        solver = SolverFactory(global_solver)
        config = ConfigBlock()
        config.declare("backup_global_solvers",ConfigValue(default=[]))
        config.declare("backup_local_solvers", ConfigValue(default=[]))
        config.declare("solve_master_globally", ConfigValue(default=True))
        config.declare("global_solver", ConfigValue(default=solver))
        config.declare("tee", ConfigValue(default=False))
        config.declare("decision_rule_order", ConfigValue(default=1))
        config.declare("objective_focus", ConfigValue(default=ObjectiveType.worst_case))
        config.declare("second_stage_variables", ConfigValue(default=master_data.master_model.scenarios[0, 0].util.second_stage_variables))
        config.declare("subproblem_file_directory", ConfigValue(default=None))
        master_soln = solve_master(master_data, config)
        self.assertEqual(master_soln.termination_condition, TerminationCondition.optimal,
                         msg="Could not solve simple master problem with solve_master function.")

# === regression test for the solver
class coefficientMatchingTests(unittest.TestCase):

    def test_coefficient_matching_correct_num_constraints_added(self):
        # Write the deterministic Pyomo model
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con = Constraint(expr=m.u ** (0.5) * m.x1 - m.u * m.x2 <= 2)
        m.eq_con = Constraint(expr =  m.u**2 * (m.x2- 1) + m.u * (m.x1**3 + 0.5) - 5 * m.u * m.x1 * m.x2 + m.u * (m.x1 + 2) == 0)
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

        coeff_matching_success, robust_infeasible = coefficient_matching(m, m.eq_con, [m.u], config)

        self.assertEqual(coeff_matching_success, True, msg="Coefficient matching was unsuccessful.")
        self.assertEqual(robust_infeasible, False, msg="Coefficient matching detected a robust infeasible constraint (1 == 0).")
        self.assertEqual(len(m.coefficient_matching_constraints), 2,
                         msg="Coefficient matching produced incorrect number of h(x,q)=0 constraints.")

        config.decision_rule_order = 1
        model_data = Block()
        model_data.working_model = m

        m.util.first_stage_variables = [m.x1]
        m.util.second_stage_variables = [m.x2]

        add_decision_rule_variables(model_data=model_data, config=config)
        add_decision_rule_constraints(model_data=model_data, config=config)

        coeff_matching_success, robust_infeasible = coefficient_matching(m, m.eq_con, [m.u], config)
        self.assertEqual(coeff_matching_success, False, msg="Coefficient matching should have been "
                                                            "unsuccessful for higher order polynomial expressions.")
        self.assertEqual(robust_infeasible, False, msg="Coefficient matching is not successful, "
                                                       "but should not be proven robust infeasible.")

    def test_coefficient_matching_robust_infeasible_proof(self):
        # Write the deterministic Pyomo model
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con = Constraint(expr=m.u ** (0.5) * m.x1 - m.u * m.x2 <= 2)
        m.eq_con = Constraint(expr =  m.u * (m.x1**3 + 0.5) - 5 * m.u * m.x1 * m.x2 + m.u * (m.x1 + 2) + m.u**2 == 0)
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

        coeff_matching_success, robust_infeasible = coefficient_matching(m, m.eq_con, [m.u], config)

        self.assertEqual(coeff_matching_success, False, msg="Coefficient matching should have been "
                                                            "unsuccessful.")
        self.assertEqual(robust_infeasible, True, msg="Coefficient matching should be proven robust infeasible.")

# === regression test for the solver
class RegressionTest(unittest.TestCase):

    @unittest.skipUnless(SolverFactory('baron').available(exception_flag=False), "Global NLP solver is not available.")
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
        results = pyros.solve(model=m,
                              first_stage_variables=m.decision_vars,
                              second_stage_variables=[],
                              uncertain_params=[m.p[1]],
                              uncertainty_set=box_set,
                              local_solver=solver,
                              global_solver=solver,
                              options={"objective_focus":ObjectiveType.nominal})
        self.assertTrue(results.pyros_termination_condition,
                         pyrosTerminationCondition.robust_feasible)

    @unittest.skipUnless(SolverFactory('baron').available(exception_flag=False),
                         "Global NLP solver is not available.")
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
        results = pyros.solve(model=m,
                              first_stage_variables=m.decision_vars,
                              second_stage_variables=[],
                              uncertain_params=[m.p[1]],
                              uncertainty_set=box_set,
                              local_solver=solver,
                              global_solver=solver,
                              options={"objective_focus": ObjectiveType.nominal,
                                       "decision_rule_order":1})
        self.assertTrue(results.pyros_termination_condition,
                        pyrosTerminationCondition.robust_feasible)

    @unittest.skipUnless(SolverFactory('baron').available(exception_flag=False),
                         "Global NLP solver is not available.")
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
        results = pyros.solve(model=m,
                              first_stage_variables=m.decision_vars,
                              second_stage_variables=[],
                              uncertain_params=[m.p[1]],
                              uncertainty_set=box_set,
                              local_solver=solver,
                              global_solver=solver,
                              options={"objective_focus": ObjectiveType.nominal,
                                       "decision_rule_order": 2})
        self.assertTrue(results.pyros_termination_condition,
                        pyrosTerminationCondition.robust_feasible)

    @unittest.skipUnless(SolverFactory('baron').available(exception_flag=False) and SolverFactory('baron').license_is_valid(),
                         "Global NLP solver is not available and licensed.")
    def test_minimize_dr_norm(self):
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.z1 = Var(initialize=0, bounds=(0,1))
        m.z2 = Var(initialize=0, bounds=(0,1))


        m.working_model = ConcreteModel()
        m.working_model.util = Block()

        m.working_model.util.second_stage_variables = [m.z1, m.z2]
        m.working_model.util.uncertain_params = [m.p1, m.p2]
        m.working_model.util.first_stage_variables = []

        m.working_model.util.first_stage_variables = []
        config = Block()
        config.decision_rule_order = 1
        config.objective_focus = ObjectiveType.nominal
        config.global_solver = SolverFactory('baron')
        config.uncertain_params = m.working_model.util.uncertain_params
        config.tee = False

        add_decision_rule_variables(model_data=m, config=config)
        add_decision_rule_constraints(model_data=m, config=config)

        # === Make master_type model
        master = ConcreteModel()
        master.scenarios = Block(NonNegativeIntegers, NonNegativeIntegers)
        master.scenarios[0, 0].transfer_attributes_from(m.working_model.clone())
        master.scenarios[0, 0].first_stage_objective = 0
        master.scenarios[0, 0].second_stage_objective = Expression(expr=(master.scenarios[0, 0].util.second_stage_variables[0] - 1)**2 +
                                    (master.scenarios[0, 0].util.second_stage_variables[1] - 1)**2)
        master.obj = Objective(expr=master.scenarios[0, 0].second_stage_objective)
        master_data = MasterProblemData()
        master_data.master_model = master
        master_data.master_model.const_efficiency_applied = False
        master_data.master_model.linear_efficiency_applied = False
        results = minimize_dr_vars(model_data=master_data, config=config)

        self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal,
                         msg="Minimize dr norm did not solve to optimality.")

    @unittest.skipUnless(
        SolverFactory('baron').available(exception_flag=False) and SolverFactory('baron').license_is_valid(),
        "Global NLP solver is not available and licensed.")
    def test_identifying_violating_param_realization(self):
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u**(0.5) - m.x2 * m.u <= 2)
        m.con2 = Constraint(expr=m.x1 ** 2 - m.x2 ** 2 * m.u == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

        # Define the uncertainty set
        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        results = pyros_solver.solve(model=m,
                                     first_stage_variables=[m.x1, m.x2],
                                     second_stage_variables=[],
                                     uncertain_params=[m.u],
                                     uncertainty_set=interval,
                                     local_solver=local_subsolver,
                                     global_solver=global_subsolver,
                                     options={
                                         "objective_focus": ObjectiveType.worst_case,
                                         "solve_master_globally": True
                                     })

        self.assertEqual(results.pyros_termination_condition, pyrosTerminationCondition.robust_optimal,
                         msg="Did not identify robust optimal solution to problem instance.")
        self.assertGreater(results.iterations, 0,
                         msg="Robust infeasible model terminated in 0 iterations (nominal case).")

    @unittest.skipUnless(
        SolverFactory('baron').available(exception_flag=False) and SolverFactory('baron').license_is_valid(),
        "Global NLP solver is not available and licensed.")
    def test_terminate_with_max_iter(self):
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u**(0.5) - m.x2 * m.u <= 2)
        m.con2 = Constraint(expr=m.x1 ** 2 - m.x2 ** 2 * m.u == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

        # Define the uncertainty set
        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        results = pyros_solver.solve(model=m,
                                     first_stage_variables=[m.x1],
                                     second_stage_variables=[m.x2],
                                     uncertain_params=[m.u],
                                     uncertainty_set=interval,
                                     local_solver=local_subsolver,
                                     global_solver=global_subsolver,
                                     options={
                                         "objective_focus": ObjectiveType.worst_case,
                                         "solve_master_globally": True,
                                         "max_iter":1,
                                         "decision_rule_order":2
                                     })

        self.assertEqual(results.pyros_termination_condition, pyrosTerminationCondition.max_iter,
                         msg="Returned termination condition is not return max_iter.")

    @unittest.skipUnless(
        SolverFactory('baron').available(exception_flag=False) and SolverFactory('baron').license_is_valid(),
        "Global NLP solver is not available and licensed.")
    def test_terminate_with_time_limit(self):
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u**(0.5) - m.x2 * m.u <= 2)
        m.con2 = Constraint(expr=m.x1 ** 2 - m.x2 ** 2 * m.u == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

        # Define the uncertainty set
        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        results = pyros_solver.solve(model=m,
                                     first_stage_variables=[m.x1, m.x2],
                                     second_stage_variables=[],
                                     uncertain_params=[m.u],
                                     uncertainty_set=interval,
                                     local_solver=local_subsolver,
                                     global_solver=global_subsolver,
                                     options={
                                         "objective_focus": ObjectiveType.worst_case,
                                         "solve_master_globally": True,
                                         "time_limit": 0.001
                                     })

        self.assertEqual(results.pyros_termination_condition, pyrosTerminationCondition.time_out,
                         msg="Returned termination condition is not return time_out.")

    @unittest.skipUnless(
        SolverFactory('baron').available(exception_flag=False) and SolverFactory('baron').license_is_valid(),
        "Global NLP solver is not available and licensed.")
    def test_discrete_separation(self):
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u**(0.5) - m.x2 * m.u <= 2)
        m.con2 = Constraint(expr=m.x1 ** 2 - m.x2 ** 2 * m.u == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

        # Define the uncertainty set
        discrete_scenarios = DiscreteScenarioSet(scenarios=[[0.25], [2.0], [1.125]])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        results = pyros_solver.solve(model=m,
                                     first_stage_variables=[m.x1, m.x2],
                                     second_stage_variables=[],
                                     uncertain_params=[m.u],
                                     uncertainty_set=discrete_scenarios,
                                     local_solver=local_subsolver,
                                     global_solver=global_subsolver,
                                     options={
                                         "objective_focus": ObjectiveType.worst_case,
                                         "solve_master_globally": True
                                     })

        self.assertEqual(results.pyros_termination_condition, pyrosTerminationCondition.robust_optimal,
                         msg="Returned termination condition is not return robust_optimal.")

    @unittest.skipUnless(
        SolverFactory('baron').available(exception_flag=False) and SolverFactory('baron').license_is_valid(),
        "Global NLP solver is not available and licensed.")
    def test_higher_order_decision_rules(self):
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con1 = Constraint(expr=m.x1 * m.u ** (0.5) - m.x2 * m.u <= 2)
        m.con2 = Constraint(expr=m.x1 ** 2 - m.x2 ** 2 * m.u == m.x3)

        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

        # Define the uncertainty set
        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        results = pyros_solver.solve(model=m,
                                     first_stage_variables=[m.x1],
                                     second_stage_variables=[m.x2],
                                     uncertain_params=[m.u],
                                     uncertainty_set=interval,
                                     local_solver=local_subsolver,
                                     global_solver=global_subsolver,
                                     options={
                                         "objective_focus": ObjectiveType.worst_case,
                                         "solve_master_globally": True,
                                         "decision_rule_order":2
                                     })

        self.assertEqual(results.pyros_termination_condition, pyrosTerminationCondition.robust_optimal,
                         msg="Returned termination condition is not return robust_optimal.")

    @unittest.skipUnless(
        SolverFactory('baron').available(exception_flag=False) and SolverFactory('baron').license_is_valid(),
        "Global NLP solver is not available and licensed.")
    def test_coefficient_matching_solve(self):

        # Write the deterministic Pyomo model
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con = Constraint(expr=m.u ** (0.5) * m.x1 - m.u * m.x2 <= 2)
        m.eq_con = Constraint(expr =  m.u**2 * (m.x2- 1) + m.u * (m.x1**3 + 0.5) - 5 * m.u * m.x1 * m.x2 + m.u * (m.x1 + 2) == 0)
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        results = pyros_solver.solve(model=m,
                                     first_stage_variables=[m.x1, m.x2],
                                     second_stage_variables=[],
                                     uncertain_params=[m.u],
                                     uncertainty_set=interval,
                                     local_solver=local_subsolver,
                                     global_solver=global_subsolver,
                                     options={
                                         "objective_focus": ObjectiveType.worst_case,
                                         "solve_master_globally": True
                                     })

        self.assertEqual(results.pyros_termination_condition, pyrosTerminationCondition.robust_optimal,
                         msg="Non-optimal termination condition from robust feasible coefficient matching problem.")
        self.assertAlmostEqual(results.final_objective_value, 6.0394, 2, msg="Incorrect objective function value.")

    def test_coefficient_matching_robust_infeasible_proof_in_pyros(self):
        # Write the deterministic Pyomo model
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con = Constraint(expr=m.u ** (0.5) * m.x1 - m.u * m.x2 <= 2)
        m.eq_con = Constraint(expr =  m.u * (m.x1**3 + 0.5) - 5 * m.u * m.x1 * m.x2 + m.u * (m.x1 + 2) + m.u**2 == 0)
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver

        results = pyros_solver.solve(model=m,
                                     first_stage_variables=[m.x1, m.x2],
                                     second_stage_variables=[],
                                     uncertain_params=[m.u],
                                     uncertainty_set=interval,
                                     local_solver=local_subsolver,
                                     global_solver=global_subsolver,
                                     options={
                                         "objective_focus": ObjectiveType.worst_case,
                                         "solve_master_globally": True
                                     })

        self.assertEqual(results.pyros_termination_condition, pyrosTerminationCondition.robust_infeasible,
                         msg="Robust infeasible problem not identified via coefficient matching.")

    def test_coefficient_matching_nonlinear_expr(self):
        # Write the deterministic Pyomo model
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.u = Param(initialize=1.125, mutable=True)

        m.con = Constraint(expr=m.u ** (0.5) * m.x1 - m.u * m.x2 <= 2)
        m.eq_con = Constraint(expr =  m.u**2 * (m.x2- 1) + m.u * (m.x1**3 + 0.5) - 5 * m.u * m.x1 * m.x2 + m.u * (m.x1 + 2) == 0)
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)

        interval = BoxSet(bounds=[(0.25, 2)])

        # Instantiate the PyROS solver
        pyros_solver = SolverFactory("pyros")

        # Define subsolvers utilized in the algorithm
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory("baron")

        # Call the PyROS solver
        try:
            results = pyros_solver.solve(model=m,
                                         first_stage_variables=[m.x1],
                                         second_stage_variables=[m.x2],
                                         uncertain_params=[m.u],
                                         uncertainty_set=interval,
                                         local_solver=local_subsolver,
                                         global_solver=global_subsolver,
                                         options={
                                             "objective_focus": ObjectiveType.worst_case,
                                             "solve_master_globally": True,
                                             "decision_rule_order":1
                                         })
        except:
            self.assertRaises(ValueError, msg="ValueError should be "
                                              "raised for general nonlinear expressions in h(x,z,q)=0 constraints.")




if __name__ == "__main__":
    unittest.main()
