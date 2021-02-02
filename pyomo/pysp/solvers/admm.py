#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
import sys
import time
import math

# TODO: run y update on the server side so that bounds can be
#       returned efficiently if enabled
# TODO: handle multi-stage
#       (this has to do with fixing distributed variable id
#        creation on the new scenario tree manager)

from pyomo.common.collections import OrderedDict
from pyomo.core import (Block, Set, Expression, Param, maximize)
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_register_unique_option,
                                    safe_declare_common_option,
                                    safe_declare_unique_option,
                                    _domain_positive,
                                    _domain_must_be_str)
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command)
from pyomo.pysp.scenariotree.manager import \
    (InvocationType,
     ScenarioTreeManager,
     ScenarioTreeManagerFactory)
from pyomo.pysp.scenariotree.manager_solver import \
    ScenarioTreeManagerSolverFactory
from pyomo.pysp.phutils import indexToString
from pyomo.pysp.solvers.spsolver import (SPSolver,
                                         SPSolverResults,
                                         SPSolverFactory)

from six.moves import xrange

thisfile = os.path.abspath(__file__)
thisfile.replace(".pyc","").replace(".py","")

_admm_group_label = "ADMM Options"
_rho_group_label = "Rho Strategy Options"

class RhoStrategy(PySPConfiguredObject):
    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()
        return options
    def __init__(self, *args, **kwds):
        super(RhoStrategy, self).__init__(*args, **kwds)
    def initialize(self, manager, x, y, z, rho):
        raise NotImplementedError
    def update_rho(self, manager, x, y, z, rho):
        raise NotImplementedError

class FixedRhoStrategy(RhoStrategy, PySPConfiguredObject):
    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()
        return options
    def __init__(self, *args, **kwds):
        super(FixedRhoStrategy, self).__init__(*args, **kwds)
    def initialize(self, manager, x, y, z, rho):
        pass
    def update_rho(self, manager, x, y, z, rho):
        pass

class AdaptiveRhoStrategy(RhoStrategy, PySPConfiguredObject):
    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()
        safe_declare_common_option(options, "verbose")
        return options
    def __init__(self, *args, **kwds):
        super(AdaptiveRhoStrategy, self).__init__(*args, **kwds)
        self._z_previous = None
        self._primal_residual_norm = None
        self._dual_residual_norm = None
        self._mu = 10.0
        self._tol = 1e-5
        self._rho_increase_factor = 2.0
        self._rho_decrease_factor = 0.5

    def snapshot_z(self, manager, z):
        for stage in manager.scenario_tree.stages[:-1]:
            for tree_node in stage.nodes:
                z_prev_node = self._z_previous[tree_node.name]
                z_node = z[tree_node.name]
                for id_ in tree_node._standard_variable_ids:
                    z_prev_node[id_] = z_node[id_]

    def compute_primal_residual_norm(self, manager, x, z):
        for stage in manager.scenario_tree.stages[:-1]:
            for tree_node in stage.nodes:
                prnorm_node = self._primal_residual_norm[tree_node.name]
                z_node = z[tree_node.name]
                for id_ in tree_node._standard_variable_ids:
                    prnorm_var = 0.0
                    for scenario in tree_node._scenarios:
                        prnorm_var += \
                            (x[scenario.name][tree_node.name][id_] - \
                             z_node[id_])**2
                    prnorm_node[id_] = math.sqrt(prnorm_var)

    def compute_dual_residual_norm(self, manager, z, rho):
        for stage in manager.scenario_tree.stages[:-1]:
            for tree_node in stage.nodes:
                drnorm_node = self._dual_residual_norm[tree_node.name]
                z_prev_node = self._z_previous[tree_node.name]
                z_node = z[tree_node.name]
                rho_node = rho[tree_node.name]
                for id_ in tree_node._standard_variable_ids:
                    drnorm_node[id_] = \
                        math.sqrt(len(tree_node.scenarios)) * \
                        rho_node[id_] * \
                        abs(z_node[id_] - z_prev_node[id_])

    def initialize(self, manager, x, y, z, rho):
        self._z_previous = {}
        self._primal_residual_norm = {}
        self._dual_residual_norm = {}
        for stage in manager.scenario_tree.stages[:-1]:
            for tree_node in stage.nodes:
                self._z_previous[tree_node.name] = {}
                self._primal_residual_norm[tree_node.name] = {}
                self._dual_residual_norm[tree_node.name] = {}
        self.snapshot_z(manager, z)

    def update_rho(self, manager, x, y, z, rho):

        first_line = ("Updating Rho Values:\n%21s %25s %16s %16s %16s"
                      % ("Action",
                         "Variable",
                         "Primal Residual",
                         "Dual Residual",
                         "New Rho"))
        first_print = True
        self.compute_primal_residual_norm(manager, x, z)
        self.compute_dual_residual_norm(manager, z, rho)
        verbose = self.get_option("verbose")
        for stage in manager.scenario_tree.stages[:-1]:
            for tree_node in stage.nodes:
                prnorm_node = self._primal_residual_norm[tree_node.name]
                drnorm_node = self._dual_residual_norm[tree_node.name]
                rho_node = rho[tree_node.name]
                for id_ in tree_node._standard_variable_ids:
                    name, index = tree_node._variable_ids[id_]
                    prnorm_var = prnorm_node[id_]
                    drnorm_var = drnorm_node[id_]
                    action = None
                    if (prnorm_var > self._mu * drnorm_var) and \
                       (prnorm_var > self._tol):
                        rho_node[id_] *= self._rho_increase_factor
                        action = "Increasing"
                    elif (drnorm_var > self._mu * prnorm_var) and \
                         (drnorm_var > self._tol):
                        rho_node[id_] *= self._rho_decrease_factor
                        action = "Decreasing"
                    if verbose:
                        if action is not None:
                            if first_print:
                                first_print = False
                                print(first_line)
                            print("%21s %25s %16g %16g %16g"
                                  % (action,
                                     name+indexToString(index),
                                     prnorm_var,
                                     drnorm_var,
                                     rho_node[id_]))

        self.snapshot_z(manager, z)

def RhoStrategyFactory(name, *args, **kwds):
    if name in RhoStrategyFactory.registered_types:
        return RhoStrategyFactory.registered_types[name](*args, **kwds)
    raise ValueError("No rho strategy registered with name: %s"
                     % (name))
RhoStrategyFactory.registered_types = {}
RhoStrategyFactory.registered_types['fixed'] = FixedRhoStrategy
RhoStrategyFactory.registered_types['adaptive'] = AdaptiveRhoStrategy


#
# NOTE: A function names beginning with EXTERNAL_ indicate
#       that the function may be executed by a different
#       process, so they should not rely on any state
#       maintained on the client scenario tree manager or
#       globals in this file.
#

def EXTERNAL_cleanup_for_admm(manager,
                              scenario):
    if manager.get_option("verbose"):
        print("Cleaning up admm modifications to scenario %s"
              % (scenario.name))
    instance = scenario._instance
    assert hasattr(scenario._instance, ".admm")
    scenario._instance.del_component(".admm")

    # The objective has changed so flag this if necessary.
    if manager.preprocessor is not None:
        manager.preprocessor.objective_updated[scenario.name] = True

def EXTERNAL_initialize_for_admm(manager,
                                 scenario):
    if manager.get_option("verbose"):
        print("Initializing scenario %s for admm algorithm"
              % (scenario.name))
    admm_block = Block(concrete=True)
    assert not hasattr(scenario._instance, ".admm")
    scenario._instance.add_component(".admm", admm_block)

    # Augment the objective with lagrangian and penalty terms
    # and weight the original objective by the scenario probability.
    # The langrangian and penalty terms will be computed after
    # the parameters are created.
    user_cost_expression = scenario._instance_cost_expression
    admm_block.cost_expression = Expression(initialize=\
        scenario.probability * user_cost_expression)
    admm_block.lagrangian_expression = Expression(initialize=0.0)
    admm_block.penalty_expression = Expression(initialize=0.0)
    # these are used in the objective, they can be toggled
    # between the expression above or something else (e.g., 0.0)
    admm_block.cost_term = Expression(
        initialize=admm_block.cost_expression)
    admm_block.lagrangian_term = Expression(
        initialize=admm_block.lagrangian_expression)
    admm_block.penalty_term = Expression(
        initialize=admm_block.penalty_expression)
    objective_direction = 1
    if manager.objective_sense == maximize:
        objective_direction = -1
    scenario._instance_objective.expr = \
        admm_block.cost_term + \
        admm_block.lagrangian_term * objective_direction + \
        admm_block.penalty_term * objective_direction

    # add objective parameters to admm block
    for tree_node in scenario.node_list[:-1]:
        assert not tree_node.is_leaf_node()
        node_block = Block(concrete=True)
        admm_block.add_component(tree_node.name,
                                 node_block)
        node_block.node_index_set = Set(
            ordered=True,
            initialize=sorted(tree_node._standard_variable_ids))
        node_block.z = Param(node_block.node_index_set,
                             initialize=0.0,
                             mutable=True)
        node_block.y = Param(node_block.node_index_set,
                             initialize=0.0,
                             mutable=True)
        node_block.rho = Param(node_block.node_index_set,
                               initialize=0.0,
                               mutable=True)

        for id_ in node_block.node_index_set:
            varname, index = tree_node._variable_ids[id_]
            var = scenario._instance.find_component(varname)[index]
            admm_block.lagrangian_expression.expr += \
                node_block.y[id_] * (var - node_block.z[id_])
            admm_block.penalty_expression.expr += \
                (node_block.rho[id_] / 2.0) * (var - node_block.z[id_])**2

    # The objective has changed so flag this if necessary.
    if manager.preprocessor is not None:
        manager.preprocessor.objective_updated[scenario.name] = True

def EXTERNAL_activate_lagrangian_term(manager,
                                      scenario):
    if manager.get_option("verbose"):
        print("Activating admm lagrangian term on scenario %s"
              % (scenario.name))
    assert hasattr(scenario._instance, ".admm")
    admm_block = getattr(scenario._instance, ".admm")
    admm_block.lagrangian_term.expr = \
        admm_block.lagrangian_expression
    # The objective has changed so flag this if necessary.
    if manager.preprocessor is not None:
        manager.preprocessor.objective_updated[scenario.name] = True

def EXTERNAL_deactivate_lagrangian_term(manager,
                                        scenario):
    if manager.get_option("verbose"):
        print("Deactivating admm lagrangian term on scenario %s"
              % (scenario.name))
    assert hasattr(scenario._instance, ".admm")
    admm_block = getattr(scenario._instance, ".admm")
    admm_block.lagrangian_term.expr = 0.0
    # The objective has changed so flag this if necessary.
    if manager.preprocessor is not None:
        manager.preprocessor.objective_updated[scenario.name] = True

def EXTERNAL_activate_penalty_term(manager,
                                   scenario):
    if manager.get_option("verbose"):
        print("Activating admm penalty term on scenario %s"
              % (scenario.name))
    assert hasattr(scenario._instance, ".admm")
    admm_block = getattr(scenario._instance, ".admm")
    admm_block.penalty_term.expr = \
        admm_block.penalty_expression
    # The objective has changed so flag this if necessary.
    if manager.preprocessor is not None:
        manager.preprocessor.objective_updated[scenario.name] = True

def EXTERNAL_deactivate_penalty_term(manager,
                                     scenario):
    if manager.get_option("verbose"):
        print("Deactivating admm penalty term on scenario %s"
              % (scenario.name))
    assert hasattr(scenario._instance, ".admm")
    admm_block = getattr(scenario._instance, ".admm")
    admm_block.penalty_term.expr = 0.0
    # The objective has changed so flag this if necessary.
    if manager.preprocessor is not None:
        manager.preprocessor.objective_updated[scenario.name] = True

def EXTERNAL_activate_probability_weighted_cost_term(manager,
                                                     scenario):
    if manager.get_option("verbose"):
        print("Activating admm probability-weighted cost term on scenario %s"
              % (scenario.name))
    assert hasattr(scenario._instance, ".admm")
    admm_block = getattr(scenario._instance, ".admm")
    admm_block.cost_term.expr = \
        admm_block.cost_expression
    # The objective has changed so flag this if necessary.
    if manager.preprocessor is not None:
        manager.preprocessor.objective_updated[scenario.name] = True

def EXTERNAL_deactivate_probability_weighted_cost_term(manager,
                                                       scenario):
    if manager.get_option("verbose"):
        print("Deactivating admm probability-weighted cost term on scenario %s"
              % (scenario.name))
    assert hasattr(scenario._instance, ".admm")
    admm_block = getattr(scenario._instance, ".admm")
    admm_block.cost_term.expr = \
        scenario._instance_cost_expression
    # The objective has changed so flag this if necessary.
    if manager.preprocessor is not None:
        manager.preprocessor.objective_updated[scenario.name] = True

def EXTERNAL_update_rho(manager,
                        scenario,
                        rho):
    if manager.get_option("verbose"):
        print("Updating admm rho parameter for scenario %s"
              % (scenario.name))
    assert hasattr(scenario._instance, ".admm")
    admm_block = getattr(scenario._instance, ".admm")
    for tree_node in scenario.node_list[:-1]:
        assert not tree_node.is_leaf_node()
        node_block = admm_block.find_component(tree_node.name)
        assert node_block is not None
        node_block.rho.store_values(rho[tree_node.name])
    # The objective has changed so flag this if necessary.
    if manager.preprocessor is not None:
        manager.preprocessor.objective_updated[scenario.name] = True

def EXTERNAL_update_y(manager,
                      scenario,
                      y):
    if manager.get_option("verbose"):
        print("Updating admm y parameter for scenario %s"
              % (scenario.name))
    assert hasattr(scenario._instance, ".admm")
    admm_block = getattr(scenario._instance, ".admm")
    for tree_node in scenario.node_list[:-1]:
        assert not tree_node.is_leaf_node()
        node_block = admm_block.find_component(tree_node.name)
        assert node_block is not None
        node_block.y.store_values(y[tree_node.name])
    # The objective has changed so flag this if necessary.
    if manager.preprocessor is not None:
        manager.preprocessor.objective_updated[scenario.name] = True

def EXTERNAL_update_z(manager,
                      scenario,
                      z):
    if manager.get_option("verbose"):
        print("Updating admm z parameter for scenario %s"
              % (scenario.name))
    assert hasattr(scenario._instance, ".admm")
    admm_block = getattr(scenario._instance, ".admm")
    for tree_node in scenario.node_list[:-1]:
        assert not tree_node.is_leaf_node()
        node_block = admm_block.find_component(tree_node.name)
        assert node_block is not None
        node_block.z.store_values(z[tree_node.name])
    # The objective has changed so flag this if necessary.
    if manager.preprocessor is not None:
        manager.preprocessor.objective_updated[scenario.name] = True

class ADMMAlgorithm(PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()
        safe_declare_common_option(options,
                                   "verbose",
                                   ap_group=_admm_group_label)
        ScenarioTreeManagerSolverFactory.register_options(
            options,
            options_prefix="subproblem_",
            setup_argparse=False)
        return options

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._manager is not None:
            self.cleanup_subproblems()
        if self._manager_solver is not None:
            self._manager_solver.close()
        self._manager_solver = None
        self._manager = None

    def __init__(self, manager, *args, **kwds):
        super(ADMMAlgorithm, self).__init__(*args, **kwds)

        if not isinstance(manager, ScenarioTreeManager):
            raise TypeError("%s requires an instance of the "
                            "ScenarioTreeManagerSolver interface as the "
                            "second argument" % (self.__class__.__name__))
        if not manager.initialized:
            raise ValueError("%s requires a scenario tree "
                             "manager that has been fully initialized"
                             % (self.__class__.__name__))

        self._manager = manager
        self._manager_solver = ScenarioTreeManagerSolverFactory(
            self._manager,
            self._options,
            options_prefix="subproblem_")

        self.initialize_subproblems()

    def initialize_subproblems(self):
        if self.get_option("verbose"):
            print("Initializing subproblems for admm")
        self._manager.invoke_function(
            "EXTERNAL_initialize_for_admm",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            oneway_call=True)

    def cleanup_subproblems(self):
        if self.get_option("verbose"):
            print("Cleaning up subproblems for admm")
        self._manager.invoke_function(
            "EXTERNAL_cleanup_for_admm",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            oneway_call=True)

    def activate_lagrangian_term(self):
        self._manager.invoke_function(
            "EXTERNAL_activate_lagrangian_term",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            oneway_call=True)

    def deactivate_lagrangian_term(self):
        self._manager.invoke_function(
            "EXTERNAL_deactivate_lagrangian_term",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            oneway_call=True)

    def activate_penalty_term(self):
        self._manager.invoke_function(
            "EXTERNAL_activate_penalty_term",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            oneway_call=True)

    def deactivate_penalty_term(self):
        self._manager.invoke_function(
            "EXTERNAL_deactivate_penalty_term",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            oneway_call=True)

    def activate_probability_weighted_cost_term(self):
        self._manager.invoke_function(
            "EXTERNAL_activate_probability_weighted_cost_term",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            oneway_call=True)

    def deactivate_probability_weighted_cost_term(self):
        self._manager.invoke_function(
            "EXTERNAL_deactivate_probability_weighted_cost_term",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            oneway_call=True)

    def run_x_update(self, x, y, z, rho):
        self._manager.invoke_function(
            "EXTERNAL_update_rho",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            function_args=(rho,),
            oneway_call=True)
        for scenario in self._manager.scenario_tree.scenarios:
            self._manager.invoke_function(
                "EXTERNAL_update_y",
                thisfile,
                invocation_type=InvocationType.OnScenario(scenario.name),
                function_args=(y[scenario.name],),
                oneway_call=True)
        self._manager.invoke_function(
            "EXTERNAL_update_z",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            function_args=(z,),
            oneway_call=True)

        self._manager_solver.solve_scenarios()

        objective = 0.0
        for scenario in self._manager.scenario_tree.scenarios:
            x_scenario = x[scenario.name]
            x_scenario_solution = scenario._x
            objective += scenario._objective
            for tree_node in scenario.node_list[:-1]:
                assert not tree_node.is_leaf_node()
                x_node = x_scenario[tree_node.name]
                x_node_solution = x_scenario_solution[tree_node.name]
                for id_ in tree_node._standard_variable_ids:
                    x_node[id_] = x_node_solution[id_]

        return objective

    def run_z_update(self, x, y, z, rho):
        primal_residual = 0.0
        dual_residual = 0.0
        x_scale = 0.0
        z_scale = 0.0
        for stage in self._manager.scenario_tree.stages[:-1]:
            for tree_node in stage.nodes:
                z_node = z[tree_node.name]
                rho_node = rho[tree_node.name]
                for id_ in tree_node._standard_variable_ids:
                    z_var_prev = z_node[id_]
                    rho_var = rho_node[id_]
                    z_var = 0.0
                    for scenario in tree_node.scenarios:
                        z_var += scenario._x[tree_node.name][id_]
                    z_var /= float(len(tree_node.scenarios))
                    if tree_node.is_variable_binary(id_) or \
                       tree_node.is_variable_integer(id_):
                        z_var = int(round(z_var))
                    dual_residual += len(tree_node.scenarios) * \
                                     rho_var**2 * \
                                     (z_var - z_var_prev)**2
                    for scenario in tree_node.scenarios:
                        x_var = scenario._x[tree_node.name][id_]
                        x_scale += x_var**2
                        primal_residual += (x_var - z_var)**2
                    z_node[id_] = z_var
                    z_scale += z_var**2
        return (math.sqrt(primal_residual),
                math.sqrt(dual_residual),
                math.sqrt(x_scale),
                math.sqrt(z_scale))

    def run_y_update(self, x, y, z, rho):
        y_scale = 0.0
        for scenario in self._manager.scenario_tree.scenarios:
            y_scenario = y[scenario.name]
            x_scenario = scenario._x
            for tree_node in scenario.node_list[:-1]:
                assert not tree_node.is_leaf_node()
                rho_node = rho[tree_node.name]
                z_node = z[tree_node.name]
                y_node = y_scenario[tree_node.name]
                x_node = x_scenario[tree_node.name]
                for id_ in tree_node._standard_variable_ids:
                    y_node[id_] += rho_node[id_] * (x_node[id_] - z_node[id_])
                    y_scale += y_node[id_]**2
        return math.sqrt(y_scale)

    def initialize_algorithm_data(self,
                                  rho_init=1.0,
                                  y_init=0.0,
                                  z_init=0.0):

        # used to check dual-feasibility of initial y
        y_sum = {}
        for stage in self._manager.scenario_tree.stages[:-1]:
            for tree_node in stage.nodes:
                y_sum_node = y_sum[tree_node.name] = \
                    dict((id_, 0.0) for id_ in tree_node._standard_variable_ids)

        x = {}
        y = {}
        for scenario in self._manager.scenario_tree.scenarios:
            x_scenario = x[scenario.name] = {}
            y_scenario = y[scenario.name] = {}
            for tree_node in scenario.node_list[:-1]:
                assert not tree_node.is_leaf_node()
                x_node = x_scenario[tree_node.name] = {}
                y_node = y_scenario[tree_node.name] = {}
                y_sum_node = y_sum[tree_node.name]
                for id_ in tree_node._standard_variable_ids:
                    x_node[id_] = None
                    if type(y_init) is dict:
                        y_node[id_] = y_init[scenario.name][tree_node.name][id_]
                    else:
                        y_node[id_] = y_init
                    y_sum_node[id_] += y_node[id_]

        # check dual-feasibility of y
        for stage in self._manager.scenario_tree.stages[:-1]:
            for tree_node in stage.nodes:
                y_sum_node = y_sum[tree_node.name]
                for id_ in tree_node._standard_variable_ids:
                    if abs(y_sum_node[id_]) > 1e-6:
                        name, index = tree_node._variable_ids[id_]
                        raise ValueError(
                            "Initial lagrange multipler estimates for non-"
                            "anticipative variable %s do not sum to zero: %s"
                            % (name+indexToString(index), repr(y_sum_node[id_])))

        rho = {}
        z = {}
        for stage in self._manager.scenario_tree.stages[:-1]:
            for tree_node in stage.nodes:
                z_node = z[tree_node.name] = {}
                rho_node = rho[tree_node.name] = {}
                for id_ in tree_node._standard_variable_ids:
                    if type(rho_init) is dict:
                        rho_node[id_] = rho_init[tree_node.name][id_]
                    else:
                        rho_node[id_] = rho_init
                    if type(z_init) is dict:
                        z_node[id_] = z_init[tree_node.name][id_]
                    else:
                        z_node[id_] = z_init

        return rho, x, y, z

    def compute_nodevector_norm(self, v):
        vnorm = 0.0
        for stage in self._manager.scenario_tree.stages[:-1]:
            for tree_node in stage.nodes:
                v_node = v[tree_node.name]
                for id_ in tree_node._standard_variable_ids:
                    vnorm += v_node[id_]**2
        return math.sqrt(vnorm)

class ADMMSolver(SPSolver, PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()
        safe_declare_common_option(options,
                                   "max_iterations",
                                   ap_group=_admm_group_label)
        safe_declare_unique_option(
            options,
            "primal_residual_relative_tolerance",
            PySPConfigValue(
                1.0e-4,
                domain=_domain_positive,
                description=(
                    "Relative primal-residual tolerance. Default is 1e-4."
                ),
                doc=None,
                visibility=0),
            ap_group=_admm_group_label)
        safe_declare_unique_option(
            options,
            "dual_residual_relative_tolerance",
            PySPConfigValue(
                1.0e-4,
                domain=_domain_positive,
                description=(
                    "Relative dual-residual tolerance. Default is 1e-4."
                ),
                doc=None,
                visibility=0),
            ap_group=_admm_group_label)
        ADMMAlgorithm._declare_options(options)
        for rstype in RhoStrategyFactory.registered_types.values():
            rstype._declare_options(options)
        assert 'adaptive' in RhoStrategyFactory.registered_types
        safe_declare_unique_option(
            options,
            "rho_strategy",
            PySPConfigValue(
                'adaptive',
                domain=_domain_must_be_str,
                description=(
                    "Rho update strategy. Choices are: %s. Default is 'adaptive'."
                    % (str(sorted(RhoStrategyFactory.registered_types.keys())))
                ),
                doc=None,
                visibility=0),
            ap_group=_admm_group_label)
        return options

    def __init__(self):
        super(ADMMSolver, self).__init__(self.register_options())
        self.set_options_to_default()
        # The following attributes will be modified by the
        # solve() method. For users that are scripting, these
        # can be accessed after the solve() method returns.
        # They will be reset each time solve() is called.
        ############################################
        self.objective_history = OrderedDict()
        self.primal_residual_history = OrderedDict()
        self.dual_residual_history = OrderedDict()
        self.iterations = None
        ############################################

    def set_options_to_default(self):
        self._options = self.register_options()

    @property
    def options(self):
        return self._options

    @property
    def name(self):
        return "admm"

    def _solve_impl(self,
                    sp,
                    rho=1.0,
                    y_init=0.0,
                    z_init=0.0,
                    output_solver_log=False):

        if len(sp.scenario_tree.stages) > 2:
            raise ValueError(
                "ADMM solver does not yet handle more "
                "than 2 time-stages")

        start_time = time.time()

        scenario_tree = sp.scenario_tree
        num_scenarios = len(scenario_tree.scenarios)
        num_stages = len(scenario_tree.stages)
        num_na_nodes = 0
        num_na_variables = 0
        num_na_continuous_variables = 0
        num_na_binary_variables = 0
        num_na_integer_variables = 0
        for stage in sp.scenario_tree.stages[:-1]:
            for tree_node in stage.nodes:
                num_na_nodes += 1
                num_na_variables += len(tree_node._standard_variable_ids)
                for id_ in tree_node._standard_variable_ids:
                    if tree_node.is_variable_binary(id_):
                        num_na_binary_variables += 1
                    elif tree_node.is_variable_integer(id_):
                        num_na_integer_variables += 1
                    else:
                        num_na_continuous_variables += 1

#        print("-"*20)
#        print("Problem Statistics".center(20))
#        print("-"*20)
#        print("Total number of scenarios.................: %10s"
#              % (num_scenarios))
#        print("Total number of time stages...............: %10s"
#              % (num_stages))
#        print("Total number of non-anticipative nodes....: %10s"
#              % (num_na_nodes))
#        print("Total number of non-anticipative variables: %10s\n#"
#              "                                continuous: %10s\n#"
#              "                                    binary: %10s\n#"
#              "                                   integer: %10s"
#              % (num_na_variables,
#                 num_na_continuous_variables,
#                 num_na_binary_variables,
#                 num_na_integer_variables))

        rel_tol_primal = \
            self.get_option("primal_residual_relative_tolerance")
        rel_tol_dual = \
            self.get_option("dual_residual_relative_tolerance")
        max_iterations = \
            self.get_option("max_iterations")

        self.objective_history = OrderedDict()
        self.primal_residual_history = OrderedDict()
        self.dual_residual_history = OrderedDict()
        self.iterations = 0
        if output_solver_log:
            print("")
        label_cols = ("{0:^4} {1:>16} {2:>8} {3:>8} {4:>12}".format(
            "iter","objective","pr_res","du_res","lg(||rho||)"))
        with ADMMAlgorithm(sp, self._options) as admm:
            rho, x, y, z = admm.initialize_algorithm_data(rho_init=rho,
                                                          y_init=y_init,
                                                          z_init=z_init)
            rho_strategy = RhoStrategyFactory(
                self.get_option("rho_strategy"),
                self._options)
            rho_strategy.initialize(sp, x, y, z, rho)
            for i in xrange(max_iterations):

                objective = \
                    admm.run_x_update(x, y, z, rho)
                (unscaled_primal_residual,
                 unscaled_dual_residual,
                 x_scale,
                 z_scale) = \
                    admm.run_z_update(x, y, z, rho)
                y_scale = \
                    admm.run_y_update(x, y, z, rho)

                # we've completed another iteration
                self.iterations += 1

                # check for convergence
                primal_rel_scale = max(1.0, x_scale, z_scale)
                dual_rel_scale = max(1.0, y_scale)
                primal_residual = unscaled_primal_residual / \
                                  math.sqrt(num_scenarios) / \
                                  primal_rel_scale
                dual_residual = unscaled_dual_residual / \
                                math.sqrt(num_na_variables) / \
                                dual_rel_scale

                self.objective_history[i] = \
                    objective
                self.primal_residual_history[i] = \
                    primal_residual
                self.dual_residual_history[i] = \
                    dual_residual

                if output_solver_log:
                    if (i % 10) == 0:
                        print(label_cols)
                    print("%4d %16.7e %8.2e %8.2e %12.2e"
                          % (i,
                             objective,
                             primal_residual,
                             dual_residual,
                             math.log(admm.compute_nodevector_norm(rho))))

                if (primal_residual < rel_tol_primal) and \
                   (dual_residual < rel_tol_dual):
                    if output_solver_log:
                        print("\nNumber of Iterations....: %s"
                              % (self.iterations))
                    break
                else:
                    rho_strategy.update_rho(sp, x, y, z, rho)

            else:
                if output_solver_log:
                    print("\nMaximum number of iterations reached: %s"
                          % (max_iterations))

        if output_solver_log:
            print("")
            print("                        {0:^24} {1:^24}".\
                  format("(scaled)", "(unscaled)"))
            print("Objective..........:    {0:^24} {1:^24.16e}".\
                  format("-", objective))
            print("Primal residual....:    {0:^24.16e} {1:^24.16e}".\
                  format(primal_residual, unscaled_primal_residual))
            print("Dual residual......:    {0:^24.16e} {1:^24.16e}".\
                  format(dual_residual, unscaled_dual_residual))
            unscaled_err = unscaled_primal_residual + \
                           unscaled_dual_residual
            err = primal_residual + dual_residual
            print("Overall error......:    {0:^24.16e} {1:^24.16e}".\
                  format(err, unscaled_err))

        results = SPSolverResults()
        results.objective = objective
        results.xhat = z
        return results

def runadmm_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()
    ADMMSolver.register_options(options)
    ScenarioTreeManagerFactory.register_options(options)
    safe_register_common_option(options,
                               "verbose")
    safe_register_common_option(options,
                               "disable_gc")
    safe_register_common_option(options,
                               "profile")
    safe_register_common_option(options,
                               "traceback")
    safe_register_common_option(options,
                                "output_solver_log")
    safe_register_common_option(options,
                               "output_scenario_tree_solution")
    safe_register_unique_option(
        options,
        "default_rho",
        PySPConfigValue(
            1.0,
            domain=_domain_positive,
            description=(
                "The default rho value for all non-anticipative "
                "variables. Default is 1.0."
            ),
            doc=None,
            visibility=0),
        ap_args=("-r", "--default-rho"),
        ap_group=_admm_group_label)

    return options

def runadmm(options):
    """
    Construct a senario tree manager and solve it
    with the ADMM solver.
    """
    start_time = time.time()
    with ScenarioTreeManagerFactory(options) as sp:
        sp.initialize()

        print("")
        print("Running ADMM solver for stochastic "
              "programming problems")
        admm = ADMMSolver()
        admm_options = admm.extract_user_options_to_dict(
            options,
            sparse=True)
        results = admm.solve(
            sp,
            options=admm_options,
            rho=options.default_rho,
            output_solver_log=options.output_solver_log)
        xhat = results.xhat
        del results.xhat
        print("")
        print(results)

        if options.output_scenario_tree_solution:
            print("Final solution (scenario tree format):")
            sp.scenario_tree.snapshotSolutionFromScenarios()
            sp.scenario_tree.pprintSolution()
            sp.scenario_tree.pprintCosts()

    print("")
    print("Total execution time=%.2f seconds"
          % (time.time() - start_time))
    return 0

#
# the main driver routine
#

def main(args=None):
    #
    # Top-level command that executes everything
    #

    #
    # Import plugins
    #
    import pyomo.environ

    #
    # Parse command-line options.
    #
    try:
        options = parse_command_line(
            args,
            runadmm_register_options,
            prog='runadmm',
            description=(
"""Optimize a stochastic program using the Alternating Direction
Method of Multipliers (ADMM) solver."""
            ))

    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(runadmm,
                          options,
                          error_label="runadmm: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

SPSolverFactory.register_solver("admm", ADMMSolver)

if __name__ == "__main__":
    sys.exit(main())
