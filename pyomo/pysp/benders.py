#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

### Ideas
# - Should be easy to warm start the benders script
#   from a history file, so one wouldn't need to start
#   from scratch

### SERIOUS TODO:
# - The ASL interface always returns duals. When the problem is a MIP,
#   they are always zero. We need to make sure that the subproblems
#   are not still MIPs after fixing the first-stage constraints;
#   otherwise, the benders cuts are invalid (because the duals are
#   artificially zero).

### TODOs:
# - feasibility cuts

import os
import logging
import time
import itertools

try:
    from collections import OrderedDict
except ImportError:                         #pragma:nocover
    from ordereddict import OrderedDict

from pyomo.util import pyomo_command
from pyomo.opt import (SolverFactory,
                       TerminationCondition,
                       PersistentSolver,
                       undefined)
from pyomo.core import (value, minimize, Set,
                        Objective, SOSConstraint,
                        Constraint, Var, RangeSet,
                        Expression, Suffix, Reals, Param)
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.beta.list_objects import XConstraintList
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_register_unique_option,
                                    safe_declare_common_option,
                                    safe_declare_unique_option,
                                    _domain_percent,
                                    _domain_nonnegative,
                                    _domain_positive_integer,
                                    _domain_must_be_str,
                                    _domain_unit_interval,
                                    _domain_tuple_of_str,
                                    _domain_tuple_of_str_or_dict)
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command)
from pyomo.pysp.scenariotree.manager import InvocationType
from pyomo.pysp.scenariotree.manager_solver import \
    (ScenarioTreeManagerSolver,
     ScenarioTreeManagerSolverClientSerial,
     ScenarioTreeManagerSolverClientPyro,
     ScenarioTreeManagerFactory)
from pyomo.pysp.phutils import find_active_objective
from pyomo.pysp.ef import create_ef_instance

from six.moves import xrange

thisfile = os.path.abspath(__file__)
thisfile.replace(".pyc","").replace(".py","")

logger = logging.getLogger('pyomo.pysp')

_benders_group_label = "Benders Options"

def EXTERNAL_deactivate_firststage_costs(manager,
                                         scenario):
    assert len(manager.scenario_tree.stages) == 2
    assert scenario in manager.scenario_tree.scenarios
    firststage = manager.scenario_tree.findRootNode().stage
    scenario._instance.find_component(
        "PYSP_BENDERS_STAGE_COST_TERM_"+firststage.name).set_value(0.0)
    if manager.preprocessor is not None:
        manager.preprocessor.objective_updated[scenario.name] = True

def EXTERNAL_activate_firststage_costs(manager,
                                       scenario):
    assert len(manager.scenario_tree.stages) == 2
    assert scenario in manager.scenario_tree.scenarios
    firststage = manager.scenario_tree.findRootNode().stage
    stagecost_var = instance.find_component(
        firststage._cost_variable[0])[firststage._cost_variable[1]]
    scenario._instance.find_component(
        "PYSP_BENDERS_STAGE_COST_TERM_"+firststage.name).set_value(stagecost_var)
    if manager.preprocessor is not None:
        manager.preprocessor.objective_updated[scenario.name] = True

def EXTERNAL_activate_fix_constraints(manager,
                                      scenario):
    assert len(manager.scenario_tree.stages) == 2
    assert scenario in manager.scenario_tree.scenarios
    fix_constraint = scenario._instance.find_component(
        "PYSP_BENDERS_FIX_XHAT_CONSTRAINT")
    fix_constraint.activate()
    if manager.preprocessor is not None:
        preprocess_constraints_list = \
            manager.preprocessor.constraints_updated_list[scenario.name]
        for constraint_data in fix_constraint.values():
            preprocess_constraints_list.append(constraint_data)

def EXTERNAL_deactivate_fix_constraints(manager,
                                        scenario):
    assert len(manager.scenario_tree.stages) == 2
    assert scenario in manager.scenario_tree.scenarios
    fix_constraint = scenario._instance.find_component(
        "PYSP_BENDERS_FIX_XHAT_CONSTRAINT")
    fix_constraint.deactivate()
    # no need to flag the preprocessor

def EXTERNAL_cleanup_from_benders(manager,
                                  scenario):

    instance = scenario._instance

    # restore stage cost expressions
    for node in scenario.node_list:
        stage = node.stage
        cost_term_name = "PYSP_BENDERS_STAGE_COST_TERM_"+stage.name
        assert hasattr(instance, cost_term_name)
        instance.del_component(cost_term_name)
    assert hasattr(scenario, "_instance_cost_expression_old")
    scenario._instance_cost_expression.set_value(
        scenario._instance_cost_expression_old)
    del scenario._instance_cost_expression_old

    if hasattr(scenario, "_remove_dual_at_benders_cleanup"):
        assert scenario._remove_dual_at_benders_cleanup
        del scenario._remove_dual_at_benders_cleanup
        if manager.scenario_tree.contains_bundles():
            found = 0
            for scenario_bundle in manager.scenario_tree.bundles:
                if scenario.name in scenario_bundle.scenario_names:
                    found += 1
                    bundle_instance = \
                        manager._bundle_binding_instance_map[scenario_bundle.name]
                    bundle_instance.del_component("dual")
            assert found == 1
        else:
            instance.del_component("dual")

    # restore cached domains
    scenario_bySymbol = instance._ScenarioTreeSymbolMap.bySymbol
    assert hasattr(instance, "PYSP_BENDERS_CACHED_DOMAINS")
    cached_domains = instance.PYSP_BENDERS_CACHED_DOMAINS
    for variable_id, domain, varbounds in cached_domains:
        vardata = scenario_bySymbol[variable_id]
        vardata.domain = domain
        vardata.setlb(varbounds[0])
        vardata.setub(varbounds[1])
    del instance.PYSP_BENDERS_CACHED_DOMAINS

    # remove fixing components
    nodal_index_set_name = "PYSP_BENDERS_FIX_XHAT_INDEX"
    assert hasattr(instance, nodal_index_set_name)
    instance.del_component(nodal_index_set_name)

    fix_param_name = "PYSP_BENDERS_FIX_XHAT_VALUE"
    assert hasattr(instance, fix_param_name)
    instance.del_component(fix_param_name)

    fix_constraint_name = "PYSP_BENDERS_FIX_XHAT_CONSTRAINT"
    assert hasattr(instance, fix_constraint_name)
    instance.del_component(fix_constraint_name)

    # The objective has changed so flag this if
    # necessary. Might as well flag the constraints as well
    # for safe measure
    if manager.preprocessor is not None:
        manager.preprocessor.objective_updated[scenario.name] = True

def EXTERNAL_initialize_for_benders(manager,
                                    scenario):
    assert len(manager.scenario_tree.stages) == 2
    assert scenario in manager.scenario_tree.scenarios

    rootnode = manager.scenario_tree.findRootNode()
    leafstage = scenario._leaf_node.stage
    instance = scenario._instance

    # disaggregate the objective into stage costs
    cost_terms = 0.0
    for node in scenario.node_list:
        stage = node.stage
        stagecost_var = instance.find_component(
            stage._cost_variable[0])[stage._cost_variable[1]]
        cost_term_name = "PYSP_BENDERS_STAGE_COST_TERM_"+stage.name
        assert not hasattr(instance, cost_term_name)
        instance.add_component(cost_term_name,
                               Expression(initialize=stagecost_var))
        cost_terms += instance.find_component(cost_term_name)
    assert not hasattr(scenario, "_instance_cost_expression_old")
    scenario._instance_cost_expression_old = \
        scenario._instance_cost_expression.expr
    # we modify this component in place so that any bundle objectives
    # get updated automatically
    scenario._instance_cost_expression.set_value(cost_terms)

    # TODO: Remove first stage constraints?
    # NOTE: If any constraints are misidentified as
    #       first-stage (e.g., because they include only
    #       first-stage variables but have data that
    #       changes), then these constraints would not be
    #       accounted for in the overall problem if we
    #       deactivate them here, so it is better leave them
    #       all active by default.
    if manager.scenario_tree.contains_bundles():
        found = 0
        for scenario_bundle in manager.scenario_tree.bundles:
            if scenario.name in scenario_bundle.scenario_names:
                found += 1
                bundle_instance = \
                    manager._bundle_binding_instance_map[scenario_bundle.name]
                if not hasattr(bundle_instance,"dual"):
                    scenario._remove_dual_at_benders_cleanup = True
                    bundle_instance.dual = Suffix(direction=Suffix.IMPORT)
                else:
                    if isinstance(bundle_instance.dual, Suffix):
                        if not bundle_instance.dual.import_enabled():
                            print("Modifying existing dual component to import "
                                  "suffix data from solver.")
                            bundle_instance.dual.set_direction(Suffix.IMPORT_EXPORT)
                    else:
                        raise TypeError(
                            "Object with name 'dual' was found on model that "
                            "is not of type 'Suffix'. The object must be renamed "
                            "in order to use the benders algorithm.")
        assert found == 1
    else:
        if not hasattr(instance,"dual"):
            scenario._remove_dual_at_benders_cleanup = True
            instance.dual = Suffix(direction=Suffix.IMPORT)
        else:
            if isinstance(instance.dual, Suffix):
                if not instance.dual.import_enabled():
                    print("Modifying existing dual component to import "
                          "suffix data from solver.")
                    instance.dual.set_direction(Suffix.IMPORT_EXPORT)
            else:
                raise TypeError(
                    "Object with name 'dual' was found on model that "
                    "is not of type 'Suffix'. The object must be renamed "
                    "in order to use the benders algorithm.")

    # Relax all first-stage variables to be continuous, cache
    # their original bounds an domains on the instance so we
    # can restore them
    scenario_bySymbol = instance._ScenarioTreeSymbolMap.bySymbol
    assert not hasattr(instance, "PYSP_BENDERS_CACHED_DOMAINS")
    cached_domains = instance.PYSP_BENDERS_CACHED_DOMAINS = []
    # GH: Question: Is it possible that there are "derived"
    #               variables that are only "derived"
    #               when additional integrality conditions
    #               are placed on them? If so, they would
    #               need to be classified as "standard" in
    #               order to run benders. That is, unless we
    #               include all node variables in the
    #               benders cuts and fixing constraints by
    #               default. For now, we do not.
    for variable_id in rootnode._variable_ids:
        vardata = scenario_bySymbol[variable_id]
        # derived variables might be Expression objects
        if not vardata.is_expression():
            tight_bounds = vardata.bounds
            domain = vardata.domain
            vardata.domain = Reals
            # Collect the var bounds after setting the domain
            # to Reals. We do this so we know if the Var bounds
            # are set by the user or come from the domain
            varbounds = vardata.bounds
            cached_domains.append((variable_id, domain, varbounds))
            vardata.setlb(tight_bounds[0])
            vardata.setub(tight_bounds[1])

    # create index sets for fixing components
    nodal_index_set_name = "PYSP_BENDERS_FIX_XHAT_INDEX"
    assert not hasattr(instance, nodal_index_set_name)
    nodal_index_set = Set(name=nodal_index_set_name,
                          ordered=True,
                          initialize=sorted(rootnode._standard_variable_ids))
    instance.add_component(nodal_index_set_name, nodal_index_set)

    fix_param_name = "PYSP_BENDERS_FIX_XHAT_VALUE"
    assert not hasattr(instance, fix_param_name)
    instance.add_component(fix_param_name,
                           Param(nodal_index_set, mutable=True, initialize=0.0))
    fix_param = instance.find_component(fix_param_name)

    fix_constraint_name = "PYSP_BENDERS_FIX_XHAT_CONSTRAINT"
    assert not hasattr(instance, fix_constraint_name)
    def fix_rule(m,variable_id):
        # NOTE: The ordering within the expression is important here; otherwise
        #       duals will be returned with the opposite sign, which affects how
        #       the cut expressions need to be generated
        return  scenario_bySymbol[variable_id] - fix_param[variable_id] == 0.0
    instance.add_component(fix_constraint_name,
                           Constraint(nodal_index_set, rule=fix_rule))
    instance.find_component(fix_constraint_name).deactivate()

    # These will flag the necessary preprocessor info
    EXTERNAL_deactivate_firststage_costs(manager, scenario)
    EXTERNAL_activate_fix_constraints(manager, scenario)

def EXTERNAL_update_fix_constraints(manager,
                                    scenario,
                                    fix_values):
    assert len(manager.scenario_tree.stages) == 2
    assert scenario in manager.scenario_tree.scenarios
    instance = scenario._instance
    fix_param_name = "PYSP_BENDERS_FIX_XHAT_VALUE"
    fix_param = instance.find_component(fix_param_name)
    fix_param.store_values(fix_values)

    fix_constraint = scenario._instance.find_component(
        "PYSP_BENDERS_FIX_XHAT_CONSTRAINT")
    if manager.preprocessor is not None:
        preprocess_constraints_list = \
            manager.preprocessor.constraints_updated_list[scenario.name]
        for constraint_data in fix_constraint.values():
            preprocess_constraints_list.append(constraint_data)

def EXTERNAL_collect_cut_data(manager,
                              scenario):
    assert len(manager.scenario_tree.stages) == 2
    assert scenario in manager.scenario_tree.scenarios

    dual_suffix = None
    sum_probability_bundle = None
    if manager.scenario_tree.contains_bundles():
        found = 0
        for scenario_bundle in manager.scenario_tree.bundles:
            if scenario.name in scenario_bundle.scenario_names:
                found += 1
                dual_suffix = \
                    manager._bundle_binding_instance_map[scenario_bundle.name].dual
                sum_probability_bundle = scenario_bundle._probability
        assert found == 1

    else:
        dual_suffix  = scenario._instance.dual
        sum_probability_bundle = scenario._probability
    rootnode = manager.scenario_tree.findRootNode()
    scenario_results = {}
    scenario_results['SSC'] = scenario._stage_costs[scenario.leaf_node.stage.name]
    duals = scenario_results['duals'] = {}
    benders_fix_constraint = scenario._instance.find_component(
        "PYSP_BENDERS_FIX_XHAT_CONSTRAINT")
    if len(dual_suffix) == 0:
        raise RuntimeError("No duals were returned with the solution for "
                           "scenario %s. This might indicate a solve failure or "
                           "that there are discrete variables in the second-stage "
                           "problem." % (scenario.name))
    for variable_id in rootnode._standard_variable_ids:
        duals[variable_id] = dual_suffix[benders_fix_constraint[variable_id]] \
                             * sum_probability_bundle \
                             / scenario._probability
    return scenario_results

class BendersOptimalityCut(object):
    __slots__ = ("xhat", "ssc", "duals")
    def __init__(self, xhat, ssc, duals):
        self.xhat = xhat
        self.ssc = ssc
        self.duals = duals

class BendersAlgorithm(PySPConfiguredObject):

    _declared_options = \
        PySPConfigBlock("Options declared for the "
                        "BendersAlgorithm class")

    safe_declare_common_option(_declared_options,
                                "verbose",
                                ap_group=_benders_group_label)
    safe_declare_unique_option(
        _declared_options,
        "max_iterations",
        PySPConfigValue(
            100,
            domain=_domain_positive_integer,
            description=(
                "The maximum number of iterations. Default is 100."
            ),
            doc=None,
            visibility=0),
        ap_group=_benders_group_label)
    safe_declare_unique_option(
        _declared_options,
        "percent_gap",
        PySPConfigValue(
            0.0001,
            domain=_domain_percent,
            description=(
                "Percent optimality gap required for convergence. "
                "Default is 0.0001%%."
            ),
            doc=None,
            visibility=0),
        ap_group=_benders_group_label)
    safe_declare_unique_option(
        _declared_options,
        "multicut_level",
        PySPConfigValue(
            1,
            domain=int,
            description=(
                "The number of cut groups added to the "
                "master benders problem each iteration. "
                "Default is 1. A number less than 1 indicates "
                "that the maximum value should be used, which "
                "is one cut group for each scenario not included "
                "in the master problem."
            ),
            doc=None,
            visibility=0),
        ap_group=_benders_group_label)
    safe_declare_unique_option(
        _declared_options,
        "optimality_gap_epsilon",
        PySPConfigValue(
            1e-10,
            domain=_domain_nonnegative,
            description=(
                "The epsilon value used in the denominator of "
                "the optimality gap calculation. Default is 1e-10."
            ),
            doc=None,
            visibility=0),
        ap_group=_benders_group_label)
    safe_declare_unique_option(
        _declared_options,
        "master_include_scenarios",
        PySPConfigValue(
            (),
            domain=_domain_tuple_of_str,
            description=(
                "A list of names of scenarios that should be included "
                "in the master problem. This option can be used multiple "
                "times from the command line to specify more than one "
                "scenario name."
            ),
            doc=None,
            visibility=0),
        ap_kwds={'action': 'append'},
        ap_group=_benders_group_label)
    safe_declare_unique_option(
        _declared_options,
        "master_disable_warmstart",
        PySPConfigValue(
            False,
            domain=bool,
            description=(
                "Disable warm-start of the benders master "
                "problem solves. Default is False."
            ),
            doc=None,
            visibility=0),
        ap_group=_benders_group_label)
    safe_declare_unique_option(
        _declared_options,
        "master_solver",
        PySPConfigValue(
            "cplex",
            domain=_domain_must_be_str,
            description=(
                "Specify the solver with which to solve "
                "the master benders problem. Default is cplex."
            ),
            doc=None,
            visibility=0),
        ap_group=_benders_group_label)
    safe_declare_unique_option(
        _declared_options,
        "master_solver_io",
        PySPConfigValue(
            None,
            domain=_domain_must_be_str,
            description=(
                "The type of IO used to execute the master "
                "solver.  Different solvers support different "
                "types of IO, but the following are common "
                "options: lp - generate LP files, nl - generate "
                "NL files, mps - generate MPS files, python - "
                "direct Python interface, os - generate OSiL XML files."
            ),
            doc=None,
            visibility=0),
        ap_group=_benders_group_label)
    safe_declare_unique_option(
        _declared_options,
        "master_mipgap",
        PySPConfigValue(
            None,
            domain=_domain_unit_interval,
            description=(
                "Specifies the mipgap for the master benders solves."
            ),
            doc=None,
            visibility=0),
        ap_group=_benders_group_label)
    safe_declare_unique_option(
        _declared_options,
        "master_solver_options",
        PySPConfigValue(
            (),
            domain=_domain_tuple_of_str_or_dict,
            description=(
                "Persistent solver options used when solving the master "
                "benders problem. This option can be used multiple times from "
                "the command line to specify more than one solver option."
            ),
            doc=None,
            visibility=0),
        ap_kwds={'action': 'append'},
        ap_group=_benders_group_label)
    safe_declare_unique_option(
        _declared_options,
        "master_output_solver_log",
        PySPConfigValue(
            False,
            domain=bool,
            description=(
                "Output solver log during solves of the master problem."
            ),
            doc=None,
            visibility=0),
        ap_group=_benders_group_label)
    safe_declare_unique_option(
        _declared_options,
        "master_keep_solver_files",
        PySPConfigValue(
            False,
            domain=bool,
            description=(
                "Retain temporary input and output files for master "
                "benders solves."
            ),
            doc=None,
            visibility=0),
        ap_group=_benders_group_label)
    safe_declare_unique_option(
        _declared_options,
        "master_symbolic_solver_labels",
        PySPConfigValue(
            False,
            domain=bool,
            description=(
                "When interfacing with the solver, use "
                "symbol names derived from the model. For "
                "example, \"my_special_variable[1_2_3]\" "
                "instead of \"v1\". Useful for "
                "debugging. When using the ASL interface "
                "(--solver-io=nl), generates corresponding "
                ".row (constraints) and .col (variables) "
                "files. The ordering in these files provides "
                "a mapping from ASL index to symbolic model "
                "names."
            ),
            doc=None,
            visibility=0),
        ap_group=_benders_group_label)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.cleanup_subproblems()
        if self._master_solver is not None:
            self._master_solver.deactivate()

    def __init__(self, manager, *args, **kwds):
        super(BendersAlgorithm, self).__init__(*args, **kwds)

        if not isinstance(manager, ScenarioTreeManagerSolver):
            raise TypeError("BendersAlgorithm requires an instance of the "
                            "ScenarioTreeManagerSolver interface as the "
                            "second argument")
        if not manager.initialized:
            raise ValueError("BendersAlgorithm requires a scenario tree "
                             "manager that has been fully initialized")
        if len(manager.scenario_tree.stages) != 2:
            raise ValueError("BendersAlgorithm requires a two-stage scenario tree")

        self._manager = manager
        print("Initializing subproblems for benders")
        self.initialize_subproblems()

        self._master_solver = None
        # setup the master solver
        self._master_solver = SolverFactory(
            self.get_option("master_solver"),
            solver_io=self.get_option("master_solver_io"))
        if isinstance(self._master_solver, PersistentSolver):
            raise TypeError("BendersAlgorithm does not support "
                            "PersistenSolver types")
        if len(self.get_option("master_solver_options")):
            if type(self.get_option("master_solver_options")) is tuple:
                self._master_solver.set_options(
                    "".join(self.get_option("master_solver_options")))
            else:
                self._master_solver.set_options(
                    self.get_option("master_solver_options"))
        if self.get_option("master_mipgap") is not None:
            self._master_solver.options.mipgap = \
                self.get_option("master_mipgap")

        # The following attributes will be modified by the
        # solve() method. For users that are scripting, these
        # can be accessed after the solve() method returns.
        # They will be reset each time solve() is called.
        ############################################
        # history of master bounds
        self.master_bound_history = OrderedDict()
        # history of objectives
        self.objective_history = OrderedDict()
        # best objective
        self.incumbent_objective = None
        # incumbent first-stage solution with best objective
        self.incumbent_xhat = None
        # |best_bound - best_objective| / (eps + |best_objective|)
        self.optimality_gap = None
        # no. of iterations completed
        self.iterations = None
        ############################################

        # The following attributes will be modified by the
        # build_master_problem() method. They will be
        # reset each time it is called. Additionally,
        # cut_pool will be appended with a new cut at
        # each iteration within the solve() method.
        self.master = None
        self.cut_pool = []
        self._num_first_stage_constraints = None

    def deactivate_firststage_costs(self):
        self._manager.invoke_function(
            "EXTERNAL_deactivate_firststage_costs",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            oneway=True)

    def activate_firststage_costs(self):
        self._manager.invoke_function(
            "EXTERNAL_activate_firststage_costs",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            oneway=True)

    def activate_fix_constraints(self):
        self._manager.invoke_function(
            "EXTERNAL_activate_fix_constraints",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            oneway=True)

    def deactivate_fix_constraints(self):
        self._manager.invoke_function(
            "EXTERNAL_deactivate_fix_constraints",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            oneway=True)

    def update_fix_constraints(self, fix_values):
        self._manager.invoke_function(
            "EXTERNAL_update_fix_constraints",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            function_args=(fix_values,),
            oneway=True)

    def collect_cut_data(self, async=False):
        return self._manager.invoke_function(
            "EXTERNAL_collect_cut_data",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            async=async)

    def initialize_subproblems(self):
        self._manager.invoke_function(
            "EXTERNAL_initialize_for_benders",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            oneway=True)

    def cleanup_subproblems(self):
        self._manager.invoke_function(
            "EXTERNAL_cleanup_from_benders",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            oneway=True)

    def generate_cut(self,
                     xhat,
                     update_stages=(),
                     return_solve_results=False):
        """
        Generate a cut for the first-stage solution xhat by
        solving the subproblems. By default, only the stage
        costs and objective values are updated on the local
        scenario tree. Setting update_stages to a list of
        state names or None (indicating all stages) can be
        used to control how much solution information is
        loaded for the variables on the scenario tree.
        """
        self.update_fix_constraints(xhat)
        solve_results = \
            self._manager.solve_subproblems(update_stages=update_stages)

        cut_data = self.collect_cut_data()
        benders_cut = BendersOptimalityCut(
            xhat,
            dict((name, cut_data[name]['SSC']) for name in cut_data),
            dict((name, cut_data[name]['duals']) for name in cut_data))

        if return_solve_results:
            return benders_cut, solve_results
        else:
            return benders_cut

    #
    # Any of the methods above this point can be called without
    # constructing a master problem.
    #

    def build_master_problem(self, **kwds):
        """
        Build a master problem to add cuts to. This method
        must called prior to calling methods like solve().
        When the optional keyword include_scenarios is used,
        it overrides the list of names of scenarios (if any)
        to include on the master that were specified on the options
        object used to initialize this class.
        """

        # check whether to override the
        # "master_include_scenarios" option
        if "include_scenarios" not in kwds:
            include_scenarios = self.get_option("master_include_scenarios")
        else:
            include_scenarios = kwds.pop("include_scenarios")
        assert len(kwds) == 0

        scenario_tree = self._manager.scenario_tree
        rootnode = scenario_tree.findRootNode()
        firststage = rootnode.stage
        objective_sense = self._manager.objective_sense

        # construct master problem
        if (include_scenarios is None) or \
           (len(include_scenarios) == 0):
            master_scenario_tree = scenario_tree.make_compressed(
                [scenario_tree.scenarios[0].name],
                normalize=False)
        else:
            print("Number of scenarios included in Benders master problem: %s"
                  % (len(include_scenarios)))
            master_scenario_tree = scenario_tree.make_compressed(
                include_scenarios,
                normalize=False)

        master_rootnode = master_scenario_tree.findRootNode()
        master_firststage = master_scenario_tree.stages[0]
        assert master_firststage is master_rootnode.stage
        master_secondstage = master_scenario_tree.stages[1]
        assert len(master_scenario_tree.stages) == 2

        master_scenario_instances = \
            scenario_tree._scenario_instance_factory.\
            construct_instances_for_scenario_tree(master_scenario_tree)
        # with the scenario instances now available, link the
        # referenced objects directly into the scenario tree.
        master_scenario_tree.linkInInstances(master_scenario_instances,
                                             create_variable_ids=True)

        master = create_ef_instance(master_scenario_tree)
        if (include_scenarios is None) or \
           (len(include_scenarios) == 0):
            master._scenarios_included = set()
            remove_ssc = True
        else:
            master._scenarios_included = \
                set(include_scenarios)
            remove_ssc = False

        #
        # Deactivate second-stage constraints
        #

        # first count the binding constraints on the top-level ef model
        num_first_stage_constraints = len(list(itertools.chain(
            master.component_data_objects(SOSConstraint,
                                          active=True,
                                          descend_into=False),
            master.component_data_objects(Constraint,
                                          active=True,
                                          descend_into=False))))
        # now count the firs-stage constraints on the scenario
        # instances included in the master ef
        for scenario in master_scenario_tree.scenarios:
            instance = scenario._instance
            for constraint_data in itertools.chain(
                    instance.component_data_objects(SOSConstraint,
                                                    active=True,
                                                    descend_into=True),
                    instance.component_data_objects(Constraint,
                                                    active=True,
                                                    descend_into=True)):
                # Note that it is possible that we are misidentifying
                # some constraints as belonging to the first stage. This
                # would be the case when no second-stage variables appear
                # in the expression but one or more rhs or first-stage variable
                # coefficients changes with scenarios.
                node = scenario.constraintNode(constraint_data,
                                               instance=instance)
                # Not sure if we want to allow variables to not be declared on
                # some stage in the scenario tree.
                                               #assume_last_stage_if_missing=True)
                if node.stage is not master_firststage:
                    assert node.stage is master_secondstage
                    if remove_ssc:
                        constraint_data.deactivate()
                else:
                    num_first_stage_constraints += 1

        self._num_first_stage_constraints = num_first_stage_constraints
        # deactivate original objective
        find_active_objective(master, safety_checks=True).deactivate()

        # add cut variable(s)
        if self.get_option("multicut_level") < 1:
            print("Using maximum number of cut groups")
            cut_bundles = [[] for scenario in scenario_tree.scenarios]
        else:
            cut_bundles = [[] for i in xrange(self.get_option("multicut_level"))]

        # TODO: Allow users some control over these cut_bundles
        cnt = 0
        len_cut_bundles = len(cut_bundles)
        assert len_cut_bundles > 0
        for scenario in scenario_tree.scenarios:
            if scenario.name not in master._scenarios_included:
                cut_bundles[cnt % len_cut_bundles].append(scenario.name)
                cnt += 1
        nonempty_cut_bundles = []
        for bundle in cut_bundles:
            if len(bundle) > 0:
                nonempty_cut_bundles.append(bundle)
        if len(nonempty_cut_bundles) != len(cut_bundles):
            if self.get_option("multicut_level") >= 1:
                print("The number of cut groups indicated by the multicut_level "
                      "option was too large. Reducing from %s to %s."
                      % (len(cut_bundles), len(nonempty_cut_bundles)))

        alpha_varname = "PYSP_BENDERS_ALPHA_SSC"
        assert not hasattr(master, alpha_varname)
        master.add_component(alpha_varname,Var())
        master_alpha = master.find_component(alpha_varname)

        alpha_bundles_index_name = "PYSP_BENDERS_BUNDLE_ALPHA_SSC_INDEX"
        assert not hasattr(master, alpha_bundles_index_name)
        master.add_component(alpha_bundles_index_name,
                             RangeSet(0,len(nonempty_cut_bundles)-1))
        bundle_alpha_index = master.find_component(alpha_bundles_index_name)

        alpha_bundles_name = "PYSP_BENDERS_BUNDLE_ALPHA_SSC"
        assert not hasattr(master, alpha_bundles_name)
        master.add_component(alpha_bundles_name,
                             Var(bundle_alpha_index))
        bundle_alpha = master.find_component(alpha_bundles_name)

        cut_bundles_list_name = "PYSP_BENDERS_CUT_BUNDLES_SSC"
        assert not hasattr(master, cut_bundles_list_name)
        setattr(master, cut_bundles_list_name, nonempty_cut_bundles)
        alpha_cut_constraint_name = "PYSP_BUNDLE_AVERAGE_ALPHA_CUT_SSC"
        assert not hasattr(master, alpha_cut_constraint_name)
        if objective_sense == minimize:
            master.add_component(
                alpha_cut_constraint_name,
                Constraint(expr=master_alpha >= sum(bundle_alpha[i]
                                                    for i in bundle_alpha_index)))
        else:
            master.add_component(
                alpha_cut_constraint_name,
                Constraint(expr=master_alpha <= sum(bundle_alpha[i]
                                                    for i in bundle_alpha_index)))

        # add new objective
        if (include_scenarios is None) or \
           (len(include_scenarios) == 0):
            assert len(master_scenario_tree.scenarios) == 1
            master_cost_expr = master.find_component(
                master_scenario_tree.scenarios[0].name).find_component(
                    master_firststage._cost_variable[0])\
                    [master_firststage._cost_variable[1]]
        else:
            # NOTE: We include the first-stage cost expression for
            #       each of the scenarios included in the master with
            #       normalized probabilities. The second stage costs
            #       are included without normalizing probabilities so
            #       that they can simply be excluded from any cut expressions
            #       without having to re-normalize anything in the cuts
            master_cost_expr = 0.0
            normalization = sum(scenario._probability
                                for scenario in master_scenario_tree.scenarios)
            for scenario in master_scenario_tree.scenarios:
                firststage_cost_expr = scenario._instance.find_component(
                    master_firststage._cost_variable[0])\
                    [master_firststage._cost_variable[1]]
                secondstage_cost_expr = scenario._instance.find_component(
                    master_secondstage._cost_variable[0])\
                    [master_secondstage._cost_variable[1]]
                master_cost_expr += scenario._probability * \
                                    (firststage_cost_expr / normalization + \
                                     secondstage_cost_expr)
        benders_objective_name = "PYSP_BENDERS_OBJECTIVE"
        assert not hasattr(master, benders_objective_name)
        master.add_component(
            benders_objective_name,
            Objective(expr=master_cost_expr + master_alpha,
                      sense=objective_sense))
        master_objective = master.find_component(
            benders_objective_name)
        cutlist_constraint_name = "PYSP_BENDERS_CUTS_SSC"
        assert not hasattr(master, cutlist_constraint_name)
        # I am using the XConstraintList prototype because
        # it is zero-based, meaning the index within self.cut_pool
        # (which stores the benders cuts objects) will correspond
        # directly with the index within this constraint.
        master.add_component(cutlist_constraint_name,
                             XConstraintList())

        self.master = master
        self.cut_pool = []

    def add_cut(self, benders_cut, ignore_cut_bundles=False):
        """
        Add the cut defined by the benders_cut object to the
        master problem. The optional keyword ignore_cut_bundles
        can be used generate the cut using the single master
        alpha cut variable rather than over the possibly many
        bundle cut groups.
        """

        if self.master is None:
            raise RuntimeError("The master problem has not been constructed."
                               "Call the build_master_problem() method to "
                               "construct it.")

        # for now, until someone figures out feasibility cuts
        assert benders_cut.__class__ is BendersOptimalityCut

        self.cut_pool.append(benders_cut)

        scenario_tree = self._manager.scenario_tree
        objective_sense = self._manager.objective_sense
        master = self.master
        rootnode = scenario_tree.findRootNode()
        master_variable = master.find_component(
            "MASTER_BLEND_VAR_"+str(rootnode.name))
        benders_cuts = master.find_component(
            "PYSP_BENDERS_CUTS_SSC")
        master_alpha = master.find_component(
            "PYSP_BENDERS_ALPHA_SSC")
        bundle_alpha = master.find_component(
            "PYSP_BENDERS_BUNDLE_ALPHA_SSC")

        xhat = benders_cut.xhat
        cut_expression = 0.0
        if not ignore_cut_bundles:
            for i, cut_scenarios in enumerate(
                    getattr(master, "PYSP_BENDERS_CUT_BUNDLES_SSC")):

                for scenario_name in cut_scenarios:
                    assert scenario_name not in master._scenarios_included
                    scenario_duals = benders_cut.duals[scenario_name]
                    scenario_ssc = benders_cut.ssc[scenario_name]
                    scenario = scenario_tree.get_scenario(scenario_name)
                    cut_expression += \
                        scenario._probability * \
                        (scenario_ssc + \
                         sum(scenario_duals[variable_id] * \
                             (master_variable[variable_id] - xhat[variable_id]) \
                             for variable_id in xhat))

                cut_expression -= bundle_alpha[i]

        else:
            for scenario in scenario_tree.scenarios:
                if scenario.name in master._scenarios_included:
                    continue
                scenario_name = scenario.name
                scenario_duals = benders_cut.duals[scenario_name]
                scenario_ssc = benders_cut.ssc[scenario_name]
                scenario = scenario_tree.get_scenario(scenario_name)
                cut_expression += \
                    scenario._probability * \
                    (scenario_ssc + \
                     sum(scenario_duals[variable_id] * \
                         (master_variable[variable_id] - xhat[variable_id]) \
                         for variable_id in xhat))

            cut_expression -= master_alpha

        if objective_sense == minimize:
            benders_cuts.append(
                _GeneralConstraintData((None,cut_expression,0.0)))
        else:
            benders_cuts.append(
                _GeneralConstraintData((0.0,cut_expression,None)))

    def extract_master_xhat(self):

        if self.master is None:
            raise RuntimeError("The master problem has not been constructed."
                               "Call the build_master_problem() method to "
                               "construct it.")
        master = self.master
        rootnode = self._manager.scenario_tree.findRootNode()
        master_variable = master.find_component(
            "MASTER_BLEND_VAR_"+str(rootnode.name))
        return dict((variable_id, master_variable[variable_id].value)
                    for variable_id in rootnode._standard_variable_ids
                    if not master_variable[variable_id].stale)

    def solve_master(self):

        if self.master is None:
            raise RuntimeError("The master problem has not been constructed."
                               "Call the build_master_problem() method to "
                               "construct it.")

        common_kwds = {
            'load_solutions':False,
            'tee':self.get_option("master_output_solver_log"),
            'keepfiles':self.get_option("master_keep_solver_files"),
            'symbolic_solver_labels':self.get_option("master_symbolic_solver_labels")}

        if (not self.get_option("master_disable_warmstart")) and \
           (self._master_solver.warm_start_capable()):
            results = self._master_solver.solve(self.master,
                                                warmstart=True,
                                                **common_kwds)
        else:
            results = self._master_solver.solve(self.master,
                                                **common_kwds)

        return results

    def solve(self, **kwds):
        """
        Run the algorithm. If one or both of the keywords max_iterations and
        percent_gap are used, they will override the values on the options
        object used to initialize this class.

        Returns the objective value for the incumbent solution.
        """

        if "max_iterations" not in kwds:
            max_iterations = self.get_option("max_iterations")
        else:
            max_iterations = _domain_positive_integer(
                kwds.pop("max_iterations"))

        if "percent_gap" not in kwds:
            percent_gap = self.get_option("percent_gap")
        else:
            percent_gap = _domain_percent(
                kwds.pop("percent_gap"))

        assert len(kwds) == 0

        start_time = time.time()

        master = self.master
        master_solver = self._master_solver

        if self.master is None:
            raise RuntimeError("The master problem has not been constructed."
                               "Call the build_master_problem() method to "
                               "construct it.")

        objective_sense = self._manager.objective_sense
        scenario_tree = self._manager.scenario_tree

        def print_dictionary(dictionary):
            #Find longest key
            longest_message = max(len(str(x[0])) for x in dictionary)

            #Find longest dictionary value
            longest_value = max(len(str(x[1])) for x in dictionary)
            for key, value in dictionary:
                print(('{0:<' + str(longest_message) + \
                       '}' '{1:^3}' '{2:<' + str(longest_value) + \
                       '}').format(key,":",value))

        print("-"*20)
        print("Problem Statistics")
        print("-"*20)

        rootnode = scenario_tree.findRootNode()
        num_discrete_firststage = len([variable_id for variable_id
                                       in rootnode._variable_ids \
                                       if rootnode.is_variable_discrete(variable_id)])
        problem_statistics = []
        problem_statistics.append(("Number of first-stage variables"   ,
                                   str(len(rootnode._variable_ids)) + \
                                   " (" + str(num_discrete_firststage)+" discrete)"))
        problem_statistics.append(("Number of first-stage constraints" ,
                                   self._num_first_stage_constraints))
        problem_statistics.append(("Number of scenarios"               ,
                                   len(rootnode.scenarios)))
        problem_statistics.append(("Number of bundles"                 ,
                                   len(scenario_tree.bundles)))
        problem_statistics.append(("Initial number of cuts in pool"    ,
                                   len(self.cut_pool)))
        problem_statistics.append(("Maximum number of iterations"      ,
                                   max_iterations))
        problem_statistics.append(("Relative convergence gap threshold",
                                   percent_gap/100.0))
        print_dictionary(problem_statistics)

        print("")
        width_log_table = 100
        print("-"*width_log_table)
        print("%6s %16s %16s %11s  %30s"
              % ("Iter","Master Bound","Best Incumbent","Gap","Solution Times [s]"))
        print("%6s %16s %16s %11s  %10s %10s %10s %10s"
              % ("","","","","Master","Sub Min","Sub Max","Cumm"))
        print("-"*width_log_table)

        master_alpha = master.find_component(
            "PYSP_BENDERS_ALPHA_SSC")
        master_bundles_alpha = master.find_component(
            "PYSP_BENDERS_BUNDLE_ALPHA_SSC")
        master_objective = master.find_component(
            "PYSP_BENDERS_OBJECTIVE")

        self.master_bound_history = OrderedDict()
        self.objective_history = OrderedDict()
        self.incumbent_objective = \
            float('inf') if (objective_sense is minimize) else float('-inf')
        self.incumbent_xhat = None
        self.optimality_gap = float('inf')
        self.iterations = 0
        output_no_gap_warning = True
        for i in xrange(1, max_iterations + 1):

            if (i == 1) and (len(self.cut_pool) == 0):
                # Avoid an unbounded problem. We may still recover
                # an xhat at which to generate a cut, but we can not
                # use the master objective as a lower bound
                master_alpha.fix(0.0)
                master_bundles_alpha.fix(0.0)

            start_time_master = time.time()
            results_master = self.solve_master()
            if len(results_master.solution) == 0:
                raise RuntimeError("Solve failed for master; no solutions generated")
            if results_master.solver.termination_condition != \
               TerminationCondition.optimal:
                logger.warning(
                    "Master solve did not generate an optimal solution:\n"
                    "Solver Status: %s\n"
                    "Solver Termination Condition: %s\n"
                    "Solution Status: %s\n"
                    % (str(results_master.solver.status),
                       str(results_master.solver.termination_condition),
                       str(results_master.solution(0).status)))
            master.solutions.load_from(results_master)
            stop_time_master = time.time()

            if master_alpha.fixed:
                assert i == 1
                assert master_alpha.value == 0.0
                current_master_bound = \
                    float('-inf') if (objective_sense is minimize) else float('inf')
                master_alpha.free()
                master_bundles_alpha.free()
            else:
                current_master_bound = value(master_objective)
                # account for any optimality gap
                solution0 = results_master.solution(0)
                if hasattr(solution0, "gap") and \
                   (solution0.gap is not None) and \
                   (solution0.gap is not undefined):
                    if objective_sense == minimize:
                        current_master_bound -= solution0.gap
                    else:
                        current_master_bound += solution0.gap
                else:
                    if output_no_gap_warning:
                        output_no_gap_warning = False
                        print("WARNING: Solver interface does not "
                              "report an optimality gap. Master bound "
                              "may be invalid.")

            self.master_bound_history[i] = current_master_bound

            new_xhat = self.extract_master_xhat()
            new_cut_info, solve_results = \
                self.generate_cut(new_xhat,
                                  return_solve_results=True)

            # compute the true objective at xhat by
            # replacing the current value of the master cut
            # variable with the true second stage costs of
            # any scenarios involved in the cuts
            self.objective_history[i] = \
                value(master_objective) - value(master_alpha) + \
                sum(scenario._probability * new_cut_info.ssc[scenario.name] \
                    for scenario in scenario_tree.scenarios
                    if scenario.name not in self.master._scenarios_included)

            incumbent_objective_prev = self.incumbent_objective
            best_master_bound = max(self.master_bound_history.values()) if \
                                (objective_sense == minimize) else \
                                min(self.master_bound_history.values())
            self.incumbent_objective = min(self.objective_history.values()) if \
                                       (objective_sense == minimize) else \
                                       max(self.objective_history.values())
            if objective_sense == minimize:
                if self.incumbent_objective < incumbent_objective_prev:
                    self.incumbent_xhat = new_xhat
            else:
                if self.incumbent_objective > incumbent_objective_prev:
                    self.incumbent_xhat = new_xhat

            self.optimality_gap = \
                abs(best_master_bound - self.incumbent_objective) / \
                (self.get_option("optimality_gap_epsilon") + \
                 abs(self.incumbent_objective))

            min_time_sub = min(solve_results.solve_time.values())
            max_time_sub = max(solve_results.solve_time.values())
            print("%6d %16.4f %16.4f %11.3f%% %10.2f %10.2f %10.2f %10.2f"
                  % (i, current_master_bound, self.incumbent_objective,
                     self.optimality_gap*100, stop_time_master - start_time_master,
                     min_time_sub, max_time_sub, time.time()-start_time))

            # Add the cut even if we exit on this iteration so
            # it ends up in the cut pool (just in case the caller
            # wants to continue the algorithm)
            self.add_cut(new_cut_info)

            # we've completed another iteration
            self.iterations += 1

            # If the optimality gap is below the convergence
            # threshold set by the user, quit the
            # loop. Otherwise, add the new cut to the master
            if self.optimality_gap*100 <= percent_gap:
                print("-" * width_log_table)
                print("Optimality gap threshold reached.")
                break
        else:
            print("-" * width_log_table)
            print("Maximum number of iterations reached.")

        print("")
        print("Re-solving at best incumbent xhat and collecting full "
              "scenario tree solution.")
        self.update_fix_constraints(self.incumbent_xhat)
        self._manager.solve_subproblems()

        return self.incumbent_objective

def runbenders_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()
    safe_register_common_option(options,
                               "verbose")
    safe_register_common_option(options,
                               "disable_gc")
    safe_register_common_option(options,
                               "profile")
    safe_register_common_option(options,
                               "traceback")
    safe_register_common_option(options,
                               "output_scenario_tree_solution")
    ScenarioTreeManagerFactory.register_options(options)
    BendersAlgorithm.register_options(options)

    return options

#
# Construct a senario tree manager and a BendersAlgorithm
# object to solve it.
#

def runbenders(options):
    import pyomo.environ

    start_time = time.time()

    with ScenarioTreeManagerFactory(options) \
         as manager:
        manager.initialize()

        # This is hard to do without a general option for
        # relaxing integrality that works with all solver
        # plugins. I think it's better left for advanced
        # users that want to implement it within a script.
        #print("Determining trivial lower bound using perfect "
        #      "information (on LP relaxation)")
        #self._manager.solve_subproblems(
        #    ephemeral_solver_options={'relax_integrality': True})
        #rootnode.updateNodeStatistics()
        #trivial_bound = sum(scenario._probability * scenario._objective
        #                    for scenario in scenario_tree.scenarios)
        #if not master_alpha.fixed:
        #    print("Determining initial alpha bound from scenario solves")
        #    benders_cut = self.generate_cut(
        #        dict((variable_id, rootnode._averages[variable_id])
        #             for variable_id in rootnode._standard_variable_ids))
        #    self.add_cut(benders_cut)

        print("")
        print("Initializing Benders decomposition algorithm for two-stage "
              "stochastic programming problems (i.e., the L-shaped method).")
        with BendersAlgorithm(manager, options) as benders:
            benders.build_master_problem()
            benders.solve()

        print("")
        print("***********************************************"
              "***********************************************")
        print(">>>THE EXPECTED SUM OF THE STAGE COST VARIABLES="
              +str(manager.scenario_tree.findRootNode().\
                   computeExpectedNodeCost())+"<<<")
        print("***********************************************"
              "***********************************************")

        if options.output_scenario_tree_solution:
            print("Final solution (scenario tree format):")
            manager.scenario_tree.snapshotSolutionFromScenarios()
            manager.scenario_tree.pprintSolution()

    print("")
    print("Total execution time=%.2f seconds"
          % (time.time() - start_time))

    return 0

#
# the main driver routine for the evaluate_xhat script.
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
            runbenders_register_options,
            prog='runbenders',
            description=(
"""Optimize a stochastic program using Generalized Benders
(i.e., the L-shaped method)"""
            ))

    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(runbenders,
                          options,
                          error_label="runbenders: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

@pyomo_command('runbenders', 'Optimize with the Benders solver')
def Benders_main(args=None):
    return main(args=args)
