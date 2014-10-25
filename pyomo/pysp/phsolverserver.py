#  _________________________________________________________________________
#
#  Coopr: A COmmon Optimization Python Repository
#  Copyright (c) 2010 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Coopr README.txt file.
#  _________________________________________________________________________

import gc         # garbage collection control.
import os
import sys
import time
import traceback
import copy

from optparse import OptionParser
import itertools
from six import iterkeys, itervalues, iteritems, advance_iterator
from six.moves import zip

try:
    import pstats
    pstats_available=True
except ImportError:
    pstats_available=False

try:
    import cProfile as profile
except ImportError:
    import profile

from coopr.core import coopr_command
from coopr.opt.base import SolverFactory, PersistentSolver
from coopr.pysp.scenariotree import *

from pyutilib.misc import import_file, PauseGC
from pyutilib.services import TempfileManager

#import Pyro.core
#import Pyro.naming
#from Pyro.errors import PyroError, NamingError

from coopr.pysp.phinit import *
from coopr.pysp.phutils import *
from coopr.pysp.phobjective import *
from coopr.pysp.ph import _PHBase
from coopr.pyomo.base.expr import identify_variables
from coopr.pysp.phsolverserverutils import TransmitType, \
                                           InvocationType
from coopr.pysp.phextension import IPHSolverServerExtension

try:
    pyro_available=True
    from pyutilib.pyro import MultiTaskWorker
    from pyutilib.pyro import TaskWorkerServer
except:
    pyro_available=False
    class MultiTaskWorker(object): pass
    class TaskWorkerServer(object): pass

class PHPyroWorker(MultiTaskWorker):

    def __init__(self, **kwds):

        MultiTaskWorker.__init__(self,**kwds)
        self._init()

    def _init(self):

        self.clear_request_types()
        # queue type, blocking, timeout
        self.push_request_type('phpyro_worker_idle',True,5)
        self._phsolverserver_map = {}

    def del_server(self, name):
        phsolver = self._phsolverserver_map[name]
        # Avoid memory leaks
        if phsolver._solver is not None:
            phsolver._solver.deactivate()
        del self._phsolverserver_map[name]

        types_to_keep = []
        for rqtype in self.current_type_order():
            if rqtype[0] != name:
                types_to_keep.append(rqtype)
        self.clear_request_types()
        for rqtype in types_to_keep:
            self.push_request_type(*rqtype)

    def process(self, data):

        result = None
        if data.action == "acknowledge":

            assert self.num_request_types() == 1
            self.clear_request_types()
            self.push_request_type(self.WORKERNAME,True,1)
            result = self.WORKERNAME

        elif (data.name == self.WORKERNAME) and (data.action == "release"):

            self.del_server(data.object_name)
            if len(self.current_type_order()) == 1:
                # Go back to making general worker requests
                # blocking with a reasonable timeout so they
                # don't overload the dispatcher
                self.pop_request_type()
                self.push_request_type(self.WORKERNAME,True,1)
            result = True

        elif (data.name == self.WORKERNAME) and (data.action == "go_idle"):

            server_names = list(self._phsolverserver_map.keys())
            for name in server_names:
                self.del_server(name)
            self._init()
            result = True

        else:

            if (data.name == self.WORKERNAME) and (data.action == "initialize"):
                current_types = self.current_type_order()
                for rqtype in current_types:
                    if data.object_name == rqtype[0]:
                        raise RuntimeError('Cannot initialize object with name:'
                                           +str(data.object_name)+
                                           ' because work queue already exists with this name')

                assert current_types[-1][0] == self.WORKERNAME
                if len(current_types) == 1:
                    # make the general worker request non blocking
                    # as we now have higher priority work to perform
                    self.pop_request_type()
                    self.push_request_type(self.WORKERNAME,False,None)

                self.push_request_type(data.object_name,True,5)
                self._phsolverserver_map[data.object_name] = _PHSolverServer()
                data.name = data.object_name
            result = self._phsolverserver_map[data.name].process(data)

        return result

class _PHSolverServer(_PHBase):

    def __init__(self, **kwds):

        _PHBase.__init__(self)

        self._first_solve = True

        # So we have access to real scenario and bundle probabilities
        self._uncompressed_scenario_tree = None

        # Maps ScenarioTreeID's on the master node ScenarioTree to ScenarioTreeID's on this
        # PHSolverServers's ScenarioTree (by node name)
        self._master_scenario_tree_id_map = {}
        self._reverse_master_scenario_tree_id_map = {}

        # global handle to ph extension plugins
        self._ph_plugins = ExtensionPoint(IPHSolverServerExtension)

    #
    # Overloading from _PHBase to add a few extra print statements
    #

    def activate_ph_objective_weight_terms(self):

        if self._verbose:
            print("Received request to activate PH objective weight terms for scenario(s)="+str(list(iterkeys(self._instances))))

        if self._initialized is False:
            raise RuntimeError("PH solver server has not been initialized!")

        _PHBase.activate_ph_objective_weight_terms(self)

        if self._verbose:
            print("Activating PH objective weight terms")

    #
    # Overloading from _PHBase to add a few extra print statements
    #

    def deactivate_ph_objective_weight_terms(self):

        if self._verbose:
            print("Received request to deactivate PH objective weight terms for scenario(s)="+str(list(iterkeys(self._instances))))

        if self._initialized is False:
            raise RuntimeError("PH solver server has not been initialized!")

        _PHBase.deactivate_ph_objective_weight_terms(self)

        if self._verbose:
            print("Deactivating PH objective weight terms")

    #
    # Overloading from _PHBase to add a few extra print statements
    #

    def activate_ph_objective_proximal_terms(self):

        if self._verbose:
            print("Received request to activate PH objective proximal terms for scenario(s)="+str(list(iterkeys(self._instances))))

        if self._initialized is False:
            raise RuntimeError("PH solver server has not been initialized!")

        _PHBase.activate_ph_objective_proximal_terms(self)

        if self._verbose:
            print("Activating PH objective proximal terms")

    #
    # Overloading from _PHBase to add a few extra print statements
    #

    def deactivate_ph_objective_proximal_terms(self):

        if self._verbose:
            print("Received request to deactivate PH objective proximal terms for scenario(s)="+str(list(iterkeys(self._instances))))

        if self._initialized is False:
            raise RuntimeError("PH solver server has not been initialized!")

        _PHBase.deactivate_ph_objective_proximal_terms(self)

        if self._verbose:
            print("Deactivating PH objective proximal terms")

    def collect_scenario_tree_data(self, tree_object_names):

        data = {}
        node_data = data['nodes'] = {}
        for node_name in tree_object_names['nodes']:
            tree_node = self._scenario_tree.get_node(node_name)
            this_node_data = node_data[node_name] = {}
            for attr_name, attr_value in iteritems(tree_node.__dict__):
                if attr_name not in ['_name',
                                     '_stage',
                                     '_parent',
                                     '_children',
                                     '_conditional_probability',
                                     '_scenarios',
                                     '_probability',
                                     '_variable_datas',
                                     '_cost_variable_datas']:
                    this_node_data[attr_name] = attr_value

        scenario_data = data['scenarios'] = {}
        for scenario_name in tree_object_names['scenarios']:
            scenario = self._scenario_tree.get_scenario(scenario_name)
            this_scenario_data = scenario_data[scenario_name] = {}
            for attr_name, attr_value in iteritems(scenario.__dict__):
                if attr_name not in ['_name',
                                     '_leaf_node',
                                     '_node_list',
                                     '_probability',
                                     '_instance',
                                     '_instance_cost_expression',
                                     '_instance_objective']:
                    this_scenario_data[attr_name] = attr_value

        return data


    def initialize(self,
                   model_location,
                   data_location,
                   object_name,
                   objective_sense,
                   solver_type,
                   solver_io,
                   scenario_bundle_specification,
                   create_random_bundles,
                   scenario_tree_random_seed,
                   default_rho,
                   linearize_nonbinary_penalty_terms,
                   retain_quadratic_binary_terms,
                   breakpoint_strategy,
                   integer_tolerance,
                   verbose):

        if verbose:
            print("Received request to initialize PH solver server")
            print("")
            print("Model source: "+model_location)
            print("Scenario Tree source: "+data_location)
            print("Solver type: "+solver_type)
            print("Scenario or bundle name: "+object_name)
            if scenario_bundle_specification != None:
                print("Scenario tree bundle specification: "
                      +scenario_bundle_specification)
            if create_random_bundles != None:
                print("Create random bundles: "+str(create_random_bundles))
            if scenario_tree_random_seed != None:
                print("Scenario tree random seed: "+ str(scenario_tree_random_seed))
            print("Linearize non-binary penalty terms: "
                  + str(linearize_nonbinary_penalty_terms))

        if self._initialized:
            raise RuntimeError("PH solver servers cannot currently be "
                               "re-initialized")

        # let plugins know if they care.
        if self._verbose:
            print("Invoking pre-initialization PHSolverServer plugins")
        for plugin in self._ph_plugins:
            plugin.pre_ph_initialization(self)

        self._objective_sense = objective_sense
        self._verbose = verbose
        self._rho = default_rho
        self._linearize_nonbinary_penalty_terms = linearize_nonbinary_penalty_terms
        self._retain_quadratic_binary_terms = retain_quadratic_binary_terms
        self._breakpoint_strategy = breakpoint_strategy
        self._integer_tolerance = integer_tolerance

        # the solver instance is persistent, applicable to all instances here.
        self._solver_type = solver_type
        self._solver_io = solver_io
        if self._verbose:
            print("Constructing solver type="+solver_type)
        self._solver = SolverFactory(solver_type,solver_io=self._solver_io)
        if self._solver == None:
            raise ValueError("Unknown solver type=" + solver_type + " specified")

        # we need the base model to construct
        # the scenarios that this server is responsible for.
        # TBD - revisit the various "weird" scenario tree arguments

        # GAH: At this point these should never be any form of
        #      compressed archive (unless I messed up the code) We
        #      want to avoid multiple unarchiving of the scenario and
        #      model directories. This should have been done once on
        #      the master node before this point, meaning these names
        #      should point to the unarchived directories.
        assert os.path.isfile(model_location)
        assert os.path.isfile(data_location)
        scenario_instance_factory = ScenarioTreeInstanceFactory(model_location,
                                                                data_location,
                                                                self._verbose)
        self._scenario_tree = scenario_instance_factory.generate_scenario_tree(
            downsample_fraction=None,
            bundles_file=scenario_bundle_specification,
            random_bundles=create_random_bundles,
            random_seed=scenario_tree_random_seed)

        if self._scenario_tree is None:
             raise RuntimeError("Unable to launch PH solver server - scenario tree construction failed.")

        scenarios_to_construct = []

        if self._scenario_tree.contains_bundles():

            # validate that the bundle actually exists.
            if self._scenario_tree.contains_bundle(object_name) is False:
                raise RuntimeError("Bundle="+object_name+" does not exist.")

            if self._verbose:
                print("Loading scenarios for bundle="+object_name)

            # bundling should use the local or "mini" scenario tree -
            # and then enable the flag to load all scenarios for this
            # instance.
            scenario_bundle = self._scenario_tree.get_bundle(object_name)
            scenarios_to_construct = scenario_bundle._scenario_names

        else:

            scenarios_to_construct.append(object_name)

        instance_factory = self._scenario_tree._scenario_instance_factory
        self._scenario_tree._scenario_instance_factory = None
        self._uncompressed_scenario_tree = copy.deepcopy(self._scenario_tree)
        self._scenario_tree._scenario_instance_factory = instance_factory
        # compact the scenario tree to reflect those instances for
        # which this ph solver server is responsible for constructing.
        self._scenario_tree.compress(scenarios_to_construct)

        instances = self._scenario_tree._scenario_instance_factory.\
                    construct_instances_for_scenario_tree(
                        self._scenario_tree)

        # with the scenario instances now available, have the scenario
        # tree compute the variable match indices at each node.
        self._scenario_tree.linkInInstances(instances,
                                            self._objective_sense,
                                            create_variable_ids=True)

        self._objective_sense = \
            self._scenario_tree._scenarios[0]._objective_sense

        self._setup_scenario_instances()

        # let plugins know if they care.
        if self._verbose:
            print("Invoking post-instance-creation PHSolverServer plugins")
        # let plugins know if they care - this callback point allows
        # users to create/modify the original scenario instances
        # and/or the scenario tree prior to creating PH-related
        # parameters, variables, and the like.
        for plugin in self._ph_plugins:
            plugin.post_instance_creation(self)

        # augment the instance with PH-specific parameters (weights,
        # rhos, etc).  this value and the linearization parameter as a
        # command-line argument.
        self._create_scenario_ph_parameters()

        # create symbol maps for easy storage/transmission of variable
        # values
        symbol_ctypes = (Var, Suffix)
        self._create_instance_symbol_maps(symbol_ctypes)

        # form the ph objective weight and proximal expressions Note:
        # The Expression objects for the weight and proximal terms
        # will be added to the instances objectives but will be
        # assigned values of 0.0, so that the original objective
        # function form is maintained.  The purpose is so that we can
        # use this shared Expression object in the bundle binding
        # instance objectives as well when we call
        # _form_bundle_binding_instances a few lines down (so
        # regeneration of bundle objective expressions is no longer
        # required before each iteration k solve.

        self.add_ph_objective_weight_terms()
        # Note: call this function on the base class to avoid the
        #       check for whether this phsolverserver has been
        #       initialized
        _PHBase.deactivate_ph_objective_weight_terms(self)
        self.add_ph_objective_proximal_terms()
        # Note: call this function on the base class to avoid the
        #       check for whether this phsolverserver has been
        #       initialized
        _PHBase.deactivate_ph_objective_proximal_terms(self)

        # create the bundle extensive form, if bundling.
        if self._scenario_tree.contains_bundles() is True:
            self._form_bundle_binding_instances(preprocess_objectives=False)

        # Delay any preprocessing of the scenario instances
        # until we are inside the solve method. This gives users a
        # chance to further modify the instances (e.g., boundsetter
        # callbacks) without having to worry about setting preprocessor
        # tags because we are going to preprocess the entire instance
        # anyway.

        # let plugins know if they care.
        if self._verbose:
            print("Invoking post-initialization PHSolverServer plugins")
        for plugin in self._ph_plugins:
            plugin.post_ph_initialization(self)

        # we're good to go!
        self._initialized = True

    def collect_results(self, scenario_name, results_flags):

        if self._scenario_tree.contains_bundles():
            results = {}
            for scenario in self._scenario_tree._scenarios:
                node_list = []
                if TransmitType.TransmitNonLeafStages(results_flags):
                    # exclude the leaf node
                    node_list.extend([n._name for n in scenario._node_list[:-1]])
                if TransmitType.TransmitAllStages(results_flags):
                    node_list.append(scenario._leaf_node._name)
                results[scenario._name] = \
                    scenario.package_current_solution(
                        translate_ids=self._reverse_master_scenario_tree_id_map,
                        node_names=node_list)
        else:
            scenario = self._scenario_tree._scenario_map[scenario_name]
            node_list = []
            if TransmitType.TransmitNonLeafStages(results_flags):
                # exclude the leaf node
                node_list.extend([n._name for n in scenario._node_list[:-1]])
            if TransmitType.TransmitAllStages(results_flags):
                node_list.append(scenario._leaf_node._name)
            results = scenario.package_current_solution(
                translate_ids=self._reverse_master_scenario_tree_id_map,
                node_names=node_list)

        return results

    def solve(self,
              object_name,
              tee,
              keepfiles,
              symbolic_solver_labels,
              output_fixed_variable_bounds,
              solver_options,
              solver_suffixes,
              warmstart,
              variable_transmission):

        if self._verbose:
            if self._scenario_tree.contains_bundles() is True:
                print("Received request to solve scenario bundle="+object_name)
            else:
                print("Received request to solve scenario instance="+object_name)

        if self._initialized is False:
            raise RuntimeError("PH solver server has not been initialized!")

        self._tee = tee
        self._symbolic_solver_labels = symbolic_solver_labels
        self._write_fixed_variables = output_fixed_variable_bounds
        self._solver_options = solver_options
        self._solver_suffixes = solver_suffixes
        self._warmstart = warmstart
        self._variable_transmission = variable_transmission

        if self._first_solve:
            # let plugins know if they care.
            if self._verbose:
                print("Invoking pre-iteration-0-solve PHSolverServer plugins")
            for plugin in self._ph_plugins:
                plugin.pre_iteration_0_solve(self)
        else:
            # let plugins know if they care.
            if self._verbose:
                print("Invoking pre-iteration-k-solve PHSolverServer plugins")
            for plugin in self._ph_plugins:
                plugin.pre_iteration_k_solve(self)

        # process input solver options - they will be persistent
        # across to the next solve.  TBD: we might want to re-think a
        # reset() of the options, or something.
        for key,value in iteritems(self._solver_options):
            if self._verbose:
                print("Processing solver option="+key+", value="+str(value))
            self._solver.options[key] = value

        # with the introduction of piecewise linearization, the form
        # of the penalty-weighted objective is no longer fixed. thus,
        # when linearizing, we need to construct (or at least modify)
        # the constraints used to compute the linearized cost terms.
        if (self._linearize_nonbinary_penalty_terms > 0):
            # These functions will do nothing if ph proximal terms are
            # not present on the model
            self.form_ph_linearized_objective_constraints()
            # if linearizing, clear the values of the PHQUADPENALTY*
            # variables.  if they have values, this can intefere with
            # warm-starting due to constraint infeasibilities.
            if self._first_solve is False:
                self._reset_instance_linearization_variables()

        self._preprocess_scenario_instances()

        if self._first_solve:

            # if we are dealing with a persisent solver plugin, go ahead
            # and compile the instance into the solver.
            if isinstance(self._solver, PersistentSolver):
                if self._scenario_tree.contains_bundles() is True:
                    # probably for no good reason, as we should be able to
                    # hand the binding instance to the compile procedure -
                    # just not tested.
                    raise RuntimeError("***We presently can't handle bundles in "
                                       "persistent solver plugins")
                else:
                    self._solver.compile_instance(
                        self._scenario_tree.get_arbitrary_scenario()._instance)

        else:

            # GAH: We may need to redefine our concept of
            #      warmstart. These values could be helpful in the
            #      nonlinear case (or could be better than 0.0, the
            #      likely default used by the solver when these
            #      initializations are not placed in the NL
            #      file. **Note: Initializations go into the NL file
            #      independent of the "warmstart" keyword

            if self._scenario_tree.contains_bundles() is True:
                # clear non-converged variables and stage cost
                # variables, to ensure feasible warm starts.
                reset_nonconverged_variables(self._scenario_tree, self._instances)
                reset_stage_cost_variables(self._scenario_tree, self._instances)
            else:
                # clear stage cost variables, to ensure feasible warm starts.
                reset_stage_cost_variables(self._scenario_tree, self._instances)

        common_solve_kwds = {
            'tee':self._tee,
            'keepfiles':keepfiles,
            'symbolic_solver_labels':self._symbolic_solver_labels,
            'output_fixed_variable_bounds':self._write_fixed_variables,
            'suffixes':self._solver_suffixes}

        results = None
        if self._scenario_tree.contains_bundles():

            if self._scenario_tree.contains_bundle(object_name) is False:
                print("Requested scenario bundle to solve not known to PH "
                      "solver server!")
                return None

            bundle_ef_instance = self._bundle_binding_instance_map[object_name]

            if  self._warmstart and self._solver.warm_start_capable():
                results = self._solver.solve(bundle_ef_instance,
                                             warmstart=True,
                                             **common_solve_kwds)
            else:
                results = self._solver.solve(bundle_ef_instance,
                                             **common_solve_kwds)

            if self._verbose:
                print("Successfully solved scenario bundle="+object_name)

            if len(results.solution) == 0:
                results.write()
                raise RuntimeError("Solve failed for bundle="+object_name+"; "
                                   "no solutions generated")

            # load the results into the instances on the server
            # side. this is non-trivial in terms of computation time,
            # for a number of reasons. plus, we don't want to pickle
            # and return results - rather, just variable-value maps.
            bundle_ef_instance.load(
                results,
                allow_consistent_values_for_fixed_vars=\
                    self._write_fixed_variables,
                comparison_tolerance_for_fixed_vars=\
                    self._comparison_tolerance_for_fixed_vars)

            if self._verbose:
                print("Successfully loaded solution for bundle="+object_name)

            variable_values = {}
            for scenario in self._scenario_tree._scenarios:
                scenario.update_solution_from_instance()
                node_list = []
                if TransmitType.TransmitNonLeafStages(self._variable_transmission):
                    # exclude the leaf node
                    node_list.extend([n._name for n in scenario._node_list[:-1]])
                if TransmitType.TransmitAllStages(self._variable_transmission):
                    node_list.append(scenario._leaf_node._name)
                variable_values[scenario._name] = \
                    scenario.package_current_solution(
                        translate_ids=self._reverse_master_scenario_tree_id_map,
                        node_names=node_list)

            suffix_values = {}

            # suffixes are stored on the master block.
            bundle_ef_instance = self._bundle_binding_instance_map[object_name]

            for scenario in self._scenario_tree._scenarios:
                # NOTE: We are only presently extracting suffix values
                #       for constraints, as this whole interface is
                #       experimental. And probably inefficient. But it
                #       does work.
                scenario_instance = self._instances[scenario._name]
                this_scenario_suffix_values = {}
                for suffix_name in self._solver_suffixes:
                    this_suffix_map = {}
                    suffix = getattr(bundle_ef_instance, suffix_name)
                    # TODO: This needs to be over all blocks
                    for constraint_name, constraint in \
                          iteritems(scenario_instance.\
                                    active_components(Constraint)):
                        this_constraint_suffix_map = {}
                        for index, constraint_data in iteritems(constraint):
                            this_constraint_suffix_map[index] = \
                                suffix.get(constraint_data)
                        this_suffix_map[constraint_name] = this_constraint_suffix_map
                    this_scenario_suffix_values[suffix_name] = this_suffix_map
                suffix_values[scenario._name] = this_scenario_suffix_values

        else:

            if object_name not in self._instances:
                print("Requested instance to solve not in PH solver "
                      "server instance collection!")
                return None

            scenario = self._scenario_tree._scenario_map[object_name]
            scenario_instance = self._instances[object_name]

            if self._warmstart and self._solver.warm_start_capable():
                results = self._solver.solve(scenario_instance,
                                             warmstart=True,
                                             **common_solve_kwds)
            else:
                results = self._solver.solve(scenario_instance,
                                             **common_solve_kwds)

            if self._verbose:
                print("Successfully solved scenario instance="+object_name)

            if len(results.solution) == 0:
                results.write()
                raise RuntimeError("Solve failed for scenario="
                                   +object_name+"; no solutions generated")

            # load the results into the instances on the server
            # side. this is non-trivial in terms of computation time,
            # for a number of reasons. plus, we don't want to pickle
            # and return results - rather, just variable-value maps.
            scenario_instance.load(
                results,
                allow_consistent_values_for_fixed_vars=\
                   self._write_fixed_variables,
                comparison_tolerance_for_fixed_vars=\
                   self._comparison_tolerance_for_fixed_vars)

            scenario.update_solution_from_instance()
            node_list = []
            if TransmitType.TransmitNonLeafStages(self._variable_transmission):
                # exclude the leaf node
                node_list.extend([n._name for n in scenario._node_list[:-1]])
            if TransmitType.TransmitAllStages(self._variable_transmission):
                node_list.append(scenario._leaf_node._name)
            variable_values = \
                scenario.package_current_solution(
                    translate_ids=self._reverse_master_scenario_tree_id_map,
                    node_names=node_list)

            if self._verbose:
                print("Successfully loaded solution for scenario="+object_name)

            # extract suffixes into a dictionary, mapping suffix names
            # to dictionaries that in turn map constraint names to
            # (index, suffix-value) pairs.
            suffix_values = {}

            # NOTE: We are only presently extracting suffix values for
            #       constraints, as this whole interface is
            #       experimental. And probably inefficient. But it
            #       does work.
            for suffix_name in self._solver_suffixes:
                this_suffix_map = {}
                suffix = getattr(scenario_instance, suffix_name)
                # TODO: This needs to be over all blocks
                for constraint_name, constraint in \
                      iteritems(scenario_instance.active_components(Constraint)):
                    this_constraint_suffix_map = {}
                    for index, constraint_data in iteritems(constraint):
                        this_constraint_suffix_map[index] = \
                            suffix.get(constraint_data)
                    this_suffix_map[constraint_name] = this_constraint_suffix_map
                suffix_values[suffix_name] = this_suffix_map

        self._solver_results[object_name] = results

        # auxilliary values are those associated with the solve
        # itself.
        auxilliary_values = {}

        solution0 = results.solution(0)
        if hasattr(solution0, "gap") and \
           (not isinstance(solution0.gap, UndefinedData)) and \
           (solution0.gap is not None):
            auxilliary_values["gap"] = solution0.gap

        # if the solver plugin doesn't populate the
        # user_time field, it is by default of type
        # UndefinedData - defined in coopr.opt.results
        if hasattr(results.solver,"user_time") and \
           (not isinstance(results.solver.user_time, UndefinedData)) and \
           (results.solver.user_time is not None):
            # the solve time might be a string, or might
            # not be - we eventually would like more
            # consistency on this front from the solver
            # plugins.
            auxilliary_values["user_time"] = \
                float(results.solver.user_time)

        solve_method_result = (variable_values, suffix_values, auxilliary_values)

        if self._first_solve is True:
            # let plugins know if they care.
            if self._verbose:
                print("Invoking post-iteration-0-solve PHSolverServer plugins")
            for plugin in self._ph_plugins:
                plugin.post_iteration_0_solve(self)
        else:
            # let plugins know if they care.
            if self._verbose:
                print("Invoking post-iteration-k-solve PHSolverServer plugins")
            for plugin in self._ph_plugins:
                plugin.post_iteration_k_solve(self)

        self._first_solve = False

        return solve_method_result

    def update_master_scenario_tree_ids(self, object_name, new_ids):

        if self._verbose:
            if self._scenario_tree.contains_bundles() is True:
                print("Received request to update xbars for bundle="+object_name)
            else:
                print("Received request to update xbars for scenario="+object_name)

        if self._initialized is False:
            raise RuntimeError("PH solver server has not been initialized!")


        for node_name, new_master_node_ids in iteritems(new_ids):
            tree_node = self._scenario_tree.get_node(node_name)
            name_index_to_id = tree_node._name_index_to_id

            self._master_scenario_tree_id_map[tree_node._name] = \
                dict((master_variable_id, name_index_to_id[name_index]) for \
                      master_variable_id, name_index in iteritems(new_master_node_ids))

            self._reverse_master_scenario_tree_id_map[tree_node._name] = \
                dict((local_variable_id, master_variable_id) for \
                     master_variable_id, local_variable_id in \
                     iteritems(self._master_scenario_tree_id_map[tree_node._name]))

    def update_xbars(self, object_name, new_xbars):

        if self._verbose:
            if self._scenario_tree.contains_bundles() is True:
                print("Received request to update xbars for bundle="+object_name)
            else:
                print("Received request to update xbars for scenario="+object_name)

        if self._initialized is False:
            raise RuntimeError("PH solver server has not been initialized!")

        for node_name, node_xbars in iteritems(new_xbars):
            tree_node = self._scenario_tree._tree_node_map[node_name]
            master_id_map = self._master_scenario_tree_id_map[tree_node._name]
            tree_node._xbars.update((master_id_map[master_id_index],val) \
                                    for master_id_index, val in iteritems(node_xbars))

    #
    # updating weights only applies to scenarios - not bundles.
    #
    def update_weights(self, scenario_name, new_weights):

        if self._verbose:
            print("Received request to update weights for scenario="+scenario_name)

        if self._initialized is False:
            raise RuntimeError("PH solver server has not been initialized!")

        if scenario_name not in self._instances:
            print("ERROR: Received request to update weights for instance not in PH solver server instance collection!")
            return None

        scenario = self._scenario_tree._scenario_map[scenario_name]
        for tree_node_name, tree_node_weights in iteritems(new_weights):
            master_id_map = self._master_scenario_tree_id_map[tree_node_name]
            scenario._w[tree_node_name].update(
                (master_id_map[master_id_index], val) \
                for master_id_index, val in \
                iteritems(tree_node_weights))

    #
    # updating rhos is only applicable to scenarios.
    #
    def update_rhos(self, scenario_name, new_rhos):

        if self._verbose:
            print("Received request to update rhos for scenario="+scenario_name)

        if self._initialized is False:
            raise RuntimeError("PH solver server has not been initialized!")

        if scenario_name not in self._instances:
            print("ERROR: Received request to update rhos for scenario="+scenario_name+", which is not in the PH solver server instance set="+str(self._instances.keys()))
            return None

        scenario = self._scenario_tree._scenario_map[scenario_name]
        for tree_node_name, tree_node_rhos in iteritems(new_rhos):
            master_id_map = self._master_scenario_tree_id_map[tree_node_name]
            scenario._rho[tree_node_name].update(
                (master_id_map[master_id_index], val) \
                for master_id_index, val in \
                iteritems(tree_node_rhos))

    #
    # updating tree node statistics is bundle versus scenario agnostic.
    #

    def update_tree_node_statistics(self,
                                    scenario_name,
                                    new_node_minimums,
                                    new_node_maximums):

        if self._verbose:
            if self._scenario_tree.contains_bundles() is True:
                print("Received request to update tree node "
                      "statistics for bundle="+scenario_name)
            else:
                print("Received request to update tree node "
                      "statistics for scenario="+scenario_name)

        if self._initialized is False:
            raise RuntimeError("PH solver server has not been initialized!")

        for tree_node_name, tree_node_minimums in iteritems(new_node_minimums):
            master_id_map = self._master_scenario_tree_id_map[tree_node_name]
            this_tree_node_minimums = \
                self._scenario_tree._tree_node_map[tree_node_name]._minimums
            this_tree_node_minimums.update(
                (master_id_map[master_id_index], value) \
                for master_id_index, value in iteritems(tree_node_minimums))

        for tree_node_name, tree_node_maximums in iteritems(new_node_maximums):
            master_id_map = self._master_scenario_tree_id_map[tree_node_name]
            this_tree_node_maximums = \
                self._scenario_tree._tree_node_map[tree_node_name]._maximums
            this_tree_node_maximums.update(
                (master_id_map[master_id_index], value) \
                for master_id_index, value in iteritems(tree_node_maximums))

    #
    # define the indicated suffix on my scenario instance. not dealing
    # with bundles right now.
    #

    def define_import_suffix(self, object_name, suffix_name):

        if self._verbose:
            if self._scenario_tree.contains_bundles() is True:
                print("Received request to define import suffix="+suffix_name+" for bundle="+object_name)
            else:
                print("Received request to define import suffix="+suffix_name+" for scenario="+object_name)

        if self._initialized is False:
            raise RuntimeError("PH solver server has not been initialized!")

        if self._scenario_tree.contains_bundles() is True:

            bundle_ef_instance = self._bundle_binding_instance_map[object_name]

            bundle_ef_instance.add_component(suffix_name, Suffix(direction=Suffix.IMPORT))

        else:

            if object_name not in self._instances:
                print("ERROR: Received request to define import suffix="+suffix_name+" for scenario="+object_name+", which is not in the collection of PH solver server instances="+str(self._instances.keys()))
                return None
            scenario_instance = self._instances[object_name]

            scenario_instance.add_component(suffix_name, Suffix(direction=Suffix.IMPORT))

    #
    # invoke the indicated function in the specified module. independent of scenario instance.
    #

    def invoke_external_function(self,
                                 object_name,
                                 invocation_type,
                                 module_name,
                                 function_name,
                                 function_args,
                                 function_kwds):

        from pyutilib.misc import import_file
        from six import iterkeys

        if self._verbose:
            if self._scenario_tree.contains_bundles():
                print("Received request to invoke external function"
                      "="+function_name+" in module="+module_name+" "
                      "for bundle="+object_name)
            else:
                print("Received request to invoke external function"
                      "="+function_name+" in module="+module_name+" "
                      "for scenario="+object_name)

        if self._initialized is False:
            raise RuntimeError("PH solver server has not been initialized!")

        scenario_tree_object = None
        if self._scenario_tree.contains_bundles():
            scenario_tree_object = self._scenario_tree._scenario_bundle_map[object_name]
        else:
            scenario_tree_object = self._scenario_tree._scenario_map[object_name]

        this_module = import_file(module_name)

        module_attrname = function_name
        subname = None
        if not hasattr(this_module, module_attrname):
            if "." in module_attrname:
                module_attrname, subname = function_name.split(".",1)
            if not hasattr(this_module, module_attrname):
                raise RuntimeError("Function="+function_name+" is not present "
                                   "in module="+module_name)

        if function_args is None:
            function_args = ()
        if function_kwds is None:
            function_kwds = {}

        call_objects = None
        if invocation_type == InvocationType.SingleInvocation:
            if self._scenario_tree.contains_bundles():
                call_objects = (object_name,self._scenario_tree._scenario_bundle_map[object_name])
            else:
                call_objects = (object_name,self._scenario_tree._scenario_map[object_name])
        elif (invocation_type == InvocationType.PerBundleInvocation) or \
             (invocation_type == InvocationType.PerBundleChainedInvocation):
            if not self._scenario_tree.contains_bundles():
                raise ValueError("Received request for bundle invocation type "
                                 "but the scenario tree does not contain bundles.")
            call_objects = iteritems(self._scenario_tree._scenario_bundle_map)
        elif (invocation_type == InvocationType.PerScenarioInvocation) or \
             (invocation_type == InvocationType.PerScenarioChainedInvocation):
            call_objects = iteritems(self._scenario_tree._scenario_map)
        elif (invocation_type == InvocationType.PerNodeInvocation) or \
             (invocation_type == InvocationType.PerNodeChainedInvocation):
            call_objects = iteritems(self._scenario_tree._tree_node_map)
        else:
            raise ValueError("Unexpected function invocation type '%s'. "
                             "Expected one of %s"
                             % (invocation_type,
                                [str(v) for v in InvocationType._values]))

        function = getattr(this_module, module_attrname)
        if subname is not None:
            function = getattr(function, subname)

        if invocation_type == InvocationType.SingleInvocation:
            call_name, call_object = call_objects
            return function(self,
                            self._scenario_tree,
                            call_object,
                            *function_args,
                            **function_kwds)
        elif (invocation_type == InvocationType.PerBundleChainedInvocation) or \
             (invocation_type == InvocationType.PerScenarioChainedInvocation) or \
             (invocation_type == InvocationType.PerNodeChainedInvocation):
            result = function_args
            for call_name, call_object in call_objects:
                result = function(self,
                                  self._scenario_tree,
                                  call_object,
                                  *result,
                                  **function_kwds)
            return result
        else:
            return dict((call_name,function(self,
                                            self._scenario_tree,
                                            call_object,
                                            *function_args,
                                            **function_kwds))
                        for call_name, call_object in call_objects)

    #
    # restore solutions for all of scenario instances.
    #

    def restoreCachedSolutions(self, object_name, cache_id, release_cache):

        if self._verbose:
            if self._scenario_tree.contains_bundles() is True:
                print("Received request to restore cached solution for bundle="+object_name)
            else:
                print("Received request to restore cached solution for scenario="+object_name)
            print("Restoring from cache id: "+str(cache_id))

        if self._initialized is False:
            raise RuntimeError("PH solver server has not been initialized!")

        if self._scenario_tree.contains_bundles() is True:
            # validate that the bundle actually exists.
            if self._scenario_tree.contains_bundle(object_name) is False:
                raise RuntimeError("Bundle="+object_name+" does not exist.")
        else:
            if object_name not in self._instances:
                raise RuntimeError(object_name+" is not in the PH solver server instance collection")

        _PHBase.restoreCachedSolutions(self, cache_id, release_cache)

    #
    # restore solutions for all of scenario instances.
    #

    def cacheSolutions(self, object_name, cache_id):

        if self._verbose:
            if self._scenario_tree.contains_bundles() is True:
                print("Received request to cache solution for bundle="+object_name)
            else:
                print("Received request to cache solution for scenario="+object_name)
            print("Caching with id: "+str(cache_id))

        if self._initialized is False:
            raise RuntimeError("PH solver server has not been initialized!")

        if self._scenario_tree.contains_bundles() is True:
            # validate that the bundle actually exists.
            if self._scenario_tree.contains_bundle(object_name) is False:
                raise RuntimeError("Bundle="+object_name+" does not exist.")
        else:
            if object_name not in self._instances:
                raise RuntimeError(object_name+" is not in the PH solver server instance collection")

        _PHBase.cacheSolutions(self, cache_id)


    #
    # fix variables as instructed by the PH client.
    #
    def update_fixed_variables(self, object_name, fixed_variables):

        if self._verbose:
            if self._scenario_tree.contains_bundles() is True:
                print("Received request to update fixed variables for "
                      "bundle="+object_name)
            else:
                print("Received request to update fixed variables for "
                      "scenario="+object_name)

        if self._initialized is False:
            raise RuntimeError("PH solver server has not been initialized!")

        for node_name, node_fixed_vars in iteritems(fixed_variables):
            tree_node = self._scenario_tree.get_node(node_name)
            tree_node._fix_queue.update(node_fixed_vars)

        self._push_fixed_to_instances()

    def process(self, data):

        suspend_gc = PauseGC()

        result = None
        if data.action == "initialize":
            result = self.initialize(data.model_location,
                                     data.data_location,
                                     data.object_name,
                                     data.objective_sense,
                                     data.solver_type,
                                     data.solver_io,
                                     data.scenario_bundle_specification,
                                     data.create_random_bundles,
                                     data.scenario_tree_random_seed,
                                     data.default_rho,
                                     data.linearize_nonbinary_penalty_terms,
                                     data.retain_quadratic_binary_terms,
                                     data.breakpoint_strategy,
                                     data.integer_tolerance,
                                     data.verbose)

        elif data.action == "collect_results":
            result = self.collect_results(data.name,
                                          data.var_config)
        elif data.action == "solve":
            # we are adding the following code because some solvers, including
            # CPLEX, are not all that robust - in that they can spontaneously
            # and sporadically fail. ultimately, this should be command-line
            # option driver.
            max_num_attempts = 2
            attempts_so_far = 0
            successful_solve = False
            while (not successful_solve):
                try:
                    attempts_so_far += 1
                    result = self.solve(data.name,
                                        data.tee,
                                        data.keepfiles,
                                        data.symbolic_solver_labels,
                                        data.output_fixed_variable_bounds,
                                        data.solver_options,
                                        data.solver_suffixes,
                                        data.warmstart,
                                        data.variable_transmission)
                    successful_solve = True
                except pyutilib.common.ApplicationError as exc:
                    print("Solve failed for object=%s - this was attempt=%d"
                          % (data.name, attempts_so_far))
                    if (attempts_so_far == max_num_attempts):
                        print("Aborting PH solver server - the maximum number "
                              "of solve attempts=%d have been executed"
                              % (max_num_attempts))
                        raise exc

            if attempts_so_far > 1:
                print("Successfully recovered from failed solve for "
                      "object=%s" % (data.name))

        elif data.action == "activate_ph_objective_proximal_terms":
            self.activate_ph_objective_proximal_terms()
            result = True

        elif data.action == "deactivate_ph_objective_proximal_terms":
            self.deactivate_ph_objective_proximal_terms()
            result = True

        elif data.action == "activate_ph_objective_weight_terms":
            self.activate_ph_objective_weight_terms()
            result = True

        elif data.action == "deactivate_ph_objective_weight_terms":
            self.deactivate_ph_objective_weight_terms()
            result = True

        elif data.action == "update_scenario_tree_ids":
            self.update_master_scenario_tree_ids(data.name,
                                                 data.new_ids)
            result = True

        elif data.action == "load_rhos":
            if self._scenario_tree.contains_bundles() is True:
                for scenario_name, scenario_instance in iteritems(self._instances):
                    self.update_rhos(scenario_name,
                                     data.new_rhos[scenario_name])
            else:
                self.update_rhos(data.name,
                                 data.new_rhos)
            result = True
            self._push_rho_to_instances()

        elif data.action == "update_fixed_variables":
            self.update_fixed_variables(data.name,
                                        data.fixed_variables)
            result = True
            self._push_fixed_to_instances()

        elif data.action == "load_weights":
            if self._scenario_tree.contains_bundles() is True:
                for scenario_name, scenario_instance in iteritems(self._instances):
                    self.update_weights(scenario_name,
                                        data.new_weights[scenario_name])
            else:
                self.update_weights(data.name,
                                    data.new_weights)
            result = True
            self._push_w_to_instances()

        elif data.action == "load_xbars":
            self.update_xbars(data.name,
                              data.new_xbars)
            result = True
            self._push_xbar_to_instances()

        elif data.action == "load_tree_node_stats":
            self.update_tree_node_statistics(data.name,
                                             data.new_mins,
                                             data.new_maxs)
            result = True

        elif data.action == "define_import_suffix":
            self.define_import_suffix(data.name,
                                      data.suffix_name)
            result = True

        elif data.action == "invoke_external_function":
           result = self.invoke_external_function(data.name,
                                                  data.invocation_type,
                                                  data.module_name,
                                                  data.function_name,
                                                  data.function_args,
                                                  data.function_kwds)

        elif data.action == "restore_cached_scenario_solutions":
            # don't pass the scenario argument - by default,
            # we restore solutions for all of our instances.
            self.restoreCachedSolutions(data.name,
                                        data.cache_id,
                                        data.release_cache)
            result = True
        elif data.action == "cache_scenario_solutions":
            # don't pass the scenario argument - by default,
            # we restore solutions for all of our instances.
            self.cacheSolutions(data.name,
                                data.cache_id)
            result = True
        elif data.action == "collect_scenario_tree_data":
            result = self.collect_scenario_tree_data(data.tree_object_names)
        else:
            raise RuntimeError("ERROR: Unknown action="+str(data.action)+" received by PH solver server")

        return result
#
# utility method to construct an option parser for ph arguments, to be
# supplied as an argument to the runph method.
#

def construct_options_parser(usage_string):

    parser = OptionParser()
    parser.add_option("--verbose",
                      help="Generate verbose output for both initialization and execution. Default is False.",
                      action="store_true",
                      dest="verbose",
                      default=False)
    parser.add_option("--profile",
                      help="Enable profiling of Python code.  The value of this option is the number of functions that are summarized.",
                      action="store",
                      dest="profile",
                      type="int",
                      default=0)
    parser.add_option("--disable-gc",
                      help="Disable the python garbage collecter. Default is False.",
                      action="store_true",
                      dest="disable_gc",
                      default=False)
    parser.add_option('--user-defined-extension',
                      help="The name of a python module specifying a user-defined PHSolverServer extension plugin.",
                      action="append",
                      dest="user_defined_extensions",
                      type="string",
                      default=[])

    parser.usage=usage_string

    return parser

#
# Execute the PH solver server daemon.
#

def run_server(options):

    # just spawn the daemon!
    TaskWorkerServer(PHPyroWorker)

def run(args=None):

    #
    # Top-level command that executes the ph solver server daemon.
    # This is segregated from phsolverserver to faciliate profiling.
    #

    #
    # Parse command-line options.
    #
    try:
        options_parser = construct_options_parser("phsolverserver [options]")
        (options, args) = options_parser.parse_args(args=args)
    except SystemExit:
        # the parser throws a system exit if "-h" is specified - catch
        # it to exit gracefully.
        return

    # for a one-pass execution, garbage collection doesn't make
    # much sense - so it is disabled by default. Because: It drops
    # the run-time by a factor of 3-4 on bigger instances.
    if options.disable_gc:
        gc.disable()
    else:
        gc.enable()

    # disable all plugins up-front. then, enable them on an as-needed
    # basis later in this function.
    ph_extension_point = ExtensionPoint(IPHSolverServerExtension)

    for plugin in ph_extension_point:
       plugin.disable()

    if len(options.user_defined_extensions) > 0:
        for this_extension in options.user_defined_extensions:
            if this_extension in sys.modules:
                print("User-defined PHSolverServer extension module="+this_extension+" already imported - skipping")
            else:
                print("Trying to import user-defined PHSolverServer extension module="+this_extension)
                # make sure "." is in the PATH.
                original_path = list(sys.path)
                sys.path.insert(0,'.')
                import_file(this_extension)
                print("Module successfully loaded")
                sys.path[:] = original_path # restore to what it was

            # now that we're sure the module is loaded, re-enable this specific plugin.
            # recall that all plugins are disabled by default in phinit.py, for various
            # reasons. if we want them to be picked up, we need to enable them explicitly.
            import inspect
            module_to_find = this_extension
            if module_to_find.rfind(".py"):
                module_to_find = module_to_find.rstrip(".py")
            if module_to_find.find("/") != -1:
                module_to_find = string.split(module_to_find,"/")[-1]

            for name, obj in inspect.getmembers(sys.modules[module_to_find], inspect.isclass):
                import coopr.core
                # the second condition gets around goofyness related to issubclass returning
                # True when the obj is the same as the test class.
                if issubclass(obj, coopr.core.plugin.SingletonPlugin) and name != "SingletonPlugin":
                    ph_extension_point = ExtensionPoint(IPHSolverServerExtension)
                    for plugin in ph_extension_point(all=True):
                        if isinstance(plugin, obj):
                            plugin.enable()

    if pstats_available and options.profile > 0:
        #
        # Call the main PH routine with profiling.
        #
        tfile = TempfileManager.create_tempfile(suffix=".profile")
        tmp = profile.runctx('run_server(options)',globals(),locals(),tfile)
        p = pstats.Stats(tfile).strip_dirs()
        p.sort_stats('time', 'cumulative')
        p = p.print_stats(options.profile)
        p.print_callers(options.profile)
        p.print_callees(options.profile)
        p = p.sort_stats('cumulative','calls')
        p.print_stats(options.profile)
        p.print_callers(options.profile)
        p.print_callees(options.profile)
        p = p.sort_stats('calls')
        p.print_stats(options.profile)
        p.print_callers(options.profile)
        p.print_callees(options.profile)
        TempfileManager.clear_tempfiles()
        ans = [tmp, None]
    else:
        #
        # Call the main PH routine without profiling.
        #
        ans = run_server(options)

    gc.enable()

    return ans

@coopr_command('phsolverserver', "Pyro-based server for PH solvers")
def main():
    import coopr.environ
    exception_trapped = False
    try:
        run()
    except IOError:
        msg = sys.exc_info()[1]
        print("IO ERROR:")
        print(msg)
        exception_trapped = True
    except pyutilib.common.ApplicationError:
        msg = sys.exc_info()[1]
        print("APPLICATION ERROR:")
        print(str(msg))
        exception_trapped = True
    except RuntimeError:
        msg = sys.exc_info()[1]
        print("RUN-TIME ERROR:")
        print(str(msg))
        exception_trapped = True
    # pyutilib.pyro tends to throw SystemExit exceptions if things
    # cannot be found or hooked up in the appropriate fashion. the
    # name is a bit odd, but we have other issues to worry about. we
    # are dumping the trace in case this does happen, so we can figure
    # out precisely who is at fault.
    except SystemExit:
        msg = sys.exc_info()[1]
        print("PH solver server encountered system error")
        print("Error: "+str(msg))
        print("Stack trace:")
        traceback.print_exc()
        exception_trapped = True
    except:
        print("Encountered unhandled exception")
        traceback.print_exc()
        exception_trapped = True

    # if an exception occurred, then we probably want to shut down all
    # Pyro components.  otherwise, the PH client may have forever
    # while waiting for results that will never arrive. there are
    # better ways to handle this at the PH client level, but until
    # those are implemented, this will suffice for cleanup.
    #NOTE: this should perhaps be command-line driven, so it can be
    #      disabled if desired.
    if exception_trapped == True:
        print("PH solver server aborted")
        shutDownPyroComponents()

