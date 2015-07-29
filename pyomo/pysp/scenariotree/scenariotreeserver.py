#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
import copy
import argparse

import pyutilib.misc
from pyutilib.misc import PauseGC
from pyutilib.misc.config import ConfigBlock
from pyutilib.pyro import (MultiTaskWorker,
                           TaskWorkerServer,
                           shutdown_pyro_components)
from pyomo.util import pyomo_command
from pyomo.opt import (SolverFactory,
                       PersistentSolver,
                       TerminationCondition,
                       SolutionStatus)
from pyomo.pysp.scenariotree import ScenarioTreeInstanceFactory
from pyomo.pysp.scenariotree.scenariotreeserverutils import \
    InvocationType
from pyomo.pysp.util.config import safe_register_common_option
from pyomo.pysp.util.misc import launch_command

from six import iteritems

class ScenarioTreeServer(MultiTaskWorker):

    def __init__(self, **kwds):

        # add for purposes of diagnostic output.
        kwds["caller_name"] = "ScenarioTreeServer"

        MultiTaskWorker.__init__(self,**kwds)

        # Requests for employement when this worker is idle
        self._idle_queue_blocking_timeout = (True, 5)

        # Requests for new jobs when this worker is aquired but owns
        # no jobs
        self._worker_queue_blocking_timeout = (True, 0.1)

        # Requests for new jobs when this worker owns at least one
        # other job
        self._assigned_worker_queue_blocking_timeout = (False, None)
        # Requests for new tasks specific to current job(s)
        self._solver_queue_blocking_timeout = (True, 0.1)

        self._init()

    def _init(self):

        self.clear_request_types()
        # queue type, blocking, timeout
        self.push_request_type('phpyro_worker_idle',
                               *self._idle_queue_blocking_timeout)
        self._worker_map = {}

    def del_worker(self, name):
        del self._worker_map[name]
        types_to_keep = []
        for rqtype in self.current_type_order():
            if rqtype[0] != name:
                types_to_keep.append(rqtype)
        self.clear_request_types()
        for rqtype in types_to_keep:
            self.push_request_type(*rqtype)

    def process(self, data):

        data = pyutilib.misc.Bunch(**data)
        result = None
        if data.action == "acknowledge":

            assert self.num_request_types() == 1
            self.clear_request_types()
            self.push_request_type(self.WORKERNAME,
                                   *self._worker_queue_blocking_timeout)
            result = self.WORKERNAME

        elif (data.name == self.WORKERNAME) and (data.action == "release"):

            self.del_worker(data.object_name)
            if len(self.current_type_order()) == 1:
                # Go back to making general worker requests
                # blocking with a reasonable timeout so they
                # don't overload the dispatcher
                self.pop_request_type()
                self.push_request_type(self.WORKERNAME,
                                       *self._worker_queue_blocking_timeout)
            result = True

        elif (data.name == self.WORKERNAME) and (data.action == "go_idle"):

            server_names = list(self._worker_map.keys())
            for name in server_names:
                self.del_worker(name)
            self._init()
            result = True

        else:

            if (data.name == self.WORKERNAME) and (data.action == "initialize"):
                current_types = self.current_type_order()
                for rqtype in current_types:
                    if data.object_name == rqtype[0]:
                        raise RuntimeError("Cannot initialize object with name '%s' "
                                           "because a work queue already exists with "
                                           "this name" % (data.object_name))

                assert current_types[-1][0] == self.WORKERNAME
                if len(current_types) == 1:
                    # make the general worker request non blocking
                    # as we now have higher priority work to perform
                    self.pop_request_type()
                    self.push_request_type(self.WORKERNAME,
                                           False,
                                           None)

                self.push_request_type(data.object_name,
                                       *self._solver_queue_blocking_timeout)
                self._worker_map[data.object_name] = _ScenarioTreeWorker()
                self._worker_map[data.object_name].WORKERNAME = self.WORKERNAME
                data.name = data.object_name

            with PauseGC() as pgc:
                result = self._worker_map[data.name].process(data)

        return result

class _ScenarioTreeWorker(object):

    def __init__(self):

        self._initialized = False
        self._verbose = False
        self._scenario_tree = None
        self._solver_manager = None
        self._instances = None
        self._objective_sense = None
        # So we have access to real scenario and bundle probabilities
        self._uncompressed_scenario_tree = None

        # Maps ScenarioTreeID's on the master node ScenarioTree to
        # ScenarioTreeID's on this ScenarioTreeWorkers's ScenarioTree
        # (by node name)
        self._master_scenario_tree_id_map = {}
        self._reverse_master_scenario_tree_id_map = {}

    def process(self, data):

        result = None
        if data.action == "initialize":
            result = self.initialize(
                data.object_name,
                data.model_location,
                data.data_location,
                data.objective_sense_option,
                data.scenario_bundle_specification,
                data.create_random_bundles,
                data.scenario_tree_random_seed,
                data.verbose)
        elif data.action == "update_master_scenario_tree_ids":
            self.update_master_scenario_tree_ids(data.name,
                                                 data.new_ids)
            result = True
        elif data.action == "collect_scenario_tree_data":
            result = self.collect_scenario_tree_data(
                data.tree_object_names)
        elif data.action == "invoke_external_function":
            result = self.invoke_external_function(
                data.name,
                data.invocation_type,
                data.module_name,
                data.function_name,
                data.function_args,
                data.function_kwds)
        else:
            raise RuntimeError("ERROR: Unknown action='%s' "
                               "received by scenario tree worker"
                               % (data.action))
        return result

    def initialize(self,
                   object_name,
                   model_location,
                   data_location,
                   objective_sense_option,
                   scenario_bundle_specification,
                   create_random_bundles,
                   scenario_tree_random_seed,
                   verbose):

        self._verbose = verbose

        if self._verbose:
            print("Received request to initialize scenario tree worker")
            print("")
            print("Model source: "+model_location)
            print("Scenario Tree source: "+str(data_location))
            print("Scenario or bundle name: "+object_name)
            if scenario_bundle_specification != None:
                print("Scenario tree bundle specification: "
                      +scenario_bundle_specification)
            if create_random_bundles != None:
                print("Create random bundles: "+str(create_random_bundles))
            if scenario_tree_random_seed != None:
                print("Scenario tree random seed: "+ str(scenario_tree_random_seed))

        if self._initialized:
            raise RuntimeError("_ScenarioTreeWorker objects cannot be "
                               "re-initialized")

        assert os.path.exists(model_location)
        assert (data_location is None) or os.path.exists(data_location)
        scenario_instance_factory = \
            ScenarioTreeInstanceFactory(model_location,
                                        data_location,
                                        self._verbose)
        self._scenario_tree = \
            scenario_instance_factory.generate_scenario_tree(
                downsample_fraction=None,
                bundles_file=scenario_bundle_specification,
                random_bundles=create_random_bundles,
                random_seed=scenario_tree_random_seed)

        if self._scenario_tree is None:
             raise RuntimeError("Unable to launch scenario tree worker - "
                                "scenario tree construction failed.")

        scenarios_to_construct = []
        if self._scenario_tree.contains_bundles():

            # validate that the bundle actually exists.
            if not self._scenario_tree.contains_bundle(object_name):
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
        # compress the scenario tree to reflect those instances for
        # which this ph solver server is responsible for constructing.
        self._scenario_tree.compress(scenarios_to_construct)

        self._instances = self._scenario_tree._scenario_instance_factory.\
                          construct_instances_for_scenario_tree(
                              self._scenario_tree)

        # with the scenario instances now available, have the scenario
        # tree compute the variable match indices at each node.
        self._scenario_tree.linkInInstances(self._instances,
                                            objective_sense_option,
                                            create_variable_ids=True)

        if self._scenario_tree.contains_bundles():
            if self._options.verbose:
                print("Forming binding instances for all scenario bundles")

            self._bundle_binding_instance_map.clear()
            self._bundle_scenario_instance_map.clear()

            if not self._scenario_tree.contains_bundles():
                raise RuntimeError("Failed to create binding instances for scenario "
                                   "bundles - no scenario bundles are defined!")

            for scenario_bundle in self._scenario_tree._scenario_bundles:

                if self._options.verbose:
                    print("Creating binding instance for scenario bundle=%s"
                          % (scenario_bundle._name))

                self._bundle_scenario_instance_map[scenario_bundle._name] = {}
                for scenario_name in scenario_bundle._scenario_names:
                    self._bundle_scenario_instance_map[scenario_bundle._name]\
                        [scenario_name] = self._instances[scenario_name]

                # IMPORTANT: The bundle variable IDs must be idential to
                #            those in the parent scenario tree - this is
                #            critical for storing results, which occurs at
                #            the full-scale scenario tree.

                scenario_bundle._scenario_tree.linkInInstances(
                    self._instances,
                    create_variable_ids=False,
                    master_scenario_tree=self._scenario_tree,
                    initialize_solution_data=False)

                bundle_ef_instance = create_ef_instance(
                    scenario_bundle._scenario_tree,
                    ef_instance_name=scenario_bundle._name,
                    verbose_output=self._verbose)

                self._bundle_binding_instance_map[scenario_bundle._name] = \
                    bundle_ef_instance

        self._objective_sense = \
            self._scenario_tree._scenarios[0]._objective_sense

        # create the bundle extensive form, if bundling.
        if self._scenario_tree.contains_bundles():
            self._form_bundle_binding_instances(preprocess_objectives=False)

        # we're good to go!
        self._initialized = True

    #
    # Update the map from local to master scenario tree ids
    #

    def update_master_scenario_tree_ids(self, object_name, new_ids):

        if self._verbose:
            if self._scenario_tree.contains_bundles():
                print("Received request to update master "
                      "scenario tree ids for bundle="+object_name)
            else:
                print("Received request to update master "
                      "scenario tree ids scenario="+object_name)

        if not self._initialized:
            raise RuntimeError("Scenario tree worker has not been initialized!")

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

    def collect_scenario_tree_data(self, tree_object_names):

        data = {}
        node_data = data['nodes'] = {}
        for node_name in tree_object_names['nodes']:
            tree_node = self._scenario_tree.get_node(node_name)
            this_node_data = node_data[node_name] = {}
            this_node_data['_variable_ids'] = tree_node._variable_ids
            this_node_data['_standard_variable_ids'] = tree_node._standard_variable_ids
            this_node_data['_variable_indices'] = tree_node._variable_indices
            this_node_data['_discrete'] = list(tree_node._discrete)
            # master will need to reconstruct
            # _derived_variable_ids
            # _name_index_to_id

        scenario_data = data['scenarios'] = {}
        for scenario_name in tree_object_names['scenarios']:
            scenario = self._scenario_tree.get_scenario(scenario_name)
            this_scenario_data = scenario_data[scenario_name] = {}
            this_scenario_data['_objective_name'] = scenario._objective_name
            this_scenario_data['_objective_sense'] = scenario._objective_sense

        return data

    #
    # Invoke the indicated function in the specified module.
    #

    def invoke_external_function(self,
                                 object_name,
                                 invocation_type,
                                 module_name,
                                 function_name,
                                 function_args,
                                 function_kwds):

        # pyutilib.Enum can not be serialized depending on the
        # serializer type used by Pyro, so we just send the
        # key name
        invocation_type = getattr(InvocationType,invocation_type)

        if self._verbose:
            if self._scenario_tree.contains_bundles():
                print("Received request to invoke external function"
                      "="+function_name+" in module="+module_name+" "
                      "for bundle="+object_name)
            else:
                print("Received request to invoke external function"
                      "="+function_name+" in module="+module_name+" "
                      "for scenario="+object_name)

        if not self._initialized:
            raise RuntimeError("Scenario tree worker has not been initialized!")

        scenario_tree_object = None
        if self._scenario_tree.contains_bundles():
            scenario_tree_object = self._scenario_tree._scenario_bundle_map[object_name]
        else:
            scenario_tree_object = self._scenario_tree._scenario_map[object_name]

        this_module = pyutilib.misc.import_file(module_name)

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
                call_objects = (object_name,
                                self._scenario_tree._scenario_bundle_map[object_name])
            else:
                call_objects = (object_name,
                                self._scenario_tree._scenario_map[object_name])
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
# utility method fill a ConfigBlock with options associated
# with the scenariotreeserver command
#

def scenariotreeserver_register_options(options):
    safe_register_common_option(options, "disable_gc")
    safe_register_common_option(options, "profile")
    safe_register_common_option(options, "traceback")
    safe_register_common_option(options, "verbose")
    safe_register_common_option(options, "pyro_hostname")

#
# Execute the scenario tree server daemon.
#
def exec_scenariotreeserver(options):

    try:
        # spawn the daemon
        TaskWorkerServer(ScenarioTreeServer,
                         host=options.pyro_hostname)
    except:
        # if an exception occurred, then we probably want to shut down
        # all Pyro components.  otherwise, the PH client may have
        # forever while waiting for results that will never
        # arrive. there are better ways to handle this at the PH
        # client level, but until those are implemented, this will
        # suffice for cleanup.
        #NOTE: this should perhaps be command-line driven, so it can
        #      be disabled if desired.
        print("Scenario tree server aborted. Sending shutdown request.")
        shutdown_pyro_components(num_retries=0)
        raise

@pyomo_command("scenariotreeserver",
               "Pyro-based server for scenario tree management")
def main(args=None):
    #
    # Top-level command that executes the scenario tree server daemon.
    #

    #
    # Import plugins
    #
    import pyomo.environ

    #
    # Parse command-line options.
    #
    options = ConfigBlock()
    scenariotreeserver_register_options(options)
    ap = argparse.ArgumentParser(prog='scenariotreeserver')
    options.initialize_argparse(ap)
    options.import_argparse(ap.parse_args(args=args))

    return launch_command(exec_scenariotreeserver,
                          options,
                          error_label="scenariotreeserver: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)

if __name__ == "__main__":
    import sys
    main(args=sys.argv[1:])
