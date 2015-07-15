#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# the intent of this module is to provide functions to interface from
# a PH client to a set of PH solver servers.

import time
import itertools

from pyutilib.enum import Enum

from six import iteritems, itervalues

InvocationType = Enum('SingleInvocation',
                      'PerBundleInvocation',
                      'PerBundleChainedInvocation',
                      'PerScenarioInvocation',
                      'PerScenarioChainedInvocation',
                      'PerNodeInvocation',
                      'PerNodeChainedInvocation')

#
# Sends a mapping between (name,index) and ScenarioTreeID so that
# phsolverservers are aware of the master nodes's ScenarioTreeID
# labeling.
#

def transmit_scenario_tree_ids(scenario_tree_manager):

    start_time = time.time()

    if scenario_tree_manager._options.verbose:
        print("Transmitting ScenarioTree variable ids to scenario tree workers")

    action_handles = []

    generate_responses = scenario_tree_manager._options.handshake_with_sppyro

    if scenario_tree_manager._scenario_tree.contains_bundles():

        for bundle in scenario_tree_manager._scenario_tree._scenario_bundles:

            ids_to_transmit = {}
            for stage in bundle._scenario_tree._stages:
                for bundle_tree_node in stage._tree_nodes:
                    # The bundle scenariotree usually isn't populated
                    # with variable value data so we need to reference
                    # the original scenariotree node
                    primary_tree_node = \
                        scenario_tree_manager._scenario_tree._tree_node_map[bundle_tree_node._name]
                    ids_to_transmit[primary_tree_node._name] = \
                        primary_tree_node._variable_ids

            action_handles.append(
                scenario_tree_manager._solver_manager.queue(
                    action="update_master_scenario_tree_ids",
                    generateResponse=generate_responses,
                    name=bundle._name,
                    new_ids=ids_to_transmit))

    else:

        for scenario in scenario_tree_manager._scenario_tree._scenarios:

            ids_to_transmit = {}
            for tree_node in scenario._node_list:
                ids_to_transmit[tree_node._name] = tree_node._variable_ids

            action_handles.append(
                scenario_tree_manager._solver_manager.queue(
                    action="update_master_scenario_tree_ids",
                    generateResponse=generate_responses,
                    name=scenario._name,
                    new_ids=ids_to_transmit))

    if generate_responses:
        scenario_tree_manager._solver_manager.wait_all(action_handles)

    end_time = time.time()

    if scenario_tree_manager._options.output_times:
        print("ScenarioTree variable ids transmission time=%.2f "
              "seconds" % (end_time - start_time))

def release_scenariotreeservers(scenario_tree_manager):

    if scenario_tree_manager._options.verbose:
        print("Releasing scenario tree servers")

    for job, worker in iteritems(scenario_tree_manager.\
                                 _sppyro_job_worker_map):

        scenario_tree_manager._solver_manager.queue(
            action="release",
            name=worker,
            object_name=job,
            generateResponse=False)
    scenario_tree_manager._sppyro_worker_jobs_map = {}
    scenario_tree_manager._sppyro_job_worker_map = {}

def initialize_scenariotree_workers(scenario_tree_manager):

    start_time = time.time()

    if scenario_tree_manager._options.verbose:
        print("Transmitting initialization information to scenario tree workers")

    if len(scenario_tree_manager._solver_manager.worker_pool) == 0:
        raise RuntimeError("No worker processes have been acquired!")

    if scenario_tree_manager._scenario_tree.contains_bundles():
        worker_jobs = [bundle._name for bundle in
                       scenario_tree_manager._scenario_tree._scenario_bundles]
    else:
        worker_jobs = [scenario._name for scenario in
                       scenario_tree_manager._scenario_tree._scenarios]

    action_handles = []
    scenario_tree_manager._sppyro_worker_jobs_map = {}
    scenario_tree_manager._sppyro_job_worker_map = {}
    for worker_name in itertools.cycle(
            scenario_tree_manager._solver_manager.worker_pool):
        if len(worker_jobs) == 0:
            break
        job_name = worker_jobs.pop()
        # both the dispatcher queue for initialization and the action name
        # are "initialize" - might be confusing, but hopefully not so
        # much.
        action_handles.append(
            scenario_tree_manager._solver_manager.queue(
                action="initialize",
                name=worker_name,
                object_name=job_name,
                model_location=(scenario_tree_manager._scenario_tree.
                                _scenario_instance_factory._model_filename),
                data_location=(scenario_tree_manager._scenario_tree.
                               _scenario_instance_factory._data_filename),
                objective_sense_option=(scenario_tree_manager._options.
                                        objective_sense_stage_based),
                scenario_bundle_specification=(scenario_tree_manager._options.
                                               scenario_bundle_specification),
                create_random_bundles=(scenario_tree_manager._options.
                                       create_random_bundles),
                scenario_tree_random_seed=(scenario_tree_manager._options.
                                           scenario_tree_random_seed),
                verbose=scenario_tree_manager._options.verbose))
        scenario_tree_manager._sppyro_worker_jobs_map.setdefault(worker_name,[]).append(job_name)
        scenario_tree_manager._sppyro_job_worker_map[job_name] = worker_name
        # This will make sure we don't come across any lingering
        # task results from a previous run that ended badly
        scenario_tree_manager._solver_manager.client.clear_queue(override_type=job_name)

    end_time = time.time()

    if scenario_tree_manager._options.output_times:
        print("Initialization transmission time=%.2f seconds"
              % (end_time - start_time))

    return action_handles

def transmit_external_function_invocation_to_worker(
        scenario_tree_manager,
        worker_name,
        module_name,
        function_name,
        invocation_type=InvocationType.SingleInvocation,
        return_action_handle=False,
        function_args=None,
        function_kwds=None):

    if scenario_tree_manager._options.verbose:
        print("Transmitting external function invocation request to "
              "scenario tree worker with name %s" % worker_name)

    generate_response = \
        scenario_tree_manager._options.handshake_with_sppyro or return_action_handle

    if scenario_tree_manager._scenario_tree.contains_bundles():
        if worker_name not in scenario_tree_manager._scenario_tree._scenario_bundle_map:
            raise ValueError("Scenario tree worker with name %s does not exist"
                             % (worker_name))
    else:
        if worker_name not in scenario_tree_manager._scenario_tree._scenario_map:
            raise ValueError("Scenario tree worker with name %s does not exist"
                             % (worker_name))

    action_handle = scenario_tree_manager._solver_manager.queue(
        action="invoke_external_function",
        name=worker_name,
        invocation_type=invocation_type.key,
        generateResponse=generate_response,
        module_name=module_name,
        function_name=function_name,
        function_kwds=function_kwds,
        function_args=function_args)

    if generate_response and (not return_action_handle):
        scenario_tree_manager._solver_manager.wait_all([action_handle])

    return action_handle if (return_action_handle) else None

def transmit_external_function_invocation(
        scenario_tree_manager,
        module_name,
        function_name,
        invocation_type=InvocationType.SingleInvocation,
        return_action_handles=False,
        function_args=None,
        function_kwds=None):

    start_time = time.time()

    if scenario_tree_manager._options.verbose:
        print("Transmitting external function invocation request "
              "to scenario tree workers")

    action_handles = []

    generate_responses = \
        scenario_tree_manager._options.handshake_with_sppyro or return_action_handles

    if scenario_tree_manager._scenario_tree.contains_bundles():

        for bundle in scenario_tree_manager._scenario_tree._scenario_bundles:

            action_handles.append(
                scenario_tree_manager._solver_manager.queue(
                    action="invoke_external_function",
                    name=bundle._name,
                    invocation_type=invocation_type.key,
                    generateResponse=generate_responses,
                    module_name=module_name,
                    function_name=function_name,
                    function_kwds=function_kwds,
                    function_args=function_args))

    else:

        for scenario in scenario_tree_manager._scenario_tree._scenarios:

            action_handles.append(
                scenario_tree_manager._solver_manager.queue(
                    action="invoke_external_function",
                    name=scenario._name,
                    invocation_type=invocation_type.key,
                    generateResponse=generate_responses,
                    module_name=module_name,
                    function_name=function_name,
                    function_kwds=function_kwds,
                    function_args=function_args))

    if generate_responses and (not return_action_handles):
        scenario_tree_manager._solver_manager.wait_all(action_handles)

    end_time = time.time()

    if scenario_tree_manager._options.output_times:
        print("External function invocation request transmission "
              "time=%.2f seconds" % (end_time - start_time))

    return action_handles if (return_action_handles) else None

def gather_scenario_tree_data(scenario_tree_manager, initialization_action_handles):

    start_time = time.time()

    if scenario_tree_manager._options.verbose:
        print("Collecting scenario tree data from scenario tree workers")

    # maps scenario names to action handles
    scenario_action_handle_map = {}
    # maps action handles to scenario names
    action_handle_scenario_map = {}

    # maps bundle names to action handles
    bundle_action_handle_map = {}
    # maps action handles to bundle names
    action_handle_bundle_map = {}

    need_node_data = \
        dict((tree_node._name,True) \
             for tree_node in scenario_tree_manager._scenario_tree._tree_nodes)
    need_scenario_data = \
        dict((scenario._name,True) \
             for scenario in scenario_tree_manager._scenario_tree._scenarios)

    if scenario_tree_manager._scenario_tree.contains_bundles():

        for scenario_bundle in scenario_tree_manager._scenario_tree._scenario_bundles:

            object_names = {}
            object_names['nodes'] = \
                [tree_node._name \
                 for scenario in scenario_bundle._scenario_tree._scenarios \
                 for tree_node in scenario._node_list \
                 if need_node_data[tree_node._name]]
            object_names['scenarios'] = \
                [scenario_name \
                 for scenario_name in scenario_bundle._scenario_names]

            new_action_handle =  scenario_tree_manager._solver_manager.queue(
                action="collect_scenario_tree_data",
                name=scenario_bundle._name,
                tree_object_names=object_names)

            bundle_action_handle_map[scenario_bundle._name] = new_action_handle
            action_handle_bundle_map[new_action_handle] = scenario_bundle._name

            for node_name in object_names['nodes']:
                need_node_data[node_name] = False
            for scenario_name in object_names['scenarios']:
                need_scenario_data[scenario_name] = False

    else:

        for scenario in scenario_tree_manager._scenario_tree._scenarios:

            object_names = {}
            object_names['nodes'] = \
                [tree_node._name for tree_node in scenario._node_list \
                 if need_node_data[tree_node._name]]
            object_names['scenarios'] = [scenario._name]

            new_action_handle = scenario_tree_manager._solver_manager.queue(
                action="collect_scenario_tree_data",
                name=scenario._name,
                tree_object_names=object_names)

            scenario_action_handle_map[scenario._name] = new_action_handle
            action_handle_scenario_map[new_action_handle] = scenario._name

            for node_name in object_names['nodes']:
                need_node_data[node_name] = False
            for scenario_name in object_names['scenarios']:
                need_scenario_data[scenario_name] = False

    assert all(not val for val in itervalues(need_node_data))
    assert all(not val for val in itervalues(need_scenario_data))

    have_node_data = \
        dict((tree_node._name,False) \
             for tree_node in scenario_tree_manager._scenario_tree._tree_nodes)
    have_scenario_data = \
        dict((scenario._name,False) \
             for scenario in scenario_tree_manager._scenario_tree._scenarios)

    if scenario_tree_manager._options.verbose:
        print("Waiting for scenario tree data extraction")

    if scenario_tree_manager._scenario_tree.contains_bundles():

        num_results_so_far = 0

        while (num_results_so_far < len(scenario_tree_manager._scenario_tree._scenario_bundles)):

            action_handle = scenario_tree_manager._solver_manager.wait_any()

            if action_handle in initialization_action_handles:
                initialization_action_handles.remove(action_handle)
                scenario_tree_manager._solver_manager.get_results(action_handle)
                continue

            bundle_results = scenario_tree_manager._solver_manager.get_results(action_handle)
            bundle_name = action_handle_bundle_map[action_handle]

            for tree_node_name, node_data in iteritems(bundle_results['nodes']):
                assert have_node_data[tree_node_name] == False
                have_node_data[tree_node_name] = True
                tree_node = scenario_tree_manager._scenario_tree.get_node(tree_node_name)
                tree_node._variable_ids.update(node_data['_variable_ids'])
                tree_node._standard_variable_ids.update(node_data['_standard_variable_ids'])
                tree_node._variable_indices.update(node_data['_variable_indices'])
                tree_node._discrete.update(node_data['_discrete'])
                # these are implied
                tree_node._derived_variable_ids = \
                    set(tree_node._variable_ids)-tree_node._standard_variable_ids
                tree_node._name_index_to_id = \
                    dict((val,key) for key,val in iteritems(tree_node._variable_ids))

            for scenario_name, scenario_data in \
                  iteritems(bundle_results['scenarios']):
                assert have_scenario_data[scenario_name] == False
                have_scenario_data[scenario_name] = True
                scenario = scenario_tree_manager._scenario_tree.get_scenario(scenario_name)
                scenario._objective_name = scenario_data['_objective_name']
                scenario._objective_sense = scenario_data['_objective_sense']

            if scenario_tree_manager._options.verbose:
                print("Successfully loaded scenario tree data "
                      "for bundle="+bundle_name)

            num_results_so_far += 1

    else:

        num_results_so_far = 0

        while (num_results_so_far < len(scenario_tree_manager._scenario_tree._scenarios)):

            action_handle = scenario_tree_manager._solver_manager.wait_any()

            if action_handle in initialization_action_handles:
                initialization_action_handles.remove(action_handle)
                scenario_tree_manager._solver_manager.get_results(action_handle)
                continue

            scenario_results = scenario_tree_manager._solver_manager.get_results(action_handle)
            scenario_name = action_handle_scenario_map[action_handle]

            for tree_node_name, node_data in iteritems(scenario_results['nodes']):
                assert have_node_data[tree_node_name] == False
                have_node_data[tree_node_name] = True
                tree_node = scenario_tree_manager._scenario_tree.get_node(tree_node_name)
                tree_node._variable_ids.update(node_data['_variable_ids'])
                tree_node._standard_variable_ids.update(node_data['_standard_variable_ids'])
                tree_node._variable_indices.update(node_data['_variable_indices'])
                tree_node._discrete.update(node_data['_discrete'])
                # these are implied
                tree_node._derived_variable_ids = \
                    set(tree_node._variable_ids)-tree_node._standard_variable_ids
                tree_node._name_index_to_id = \
                    dict((val,key) for key,val in iteritems(tree_node._variable_ids))

            for scenario_name, scenario_data in \
                  iteritems(scenario_results['scenarios']):
                assert have_scenario_data[scenario_name] == False
                have_scenario_data[scenario_name] = True
                scenario = scenario_tree_manager._scenario_tree.get_scenario(scenario_name)
                scenario._objective_name = scenario_data['_objective_name']
                scenario._objective_sense = scenario_data['_objective_sense']

            if scenario_tree_manager._options.verbose:
                print("Successfully loaded scenario tree data for "
                      "scenario="+scenario_name)

            num_results_so_far += 1

    assert all(have_node_data)
    assert all(have_scenario_data)

    if len(initialization_action_handles):
        if scenario_tree_manager._options.verbose:
            print("Waiting on remaining scenario tree worker initializations")
        scenario_tree_manager._solver_manager.wait_all(initialization_action_handles)
        while len(initialization_action_handles):
            initialization_action_handles.pop()

    end_time = time.time()

    if scenario_tree_manager._options.output_times:
        print("Scenario tree data collection time=%.2f seconds"
              % (end_time - start_time))
