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

from pyomo.core import *

from six import iteritems, itervalues

InvocationType = Enum('SingleInvocation',
                      'PerBundleInvocation',
                      'PerBundleChainedInvocation',
                      'PerScenarioInvocation',
                      'PerScenarioChainedInvocation',
                      'PerNodeInvocation',
                      'PerNodeChainedInvocation')

class TransmitType(object):

    # Transmit stale variables
    stale       = 0b0000001
    # Transmit fixed variables
    fixed       = 0b0000010
    # Transmit derived variables
    derived     = 0b0000100
    # Transmit blended variables
    blended     = 0b0001000

    @classmethod
    def TransmitStale(cls, flag):
        return (flag & cls.stale) == cls.stale
    @classmethod
    def TransmitFixed(cls, flag):
        return (flag & cls.fixed) == cls.fixed
    @classmethod
    def TransmitDerived(cls, flag):
        return (flag & cls.derived) == cls.derived

    # Scenario tree variables
    nonleaf_stages     = 0b0010000
    all_stages         = 0b0110000

    @classmethod
    def TransmitNonLeafStages(cls, flag):
        return flag & cls.nonleaf_stages == cls.nonleaf_stages
    @classmethod
    def TransmitAllStages(cls, flag):
        return flag & cls.all_stages == cls.all_stages

def collect_full_results(ph, var_config):

    start_time = time.time()

    if ph._verbose:
        print("Collecting results from PH solver servers")

    scenario_action_handle_map = {} # maps scenario names to action handles
    action_handle_scenario_map = {} # maps action handles to scenario names

    bundle_action_handle_map = {} # maps bundle names to action handles
    action_handle_bundle_map = {} # maps action handles to bundle names

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:

            new_action_handle =  ph._solver_manager.queue(
                action="collect_results",
                queue_name=ph._phpyro_job_worker_map[bundle._name],
                name=bundle._name,
                var_config=var_config)

            bundle_action_handle_map[bundle._name] = new_action_handle
            action_handle_bundle_map[new_action_handle] = bundle._name

    else:

        for scenario in ph._scenario_tree._scenarios:

            new_action_handle = ph._solver_manager.queue(
                action="collect_results",
                queue_name=ph._phpyro_job_worker_map[scenario._name],
                name=scenario._name,
                var_config=var_config)

            scenario_action_handle_map[scenario._name] = new_action_handle
            action_handle_scenario_map[new_action_handle] = scenario._name
    ph._solver_manager.end_bulk()

    if ph._scenario_tree.contains_bundles():

        if ph._verbose:
            print("Waiting for bundle results extraction")

        num_results_so_far = 0

        while (num_results_so_far < len(ph._scenario_tree._scenario_bundles)):

            action_handle = ph._solver_manager.wait_any()
            try:
                bundle_name = action_handle_bundle_map[action_handle]
            except KeyError:
                if action_handle in ph._queued_solve_action_handles:
                    ph._queued_solve_action_handles.discard(action_handle)
                    print("WARNING: Discarding uncollected solve action handle "
                          "with id=%d encountered during bundle results collection"
                          % (action_handle.id))
                    continue
                else:
                    known_action_handles = \
                        sorted((ah.id for ah in action_handle_scenario_map))
                    raise RuntimeError("PH client received an unknown action "
                                       "handle=%d from the dispatcher; known "
                                       "action handles are: %s"
                                       % (action_handle.id,
                                          str(known_action_handles)))

            bundle_results = ph._solver_manager.get_results(action_handle)

            for scenario_name, scenario_results in iteritems(bundle_results):
                scenario = ph._scenario_tree._scenario_map[scenario_name]
                scenario.set_solution(scenario_results)

            if ph._verbose:
                print("Successfully loaded solution for bundle="+bundle_name)

            num_results_so_far += 1

    else:

        if ph._verbose:
            print("Waiting for scenario results extraction")

        num_results_so_far = 0

        while (num_results_so_far < len(ph._scenario_tree._scenarios)):

            action_handle = ph._solver_manager.wait_any()
            try:
                scenario_name = action_handle_scenario_map[action_handle]
            except KeyError:
                if action_handle in ph._queued_solve_action_handles:
                    ph._queued_solve_action_handles.discard(action_handle)
                    print("WARNING: Discarding uncollected solve action handle "
                          "with id=%d encountered during scenario results collection"
                          % (action_handle.id))
                    continue
                else:
                    known_action_handles = \
                        sorted((ah.id for ah in action_handle_scenario_map))
                    raise RuntimeError("PH client received an unknown action "
                                       "handle=%d from the dispatcher; known "
                                       "action handles are: %s"
                                       % (action_handle.id,
                                          str(known_action_handles)))

            scenario_results = ph._solver_manager.get_results(action_handle)
            scenario = ph._scenario_tree._scenario_map[scenario_name]
            scenario.set_solution(scenario_results)

            if ph._verbose:
                print("Successfully loaded solution for scenario="+scenario_name)

            num_results_so_far += 1

    end_time = time.time()

    if ph._output_times:
        print("Results collection time=%.2f seconds" % (end_time - start_time))

#
# Sends a mapping between (name,index) and ScenarioTreeID so that
# phsolverservers are aware of the master nodes's ScenarioTreeID
# labeling.
#

def warmstart_scenario_instances(ph):

    start_time = time.time()

    if ph._verbose:
        print("Collecting warmstart from PH solver servers")

    scenario_action_handle_map = {} # maps scenario names to action handles
    action_handle_scenario_map = {} # maps action handles to scenario names

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:

            for scenario_name in bundle._scenario_names:

                new_action_handle =  ph._solver_manager.queue(
                    action="collect_warmstart",
                    queue_name=ph._phpyro_job_worker_map[bundle._name],
                    name=bundle._name,
                    scenario_name=scenario_name)

                scenario_action_handle_map[scenario_name] = new_action_handle
                action_handle_scenario_map[new_action_handle] = scenario_name

    else:

        for scenario in ph._scenario_tree._scenarios:

            new_action_handle = ph._solver_manager.queue(
                action="collect_warmstart",
                queue_name=ph._phpyro_job_worker_map[scenario._name],
                name=scenario._name,
                scenario_name=scenario._name)

            scenario_action_handle_map[scenario._name] = new_action_handle
            action_handle_scenario_map[new_action_handle] = scenario._name
    ph._solver_manager.end_bulk()

    if ph._verbose:
        print("Waiting for warmstart results")

    num_results_so_far = 0

    while (num_results_so_far < len(ph._scenario_tree._scenarios)):

        action_handle = ph._solver_manager.wait_any()
        try:
            scenario_name = action_handle_scenario_map[action_handle]
        except KeyError:
            if action_handle in ph._queued_solve_action_handles:
                ph._queued_solve_action_handles.discard(action_handle)
                print("WARNING: Discarding uncollected solve action handle "
                      "with id=%d encountered during scenario results collection"
                      % (action_handle.id))
                continue
            else:
                known_action_handles = \
                    sorted((ah.id for ah in action_handle_scenario_map))
                raise RuntimeError("PH client received an unknown action "
                                   "handle=%d from the dispatcher; known "
                                   "action handles are: %s"
                                   % (action_handle.id,
                                      str(known_action_handles)))

        scenario_results = ph._solver_manager.get_results(action_handle)
        scenario = ph._scenario_tree._scenario_map[scenario_name]
        var_sm_bySymbol = scenario._instance._PHInstanceSymbolMaps[Var].bySymbol
        for symbol, val in iteritems(scenario_results):
            var_sm_bySymbol[symbol].value = val

        if ph._verbose:
            print("Successfully loaded warmstart for scenario="+scenario_name)

        num_results_so_far += 1

    end_time = time.time()

    if ph._output_times:
        print("Warmstart collection time=%.2f seconds" % (end_time - start_time))

#
# Sends a mapping between (name,index) and ScenarioTreeID so that
# phsolverservers are aware of the master nodes's ScenarioTreeID
# labeling.
#

def transmit_scenario_tree_ids(ph):

    start_time = time.time()

    if ph._verbose:
        print("Transmitting ScenarioTree variable ids to PH solver servers")

    action_handles = []

    generate_responses = ph._handshake_with_phpyro

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:

            ids_to_transmit = {}
            for stage in bundle._scenario_tree._stages:
                for bundle_tree_node in stage._tree_nodes:
                    # The bundle scenariotree usually isn't populated
                    # with variable value data so we need to reference
                    # the original scenariotree node
                    primary_tree_node = \
                        ph._scenario_tree._tree_node_map[bundle_tree_node._name]
                    ids_to_transmit[primary_tree_node._name] = \
                        primary_tree_node._variable_ids

            action_handles.append( ph._solver_manager.queue(
                action="update_scenario_tree_ids",
                queue_name=ph._phpyro_job_worker_map[bundle._name],
                generateResponse=generate_responses,
                name=bundle._name,
                new_ids=ids_to_transmit) )

    else:

        for scenario in ph._scenario_tree._scenarios:

            ids_to_transmit = {}
            for tree_node in scenario._node_list:
                ids_to_transmit[tree_node._name] = tree_node._variable_ids

            action_handles.append( ph._solver_manager.queue(
                action="update_scenario_tree_ids",
                queue_name=ph._phpyro_job_worker_map[scenario._name],
                generateResponse=generate_responses,
                name=scenario._name,
                new_ids=ids_to_transmit) )
    ph._solver_manager.end_bulk()

    if generate_responses:
        ph._solver_manager.wait_all(action_handles)

    end_time = time.time()

    if ph._output_times:
        print("ScenarioTree variable ids transmission time=%.2f "
              "seconds" % (end_time - start_time))

def transmit_weights(ph):

    start_time = time.time()

    if ph._verbose:
        print("Transmitting instance weights to PH solver servers")

    action_handles = []

    generate_responses = ph._handshake_with_phpyro

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:

            # map from scenario name to the corresponding weight map
            weights_to_transmit = {}

            for scenario in bundle._scenario_tree._scenarios:
                scenario_name = scenario._name

                # Skip the leaf nodes (scenario._w usually doesn't
                # store a value for variables on the leaf node)
                weights_to_transmit[scenario._name] = \
                    ph._scenario_tree._scenario_map[scenario_name]._w

            action_handles.append( ph._solver_manager.queue(
                action="load_weights",
                queue_name=ph._phpyro_job_worker_map[bundle._name],
                generateResponse=generate_responses,
                name=bundle._name,
                new_weights=weights_to_transmit) )

    else:

        for scenario in ph._scenario_tree._scenarios:

            # Skip the leaf nodes (scenario._w usually doesn't store a value
            # for variables on the leaf node)
            action_handles.append( ph._solver_manager.queue(
                action="load_weights",
                queue_name=ph._phpyro_job_worker_map[scenario._name],
                generateResponse=generate_responses,
                name=scenario._name,
                new_weights=scenario._w) )
    ph._solver_manager.end_bulk()

    if generate_responses:
        ph._solver_manager.wait_all(action_handles)

    end_time = time.time()

    if ph._output_times:
        print("Weight transmission time=%.2f seconds" % (end_time - start_time))

#
# a utility to transmit - across the PH solver manager - the
# the current xbar values for each non-leaf scenario tree node
#

def transmit_xbars(ph):

    start_time = time.time()

    if ph._verbose:
        print("Transmitting instance xbars to PH solver servers")

    action_handles = []

    generate_responses = ph._handshake_with_phpyro

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:

            xbars_to_transmit = {}
            # Skip the leaf nodes
            for stage in bundle._scenario_tree._stages[:-1]:
                for bundle_tree_node in stage._tree_nodes:
                    # The bundle scenariotree usually isn't populated
                    # with variable value data so we need to reference
                    # the original scenariotree node
                    primary_tree_node = \
                        ph._scenario_tree._tree_node_map[bundle_tree_node._name]
                    xbars_to_transmit[primary_tree_node._name] = \
                        primary_tree_node._xbars

            action_handles.append( ph._solver_manager.queue(
                action="load_xbars",
                queue_name=ph._phpyro_job_worker_map[bundle._name],
                generateResponse=generate_responses,
                name=bundle._name,
                new_xbars=xbars_to_transmit) )

    else:

        for scenario in ph._scenario_tree._scenarios:

            # Skip the leaf nodes
            xbars_to_transmit = {}
            xbars_to_transmit = dict((tree_node._name, tree_node._xbars) \
                                     for tree_node in scenario._node_list[:-1])

            action_handles.append( ph._solver_manager.queue(
                action="load_xbars",
                queue_name=ph._phpyro_job_worker_map[scenario._name],
                generateResponse=generate_responses,
                name=scenario._name,
                new_xbars=xbars_to_transmit) )
    ph._solver_manager.end_bulk()

    if generate_responses:
        ph._solver_manager.wait_all(action_handles)

    end_time = time.time()

    if ph._output_times:
        print("Xbar transmission time=%.2f seconds" % (end_time - start_time))


def _transmit_init(ph, worker_name, object_name):

    # both the dispatcher queue for initialization and the action name
    # are "initialize" - might be confusing, but hopefully not so
    # much.

    ah = ph._solver_manager.queue(
        action="initialize",
        queue_name=worker_name,
        name=worker_name,
        model_location=ph._scenario_tree._scenario_instance_factory._model_filename,
        data_location=ph._scenario_tree._scenario_instance_factory._scenario_tree_filename,
        objective_sense=ph._objective_sense_option,
        object_name=object_name,
        solver_type=ph._solver_type,
        solver_io=ph._solver_io,
        scenario_bundle_specification=ph._scenario_bundle_specification,
        create_random_bundles=ph._create_random_bundles,
        scenario_tree_random_seed=ph._scenario_tree_random_seed,
        default_rho=ph._rho,
        linearize_nonbinary_penalty_terms=\
        ph._linearize_nonbinary_penalty_terms,
        retain_quadratic_binary_terms=ph._retain_quadratic_binary_terms,
        breakpoint_strategy=ph._breakpoint_strategy,
        integer_tolerance=ph._integer_tolerance,
        output_solver_results=ph._output_solver_results,
        verbose=ph._verbose,
        compile_scenario_instances=ph._options.compile_scenario_instances)

    return ah

def release_phsolverservers(ph):

    if ph._verbose:
        print("Revoking PHPyroWorker job assignments")

    ph._solver_manager.begin_bulk()
    for job, worker in iteritems(ph._phpyro_job_worker_map):
        ph._solver_manager.queue(action="release",
                                 queue_name=ph._phpyro_job_worker_map[job],
                                 name=worker,
                                 object_name=job,
                                 generateResponse=False)
    ph._solver_manager.end_bulk()

    ph._phpyro_worker_jobs_map = {}
    ph._phpyro_job_worker_map = {}

def initialize_ph_solver_servers(ph):

    start_time = time.time()

    if ph._verbose:
        print("Transmitting initialization information to PH solver servers")

    if len(ph._solver_manager.server_pool) == 0:
        raise RuntimeError("No PHSolverServer processes have been acquired!")

    if ph._scenario_tree.contains_bundles():
        worker_jobs = [bundle._name for bundle in ph._scenario_tree._scenario_bundles]
    else:
        worker_jobs = [scenario._name for scenario in ph._scenario_tree._scenarios]

    action_handles = []
    ph._phpyro_worker_jobs_map = {}
    ph._phpyro_job_worker_map = {}
    for worker_name in itertools.cycle(ph._solver_manager.server_pool):
        if len(worker_jobs) == 0:
            break
        job_name = worker_jobs.pop()
        action_handles.append(_transmit_init(ph, worker_name, job_name))
        ph._phpyro_worker_jobs_map.setdefault(worker_name,[]).append(job_name)
        ph._phpyro_job_worker_map[job_name] = worker_name
    end_time = time.time()

    if ph._output_times:
        print("Initialization transmission time=%.2f seconds"
              % (end_time - start_time))

    #ph._solver_manager.wait_all(action_handles)
    return action_handles

#
# a utility to transmit to each PH solver server the current rho
# values for each problem instance.
#

def transmit_rhos(ph):

    start_time = time.time()

    if ph._verbose:
        print("Transmitting instance rhos to PH solver servers")

    action_handles = []

    generate_responses = ph._handshake_with_phpyro

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:

            # map from scenario name to the corresponding rho map
            rhos_to_transmit = {}

            for scenario in bundle._scenario_tree._scenarios:
                # Skip the leaf nodes (scenario._rho usually doesn't
                # store a value for variables on the leaf node)
                rhos_to_transmit[scenario._name] = \
                    ph._scenario_tree._scenario_map[scenario._name]._rho

            action_handles.append( ph._solver_manager.queue(
                action="load_rhos",
                queue_name=ph._phpyro_job_worker_map[bundle._name],
                name=bundle._name,
                generateResponse=generate_responses,
                new_rhos=rhos_to_transmit) )

    else:

        for scenario in ph._scenario_tree._scenarios:

            # Skip the leaf nodes (scenario._rho usually doesn't store
            # a value for variables on the leaf node)
            action_handles.append( ph._solver_manager.queue(
                action="load_rhos",
                queue_name=ph._phpyro_job_worker_map[scenario._name],
                name=scenario._name,
                generateResponse=generate_responses,
                new_rhos=scenario._rho) )
    ph._solver_manager.end_bulk()

    if generate_responses:
        ph._solver_manager.wait_all(action_handles)

    end_time = time.time()

    if ph._output_times:
        print("Rho transmission time=%.2f seconds" % (end_time - start_time))

#
# a utility to transmit - across the PH solver manager - the current
# scenario tree node statistics to each of my problem instances. done
# prior to each PH iteration k.
#

def transmit_tree_node_statistics(ph):

    start_time = time.time()

    if ph._verbose:
        print("Transmitting tree node statistics to PH solver servers")

    action_handles = []

    generate_responses = ph._handshake_with_phpyro

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:

            tree_node_minimums = {}
            tree_node_maximums = {}
            # iterate over the tree nodes in the bundle scenario tree - but
            # there aren't any statistics there - be careful!
            # TBD - we need to form these statistics! right now, they are
            #       beyond the bundle.
            # We ignore the leaf nodes
            for stage in bundle._scenario_tree._stages[:-1]:
                for bundle_tree_node in stage._tree_nodes:
                    primary_tree_node = \
                        ph._scenario_tree._tree_node_map[bundle_tree_node._name]
                    tree_node_minimums[primary_tree_node._name] = \
                        primary_tree_node._minimums
                    tree_node_maximums[primary_tree_node._name] = \
                        primary_tree_node._maximums

            action_handles.append( ph._solver_manager.queue(
                action="load_tree_node_stats",
                queue_name=ph._phpyro_job_worker_map[bundle._name],
                name=bundle._name,
                generateResponse=generate_responses,
                new_mins=tree_node_minimums,
                new_maxs=tree_node_maximums) )

    else:

        for scenario in ph._scenario_tree._scenarios:

            tree_node_minimums = {}
            tree_node_maximums = {}

            # Skip the leaf nodes
            for tree_node in scenario._node_list[:-1]:
                tree_node_minimums[tree_node._name] = tree_node._minimums
                tree_node_maximums[tree_node._name] = tree_node._maximums

            action_handles.append( ph._solver_manager.queue(
                action="load_tree_node_stats",
                queue_name=ph._phpyro_job_worker_map[scenario._name],
                name=scenario._name,
                generateResponse=generate_responses,
                new_mins=tree_node_minimums,
                new_maxs=tree_node_maximums) )
    ph._solver_manager.end_bulk()

    if generate_responses:
        ph._solver_manager.wait_all(action_handles)

    end_time = time.time()

    if ph._output_times:
        print("Tree node statistics transmission time="
              "%.2f seconds" % (end_time - start_time))

#
# a utility to activate - across the PH solver manager - weighted
# penalty objective terms.
#

def activate_ph_objective_weight_terms(ph):

    if ph._verbose:
        print("Transmitting request to activate PH objective "
              "weight terms to PH solver servers")

    action_handles = []

    generate_responses = ph._handshake_with_phpyro

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:
            action_handles.append( ph._solver_manager.queue(
                action="activate_ph_objective_weight_terms",
                queue_name=ph._phpyro_job_worker_map[bundle._name],
                generateResponse=generate_responses,
                name=bundle._name) )

    else:

        for scenario in ph._scenario_tree._scenarios:
            action_handles.append( ph._solver_manager.queue(
                action="activate_ph_objective_weight_terms",
                queue_name=ph._phpyro_job_worker_map[scenario._name],
                generateResponse=generate_responses,
                name=scenario._name) )
    ph._solver_manager.end_bulk()

    if generate_responses:
        ph._solver_manager.wait_all(action_handles)

#
# a utility to deactivate - across the PH solver manager - weighted
# penalty objective terms.
#

def deactivate_ph_objective_weight_terms(ph):

    if ph._verbose:
        print("Transmitting request to deactivate PH objective "
              "weight terms to PH solver servers")

    action_handles = []

    generate_responses = ph._handshake_with_phpyro

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:
            action_handles.append( ph._solver_manager.queue(
                action="deactivate_ph_objective_weight_terms",
                queue_name=ph._phpyro_job_worker_map[bundle._name],
                generateResponse=generate_responses,
                name=bundle._name) )

    else:

        for scenario in ph._scenario_tree._scenarios:
            action_handles.append( ph._solver_manager.queue(
                action="deactivate_ph_objective_weight_terms",
                queue_name=ph._phpyro_job_worker_map[scenario._name],
                generateResponse=generate_responses,
                name=scenario._name) )
    ph._solver_manager.end_bulk()

    if generate_responses:
        ph._solver_manager.wait_all(action_handles)


#
# a utility to activate - across the PH solver manager - proximal
# penalty objective terms.
#

def activate_ph_objective_proximal_terms(ph):

    if ph._verbose:
        print("Transmitting request to activate PH objective "
              "proximal terms to PH solver servers")

    action_handles = []

    generate_responses = ph._handshake_with_phpyro

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:
            action_handles.append( ph._solver_manager.queue(
                action="activate_ph_objective_proximal_terms",
                queue_name=ph._phpyro_job_worker_map[bundle._name],
                generateResponse=generate_responses,
                name=bundle._name) )

    else:

        for scenario in ph._scenario_tree._scenarios:
            action_handles.append( ph._solver_manager.queue(
                action="activate_ph_objective_proximal_terms",
                queue_name=ph._phpyro_job_worker_map[scenario._name],
                generateResponse=generate_responses,
                name=scenario._name) )
    ph._solver_manager.end_bulk()

    if generate_responses:
        ph._solver_manager.wait_all(action_handles)

#
# a utility to deactivate - across the PH solver manager - proximal
# penalty objective terms.
#

def deactivate_ph_objective_proximal_terms(ph):

    if ph._verbose:
        print("Transmitting request to deactivate PH objective "
              "proximal terms to PH solver servers")

    action_handles = []

    generate_responses = ph._handshake_with_phpyro

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:
            action_handles.append( ph._solver_manager.queue(
                action="deactivate_ph_objective_proximal_terms",
                queue_name=ph._phpyro_job_worker_map[bundle._name],
                generateResponse=generate_responses,
                name=bundle._name) )

    else:

        for scenario in ph._scenario_tree._scenarios:
            action_handles.append( ph._solver_manager.queue(
                action="deactivate_ph_objective_proximal_terms",
                queue_name=ph._phpyro_job_worker_map[scenario._name],
                generateResponse=generate_responses,
                name=scenario._name) )
    ph._solver_manager.end_bulk()

    if generate_responses:
        ph._solver_manager.wait_all(action_handles)


def transmit_fixed_variables(ph):

    start_time = time.time()

    if ph._verbose:
        print("Synchronizing fixed variable status with PH solver servers")

    action_handles = []

    generate_responses = ph._handshake_with_phpyro

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:

            transmit_variables = False
            for bundle_tree_node in bundle._scenario_tree._tree_nodes:
                    primary_tree_node = \
                        ph._scenario_tree._tree_node_map[bundle_tree_node._name]
                    if len(primary_tree_node._fix_queue):
                        transmit_variables = True
                        break

            if transmit_variables:
                # map from node name to the corresponding list of
                # fixed variables
                fixed_variables_to_transmit = {}

                # Just send the entire state of fixed variables
                # on each node (including leaf nodes)
                for bundle_tree_node in bundle._scenario_tree._tree_nodes:
                    primary_tree_node = \
                        ph._scenario_tree._tree_node_map[bundle_tree_node._name]
                    fixed_variables_to_transmit[primary_tree_node._name] = \
                        primary_tree_node._fix_queue

                action_handles.append( ph._solver_manager.queue(
                    action="update_fixed_variables",
                    queue_name=ph._phpyro_job_worker_map[bundle._name],
                    name=bundle._name,
                    generateResponse=generate_responses,
                    fixed_variables=fixed_variables_to_transmit) )
            else:
                if ph._verbose:
                    print("No synchronization was needed for bundle %s"
                          % (bundle._name))

    else:

        for scenario in ph._scenario_tree._scenarios:

            transmit_variables = False
            for tree_node in scenario._node_list:
                if len(tree_node._fix_queue):
                    transmit_variables = True
                    break

            if transmit_variables:

                fixed_variables_to_transmit = \
                    dict((tree_node._name, tree_node._fix_queue)
                         for tree_node in scenario._node_list)

                action_handles.append( ph._solver_manager.queue(
                    action="update_fixed_variables",
                    queue_name=ph._phpyro_job_worker_map[scenario._name],
                    name=scenario._name,
                    generateResponse=generate_responses,
                    fixed_variables=fixed_variables_to_transmit) )
            else:
                if ph._verbose:
                    print("No synchronization was needed for scenario %s"
                          % (scenario._name))
    ph._solver_manager.end_bulk()

    if generate_responses:
        ph._solver_manager.wait_all(action_handles)

    end_time = time.time()

    if ph._output_times:
        print("Fixed variable synchronization time="
              "%.2f seconds" % (end_time - start_time))

def transmit_external_function_invocation_to_worker(
        ph,
        worker_name,
        module_name,
        function_name,
        invocation_type=InvocationType.SingleInvocation,
        return_action_handle=False,
        function_args=None,
        function_kwds=None):

    if ph._verbose:
        print("Transmitting external function invocation request to PH "
              "solver server with name %s" % worker_name)

    generate_response = ph._handshake_with_phpyro or return_action_handle

    if ph._scenario_tree.contains_bundles():
        if worker_name not in ph._scenario_tree._scenario_bundle_map:
            raise ValueError("PH solver server with name %s does not exist"
                             % (worker_name))
    else:
        if worker_name not in ph._scenario_tree._scenario_map:
            raise ValueError("PH solver server with name %s does not exist"
                             % (worker_name))

    action_handle = ph._solver_manager.queue(action="invoke_external_function",
                                             queue_name=ph._phpyro_job_worker_map[worker_name],
                                             name=worker_name,
                                             invocation_type=invocation_type.key,
                                             generateResponse=generate_response,
                                             module_name=module_name,
                                             function_name=function_name,
                                             function_kwds=function_kwds,
                                             function_args=function_args)

    if generate_response and (not return_action_handle):
        ph._solver_manager.wait_all([action_handle])

    return action_handle if (return_action_handle) else None

def transmit_external_function_invocation(
        ph,
        module_name,
        function_name,
        invocation_type=InvocationType.SingleInvocation,
        return_action_handles=False,
        function_args=None,
        function_kwds=None):

    start_time = time.time()

    if ph._verbose:
        print("Transmitting external function invocation request "
              "to PH solver servers")

    action_handles = []

    generate_responses = ph._handshake_with_phpyro or return_action_handles

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:

            action_handles.append(
                ph._solver_manager.queue(
                    action="invoke_external_function",
                    queue_name=ph._phpyro_job_worker_map[bundle._name],
                    name=bundle._name,
                    invocation_type=invocation_type.key,
                    generateResponse=generate_responses,
                    module_name=module_name,
                    function_name=function_name,
                    function_kwds=function_kwds,
                    function_args=function_args))

    else:

        for scenario in ph._scenario_tree._scenarios:

            action_handles.append(
                ph._solver_manager.queue(
                    action="invoke_external_function",
                    queue_name=ph._phpyro_job_worker_map[scenario._name],
                    name=scenario._name,
                    invocation_type=invocation_type.key,
                    generateResponse=generate_responses,
                    module_name=module_name,
                    function_name=function_name,
                    function_kwds=function_kwds,
                    function_args=function_args))
    ph._solver_manager.end_bulk()

    if generate_responses and (not return_action_handles):
        ph._solver_manager.wait_all(action_handles)

    end_time = time.time()

    if ph._output_times:
        print("External function invocation request transmission "
              "time=%.2f seconds" % (end_time - start_time))

    return action_handles if (return_action_handles) else None

#
# a utility to define model-level import suffixes - across the PH
# solver manager, on all instances.
#

def define_import_suffix(ph, suffix_name):

    if ph._verbose:
        print("Transmitting request to define suffix=%s to PH "
              "solver servers" % (suffix_name))

    action_handles = []

    generate_responses = ph._handshake_with_phpyro

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:
            action_handles.append( ph._solver_manager.queue(
                action="define_import_suffix",
                queue_name=ph._phpyro_job_worker_map[bundle._name],
                generateResponse=generate_responses,
                name=bundle._name,
                suffix_name = suffix_name))

    else:

        for scenario in ph._scenario_tree._scenarios:
            action_handles.append( ph._solver_manager.queue(
                action="define_import_suffix",
                queue_name=ph._phpyro_job_worker_map[scenario._name],
                generateResponse=generate_responses,
                name=scenario._name,
                suffix_name = suffix_name))
    ph._solver_manager.end_bulk()

    if generate_responses:
        ph._solver_manager.wait_all(action_handles)

#
# a utility to request that each PH solver server restore cached
# solutions to their scenario instances.
#

def restore_cached_scenario_solutions(ph, cache_id, release_cache):

    if ph._verbose:
        print("Transmitting request to restore cached scenario solutions "
              "to PH solver servers")

    action_handles = []

    generate_responses = ph._handshake_with_phpyro

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:
            action_handles.append( ph._solver_manager.queue(
                action="restore_cached_scenario_solutions",
                queue_name=ph._phpyro_job_worker_map[bundle._name],
                cache_id=cache_id,
                release_cache=release_cache,
                generateResponse=generate_responses,
                name=bundle._name) )

    else:

        for scenario in ph._scenario_tree._scenarios:

            action_handles.append( ph._solver_manager.queue(
                action="restore_cached_scenario_solutions",
                queue_name=ph._phpyro_job_worker_map[scenario._name],
                cache_id=cache_id,
                release_cache=release_cache,
                generateResponse=generate_responses,
                name=scenario._name) )
    ph._solver_manager.end_bulk()

    if generate_responses:
        ph._solver_manager.wait_all(action_handles)

#
# a utility to request that each PH solver server cache
# solutions to their scenario instances.
#

def cache_scenario_solutions(ph, cache_id):

    if ph._verbose:
        print("Transmitting request to cache scenario solutions "
              "to PH solver servers")

    action_handles = []

    generate_responses = ph._handshake_with_phpyro

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:
            action_handles.append( ph._solver_manager.queue(
                action="cache_scenario_solutions",
                queue_name=ph._phpyro_job_worker_map[bundle._name],
                cache_id=cache_id,
                generateResponse=generate_responses,
                name=bundle._name) )

    else:

        for scenario in ph._scenario_tree._scenarios:
            action_handles.append( ph._solver_manager.queue(
                action="cache_scenario_solutions",
                queue_name=ph._phpyro_job_worker_map[scenario._name],
                cache_id=cache_id,
                generateResponse=generate_responses,
                name=scenario._name) )
    ph._solver_manager.end_bulk()

    if generate_responses:
        ph._solver_manager.wait_all(action_handles)

def gather_scenario_tree_data(ph, initialization_action_handles):

    start_time = time.time()

    if ph._verbose:
        print("Collecting scenario tree data from PH solver servers")

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
             for tree_node in ph._scenario_tree._tree_nodes)
    need_scenario_data = \
        dict((scenario._name,True) \
             for scenario in ph._scenario_tree._scenarios)

    ph._solver_manager.begin_bulk()
    if ph._scenario_tree.contains_bundles():

        for bundle in ph._scenario_tree._scenario_bundles:

            object_names = {}
            object_names['nodes'] = \
                [tree_node._name \
                 for scenario in bundle._scenario_tree._scenarios \
                 for tree_node in scenario._node_list \
                 if need_node_data[tree_node._name]]
            object_names['scenarios'] = \
                [scenario_name \
                 for scenario_name in bundle._scenario_names]

            new_action_handle =  ph._solver_manager.queue(
                action="collect_scenario_tree_data",
                queue_name=ph._phpyro_job_worker_map[bundle._name],
                name=bundle._name,
                tree_object_names=object_names)

            bundle_action_handle_map[bundle._name] = new_action_handle
            action_handle_bundle_map[new_action_handle] = bundle._name

            for node_name in object_names['nodes']:
                need_node_data[node_name] = False
            for scenario_name in object_names['scenarios']:
                need_scenario_data[scenario_name] = False

    else:

        for scenario in ph._scenario_tree._scenarios:

            object_names = {}
            object_names['nodes'] = \
                [tree_node._name for tree_node in scenario._node_list \
                 if need_node_data[tree_node._name]]
            object_names['scenarios'] = [scenario._name]

            new_action_handle = ph._solver_manager.queue(
                action="collect_scenario_tree_data",
                queue_name=ph._phpyro_job_worker_map[scenario._name],
                name=scenario._name,
                tree_object_names=object_names)

            scenario_action_handle_map[scenario._name] = new_action_handle
            action_handle_scenario_map[new_action_handle] = scenario._name

            for node_name in object_names['nodes']:
                need_node_data[node_name] = False
            for scenario_name in object_names['scenarios']:
                need_scenario_data[scenario_name] = False
    ph._solver_manager.end_bulk()

    assert all(not val for val in itervalues(need_node_data))
    assert all(not val for val in itervalues(need_scenario_data))

    have_node_data = \
        dict((tree_node._name,False) \
             for tree_node in ph._scenario_tree._tree_nodes)
    have_scenario_data = \
        dict((scenario._name,False) \
             for scenario in ph._scenario_tree._scenarios)

    if ph._verbose:
        print("Waiting for scenario tree data extraction")

    if ph._scenario_tree.contains_bundles():

        num_results_so_far = 0

        while (num_results_so_far < len(ph._scenario_tree._scenario_bundles)):

            action_handle = ph._solver_manager.wait_any()

            if action_handle in initialization_action_handles:
                initialization_action_handles.remove(action_handle)
                ph._solver_manager.get_results(action_handle)
                continue

            bundle_results = ph._solver_manager.get_results(action_handle)
            bundle_name = action_handle_bundle_map[action_handle]

            for tree_node_name, node_data in iteritems(bundle_results['nodes']):
                assert have_node_data[tree_node_name] == False
                have_node_data[tree_node_name] = True
                tree_node = ph._scenario_tree.get_node(tree_node_name)
                tree_node._variable_ids.update(node_data['_variable_ids'])
                tree_node._standard_variable_ids.update(node_data['_standard_variable_ids'])
                tree_node._variable_indices.update(node_data['_variable_indices'])
                tree_node._integer.update(node_data['_integer'])
                tree_node._binary.update(node_data['_binary'])
                tree_node._semicontinuous.update(node_data['_semicontinuous'])
                # these are implied
                tree_node._derived_variable_ids = \
                    set(tree_node._variable_ids)-tree_node._standard_variable_ids
                tree_node._name_index_to_id = \
                    dict((val,key) for key,val in iteritems(tree_node._variable_ids))

            for scenario_name, scenario_data in \
                  iteritems(bundle_results['scenarios']):
                assert have_scenario_data[scenario_name] == False
                have_scenario_data[scenario_name] = True
                scenario = ph._scenario_tree.get_scenario(scenario_name)
                scenario._objective_name = scenario_data['_objective_name']
                scenario._objective_sense = scenario_data['_objective_sense']
                # rhos may have been modified with rhosetter callback
                scenario._rho.update(scenario_data['_rho'])
                # initialize _w, _rho, and _x, keys after this loop

            if ph._verbose:
                print("Successfully loaded scenario tree data "
                      "for bundle="+bundle_name)

            num_results_so_far += 1

    else:

        num_results_so_far = 0

        while (num_results_so_far < len(ph._scenario_tree._scenarios)):

            action_handle = ph._solver_manager.wait_any()

            if action_handle in initialization_action_handles:
                initialization_action_handles.remove(action_handle)
                ph._solver_manager.get_results(action_handle)
                continue

            scenario_results = ph._solver_manager.get_results(action_handle)
            scenario_name = action_handle_scenario_map[action_handle]
            for tree_node_name, node_data in iteritems(scenario_results['nodes']):
                assert have_node_data[tree_node_name] == False
                have_node_data[tree_node_name] = True
                tree_node = ph._scenario_tree.get_node(tree_node_name)
                tree_node._variable_ids.update(node_data['_variable_ids'])
                tree_node._standard_variable_ids.update(node_data['_standard_variable_ids'])
                tree_node._variable_indices.update(node_data['_variable_indices'])
                tree_node._integer.update(node_data['_integer'])
                tree_node._binary.update(node_data['_binary'])
                tree_node._semicontinuous.update(node_data['_semicontinuous'])
                # these are implied
                tree_node._derived_variable_ids = \
                    set(tree_node._variable_ids)-tree_node._standard_variable_ids
                tree_node._name_index_to_id = \
                    dict((val,key) for key,val in iteritems(tree_node._variable_ids))

            for scenario_name, scenario_data in \
                  iteritems(scenario_results['scenarios']):
                assert have_scenario_data[scenario_name] == False
                have_scenario_data[scenario_name] = True
                scenario = ph._scenario_tree.get_scenario(scenario_name)
                scenario._objective_name = scenario_data['_objective_name']
                scenario._objective_sense = scenario_data['_objective_sense']
                # rhos may have been modified with rhosetter callback
                scenario._rho.update(scenario_data['_rho'])
                # initialize _w and _x keys after this loop

            if ph._verbose:
                print("Successfully loaded scenario tree data for "
                      "scenario="+scenario_name)

            num_results_so_far += 1

    assert all(have_node_data)
    assert all(have_scenario_data)

    for tree_node in ph._scenario_tree._tree_nodes:
        tree_node._minimums = dict.fromkeys(tree_node._variable_ids,0)
        tree_node._maximums = dict.fromkeys(tree_node._variable_ids,0)
        # this is the true variable average at the node (unmodified)
        tree_node._averages = dict.fromkeys(tree_node._variable_ids,0)
        # this is the xbar used in the PH objective.
        tree_node._xbars = dict.fromkeys(tree_node._standard_variable_ids,0.0)
        # this is the blend used in the PH objective
        tree_node._blend = dict.fromkeys(tree_node._standard_variable_ids,1)
        # For the dual ph algorithm
        tree_node._wbars = dict.fromkeys(tree_node._standard_variable_ids,None)
        for scenario in tree_node._scenarios:
            scenario._x[tree_node._name] = \
                dict.fromkeys(tree_node._variable_ids,None)
            if not tree_node.is_leaf_node():
                scenario._w[tree_node._name] = \
                    dict.fromkeys(tree_node._standard_variable_ids,0.0)

    if len(initialization_action_handles):
        if ph._verbose:
            print("Waiting on remaining PHSolverServer initializations")
        ph._solver_manager.wait_all(initialization_action_handles)
        while len(initialization_action_handles):
            initialization_action_handles.pop()

    end_time = time.time()

    if ph._output_times:
        print("Scenario tree data collection time=%.2f seconds"
              % (end_time - start_time))
