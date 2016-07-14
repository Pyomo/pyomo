#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("ScenarioTreeManagerSolverWorkerPyro",)

import time

from pyomo.opt import SolverFactory
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import (PySPConfigBlock,
                                    safe_declare_common_option)
from pyomo.pysp.scenariotree.manager_worker_pyro import \
    ScenarioTreeManagerWorkerPyro
from pyomo.pysp.scenariotree.manager_solver import \
    (_ScenarioTreeManagerSolverWorker,
     ScenarioTreeManagerSolver)
from six import iteritems

#
# A full implementation of the ScenarioTreeManagerSolver and
# ScenarioTreeManager interfaces designed to be used by Pyro-based
# client-side ScenarioTreeManagerSolver implementations.
#

class ScenarioTreeManagerSolverWorkerPyro(ScenarioTreeManagerWorkerPyro,
                                          _ScenarioTreeManagerSolverWorker,
                                          ScenarioTreeManagerSolver,
                                          PySPConfiguredObject):

    _declared_options = \
        PySPConfigBlock("Options declared for the "
                        "ScenarioTreeManagerSolverWorkerPyro class")

    def __init__(self, *args, **kwds):

        super(ScenarioTreeManagerSolverWorkerPyro, self).\
            __init__(*args, **kwds)
        # Maps ScenarioTree variable IDs on the client-side to
        # ScenarioTree variable IDs on this worker (by node name)
        self._master_scenario_tree_id_map = {}
        self._reverse_master_scenario_tree_id_map = {}

    #
    # Abstract methods for ScenarioTreeManager:
    #

    # override what is implemented by ScenarioTreeSolverWorkerPyro
    def _init(self, *args, **kwds):
        super(ScenarioTreeManagerSolverWorkerPyro, self).\
            _init(*args, **kwds)
        super(ScenarioTreeManagerSolverWorkerPyro, self).\
            _init_solver_worker()

    # Update the map from local to master scenario tree ids
    def _update_master_scenario_tree_ids_for_client(self,
                                                    object_name,
                                                    new_ids):

        if self.get_option("verbose"):
            if self._scenario_tree.contains_bundles():
                print("Received request to update master "
                      "scenario tree ids for bundle="+object_name)
            else:
                print("Received request to update master "
                      "scenario tree ids scenario="+object_name)

        for node_name, new_master_node_ids in iteritems(new_ids):
            tree_node = self._scenario_tree.get_node(node_name)
            name_index_to_id = tree_node._name_index_to_id

            self._master_scenario_tree_id_map[tree_node.name] = \
                dict((master_variable_id, name_index_to_id[name_index])
                     for master_variable_id, name_index
                     in iteritems(new_master_node_ids))

            self._reverse_master_scenario_tree_id_map[tree_node.name] = \
                dict((local_variable_id, master_variable_id)
                     for master_variable_id, local_variable_id
                     in iteritems(self._master_scenario_tree_id_map\
                                  [tree_node.name]))

    def _collect_scenario_tree_data_for_client(self, tree_object_names):

        data = {}
        node_data = data['nodes'] = {}
        for node_name in tree_object_names['nodes']:
            tree_node = self._scenario_tree.get_node(node_name)
            this_node_data = node_data[node_name] = {}
            this_node_data['_variable_ids'] = tree_node._variable_ids
            this_node_data['_standard_variable_ids'] = \
                tree_node._standard_variable_ids
            this_node_data['_variable_indices'] = tree_node._variable_indices
            this_node_data['_integer'] = tuple(tree_node._integer)
            this_node_data['_binary'] = tuple(tree_node._binary)
            this_node_data['_semicontinuous'] = \
                tuple(tree_node._semicontinuous)
            # master will need to reconstruct
            # _derived_variable_ids
            # _name_index_to_id

        scenario_data = data['scenarios'] = {}
        for scenario_name in tree_object_names['scenarios']:
            scenario = self._scenario_tree.get_scenario(scenario_name)
            this_scenario_data = scenario_data[scenario_name] = {}
            this_scenario_data['_objective_name'] = scenario._objective_name
            this_scenario_data['_objective_sense'] = \
                scenario._objective_sense

        return data

    # TODO: functionality for returning suffixes
    def _solve_objects_for_client(self,
                                  object_type,
                                  objects,
                                  update_stages,
                                  ephemeral_solver_options,
                                  disable_warmstart):

        if self.get_option("verbose"):
            print("Received request to queue solves for %s" % (object_type))

        # suppress any verbose output within _solve_objects
        # as it will be repeated by the client
        self_verbose = self.get_option("verbose")
        self_output_times = self.get_option("output_times")
        setattr(self._options,
                self.get_full_option_name("verbose"),
                False)
        setattr(self._options,
                self.get_full_option_name("output_times"),
                False)
        manager_results = super(ScenarioTreeManagerSolverWorkerPyro, self).\
                          _solve_objects(object_type,
                                         objects,
                                         update_stages,
                                         ephemeral_solver_options,
                                         disable_warmstart,
                                         False, # check_status
                                         False) # async
        setattr(self._options,
                self.get_full_option_name("verbose"),
                self_verbose)
        setattr(self._options,
                self.get_full_option_name("output_times"),
                self_output_times)

        if object_type == 'bundles':
            if objects is None:
                objects = self._scenario_tree._scenario_bundle_map
        else:
            assert object_type == 'scenarios'
            if objects is None:
                objects = self._scenario_tree._scenario_map

        results = {}
        for object_name in objects:

            manager_object_results = \
                manager_results.results_for(object_name)
            # Convert enums to strings to avoid difficult
            # behavior related to certain Pyro serializer
            # settings
            manager_object_results['solver_status'] = \
                str(manager_object_results['solver_status'])
            manager_object_results['termination_condition'] = \
                str(manager_object_results['termination_condition'])
            manager_object_results['solution_status'] = \
                str(manager_object_results['solution_status'])

            if object_type == 'bundles':
                solution = {}
                for scenario_name in self._scenario_tree.\
                       get_bundle(object_name).scenario_names:
                    scenario = self._scenario_tree.get_scenario(
                        scenario_name)
                    solution[scenario_name] = \
                        scenario.copy_solution(
                            translate_ids=\
                            self._reverse_master_scenario_tree_id_map)
            else:
                scenario = self._scenario_tree.get_scenario(object_name)
                solution = scenario.copy_solution(
                    translate_ids=\
                    self._reverse_master_scenario_tree_id_map)

            results[object_name] = (manager_object_results, solution)

        return results

    def _update_fixed_variables_for_client(self, fixed_variables):

        print("Received request to update fixed statuses on "
              "scenario tree nodes")

        for node_name, node_fixed_vars in iteritems(fixed_variables):
            tree_node = self._scenario_tree.get_node(node_name)
            node_variable_id_map = \
                self._master_scenario_tree_id_map[node_name]
            tree_node._fix_queue.update(
                (node_variable_id_map[master_variable_id],
                 node_fixed_vars[master_variable_id])
                for master_variable_id in node_fixed_vars)

        self.push_fix_queue_to_instances()
