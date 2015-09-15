#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("ScenarioTreeSolverWorker",)

import time

from pyutilib.misc.config import ConfigBlock
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import safe_register_common_option
from pyomo.pysp.scenariotree.scenariotreeworkerbasic import \
    ScenarioTreeWorkerBasic
from pyomo.pysp.scenariotree.scenariotreesolvermanager import \
    (_ScenarioTreeSolverWorkerImpl,
     _ScenarioTreeSolverManager)
from six import iteritems

class ScenarioTreeSolverWorker(ScenarioTreeWorkerBasic,
                               _ScenarioTreeSolverWorkerImpl,
                               _ScenarioTreeSolverManager,
                               PySPConfiguredObject):

    _registered_options = \
        ConfigBlock("Options registered for the ScenarioTreeWorkerSolver class")

    def __init__(self, *args, **kwds):

        super(ScenarioTreeSolverWorker, self).__init__(*args, **kwds)

        # Maps ScenarioTreeID's on the master node ScenarioTree to
        # ScenarioTreeID's on this ScenarioTreeWorkers's ScenarioTree
        # (by node name)
        self._master_scenario_tree_id_map = {}
        self._reverse_master_scenario_tree_id_map = {}

    #
    # Update the map from local to master scenario tree ids
    #

    def update_master_scenario_tree_ids(self, object_name, new_ids):

        if self._options.verbose:
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

    def collect_scenario_tree_data(self, tree_object_names):

        data = {}
        node_data = data['nodes'] = {}
        for node_name in tree_object_names['nodes']:
            tree_node = self._scenario_tree.get_node(node_name)
            this_node_data = node_data[node_name] = {}
            this_node_data['_variable_ids'] = tree_node._variable_ids
            this_node_data['_standard_variable_ids'] = \
                tree_node._standard_variable_ids
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
