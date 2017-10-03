#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

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

class ScenarioTreeManagerSolverWorkerPyro(_ScenarioTreeManagerSolverWorker,
                                          ScenarioTreeManagerSolver,
                                          PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()
        return options

    def __init__(self,
                 server,
                 worker_name,
                 base_worker_name,
                 *args,
                 **kwds):
        assert len(args) == 0
        options = self.register_options()
        for name, val in iteritems(kwds):
            options.get(name).set_value(val)
        self._server = server
        self._worker_name = worker_name
        manager = self._server._worker_map[base_worker_name]
        super(ScenarioTreeManagerSolverWorkerPyro, self).\
            __init__(manager, options)

    #
    # Abstract methods for ScenarioTreeManager:
    #

    def _solve_objects_for_client(self,
                                  object_type,
                                  objects,
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
                objects = self.manager.scenario_tree._scenario_bundle_map
        else:
            assert object_type == 'scenarios'
            if objects is None:
                objects = self.manager.scenario_tree._scenario_map

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
            results[object_name] = manager_object_results

        return results

    def _update_fixed_variables_for_client(self, fixed_variables):

        print("Received request to update fixed statuses on "
              "scenario tree nodes")

        for node_name, node_fixed_vars in iteritems(fixed_variables):
            tree_node = self._scenario_tree.get_node(node_name)
            tree_node._fix_queue.update(node_fixed_vars)

        self.push_fix_queue_to_instances()

# register this worker with the pyro server
from pyomo.pysp.scenariotree.server_pyro import RegisterWorker
RegisterWorker('ScenarioTreeManagerSolverWorkerPyro',
               ScenarioTreeManagerSolverWorkerPyro)
