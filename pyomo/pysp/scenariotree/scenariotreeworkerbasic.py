#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("ScenarioTreeWorkerBasic",)

import time

from pyutilib.misc.config import ConfigBlock
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import safe_register_common_option
from pyomo.pysp.scenariotree.scenariotreeserverutils \
    import (WorkerInitType,
            InvocationType)
from pyomo.pysp.scenariotree.scenariotreemanager \
    import (_ScenarioTreeWorkerImpl,
            _ScenarioTreeManager)

from six import iteritems, string_types

class ScenarioTreeWorkerBasic(_ScenarioTreeWorkerImpl,
                              _ScenarioTreeManager,
                              PySPConfiguredObject):

    _registered_options = \
        ConfigBlock("Options registered for the ScenarioTreeWorker class")

    #
    # scenario instance construction
    #
    safe_register_common_option(_registered_options,
                                "objective_sense_stage_based")
    safe_register_common_option(_registered_options,
                                "compile_scenario_instances")
    safe_register_common_option(_registered_options,
                                "output_instance_construction_time")

    #
    # various
    #
    safe_register_common_option(_registered_options,
                                "verbose")
    safe_register_common_option(_registered_options,
                                "profile_memory")

    def __init__(self,
                 server_name,
                 full_scenario_tree,
                 worker_name,
                 worker_init,
                 *args,
                 **kwds):

        super(ScenarioTreeWorkerBasic, self).__init__(*args, **kwds)
        # pyutilib.Enum can not be serialized depending on the
        # serializer type used by Pyro, so we just send the
        # key name

        if self._options.verbose:
            print("Initializing ScenarioTreeWorkerBasic with options:")
            self.display_options()
            print("")

        # The name of the scenario tree server owning this worker
        self._server_name = server_name
        # So we have access to real scenario and bundle probabilities
        self._full_scenario_tree = full_scenario_tree
        self._worker_name = worker_name

        scenarios_to_construct = []
        if worker_init.type_ == WorkerInitType.Scenarios:
            assert type(worker_init.names) in (list, tuple)
            assert len(worker_init.names) > 0
            assert worker_init.data is None

            if self._options.verbose:
                print("Constructing worker with name %s for scenarios: %s"
                      % (worker_name, str(worker_init.names)))

            scenarios_to_construct.extend(worker_init.names)

        elif worker_init.type_ == WorkerInitType.Bundles:
            assert type(worker_init.names) in (list, tuple)
            assert type(worker_init.data) is dict
            assert len(worker_init.names) > 0

            if self._options.verbose:
                print("Constructing worker with name %s for bundle list:"
                      % (worker_name))
                for bundle_name in worker_init.names:
                    assert type(worker_init.data[bundle_name]) in (list, tuple)
                    print("  - %s: %s" % (bundle_name, worker_init.data[bundle_name]))

            for bundle_name in worker_init.names:
                assert type(worker_init.data[bundle_name]) in (list, tuple)
                scenarios_to_construct.extend(worker_init.data[bundle_name])

        # compress the scenario tree to reflect those instances for
        # which this ph solver server is responsible for constructing.
        self._scenario_tree = \
            self._full_scenario_tree.make_compressed(scenarios_to_construct,
                                                     normalize=False)

        self._instances = \
            self._full_scenario_tree._scenario_instance_factory.\
            construct_instances_for_scenario_tree(
                self._scenario_tree,
                output_instance_construction_time=\
                   self._options.output_instance_construction_time,
                profile_memory=self._options.profile_memory,
                compile_scenario_instances=self._options.compile_scenario_instances)

        # with the scenario instances now available, have the scenario
        # tree compute the variable match indices at each node.
        self._scenario_tree.linkInInstances(
            self._instances,
            objective_sense=self._options.objective_sense_stage_based,
            create_variable_ids=True)

        self._objective_sense = \
            self._scenario_tree._scenarios[0]._objective_sense
        assert all(_s._objective_sense == self._objective_sense
                   for _s in self._scenario_tree._scenarios)

        #
        # Create bundle if needed
        #
        if worker_init.type_ == WorkerInitType.Bundles:
            for bundle_name in worker_init.names:
                assert not self._scenario_tree.contains_bundle(bundle_name)
                self.add_bundle(bundle_name, worker_init.data[bundle_name])
                assert self._scenario_tree.contains_bundle(bundle_name)

    def assign_data(self, name, data):
        if self._options.verbose:
            print("Received request to assign data to attribute name %s on "
                  "scenario tree worker %s" % (name, self._worker_name))
        setattr(self, name, data)

    def _close_impl(self):
        _ScenarioTreeWorkerImpl._close_impl(self)
        ignored_options = dict((_c._name, _c.value(False))
                               for _c in self._options.unused_user_values())
        if len(ignored_options):
            print("")
            print("*** WARNING: The following options were explicitly "
                  "set but never accessed by worker %s: "
                  % (self._worker_name))
            for name in ignored_options:
                print(" - %s: %s" % (name, ignored_options[name]))
            print("*** If you believe this is a bug, please report it "
                  "to the PySP developers.")
            print("")

    #
    # Invoke the indicated function in the specified module.
    #

    def invoke_external_function(self,
                                 module_name,
                                 function_name,
                                 invocation_type=InvocationType.SingleInvocation,
                                 function_args=None,
                                 function_kwds=None):

        start_time = time.time()

        if self._options.verbose:
            print("Received request to invoke external function"
                  "="+function_name+" in module="+module_name)

        # pyutilib.Enum can not be serialized depending on the
        # serializer type used by Pyro, so we just send the
        # key name in that case
        if isinstance(invocation_type, string_types):
            invocation_type = getattr(InvocationType, invocation_type)

        result = self._invoke_external_function_impl(module_name,
                                                     function_name,
                                                     invocation_type=invocation_type,
                                                     function_args=function_args,
                                                     function_kwds=function_kwds)

        end_time = time.time()
        if self._options.output_times:
            print("External function invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result
