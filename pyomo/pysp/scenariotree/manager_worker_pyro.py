#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ("ScenarioTreeManagerWorkerPyro",)

import time

from pyomo.common.dependencies import dill, dill_available
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import (PySPConfigBlock,
                                    safe_declare_common_option)
from pyomo.pysp.scenariotree.manager \
    import (_ScenarioTreeManagerWorker,
            ScenarioTreeManager,
            InvocationType)

import six
from six import iteritems, string_types

#
# A full implementation of the ScenarioTreeManager interface
# designed to be used by Pyro-based ScenarioTreeManagerClient
# implementations.
#

class ScenarioTreeManagerWorkerPyro(_ScenarioTreeManagerWorker,
                                    ScenarioTreeManager,
                                    PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()

        #
        # scenario instance construction
        #
        safe_declare_common_option(options,
                                   "objective_sense_stage_based")
        safe_declare_common_option(options,
                                   "output_instance_construction_time")
        safe_declare_common_option(options,
                                   "compile_scenario_instances")

        #
        # various
        #
        safe_declare_common_option(options,
                                   "verbose")
        safe_declare_common_option(options,
                                   "profile_memory")

        return options

    @property
    def server(self):
        return self._server

    @property
    def modules_imported(self):
        assert self._server is not None
        return self._server._modules_imported

    @property
    def uncompressed_scenario_tree(self):
        assert self._server is not None
        return self._server._full_scenario_tree

    @property
    def MPI(self):
        assert self._server is not None
        return self._server.MPI

    def __init__(self,
                 server,
                 worker_name,
                 init_args,
                 *args,
                 **kwds):
        assert len(args) == 0
        options = self.register_options()
        for name, val in iteritems(kwds):
            options.get(name).set_value(val)
        super(ScenarioTreeManagerWorkerPyro, self).__init__(options)

        # The name of the scenario tree server owning this worker
        self._server = server
        # The name of this worker on this server
        self._worker_name = worker_name
        self.mpi_comm_tree = {}
        self.initialize(*init_args)

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

    def _update_fixed_variables_for_client(self, fixed_variables):

        if self.get_option("verbose"):
            print("Received request to update fixed statuses on "
                  "scenario tree nodes")

        for node_name, node_fixed_vars in iteritems(fixed_variables):
            tree_node = self._scenario_tree.get_node(node_name)
            tree_node._fix_queue.update(node_fixed_vars)

        self.push_fix_queue_to_instances()

    #
    # Abstract methods for ScenarioTreeManager:
    #

    def _init(self,
              init_type,
              init_names,
              init_data):
        # check to make sure no base class has implemented _init
        try:
            super(ScenarioTreeManagerWorkerPyro, self)._init()
        except NotImplementedError:
            pass
        else:
            assert False, "developer error"

        scenarios_to_construct = []
        if init_type == "scenarios":
            assert type(init_names) in (list, tuple)
            assert len(init_names) > 0
            assert init_data is None

            if self.get_option("verbose"):
                print("Initializing worker with name %s for scenarios: %s"
                      % (self._worker_name, str(init_names)))

            scenarios_to_construct.extend(init_names)

        elif init_type == "bundles":
            assert type(init_names) in (list, tuple)
            assert type(init_data) is dict
            assert len(init_names) > 0
            assert len(init_names) == len(init_data)

            if self.get_option("verbose"):
                print("Initializing worker with name %s for bundle list:"
                      % (self._worker_name))
                for bundle_name in init_names:
                    assert type(init_data[bundle_name]) in (list, tuple)
                    print("  - %s: %s" % (bundle_name, init_data[bundle_name]))

            for bundle_name in init_names:
                assert type(init_data[bundle_name]) in (list, tuple)
                scenarios_to_construct.extend(init_data[bundle_name])
        else:
            raise ValueError("Invalid worker init type: %s" % (init_type))

        # compress the scenario tree to reflect those instances for
        # which this ph solver server is responsible for constructing.
        self._scenario_tree = \
            self.uncompressed_scenario_tree.make_compressed(
                scenarios_to_construct,
                normalize=False)
        self._instances = \
            self.uncompressed_scenario_tree._scenario_instance_factory.\
            construct_instances_for_scenario_tree(
                self._scenario_tree,
                output_instance_construction_time=\
                   self.get_option("output_instance_construction_time"),
                profile_memory=self.get_option("profile_memory"),
                compile_scenario_instances=self.get_option("compile_scenario_instances"))

        # with the scenario instances now available, have the scenario
        # tree compute the variable match indices at each node.
        self._scenario_tree.linkInInstances(
            self._instances,
            objective_sense=self.get_option("objective_sense_stage_based"),
            create_variable_ids=True)

        self._objective_sense = \
            self._scenario_tree._scenarios[0]._objective_sense
        assert all(_s._objective_sense == self._objective_sense
                   for _s in self._scenario_tree._scenarios)

        #
        # Create bundle if needed
        #
        if init_type == "bundles":
            for bundle_name in init_names:
                assert not self._scenario_tree.contains_bundle(bundle_name)
                self._scenario_tree.add_bundle(bundle_name,
                                               init_data[bundle_name])
                self._init_bundle(bundle_name,
                                  init_data[bundle_name])
                assert self._scenario_tree.contains_bundle(bundle_name)

        # now generate the process communicators
        root_comm = self.server.mpi_comm_workers
        if self.MPI is None:
            assert root_comm is None
        else:
            assert root_comm is not None
            root_node = self._scenario_tree.findRootNode()
            self.mpi_comm_tree[root_node.name] = root_comm.Dup()
            # loop over all nodes except the root and leaf
            # nodes and create a communicator between all
            # processes that reference a node
            for stage in self.uncompressed_scenario_tree.stages[1:-1]:
                for node in stage.nodes:
                    if self._scenario_tree.contains_node(node.name):
                        self.mpi_comm_tree[node.name] = \
                            self.mpi_comm_tree[node.parent.name].\
                            Split(0)
                    elif node.parent.name in self.mpi_comm_tree:
                        self.mpi_comm_tree[node.parent.name].\
                            Split(self.MPI.UNDEFINED)

    # Override the implementation on _ScenarioTreeManagerWorker
    def _close_impl(self):
        super(ScenarioTreeManagerWorkerPyro, self)._close_impl()
        self._options.check_usage(error=False)
        for comm in self.mpi_comm_tree.values():
            comm.Free()

    def _invoke_function_impl(self,
                              function,
                              module_name=None,
                              invocation_type=InvocationType.Single,
                              function_args=(),
                              function_kwds=None):

        start_time = time.time()

        if self.get_option("verbose"):
            if module_name is not None:
                print("Received request to invoke function=%s "
                      "in module=%s" % (str(function), str(module_name)))
            else:
                print("Received request to invoke anonymous "
                      "function serialized using the dill module")

        # InvocationType is transmitted as (key, data) to
        # avoid issues with Pyro, so this function accepts a
        # tuple and converts back to InvocationType
        if type(invocation_type) is tuple:
            _invocation_type_key, _invocation_type_data = invocation_type
            assert isinstance(_invocation_type_key, string_types)
            invocation_type = getattr(InvocationType,
                                      _invocation_type_key)
            if _invocation_type_data is not None:
                invocation_type = invocation_type(_invocation_type_data)

        # here we assume that if the module_name is None,
        # then a function was serialized by the fill module
        # before being transmitted
        if module_name is None:
            assert dill_available
            function = dill.loads(function)

        result = self._invoke_function_by_worker(
            function,
            module_name=module_name,
            invocation_type=invocation_type,
            function_args=function_args,
            function_kwds=function_kwds)

        end_time = time.time()
        if self.get_option("output_times") or \
           self.get_option("verbose"):
            print("External function invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    def _invoke_method_impl(self,
                            method_name,
                            method_args=(),
                            method_kwds=None):

        start_time = time.time()

        if self.get_option("verbose"):
            print("Received request to invoke method="+method_name)

        if method_kwds is None:
            method_kwds = {}
        result = getattr(self, method_name)(*method_args, **method_kwds)

        end_time = time.time()
        if self.get_option("output_times") or \
           self.get_option("verbose"):
            print("Method invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    #
    # Override the invoke_function and invoke_method interface methods
    # on ScenarioTreeManager
    # ** NOTE **: These version are meant to be invoked locally.
    #             The client-side will always invoke the *_impl
    #             methods, which do not accept the async_call or
    #             oneway_call keywords. When invoked here, the
    #             async_call and oneway_call keywords behave like they
    #             do for the Serial solver manager (they are
    #             a dummy interface)
    #

    def invoke_function(self,
                        function,
                        module_name=None,
                        invocation_type=InvocationType.Single,
                        function_args=(),
                        function_kwds=None,
                        async_call=False,
                        oneway_call=False):
        """This function is an override of that on the
        ScenarioTreeManager interface. It should not be invoked by a
        client, but only locally (e.g., inside a local function
        invocation transmitted by the client).
        """
        if async_call and oneway_call:
            raise ValueError("async oneway calls do not make sense")
        invocation_type = _map_deprecated_invocation_type(invocation_type)

        if not isinstance(function, six.string_types):
            if module_name is not None:
                raise ValueError(
                    "The module_name keyword must be None "
                    "when the function argument is not a string.")
        else:
            if module_name is None:
                raise ValueError(
                    "A module name is required when "
                    "a function name is given")

        self._invoke_function_impl(function,
                                   module_name=module_name,
                                   invocation_type=invocation_type,
                                   function_args=function_args,
                                   function_kwds=function_kwds)

        if not oneway_call:
            if invocation_type == InvocationType.Single:
                result = {self._worker_name: result}
        if async_call:
            result = self.AsyncResult(None, result=result)

        return result

    def invoke_method(self,
                      method_name,
                      method_args=(),
                      method_kwds=None,
                      async_call=False,
                      oneway_call=False):
        """This function is an override of that on the
        ScenarioTreeManager interface. It should not be invoked by a
        client, but only locally (e.g., inside a local function
        invocation transmitted by the client).

        """
        if async_call and oneway_call:
            raise ValueError("async oneway calls do not make sense")

        if method_kwds is None:
            method_kwds = {}
        result = getattr(self, method_name)(*method_args, **method_kwds)

        if not oneway_call:
            result = {self._worker_name: result}
        if async_call:
            result = self.AsyncResult(None, result=result)

        return result

    #
    # Helper methods that can be invoked by the client
    #

    def assign_data(self, name, data):
        if self.get_option("verbose"):
            print("Received request to assign data to attribute name %s on "
                  "scenario tree worker %s" % (name, self._worker_name))
        setattr(self, name, data)

# register this worker with the pyro server
from pyomo.pysp.scenariotree.server_pyro import RegisterWorker
RegisterWorker('ScenarioTreeManagerWorkerPyro',
               ScenarioTreeManagerWorkerPyro)
