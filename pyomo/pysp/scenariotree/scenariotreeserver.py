#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("ScenarioTreeServer", "RegisterScenarioTreeWorker")

import os
import sys
import copy
import argparse
import logging
import traceback

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
from pyomo.pysp.scenariotree.scenariotreeserverutils \
    import (SPPyroScenarioTreeServer_ProcessTaskError,
            WorkerInitType)
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import safe_register_common_option
from pyomo.pysp.util.misc import launch_command

from six import iteritems

logger = logging.getLogger('pyomo.pysp')

class SPPyroScenarioTreeServer(MultiTaskWorker, PySPConfiguredObject):

    # Maps name to a registered worker class to instantiate
    _registered_workers = {}

    _registered_options = \
        ConfigBlock("Options registered for the SPPyroScenarioTreeServer class")

    #
    # scenario instance construction
    #
    safe_register_common_option(_registered_options,
                                "model_location")
    safe_register_common_option(_registered_options,
                                "scenario_tree_location")

    #
    # scenario tree generation
    #
    safe_register_common_option(_registered_options,
                                "scenario_tree_random_seed")
    safe_register_common_option(_registered_options,
                                "scenario_tree_downsample_fraction")

    #
    # various
    #
    safe_register_common_option(_registered_options,
                                "verbose")

    @classmethod
    def get_registered_worker_type(cls, name):
        if name in cls._registered_workers:
            return cls._registered_workers[name]
        raise KeyError("No worker type has been registered under the name "
                       "'%s' for SPPyroScenarioTreeServer" % (name))

    def __init__(self, *args, **kwds):

        # add for purposes of diagnostic output.
        kwds["caller_name"] = "SPPyroScenarioTreeServer"
        MultiTaskWorker.__init__(self, *args, **kwds)
        # This classes options get updated during the "setup" phase
        options = self.register_options()
        PySPConfiguredObject.__init__(self, options)

        self._global_verbose = kwds.get('verbose', False)

        # Requests for employement when this worker is idle
        self._idle_queue_blocking_timeout = (True, 5)

        # Requests for new jobs when this worker is acquired but owns
        # no jobs
        self._worker_queue_blocking_timeout = (True, 0.1)

        # Requests for new jobs when this worker owns at least one
        # other job
        self._assigned_worker_queue_blocking_timeout = (False, None)
        # Requests for new tasks specific to current job(s)
        self._solver_queue_blocking_timeout = (True, 0.1)

        #
        # These will be used by all subsequent workers created
        # by this server. Their creation can eat up a nontrivial
        # amount of initialization time when a large number of
        # workers are created on this server, so we only create
        # them once.
        #
        self._scenario_instance_factory = None
        self._full_scenario_tree = None

        self._init()

    def _init(self):

        self.clear_request_types()
        # queue type, blocking, timeout
        self.push_request_type('sppyro_server_idle',
                               *self._idle_queue_blocking_timeout)
        self._worker_map = {}
        self._scenario_instance_factory = None
        self._full_scenario_tree = None

    def remove_worker(self, name):
        self._worker_map[name].close()
        del self._worker_map[name]
        types_to_keep = []
        for rqtype in self.current_type_order():
            if rqtype[0] != name:
                types_to_keep.append(rqtype)
        self.clear_request_types()
        for rqtype in types_to_keep:
            self.push_request_type(*rqtype)

    def process(self, data):
        try:
            return self._process(data)
        except:
            logger.error(
                "Scenario tree server %s caught an exception of type "
                "%s while processing a task. Going idle."
                % (self.WORKERNAME, sys.exc_info()[0].__name__))
            traceback.print_exception(*sys.exc_info())
            self._init()
            return (SPPyroScenarioTreeServer_ProcessTaskError,
                    traceback.format_exc())

    def _process(self, data):
        data = pyutilib.misc.Bunch(**data)
        result = None
        if data.action == 'SPPyroScenarioTreeServer_acknowledge':
            if self._global_verbose:
                print("Scenario tree server %s acknowledging work request"
                      % (self.WORKERNAME))
            assert self.num_request_types() == 1
            self.clear_request_types()
            self.push_request_type(self.WORKERNAME,
                                   *self._worker_queue_blocking_timeout)
            result = self.WORKERNAME

        elif data.action == 'SPPyroScenarioTreeServer_setup':
            options = self.register_options()
            for name, val in iteritems(data.options):
                options.get(name).set_value(val)
            self.set_options(options)
            self._options.verbose = self._options.verbose | self._global_verbose
            assert self._scenario_instance_factory is None
            assert self._full_scenario_tree is None
            if self._options.verbose:
                print("Received request to setup scenario tree server")
                print("Options:")
                self.display_options()

            # Make sure these are not archives
            assert os.path.exists(self._options.model_location)
            assert (self._options.scenario_tree_location is None) or \
                os.path.exists(self._options.scenario_tree_location)
            self._scenario_instance_factory = \
                ScenarioTreeInstanceFactory(
                    self._options.model_location,
                    scenario_tree_location=self._options.scenario_tree_location,
                    verbose=self._options.verbose)

            self._full_scenario_tree = \
                self._scenario_instance_factory.generate_scenario_tree(
                    downsample_fraction=self._options.scenario_tree_downsample_fraction,
                    random_seed=self._options.scenario_tree_random_seed)

            if self._full_scenario_tree is None:
                 raise RuntimeError("Unable to launch scenario tree worker - "
                                    "scenario tree construction failed.")

            result = True

        elif data.action == 'SPPyroScenarioTreeServer_release':
            if self._global_verbose:
                print("Scenario tree server %s releasing worker: %s"
                      % (self.WORKERNAME, data.worker_name))
            self.remove_worker(data.worker_name)
            if len(self.current_type_order()) == 1:
                # Go back to making general worker requests
                # blocking with a reasonable timeout so they
                # don't overload the dispatcher
                self.pop_request_type()
                self.push_request_type(self.WORKERNAME,
                                       *self._worker_queue_blocking_timeout)
            result = True

        elif data.action == 'SPPyroScenarioTreeServer_idle':
            if self._global_verbose:
                print("Scenario tree server %s going into idle mode"
                      % (self.WORKERNAME))
            server_names = list(self._worker_map.keys())
            for name in server_names:
                self.remove_worker(name)

            ignored_options = dict((_c._name, _c.value(False))
                                   for _c in self._options.unused_user_values())
            if len(ignored_options):
                print("")
                print("*** WARNING: The following options were "
                      "explicitly set but never accessed by server %s: "
                      % (self.WORKERNAME))
                for name in ignored_options:
                    print(" - %s: %s" % (name, ignored_options[name]))
                print("*** If you believe this is a bug, please report it "
                      "to the PySP developers.")
                print("")

            self._init()
            result = True

        elif data.action == 'SPPyroScenarioTreeServer_initialize':

            worker_name = data.worker_name
            if self._options.verbose:
                print("Received request to initialize scenario tree worker "
                      "named %s" % (worker_name))

            assert self._scenario_instance_factory is not None
            assert self._full_scenario_tree is not None

            current_types = self.current_type_order()
            for rqtype in current_types:
                if worker_name == rqtype[0]:
                    raise RuntimeError(
                        "Cannot initialize worker with name '%s' "
                        "because a work queue already exists with "
                        "this name" % (worker_name))

            if len(current_types) == 1:
                assert current_types[-1][0] == self.WORKERNAME
                # make the general worker request non blocking
                # as we now have higher priority work to perform
                self.pop_request_type()
                self.push_request_type(self.WORKERNAME,
                                       False,
                                       None)

            self.push_request_type(worker_name,
                                   *self._solver_queue_blocking_timeout)

            worker_type = self._registered_workers[data.worker_type]
            options = worker_type.register_options()
            for name, val in iteritems(data.options):
                options.get(name).set_value(val)
            init_type = getattr(WorkerInitType, data.init_type)
            self._worker_map[worker_name] = worker_type(
                self.WORKERNAME,
                self._full_scenario_tree,
                worker_name,
                init_type,
                data.init_data,
                options)

            result = True
        else:

            #with PauseGC() as pgc:
            result = getattr(self._worker_map[data.worker_name], data.action)\
                     (*data.args, **data.kwds)

        return result

def RegisterScenarioTreeWorker(name, class_type):
    assert name not in SPPyroScenarioTreeServer._registered_workers, \
        ("The name %s is already registered for another worker class"
         % (name))
    SPPyroScenarioTreeServer._registered_workers[name] = class_type

#
# Register some known, trusted workers
#
from pyomo.pysp.scenariotree.scenariotreeworkerbasic import \
    ScenarioTreeWorkerBasic
RegisterScenarioTreeWorker('ScenarioTreeWorkerBasic',
                           ScenarioTreeWorkerBasic)


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
        TaskWorkerServer(SPPyroScenarioTreeServer,
                         host=options.pyro_hostname,
                         verbose=options.verbose)
    except:
        # if an exception occurred, then we probably want to shut down
        # all Pyro components.  otherwise, the PH client may have
        # forever while waiting for results that will never
        # arrive. there are better ways to handle this at the PH
        # client level, but until those are implemented, this will
        # suffice for cleanup.
        #NOTE: this should perhaps be command-line driven, so it can
        #      be disabled if desired.
        print("SPPyroScenarioTreeServer aborted. Sending shutdown request.")
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
