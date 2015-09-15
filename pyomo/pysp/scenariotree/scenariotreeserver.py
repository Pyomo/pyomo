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
import socket
import copy
import argparse
import logging
import traceback

import pyutilib.misc
from pyutilib.misc import PauseGC
from pyutilib.misc.config import ConfigValue, ConfigBlock
from pyutilib.pyro import (TaskWorker,
                           TaskWorkerServer,
                           shutdown_pyro_components,
                           TaskProcessingError)
from pyomo.util import pyomo_command
from pyomo.opt import (SolverFactory,
                       PersistentSolver,
                       TerminationCondition,
                       SolutionStatus)
from pyomo.opt.parallel.manager import ActionManagerError
from pyomo.pysp.util.misc import launch_command, load_external_module
from pyomo.pysp.util.config import (safe_register_common_option,
                                    safe_declare_common_option,
                                    safe_declare_unique_option,
                                    _domain_tuple_of_str)
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory
from pyomo.pysp.scenariotree.scenariotreeserverutils import \
    (WorkerInitType,
     WorkerInit)

from six import iteritems

logger = logging.getLogger('pyomo.pysp')

class SPPyroScenarioTreeServer(TaskWorker, PySPConfiguredObject):

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
        kwds["caller_name"] = "PH Pyro Server"
        kwds["name"] = ("PySPWorker_%d@%s" % (os.getpid(), socket.gethostname()))
        TaskWorker.__init__(self, **kwds)
        # This classes options get updated during the "setup" phase
        options = self.register_options()
        PySPConfiguredObject.__init__(self, options)

        self.type = self.WORKERNAME
        self.block = True
        self.timeout = None
        self._worker_map = {}

        #
        # These will be used by all subsequent workers created
        # by this server. Their creation can eat up a nontrivial
        # amount of initialization time when a large number of
        # workers are created on this server, so we only create
        # them once.
        #
        self._scenario_instance_factory = None
        self._full_scenario_tree = None

    def reset(self):
        if self._scenario_instance_factory is not None:
            self._scenario_instance_factory.close()
        self._scenario_instance_factory = None
        self._full_scenario_tree = None
        for worker_name in list(self._worker_map):
            self.remove_worker(worker_name)

    def remove_worker(self, name):
        self._worker_map[name].close()
        del self._worker_map[name]

    def process(self, data):
        try:
            return self._process(data)
        except:
            logger.error(
                "Scenario tree server %s caught an exception of type "
                "%s while processing a task. Going idle."
                % (self.WORKERNAME, sys.exc_info()[0].__name__))
            traceback.print_exception(*sys.exc_info())
            self._worker_error = True
            return TaskProcessingError(traceback.format_exc())

    def _process(self, data):

        data = pyutilib.misc.Bunch(**data)
        result = None
        if not data.action.startswith('SPPyroScenarioTreeServer_'):
            #with PauseGC() as pgc:
            result = getattr(self._worker_map[data.worker_name], data.action)\
                     (*data.args, **data.kwds)

        elif data.action == 'SPPyroScenarioTreeServer_setup':
            options = self.register_options()
            for name, val in iteritems(data.options):
                options.get(name).set_value(val)
            self.set_options(options)
            self._options.verbose = self._options.verbose | self._verbose
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

        elif data.action == "SPPyroScenarioTreeServer_initialize":

            worker_name = data.worker_name
            if self._options.verbose:
                print("Received request to initialize scenario tree worker "
                      "named %s" % (worker_name))

            assert self._scenario_instance_factory is not None
            assert self._full_scenario_tree is not None


            if worker_name in self._worker_map:
                raise RuntimeError(
                    "Cannot initialize worker with name '%s' "
                    "because a work queue already exists with "
                    "this name" % (worker_name))

            worker_type = self._registered_workers[data.worker_type]
            options = worker_type.register_options()
            for name, val in iteritems(data.options):
                options.get(name).set_value(val)

            #
            # Depending on the Pyro serializer, the namedtuple
            # may be been converted to a tuple
            #
            if not isinstance(data.worker_init, WorkerInit):
                assert type(data.worker_init) is tuple
                data.worker_init = WorkerInit(type_=data.worker_init[0],
                                              names=data.worker_init[1],
                                              data=data.worker_init[2])

            # replace enum string representation with the actual enum
            # object now that we've unserialized the Pyro data
            worker_init = WorkerInit(type_=getattr(WorkerInitType,
                                                   data.worker_init.type_),
                                     names=data.worker_init.names,
                                     data=data.worker_init.data)
            self._worker_map[worker_name] = worker_type(
                self.WORKERNAME,
                self._full_scenario_tree,
                worker_name,
                worker_init,
                options)

            result = True

        elif data.action == "SPPyroScenarioTreeServer_release":

            if self._options.verbose:
                print("Scenario tree server %s releasing worker: %s"
                      % (self.WORKERNAME, data.worker_name))
            self.remove_worker(data.worker_name)
            result = True

        elif data.action == "SPPyroScenarioTreeServer_reset":

            if self._options.verbose:
                print("Scenario tree server received reset request")
            self.reset()
            result = True

        elif data.action == "SPPyroScenarioTreeServer_shutdown":

            if self._options.verbose:
                print("Scenario tree server received shutdown request")
            self.reset()
            self._worker_shutdown = True
            result = True

        else:
            raise ValueError("Invalid SPPyroScenarioTreeServer command: %s"
                             % (data.action))

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
from pyomo.pysp.scenariotree.scenariotreesolverworker import \
    ScenarioTreeSolverWorker
RegisterScenarioTreeWorker('ScenarioTreeSolverWorker',
                           ScenarioTreeSolverWorker)

#
# utility method fill a ConfigBlock with options associated
# with the scenariotreeserver command
#

def scenariotreeserver_register_options(options):
    safe_declare_common_option(options, "disable_gc")
    safe_declare_common_option(options, "profile")
    safe_declare_common_option(options, "traceback")
    safe_declare_common_option(options, "verbose")
    safe_declare_common_option(options, "pyro_hostname")
    safe_declare_unique_option(
        options,
        "import_module",
        ConfigValue(
            (),
            domain=_domain_tuple_of_str,
            description=(
                "The name of a user-defined python module to import that, "
                "e.g., registers a user-defined scenario tree worker class."
            ),
            doc=None,
            visibility=0))

#
# Execute the scenario tree server daemon.
#
def exec_scenariotreeserver(options):

    for module_name in options.import_module:
        load_external_module(module_name)

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
