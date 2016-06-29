#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("ScenarioTreeServerPyro",
           "RegisterWorker")

import os
import sys
import socket
import copy
import argparse
import logging
import traceback
import base64
try:
    import cPickle as pickle
except:
    import pickle

import pyutilib.misc
from pyutilib.misc import PauseGC
from pyutilib.pyro import (TaskWorker,
                           TaskWorkerServer,
                           shutdown_pyro_components,
                           TaskProcessingError,
                           using_pyro4)
if using_pyro4:
    import Pyro4

from pyomo.util import pyomo_command
from pyomo.opt import (SolverFactory,
                       PersistentSolver,
                       TerminationCondition,
                       SolutionStatus)
from pyomo.opt.parallel.manager import ActionManagerError
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command,
                                  load_external_module)
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_declare_common_option,
                                    safe_register_common_option,
                                    safe_register_unique_option,
                                    _domain_tuple_of_str)
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory
from pyomo.pysp.scenariotree.server_pyro_utils import \
    (WorkerInitType,
     WorkerInit)

import six
from six import iteritems

logger = logging.getLogger('pyomo.pysp')

class ScenarioTreeServerPyro(TaskWorker, PySPConfiguredObject):

    # Maps name to a registered worker class to instantiate
    _registered_workers = {}

    _declared_options = \
        PySPConfigBlock("Options declared for the "
                        "ScenarioTreeServerPyro class")

    #
    # scenario instance construction
    #
    safe_declare_common_option(_declared_options,
                               "model_location")
    safe_declare_common_option(_declared_options,
                               "scenario_tree_location")

    #
    # scenario tree generation
    #
    safe_declare_common_option(_declared_options,
                               "scenario_tree_random_seed")
    safe_declare_common_option(_declared_options,
                               "scenario_tree_downsample_fraction")

    #
    # various
    #
    safe_declare_common_option(_declared_options,
                               "verbose")

    @classmethod
    def get_registered_worker_type(cls, name):
        if name in cls._registered_workers:
            return cls._registered_workers[name]
        raise KeyError("No worker type has been registered under the name "
                       "'%s' for ScenarioTreeServerPyro" % (name))

    def __init__(self, *args, **kwds):


        # add for purposes of diagnostic output.
        kwds["name"] = ("ScenarioTreeServerPyro_%d@%s"
                        % (os.getpid(), socket.gethostname()))
        kwds["caller_name"] = kwds["name"]
        self._modules_imported = kwds.pop('modules_imported', {})

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
        self._worker_task_return_queue = self._current_task_client
        try:
            # The only reason we are go through this much
            # effort to deal with the serpent serializer
            # is because it is the default in Pyro4.
            if using_pyro4 and \
               (Pyro4.config.SERIALIZER == 'serpent'):
                if six.PY3:
                    assert type(data) is dict
                    assert data['encoding'] == 'base64'
                    data = base64.b64decode(data['data'])
                else:
                    assert type(data) is unicode
                    data = str(data)
            return pickle.dumps(self._process(pickle.loads(data)))
        except:
            logger.error(
                "Scenario tree server %s caught an exception of type "
                "%s while processing a task. Going idle."
                % (self.WORKERNAME, sys.exc_info()[0].__name__))
            traceback.print_exception(*sys.exc_info())
            self._worker_error = True
            return pickle.dumps(TaskProcessingError(traceback.format_exc()))

    def _process(self, data):
        data = pyutilib.misc.Bunch(**data)
        result = None
        if not data.action.startswith('ScenarioTreeServerPyro_'):
            #with PauseGC() as pgc:
            result = getattr(self._worker_map[data.worker_name], data.action)\
                     (*data.args, **data.kwds)

        elif data.action == 'ScenarioTreeServerPyro_setup':
            options = self.register_options()
            for name, val in iteritems(data.options):
                options.get(name).set_value(val)
            self.set_options(options)
            self._options.verbose = self._options.verbose | self._verbose
            assert self._scenario_instance_factory is None
            assert self._full_scenario_tree is None
            if self._options.verbose:
                print("Server %s received setup request."
                      % (self.WORKERNAME))
                print("Options:")
                self.display_options()

            # Make sure these are not archives
            assert os.path.exists(self._options.model_location)
            assert (self._options.scenario_tree_location is None) or \
                os.path.exists(self._options.scenario_tree_location)
            self._scenario_instance_factory = \
                ScenarioTreeInstanceFactory(
                    self._options.model_location,
                    self._options.scenario_tree_location)

            #
            # Try to prevent unnecessarily re-importing the model module
            # if other callbacks are in the same location. Doing so might
            # have serious consequences.
            #
            if self._scenario_instance_factory._model_module is not None:
                self._modules_imported[self._scenario_instance_factory.\
                                       _model_filename] = \
                    self._scenario_instance_factory._model_module
            if self._scenario_instance_factory._scenario_tree_module is not None:
                self._modules_imported[self._scenario_instance_factory.\
                                       _scenario_tree_filename] = \
                    self._scenario_instance_factory._scenario_tree_module

            self._full_scenario_tree = \
                self._scenario_instance_factory.generate_scenario_tree(
                    downsample_fraction=self._options.scenario_tree_downsample_fraction,
                    random_seed=self._options.scenario_tree_random_seed,
                    verbose=self._options.verbose)

            if self._full_scenario_tree is None:
                 raise RuntimeError("Unable to launch scenario tree worker - "
                                    "scenario tree construction failed.")

            result = True

        elif data.action == "ScenarioTreeServerPyro_initialize":

            worker_name = data.worker_name
            if self._options.verbose:
                print("Server %s received request to initialize "
                      "scenario tree worker with name %s."
                      % (self.WORKERNAME, worker_name))

            assert self._scenario_instance_factory is not None
            assert self._full_scenario_tree is not None

            if worker_name in self._worker_map:
                raise RuntimeError(
                    "Server %s Cannot initialize worker with name '%s' "
                    "because a worker already exists with that name."
                     % (self.WORKERNAME, worker_name))

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
            self._worker_map[worker_name] = worker_type(options)
            self._worker_map[worker_name].initialize(
                self.WORKERNAME,
                self._full_scenario_tree,
                worker_name,
                worker_init,
                self._modules_imported)

            result = True

        elif data.action == "ScenarioTreeServerPyro_release":

            if self._options.verbose:
                print("Server %s releasing worker: %s"
                      % (self.WORKERNAME, data.worker_name))
            self.remove_worker(data.worker_name)
            result = True

        elif data.action == "ScenarioTreeServerPyro_reset":

            if self._options.verbose:
                print("Server %s received reset request"
                      % (self.WORKERNAME))
            self.reset()
            result = True

        elif data.action == "ScenarioTreeServerPyro_shutdown":

            if self._options.verbose:
                print("Server %s received shutdown request"
                      % (self.WORKERNAME))
            self.reset()
            self._worker_shutdown = True
            result = True

        else:
            raise ValueError("Server %s: Invalid command: %s"
                             % (self.WORKERNAME, data.action))

        return result

def RegisterWorker(name, class_type):
    assert name not in ScenarioTreeServerPyro._registered_workers, \
        ("The name %s is already registered for another worker class"
         % (name))
    ScenarioTreeServerPyro._registered_workers[name] = class_type

#
# Register some known, trusted workers
#
from pyomo.pysp.scenariotree.manager_worker_pyro import \
    ScenarioTreeManagerWorkerPyro
RegisterWorker('ScenarioTreeManagerWorkerPyro',
               ScenarioTreeManagerWorkerPyro)
from pyomo.pysp.scenariotree.manager_solver_worker_pyro import \
    ScenarioTreeManagerSolverWorkerPyro
RegisterWorker('ScenarioTreeManagerSolverWorkerPyro',
               ScenarioTreeManagerSolverWorkerPyro)

#
# utility method fill a PySPConfigBlock with options associated
# with the scenariotreeserver command
#

def scenariotreeserver_register_options(options=None):
    if options is None:
        options = PySPConfigBlock()
    safe_register_common_option(options, "disable_gc")
    safe_register_common_option(options, "profile")
    safe_register_common_option(options, "traceback")
    safe_register_common_option(options, "verbose")
    safe_register_common_option(options, "pyro_host")
    safe_register_common_option(options, "pyro_port")
    safe_register_unique_option(
        options,
        "import_module",
        PySPConfigValue(
            (),
            domain=_domain_tuple_of_str,
            description=(
                "The name of a user-defined python module to import that, "
                "e.g., registers a user-defined scenario tree worker class."
            ),
            doc=None,
            visibility=0))

    return options
#
# Execute the scenario tree server daemon.
#
def exec_scenariotreeserver(options):

    modules_imported = {}
    for module_name in options.import_module:
        if module_name in sys.modules:
            modules_imported[module_name] = sys.modules[module_name]
        else:
            modules_imported[module_name] = \
                load_external_module(module_name, clear_cache=True)[0]

    try:
        # spawn the daemon
        TaskWorkerServer(ScenarioTreeServerPyro,
                         host=options.pyro_host,
                         port=options.pyro_port,
                         verbose=options.verbose,
                         modules_imported=modules_imported)
    except:
        # if an exception occurred, then we probably want to shut down
        # all Pyro components.  otherwise, the PH client may have
        # forever while waiting for results that will never
        # arrive. there are better ways to handle this at the PH
        # client level, but until those are implemented, this will
        # suffice for cleanup.
        #NOTE: this should perhaps be command-line driven, so it can
        #      be disabled if desired.
        print("ScenarioTreeServerPyro aborted. Sending shutdown request.")
        shutdown_pyro_components(host=options.pyro_host,
                                 port=options.pyro_port,
                                 num_retries=0)
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
    try:
        options = parse_command_line(
            args,
            scenariotreeserver_register_options,
            prog='scenariotreeserver',
            description=(
"""Launches a scenariotreeserver process to manage workers in a
distributed scenario tree."""
            ))

    except SystemExit as _exc:
        # the parser throws a system exit if "-h" is specified
        # - catch it to exit gracefully.
        return _exc.code

    return launch_command(exec_scenariotreeserver,
                          options,
                          error_label="scenariotreeserver: ",
                          disable_gc=options.disable_gc,
                          profile_count=options.profile,
                          traceback=options.traceback)
