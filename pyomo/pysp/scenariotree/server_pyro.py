#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ("ScenarioTreeServerPyro",
           "RegisterWorker")

import os
import six
import sys
import socket
import logging
import traceback
import base64
try:
    import cPickle as pickle
except:                                           #pragma:nocover
    import pickle

from pyomo.common.collections import Bunch
from pyomo.common.dependencies import attempt_import, dill, dill_available
from pyomo.common import pyomo_command
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command,
                                  load_external_module)
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_register_unique_option,
                                    _domain_tuple_of_str)
from pyomo.pysp.scenariotree.tree_structure import \
    ScenarioTree
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory

pyu_pyro = attempt_import('pyutilib.pyro', alt_names=['pyu_pyro'])[0]
Pyro4 = attempt_import('Pyro4')[0]

logger = logging.getLogger('pyomo.pysp')

class ScenarioTreeServerPyro(pyu_pyro.TaskWorker):

    # Maps name to a registered worker class to instantiate
    _registered_workers = {}

    @classmethod
    def get_registered_worker_type(cls, name):
        if name in cls._registered_workers:
            return cls._registered_workers[name]
        raise KeyError("No worker type has been registered under the name "
                       "'%s' for ScenarioTreeServerPyro" % (name))

    def __init__(self, *args, **kwds):

        mpi = kwds.pop('mpi', None)
        # add for purposes of diagnostic output.
        kwds["name"] = ("ScenarioTreeServerPyro_%d@%s"
                        % (os.getpid(), socket.gethostname()))
        if mpi is not None:
            assert len(mpi) == 2
            kwds["name"] += "_MPIRank_"+str(mpi[1].rank)
        kwds["caller_name"] = kwds["name"]
        self._modules_imported = kwds.pop('modules_imported', {})

        pyu_pyro.TaskWorker.__init__(self, **kwds)
        assert hasattr(self, "_bulk_task_collection")
        self._bulk_task_collection = True
        self._contiguous_task_processing = False

        self.type = self.WORKERNAME
        self.block = True
        self.timeout = None
        self._worker_map = {}
        self._init_verbose = self._verbose

        # A reference to the mpi4py.MPI namespace
        self.MPI = None
        # The communicator and group associated with all processors
        self.mpi_comm_world = None
        self.mpi_group_world = None
        # The communicator associated with the workers assigned
        # to the current current client
        self.mpi_comm_workers = None
        if mpi is not None:
            assert len(mpi) == 2
            self.MPI = mpi[0]
            self.mpi_comm_world = mpi[1]
            self.mpi_group_world = self.mpi_comm_world.Get_group()

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
        if self.mpi_comm_workers is not None:
            self.mpi_comm_workers.Free()
            self.mpi_comm_workers = None
        self._verbose = self._init_verbose

    def remove_worker(self, name):
        self._worker_map[name].close()
        del self._worker_map[name]

    def process(self, data):
        self._worker_task_return_queue = self._current_task_client
        try:
            # The only reason we are go through this much
            # effort to deal with the serpent serializer
            # is because it is the default in Pyro4.
            if pyu_pyro.using_pyro4 and \
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
            return pickle.dumps(pyu_pyro.TaskProcessingError(
                traceback.format_exc()))

    def _process(self, data):
        data = Bunch(**data)
        result = None
        if not data.action.startswith('ScenarioTreeServerPyro_'):
            result = getattr(self._worker_map[data.worker_name], data.action)\
                     (*data.args, **data.kwds)

        elif data.action == 'ScenarioTreeServerPyro_setup':
            model_input = data.options.pop('model', None)
            if model_input is None:
                model_input = data.options.pop('model_callback')
                assert dill_available
                model_input = dill.loads(model_input)

            scenario_tree_input = data.options.pop('scenario_tree')
            data_input = data.options.pop('data')
            mpi_group = data.options.pop("mpi_group",None)
            verbose = data.options.pop("verbose", False)
            assert len(data.options) == 0
            self._verbose |= verbose
            assert self._scenario_instance_factory is None
            assert self._full_scenario_tree is None
            if self._verbose:
                print("Server %s received setup request."
                      % (self.WORKERNAME))

            # Make sure these are not archives
            assert (not isinstance(model_input, six.string_types)) or \
                os.path.exists(model_input)
            assert isinstance(scenario_tree_input, ScenarioTree)
            self._scenario_instance_factory = \
                ScenarioTreeInstanceFactory(
                    model_input,
                    scenario_tree_input,
                    data=data_input)

            #
            # Try to prevent unnecessarily re-importing the model module
            # if other callbacks are in the same location. Doing so might
            # have serious consequences.
            #
            if self._scenario_instance_factory._model_module is not None:
                self._modules_imported[self._scenario_instance_factory.\
                                       _model_filename] = \
                    self._scenario_instance_factory._model_module
            assert self._scenario_instance_factory._scenario_tree_module is None

            self._full_scenario_tree = \
                 self._scenario_instance_factory.generate_scenario_tree()

            assert self.mpi_comm_workers is None
            if self.mpi_comm_world is not None:
                assert self.mpi_group_world is not None
                assert mpi_group is not None
                mpi_group = self.mpi_group_world.Incl(mpi_group)
                self.mpi_comm_workers = \
                    self.mpi_comm_world.Create_group(mpi_group)
            else:
                assert mpi_group is None

            if self._full_scenario_tree is None:
                 raise RuntimeError("Unable to launch scenario tree worker - "
                                    "scenario tree construction failed.")

            result = True

        elif data.action == "ScenarioTreeServerPyro_initialize":

            worker_name = data.worker_name
            if self._verbose:
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

            self._worker_map[worker_name] = worker_type(
                self,
                worker_name,
                *data.init_args,
                **data.init_kwds)
            result = True

        elif data.action == "ScenarioTreeServerPyro_release":

            if self._verbose:
                print("Server %s releasing worker: %s"
                      % (self.WORKERNAME, data.worker_name))
            self.remove_worker(data.worker_name)
            result = True

        elif data.action == "ScenarioTreeServerPyro_reset":

            if self._verbose:
                print("Server %s received reset request"
                      % (self.WORKERNAME))
            self.reset()
            result = True

        elif data.action == "ScenarioTreeServerPyro_shutdown":

            if self._verbose:
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
    if name in ScenarioTreeServerPyro._registered_workers:
        raise ValueError("The name %s is already registered "
                         "for another worker class"
                         % (name))
    ScenarioTreeServerPyro._registered_workers[name] = class_type

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
        "mpi",
        PySPConfigValue(
            False,
            domain=bool,
            description=(
                "Activate MPI based functionality. "
                "Requires the mpi4py module."
            ),
            doc=None,
            visibility=0))
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

    mpi = None
    if options.mpi:
        import mpi4py
        # This import calls MPI_Init
        import mpi4py.MPI
        mpi = (mpi4py.MPI, mpi4py.MPI.COMM_WORLD)

    modules_imported = {}
    for module_name in options.import_module:
        if module_name in sys.modules:
            modules_imported[module_name] = sys.modules[module_name]
        else:
            modules_imported[module_name] = \
                load_external_module(module_name,
                                     clear_cache=True,
                                     verbose=True)[0]

    try:
        # spawn the daemon
        pyu_pyro.TaskWorkerServer(ScenarioTreeServerPyro,
                                 host=options.pyro_host,
                                 port=options.pyro_port,
                                 verbose=options.verbose,
                                 modules_imported=modules_imported,
                                 mpi=mpi)
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
        pyu_pyro.shutdown_pyro_components(host=options.pyro_host,
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
