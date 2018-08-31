#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


__all__ = []

import base64
import ast
try:
    import cPickle as pickle
except:
    import pickle

import pyutilib.pyro
from pyutilib.pyro import using_pyro4, TaskProcessingError
import pyutilib.misc
from pyomo.opt.base import OptSolver
from pyomo.opt.parallel.manager import ActionManagerError, ActionStatus
from pyomo.opt.parallel.async_solver import (AsynchronousSolverManager,
                                             SolverManagerFactory)
from pyomo.opt.parallel.pyro import PyroAsynchronousActionManager
from pyomo.core.base import Block
import pyomo.core.base.suffix

from pyomo.core.kernel.block import IBlock
import pyomo.core.kernel.suffix

import six

@SolverManagerFactory.register('pyro', doc="Execute solvers remotely using pyro")
class SolverManager_Pyro(PyroAsynchronousActionManager, AsynchronousSolverManager):


    def __init__(self, *args, **kwds):
        self._opt_data = {}
        self._args = {}
        self._client = None
        super(SolverManager_Pyro, self).__init__(*args, **kwds)

    def clear(self):
        """Clear manager state"""
        super(SolverManager_Pyro, self).clear()
        self.client = self._create_client()
        assert len(self._dispatcher_name_to_client) == 1
        self.client.clear_queue()
        self._opt_data = {}
        self._args = {}

    #
    # Abstract Methods
    #

    def _get_dispatcher_name(self, queue_name):
        assert queue_name is None
        assert len(self._dispatcher_name_to_client) == 1
        assert self.client.URI in self._dispatcher_name_to_client
        return self.client.URI

    def _get_task_data(self, ah, *args, **kwds):

        opt = kwds.pop('solver', kwds.pop('opt', None))
        if opt is None:
            raise ActionManagerError(
                "No solver passed to %s, use keyword option 'solver'"
                % (type(self).__name__) )
        if isinstance(opt, six.string_types):
            opt = SolverFactory(opt, solver_io=kwds.pop('solver_io', None))

        #
        # The following block of code is taken from the OptSolver.solve()
        # method, which we do not directly invoke with this interface
        #

        #
        # If the inputs are models, then validate that they have been
        # constructed! Collect suffix names to try and import from solution.
        #
        for arg in args:
            if isinstance(arg, (Block, IBlock)):
                if isinstance(arg, Block):
                    if not arg.is_constructed():
                        raise RuntimeError(
                            "Attempting to solve model=%s with unconstructed "
                            "component(s)" % (arg.name))
                # import suffixes must be on the top-level model
                if isinstance(arg, Block):
                    model_suffixes = list(name for (name,comp) \
                                          in pyomo.core.base.suffix.\
                                          active_import_suffix_generator(arg))
                else:
                    assert isinstance(arg, IBlock)
                    model_suffixes = list(comp.storage_key for comp \
                                          in pyomo.core.base.suffix.\
                                          import_suffix_generator(arg,
                                                                  active=True,
                                                                  descend_into=False))
                if len(model_suffixes) > 0:
                    kwds_suffixes = kwds.setdefault('suffixes',[])
                    for name in model_suffixes:
                        if name not in kwds_suffixes:
                            kwds_suffixes.append(name)

        #
        # Handle ephemeral solvers options here. These
        # will override whatever is currently in the options
        # dictionary, but we will reset these options to
        # their original value at the end of this method.
        #
        ephemeral_solver_options = {}
        ephemeral_solver_options.update(kwds.pop('options', {}))
        ephemeral_solver_options.update(
            OptSolver._options_string_to_dict(kwds.pop('options_string', '')))

        #
        # Force pyomo.opt to ignore tests for availability, at least locally.
        #
        del_available = bool('available' not in kwds)
        kwds['available'] = True
        opt._presolve(*args, **kwds)
        problem_file_string = None
        with open(opt._problem_files[0], 'r') as f:
            problem_file_string = f.read()

        #
        # Delete this option, to ensure that the remote worker does the check for
        # availability.
        #
        if del_available:
            del kwds['available']

        #
        # We can't pickle the options object itself - so extract a simple
        # dictionary of solver options and re-construct it on the other end.
        #
        solver_options = {}
        for key in opt.options:
            solver_options[key]=opt.options[key]
        solver_options.update(ephemeral_solver_options)

        #
        # NOTE: let the distributed node deal with the warm-start
        # pick up the warm-start file, if available.
        #
        warm_start_file_string = None
        warm_start_file_name = None
        if hasattr(opt,  "_warm_start_solve"):
            if opt._warm_start_solve  and \
               (opt._warm_start_file_name is not None):
                warm_start_file_name = opt._warm_start_file_name
                with open(warm_start_file_name, 'r') as f:
                    warm_start_file_string = f.read()

        data = pyutilib.misc.Bunch(opt=opt.type, \
                                   file=problem_file_string, \
                                   filename=opt._problem_files[0], \
                                   warmstart_file=warm_start_file_string, \
                                   warmstart_filename=warm_start_file_name, \
                                   kwds=kwds, \
                                   solver_options=solver_options, \
                                   suffixes=opt._suffixes)

        self._args[ah.id] = args
        self._opt_data[ah.id] = (opt._smap_id,
                                 opt._load_solutions,
                                 opt._select_index,
                                 opt._default_variable_value)

        return data

    def _download_results(self):

        results = self.client.get_results(override_type=self.client.CLIENTNAME,
                                          block=True,
                                          timeout=None)
        for task in results:
            self.queued_action_counter -= 1
            ah = self.event_handle.get(task['id'], None)
            if ah is None:
                # if we are here, this is really bad news!
                raise RuntimeError(
                    "The %s found results for task with id=%s"
                    " - but no corresponding action handle "
                    "could be located!" % (type(self).__name__, task['id']))
            if type(task['result']) is TaskProcessingError:
                ah.status = ActionStatus.error
                self.event_handle[ah.id].update(ah)
                raise RuntimeError(
                    "Dispatcher reported a processing error "
                    "for task with id=%s. Reason: \n%s"
                    % (task['id'], task['result'].args[0]))
            else:
                ah.status = ActionStatus.done
                self.event_handle[ah.id].update(ah)

                (smap_id,
                 load_solutions,
                 select_index,
                 default_variable_value) = self._opt_data[task['id']]
                del self._opt_data[task['id']]

                args = self._args[task['id']]
                del self._args[task['id']]

                results = task['result']
                if using_pyro4:
                    # These two conversions are in place to unwrap
                    # the hacks placed in the pyro_mip_server
                    # before transmitting the results
                    # object. These hacks are put in place to
                    # avoid errors when transmitting the pickled
                    # form of the results object with the default Pyro4
                    # serializer (Serpent)
                    if six.PY3:
                        results = base64.decodebytes(
                            ast.literal_eval(results))
                    else:
                        results = base64.decodestring(results)

                results = pickle.loads(results)

                # Tag the results object with the symbol map id.
                results._smap_id = smap_id

                if isinstance(args[0], Block):
                    _model = args[0]
                    if load_solutions:
                        _model.solutions.load_from(
                            results,
                            select=select_index,
                            default_variable_value=default_variable_value)
                        results._smap_id = None
                        results.solution.clear()
                    else:
                        results._smap = _model.solutions.symbol_map[smap_id]
                        _model.solutions.delete_symbol_map(smap_id)

                self.results[ah.id] = results

    def shutdown_workers(self):

        shutdown_task = pyutilib.pyro.Task(
            data={'action':'Pyomo_pyro_mip_server_shutdown'},
            id=float('inf'),
            generateResponse=False)

        dispatcher = self.client.dispatcher
        workers = dispatcher.acquire_available_workers()
        dispatcher.release_acquired_workers(workers)
        for worker_name in workers:
            self.client.add_task(shutdown_task,
                                 verbose=self._verbose > 1)

if pyutilib.pyro.Pyro is None:
    SolverManagerFactory.unregister('pyro')
