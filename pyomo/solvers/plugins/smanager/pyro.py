#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________


__all__ = []

import base64
import ast
try:
    import cPickle as pickle
except:
    import pickle
import six

import pyutilib.pyro
import pyutilib.misc

import pyomo.util.plugin
from pyomo.opt.parallel.manager import *
from pyomo.opt.parallel.async_solver import *



class SolverManager_Pyro(AsynchronousSolverManager):

    pyomo.util.plugin.alias('pyro', doc="Execute solvers remotely using pyro")

    def __init__(self, host=None):

        self.host = host
        AsynchronousActionManager.__init__(self)

    def clear(self):
        """
        Clear manager state
        """
        AsynchronousSolverManager.clear(self)
        self.client = pyutilib.pyro.Client(host=self.host)
        self._opt = None
        self._verbose = False
        self._ah = {}       # maps task ids to their corresponding action handle.
        self._smap_id = {}  # maps task ids to the corresponding symbol map ids.
        self._args = {}     # maps task ids to the corresponding queued arguments
        self.load_solution = True

    def _perform_queue(self, ah, *args, **kwds):
        """
        Perform the queue operation.  This method returns the ActionHandle,
        and the ActionHandle status indicates whether the queue was successful.
        """
        if 'opt' in kwds:
            self._opt = kwds['opt']
            del kwds['opt']
        else:
            raise ActionManagerError("No solver passed to SolverManager_Pyro, method=_perform_queue; use keyword option \"opt\"")

        if 'verbose' in kwds:
            self._verbose = kwds['verbose']
            del kwds['verbose']
        load_solutions = kwds.get('load_solutions', load_solutions)

        #
        # Force pyomo.opt to ignore tests for availability, at least locally.
        #
        kwds['available'] = True
        self._opt._presolve(*args, **kwds)
        problem_file_string = open(self._opt._problem_files[0],'r').read()
        #
        # Delete this option, to ensure that the remote worker does the check for
        # availability.
        #
        del kwds['available']
        #
        # We can't pickle the options object itself - so extract a simple
        # dictionary of solver options and re-construct it on the other end.
        #
        solver_options = {}
        for key in self._opt.options:
            solver_options[key]=self._opt.options[key]
        #
        # NOTE: let the distributed node deal with the warm-start
        # pick up the warm-start file, if available.
        #
        warm_start_file_string = None
        warm_start_file_name = None
        if hasattr(self._opt,  "warm_start_solve"):
            if (self._opt.warm_start_solve is True) and (self._opt.warm_start_file_name is not None):
                warm_start_file_name = self._opt.warm_start_file_name
        #
        data = pyutilib.misc.Bunch(opt=self._opt.type, \
                                 file=problem_file_string, \
                                 filename=self._opt._problem_files[0], \
                                 warmstart_file=warm_start_file_string, \
                                 warmstart_filename=warm_start_file_name, \
                                 kwds=kwds, \
                                 solver_options=solver_options, \
                                 suffixes=self._opt.suffixes)

        task = pyutilib.pyro.Task(data=data.copy(), id=ah.id)
        self.client.add_task(task, verbose=self._verbose)
        self._ah[task['id']] = ah
        self._smap_id[task['id']] = self._opt._smap_id
        self._args[task['id']] = args
        #
        return ah

    def _perform_wait_any(self):
        """
        Perform the wait_any operation.  This method returns an
        ActionHandle with the results of waiting.  If None is returned
        then the ActionManager assumes that it can call this method again.
        Note that an ActionHandle can be returned with a dummy value,
        to indicate an error.
        """
        from pyomo.core import Model

        if self.client.num_results() > 0:
            # this protects us against the case where we get an action
            # handle that we didn't know about or expect.
            while(True):
                task = self.client.get_result()
                if task['id'] in self._ah:

                    ah = self._ah[task['id']]
                    del self._ah[task['id']]

                    smap_id = self._smap_id[task['id']]
                    del self._smap_id[task['id']]

                    args = self._args[task['id']]
                    del self._args[task['id']]

                    ah.status = ActionStatus.done

                    pickled_results = task['result']
                    if six.PY3:
                        # These two conversions are in place to unwrap
                        # the hacks placed in the pyro_mip_server
                        # before transmitting the results
                        # object. These hacks are put in place to
                        # avoid errors when transmitting the pickled
                        # form of the results object with Pyro4.
                        pickled_results = \
                            base64.decodebytes(
                                ast.literal_eval(pickled_results))
                    self.results[ah.id] = pickle.loads(pickled_results)

                    # Tag the results object with the symbol map id.
                    self.results[ah.id]._smap_id = smap_id

                    if load_solution and isinstance(args[0],Model):
                        args[0].solutions.load_from(self.results[ah.id])
                    return ah


if pyutilib.pyro.Pyro is None:
    SolverManagerFactory.deactivate('pyro')
