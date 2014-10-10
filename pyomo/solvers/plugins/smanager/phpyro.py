#  _________________________________________________________________________
#
#  Coopr: A COmmon Optimization Python Repository
#  Copyright (c) 2011 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Coopr README.txt file.
#  _________________________________________________________________________


__all__ = ["PHSolverServerAction"]

try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    import Pyro.core
    import pyutilib.pyro
    using_pyro=True
except ImportError:
    using_pyro=False
import time

import pyutilib.misc
from pyutilib.enum import Enum

import coopr.core.plugin
from coopr.opt.parallel.manager import *
from coopr.opt.parallel.async_solver import *
from coopr.opt.results import SolverResults


# 
# an enumerated type used to define specific actions for a PH solver server.
#

PHSolverServerAction = Enum(
   'solve' # perform a scenario solve using whatever data is available
)

#
# a specialized asynchronous solver manager for Progressive Hedging.
#

class SolverManager_PHPyro(AsynchronousSolverManager):

    coopr.core.plugin.alias('phpyro', doc="Specialized PH solver manager that uses pyro")

    def clear(self):
        """
        Clear manager state
        """

        AsynchronousSolverManager.clear(self)

        # the client-side interface to the dispatch server.
        self.client = pyutilib.pyro.Client()

        # only useful for debugging communication patterns - results
        # in a ton of output.
        self._verbose = False

        # map from task id to the corresponding action handle.
        # we only retain entries for tasks for which we expect
        # a result/response.
        self._ah = {}

        # the list of cached results obtained from the dispatch server.
        # to avoid communication overhead, grab any/all results available,
        # and then cache them here - but return one-at-a-time via
        # the standard _perform_wait_any interface. the elements in this
        # list are simply tasks - at this point, we don't care about the
        # queue name associated with the task.
        self._results_waiting = []

    #
    # a utility to extract a single result from the _results_waiting list.
    #

    def _extract_result(self):

        if len(self._results_waiting) == 0:
            raise RuntimeError("There are no results available for extraction from the PHPyro solver manager - call to _extract_result is not valid.")
        
        task = self._results_waiting.pop(0)

        if task.id in self._ah:
            ah = self._ah[task.id]
            self._ah[task.id] = None
            ah.status = ActionStatus.done
            # TBD - what is the 'results' object - can we just load results directly into there?
            self.results[ah.id] = pickle.loads(task.result)
            return ah
        else:
            # if we are here, this is really bad news!
            raise RuntimeError("The PHPyro solver manager found results for task with id=" + str(task.id) + " - but no corresponding action handle could be located!")

    def _perform_queue(self, ah, *args, **kwds):
        """
        Perform the queue operation.  This method returns the ActionHandle,
        and the ActionHandle status indicates whether the queue was successful.
        """

        # the PH solver server expects no non-keyword arguments. 
        if len(args) > 0:
           raise RuntimeError("ERROR: The _perform_queue method of PH pyro solver manager received position input arguments, but accepts none.")

        if "action" not in kwds:
           raise RuntimeError("ERROR: No 'action' keyword supplied to _perform_queue method of PH pyro solver manager")

        if "name" not in kwds:
           raise RuntimeError("ERROR: No 'name' keyword supplied to _perform_queue method of PH pyro solver manager")
        instance_name = kwds["name"]

        if "verbose" in kwds:
            self._verbose = kwds["verbose"]
        else:
            # we always want to pass a verbose flag to the solver server.
            kwds["verbose"] = False 

        if "generateResponse" in kwds:
            generateResponse = kwds.pop("generateResponse")
        else:
            generateResponse = True

        #
        # Pickle everything into one big data object via the "Bunch" command and post the task.
        #
        data = pyutilib.misc.Bunch(**kwds)

        # NOTE: the task type (type=) should be the name of the scenario/bundle!

        task = pyutilib.pyro.Task(data=data, id=ah.id, generateResponse=generateResponse)
        self.client.add_task(task, verbose=self._verbose, override_type=instance_name)

        # only populate the action_handle-to-task dictionary is a response is expected.
        if generateResponse is True:
            self._ah[task.id] = ah

        return ah

    def _perform_wait_any(self):
        """
        Perform the wait_any operation.  This method returns an
        ActionHandle with the results of waiting.  If None is returned
        then the ActionManager assumes that it can call this method again.
        Note that an ActionHandle can be returned with a dummy value,
        to indicate an error.
        """

        if len(self._results_waiting) > 0:
            return self._extract_result()
            
        elif len(self.client.queues_with_results()) > 0:

            all_results = self.client.get_results_all_queues()

            for task in all_results:
                self._results_waiting.append(task)                
        else:

          # if the queues are all empty, wait some time for things to fill up.
          # constantly pinging dispatch servers wastes their time, and inhibits
          # task worker communication. the good thing about queues_to_check
          # is that it simultaneously grabs information for any queues with
          # results => one client query can yield many results.
          
          # TBD: We really need to parameterize the time-out value, but it
          #      isn't clear how to propagate this though the solver manager
          #      interface layers.
          time.sleep(0.01)

if not using_pyro:
    SolverManagerFactory.deactivate('phpyro')
