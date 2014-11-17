#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2011 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________


__all__ = ["PHSolverServerAction"]

import sys
import time

import pyutilib.pyro
import pyutilib.misc
from pyutilib.enum import Enum

import pyomo.util.plugin
from pyomo.opt.parallel.manager import *
from pyomo.opt.parallel.async_solver import *
from pyomo.opt.results import SolverResults

import six

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

    pyomo.util.plugin.alias('phpyro', doc="Specialized PH solver manager that uses pyro")

    def __init__(self, host=None):

        # the PHPyroWorker objects associated with this manager
        self.worker_pool = []
        self.host = host
        AsynchronousActionManager.__init__(self)

    def clear(self):
        """
        Clear manager state
        """

        AsynchronousSolverManager.clear(self)

        # the client-side interface to the dispatch server.
        self.client = pyutilib.pyro.Client(host=self.host)

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

        if len(self.worker_pool):
            self.release_workers()

    #
    # a utility to extract a single result from the _results_waiting
    # list.
    #

    def _extract_result(self):

        if len(self._results_waiting) == 0:
            raise RuntimeError("There are no results available for "
                               "extraction from the PHPyro solver manager "
                               "- call to _extract_result is not valid.")

        task = self._results_waiting.pop(0)

        if task['id'] in self._ah:
            ah = self._ah[task['id']]
            self._ah[task['id']] = None
            ah.status = ActionStatus.done
            # TBD - what is the 'results' object - can we just load
            # results directly into there?
            self.results[ah.id] = task['result']
            return ah
        else:
            # if we are here, this is really bad news!
            raise RuntimeError("The PHPyro solver manager found "
                               "results for task with id="+str(task['id'])+
                               " - but no corresponding action handle "
                               "could be located!")

    def _perform_queue(self, ah, *args, **kwds):
        """
        Perform the queue operation.  This method returns the
        ActionHandle, and the ActionHandle status indicates whether
        the queue was successful.
        """

        # the PH solver server expects no non-keyword arguments.
        if len(args) > 0:
           raise RuntimeError("ERROR: The _perform_queue method of PH "
                              "pyro solver manager received position input "
                              "arguments, but accepts none.")

        if "action" not in kwds:
           raise RuntimeError("ERROR: No 'action' keyword supplied to "
                              "_perform_queue method of PH pyro solver manager")

        if "name" not in kwds:
           raise RuntimeError("ERROR: No 'name' keyword supplied to "
                              "_perform_queue method of PH pyro solver manager")

        name = kwds["name"]

        if "verbose" in kwds:
            self._verbose = kwds["verbose"]
        else:
            # we always want to pass a verbose flag to the solver server.
            kwds["verbose"] = False

        if "generateResponse" in kwds:
            generateResponse = kwds.pop("generateResponse")
        else:
            generateResponse = True

        task = pyutilib.pyro.Task(data=kwds,
                                  id=ah.id,
                                  generateResponse=generateResponse)

        self.client.add_task(task, verbose=self._verbose, override_type=name)

        # only populate the action_handle-to-task dictionary is a
        # response is expected.
        if generateResponse is True:
            self._ah[task['id']] = ah

        return ah

    def _perform_wait_any(self):
        """
        Perform the wait_any operation.  This method returns an
        ActionHandle with the results of waiting.  If None is returned
        then the ActionManager assumes that it can call this method
        again.  Note that an ActionHandle can be returned with a dummy
        value, to indicate an error.
        """

        if len(self._results_waiting) > 0:
            return self._extract_result()

        elif len(self.client.queues_with_results()) > 0:

            all_results = self.client.get_results_all_queues()

            for task in all_results:
                self._results_waiting.append(task)
        else:

          # if the queues are all empty, wait some time for things to
          # fill up.  constantly pinging dispatch servers wastes their
          # time, and inhibits task worker communication. the good
          # thing about queues_to_check is that it simultaneously
          # grabs information for any queues with results => one
          # client query can yield many results.

          # TBD: We really need to parameterize the time-out value,
          #      but it isn't clear how to propagate this though the
          #      solver manager interface layers.
          time.sleep(0.01)

    def acquire_workers(self,num,timeout):

        """
        print("Attempting to acquire "+str(num)+" workers")
        if timeout is None:
            print("Timeout has been disabled")
        else:
            print("Automatic timeout in "+str(timeout)+" seconds")
        """
        workers_acquired = []
        wait_start = time.time()
        while(len(workers_acquired) < num):
            data = {'action':'acknowledge'}
            task = pyutilib.pyro.Task(data=data, generateResponse=True)
            self.client.add_task(task,
                                 verbose=self._verbose,
                                 override_type="phpyro_worker_idle")
            task = None
            while task is None:
                task = self.client.get_result(override_type='phpyro_worker_idle',
                                              block=True,
                                              timeout=0.1)
                if task is not None:
                    ####six.print_('.',end="")
                    workername = task['result']
                    workers_acquired.append(task['result'])
                    # Make sure this worker doesn't have any requests
                    # under its name from a previous run
                    self.client.clear_queue(override_type=workername)
                else:
                    if self._verbose:
                        six.print_('x',end="")
                        sys.stdout.flush()
                        time.sleep(1)
                if (timeout is not None) and \
                   ((time.time()-wait_start) > timeout):
                    break
            if (timeout is not None) and \
               ((time.time()-wait_start) > timeout):
                break

        """
        print("")
        if len(workers_acquired) < num:
            print("Wait time limit exceeded...")
            if len(workers_acquired) == 0:
                raise RuntimeError("No workers found within time limit!")
            print("Proceeding with "+str(len(workers_acquired))+" workers")
        else:
            print("All Workers acquired")
        """
        self.worker_pool.extend(workers_acquired)

    def release_workers(self):

        ###print("Releasing PHPyro workers")
        # tell workers to become idle
        action_handles = []
        for worker in self.worker_pool:
            data = {'name':worker,'action':'go_idle'}
            task = pyutilib.pyro.Task(data=data, generateResponse=False)
            self.client.add_task(task,
                                 verbose=self._verbose,
                                 override_type=worker)
        self.worker_pool = []

if pyutilib.pyro.Pyro is None:
    SolverManagerFactory.deactivate('phpyro')
