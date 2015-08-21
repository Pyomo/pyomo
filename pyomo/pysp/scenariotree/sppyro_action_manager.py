#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("SPPyroAsyncActionManager",)

import sys
import time
import itertools
from collections import defaultdict

import pyutilib.pyro
from pyutilib.pyro import using_pyro3, using_pyro4
from pyutilib.pyro import Pyro as _pyro
from pyomo.opt.parallel.manager \
    import AsynchronousActionManager, ActionStatus
from pyomo.pysp.scenariotree.scenariotreeserverutils \
    import SPPyroScenarioTreeServer_ProcessTaskError

import six
from six import advance_iterator, iteritems, itervalues
from six.moves import xrange

_connection_problem = None
if using_pyro3:
    _connection_problem = _pyro.errors.ConnectionDeniedError
elif using_pyro4:
    _connection_problem = _pyro.errors.TimeoutError

#
# a specialized asynchronous action manager for the SPPyroScenarioTreeServer
#

class SPPyroAsyncActionManager(AsynchronousActionManager):

    def __init__(self, host=None, verbose=0):

        # the SPPyroScenarioTreeServer objects associated with this manager
        self.server_pool = []
        self.host = host
        self._verbose = verbose
        self._bulk_transmit_mode = False
        self._bulk_task_dict = {}
        self.client = None
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
        AsynchronousActionManager.__init__(self)

    def clear(self):
        """
        Clear manager state
        """
        AsynchronousActionManager.clear(self)
        self.close()
        # the client-side interface to the dispatch server.
        self.client = pyutilib.pyro.Client(host=self.host)

    def close(self):
        """Close the manager."""
        if len(self.server_pool):
            self.release_servers()
        if self.client is not None:
            self.client.close()
            self.client = None
        self._ah = {}
        if len(self._results_waiting):
            print("WARNING: SPPyroAsyncActionManager is closing with local "
                  "results waiting to be processed.")
        if len(self._bulk_task_dict):
            print("WARNING: SPPyroAsyncActionManager is closing with local "
                  "tasks waiting to be transmitted.")
        self._results_waiting = []
        self._bulk_transmit_mode = False
        self._bulk_task_dict = {}

    def begin_bulk(self):
        self._bulk_transmit_mode = True

    def end_bulk(self):
        self._bulk_transmit_mode = False
        if len(self._bulk_task_dict):
            client_verbose = False
            if self._verbose > 1:
                client_verbose = True
            self.client.add_tasks(self._bulk_task_dict,
                                  verbose=client_verbose)
            self._bulk_task_dict = {}

    #
    # a utility to extract a single result from the _results_waiting
    # list.
    #

    def _extract_result(self):

        if len(self._results_waiting) == 0:
            raise RuntimeError(
                "There are no results available for "
                "extraction from the SPPyroAsyncActionManager "
                "- call to _extract_result is not valid.")

        task = self._results_waiting.pop(0)
        if (type(task['result']) is tuple) and \
           (len(task['result']) == 2) and \
           (task['result'][0].startswith(
               SPPyroScenarioTreeServer_ProcessTaskError)):
            raise RuntimeError(
                "SPPyroScenarioTreeServer reported a processing error for task "
                "with id=%s. Reason: \n%s" % (task['id'], task['result'][1]))
        if task['id'] in self._ah:
            ah = self._ah[task['id']]
            self._ah[task['id']] = None
            ah.status = ActionStatus.done
            self.results[ah.id] = task['result']
            return ah
        else:
            # if we are here, this is really bad news!
            raise RuntimeError(
                "The SPPyroAsyncActionManager found "
                "results for task with id="+str(task['id'])+
                " - but no corresponding action handle "
                "could be located!")

    #
    # Perform the queue operation. This method returns the
    # ActionHandle, and the ActionHandle status indicates whether
    # the queue was successful.
    #
    def _perform_queue(self,
                       ah,
                       queue_name,
                       generate_response=True,
                       **kwds):

        task = pyutilib.pyro.Task(data=kwds,
                                  id=ah.id,
                                  generateResponse=generate_response)

        if self._bulk_transmit_mode:
            if queue_name not in self._bulk_task_dict:
                self._bulk_task_dict[queue_name] = []
            self._bulk_task_dict[queue_name].append(task)
        else:
            client_verbose = False
            if self._verbose > 1:
                client_verbose = True
            self.client.add_task(task,
                                 verbose=client_verbose,
                                 override_type=queue_name)

        # only populate the action_handle-to-task dictionary is a
        # response is expected.
        if generate_response:
            self._ah[task['id']] = ah

        return ah

    #
    # Perform the wait_any operation. This method returns an
    # ActionHandle with the results of waiting. If None is returned
    # then the ActionManager assumes that it can call this method
    # again. Note that an ActionHandle can be returned with a dummy
    # value, to indicate an error.
    #
    def _perform_wait_any(self):

        if len(self._results_waiting) > 0:
            return self._extract_result()

        all_results = self.client.get_results_all_queues()

        if len(all_results) > 0:
            for task in all_results:
                self._results_waiting.append(task)
        else:

          # If the queues are all empty, wait some time for things to
          # fill up. Constantly pinging dispatch servers wastes their
          # time, and inhibits task server communication. The good
          # thing about queues_to_check is that it simultaneously
          # grabs information for any queues with results => one
          # client query can yield many results.

          # TBD: We really need to parameterize the time-out value,
          #      but it isn't clear how to propagate this though the
          #      solver manager interface layers.
          time.sleep(0.01)

    def acquire_servers(self, num, timeout):

        if self._verbose:
            print("Attempting to acquire "+str(num)+" scenario tree servers")
            if timeout is None:
                print("Timeout has been disabled")
            else:
                print("Automatic timeout in "+str(timeout)+" seconds")

        client_verbose = False
        if self._verbose > 1:
            client_verbose = True

        servers_acquired = []
        wait_start = time.time()
        while(len(servers_acquired) < num):
            data = {'action': 'SPPyroScenarioTreeServer_acknowledge'}
            task = pyutilib.pyro.Task(data=data,
                                      id=float('-inf'),
                                      generateResponse=True)
            self.client.add_task(task,
                                 verbose=client_verbose,
                                 override_type="sppyro_server_idle")
            task = None
            while task is None:
                task = self.client.get_result(override_type='sppyro_server_idle',
                                              block=True,
                                              timeout=0.1)
                if task is not None:
                    if client_verbose:
                        six.print_('.',end="")
                    servername = task['result']
                    servers_acquired.append(task['result'])
                    # Make sure this server doesn't have any requests
                    # under its name from a previous run
                    self.client.clear_queue(override_type=servername)
                else:
                    if client_verbose:
                        six.print_('x',end="")
                        sys.stdout.flush()
                        time.sleep(1)
                if (timeout is not None) and \
                   ((time.time()-wait_start) > timeout):
                    break
            if (timeout is not None) and \
               ((time.time()-wait_start) > timeout):
                break

        if self._verbose:
            print("")
            if len(servers_acquired) < num:
                print("Wait time limit exceeded...")
                if len(servers_acquired) == 0:
                    raise RuntimeError(
                        "No scenario tree servers found within time limit!")
                print("Proceeding with %s scenario tree servers"
                      % (len(servers_acquired)))
            else:
                print("All scenario tree servers acquired")

        self.server_pool.extend(servers_acquired)

    def release_servers(self):

        if self._verbose:
            print("Releasing scenario tree servers")

        client_verbose = False
        if self._verbose > 1:
            client_verbose = True

        # tell servers to become idle
        action_handles = []
        data = {'action': 'SPPyroScenarioTreeServer_idle'}
        for server_name in self.server_pool:
            task = pyutilib.pyro.Task(data=data,
                                      id=float('inf'),
                                      generateResponse=False)
            self.client.add_task(task,
                                 verbose=client_verbose,
                                 override_type=server_name)
        self.server_pool = []
