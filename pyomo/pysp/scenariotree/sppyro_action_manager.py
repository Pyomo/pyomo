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
from pyutilib.pyro import using_pyro3, using_pyro4, TaskProcessingError
from pyutilib.pyro import Pyro as _pyro
from pyutilib.pyro.util import _connection_problem
from pyomo.opt.parallel.manager import \
    (AsynchronousActionManager,
     ActionManagerError,
     ActionStatus,
     ActionHandle)

try:
    from collections import OrderedDict
except ImportError:                         #pragma:nocover
    from ordereddict import OrderedDict

import six
from six import advance_iterator, iteritems, itervalues
from six.moves import xrange

#
# a specialized asynchronous action manager for the SPPyroScenarioTreeServer
#

class SPPyroAsyncActionManager(AsynchronousActionManager):

    def __init__(self, host=None, verbose=0):

        # the SPPyroScenarioTreeServer objects associated with this
        # manager
        self.server_pool = []
        self.host = host
        self._verbose = verbose
        self._bulk_transmit_mode = False
        self._bulk_task_dict = {}
        self._dispatcher_name_to_client = {}
        self._server_name_to_dispatcher_name = {}
        self._dispatcher_name_to_server_names = {}
        self._dispatcher_proxies = {}
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
        self._results_waiting = OrderedDict()
        self._last_extracted_ah = None
        AsynchronousActionManager.__init__(self)

    def clear(self):
        """
        Clear manager state
        """
        AsynchronousActionManager.clear(self)
        self.close()

    def close(self):
        """Close the manager."""
        if len(self.server_pool):
            self.release_servers()
        self._ah = {}
        if self.queued_action_counter > 0:
            print("WARNING: SPPyroAsyncActionManager is closing with %s tasks "
                  "still in queue." % (self.queued_action_counter))
        if len(self._results_waiting):
            print("WARNING: SPPyroAsyncActionManager is closing with %s local "
                  "results waiting to be processed." % (len(self._results_waiting)))
        if len(self._bulk_task_dict):
            print("WARNING: SPPyroAsyncActionManager is closing with %s local "
                  "tasks waiting to be transmitted." % (len(self._bulk_task_dict)))
        self._results_waiting = OrderedDict()
        self._bulk_transmit_mode = False
        self._bulk_task_dict = {}

    def begin_bulk(self):
        self._bulk_transmit_mode = True

    def end_bulk(self):
        self._bulk_transmit_mode = False
        if len(self._bulk_task_dict):
            for dispatcher_name in self._bulk_task_dict:
                client = self._dispatcher_name_to_client[dispatcher_name]
                client.add_tasks(self._bulk_task_dict[dispatcher_name],
                                 verbose=self._verbose > 1)
        self._bulk_task_dict = {}

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

        dispatcher_name = self._server_name_to_dispatcher_name[queue_name]
        if self._bulk_transmit_mode:
            if dispatcher_name not in self._bulk_task_dict:
                self._bulk_task_dict[dispatcher_name] = dict()
            if queue_name not in self._bulk_task_dict[dispatcher_name]:
                self._bulk_task_dict[dispatcher_name][queue_name] = []
            self._bulk_task_dict[dispatcher_name][queue_name].append(task)
        else:
            client = self._dispatcher_name_to_client[dispatcher_name]
            client.add_task(task,
                            verbose=self._verbose > 1,
                            override_type=queue_name)

        # only populate the action_handle-to-task dictionary is a
        # response is expected.
        if generate_response:
            self._ah[task['id']] = ah
        else:
            self.queued_action_counter -= 1

        return ah

    def wait_all(self, *args):
        """
        Wait for all actions to complete.  The arguments to this method
        are expected to be ActionHandle objects or iterators that return
        ActionHandle objects.  If no arguments are provided, then this
        method will terminate after all queued actions are complete.
        """
        #
        # Collect event handlers from the arguments
        #
        ahs = set()
        if len(args) > 0:
            for item in args:
                if type(item) is ActionHandle:
                    ahs.add(item)
                elif type(item) in (list, tuple, dict):
                    for ah in item:
                        if type(ah) is not ActionHandle:     #pragma:nocover
                            raise ActionManagerError("Bad argument type %s" % str(ah))
                        ahs.add(ah)
                else:                       #pragma:nocover
                    raise ActionManagerError("Bad argument type %s" % str(item))
        results = {}
        if len(ahs):
            while len(ahs) > 0:
                ahs_waiting = [ah for ah in ahs if ah.id in self._results_waiting]
                for ah in ahs_waiting:
                    self._extract_result(ah=ah)
                    results[ah] = self.get_results(ah)
                    self.queued_action_counter -= 1
                    ahs.discard(ah)
                if len(ahs):
                    self._download_results()
        else:
            #
            # Iterate until all ah's have completed
            #
            while self.queued_action_counter > 0:
                ah = self.wait_any()
                results[ah] = self.get_results(ah)
        return results

    def wait_any(self):
        """
        Wait for any action to complete, and return the
        corresponding ActionHandle.
        """
        ah = self._perform_wait_any()
        self.queued_action_counter -= 1
        self.event_handle[ah.id].update(ah)
        return ah

    def wait_for(self, ah):
        """
        Wait for the specified action to complete.
        """
        while True:
            if ah.id in self._results_waiting:
                self._extract_result(ah=ah)
                break
            else:
                self._download_results()
        self.queued_action_counter -= 1
        self.event_handle[ah.id].update(ah)
        return self.get_results(ah)

    def return_to_queue(self, ah):
        self.queued_action_counter += 1
        self._results_waiting[ah.id] = None

    #
    # Perform the wait_any operation. This method returns an
    # ActionHandle with the results of waiting. If None is returned
    # then the ActionManager assumes that it can call this method
    # again. Note that an ActionHandle can be returned with a dummy
    # value, to indicate an error.
    #
    def _perform_wait_any(self):

        while len(self._results_waiting) == 0:
            self._download_results()
        ah = self._extract_result()
        if ah == self._last_extracted_ah:
            self._download_results()
            self.return_to_queue(ah)
        return self._extract_result()

    def _download_results(self):

        found_results = False
        for client in itervalues(self._dispatcher_name_to_client):
            results = client.get_results_all_queues()
            if len(results) > 0:
                found_results = True
                for task in results:
                    if type(task['result']) is TaskProcessingError:
                        raise RuntimeError(
                            "SPPyroScenarioTreeServer reported a processing error "
                            "for task with id=%s. Reason: \n%s"
                            % (task['id'], task['result'].message))
                    else:
                        self._results_waiting[task['id']] = task

        if not found_results:
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

    #
    # a utility to extract a single result from the _results_waiting
    # list.
    #

    def _extract_result(self, ah=None):

        if len(self._results_waiting) == 0:
            raise RuntimeError(
                "There are no results available for "
                "extraction from the SPPyroAsyncActionManager "
                "- call to _extract_result is not valid.")

        if ah is not None:
            task = self._results_waiting.pop(ah.id)
        else:
            ah, task = self._results_waiting.popitem(last=False)
        if task is None:
            assert ah.id in self.results
            return ah
        elif task['id'] in self._ah:
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

    def acquire_servers(self, servers_requested, timeout=None):

        if self._verbose:
            print("Attempting to acquire %s scenario tree servers"
                  % (servers_requested))
            if timeout is None:
                print("Timeout has been disabled")
            else:
                print("Automatic timeout in %s seconds" % (timeout))

        assert len(self.server_pool) == 0
        assert len(self._dispatcher_name_to_client) == 0
        assert len(self._server_name_to_dispatcher_name) == 0
        assert len(self._dispatcher_name_to_server_names) == 0
        assert len(self._dispatcher_proxies) == 0
        #
        # This process consists of the following steps:
        #
        # (1) Obtain the list of dispatchers from the nameserver
        # (2) Acquire all workers currently registered on each dispatcher
        # (3) Repeat (1) and (2) until we reach the timeout (if it exists)
        #     or until we obtain the number of servers requested
        # (4) Release any servers we don't need on dispatchers
        #
        wait_start = time.time()
        dispatcher_registered_servers = defaultdict(list)
        dispatcher_servers_to_release = defaultdict(list)
        dispatcher_proxies = {}
        servers_acquired = 0
        while servers_acquired < servers_requested:

            if (timeout is not None) and \
               ((time.time()-wait_start) > timeout):
                print("Timeout reached before %s servers could be acquired. "
                      "Proceeding with %s servers."
                      % (servers_requested, servers_acquired))
                break

            try:
                dispatchers = pyutilib.pyro.util.get_dispatchers(
                    host=self.host,
                    caller_name="Client")
            except _connection_problem:
                print("Failed to obtain one or more dispatchers from nameserver")
                continue
            for (name, uri) in dispatchers:
                dispatcher = None
                server_names = None
                if name not in dispatcher_proxies:
                    # connect to the dispatcher
                    if using_pyro3:
                        dispatcher = _pyro.core.getProxyForURI(uri)
                    else:
                        dispatcher = _pyro.Proxy(uri)
                        dispatcher._pyroTimeout = 10
                    try:
                        server_names = dispatcher.acquire_available_workers()
                    except _connection_problem:
                        if using_pyro4:
                            dispatcher._pyroRelease()
                        else:
                            dispatcher._release()
                        continue
                    dispatcher_proxies[name] = dispatcher
                    if using_pyro4:
                        dispatcher._pyroTimeout = None
                else:
                    dispatcher = dispatcher_proxies[name]
                    server_names = dispatcher.acquire_available_workers()

                # collect the list of registered PySP workers
                servers_to_release = dispatcher_servers_to_release[name]
                registered_servers = dispatcher_registered_servers[name]
                for server_name in server_names:
                    if server_name.startswith("PySPWorker_"):
                        registered_servers.append(server_name)
                    else:
                        servers_to_release.append(server_name)

                if (timeout is not None) and \
                   ((time.time()-wait_start) > timeout):
                    break

            servers_acquired = sum(len(_serverlist) for _serverlist
                                   in itervalues(dispatcher_registered_servers))

        for name, servers_to_release in iteritems(dispatcher_servers_to_release):
            dispatcher_proxies[name].release_acquired_workers(servers_to_release)
        del dispatcher_servers_to_release

        #
        # Decide which servers we will utilize and do this in such a way
        # as to balance the workload we place on each dispatcher
        #
        server_to_dispatcher_map = {}
        dispatcher_servers_utilized = defaultdict(list)
        servers_utilized = 0
        dispatcher_names = itertools.cycle(dispatcher_registered_servers.keys())
        while servers_utilized < min(servers_requested, servers_acquired):
            name = advance_iterator(dispatcher_names)
            if len(dispatcher_registered_servers[name]) > 0:
                servername = dispatcher_registered_servers[name].pop()
                server_to_dispatcher_map[servername] = name
                dispatcher_servers_utilized[name].append(servername)
                servers_utilized += 1

        # copy the keys as we are modifying this list
        for name in list(dispatcher_proxies.keys()):
            dispatcher = dispatcher_proxies[name]
            servers = dispatcher_servers_utilized[name]
            if len(dispatcher_registered_servers[name]) > 0:
                # release any servers we do not need
                dispatcher.release_acquired_workers(
                    dispatcher_registered_servers[name])
            if len(servers) == 0:
                # release the proxy to this dispatcher,
                # we don't need it
                if using_pyro4:
                    dispatcher._pyroRelease()
                else:
                    dispatcher._release()
                del dispatcher_proxies[name]
            else:
                client = pyutilib.pyro.Client(host=self.host,
                                              dispatcher=dispatcher)
                self._dispatcher_name_to_client[name] = client
                self._dispatcher_name_to_server_names[name] = servers
                for servername in servers:
                    self._server_name_to_dispatcher_name[servername] = name
                    self.server_pool.append(servername)
        self._dispatcher_proxies = dispatcher_proxies

    def release_servers(self):

        if self._verbose:
            print("Releasing scenario tree servers")

        for name in self._dispatcher_proxies:
            dispatcher = self._dispatcher_proxies[name]
            servers = self._dispatcher_name_to_server_names[name]
            # tell dispatcher that the servers we have acquired are no
            # longer needed
            dispatcher.release_acquired_workers(servers)
            # the client will release the dispatcher proxy
            self._dispatcher_name_to_client[name].close()

        self.server_pool = []
        self._dispatcher_name_to_client = {}
        self._server_name_to_dispatcher_name = {}
        self._dispatcher_name_to_server_names = {}
        self._dispatcher_proxies = {}
