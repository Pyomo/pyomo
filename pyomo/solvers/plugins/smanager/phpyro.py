#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


__all__ = ["SolverManager_PHPyro"]

import sys
import time
import itertools
from collections import defaultdict

import pyutilib.pyro
from pyutilib.pyro import using_pyro3, using_pyro4
from pyutilib.pyro import Pyro as _pyro
from pyutilib.pyro.util import _connection_problem
from pyomo.opt.parallel.manager import *
from pyomo.opt.parallel.async_solver import *

import six
from six import advance_iterator, iteritems, itervalues
from six.moves import xrange

#
# a specialized asynchronous solver manager for Progressive Hedging.
#

@SolverManagerFactory.register('phpyro',
                            doc="Specialized PH solver manager that uses pyro")
class SolverManager_PHPyro(AsynchronousSolverManager):

    def __init__(self, host=None, port=None, verbose=False):

        # the PHPyroWorker objects associated with this manager
        self.host = host
        self.port = port
        self._verbose = verbose
        self._bulk_transmit_mode = False
        self._bulk_task_dict = {}
        self.server_pool = []
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
        self._results_waiting = []
        AsynchronousActionManager.__init__(self)

    def clear(self):
        """
        Clear manager state
        """

        AsynchronousSolverManager.clear(self)

        if len(self.server_pool):
            self.release_servers()

        self._bulk_transmit_mode = False
        self._bulk_task_dict = {}

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

    def begin_bulk(self):
        self._bulk_transmit_mode = True

    def end_bulk(self):
        self._bulk_transmit_mode = False
        if len(self._bulk_task_dict):
            for dispatcher_name in self._bulk_task_dict:
                client = self._dispatcher_name_to_client[dispatcher_name]
                client.add_tasks(self._bulk_task_dict[dispatcher_name])
        self._bulk_task_dict = {}

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

        if "queue_name" not in kwds:
            raise RuntimeError("ERROR: No 'queue_name' keyword supplied to "
                               "_perform_queue method of PH pyro solver manager")

        queue_name = kwds["queue_name"]

        if "verbose" not in kwds:
            # we always want to pass a verbose flag to the solver server.
            kwds["verbose"] = False

        if "generateResponse" in kwds:
            generateResponse = kwds.pop("generateResponse")
        else:
            generateResponse = True

        task = pyutilib.pyro.Task(data=kwds,
                                  id=ah.id,
                                  generateResponse=generateResponse)

        dispatcher_name = self._server_name_to_dispatcher_name[queue_name]
        if self._bulk_transmit_mode:
            if dispatcher_name not in self._bulk_task_dict:
                self._bulk_task_dict[dispatcher_name] = dict()
            if queue_name not in self._bulk_task_dict[dispatcher_name]:
                self._bulk_task_dict[dispatcher_name][queue_name] = []
            self._bulk_task_dict[dispatcher_name][queue_name].append(task)
        else:
            client = self._dispatcher_name_to_client[dispatcher_name]
            client.add_task(task, verbose=self._verbose, override_type=queue_name)

        # only populate the action_handle-to-task dictionary is a
        # response is expected.
        if generateResponse:
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

        all_results = []
        for client in itervalues(self._dispatcher_name_to_client):
            all_results.extend(client.get_results_all_queues())

        if len(all_results) > 0:

            for task in all_results:
                self._results_waiting.append(task)
        else:

          # if the queues are all empty, wait some time for things to
          # fill up.  constantly pinging dispatch servers wastes their
          # time, and inhibits task server communication. the good
          # thing about queues_to_check is that it simultaneously
          # grabs information for any queues with results => one
          # client query can yield many results.

          # TBD: We really need to parameterize the time-out value,
          #      but it isn't clear how to propagate this though the
          #      solver manager interface layers.
          time.sleep(0.01)

    def acquire_servers(self, servers_requested, timeout=None):
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

            dispatchers = pyutilib.pyro.util.get_dispatchers(
                host=self.host,
                port=self.port,
                caller_name="Client")
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
                # when we initialize a client directly with a dispatcher
                # proxy it does not need to know the nameserver host or port
                client = pyutilib.pyro.Client(dispatcher=dispatcher)
                self._dispatcher_name_to_client[name] = client
                self._dispatcher_name_to_server_names[name] = servers
                for servername in servers:
                    self._server_name_to_dispatcher_name[servername] = name
                    self.server_pool.append(servername)
        self._dispatcher_proxies = dispatcher_proxies

    def release_servers(self, shutdown=False):

        shutdown_task = pyutilib.pyro.Task(data={'action':'shutdown'},
                                           id=float('inf'),
                                           generateResponse=False)
        # copy the keys as we are modifying this list
        for name in self._dispatcher_proxies:
            dispatcher = self._dispatcher_proxies[name]
            servers = self._dispatcher_name_to_server_names[name]
            # tell dispatcher that the servers we have acquired are
            # no longer needed
            dispatcher.release_acquired_workers(servers)
            client = self._dispatcher_name_to_client[name]
            if shutdown:
                for server_name in servers:
                    client.add_task(shutdown_task,
                                    verbose=self._verbose,
                                    override_type=server_name)
            # the client will release the dispatcher proxy
            client.close()

        self.server_pool = []
        self._dispatcher_name_to_client = {}
        self._server_name_to_dispatcher_name = {}
        self._dispatcher_name_to_server_names = {}
        self._dispatcher_proxies = {}

if pyutilib.pyro.Pyro is None:
    SolverManagerFactory.unregister('phpyro')
