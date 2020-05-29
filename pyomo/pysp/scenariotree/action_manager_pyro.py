#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ("ScenarioTreeActionManagerPyro",)

import time
import itertools
import logging
from collections import defaultdict
import base64
try:
    import cPickle as pickle
except:
    import pickle

from pyomo.common.dependencies import attempt_import
from pyomo.opt.parallel.manager import ActionStatus
from pyomo.opt.parallel.pyro import PyroAsynchronousActionManager

pyu_pyro = attempt_import('pyutilib.pyro', alt_names=['pyu_pyro'])[0]
Pyro4 = attempt_import('Pyro4')[0]

import six
from six import advance_iterator, iteritems, itervalues

logger = logging.getLogger('pyomo.pysp')

#
# a specialized asynchronous action manager for the scenariotreeserver
#

class ScenarioTreeActionManagerPyro(PyroAsynchronousActionManager):

    def __init__(self, *args, **kwds):
        super(ScenarioTreeActionManagerPyro, self).__init__(*args, **kwds)
        # the SPPyroScenarioTreeServer objects associated with
        # this manager
        self.server_pool = []
        self._server_name_to_dispatcher_name = {}
        self._dispatcher_name_to_server_names = {}
        # tells the action manager to ignore task errors
        # (it will still report them, just take no action)
        self.ignore_task_errors = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the manager."""
        if len(self.server_pool):
            self.release_servers()
        super(ScenarioTreeActionManagerPyro, self).close()

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
                dispatchers = pyu_pyro.util.get_dispatchers(
                    host=self.host,
                    port=self.port,
                    caller_name="Client")
            except pyu_pyro.util._connection_problem:
                print("Failed to obtain one or more dispatchers from nameserver")
                continue
            for (name, uri) in dispatchers:
                dispatcher = None
                server_names = None
                if name not in dispatcher_proxies:
                    # connect to the dispatcher
                    if pyu_pyro.using_pyro3:
                        dispatcher = pyu_pyro.Pyro.core.getProxyForURI(uri)
                    else:
                        dispatcher = pyu_pyro.Pyro.Proxy(uri)
                        dispatcher._pyroTimeout = 10
                    try:
                        server_names = dispatcher.acquire_available_workers()
                    except pyu_pyro.util._connection_problem:
                        if pyu_pyro.using_pyro4:
                            dispatcher._pyroRelease()
                        else:
                            dispatcher._release()
                        continue
                    dispatcher_proxies[name] = dispatcher
                    if pyu_pyro.using_pyro4:
                        dispatcher._pyroTimeout = None
                else:
                    dispatcher = dispatcher_proxies[name]
                    server_names = dispatcher.acquire_available_workers()

                # collect the list of registered PySP workers
                servers_to_release = dispatcher_servers_to_release[name]
                registered_servers = dispatcher_registered_servers[name]
                for server_name in server_names:
                    if server_name.startswith("ScenarioTreeServerPyro_"):
                        registered_servers.append(server_name)
                    else:
                        servers_to_release.append(server_name)

                if (timeout is not None) and \
                   ((time.time()-wait_start) > timeout):
                    break

            servers_acquired = sum(len(_serverlist) for _serverlist
                                   in itervalues(dispatcher_registered_servers))
            # Don't overload the nameserver while trying to
            # collect dispatchers with registered workers.
            # If you haven't found them after the first few tries,
            # it's very likely that you are not going to.
            time.sleep(0.5)

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
        dispatcher_proxies_byURI = {}
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
                if pyu_pyro.using_pyro4:
                    dispatcher._pyroRelease()
                else:
                    dispatcher._release()
                del dispatcher_proxies[name]
            else:
                # when we initialize a client directly with a dispatcher
                # proxy it does not need to know the nameserver host or port
                client = self._create_client(dispatcher=dispatcher)
                self._dispatcher_name_to_server_names[client.URI] = servers
                dispatcher_proxies_byURI[client.URI] = dispatcher
                for servername in servers:
                    self._server_name_to_dispatcher_name[servername] = client.URI
                    self.server_pool.append(servername)
        self._dispatcher_proxies = dispatcher_proxies_byURI

    def release_servers(self):

        if self._verbose:
            print("Releasing scenario tree servers")

        for name in self._dispatcher_proxies:
            dispatcher = self._dispatcher_proxies[name]
            servers = self._dispatcher_name_to_server_names[name]
            # tell dispatcher that the servers we have acquired are no
            # longer needed
            dispatcher.release_acquired_workers(servers)

        self.server_pool = []
        self._server_name_to_dispatcher_name = {}
        self._dispatcher_name_to_server_names = {}

    #
    # Abstract Methods
    #

    def _get_dispatcher_name(self, queue_name):
        return self._server_name_to_dispatcher_name[queue_name]

    def _get_task_data(self, ah, **kwds):
        # Doing this serves two purposes:
        #   (1) It avoids issues with transmitting user-defined
        #       types over the wire that the dispatcher is not
        #       aware of (and therefore unable to de-serialize)
        #   (2) It improves performance on the dispatcher
        #       because de-serialization (and
        #       re-serialization) of raw bytes should be
        #       about as trivial as you can get for any
        #       serializer that Pyro/Pyro4 happens to be
        #       configured with (pickle is the fastest,
        #       but that is not the default in Pyro4 for
        #       security reasons).
        return pickle.dumps(kwds)

    def _download_results(self):

        found_results = False
        for client in itervalues(self._dispatcher_name_to_client):
            if len(self._dispatcher_name_to_client) == 1:
                # if there is a single dispatcher then we can do
                # a more efficient blocking call
                results = client.get_results(override_type=client.CLIENTNAME,
                                             block=True,
                                             timeout=None)
            else:
                results = client.get_results(override_type=client.CLIENTNAME,
                                             block=False)
            if len(results) > 0:
                found_results = True
                for task in results:
                    self.queued_action_counter -= 1

                    # The only reason we are go through this much
                    # effort to deal with the serpent serializer
                    # is because it is the default in Pyro4.
                    if pyu_pyro.using_pyro4 and \
                       (Pyro4.config.SERIALIZER == 'serpent'):
                        if six.PY3:
                            assert type(task['result']) is dict
                            assert task['result']['encoding'] == 'base64'
                            task['result'] = base64.b64decode(task['result']['data'])
                        else:
                            assert type(task['result']) is unicode
                            task['result'] = str(task['result'])
                    # ** See note in _get_task_data about why we pickle
                    #    all communication
                    task['result'] = pickle.loads(task['result'])

                    ah = self.event_handle.get(task['id'], None)
                    if ah is None:
                        # if we are here, this is really bad news!
                        raise RuntimeError(
                            "The %s found results for task with id=%s"
                            " - but no corresponding action handle "
                            "could be located! Showing task result "
                            "below:\n%s" % (type(self).__name__,
                                            task['id'],
                                            task.get('result', None)))
                    if type(task['result']) is pyu_pyro.TaskProcessingError:
                        ah.status = ActionStatus.error
                        self.event_handle[ah.id].update(ah)
                        msg = ("ScenarioTreeServer reported a processing "
                               "error for task with id=%s. Reason: \n%s"
                               % (task['id'], task['result'].args[0]))
                        if not self.ignore_task_errors:
                            raise RuntimeError(msg)
                        elif self.ignore_task_errors == 1:
                            logger.warning(msg)
                        # any value other than 0 or 1 will
                        # silently ignore task errors
                    else:
                        ah.status = ActionStatus.done
                        self.event_handle[ah.id].update(ah)
                        self.results[ah.id] = task['result']

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
