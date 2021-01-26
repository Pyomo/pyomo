#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ("PyroAsynchronousActionManager",)

from pyomo.common.collections import OrderedDict
from pyomo.common.dependencies import attempt_import
from pyomo.opt.parallel.manager import \
    (AsynchronousActionManager,
     ActionStatus)

pyu_pyro = attempt_import('pyutilib.pyro', alt_names=['pyu_pyro'])[0]


#
# a specialized asynchronous action manager for Pyro based managers
#

class PyroAsynchronousActionManager(AsynchronousActionManager):

    def __init__(self, host=None, port=None, verbose=0):

        self.host = host
        self.port = port
        self._verbose = verbose
        self._paused = False
        self._paused_task_dict = {}
        self._dispatcher_name_to_client = {}
        self._dispatcher_proxies = {}
        # map from task id to the corresponding action handle.
        # we only retain entries for tasks for which we expect
        # a result/response.
        self._last_extracted_ah_id = None
        super(PyroAsynchronousActionManager, self).__init__()
        # the list of cached results obtained from the dispatch server.
        # to avoid communication overhead, grab any/all results available,
        # and then cache them here - but return one-at-a-time via
        # the standard _perform_wait_any interface. the elements in this
        # list are simply tasks - at this point, we don't care about the
        # queue name associated with the task.
        self.results = OrderedDict()

    def clear(self):
        """
        Clear manager state
        """
        super(PyroAsynchronousActionManager, self).clear()
        self.results = OrderedDict()

    def close(self):
        """Close the manager."""
        if len(self.results):
            print("WARNING: %s is closing with %s local "
                  "results waiting to be processed."
                  % (type(self).__name__, len(self.results)))
        if len(self._paused_task_dict):
            print("WARNING: %s is closing with %s paused "
                  "tasks waiting to be queued."
                  % (type(self).__name__, len(self._paused_task_dict)))
        self.results = OrderedDict()
        self._paused = False
        self._paused_task_dict = {}
        for client in self._dispatcher_name_to_client.values():
            # the client will release the dispatcher proxy
            client.close()
        self._dispatcher_name_to_client = {}
        self._dispatcher_proxies = {}

    def pause(self):
        self._paused = True

    def unpause(self):
        self._paused = False
        if len(self._paused_task_dict):
            for dispatcher_name in self._paused_task_dict:
                client = self._dispatcher_name_to_client[dispatcher_name]
                client.add_tasks(self._paused_task_dict[dispatcher_name],
                                 verbose=self._verbose > 1)
        self._paused_task_dict = {}

    def get_results(self, ah):
        return self.results.pop(ah.id, None)

    def wait_all(self, *args):
        """
        Wait for all actions to complete.  The arguments to this method
        are expected to be ActionHandle objects or iterators that return
        ActionHandle objects.  If no arguments are provided, then this
        method will terminate after all queued actions are complete.
        """
        # Collect event handlers from the arguments
        ahs = self._flatten(*args)
        if len(ahs):
            while len(ahs) > 0:
                ahs.difference_update([ah for ah in ahs if ah.id in self.results])
                if len(ahs):
                    self._download_results()
        else:
            while self.queued_action_counter > 0:
                self._download_results()

    def wait_any(self, *args):
        # Collect event handlers from the arguments
        ahs = self._flatten(*args)
        if len(ahs):
            while (1):
                for ah in ahs:
                    if ah.id in self.results:
                        return ah
                self._download_results()
        else:
            while len(self.results) == 0:
                self._download_results()
            ah_id, result = self.results.popitem(last=False)
            if ah_id == self._last_extracted_ah_id:
                self._last_extracted_ah_id = ah_id
                self._download_results()
                self.results[ah_id] = result
                ah_id, result = self.results.popitem(last=False)
            self.results[ah_id] = result
            return self.event_handle[ah_id]

    def wait_for(self, ah):
        """
        Wait for the specified action to complete.
        """
        while (1):
            if ah.id in self.results:
                break
            else:
                self._download_results()
        return self.get_results(ah)

    def _create_client(self, dispatcher=None):
        if dispatcher is None:
            client = pyu_pyro.Client(host=self.host, port=self.port)
        else:
            client = pyu_pyro.Client(dispatcher=dispatcher)
        if client.URI in self._dispatcher_name_to_client:
            self._dispatcher_name_to_client[client.URI].close()
        self._dispatcher_name_to_client[client.URI] = client
        return client

    #
    # Perform the queue operation. This method returns the
    # ActionHandle, and the ActionHandle status indicates whether
    # the queue was successful.
    #
    def _perform_queue(self,
                       ah,
                       *args,
                       **kwds):

        queue_name = kwds.pop('queue_name', None)
        generate_response = kwds.pop('generate_response', True)

        dispatcher_name = self._get_dispatcher_name(queue_name)
        task_data = self._get_task_data(ah, *args, **kwds)
        task = pyu_pyro.Task(data=task_data,
                            id=ah.id,
                            generateResponse=generate_response)

        if self._paused:
            if dispatcher_name not in self._paused_task_dict:
                self._paused_task_dict[dispatcher_name] = dict()
            if queue_name not in self._paused_task_dict[dispatcher_name]:
                self._paused_task_dict[dispatcher_name][queue_name] = []
            self._paused_task_dict[dispatcher_name][queue_name].append(task)
        else:
            client = self._dispatcher_name_to_client[dispatcher_name]
            client.add_task(task,
                            verbose=self._verbose > 1,
                            override_type=queue_name)

        # only populate the action_handle-to-task dictionary is a
        # response is expected.
        if not generate_response:
            ah.status = ActionStatus.done
            self.event_handle[ah.id].update(ah)
            self.queued_action_counter -= 1

        return ah

    #
    # Abstract Methods
    #

    def _get_dispatcher_name(self, queue_name):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _get_task_data(self, ah, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _download_results(self):
        raise NotImplementedError(type(self).__name__+": This method is abstract")
