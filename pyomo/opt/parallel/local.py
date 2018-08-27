#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


__all__ = ()

import time

try:
    from collections import OrderedDict
except ImportError:                         #pragma:nocover
    from ordereddict import OrderedDict

import pyomo.opt
from pyomo.opt.parallel.manager import (ActionManagerError,
                                        ActionStatus,
                                        ActionHandle)
from pyomo.opt.parallel.async_solver import AsynchronousSolverManager, SolverManagerFactory

import six
from six import string_types


@SolverManagerFactory.register("serial", doc="Synchronously execute solvers locally")
class SolverManager_Serial(AsynchronousSolverManager):

    def clear(self):
        """
        Clear manager state
        """
        super(SolverManager_Serial, self).clear()
        self.results = OrderedDict()

    def _perform_queue(self, ah, *args, **kwds):
        """
        Perform the queue operation.  This method returns the ActionHandle,
        and the ActionHandle status indicates whether the queue was successful.
        """

        opt = kwds.pop('solver', kwds.pop('opt', None))
        if opt is None:
            raise ActionManagerError(
                "No solver passed to %s, use keyword option 'solver'"
                % (type(self).__name__) )

        time_start = time.time()
        if isinstance(opt, string_types):
            with pyomo.opt.SolverFactory(opt) as _opt:
                results = _opt.solve(*args, **kwds)
        else:
            results = opt.solve(*args, **kwds)
        results.pyomo_solve_time = time.time()-time_start

        self.results[ah.id] = results
        ah.status = ActionStatus.done
        self.event_handle[ah.id].update(ah)

        return ah

    def _perform_wait_any(self):
        """
        Perform the wait_any operation.  This method returns an
        ActionHandle with the results of waiting.  If None is returned
        then the ActionManager assumes that it can call this method again.
        Note that an ActionHandle can be returned with a dummy value,
        to indicate an error.
        """
        if len(self.results) > 0:
            ah_id, result = self.results.popitem(last=False)
            self.results[ah_id] = result
            return self.event_handle[ah_id]
        return ActionHandle(error=True,
                            explanation=("No queued evaluations available in "
                                         "the 'serial' solver manager, which "
                                         "executes solvers synchronously"))
