#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________


__all__ = []

from pyomo.util.plugin import alias

import pyomo.opt
from pyomo.opt.parallel.manager import *
from pyomo.opt.parallel.async_solver import *

import six
using_py3 = six.PY3

class SolverManager_Serial(AsynchronousSolverManager):

    alias("serial", doc="Synchronously execute solvers locally")

    def clear(self):
        """
        Clear manager state
        """
        AsynchronousSolverManager.clear(self)
        self._ah_list = []
        self._opt = None

    def _perform_queue(self, ah, *args, **kwds):
        """
        Perform the queue operation.  This method returns the ActionHandle,
        and the ActionHandle status indicates whether the queue was successful.
        """
        if 'opt' in kwds:
            self._opt = kwds['opt']
            del kwds['opt']
        if self._opt is None:
            raise ActionManagerError("Undefined solver")

        if using_py3:
            if isinstance(self._opt, str):
                solver = pyomo.opt.SolverFactory(self._opt)
            else:
                solver = self._opt
        else:
            if isinstance(self._opt, basestring):
                solver = pyomo.opt.SolverFactory(self._opt)
            else:
                solver = self._opt
        self.results[ah.id] = solver.solve(*args, **kwds)
        ah.status = ActionStatus.done
        self._ah_list.append(ah)
        return ah

    def _perform_wait_any(self):
        """
        Perform the wait_any operation.  This method returns an
        ActionHandle with the results of waiting.  If None is returned
        then the ActionManager assumes that it can call this method again.
        Note that an ActionHandle can be returned with a dummy value,
        to indicate an error.
        """
        if len(self._ah_list) > 0:
            return self._ah_list.pop()
        return ActionHandle(error=True, explanation="No queued evaluations available in the 'serial' solver manager, which executes solvers synchronously")
