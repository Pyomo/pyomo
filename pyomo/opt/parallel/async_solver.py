#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________


__all__ = ['AsynchronousSolverManager', 'SolverManagerFactory']

from pyomo.util.plugin import *
from pyomo.opt.parallel.manager import *


class ISolverManager(Interface):
    pass


SolverManagerFactory = CreatePluginFactory(ISolverManager)


class AsynchronousSolverManager(AsynchronousActionManager, Plugin):

    implements(ISolverManager)

    def __init__(self, **kwds):
        AsynchronousActionManager.__init__(self)
        Plugin.__init__(self, **kwds)

    def solve(self, *args, **kwds):
        return self.execute(*args, **kwds)

    def solve_all(self, solver, instances, **kwds):
        """
        A simple utility to apply a solver to a list of problem instances.
        The solver is applied asynchronously and a barrier synchronization
        is performed to finalize all results.  All keywords are passed
        to each invocation of the solver, and the results are loaded
        into each instance.

        The solver manager manages this process, and the solver is used to
        manage each invocation of the solver.
        """
        action_handles = []
        instance_map = {}
        kwds['opt'] = solver
        for instance in instances:
            action_handle = self.queue(instance, **kwds)
            instance_map[action_handle] = instance
            action_handles.append(action_handle)
        self.wait_all(action_handles)
        for action_handle in action_handles:
            results = self.get_results(action_handle)
            instance_map[action_handle].solutions.load_from(results)

