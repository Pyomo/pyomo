#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


__all__ = ['AsynchronousSolverManager', 'SolverManagerFactory']

from pyomo.common import Factory
from pyomo.opt.parallel.manager import AsynchronousActionManager



SolverManagerFactory = Factory('solver manager')


class AsynchronousSolverManager(AsynchronousActionManager):

    def __init__(self, **kwds):
        AsynchronousActionManager.__init__(self)

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

    #
    # Support "with" statements. Forgetting to call deactivate
    # on Plugins is a common source of memory leaks
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

