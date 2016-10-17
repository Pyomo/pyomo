#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("SPSolverResults",
           "SPSolver",
           "SPSolverFactory")

import time
import logging

from pyomo.opt import undefined

from pyomo.pysp.util.configured_object import PySPConfiguredObject

logger = logging.getLogger('pyomo.pysp')

class SPSolverResults(object):

    def __init__(self):
        self.objective = undefined
        self.objective_interval = undefined
        self.bound = undefined
        self.bound_interval = undefined
        self.solver_name = undefined
        self.solver_time = undefined
        self.solver_status = undefined

    def __str__(self):
        attrs = vars(self)
        order = ['objective',
                 'objective_interval',
                 'bound',
                 'bound_interval',
                 'solver_name',
                 'solver_status',
                 'solver_time']
        # sort by order above, then by name
        names = sorted(list(attrs.keys()),
                       key=lambda x: order.index(x) if (x in order) else x)
        out =  "SPSolverResults:\n"
        for name in names:
            out += "  %s: %s\n" % (name, attrs[name])
        return out

class SPSolver(PySPConfiguredObject):

    def __init__(self, *args, **kwds):
        super(SPSolver, self).__init__(*args, **kwds)
        self._name = None

    def solve(self, *args, **kwds):
        raise NotImplementedError

    @property
    def options(self):
        return self._options

    @property
    def name(self):
        self._name

    def solve(self, sp, *args, **kwds):

        start = time.time()

        scenario_tree = sp.scenario_tree

        num_scenarios = len(scenario_tree.scenarios)
        num_stages = len(scenario_tree.stages)
        num_na_variables = 0
        num_na_continuous_variables = 0
        for stage in scenario_tree.stages[:-1]:
            for tree_node in stage.nodes:
                num_na_variables += len(tree_node._standard_variable_ids)
                for id_ in tree_node._standard_variable_ids:
                    if not tree_node.is_variable_discrete(id_):
                        num_na_continuous_variables += 1

        if kwds.get('output_solver_log', False):
            print("-"*20)
            print("Problem Statistics".center(20))
            print("-"*20)
            print("Total number of time stages...............: %10s"
                  % (num_stages))
            print("Total number of non-anticipative variables: %10s\n"
                  "                                continuous: %10s\n"
                  "                                  discrete: %10s"
                  % (num_na_variables,
                     num_na_continuous_variables,
                     num_na_variables - num_na_continuous_variables))

        results = self._solve_impl(sp, *args, **kwds)

        stop = time.time()
        results.pysp_time = stop - start
        results.solver_name = self.name

        return results

    def _solve_impl(self, *args, **kwds):
        raise NotImplementedError

def SPSolverFactory(solver_name, options=None, **kwds):
    if solver_name in SPSolverFactory._registered_solvers:
        type_ = SPSolverFactory._registered_solvers[solver_name]
        if options is None:
            options = type_.register_options()
        return type_(options, **kwds)
    else:
        raise ValueError(
            "No SPSolver object has been registered with name: %s"
            % (solver_name))
SPSolverFactory._registered_solvers = {}

def _register_solver(name, type_):
    if not issubclass(type_, SPSolver):
        raise TypeError("Can not register SP solver type '%s' "
                        "because it is not derived from type '%s'"
                        % (type_, SPSolver))
    SPSolverFactory._registered_solvers[name] = type_
SPSolverFactory.register_solver = _register_solver
