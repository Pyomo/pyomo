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

import pyutilib.misc

from pyomo.opt import undefined

from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import PySPConfigBlock

logger = logging.getLogger('pyomo.pysp')

class SPSolverResults(object):

    def __init__(self):
        self.objective = undefined
        self.bound = undefined
        self.solver_name = undefined
        self.solver_time = undefined
        self.solver_status = undefined

    def __str__(self):
        attrs = vars(self)
        names = sorted(list(attrs.keys()))
        out =  "SPSolverResults:\n"
        for name in names:
            out += "  %s: %s\n" % (name, attrs[name])
        return out

class SPSolver(PySPConfiguredObject):

    def __init__(self, *args, **kwds):
        super(SPSolver, self).__init__(*args, **kwds)
        self._name = None
        self._solver_options = pyutilib.misc.Options()

    @property
    def options(self):
        return self._solver_options

    @property
    def name(self):
        return self._name

    def solve(self, sp, *args, **kwds):

        start = time.time()

        tmp_options = kwds.pop('options', None)
        orig_options = self.options
        if tmp_options is not None:
            self._solver_options = pyutilib.misc.Options()
            for key, val in orig_options.items():
                self._solver_options[key] = val
            for key, val in tmp_options.items():
                self._solver_options[key] = val

        try:
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
        finally:
            if tmp_options is not None:
                self._solver_options = orig_options

        return results

    def _solve_impl(self, *args, **kwds):
        raise NotImplementedError

def SPSolverFactory(solver_name, **kwds):
    if solver_name in SPSolverFactory._registered_solvers:
        type_ = SPSolverFactory._registered_solvers[solver_name]
        config = type_.register_options()
        for key, val in kwds.items():
            config[key] = val
        return type_(config)
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
