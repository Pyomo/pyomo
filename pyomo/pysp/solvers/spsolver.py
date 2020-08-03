#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ("SPSolverResults",
           "SPSolver",
           "SPSolverFactory")

import sys
import time
import logging

from pyomo.opt import UndefinedData
from pyomo.core import ComponentUID
from pyomo.pysp.embeddedsp import EmbeddedSP

from six import StringIO

logger = logging.getLogger('pyomo.pysp')

# TODO:
# Things to test for particular solvers:
#   - serial vs pyro scenario tree manager
#   - behavior on multi-stage
#   - access to xhat
#   - access to derived variables in xhat
#   - solve status (need an enum?)
#   - solutions loaded into scenario instances
#   - solution loaded into reference model
#   - results.objective
#   - results.bound
#   - ddsip validate problem (like in smpsutils)

class SPSolverStatus(object):
    def __init__(self):
        self.name = None
        self.status = None
        self.termination_condition = None

    def pprint(self, ostream=sys.stdout):
        attrs = vars(self)
        names = sorted(list(attrs.keys()))
        first = ('name','status','termination_condition')
        for cnt, name in enumerate(first):
            names.remove(name)
            if cnt > 0:
                ostream.write('\n')
            ostream.write('%s: %s'
                          % (name, getattr(self, name)))
        for name in names:
            ostream.write('\n%s: %s'
                          % (name, getattr(self, name)))

    def __str__(self):
        tmp = StringIO()
        self.pprint(ostream=tmp)
        return tmp.getvalue()

class SPSolverResults(object):

    def __init__(self):
        self.objective = None
        self.bound = None
        self.status = None
        self.xhat_loaded = False
        self.xhat = None
        self.solver = SPSolverStatus()

    def pprint(self, ostream=sys.stdout):
        attrs = vars(self)
        names = sorted(list(attrs.keys()))
        first = ('status','objective','bound','xhat_loaded')
        for name in first:
            names.remove(name)
            ostream.write('%s: %s\n'
                          % (name, getattr(self, name)))
        names.remove('solver')
        for name in names:
            ostream.write('%s: %s\n'
                          % (name, getattr(self, name)))
        tmp = StringIO()
        self.solver.pprint(ostream=tmp)
        ostream.write("solver:")
        for line in tmp.getvalue().splitlines():
            ostream.write("\n  "+line)

    def __str__(self):
        tmp = StringIO()
        self.pprint(ostream=tmp)
        return tmp.getvalue()

class SPSolver(object):

    @property
    def set_options_to_default(self):
        """Reset all options on the solver object to their
        default value"""
        raise NotImplementedError                  #pragma:nocover

    @property
    def options(self):
        """Access the solver options"""
        raise NotImplementedError                  #pragma:nocover

    def _solve_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    @property
    def name(self):
        """The registered solver name"""
        raise NotImplementedError                  #pragma:nocover

    def solve(self, sp, *args, **kwds):
        """
        Solve a stochastic program.

        Args:
            sp: The stochastic program to solve.
            reference_model: A pyomo model with at least the
                set of non-anticipative variable objects
                that were declared on th e scenario tree of
                the stochastic program. If this keyword is
                changed from its default value of None, the
                non leaf-stage variable values in the
                solution will be stored into the variable
                objects on the reference model. Otherwise,
                the results object that is returned will
                contain a solution dictionary called 'xhat'
                that stores the solution nested by tree node
                name.
            options: Ephemeral solver options that will
                temporarily overwrite any matching options
                currently set for the solver.
            output_solver_log (bool): Stream the solver
                output during the solve.
            *args: Passed to the derived solver class
                (see the _solve_impl method).
            **kwds: Passed to the derived solver class
                (see the _solve_impl method).

        Returns: A results object with information about the solution.
        """

        start = time.time()

        reference_model = kwds.pop('reference_model', None)

        tmp_options = kwds.pop('options', None)
        orig_options = self.options
        if tmp_options is not None:
            self.set_options_to_default()
            for opt in orig_options.user_values():
                self.options[opt.name()] = opt.value(accessValue=False)
            for key, val in tmp_options.items():
                self.options[key] = val

        # reset the _userAccessed flag on all options
        # so we can verify that all set options are used
        # each time
        for key in self.options:
            self.options.get(key)._userAccessed = False

        try:
            if isinstance(sp, EmbeddedSP):
                num_scenarios = "<unknown>"
                num_stages = len(sp.time_stages)
                num_na_variables = 0
                num_na_continuous_variables = 0
                for stage in sp.time_stages[:-1]:
                    for var,derived in sp.stage_to_variables_map[stage]:
                        if not derived:
                            num_na_variables += 1
                            if var.is_continuous():
                                num_na_continuous_variables += 1
            else:
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
                print("\n")
                print("-"*20)
                print("Problem Statistics".center(20))
                print("-"*20)
                print("Total number of scenarios.................: %10s"
                      % (num_scenarios))
                if scenario_tree.contains_bundles():
                    print("Total number of scenario bundles..........: %10s"
                          % (len(scenario_tree.bundles)))
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
            results.solver.pysp_time = stop - start
            results.solver.name = self.name
            if (results.status is None) or \
               isinstance(results.status, UndefinedData):
                results.status = results.solver.termination_condition
            results.xhat_loaded = False
            if (reference_model is not None):
                # TODO: node/stage costs
                if results.xhat is not None:
                    xhat = results.xhat
                    for tree_obj_name in xhat:
                        tree_obj_solution = xhat[tree_obj_name]
                        for id_ in tree_obj_solution:
                            var = ComponentUID(id_).\
                                  find_component(reference_model)
                            if not var.is_expression_type():
                                var.value = tree_obj_solution[id_]
                                var.stale = False
                    results.xhat_loaded = True
                del results.xhat

        finally:

            # warn about ignored options
            self.options.check_usage(error=False)

            # reset options (if temporary ones were provided)
            if tmp_options is not None:
                current_options = self.options
                self.set_options_to_default()
                for opt in orig_options.user_values():
                    current = current_options.get(opt.name())
                    self.options[opt.name()] = \
                        opt.value(accessValue=current._userAccessed)

        return results

def SPSolverFactory(solver_name, *args, **kwds):
    if solver_name in SPSolverFactory._registered_solvers:
        type_ = SPSolverFactory._registered_solvers[solver_name]
        return type_(*args, **kwds)
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
