#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("ScenarioTreeSolverManagerSerial",
           "ScenarioTreeSolverManagerSPPyro")

# TODO: handle pyro as the solver manager
# TODO: handle warmstart keyword

# TODO:
#self._objective_sense = \
#    self._scenario_tree._scenarios[0]._objective_sense
#assert all(_s._objective_sense == self._objective_sense
#           for _s in self._scenario_tree._scenarios)

# TODO
# Transmit persistent solver options
#solver_options = {}
#for key in self._solver.options:
#    solver_options[key]=self._solver.options[key]

# TODO
# make sure preprocessor gets constructed
# on scenario tree worker for bundles

# TODO
# make sure worker properly closes solver manager (including pyro)
# and solvers

# TODO sppyro_transmit_leaf_stage_variable_solutions
#      change name and apply same behavior to serial class

import math
import time
from collections import defaultdict

from pyutilib.misc import Bunch
from pyomo.opt import (UndefinedData,
                       undefined,
                       SolverFactory,
                       SolutionStatus,
                       TerminationCondition)
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option)
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.scenariotree.preprocessor import ScenarioTreePreprocessor
from pyomo.pysp.scenariotree.scenariotreemanager import \
    (ScenarioTreeManagerSerial,
     ScenarioTreeManagerSPPyro)

from six import itervalues, iteritems

#
# This class is designed to manage simple solve invocations
# as well as solution caching across distributed or serial
# scenario tree management
#

class _ScenarioTreeSolverWorkerImpl(PySPConfiguredObject):

    _registered_options = \
        PySPConfigBlock("Options registered for the "
                        "_ScenarioTreeSolverWorkerImpl class")

    safe_register_common_option(_registered_options,
                                "pyro_host")
    safe_register_common_option(_registered_options,
                                "pyro_port")
    safe_register_common_option(_registered_options,
                                "shutdown_pyro")

    ScenarioTreePreprocessor.register_options(_registered_options)

    def __init__(self, *args, **kwds):
        # solver related objects
        self._scenario_solvers = {}
        self._bundle_solvers = {}
        self._preprocessor = None
        self._solver_manager = None

        # there are situations in which it is valuable to snapshot /
        # store the solutions associated with the scenario
        # instances. for example, when one wants to use a warm-start
        # from a particular iteration solve, following a modification
        # and re-solve of the problem instances in a user-defined
        # callback. the following nested dictionary is intended to
        # serve that purpose. The nesting is dependent on whether
        # bundling and or phpyro is in use
        self._cached_solutions = {}
        self._cached_scenariotree_solutions = {}

        # results objects from the most recent round of solves These
        # may hold more information than just variable values and so
        # can be useful to hold on to until the next round of solves
        # (keys are bundle name or scenario name)
        self._solver_results = {}

        super(_ScenarioTreeSolverWorkerImpl, self).__init__(*args, **kwds)

        # initialize the preprocessor
        self._preprocessor = ScenarioTreePreprocessor(self._options)
        # initialize the solver manager
        self._solver_manager = SolverManagerFactory(
            self._options.solver_manager,
            host=self.get_option('pyro_host'),
            port=self.get_option('pyro_port'))

    #
    # Creates a deterministic symbol map for variables on an
    # instance. This allows convenient transmission of information to
    # and from ScenarioTreeSolverWorkers and makes it easy to save solutions
    # using a pickleable dictionary of symbols -> values
    #
    def _create_instance_symbol_maps(self, ctypes):

        for instance in itervalues(self._instances):

            create_block_symbol_maps(instance, ctypes)

    def push_fix_queue_to_instances(self):

        for tree_node in self._scenario_tree._tree_nodes:

            if len(tree_node._fix_queue):
                for scenario in tree_node._scenarios:
                    scenario_name = scenario.name
                    for variable_id, (fixed_status, new_value) in \
                          iteritems(tree_node._fix_queue):
                        variable_name, index = \
                            tree_node._variable_ids[variable_id]
                        if fixed_status == tree_node.VARIABLE_FREED:
                            self._preprocessor.\
                                freed_variables[scenario_name].\
                                append((variable_name, index))
                        elif fixed_status == tree_node.VARIABLE_FIXED:
                            self._preprocessor.\
                                fixed_variables[scenario_name].\
                                append((variable_name, index))
                tree_node.push_fix_queue_to_instances()

    #
    # Override some methods on _ScenarioTreeWorkerImpl:
    # NOTE: Be careful with MRO here. (1) Need to use super
    #       and (2) this class needs to appear BEFORE
    #       _ScenarioTreeWorkerImpl in the inheritance order
    #

    def _close_impl(self):
        super(_ScenarioTreeSolverWorkerImpl, self)._close_impl()
        ignored_options = dict((_c._name, _c.value(False))
                               for _c in self._options.unused_user_values())
        if len(ignored_options):
            print("")
            print("*** WARNING: The following options were explicitly "
                  "set but never accessed by worker %s: "
                  % (self._worker_name))
            for name in ignored_options:
                print(" - %s: %s" % (name, ignored_options[name]))
            print("*** If you believe this is a bug, please report it "
                  "to the PySP developers.")
            print("")

    #
    # Adds to functionality on _ScenarioTreeWorkerImpl
    # by registering the bundle instance with the preprocessor
    # and creating a solver for the bundle
    #

    def add_bundle(self, bundle_name, scenario_list):
        super(_ScenarioTreeSolverWorkerImpl, self).add_bundle(bundle_name,
                                                              scenario_list)
        assert bundle_name not in self._bundle_solvers
        self._bundle_solvers[bundle_name] = \
            SolverFactory(self._options.solver,
                          solver_io=self._options.solver_io)
        if self._preprocessor is not None:
            self._preprocessor.add_bundle(
                self._scenario_tree.get_bundle(bundle_name),
                self._bundle_binding_instance_map[bundle_name],
                self._bundle_solvers[bundle_name])

    #
    # Adds to functionality on _ScenarioTreeSolverWorkerImpl
    # by registering the bundle instance with the preprocessor
    # and creating a solver for the bundle
    #

    def remove_bundle(self, bundle_name):
        assert bundle_name in self._bundle_solvers
        if self._preprocessor is not None:
            self._preprocessor.remove_bundle(
                self._scenario_tree.get_bundle(bundle_name))
        self._bundle_solvers[bundle_name].deactivate()
        del self._bundle_solvers[bundle_name]
        super(_ScenarioTreeSolverWorkerImpl, self).remove_bundle(bundle_name)


    #
    # Queue the bundle or scenario solve requests
    # on the solver manager
    #

    def _queue_object_solves(self,
                             object_type,
                             objects,
                             ephemeral_solver_options,
                             disable_warmstart):

        if self._options.verbose:
            print("Queuing %s solves" % (object_type[:-1]))

        assert object_type in ('bundles', 'scenarios')

        solver_dict = None
        instance_dict = None
        if object_type == 'bundles':
            if objects is None:
                objects = self._scenario_tree._scenario_bundle_map
            solver_dict = self._bundle_solvers
            instance_dict = self._bundle_binding_instance_map
            for bundle_name in objects:
                for scenario_name in self._scenario_tree.get_bundle(bundle_name).\
                       scenario_names:
                    self._scenario_tree.get_scenario(scenario_name).\
                        _instance_objective.deactivate()
            if self._preprocessor is not None:
                self._preprocessor.preprocess_bundles(bundles=objects)
        else:
            if objects is None:
                objects = self._scenario_tree._scenario_map
            solver_dict = self._scenario_solvers
            instance_dict = self._instances
            if self._scenario_tree.contains_bundles():
                for scenario_name in objects:
                    self._scenario_tree.get_scenario(scenario_name).\
                        _instance_objective.activate()
            if self._preprocessor is not None:
                self._preprocessor.preprocess_scenarios(scenarios=objects)
        assert solver_dict is not None
        assert instance_dict is not None

        # setup common solve keywords
        common_kwds = {}
        common_kwds['tee'] = self._options.output_solver_logs
        common_kwds['keepfiles'] = self._options.keep_solver_files
        common_kwds['symbolic_solver_labels'] = \
            self._options.symbolic_solver_labels
        # we always rely on ourselves to load solutions - we control
        # the error checking and such.
        common_kwds['load_solutions'] = False

        # Load preprocessor related io_options
        if self._preprocessor is not None:
            common_kwds.update(self._preprocessor.get_solver_keywords())

        # Load solver options
        solver_options = {}
        for key in self._options.solver_options:
            solver_options[key] = self._options.solver_options[key]
        if self._options.mipgap is not None:
            solver_options['mipgap'] = self._options.mipgap
        common_kwds['options'] = solver_options

        #
        # override "persistent" values that are included from this
        # classes registered options
        #
        if ephemeral_solver_options is not None:
            common_kwds['options'].update(ephemeral_solver_options)

        # maps action handles to subproblem names
        action_handle_data = {}
        for object_name in objects:

            opt = solver_dict[object_name]
            instance = instance_dict[object_name]
            if (not self._options.disable_warmstart) and \
               (not disable_warmstart) and \
               opt.warm_start_capable():
                new_action_handle = \
                    self._solver_manager.queue(instance,
                                               opt=opt,
                                               warmstart=True,
                                               **common_kwds)
            else:
                new_action_handle = \
                    self._solver_manager.queue(instance,
                                               opt=opt,
                                               **common_kwds)

            action_handle_data[new_action_handle] = object_name

        return self.AsyncResult(
            self._solver_manager, action_handle_data=action_handle_data)

    def _process_bundle_solve_result(self, bundle_name, results):

        bundle = self._scenario_tree.get_bundle(bundle_name)
        bundle_instance = self._bundle_binding_instance_map[bundle.name]

        # if the solver plugin doesn't populate the
        # user_time field, it is by default of type
        # UndefinedData - defined in pyomo.opt.results
        if hasattr(results.solver,"user_time") and \
           (not isinstance(results.solver.user_time,
                           UndefinedData)) and \
           (results.solver.user_time is not None):
            # the solve time might be a string, or might
            # not be - we eventually would like more
            # consistency on this front from the solver
            # plugins.
            self._solve_times[bundle_name] = \
                float(results.solver.user_time)
        elif hasattr(results.solver,"time"):
            solve_time = results.solver.time
            self._solve_times[bundle_name] = float(results.solver.time)

        if hasattr(results,"pyomo_solve_time"):
            self._pyomo_solve_times[bundle_name] = \
                results.pyomo_solve_time

        if (len(results.solution) == 0) or \
           (results.solution(0).status == \
           SolutionStatus.infeasible) or \
           (results.solver.termination_condition == \
            TerminationCondition.infeasible):
            # solve failed
            return ("No solution returned or status infeasible: \n%s"
                    % (results.write()))

        if self._options.output_solver_results:
            print("Results for bundle=%s" % (bundle_name))
            results.write(num=1)

        results_sm = results._smap
        bundle_instance.solutions.load_from(
            results,
            allow_consistent_values_for_fixed_vars=\
               not self._options.preprocess_fixed_variables,
            comparison_tolerance_for_fixed_vars=\
               self._options.comparison_tolerance_for_fixed_variables,
            ignore_fixed_vars=self._options.preprocess_fixed_variables)
        self._solver_results[bundle_name] = \
            (results, results_sm)

        solution0 = results.solution(0)
        if hasattr(solution0, "gap") and \
           (solution0.gap is not None):
            self._gaps[bundle_name] = solution0.gap

        self._solution_status[bundle_name] = solution0.status

        for scenario_name in bundle._scenario_names:
            scenario = self._scenario_tree._scenario_map[scenario_name]
            scenario.update_solution_from_instance()

        return None

    def _process_scenario_solve_result(self, scenario_name, results):

        scenario = self._scenario_tree.get_scenario(scenario_name)
        scenario_instance = scenario._instance
        if self._scenario_tree.contains_bundles():
            scenario._instance_objective.deactivate()

        # if the solver plugin doesn't populate the
        # user_time field, it is by default of type
        # UndefinedData - defined in pyomo.opt.results
        if hasattr(results.solver,"user_time") and \
           (not isinstance(results.solver.user_time,
                           UndefinedData)) and \
           (results.solver.user_time is not None):
            # the solve time might be a string, or might
            # not be - we eventually would like more
            # consistency on this front from the solver
            # plugins.
            self._solve_times[scenario_name] = \
                float(results.solver.user_time)
        elif hasattr(results.solver,"time"):
            self._solve_times[scenario_name] = \
                float(results.solver.time)

        if hasattr(results,"pyomo_solve_time"):
            self._pyomo_solve_times[scenario_name] = \
                results.pyomo_solve_time

        if (len(results.solution) == 0) or \
           (results.solution(0).status == \
           SolutionStatus.infeasible) or \
           (results.solver.termination_condition == \
            TerminationCondition.infeasible):
            # solve failed
            return ("No solution returned or status infeasible: \n%s"
                    % (results.write()))

        if self._options.output_solver_results:
            print("Results for scenario="+scenario_name)
            results.write(num=1)

        # TBD: Technically, we should validate that there
        #      is only a single solution. Or at least warn
        #      if there are multiple.
        results_sm = results._smap
        scenario_instance.solutions.load_from(
            results,
            allow_consistent_values_for_fixed_vars=\
               not self._options.preprocess_fixed_variables,
            comparison_tolerance_for_fixed_vars=\
               self._options.comparison_tolerance_for_fixed_variables,
            ignore_fixed_vars=self._options.preprocess_fixed_variables)
        self._solver_results[scenario.name] = (results, results_sm)

        scenario.update_solution_from_instance()

        solution0 = results.solution(0)
        if hasattr(solution0, "gap") and \
           (solution0.gap is not None):
            self._gaps[scenario_name] = solution0.gap

        self._solution_status[scenario_name] = solution0.status

        return None

#
# This class is meant to serve as an ADDITIONAL base class
# to _ScenarioTreeManager and adds solve and preprocessing
# related functionality (it does not work as a stand-alone
# base class)
#

class _ScenarioTreeSolverManager(PySPConfiguredObject):

    _registered_options = \
        PySPConfigBlock("Options registered for the "
                        "_ScenarioTreeSolverManager class")

    #
    # solve and I/O related
    #
    safe_register_common_option(_registered_options,
                                "symbolic_solver_labels")
    safe_register_common_option(_registered_options,
                                "mipgap")
    safe_register_common_option(_registered_options,
                                "solver_options")
    safe_register_common_option(_registered_options,
                                "solver")
    safe_register_common_option(_registered_options,
                                "solver_io")
    safe_register_common_option(_registered_options,
                                "solver_manager")
    safe_register_common_option(_registered_options,
                                "disable_warmstart")
    safe_register_common_option(_registered_options,
                                "output_solver_logs")
    safe_register_common_option(_registered_options,
                                "output_solver_results")
    safe_register_common_option(_registered_options,
                                "keep_solver_files")
    safe_register_common_option(_registered_options,
                                "comparison_tolerance_for_fixed_variables")

    def __init__(self, *args, **kwds):
        super(_ScenarioTreeSolverManager, self).__init__(*args, **kwds)

        self._objective_sense = None

        # maps scenario name (or bundle name, in the case of bundling)
        # to the last solve time reported for the corresponding
        # sub-problem.
        # presently user time, due to deficiency in solver plugins. ultimately
        # want wall clock time for PH reporting purposes.
        self._solve_times = defaultdict(lambda: undefined)

        # similar to the above, but the time consumed by the invocation
        # of the solve() method on whatever solver plugin was used.
        self._pyomo_solve_times = defaultdict(lambda: undefined)

        # maps scenario name (or bundle name, in the case of bundling)
        # to the last gap reported by the solver when solving the
        # associated instance. if there is no entry, then there has
        # been no solve.
        self._gaps = defaultdict(lambda: undefined)

        # maps scenario name (or bundle name, in the case of bundling)
        # to the last solution status reported by the solver when solving the
        # associated instance. if there is no entry, then there has
        # been no solve.
        self._solution_status = defaultdict(lambda: undefined)

        # the preprocessor for instances (this can be None)
        self._preprocessor = None

    def _solve_objects(self,
                       object_type,
                       objects,
                       ephemeral_solver_options,
                       disable_warmstart,
                       exception_on_failure,
                       process_results,
                       async):

        assert object_type in ('bundles', 'scenarios')
        if not process_results:
            if exception_on_failure:
                raise ValueError("'exception_on_failure' can only be set to "
                                 "True when processing solve results")

        # queue the solves
        _async_solve_result = self._queue_object_solves(
            object_type,
            objects,
            ephemeral_solver_options,
            disable_warmstart)
        async_solve_result = self.AsyncResultCallback(
            lambda: (object_type, _async_solve_result.complete()))

        def _complete_solve():
            if process_results:
                return self.process_solve_results(async_solve_result)
            else:
                return async_solve_result.complete()

        result = self.AsyncResultCallback(_complete_solve)
        if not async:
            result = result.complete()
        return result

    #
    # Abstract methods defined on base class
    #

    def _queue_object_solves(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _process_bundle_solve_result(self, bundle_name, results):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _process_scenario_solve_result(self, scenario_name, results):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    #
    # Interface
    #

    def get_objective_sense(self):
        return self._objective_sense

    def report_bundle_objectives(self):

        assert self._scenario_tree.contains_bundles()

        max_name_len = max(len(str(_scenario_bundle.name)) \
                           for _scenario_bundle in \
                           self._scenario_tree._scenario_bundles)
        max_name_len = max((len("Scenario Bundle"), max_name_len))
        line = (("  %-"+str(max_name_len)+"s    ") % "Scenario Bundle")
        line += ("%-16s %-16s %-16s %-16s"
                 % ("Cost",
                    "Objective",
                    "Objective Gap",
                    "Solution Status "))
        if self._options.output_times:
            line += (" %-14s" % ("Solve Time"))
            line += (" %-14s" % ("Pyomo Time"))
        print(line)
        for scenario_bundle in self._scenario_tree._scenario_bundles:

            bundle_gap = self._gaps[scenario_bundle.name]
            bundle_status = self._solution_status[scenario_bundle.name]
            if isinstance(bundle_status, UndefinedData):
                bundle_status = "None Reported"
            bundle_objective_value = 0.0
            bundle_cost_value = 0.0
            for scenario in scenario_bundle._scenario_tree._scenarios:
                # The objective must be taken from the scenario
                # objects on PH full scenario tree
                scenario_objective = \
                    self._scenario_tree.get_scenario(scenario.name)._objective
                scenario_cost = \
                    self._scenario_tree.get_scenario(scenario.name)._cost
                # And we need to make sure to use the
                # probabilities assigned to scenarios in the
                # compressed bundle scenario tree
                bundle_objective_value += scenario_objective * \
                                          scenario._probability
                bundle_cost_value += scenario_cost * \
                                     scenario._probability

            line = ("  %-"+str(max_name_len)+"s    ")
            line += ("%-16.4f %-16.4f")
            if (not isinstance(bundle_gap, UndefinedData)) and \
               (bundle_gap is not None):
                line += (" %-16.4f")
            else:
                bundle_gap = "None Reported"
                line += (" %-16s")
            line += (" %-16s")
            line %= (scenario_bundle.name,
                     bundle_cost_value,
                     bundle_objective_value,
                     bundle_gap,
                     bundle_status)
            if self._options.output_times:
                solve_time = self._solve_times.get(scenario_bundle.name)
                if (not isinstance(solve_time, UndefinedData)) and \
                   (solve_time is not None):
                    line += (" %-14.2f"
                             % (solve_time))
                else:
                    line += (" %-14s" % "Not Reported")

                pyomo_solve_time = self._pyomo_solve_times.get(scenario_bundle.name)
                if (not isinstance(pyomo_solve_time, UndefinedData)) and \
                   (pyomo_solve_time is not None):
                    line += (" %-14.2f"
                             % (pyomo_solve_time))
                else:
                    line += (" %-14s" % "None Reported")
            print(line)
        print("")

    def report_scenario_objectives(self):

        max_name_len = max(len(str(_scenario.name)) \
                           for _scenario in self._scenario_tree._scenarios)
        max_name_len = max((len("Scenario"), max_name_len))
        line = (("  %-"+str(max_name_len)+"s    ") % "Scenario")
        line += ("%-16s %-16s %-16s %-16s"
                 % ("Cost",
                    "Objective",
                    "Objective Gap",
                    "Solution Status "))
        if self._options.output_times:
            line += (" %-14s" % ("Solve Time"))
            line += (" %-14s" % ("Pyomo Time"))
        print(line)
        for scenario in self._scenario_tree._scenarios:
            objective_value = scenario._objective
            scenario_cost = scenario._cost
            gap = self._gaps.get(scenario.name)
            status = self._solution_status[scenario.name]
            if isinstance(status, UndefinedData):
                status = "None Reported"
            line = ("  %-"+str(max_name_len)+"s    ")
            line += ("%-16.4f %-16.4f")
            if (not isinstance(gap, UndefinedData)) and (gap is not None):
                line += (" %-16.4f")
            else:
                gap = "None Reported"
                line += (" %-16s")
            line += (" %-16s")
            line %= (scenario.name,
                     scenario_cost,
                     objective_value,
                     gap,
                     status)
            if self._options.output_times:
                solve_time = self._solve_times.get(scenario.name)
                if (not isinstance(solve_time, UndefinedData)) and \
                   (solve_time is not None):
                    line += (" %-14.2f"
                             % (solve_time))
                else:
                    line += (" %-14s" % "None Reported")

                pyomo_solve_time = self._pyomo_solve_times.get(scenario.name)
                if (not isinstance(pyomo_solve_time, UndefinedData)) and \
                   (pyomo_solve_time is not None):
                    line += (" %-14.2f"
                             % (pyomo_solve_time))
                else:
                    line += (" %-14s" % "None Reported")
            print(line)
        print("")

    #
    # Solve scenarios or bundles (if they exist)
    #

    def solve_subproblems(self,
                          subproblems=None,
                          **kwds):
        ret = None
        if self._scenario_tree.contains_bundles():
            ret = self.solve_bundles(bundles=subproblems,
                                     **kwds)
        else:
            ret = self.solve_scenarios(scenarios=subproblems,
                                       **kwds)
        return ret

    #
    # Solve scenarios (ignoring bundles even if they exists)
    #

    def solve_scenarios(self,
                        scenarios=None,
                        ephemeral_solver_options=None,
                        disable_warmstart=False,
                        exception_on_failure=False,
                        process_results=True,
                        async=False):
        return self._solve_objects('scenarios',
                                   scenarios,
                                   ephemeral_solver_options,
                                   disable_warmstart,
                                   exception_on_failure,
                                   process_results,
                                   async)

    #
    # Solve scenario bundles (they must exists)
    #

    def solve_bundles(self,
                      bundles=None,
                      ephemeral_solver_options=None,
                      disable_warmstart=False,
                      exception_on_failure=False,
                      process_results=True,
                      async=False):
        if not self._scenario_tree.contains_bundles():
            raise RuntimeError(
                "Unable to solve bundles. Bundling "
                "does not seem to be activated.")
        return self._solve_objects('bundles',
                                   bundles,
                                   ephemeral_solver_options,
                                   disable_warmstart,
                                   exception_on_failure,
                                   process_results,
                                   async)

    #
    # Process and load previously queued solve results
    #

    def process_solve_results(self,
                              solve_result_data,
                              exception_on_failure=False):

        failures = []

        if isinstance(solve_result_data, self.Async):
            # After complete is called the first time,
            # any remaining calls will immediately return
            # the results, so it does not matter if the
            # user has already called complete
            solve_result_data = solve_result_data.complete()

        object_type, solver_results = solve_result_data
        assert object_type in ('bundles', 'scenarios')
        _process_function = self._process_bundle_solve_result \
                            if (object_type == 'bundles') else \
                               self._process_scenario_solve_result

        for object_name in solver_results:

            results = solver_results[object_name]

            if self._options.verbose:
                print("Processing results for %s=%s"
                      % (object_type[:-1],
                         object_name))

            start_load = time.time()
            #
            # This method is expected to update:
            #  - self._solve_times
            #  - self._pyomo_solve_times
            #  - self._gaps
            #  - self._solution_status
            #  - self._solver_results
            #  - solutions on scenario objects
            self._gaps[object_name] = undefined
            self._solution_status[object_name] = undefined
            self._solve_times[object_name] = undefined
            self._pyomo_solve_times[object_name] = undefined
            if object_type == 'scenarios':
                if self._scenario_tree.contains_bundles():
                    _bundle_name = self._scenario_to_bundle_map[object_name]
                    self._gaps[_bundle_name] = undefined
                    self._solution_status[_bundle_name] = undefined
                    self._solve_times[_bundle_name] = undefined
                    self._pyomo_solve_times[_bundle_name] = undefined
            else:
                assert object_type == 'bundles'
                for _scenario_name in self._scenario_tree.get_bundle(object_name).scenario_names:
                    self._gaps[_scenario_name] = undefined
                    self._solution_status[_scenario_name] = undefined
                    self._solve_times[_scenario_name] = undefined
                    self._pyomo_solve_times[_scenario_name] = undefined
            failure_msg = _process_function(object_name, results)

            if failure_msg is not None:
                failures.append(object_name)
                if self._options.verbose:
                    print("Solve failed for %s=%s; Message:\n%s"
                          % (object_type[:-1], object_name, failure_msg))

            else:
                if self._options.verbose:
                    print("Successfully loaded solution for %s=%s"
                          % (object_type[:-1], object_name))
                if self._options.output_times:
                    print("Time loading results for %s %s=%0.2f seconds"
                          % (object_type[:-1],
                             object_name,
                             time.time() - start_load))

        if len(failures) > 0:
            print(" ** At least one of the %s failed to solve! ** "
                  % (object_type))
            print(" Failed %s:" % (object_type))
            for failure in sorted(failures):
                print("   "+str(failure))
                if exception_on_failure:
                    raise RuntimeError("Failed to obtain a solution for "
                                       "the following %s: %s"
                                       % (object_type, str(failures)))

        if self._options.output_times:
            if len(failures) > 0:
                print("Skipping timing statistics due to one or more "
                      "solve failures" % (object_type))
            else:
                # if any of the solve times are of type
                # pyomo.opt.results.container.UndefinedData, then don't
                # output timing statistics.
                solve_times = [self._solve_times[object_name]
                               for object_name in solver_results]
                if any(isinstance(x, UndefinedData)
                       for x in solve_times):
                    print("At least one of the %s had an undefined solve time - "
                          "skipping timing statistics" % (object_type))
                else:
                    solve_times = [float(x) for x in solve_times]
                    mean = sum(solve_times) / float(len(solve_times))
                    std_dev = math.sqrt(sum(pow(x-mean,2.0) for x in solve_times) / \
                                        float(len(solve_times)))
                    print("Solve time statistics for %s - Min: "
                          "%0.2f Avg: %0.2f Max: %0.2f StdDev: %0.2f (seconds)"
                          % (object_type,
                             min(solve_times),
                             mean,
                             max(solve_times),
                             std_dev))

                # if any of the solve times are of type
                # pyomo.opt.results.container.UndefinedData, then don't
                # output timing statistics.
                pyomo_solve_times = [self._pyomo_solve_times[object_name]
                                     for object_name in solver_results]
                if any(isinstance(x, UndefinedData)
                       for x in pyomo_solve_times):
                    print("At least one of the %s had an undefined pyomo solve time - "
                          "skipping timing statistics" % (object_type))
                else:
                    pyomo_solve_times = [float(x) for x in pyomo_solve_times]
                    mean = sum(pyomo_solve_times) / float(len(pyomo_solve_times))
                    std_dev = \
                        math.sqrt(sum(pow(x-mean,2.0) for x in pyomo_solve_times) / \
                                  float(len(pyomo_solve_times)))
                    print("Pyomo solve time statistics for %s - Min: "
                          "%0.2f Avg: %0.2f Max: %0.2f StdDev: %0.2f (seconds)"
                          % (object_type,
                             min(pyomo_solve_times),
                             mean,
                             max(pyomo_solve_times),
                             std_dev))

        return failures

class ScenarioTreeSolverManagerSerial(ScenarioTreeManagerSerial,
                                      _ScenarioTreeSolverWorkerImpl,
                                      _ScenarioTreeSolverManager,
                                      PySPConfiguredObject):

    _registered_options = \
        PySPConfigBlock("Options registered for the "
                        "ScenarioTreeSolverManagerSerial class")

    def __init__(self, *args, **kwds):
        super(ScenarioTreeSolverManagerSerial, self).__init__(*args, **kwds)

    #
    # Abstract methods for _ScenarioTreeManagerImpl:
    #

    def _init(self):
        handle = super(ScenarioTreeSolverManagerSerial, self)._init()
        assert self._preprocessor is not None
        for scenario in self._scenario_tree._scenarios:
            assert scenario._instance is not None
            solver = self._scenario_solvers[scenario.name] = \
                SolverFactory(self._options.solver,
                              solver_io=self._options.solver_io)
            if self._preprocessor is not None:
                self._preprocessor.add_scenario(scenario,
                                                scenario._instance,
                                                solver)
        for bundle in self._scenario_tree._scenario_bundles:
            solver = self._bundle_solvers[bundle.name] = \
                SolverFactory(self._options.solver,
                              solver_io=self._options.solver_io)
            bundle_instance = \
                self._bundle_binding_instance_map[bundle.name]
            if self._preprocessor is not None:
                self._preprocessor.add_bundle(bundle,
                                              bundle_instance,
                                              solver)

        return handle

class ScenarioTreeSolverManagerSPPyro(ScenarioTreeManagerSPPyro,
                                      _ScenarioTreeSolverManager):

    _registered_options = \
        PySPConfigBlock("Options registered for the "
                        "ScenarioTreeSolverManagerSPPyro class")

    safe_register_common_option(
        _registered_options,
        "sppyro_transmit_leaf_stage_variable_solutions")

    default_registered_worker_name = 'ScenarioTreeSolverWorker'

    def __init__(self, *args, **kwds):
        super(ScenarioTreeSolverManagerSPPyro, self).__init__(*args, **kwds)

    def _get_queue_solve_kwds(self):
        args, kwds = self._get_common_solve_inputs()
        assert len(args) == 0
        kwds['solver_suffixes'] = []
        if not self._options.disable_warmstart:
            kwds['warmstart'] = True
        # TODO
        #kwds['variable_transmission'] = \
        #    self._phpyro_variable_transmission_flags
        return args, kwds

    def _request_scenario_tree_data(self):

        start_time = time.time()

        if self._options.verbose:
            print("Broadcasting requests to collect scenario tree "
                  "data from workers")

        # maps scenario or bundle name to async object
        async_results = {}

        need_node_data = dict((tree_node.name, True)
                              for tree_node in self._scenario_tree._tree_nodes)
        need_scenario_data = dict((scenario.name,True)
                                  for scenario in self._scenario_tree._scenarios)

        assert not self._transmission_paused
        self.pause_transmit()
        if self._scenario_tree.contains_bundles():

            for bundle in self._scenario_tree._scenario_bundles:

                object_names = {}
                object_names['nodes'] = \
                    [tree_node.name
                     for scenario in bundle._scenario_tree._scenarios
                     for tree_node in scenario.node_list
                     if need_node_data[tree_node.name]]
                object_names['scenarios'] = \
                    [scenario_name \
                     for scenario_name in bundle._scenario_names]

                async_results[bundle.name] = \
                    self.invoke_method_on_worker(
                        self.get_worker_for_bundle(bundle.name),
                        "_collect_scenario_tree_data_for_client",
                        method_args=(object_names,),
                        async=True)

                for node_name in object_names['nodes']:
                    need_node_data[node_name] = False
                for scenario_name in object_names['scenarios']:
                    need_scenario_data[scenario_name] = False

        else:

            for scenario in self._scenario_tree._scenarios:

                object_names = {}
                object_names['nodes'] = \
                    [tree_node.name for tree_node in scenario.node_list \
                     if need_node_data[tree_node.name]]
                object_names['scenarios'] = [scenario.name]

                async_results[scenario.name] = \
                    self.invoke_method_on_worker(
                        self.get_worker_for_scenario(scenario.name),
                        "_collect_scenario_tree_data_for_client",
                        method_args=(object_names,),
                        async=True)

                for node_name in object_names['nodes']:
                    need_node_data[node_name] = False
                for scenario_name in object_names['scenarios']:
                    need_scenario_data[scenario_name] = False

        self.unpause_transmit()

        assert all(not val for val in itervalues(need_node_data))
        assert all(not val for val in itervalues(need_scenario_data))

        return async_results

    def _gather_scenario_tree_data(self, async_results):

        start_time = time.time()

        have_node_data = dict((tree_node.name, False)
                              for tree_node in self._scenario_tree._tree_nodes)
        have_scenario_data = dict((scenario.name, False)
                                  for scenario in self._scenario_tree._scenarios)

        if self._options.verbose:
            print("Waiting for scenario tree data collection")

        if self._scenario_tree.contains_bundles():

            for bundle_name in async_results:

                results = async_results[bundle_name].complete()

                for tree_node_name, node_data in iteritems(results['nodes']):
                    assert have_node_data[tree_node_name] == False
                    have_node_data[tree_node_name] = True
                    tree_node = self._scenario_tree.get_node(tree_node_name)
                    tree_node._variable_ids.update(node_data['_variable_ids'])
                    tree_node._standard_variable_ids.update(
                        node_data['_standard_variable_ids'])
                    tree_node._variable_indices.update(
                        node_data['_variable_indices'])
                    tree_node._discrete.update(node_data['_discrete'])
                    # these are implied
                    tree_node._derived_variable_ids = \
                        set(tree_node._variable_ids) - \
                        tree_node._standard_variable_ids
                    tree_node._name_index_to_id = \
                        dict((val,key)
                             for key,val in iteritems(tree_node._variable_ids))

                for scenario_name, scenario_data in \
                      iteritems(results['scenarios']):
                    assert have_scenario_data[scenario_name] == False
                    have_scenario_data[scenario_name] = True
                    scenario = self._scenario_tree.get_scenario(scenario_name)
                    scenario._objective_name = scenario_data['_objective_name']
                    scenario._objective_sense = scenario_data['_objective_sense']

                if self._options.verbose:
                    print("Successfully loaded scenario tree data "
                          "for bundle="+bundle_name)

        else:

            for scenario_name in async_results:

                results = async_results[scenario_name].complete()

                for tree_node_name, node_data in iteritems(results['nodes']):
                    assert have_node_data[tree_node_name] == False
                    have_node_data[tree_node_name] = True
                    tree_node = self._scenario_tree.get_node(tree_node_name)
                    tree_node._variable_ids.update(node_data['_variable_ids'])
                    tree_node._standard_variable_ids.update(
                        node_data['_standard_variable_ids'])
                    tree_node._variable_indices.update(
                        node_data['_variable_indices'])
                    tree_node._discrete.update(node_data['_discrete'])
                    # these are implied
                    tree_node._derived_variable_ids = \
                        set(tree_node._variable_ids) - \
                        tree_node._standard_variable_ids
                    tree_node._name_index_to_id = \
                        dict((val,key)
                             for key,val in iteritems(tree_node._variable_ids))

                for scenario_name, scenario_data in \
                      iteritems(results['scenarios']):
                    assert have_scenario_data[scenario_name] == False
                    have_scenario_data[scenario_name] = True
                    scenario = self._scenario_tree.get_scenario(scenario_name)
                    scenario._objective_name = scenario_data['_objective_name']
                    scenario._objective_sense = scenario_data['_objective_sense']

                if self._options.verbose:
                    print("Successfully loaded scenario tree data for "
                          "scenario="+scenario_name)

        assert all(have_node_data)
        assert all(have_scenario_data)

        if self._options.verbose:
            print("Scenario tree instance data successfully "
                  "collected")

        if self._options.output_times:
            print("Scenario tree data collection time=%.2f seconds"
                  % (time.time() - start_time))

    #
    # Sends a mapping between (name,index) and ScenarioTreeID so that
    # phsolverservers are aware of the master nodes's ScenarioTreeID
    # labeling.
    #

    def _transmit_scenario_tree_ids(self):

        start_time = time.time()

        if self._options.verbose:
            print("Broadcasting scenario tree id mapping"
                  "to scenario tree servers")

        assert not self._transmission_paused
        self.pause_transmit()
        if self._scenario_tree.contains_bundles():

            for bundle in self._scenario_tree._scenario_bundles:

                ids_to_transmit = {}
                for stage in bundle._scenario_tree._stages:
                    for bundle_tree_node in stage._tree_nodes:
                        # The bundle scenariotree usually isn't populated
                        # with variable value data so we need to reference
                        # the original scenariotree node
                        primary_tree_node = self._scenario_tree.\
                                            _tree_node_map[bundle_tree_node.name]
                        ids_to_transmit[primary_tree_node.name] = \
                            primary_tree_node._variable_ids

                    self.invoke_method_on_worker(
                        self.get_worker_for_bundle(bundle.name),
                        "_update_master_scenario_tree_ids_for_client",
                        method_args=(bundle.name, ids_to_transmit),
                        oneway=True)

        else:

            for scenario in self._scenario_tree._scenarios:

                ids_to_transmit = {}
                for tree_node in scenario.node_list:
                    ids_to_transmit[tree_node.name] = tree_node._variable_ids

                self.invoke_method_on_worker(
                    self.get_worker_for_scenario(scenario.name),
                    "_update_master_scenario_tree_ids_for_client",
                    method_args=(scenario.name, ids_to_transmit),
                    oneway=True)

        self.unpause_transmit()

        end_time = time.time()

        if self._options.output_times:
            print("Scenario tree variable ids broadcast time=%.2f "
                  "seconds" % (time.time() - start_time))

    #
    # Abstract methods
    #

    def _init(self):
        init_handle = \
            super(ScenarioTreeSolverManagerSPPyro, self)._init()
        assert self._action_manager is not None

        async_results = self._request_scenario_tree_data()

        def _complete_init():
            self._gather_scenario_tree_data(async_results)
            self._transmit_scenario_tree_ids()
            return None

        async_callback = self.AsyncResultCallback(_complete_init)
        return self.AsyncResultChain(
            results=[init_handle, async_callback],
            return_index=0)

    def _setup_scenario_solve(self, scenario_name):
        args, kwds = self._get_queue_solve_kwds()
        assert len(args) == 0
        kwds['action'] = "solve_scenario"
        kwds['name'] = scenario_name
        return args, kwds

    def _setup_bundle_solve(self, bundle_name):
        args, kwds = self._get_queue_solve_kwds()
        assert len(args) == 0
        kwds['action'] = "solve_bundle"
        kwds['name'] = bundle_name
        return args, kwds

    #
    # Queue the bundle or scenario solve requests
    # on the solver manager
    #

    def _queue_object_solves(self,
                             object_type,
                             objects,
                             ephemeral_solver_options,
                             disable_warmstart):

        assert object_type in ('bundles', 'scenarios')

        if self._options.verbose:
            print("Transmitting solve requests for %s" % (object_type))

        worker_names = None
        worker_objects = {}
        if object_type == 'bundles':
            if objects is not None:
                worker_names = set()
                for bundle_name in objects:
                    worker_names.add(self.get_worker_for_bundle(bundle_name))
                    worker_objects.setdefault(worker_name, []).append(bundle_name)
        else:
            if objects is not None:
                worker_names = set()
                for scenario_name in objects:
                    worker_names.add(self.get_worker_for_senario(scenario_name))
                    worker_objects.setdefault(worker_name, []).append(scenario_name)
        if worker_names is None:
            worker_names = self._sppyro_worker_list

        was_paused = self.pause_transmit()
        action_handle_data = {}
        for worker_name in worker_names:
            action_handle_data[self._invoke_method_on_worker(
                worker_name,
                "_solve_objects_for_client",
                method_args=(object_type,
                             worker_objects.get(worker_name, None),
                             ephemeral_solver_options,
                             disable_warmstart),
                oneway=False)] = worker_name
        if not was_paused:
            self.unpause_transmit()

        return self.AsyncResult(
            self._action_manager,
            action_handle_data=action_handle_data,
            map_result=(lambda ah_to_result: \
                        dict((key, result[key])
                             for result in itervalues(ah_to_result)
                             for key in result)))

    def _process_bundle_solve_result(self, bundle_name, results):

        auxilliary_values = results['auxilliary_values']
        self._solve_times[bundle_name] = auxilliary_values['time']
        self._pyomo_solve_times[bundle_name] = auxilliary_values['pyomo_solve_time']
        self._gaps[bundle_name] = auxilliary_values['gaps']
        self._solution_status[bundle_name] = auxilliary_values['solution_status']

        if 'solution' not in results:
            # solve failed
            return "Solve status reported by scenario tree server"
        else:
            solution = results['solution']
            for scenario_name in self._scenario_tree.\
                   get_bundle(bundle_name).scenario_names:
                scenario = self._scenario_tree.get_scenario(scenario_name)
                scenario.update_current_solution(solution[scenario_name])

        return None

    def _process_scenario_solve_result(self, scenario_name, results):

        # TODO: Use these keywords to perform some
        #       validation of fixed variable values in the
        #       results returned
        #allow_consistent_values_for_fixed_vars=\
        #   not self._options.preprocess_fixed_variables,
        #comparison_tolerance_for_fixed_vars=\
        #   self._options.comparison_tolerance_for_fixed_variables,

        auxilliary_values = results['auxilliary_values']
        self._solve_times[scenario_name] = auxilliary_values['time']
        self._pyomo_solve_times[scenario_name] = auxilliary_values['pyomo_solve_time']
        self._gaps[scenario_name] = auxilliary_values['gaps']
        self._solution_status[scenario_name] = auxilliary_values['solution_status']

        if 'solution' not in results:
            # solve failed
            return "Solve status reported by scenario tree server"
        else:
            scenario = self._scenario_tree.get_scenario(scenario_name)
            scenario.update_current_solution(results['solution'])

        return None

    def push_fix_queue_to_instances(self):

        start_time = time.time()

        if self._options.verbose:
            print("Synchronizing fixed variable statuses with scenario instances")

        was_paused = self.pause_transmit()
        if self._scenario_tree.contains_bundles():

            for bundle in self._scenario_tree._scenario_bundles:

                transmit_variables = False
                for bundle_tree_node in bundle._scenario_tree._tree_nodes:
                        primary_tree_node = \
                            self._scenario_tree._tree_node_map[bundle_tree_node.name]
                        if len(primary_tree_node._fix_queue):
                            transmit_variables = True
                            break

                if transmit_variables:
                    # map from node name to the corresponding list of
                    # fixed variables
                    fixed_variables_to_transmit = {}

                    # Just send the entire state of fixed variables
                    # on each node (including leaf nodes)
                    for bundle_tree_node in bundle._scenario_tree._tree_nodes:
                        primary_tree_node = \
                            self._scenario_tree._tree_node_map[bundle_tree_node.name]
                        fixed_variables_to_transmit[primary_tree_node.name] = \
                            primary_tree_node._fix_queue

                    self._invoke_method_on_worker(
                        self.get_worker_for_bundle(bundle.name),
                        "_update_fixed_variables_for_client",
                        method_args=(bundle.name, fixed_variables_to_transmit),
                        oneway=True)

                else:
                    if self._options.verbose:
                        print("No synchronization was needed for bundle %s"
                              % (bundle.name))

        else:

            for scenario in self._scenario_tree._scenarios:

                transmit_variables = False
                for tree_node in scenario.node_list:
                    if len(tree_node._fix_queue):
                        transmit_variables = True
                        break

                if transmit_variables:

                    fixed_variables_to_transmit = \
                        dict((tree_node.name, tree_node._fix_queue)
                             for tree_node in scenario.node_list)

                    self._invoke_method_on_worker(
                        self.get_worker_for_scenario(scenario.name),
                        "_update_fixed_variables_for_client",
                        method_args=(scenario.name, fixed_variables_to_transmit),
                        oneway=True)

                else:
                    if self._options.verbose:
                        print("No synchronization was needed for scenario %s"
                              % (scenario.name))

        if not was_paused:
            self.unpause_transmit()

        if self._options.output_times:
            print("Fixed variable synchronization time="
                  "%.2f seconds" % (time.time() - start_time))
