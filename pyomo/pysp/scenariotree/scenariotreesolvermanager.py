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

import math
import time

from pyutilib.misc.config import (ConfigValue,
                                  ConfigBlock)
from pyutilib.misc import Bunch
from pyomo.opt import (UndefinedData,
                       undefined,
                       SolverFactory,
                       SolutionStatus,
                       TerminationCondition)
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.pysp.util.config import safe_register_common_option
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
        ConfigBlock("Options registered for the _ScenarioTreeSolverWorkerImpl class")

    safe_register_common_option(_registered_options,
                                "pyro_hostname")
    safe_register_common_option(_registered_options,
                                "shutdown_pyro")

    ScenarioTreePreprocessor.register_options(_registered_options)

    def __init__(self, *args, **kwds):
        super(_ScenarioTreeSolverWorkerImpl, self).__init__(*args, **kwds)

        # solver related objects
        self._scenario_solvers = {}
        self._bundle_solvers = {}
        self._preprocessor = ScenarioTreePreprocessor(*args, **kwds)
        self._solver_manager = SolverManagerFactory(
            self._options.solver_manager,
            host=self.get_option('pyro_hostname'))

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

    #
    # Adds to functionality on _ScenarioTreeWorkerImpl
    # by registering the bundle instance with the preprocessor
    # and creating a solver for the bundle
    #

    def add_bundle(self, bundle_name, scenario_list):
        _ScenarioTreeWorkerImpl.add_bundle(self, bundle_name, scenario_list)
        assert bundle_name not in bundle_solvers
        self._bundle_solvers[bundle_name] = \
            SolverFactory(self._options.solver,
                          solver_io=self._options.solver_io)
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
        assert bundle_name in bundle_solvers
        self._preprocessor.remove_bundle(
            self._scenario_tree.get_bundle(bundle_name))
        self._bundle_solvers[bundle_name].deactivate()
        del self._bundle_solvers[bundle_name]
        _ScenarioTreeWorkerImpl.remove_bundle(self, bundle_name)

    #
    # Creates a deterministic symbol map for variables on an
    # instance. This allows convenient transmission of information to
    # and from ScenarioTreeSolverWorkers and makes it easy to save solutions
    # using a pickleable dictionary of symbols -> values
    #
    def _create_instance_symbol_maps(self, ctypes):

        for instance in itervalues(self._instances):

            create_block_symbol_maps(instance, ctypes)

    # restores the variable values for all of the scenario instances
    # that I maintain.  restoration proceeds from the
    # self._cached_solutions map. if this is not populated (via
    # setting cache_results=True when calling solve_subproblems), then
    # an exception will be thrown.
    def restore_cached_solutions(self, cache_id, release_cache):

        cache = self._cached_scenariotree_solutions.get(cache_id,None)
        if cache is None:
            raise RuntimeError(
                "Scenario tree solution cache with id '%s' does not exist"
                % (cache_id))

        if release_cache and (cache is not None):
            del self._cached_scenariotree_solutions[cache_id]

        for scenario in self._scenario_tree._scenarios:

            scenario.update_current_solution(cache[scenario.name])

        if (not len(self._bundle_binding_instance_map)) and \
           (not len(self._instances)):
            return

        cache = self._cached_solutions.get(cache_id,None)
        if cache is None:
                raise RuntimeError(
                    "Solver solution cache with id %s does not exist"
                    % (cache_id))

        if release_cache and (cache is not None):
            del self._cached_solutions[cache_id]

        if self._scenario_tree.contains_bundles():

            for bundle_name in self._bundle_binding_instance_map:

                bundle_ef_instance = \
                    self._bundle_binding_instance_map[bundle_name]

                (results, results_sm), fixed_results = cache[bundle_name]

                for scenario_name in fixed_results:
                    scenario_fixed_results = fixed_results[scenario_name]
                    scenario_instance = self._instances[scenario_name]
                    bySymbol = scenario_instance.\
                               _PySPInstanceSymbolMaps[Var].bySymbol
                    for instance_id, varvalue, stale_flag in \
                           scenario_fixed_results:
                        vardata = bySymbol[instance_id]
                        vardata.fix(varvalue)

                bundle_ef_instance.solutions.add_symbol_map(results_sm)
                bundle_ef_instance.solutions.load_from(
                    results,
                    allow_consistent_values_for_fixed_vars=\
                       not self._options.preprocess_fixed_variables,
                    comparison_tolerance_for_fixed_vars=\
                       self._options.comparison_tolerance_for_fixed_variables)
                for scenario_name in fixed_results:
                    scenario_fixed_results = fixed_results[scenario_name]
                    scenario_instance = self._instances[scenario_name]
                    bySymbol = scenario_instance.\
                               _PySPInstanceSymbolMaps[Var].bySymbol
                    for instance_id, varvalue, stale_flag in \
                           scenario_fixed_results:
                        vardata = bySymbol[instance_id]
                        assert vardata.fixed
                        vardata.stale = stale_flag

        else:

            for scenario_name in self._instances:

                scenario_instance = self._instances[scenario_name]
                (results, results_sm), fixed_results = cache[scenario_name]

                bySymbol = scenario_instance.\
                           _PySPInstanceSymbolMaps[Var].bySymbol
                for instance_id, varvalue, stale_flag in fixed_results:
                    vardata = bySymbol[instance_id]
                    vardata.fix(varvalue)

                scenario_instance.solutions.add_symbol_map(results_sm)
                scenario_instance.solutions.load_from(
                    results,
                    allow_consistent_values_for_fixed_vars=\
                       not self._options.preprocess_fixed_variables,
                    comparison_tolerance_for_fixed_vars=\
                       self._options.comparison_tolerance_for_fixed_variables)
                bySymbol = scenario_instance.\
                           _PySPInstanceSymbolMaps[Var].bySymbol
                for instance_id, varvalue, stale_flag in fixed_results:
                    vardata = bySymbol[instance_id]
                    assert vardata.fixed
                    vardata.stale = stale_flag

    def cache_solutions(self, cache_id):

        for scenario in self._scenario_tree._scenarios:
            self._cached_scenariotree_solutions.\
                setdefault(cache_id,{})[scenario.name] = \
                    scenario.package_current_solution()

        if self._scenario_tree.contains_bundles():

            for bundle in self._scenario_tree._scenario_bundles:

                fixed_results = {}
                for scenario_name in bundle._scenario_names:

                    scenario_instance = scenario_map[scenario_name]
                    fixed_results[scenario_name] = \
                        tuple((instance_id, vardata.value, vardata.stale) \
                              for instance_id, vardata in \
                              iteritems(scenario_instance.\
                                        _PySPInstanceSymbolMaps[Var].\
                                        bySymbol) \
                              if vardata.fixed)

                self._cached_solutions.\
                    setdefault(cache_id,{})[bundle_name] = \
                        (self._solver_results[bundle_name],
                         fixed_results)

        else:

            for scenario_name in self._instances:

                scenario_instance = self._instances[scenario_name]
                fixed_results = \
                    tuple((instance_id, vardata.value, vardata.stale) \
                          for instance_id, vardata in \
                          iteritems(scenario_instance.\
                                    _PySPInstanceSymbolMaps[Var].bySymbol) \
                          if vardata.fixed)

                self._cached_solutions.\
                    setdefault(cache_id,{})[scenario_name] = \
                        (self._solver_results[scenario_name],
                         fixed_results)

    def _push_fix_queue_to_instances(self):

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

    def _push_all_node_fixed_to_instances(self):

        for tree_node in self._scenario_tree._tree_nodes:

            tree_node.push_all_fixed_to_instances()

            # flag the preprocessor
            for scenario in tree_node._scenarios:

                for variable_id in tree_node._fixed:

                    self._preprocessor.\
                        fixed_variables[scenario.name].\
                        append(tree_node._variable_ids[variable_id])

#
# This class is meant to serve as an ADDITIONAL base class
# to _ScenarioTreeManager and adds solve and preprocessing
# related functionality (it does not work as a stand-alone
# base class)
#

class _ScenarioTreeSolverManager(PySPConfiguredObject):

    _registered_options = \
        ConfigBlock("Options registered for the _ScenarioTreeSolverManager class")

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
        self._solve_times = {}

        # similar to the above, but the time consumed by the invocation
        # of the solve() method on whatever solver plugin was used.
        self._pyomo_solve_times = {}

        # maps scenario name (or bundle name, in the case of bundling)
        # to the last gap reported by the solver when solving the
        # associated instance. if there is no entry, then there has
        # been no solve.
        self._gaps = {}

        # maps scenario name (or bundle name, in the case of bundling)
        # to the last solution status reported by the solver when solving the
        # associated instance. if there is no entry, then there has
        # been no solve.
        self._solution_status = {}

        # seconds, over course of solve()
        self._cumulative_solve_time = 0.0

        # the preprocessor for instances (this can be None)
        self._preprocessor = None

    # minimize copy-pasted code
    def _solve_objects(self,
                       object_type,
                       objects,
                       ephemeral_solver_options,
                       disable_warmstart,
                       exception_on_failure):

        assert object_type in ('bundles', 'scenarios')
        iteration_start_time = time.time()

        # queue the solves
        if object_type == 'bundles':
            action_handle_data = \
                self.queue_bundle_solves(
                    bundles=objects,
                    ephemeral_solver_options=ephemeral_solver_options,
                    disable_warmstart=disable_warmstart)
        else:
            assert object_type == 'scenarios'
            action_handle_data = \
                self.queue_scenario_solves(
                    scenarios=objects,
                    ephemeral_solver_options=ephemeral_solver_options,
                    disable_warmstart=disable_warmstart)

        failures = self.wait_for_and_process_solves(action_handle_data)

        iteration_end_time = time.time()
        self._cumulative_solve_time += \
            (iteration_end_time - iteration_start_time)

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
                               for object_name in objects]
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
                               for object_name in objects]
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

            print("Overall solve time for %s=%.2f seconds"
                  % (object_type, iteration_end_time - iteration_start_time))

        return failures

    # minimize copy-pasted code
    def _queue_object_solves(self,
                             object_type,
                             objects,
                             ephemeral_solver_options,
                             disable_warmstart):

        assert object_type in ('bundles', 'scenarios')

        for object_name in objects:
            self._gaps[object_name] = undefined
            self._solution_status[object_name] = undefined
            self._solve_times[object_name] = undefined
            self._pyomo_solve_times[object_name] = undefined

        # maps action handles to subproblem names
        action_handle_name_map = {}

        for object_name in objects:

            if self._options.verbose:
                print("Queuing solve for %s=%s"
                      % (object_type[:-1], object_name))

            # preprocess and gather kwds are solve command
            if object_type == 'bundles':
                solve_args, solve_kwds = self._setup_bundle_solve(object_name)
            else:
                assert object_type == 'scenarios'
                solve_args, solve_kwds = self._setup_scenario_solve(object_name)

            #
            # override "persistent" values that are included from this
            # classes registered options
            #
            if ephemeral_solver_options is not None:
                solve_kwds['options'].update(ephemeral_solver_options)
            if disable_warmstart:
                if 'warmstart' in solve_kwds:
                    del solve_kwds['warmstart']

            new_action_handle = \
                self._solver_manager.queue(*solve_args, **solve_kwds)

            action_handle_name_map[new_action_handle] = object_name

        return Bunch(object_type=object_type,
                     ah_dict=action_handle_name_map)

    # minimize copy-paste code
    def _get_common_solve_inputs(self):
        common_kwds = {}
        common_kwds['tee'] = self._options.output_solver_logs
        common_kwds['keepfiles'] = self._options.keep_solver_files
        common_kwds['symbolic_solver_labels'] = \
            self._options.symbolic_solver_labels
        # we always rely on ourselves to load solutions - we control
        # the error checking and such.
        common_kwds['load_solutions'] = False

        #
        # Load preprocessor related io_options
        #
        if self._preprocessor is not None:
            common_kwds.update(self._preprocessor.get_solver_keywords())

        #
        # Load solver options
        #
        solver_options = {}
        for key in self._options.solver_options:
            solver_options[key] = self._options.solver_options[key]
        if self._options.mipgap is not None:
            solver_options['mipgap'] = self._options.mipgap
        common_kwds['options'] = solver_options

        return (), common_kwds

    #
    # Abstract methods defined on base class
    #

    def _setup_scenario_solve(self, scenario_name):
        raise NotImplementedError("This method is abstract")

    def _setup_bundle_solve(self, bundle_name):
        raise NotImplementedError("This method is abstract")

    def _process_solve_result(self, object_type, object_name, results):
        raise NotImplementedError("This method is abstract")

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
    # Queue solves for scenarios or bundles (if they exist)
    #

    def queue_subproblem_solves(self,
                                subproblems=None,
                                **kwds):
        ret = None
        if self._scenario_tree.contains_bundles():
            ret = self.queue_bundle_solves(bundles=subproblems,
                                           **kwds)
        else:
            ret = self.queue_scenario_solves(scenarios=subproblems,
                                             **kwds)
        return ret

    #
    # Solve scenarios (ignoring bundles even if they exists)
    #

    def solve_scenarios(self,
                        scenarios=None,
                        ephemeral_solver_options=None,
                        disable_warmstart=False,
                        exception_on_failure=False):
        if scenarios is None:
            scenarios = self._scenario_tree._scenario_map.keys()
        return self._solve_objects('scenarios',
                                   scenarios,
                                   ephemeral_solver_options,
                                   disable_warmstart,
                                   exception_on_failure)

    #
    # Queue solves for scenarios (ignoring bundles even if they exists)
    #

    def queue_scenario_solves(self,
                              scenarios=None,
                              ephemeral_solver_options=None,
                              disable_warmstart=False):

        if scenarios is None:
            scenarios = self._scenario_tree._scenario_map.keys()
        return self._queue_object_solves('scenarios',
                                         scenarios,
                                         ephemeral_solver_options,
                                         disable_warmstart)

    #
    # Solve scenario bundles (they must exists)
    #

    def solve_bundles(self,
                      bundles=None,
                      ephemeral_solver_options=None,
                      disable_warmstart=False,
                      exception_on_failure=False):

        if not self._scenario_tree.contains_bundles():
            raise RuntimeError(
                "Unable to solve bundles. Bundling "
                "does not seem to be activated.")
        if bundles is None:
            bundles = self._scenario_tree._scenario_bundle_map.keys()
        return self._solve_objects('bundles',
                                   bundles,
                                   ephemeral_solver_options,
                                   disable_warmstart,
                                   exception_on_failure)

    #
    # Queue solves for scenario bundles (they must exists)
    #

    def queue_bundle_solves(self,
                            bundles=None,
                            ephemeral_solver_options=None,
                            disable_warmstart=False):

        if not self._scenario_tree.contains_bundles():
            raise RuntimeError(
                "Unable to solve bundles. Bundling "
                "does not seem to be activated.")
        if bundles is None:
            bundles = self._scenario_tree._scenario_bundle_map.keys()
        return self._queue_object_solves('bundles',
                                         bundles,
                                         ephemeral_solver_options,
                                         disable_warmstart)

    #
    # Wait for solve action handles returned by one of the
    # queue_*_solves methods.
    #

    def wait_for_and_process_solves(self, action_handle_data):

        failures = []

        object_type = action_handle_data.object_type
        assert object_type in ('bundles', 'scenarios')

        subproblem_count = len(action_handle_data.ah_dict)

        if self._options.verbose:
            print("Waiting for %s %s to complete solve request"
                  % (subproblem_count, object_type))

        num_results_so_far = 0
        while (num_results_so_far < subproblem_count):

            action_handle = self._solver_manager.wait_any()
            results = self._solver_manager.get_results(action_handle)

            # there are cases, if the dispatchers and name servers are not
            # correctly configured, in which you may get an action handle
            # that you didn't expect. in this case, punt with a sane
            # message, as there isn't much else you can do.
            try:
                object_name = action_handle_data.ah_dict[action_handle]
            except KeyError:
                known_action_handles = \
                    sorted((ah.id for ah in action_handle_data.ah_dict))
                raise RuntimeError(
                    "Client received an unknown action handle=%d from "
                    "the dispatcher; known action handles are: %s"
                    % (action_handle.id, str(known_action_handles)))

            num_results_so_far += 1

            if self._options.verbose:
                print("Results obtained for %s=%s"
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
            failure_msg = self._process_solve_result(object_type,
                                                     object_name,
                                                     results)

            if failure_msg is not None:
                failures.append(object_name)
                if self._options.verbose:
                    print("Solve failed for %s=%s; Message:\n%s"
                          % (object_type[:-1], object_name, failure_msg))

            else:
                if self._options.verbose:
                    print("Successfully loaded solution for %s=%s - "
                          "waiting on %d more" % (object_type[:-1],
                                                  object_name,
                                                  subproblem_count-num_results_so_far))
                if self._options.output_times:
                    print("Time loading results for %s %s=%0.2f seconds"
                          % (object_type[:-1], object_name, time.time()-start_load))


        return failures

class ScenarioTreeSolverManagerSerial(ScenarioTreeManagerSerial,
                                      _ScenarioTreeSolverWorkerImpl,
                                      _ScenarioTreeSolverManager,
                                      PySPConfiguredObject):

    _registered_options = \
        ConfigBlock("Options registered for the ScenarioTreeSolverManagerSerial class")

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
            self._preprocessor.add_scenario(scenario,
                                            scenario._instance,
                                            solver)
        for bundle in self._scenario_tree._scenario_bundles:
            solver = self._bundle_solvers[bundle.name] = \
                SolverFactory(self._options.solver,
                              solver_io=self._options.solver_io)
            bundle_instance = \
                self._bundle_binding_instance_map[bundle.name]
            self._preprocessor.add_bundle(bundle,
                                          bundle_instance,
                                          solver)
        return handle

    #
    # Abstract methods for _ScenarioTreeSolverManager:
    #

    def _setup_scenario_solve(self, scenario_name):
        args, kwds = self._get_common_solve_inputs()
        assert len(args) == 0
        kwds['opt'] = self._scenario_solvers[scenario_name]
        if (not self._options.disable_warmstart) and \
           kwds['opt'].warm_start_capable():
            kwds['warmstart'] = True
        args = (self._instances[scenario_name],)
        if self._scenario_tree.contains_bundles():
            self._scenario_tree.get_scenario(scenario_name).\
                _instance_objective.activate()
        self._preprocessor.preprocess_scenarios(scenarios=[scenario_name])
        return args, kwds

    def _setup_bundle_solve(self, bundle_name):
        args, kwds = self._get_common_solve_inputs()
        assert len(args) == 0
        kwds['opt'] = self._bundle_solvers[bundle_name]
        if (not self._options.disable_warmstart) and \
           kwds['opt'].warm_start_capable():
            kwds['warmstart'] = True
        args = (self._bundle_binding_instance_map[bundle_name],)
        self._preprocessor.preprocess_bundles(bundles=[bundle_name])
        return args, kwds

    def _process_solve_result(self, object_type, object_name, results):

        assert object_type in ('scenarios', 'bundles')

        if object_type == 'bundles':
            bundle = self._scenario_tree.get_bundle(object_name)
            bundle_instance = self._bundle_binding_instance_map[bundle.name]

            if (len(results.solution) == 0) or \
               (results.solution(0).status == \
               SolutionStatus.infeasible) or \
               (results.solver.termination_condition == \
                TerminationCondition.infeasible):
                # solve failed
                return ("No solution returned or status infeasible: \n%s"
                        % (results.write()))

            if self._options.output_solver_results:
                print("Results for bundle=%s" % (object_name))
                results.write(num=1)

            results_sm = results._smap
            bundle_instance.solutions.load_from(
                results,
                allow_consistent_values_for_fixed_vars=\
                   not self._options.preprocess_fixed_variables,
                comparison_tolerance_for_fixed_vars=\
                   self._options.comparison_tolerance_for_fixed_variables,
                ignore_fixed_vars=self._options.preprocess_fixed_variables)
            self._solver_results[object_name] = \
                (results, results_sm)

            solution0 = results.solution(0)
            if hasattr(solution0, "gap") and \
               (solution0.gap is not None):
                self._gaps[object_name] = solution0.gap

            self._solution_status[object_name] = solution0.status

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
                self._solve_times[object_name] = \
                    float(results.solver.user_time)
            elif hasattr(results.solver,"time"):
                solve_time = results.solver.time
                self._solve_times[object_name] = float(results.solver.time)

            if hasattr(results,"pyomo_solve_time"):
                self._pyomo_solve_times[object_name] = results.pyomo_solve_time

            for scenario_name in bundle._scenario_names:
                scenario = self._scenario_tree._scenario_map[scenario_name]
                scenario.update_solution_from_instance()

        else:
            assert object_type == 'scenarios'

            scenario = self._scenario_tree.get_scenario(object_name)
            scenario_instance = scenario._instance
            if self._scenario_tree.contains_bundles():
                scenario._instance_objective.deactivate()

            if (len(results.solution) == 0) or \
               (results.solution(0).status == \
               SolutionStatus.infeasible) or \
               (results.solver.termination_condition == \
                TerminationCondition.infeasible):
                # solve failed
                return ("No solution returned or status infeasible: \n%s"
                        % (results.write()))

            if self._options.output_solver_results:
                print("Results for scenario="+object_name)
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
                self._gaps[object_name] = solution0.gap

            self._solution_status[object_name] = solution0.status

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
                self._solve_times[object_name] = \
                    float(results.solver.user_time)
            elif hasattr(results.solver,"time"):
                self._solve_times[object_name] = \
                    float(results.solver.time)

            if hasattr(results,"pyomo_solve_time"):
                self._pyomo_solve_times[object_name] = results.pyomo_solve_time

        return None

class ScenarioTreeSolverManagerSPPyro(ScenarioTreeManagerSPPyro,
                                      _ScenarioTreeSolverManager):

    _registered_options = \
        ConfigBlock("Options registered for the ScenarioTreeSolverManagerSPPyro class")

    safe_register_common_option(_registered_options,
                                "sppyro_transmit_leaf_stage_variable_solutions")

    default_registered_worker_name = 'ScenarioTreeSolverWorker'

    def __init__(self, *args, **kwds):
        self._solver_manager = None
        super(ScenarioTreeSolverManagerSPPyro, self).__init__(*args, **kwds)

    def _get_queue_solve_kwds(self):
        args, kwds = self._get_common_solve_inputs()
        assert len(args) == 0
        kwds['options'] = solver_options
        kwds['solver_suffixes'] = []
        kwds['warmstart'] = warmstart
        kwds['variable_transmission'] = \
            self._phpyro_variable_transmission_flags
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
                     for tree_node in scenario._node_list
                     if need_node_data[tree_node.name]]
                object_names['scenarios'] = \
                    [scenario_name \
                     for scenario_name in bundle._scenario_names]

                async_results[bundle.name] = \
                    self.invoke_method_on_worker(
                        self.get_worker_for_bundle(bundle.name),
                        "collect_scenario_tree_data",
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
                    [tree_node.name for tree_node in scenario._node_list \
                     if need_node_data[tree_node.name]]
                object_names['scenarios'] = [scenario.name]

                async_results[scenario.name] = \
                    self.invoke_method_on_worker(
                        self.get_worker_for_scenario(scenario.name),
                        "collect_scenario_tree_data",
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
                  % (end_time - start_time))

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
                        "update_master_scenario_tree_ids",
                        method_args=(bundle.name, ids_to_transmit),
                        oneway=True)

        else:

            for scenario in self._scenario_tree._scenarios:

                ids_to_transmit = {}
                for tree_node in scenario._node_list:
                    ids_to_transmit[tree_node.name] = tree_node._variable_ids

                self.invoke_method_on_worker(
                    self.get_worker_for_scenario(scenario.name),
                    "update_master_scenario_tree_ids",
                    method_args=(scenario.name, ids_to_transmit),
                    oneway=True)

        self.unpause_transmit()

        end_time = time.time()

        if self._options.output_times:
            print("Scenario tree variable ids broadcast time=%.2f "
                  "seconds" % (end_time - start_time))

    #
    # Abstract methods
    #

    def _init(self):
        init_handle = \
            super(ScenarioTreeSolverManagerSPPyro, self)._init()
        assert self._action_manager is not None
        self._solver_manager = self._action_manager

        async_results = self._request_scenario_tree_data()

        def _complete_init():
            self._gather_scenario_tree_data(async_results)
            self._transmit_scenario_tree_ids()

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
        kwds = self._get_queue_solve_kwds()
        assert len(args) == 0
        kwds['action'] = "solve_bundle"
        kwds['name'] = bundle_name
        return args, kwds

    def _process_solve_result(self, object_type, object_name, results):

        assert object_type in ('scenarios', 'bundles')

        if object_type == 'bundles':
            assert self._scenario_tree.contains_bundles()

            if len(results) == 0:
                # solve failed
                return "Solve status reported by scenario tree server"

            for scenario_name, scenario_solution in \
                              iteritems(results[0]):
                scenario = self._scenario_tree._scenario_map[scenario_name]
                scenario.update_current_solution(scenario_solution)

            auxilliary_values = results[2]
            if "gap" in auxilliary_values:
                self._gaps[object_name] = auxilliary_values["gap"]

            self._solution_status[object_name] = auxilliary_values["solution_status"]

            if "user_time" in auxilliary_values:
                self._solve_times[object_name] = \
                    auxilliary_values["user_time"]
            elif "time" in auxilliary_values:
                self._solve_times[object_name] = \
                    auxilliary_values["time"]

            self._pyomo_solve_times[object_name] = auxilliary_values["pyomo_solve_time"]

        else:
            assert object_type == 'scenarios'
            if len(results) == 0:
                # solve failed
                return "Solve status reported by scenario tree server"

            # TODO: Use these keywords to perform some
            #       validation of fixed variable values in the
            #       results returned
            #allow_consistent_values_for_fixed_vars=\
            #   not self._options.preprocess_fixed_variables,
            #comparison_tolerance_for_fixed_vars=\
            #   self._options.comparison_tolerance_for_fixed_variables,
            # results[0] are variable values
            # results[1] are suffix values
            # results[2] are auxilliary values
            scenario.update_current_solution(results[0])

            auxilliary_values = results[2]
            if "gap" in auxilliary_values:
                self._gaps[object_name] = auxilliary_values["gap"]

            self._solution_status[object_name] = auxilliary_values["solution_status"]

            if "user_time" in auxilliary_values:
                self._solve_times[object_name] = \
                    auxilliary_values["user_time"]
            elif "time" in auxilliary_values:
                self._solve_times[object_name] = \
                    auxilliary_values["time"]

            self._pyomo_solve_times[object_name] = auxilliary_values["pyomo_solve_time"]

        return None
