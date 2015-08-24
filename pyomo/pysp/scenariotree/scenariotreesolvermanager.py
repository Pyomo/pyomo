#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("ScenarioTreeSolverManager",)

# TODO: handle pyro as the solver manager
# TODO: handle warmstart keyword
# TODO: implement sppyro worker side
# TODO: separate solver_manager from scenario_tree_manager

import time

from pyutilib.misc.config import (ConfigValue,
                                  ConfigBlock)

from pyomo.opt import (UndefinedData,
                       undefined,
                       SolverFactory)
from pyomo.pysp.util.config import safe_register_common_option
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.scenariotree.preprocessor import ScenarioTreePreprocessor
from pyomo.pysp.scenariotree.scenariotreemanager import (
    ScenarioTreeManagerSerial, ScenarioTreeManagerSPPyro, )
import pyomo.pysp.scenariotree.scenariotreeserverutils

#
# This class is designed to manage simple solve invocations
# as well as solution caching across distributed or serial
# scenario tree management
#

class _ScenarioTreeSolverManagerImpl(object):

    def __init__(self, *args, **kwds):
        super(_ScenarioTreeSolverManagerImpl, self).__init__(*args, **kwds)

        # solver related objects
        self._solver = None
        self._preprocessor = None
        self._comparison_tolerance_for_fixed_vars = 1e-5

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
    # Creates a deterministic symbol map for variables on an
    # instance. This allows convenient transmission of information to
    # and from PHSolverServers and makes it easy to save solutions
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

            scenario.update_current_solution(cache[scenario._name])

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
                       self._comparison_tolerance_for_fixed_vars)
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
                       self._comparison_tolerance_for_fixed_vars)
                bySymbol = scenario_instance.\
                           _PySPInstanceSymbolMaps[Var].bySymbol
                for instance_id, varvalue, stale_flag in fixed_results:
                    vardata = bySymbol[instance_id]
                    assert vardata.fixed
                    vardata.stale = stale_flag

    def cache_solutions(self, cache_id):

        for scenario in self._scenario_tree._scenarios:
            self._cached_scenariotree_solutions.\
                setdefault(cache_id,{})[scenario._name] = \
                    scenario.package_current_solution()

        if self._scenario_tree.contains_bundles():

            for bundle_name in self._bundle_scenario_map:

                scenario_map = \
                    self._bundle_scenario_map[bundle_name]
                fixed_results = {}
                for scenario_name in scenario_map:

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
                    scenario_name = scenario._name
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
                        fixed_variables[scenario._name].\
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
                                "disable_warmstarts")
    safe_register_common_option(_registered_options,
                                "output_solver_logs")
    safe_register_common_option(_registered_options,
                                "output_solver_results")
    safe_register_common_option(_registered_options,
                                "keep_solver_files")

    def __init__(self, *args, **kwds):
        super(_ScenarioTreeSolverManager, self).__init__(*args, **kwds)

        # maps scenario name (or bundle name, in the case of bundling)
        # to the last solve time reported for the corresponding
        # sub-problem.
        # presently user time, due to deficiency in solver plugins. ultimately
        # want wall clock time for PH reporting purposes.
        self._solve_times = {}

        # maps scenario name (or bundle name, in the case of bundling)
        # to the last gap reported by the solver when solving the
        # associated instance. if there is no entry, then there has
        # been no solve.
        # NOTE: This dictionary could expand significantly, as we
        #       identify additional solve-related information
        #       associated with an instance.
        self._gaps = {}

        # seconds, over course of solve()
        self._cumulative_solve_time = 0.0

    # minimize copy-pasted code
    def _solve_objects(object_type,
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

        subproblems, failures = self.wait_for_and_process_solves(action_handle_data)

        # do some error checking reporting
        if len(self._solve_times) > 0:
            # if any of the solve times are of type
            # pyomo.opt.results.container.UndefinedData, then don't
            # output timing statistics.
            undefined_detected = False
            for this_time in itervalues(self._solve_times):
                if isinstance(this_time, UndefinedData):
                    undefined_detected=True
            if undefined_detected:
                print("At least one of the %s had an undefined solve time - "
                      "skipping timing statistics" % (object_type))
            else:
                mean = sum(self._solve_times.values()) / \
                        float(len(self._solve_times.values()))
                std_dev = sqrt(
                    sum(pow(x-mean,2.0) for x in self._solve_times.values()) /
                    float(len(self._solve_times.values())))
                if self._options.output_times:
                    print("Solve time statistics for %s - Min: "
                          "%0.2f Avg: %0.2f Max: %0.2f StdDev: %0.2f (seconds)"
                          % (object_type,
                             min(self._solve_times.values()),
                             mean,
                             max(self._solve_times.values()),
                             std_dev))

        iteration_end_time = time.time()
        self._cumulative_solve_time += (iteration_end_time - iteration_start_time)

        if self._options.output_times:
            print("Solve time for %s=%.2f seconds"
                  % (object_type, iteration_end_time - iteration_start_time))

        if len(failures):
            print(" ** At least one of the %s failed to solve! ** " % (object_type))
            print(" Failed %s:" % (object_type))
            for failure in sorted(failures):
                print("   "+str(failure))
            if exception_on_failure:
                raise RuntimeError("Failed to obtain a solution for "
                                   "the following %s: %s"
                                   % (object_type, str(failures)))

        return failures

    # minimize copy-pasted code
    def _queue_object_solves(object_type,
                             objects,
                             ephemeral_solver_options,
                             disable_warmstart):

        assert object_type in ('bundles', 'scenarios')

        self._gaps = {}
        self._solve_times = {}
        for object_name in objects:
            self._gaps[object_name] = undefined
            self._solve_times[object_name] = undefined

        # maps action handles to subproblem names
        action_handle_name_map = {}

        # preprocess and gather kwds are solve command
        if object_type == 'bundles':
            if self._preprocessor is not None:
                self._preprocessor.preprocess_bundles(bundles=objects)
            solve_args, solve_kwds = self._get_queue_bundle_solve_inputs()
        else:
            assert object_type == 'scenarios'
            if self._preprocessor is not None:
                self._preprocessor.preprocess_scenarios(scenarios=objects)
            solve_args, solve_kwds = self._get_queue_scenario_solve_inputs()

        #
        # override "persistent" values that are included from this
        # classes registered options
        #
        solve_kwds['solver_options'].update(ephemeral_solver_options)
        if disable_warmstart:
            if 'warmstart' in solve_kwds:
                del solve_kwds['warmstart']

        for object_name in objects:

            if self._options.verbose:
                print("Queuing solve for %s=%s"
                      % (object_type[:-1], object_name))

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
            self._self.options.symbolic_solver_labels
        # we always rely on ourselves to load solutions - we control
        # the error checking and such.
        common_kwds['load_solutions'] = False

        #
        # Load preprocessor related io_options
        #
        if self._preprocesssor is not None:
            common_kwds.update(self._preprocessor.get_solver_keywords())

        #
        # Load solver options
        #
        solver_options = {}
        for key in self._options.solver_options:
            solver_options[key] = self._options.solver_options[key]
        if self._options.mipgap is not None:
            solver_options['mipgap'] = self._options.mipgap
        common_kwds['solver_options'] = solver_options

        return (), common_kwds

    #
    # Abstract methods defined on base class
    #

    def _get_queue_scenario_solve_kwds(self, scenario_name):
        raise NotImplementedError("This method is abstract")

    def _get_queue_bundle_solve_kwds(self, bundle_name):
        raise NotImplementedError("This method is abstract")

    def _process_solve_result(self, object_type, object_name, results):
        raise NotImplementedError("This method is abstract")

    #
    # Interface
    #

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
            scenario = self._instances.keys()
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
            bundles = self._bundle_scenario_map.keys()
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
        subproblems = []

        assert action_handle_data.object_type in ('bundles', 'scenarios')

        subproblem_count = len(action_handle_data.ah_dict)

        if self._options.verbose:
            print("Waiting for %s %s to complete solve request"
                  % (len(subproblem_count), action_handle_data.object_type))

        num_results_so_far = 0
        while (num_results_so_far < subproblem_count):

            action_handle = self._solver_manager.wait_any(action_handle_data.ah_dict)
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

            subproblems.append(object_name)

            num_results_so_far += 1

            if self._options.verbose:
                print("Results obtained for %s=%s" % (object_type[:-1], object_name))

            start_load = time.time()
            #
            # This method is expected to:
            #  - update self._solve_times
            #  - update self._gaps
            #  - update self._solver_results
            #  - update solutions on scenario objects
            failure_msg = self._process_solve_result(object_type, object_name, results)

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


        return subproblems, failures

class ScenarioTreeSolverManagerSerial(ScenarioTreeManagerSerial,
                                      _ScenarioTreeSolverManagerImpl,
                                      _ScenarioTreeSolverManager,
                                      PySPConfiguredObject):

    _registered_options = \
        ConfigBlock("Options registered for the ScenarioTreeSolverManagerSerial class")

    def __init__(self, *args, **kwds):
        self._preprocessor = ScenarioTreePreprocessor(options)
        super(ScenarioTreeSolverManagerSerial, self).__init__(*args, **kwds)

    def _get_queue_solve_kwds(self):
        args, kwds = self._get_common_solve_inputs()
        assert len(args) == 0
        kwds['opt'] = self._solver
        if (not self._options.disable_warmstart) and \
           self._solver.warm_start_capable():
            kwds['warmstart'] = True
        return args, kwds

    #
    # Abstract methods for _ScenarioTreeManagerImpl:
    #

    def _init(self):
        ScenarioTreeManagerSerial._init(self)

    #
    # Abstract methods for _ScenarioTreeSolverManager:
    #

    def _get_queue_scenario_solve_kwds(self, scenario_name):
        args, kwds = self._get_queue_solve_kwds()
        assert len(args) == 0
        args = (self._instances[scenario_name],)
        return args, kwds

    def _get_queue_bundle_solve_kwds(self, bundle_name):
        args, kwds = self._get_queue_solve_kwds()
        assert len(args) == 0
        args = (self._bundle_binding_instance_map[bundle_name],)
        return args, kwds

    def _process_solve_result(self, object_type, object_name, results):

        assert object_type in ('scenarios', 'bundles')

        if object_type == 'bundles':
            assert self._scenario_tree.contains_bundles()

            if (len(bundle_results.solution) == 0) or \
               (bundle_results.solution(0).status == \
               SolutionStatus.infeasible) or \
               (bundle_results.solver.termination_condition == \
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
                self._write_fixed_variables,
                comparison_tolerance_for_fixed_vars=\
                self._comparison_tolerance_for_fixed_vars,
                ignore_fixed_vars=not self._write_fixed_variables)
            self._solver_results[object_name] = \
                (results, results_sm)

            solution0 = results.solution(0)
            if hasattr(solution0, "gap") and \
               (solution0.gap is not None):
                self._gaps[object_name] = solution0.gap

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

            scenario_bundle = \
                self._scenario_tree._scenario_bundle_map[object_name]
            for scenario_name in scenario_bundle._scenario_names:
                scenario = self._scenario_tree._scenario_map[scenario_name]
                scenario.update_solution_from_instance()

        else:
            assert object_type == 'scenarios'

            instance = scenario._instance

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
            instance.solutions.load_from(
                results,
                allow_consistent_values_for_fixed_vars=\
                    self._write_fixed_variables,
                comparison_tolerance_for_fixed_vars=\
                    self._comparison_tolerance_for_fixed_vars,
                ignore_fixed_vars=not self._write_fixed_variables)
            self._solver_results[scenario._name] = (results, results_sm)

            scenario.update_solution_from_instance()

            solution0 = results.solution(0)
            if hasattr(solution0, "gap") and \
               (solution0.gap is not None):
                self._gaps[scenario_name] = solution0.gap

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

        return None

class ScenarioTreeSolverManagerSPPyro(ScenarioTreeManagerSPPyro,
                                      _ScenarioTreeSolverManager):

    _registered_options = \
        ConfigBlock("Options registered for the ScenarioTreeSolverManagerSPPyro class")

    safe_register_common_option(_registered_options,
                                "sppyro_transmit_leaf_stage_variable_solutions")

    def __init__(self, *args, **kwds):
        self._solver_manager = None
        super(ScenarioTreeSolverManagerSPPyro, self).__init__(*args, **kwds)

    def _get_queue_solve_kwds(self):
        args, kwds = self._get_common_solve_inputs()
        assert len(args) == 0

        kwds['solver_options'] = solver_options
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

        # maps action handles to worker name
        action_handle_to_worker_map = {}

        need_node_data = dict((tree_node._name, True)
                              for tree_node in self._scenario_tree._tree_nodes)
        need_scenario_data = dict((scenario._name,True)
                                  for scenario in self._scenario_tree._scenarios)

        if self._scenario_tree.contains_bundles():

            for scenario_bundle in self._scenario_tree._scenario_bundles:

                object_names = {}
                object_names['nodes'] = \
                    [tree_node._name
                     for scenario in scenario_bundle._scenario_tree._scenarios
                     for tree_node in scenario._node_list
                     if need_node_data[tree_node._name]]
                object_names['scenarios'] = \
                    [scenario_name \
                     for scenario_name in scenario_bundle._scenario_names]

                new_action_handle =  self._action_manager.queue(
                    action="collect_scenario_tree_data",
                    name=scenario_bundle._name,
                    tree_object_names=object_names)

                action_handle_to_worker_map[new_action_handle] = \
                    scenario_bundle._name

                for node_name in object_names['nodes']:
                    need_node_data[node_name] = False
                for scenario_name in object_names['scenarios']:
                    need_scenario_data[scenario_name] = False

        else:

            for scenario in self._scenario_tree._scenarios:

                object_names = {}
                object_names['nodes'] = \
                    [tree_node._name for tree_node in scenario._node_list \
                     if need_node_data[tree_node._name]]
                object_names['scenarios'] = [scenario._name]

                new_action_handle = self._action_manager.queue(
                    action="collect_scenario_tree_data",
                    name=scenario._name,
                    tree_object_names=object_names)

                action_handle_to_worker_map[new_action_handle] = scenario._name

                for node_name in object_names['nodes']:
                    need_node_data[node_name] = False
                for scenario_name in object_names['scenarios']:
                    need_scenario_data[scenario_name] = False

        assert all(not val for val in itervalues(need_node_data))
        assert all(not val for val in itervalues(need_scenario_data))

        return action_handle_to_worker_map

    def _gather_scenario_tree_data(self,
                                   action_handle_to_worker_map,
                                   initialization_action_handles):

        have_node_data = dict((tree_node._name, False)
                              for tree_node in self._scenario_tree._tree_nodes)
        have_scenario_data = dict((scenario._name, False)
                                  for scenario in self._scenario_tree._scenarios)

        if self._options.verbose:
            print("Waiting scenario tree data collection")

        if self._scenario_tree.contains_bundles():

            num_results_so_far = 0

            while (num_results_so_far < len(self._scenario_tree._scenario_bundles)):

                action_handle = self._action_manager.wait_any()

                if action_handle in initialization_action_handles:
                    initialization_action_handles.remove(action_handle)
                    self._action_manager.get_results(action_handle)
                    continue

                bundle_results = self._action_manager.get_results(action_handle)
                bundle_name = action_handle_to_worker_map[action_handle]

                for tree_node_name, node_data in iteritems(bundle_results['nodes']):
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
                      iteritems(bundle_results['scenarios']):
                    assert have_scenario_data[scenario_name] == False
                    have_scenario_data[scenario_name] = True
                    scenario = self._scenario_tree.get_scenario(scenario_name)
                    scenario._objective_name = scenario_data['_objective_name']
                    scenario._objective_sense = scenario_data['_objective_sense']

                if self._options.verbose:
                    print("Successfully loaded scenario tree data "
                          "for bundle="+bundle_name)

                num_results_so_far += 1

        else:

            num_results_so_far = 0

            while (num_results_so_far < len(self._scenario_tree._scenarios)):

                action_handle = self._action_manager.wait_any()

                if action_handle in initialization_action_handles:
                    initialization_action_handles.remove(action_handle)
                    self._action_manager.get_results(action_handle)
                    continue

                scenario_results = self._action_manager.get_results(action_handle)
                scenario_name = action_handle_to_worker_map[action_handle]

                for tree_node_name, node_data in iteritems(scenario_results['nodes']):
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
                      iteritems(scenario_results['scenarios']):
                    assert have_scenario_data[scenario_name] == False
                    have_scenario_data[scenario_name] = True
                    scenario = self._scenario_tree.get_scenario(scenario_name)
                    scenario._objective_name = scenario_data['_objective_name']
                    scenario._objective_sense = scenario_data['_objective_sense']

                if self._options.verbose:
                    print("Successfully loaded scenario tree data for "
                          "scenario="+scenario_name)

                num_results_so_far += 1

        assert all(have_node_data)
        assert all(have_scenario_data)

        if self._options.output_times:
            print("Scenario tree data collection time=%.2f seconds"
                  % (end_time - start_time))

    #
    # Sends a mapping between (name,index) and ScenarioTreeID so that
    # phsolverservers are aware of the master nodes's ScenarioTreeID
    # labeling.
    #

    def transmit_scenario_tree_ids(self):

        start_time = time.time()

        if self._options.verbose:
            print("Transmitting scenario tree variable information to workers")

        action_handles = []

        generate_responses = self._options.handshake_with_sppyro

        if self._scenario_tree.contains_bundles():

            for bundle in self._scenario_tree._scenario_bundles:

                ids_to_transmit = {}
                for stage in bundle._scenario_tree._stages:
                    for bundle_tree_node in stage._tree_nodes:
                        # The bundle scenariotree usually isn't populated
                        # with variable value data so we need to reference
                        # the original scenariotree node
                        primary_tree_node = self._scenario_tree.\
                                            _tree_node_map[bundle_tree_node._name]
                        ids_to_transmit[primary_tree_node._name] = \
                            primary_tree_node._variable_ids

                action_handles.append(
                    self._action_manager.queue(
                        action="update_master_scenario_tree_ids",
                        generate_response=generate_responses,
                        name=bundle._name,
                        new_ids=ids_to_transmit))

        else:

            for scenario in self._scenario_tree._scenarios:

                ids_to_transmit = {}
                for tree_node in scenario._node_list:
                    ids_to_transmit[tree_node._name] = tree_node._variable_ids

                action_handles.append(
                    self._action_manager.queue(
                        action="update_master_scenario_tree_ids",
                        generate_response=generate_responses,
                        name=scenario._name,
                        new_ids=ids_to_transmit))

        if generate_responses:
            self._action_manager.wait_all(action_handles)

        end_time = time.time()

        if self._options.output_times:
            print("ScenarioTree variable ids transmission time=%.2f "
                  "seconds" % (end_time - start_time))

    #
    # Abstract methods
    #

    def _init(self):
        initialization_action_handles = \
            super(ScenarioTreeSolverManagerSPPyro, self)._init()
        assert self._action_manager is not None
        self._solver_manager = self._action_manager
        scenario_tree_data_action_handle_map = \
            self._request_scenario_tree_data()

        return (scenario_tree_data_action_handle_map,
                initialization_action_handles)

        # TODO
        # Transmit persistent solver options
        #solver_options = {}
        #for key in self._solver.options:
        #    solver_options[key]=self._solver.options[key]

    def _complete_init(self, action_handle_data):

        assert type(action_handle_data) is tuple
        assert len(action_handle_data) == 2
        (scenario_tree_data_action_handle_map,
         initialization_action_handles) = action_handle_map

        self._collect_scenario_tree_data(scenario_tree_data_action_handle_map,
                                         initialization_action_handles)

        if self._options.verbose:
            print("Scenario tree instance data successfully "
                  "collected")

            if self._options.verbose:
                print("Broadcasting scenario tree id mapping"
                      "to scenario tree servers")

            pyomo.pysp.scenariotree.\
                scenariotreeserverutils.transmit_scenario_tree_ids(self)

            if self._options.verbose:
                print("Scenario tree ids successfully sent")

            if self._options.verbose:
                print("ScenarioTreeManagerSPPyro is successfully "
                      "initialized")

        super(ScenarioTreeSolverManagerSPPyro, self).\
                _complete_init(initialization_action_handles)

    def _get_queue_scenario_solve_kwds(self, scenario_name):
        args, kwds = self._get_queue_solve_kwds()
        assert len(args) == 0
        kwds['action'] = "solve_scenario"
        kwds['name'] = scenario_name
        return args, kwds

    def _get_queue_bundle_solve_kwds(self, bundle_name):
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
                self._gaps[bundle_name] = auxilliary_values["gap"]

            if "user_time" in auxilliary_values:
                self._solve_times[bundle_name] = \
                    auxilliary_values["user_time"]
            elif "time" in auxilliary_values:
                self._solve_times[bundle_name] = \
                    auxilliary_values["time"]

        else:
            assert object_type == 'scenarios'
            if len(results) == 0:
                # solve failed
                return "Solve status reported by scenario tree server"

            # TODO: Use these keywords to perform some
            #       validation of fixed variable values in the
            #       results returned
            #allow_consistent_values_for_fixed_vars =\
            #    self._write_fixed_variables,
            #comparison_tolerance_for_fixed_vars =\
            #    self._comparison_tolerance_for_fixed_vars
            # results[0] are variable values
            # results[1] are suffix values
            # results[2] are auxilliary values
            scenario.update_current_solution(results[0])

            auxilliary_values = results[2]
            if "gap" in auxilliary_values:
                self._gaps[scenario_name] = auxilliary_values["gap"]
            if "user_time" in auxilliary_values:
                self._solve_times[scenario_name] = \
                    auxilliary_values["user_time"]
            elif "time" in auxilliary_values:
                self._solve_times[scenario_name] = \
                    auxilliary_values["time"]

        return None
