#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("ScenarioTreeManagerSolverClientSerial",
           "ScenarioTreeManagerSolverClientPyro",
           "ScenarioTreeManagerFactory")

# TODO: handle pyro as the solver manager when even when the
#       pyro scenario tree manager is used

import math
import time

from pyutilib.pyro import shutdown_pyro_components
from pyomo.opt import (UndefinedData,
                       undefined,
                       SolverFactory,
                       SolutionStatus,
                       TerminationCondition)
from pyomo.opt.base.solvers import OptSolver
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_declare_common_option,
                                    safe_register_common_option)
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.scenariotree.preprocessor import ScenarioTreePreprocessor
from pyomo.pysp.scenariotree.manager import \
    (ScenarioTreeManager,
     _ScenarioTreeManagerWorker,
     ScenarioTreeManagerClientSerial,
     ScenarioTreeManagerClientPyro)

from six import itervalues, iteritems

#
# This class adds additional functionality to the ScenarioTreeManager
# manager interface relating to preprocessing and solving of scenarios
# and bundles as well as fixing and freeing scenario tree variables.
#

class ScenarioTreeManagerSolver(ScenarioTreeManager,
                                PySPConfiguredObject):

    _declared_options = \
        PySPConfigBlock("Options declared for the "
                        "ScenarioTreeManagerSolver class")

    #
    # solve and I/O related
    #
    safe_declare_common_option(_declared_options,
                               "symbolic_solver_labels")
    safe_declare_common_option(_declared_options,
                               "mipgap")
    safe_declare_common_option(_declared_options,
                               "solver_options")
    safe_declare_common_option(_declared_options,
                               "solver")
    safe_declare_common_option(_declared_options,
                               "solver_io")
    safe_declare_common_option(_declared_options,
                               "solver_manager")
    safe_declare_common_option(_declared_options,
                               "disable_warmstart")
    safe_declare_common_option(_declared_options,
                               "disable_advanced_preprocessing")
    safe_declare_common_option(_declared_options,
                               "output_solver_log")
    safe_declare_common_option(_declared_options,
                               "output_solver_results")
    safe_declare_common_option(_declared_options,
                               "keep_solver_files")
    safe_declare_common_option(_declared_options,
                               "comparison_tolerance_for_fixed_variables")

    def __init__(self, *args, **kwds):
        if self.__class__ is ScenarioTreeManagerSolver:
            raise NotImplementedError(
                "%s is an abstract class for subclassing" % self.__class__)

        super(ScenarioTreeManagerSolver, self).__init__(*args, **kwds)

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

        # maps scenario name (or bundle name, in the case of
        # bundling) to the solver termination condition
        # associated with the previous round of solves. if
        # there is no entry or it is undefined, then the
        # object was not solved
        self._termination_condition = {}

        # maps scenario name (or bundle name, in the case of
        # bundling) to the solution status associated with
        # the previous round of solves. if there is no entry
        # or it is undefined, then the object was not solved
        # during the previous solve invocation
        self._solution_status = {}

        # the preprocessor for instances (this can be None)
        self._preprocessor = None

    def _solve_objects(self,
                       object_type,
                       objects,
                       update_stages,
                       ephemeral_solver_options,
                       disable_warmstart,
                       exception_on_failure,
                       check_status,
                       async):

        assert object_type in ('bundles', 'scenarios')

        if update_stages is not None:
            for stage_name in update_stages:
                if stage_name not in self._scenario_tree._stage_map:
                    raise ValueError("Can not update solution at stage %s. "
                                     "A scenario tree stage with this name "
                                     "does not exist." % (stage_name))

        # queue the solves
        _async_solve_result = self._queue_object_solves(
            object_type,
            objects,
            update_stages,
            ephemeral_solver_options,
            disable_warmstart)

        result = self.AsyncResultCallback(
            lambda: self._process_solve_results(
                object_type,
                update_stages,
                _async_solve_result.complete(),
                exception_on_failure,
                check_status))
        if not async:
            result = result.complete()
        return result

    #
    # Interface
    #

    @property
    def objective_sense(self):
        """Return the objective sense declared for all
        subproblems."""
        return self._objective_sense

    @property
    def pyomo_solve_times(self):
        """Return a dictionary with the pyomo solve times
        for all objects from the most recent round of solves."""
        return self._pyomo_solve_times

    @property
    def solve_times(self):
        """Return a dictionary with solve times for all
        objects from the most recent round of solves."""
        return self._solve_times

    @property
    def gaps(self):
        """Return a dictionary with solution gaps for all
        objects from the most recent round of solves."""
        return self._gaps

    @property
    def termination_condition(self):
        """Return a dictionary with solver termination conditions
        for all objects from the most recent round of solves."""
        return self._termination_condition

    @property
    def solution_status(self):
        """Return a dictionary with solution statuses for
        all objects from the most recent round of solves."""
        return self._solution_status

    def _report_bundle_objectives(self, scenario_bundles):
        """Report the objective value and other information that
        corresponds to the most recently completed solve of the
        scenario bundles. Raises assertion error if the scenario tree
        does not contain bundles."""
        assert self._scenario_tree.contains_bundles()

        max_name_len = max(len(str(_scenario_bundle_name)) \
                           for _scenario_bundle_name in scenario_bundles)
        max_name_len = max((len("Scenario Bundle"), max_name_len))
        line = (("  %-"+str(max_name_len)+"s    ") % "Scenario Bundle")
        line += ("%-16s %-16s %-16s %-16s"
                 % ("Cost",
                    "Objective",
                    "Objective Gap",
                    "Solution Status "))
        if self._options.output_times or \
           self._options.verbose:
            line += (" %-14s" % ("Solve Time"))
            line += (" %-14s" % ("Pyomo Time"))
        print(line)
        for scenario_bundle_name in scenario_bundles:

            scenario_bundle = self._scenario_tree.get_bundle(scenario_bundle_name)
            bundle_gap = self._gaps[scenario_bundle_name]
            bundle_status = self._solution_status[scenario_bundle_name]
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
            if self._options.output_times or \
               self._options.verbose:
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

    def _report_scenario_objectives(self, scenarios):
        """Report the objective value and other information that
        corresponds to the most recently completed solve of the
        scenario (or bundles they are contained in)."""

        max_name_len = max(len(str(_scenario_name)) \
                           for _scenario_name in scenarios)
        max_name_len = max((len("Scenario"), max_name_len))
        line = (("  %-"+str(max_name_len)+"s    ") % "Scenario")
        line += ("%-16s %-16s %-16s %-16s"
                 % ("Cost",
                    "Objective",
                    "Objective Gap",
                    "Solution Status "))
        if self._options.output_times or \
           self._options.verbose:
            line += (" %-14s" % ("Solve Time"))
            line += (" %-14s" % ("Pyomo Time"))
        print(line)
        for scenario_name in scenarios:
            scenario = self._scenario_tree.get_scenario(scenario_name)
            objective_value = scenario._objective
            scenario_cost = scenario._cost
            gap = self._gaps.get(scenario_name)
            status = self._solution_status[scenario_name]
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
            line %= (scenario_name,
                     scenario_cost,
                     objective_value,
                     gap,
                     status)
            if self._options.output_times or \
               self._options.verbose:
                solve_time = self._solve_times.get(scenario_name)
                if (not isinstance(solve_time, UndefinedData)) and \
                   (solve_time is not None):
                    line += (" %-14.2f"
                             % (solve_time))
                else:
                    line += (" %-14s" % "None Reported")
                pyomo_solve_time = self._pyomo_solve_times.get(scenario_name)
                if (not isinstance(pyomo_solve_time, UndefinedData)) and \
                   (pyomo_solve_time is not None):
                    line += (" %-14.2f"
                             % (pyomo_solve_time))
                else:
                    line += (" %-14s" % "None Reported")
            print(line)
        print("")

    def solve_subproblems(self,
                          subproblems=None,
                          **kwds):
        """Solve scenarios or bundles (if they exist)."""
        ret = None
        if self._scenario_tree.contains_bundles():
            ret = self.solve_bundles(bundles=subproblems,
                                     **kwds)
        else:
            ret = self.solve_scenarios(scenarios=subproblems,
                                       **kwds)
        return ret

    def solve_scenarios(self,
                        scenarios=None,
                        update_stages=None,
                        ephemeral_solver_options=None,
                        disable_warmstart=False,
                        exception_on_failure=False,
                        check_status=True,
                        async=False):
        """Solve scenarios (ignoring bundles even if they exists)."""
        return self._solve_objects('scenarios',
                                   scenarios,
                                   update_stages,
                                   ephemeral_solver_options,
                                   disable_warmstart,
                                   exception_on_failure,
                                   check_status,
                                   async)

    def solve_bundles(self,
                      bundles=None,
                      update_stages=None,
                      ephemeral_solver_options=None,
                      disable_warmstart=False,
                      exception_on_failure=False,
                      check_status=True,
                      async=False):
        """Solve the bundles (they must exists)."""
        if not self._scenario_tree.contains_bundles():
            raise RuntimeError(
                "Unable to solve bundles. No bundles exist")
        return self._solve_objects('bundles',
                                   bundles,
                                   update_stages,
                                   ephemeral_solver_options,
                                   disable_warmstart,
                                   exception_on_failure,
                                   check_status,
                                   async)

    def _process_solve_results(self,
                               object_type,
                               update_stages,
                               solve_results,
                               exception_on_failure,
                               check_status):
        """Process and load previously queued solve results."""
        failures = []

        assert object_type in ('bundles', 'scenarios')
        _process_function = self._process_bundle_solve_result \
                            if (object_type == 'bundles') else \
                               self._process_scenario_solve_result

        self._solve_times = {}
        self._pyomo_solve_times = {}
        self._gaps = {}
        self._termination_condition = {}
        self._solution_status = {}
        for object_name in solve_results:

            results = solve_results[object_name]

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
            #  - self._termination_condition
            #  - self._solver_results
            #  - solutions on scenario objects
#            self._gaps[object_name] = undefined
#            self._solution_status[object_name] = undefined
#            self._termination_condition[object_name] = undefined
#            self._solve_times[object_name] = undefined
#            self._pyomo_solve_times[object_name] = undefined
#            if object_type == 'scenarios':
#                if self._scenario_tree.contains_bundles():
#                    _bundle_name = self._scenario_to_bundle_map[object_name]
#                    self._gaps[_bundle_name] = undefined
#                    self._solution_status[_bundle_name] = undefined
#                    self._termination_condition[_bundle_name] = undefined
#                    self._solve_times[_bundle_name] = undefined
#                    self._pyomo_solve_times[_bundle_name] = undefined
#            else:
#                assert object_type == 'bundles'
#                for _scenario_name in self._scenario_tree.\
#                       get_bundle(object_name).scenario_names:
#                    self._gaps[_scenario_name] = undefined
#                    self._solution_status[_scenario_name] = undefined
#                    self._termination_condition[_scenario_name] = undefined
#                    self._solve_times[_scenario_name] = undefined
#                    self._pyomo_solve_times[_scenario_name] = undefined
            _process_function(object_name, update_stages, results)

            if check_status:
                if ((self._solution_status[object_name] !=
                    SolutionStatus.optimal) or \
                    (self._termination_condition[object_name] != \
                     TerminationCondition.optimal)):

                    failures.append(object_name)
                    if self._options.verbose:
                        print("Solve failed for %s=%s"
                              % (object_type[:-1], object_name))
                else:
                    if self._options.verbose:
                        print("Successfully completed solve for %s=%s"
                              % (object_type[:-1], object_name))
            else:
                if self._options.verbose:
                    print("Solve for %s=%s has completed. "
                          "Skipping status check."
                          % (object_type[:-1], object_name))

            if self._options.output_times or \
               self._options.verbose:
                print("Time loading results for %s %s=%0.2f seconds"
                      % (object_type[:-1],
                         object_name,
                         time.time() - start_load))

        if len(failures) > 0:
            if self._options.verbose:
                print(" ** At least one of the %s failed to solve! ** "
                      % (object_type))
                print(" Failed %s:" % (object_type))
                for failure in sorted(failures):
                    print("   "+str(failure))
            if exception_on_failure:
                raise RuntimeError("Failed to obtain a solution for "
                                   "the following %s: %s"
                                   % (object_type, str(failures)))

        if self._options.verbose:
            if object_type == 'scenarios':
                self._report_scenario_objectives(solve_results)
            else:
                assert object_type == 'bundles'
                self._report_bundle_objectives(solve_results)

        if self._options.output_times or \
           self._options.verbose:
            if len(failures) > 0:
                print("Skipping timing statistics due to one or more "
                      "solve failures" % (object_type))
            else:
                # if any of the solve times are of type
                # pyomo.opt.results.container.UndefinedData, then don't
                # output timing statistics.
                solve_times = [self._solve_times[object_name]
                               for object_name in solve_results]
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
                                     for object_name in solve_results]
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

    def push_fix_queue_to_instances(self):
        """Pushed the fixed queue on the scenario tree nodes onto the
        actual variables on the scenario instances.

        * NOTE: This function is poorly named and this functionality
                will likely be changed in the near future. Ideally, fixing
                would be done through the scenario tree manager, rather
                than through the scenario tree (with this function being invoked
                afterward.)

        """
        if self._options.verbose:
            print("Synchronizing fixed variable statuses on scenario tree nodes")
        node_count = self._push_fix_queue_to_instances_impl()
        if node_count > 0:
            if self._options.verbose:
                print("Updated fixed statuses on %s scenario tree nodes"
                      % (node_count))
        else:
            if self._options.verbose:
                print("No synchronization was needed for scenario tree nodes")

    #
    # Methods defined by derived class that are not
    # part of the user interface
    #

    def _queue_object_solves(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _process_bundle_solve_result(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _process_scenario_solve_result(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _push_fix_queue_to_instances_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

#
# A partial implementation of the ScenarioTreeManagerSolver
# interface that is common to both the Serial scenario
# tree manager solver as well as the Pyro-based manager solver worker
#

class _ScenarioTreeManagerSolverWorker(_ScenarioTreeManagerWorker,
                                       PySPConfiguredObject):

    _declared_options = \
        PySPConfigBlock("Options declared for the "
                        "_ScenarioTreeManagerSolverWorker class")

    # options for controlling the solver manager
    # (not the scenario tree manager)
    safe_declare_common_option(_declared_options,
                               "pyro_host")
    safe_declare_common_option(_declared_options,
                               "pyro_port")

    ScenarioTreePreprocessor.register_options(_declared_options)

    @property
    def preprocessor(self):
        return self._preprocessor

    def __init__(self, *args, **kwds):
        if self.__class__ is _ScenarioTreeManagerSolverWorker:
            raise NotImplementedError(
                "%s is an abstract class for subclassing" % self.__class__)

        super(_ScenarioTreeManagerSolverWorker, self).__init__(*args, **kwds)

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

    # invoke after the scenario and bundle instances have been created
    def _init_solver_worker(self):

        # initialize the preprocessor
        self._preprocessor = None
        if not self._options.disable_advanced_preprocessing:
            self._preprocessor = ScenarioTreePreprocessor(self._options)

        # initialize the solver manager
        self._solver_manager = SolverManagerFactory(
            self._options.solver_manager,
            host=self.get_option('pyro_host'),
            port=self.get_option('pyro_port'))
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

        self._objective_sense = \
            self._scenario_tree._scenarios[0]._objective_sense
        assert all(_s._objective_sense == self._objective_sense
                   for _s in self._scenario_tree._scenarios)

    #
    # Creates a deterministic symbol map for variables on an
    # instance. This allows convenient transmission of information to
    # and from ScenarioTreeSolverWorkers and makes it easy to save solutions
    # using a pickleable dictionary of symbols -> values
    #
    def _create_instance_symbol_maps(self, ctypes):

        for instance in itervalues(self._instances):

            create_block_symbol_maps(instance, ctypes)

    #
    # Override some methods for ScenarioTreeManager that
    # were implemented by _ScenarioTreeManagerWorker:
    #

    def _close_impl(self):
        super(_ScenarioTreeManagerSolverWorker, self)._close_impl()
        if self._solver_manager is not None:
            self._solver_manager.deactivate()
            self._solver_manager = None
        for solver in self._scenario_solvers.values():
            solver.deactivate()
        self._scenario_solvers = {}
        for solver in self._bundle_solvers.values():
            solver.deactivate()
        self._bundle_solvers = {}
        self._preprocessor = None
        self._objective_sense = None

    # Adds to functionality to _ScenarioTreeManagerWorker by
    # registering the bundle instance with the preprocessor and
    # creating a solver for the bundle
    def _add_bundle_impl(self, bundle_name, scenario_list):
        super(_ScenarioTreeManagerSolverWorker, self)._add_bundle_impl(bundle_name,
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

    # Adds to functionality to _ScenarioTreeManagerWorker by
    # registering the bundle instance with the preprocessor and
    # creating a solver for the bundle
    def _remove_bundle_impl(self, bundle_name):
        assert bundle_name in self._bundle_solvers
        if self._preprocessor is not None:
            self._preprocessor.remove_bundle(
                self._scenario_tree.get_bundle(bundle_name))
        self._bundle_solvers[bundle_name].deactivate()
        del self._bundle_solvers[bundle_name]
        super(_ScenarioTreeManagerSolverWorker, self).\
            _remove_bundle_impl(bundle_name)

    #
    # Abstract methods for ScenarioTreeManagerSolver:
    #

    def _queue_object_solves(self,
                             object_type,
                             objects,
                             update_stages,
                             ephemeral_solver_options,
                             disable_warmstart):

        if self._options.verbose:
            print("Queuing %s solves" % (object_type[:-1]))

        assert object_type in ('bundles', 'scenarios')
        # unused: update_stages

        solver_dict = None
        instance_dict = None
        if object_type == 'bundles':
            if objects is None:
                objects = self._scenario_tree._scenario_bundle_map
            solver_dict = self._bundle_solvers
            instance_dict = self._bundle_binding_instance_map
            for bundle_name in objects:
                for scenario_name in self._scenario_tree.\
                    get_bundle(bundle_name).\
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
        common_kwds['tee'] = self._options.output_solver_log
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
        if type(self._options.solver_options) is tuple:
            solver_options.update(
                OptSolver._options_string_to_dict(
                    "".join(self._options.solver_options)))
        else:
            solver_options.update(self._options.solver_options)
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

    def _process_bundle_solve_result(self,
                                     bundle_name,
                                     update_stages,
                                     results):

        bundle = self._scenario_tree.get_bundle(bundle_name)
        bundle_instance = self._bundle_binding_instance_map[bundle.name]

        if self._options.output_solver_results:
            print("Results for bundle=%s" % (bundle_name))
            results.write(num=1)

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
            self._solve_times[bundle_name] = \
                float(results.solver.time)
        else:
            self._solve_times[bundle_name] = undefined

        if hasattr(results,"pyomo_solve_time"):
            self._pyomo_solve_times[bundle_name] = \
                results.pyomo_solve_time
        else:
            self._pyomo_solve_times[bundle_name] = undefined

        self._termination_condition[bundle_name] = \
            results.solver.termination_condition

        if len(results.solution) > 0:
            assert len(results.solution) == 1

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
            else:
                self._gaps[bundle_name] = undefined

            self._solution_status[bundle_name] = solution0.status

            for scenario_name in bundle._scenario_names:
                scenario = self._scenario_tree._scenario_map[scenario_name]
                scenario.update_solution_from_instance(stages=update_stages)

        else:

            self._gaps[bundle_name] = undefined
            self._solution_status[bundle_name] = undefined

    def _process_scenario_solve_result(self,
                                       scenario_name,
                                       update_stages,
                                       results):

        scenario = self._scenario_tree.get_scenario(scenario_name)
        scenario_instance = scenario._instance
        if self._scenario_tree.contains_bundles():
            scenario._instance_objective.deactivate()

        if self._options.output_solver_results:
            print("Results for scenario="+scenario_name)
            results.write(num=1)

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
        else:
            self._solver_times[scenario_name] = undefined

        if hasattr(results,"pyomo_solve_time"):
            self._pyomo_solve_times[scenario_name] = \
                results.pyomo_solve_time
        else:
            self._pyomo_solve_times[scenario_name] = undefined

        self._termination_condition[scenario_name] = \
            results.solver.termination_condition

        if len(results.solution) > 0:
            assert len(results.solution) == 1

            results_sm = results._smap
            scenario_instance.solutions.load_from(
                results,
                allow_consistent_values_for_fixed_vars=\
                   not self._options.preprocess_fixed_variables,
                comparison_tolerance_for_fixed_vars=\
                   self._options.comparison_tolerance_for_fixed_variables,
                ignore_fixed_vars=self._options.preprocess_fixed_variables)
            self._solver_results[scenario.name] = (results, results_sm)

            scenario.update_solution_from_instance(stages=update_stages)

            solution0 = results.solution(0)
            if hasattr(solution0, "gap") and \
               (solution0.gap is not None):
                self._gaps[scenario_name] = solution0.gap
            else:
                self._gaps[scenario_name] = undefined

            self._solution_status[scenario_name] = solution0.status

        else:

            self._gaps[scenario_name] = undefined
            self._solution_status[scenario_name] = undefined

    def _push_fix_queue_to_instances_impl(self):

        node_count = 0
        for tree_node in self._scenario_tree._tree_nodes:

            if len(tree_node._fix_queue):
                node_count += 1
                if self._preprocessor is not None:
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

        return node_count

class ScenarioTreeManagerSolverClientSerial(ScenarioTreeManagerClientSerial,
                                            _ScenarioTreeManagerSolverWorker,
                                            ScenarioTreeManagerSolver,
                                            PySPConfiguredObject):

    _declared_options = \
        PySPConfigBlock("Options declared for the "
                        "ScenarioTreeManagerSolverClientSerial class")

    safe_declare_common_option(_declared_options,
                               "pyro_shutdown")

    def __init__(self, *args, **kwds):
        super(ScenarioTreeManagerSolverClientSerial, self).\
            __init__(*args, **kwds)

    #
    # Override some methods for ScenarioTreeManager that
    # were implemented by _ScenarioTreeManagerWorkerSolver:
    #

    def _close_impl(self):
        super(ScenarioTreeManagerSolverClientSerial, self)._close_impl()
        if self._options.pyro_shutdown:
            print("Shutting down Pyro components.")
            shutdown_pyro_components(
                host=self._options.pyro_host,
                port=self._options.pyro_port,
                num_retries=0,
                caller_name=self.__class__.__name__)

    #
    # Override methods for ScenarioTreeManagerClient that
    # were implemented by ScenarioTreeManagerSerial:
    #

    def _init_client(self):
        handle = super(ScenarioTreeManagerSolverClientSerial, self).\
                 _init_client()
        # construct the preprocessor and solver manager
        # (this is implemented by _ScenarioTreeManagerSolverWorker)
        super(ScenarioTreeManagerSolverClientSerial, self).\
            _init_solver_worker()
        return handle

    #
    # Abstract methods for ScenarioTreeManagerSolver:
    #

    # implemented by _ScenarioTreeManagerSolverWorker
    #def _queue_object_solves(...)

    # implemented by _ScenarioTreeManagerSolverWorker
    #def _process_bundle_solve_result(...)

    # implemented by _ScenarioTreeManagerSolverWorker
    #def _process_scenario_solve_result(...)

    # implemented by _ScenarioTreeManagerSolverWorker
    #def _push_fix_queue_to_instances_impl(...)

class ScenarioTreeManagerSolverClientPyro(ScenarioTreeManagerClientPyro,
                                          ScenarioTreeManagerSolver):

    _declared_options = \
        PySPConfigBlock("Options declared for the "
                        "ScenarioTreeManagerSolverClientPyro class")

    default_registered_worker_name = 'ScenarioTreeManagerSolverWorkerPyro'

    def __init__(self, *args, **kwds):
        super(ScenarioTreeManagerSolverClientPyro, self).\
            __init__(*args, **kwds)

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
                    tree_node._variable_ids.update(
                        node_data['_variable_ids'])
                    tree_node._standard_variable_ids.update(
                        node_data['_standard_variable_ids'])
                    tree_node._variable_indices.update(
                        node_data['_variable_indices'])
                    tree_node._integer.update(node_data['_integer'])
                    tree_node._binary.update(node_data['_binary'])
                    tree_node._semicontinuous.update(
                        node_data['_semicontinuous'])
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
                    tree_node._variable_ids.update(
                        node_data['_variable_ids'])
                    tree_node._standard_variable_ids.update(
                        node_data['_standard_variable_ids'])
                    tree_node._variable_indices.update(
                        node_data['_variable_indices'])
                    tree_node._integer.update(node_data['_integer'])
                    tree_node._binary.update(node_data['_binary'])
                    tree_node._semicontinuous.update(
                        node_data['_semicontinuous'])
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

        self._objective_sense = \
            self._scenario_tree._scenarios[0]._objective_sense
        assert all(_s._objective_sense == self._objective_sense
                   for _s in self._scenario_tree._scenarios)

        assert all(itervalues(have_node_data))
        assert all(itervalues(have_scenario_data))

        if self._options.verbose:
            print("Scenario tree instance data successfully "
                  "collected")

        if self._options.output_times or \
           self._options.verbose:
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

        if self._options.output_times or \
           self._options.verbose:
            print("Scenario tree variable ids broadcast time=%.2f "
                  "seconds" % (time.time() - start_time))

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
    # Override methods for ScenarioTreeManagerClient that
    # were implemented by ScenarioTreeManagerPyro:
    #

    def _init_client(self):
        init_handle = \
            super(ScenarioTreeManagerSolverClientPyro, self)._init_client()
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

    #
    # Abstract methods for ScenarioTreeManagerSolver:
    #

    def _queue_object_solves(self,
                             object_type,
                             objects,
                             update_stages,
                             ephemeral_solver_options,
                             disable_warmstart):

        assert object_type in ('bundles', 'scenarios')

        if self._options.verbose:
            print("Transmitting solve requests for %s" % (object_type))

        worker_names = None
        worker_map = {}
        if objects is not None:
            if object_type == 'bundles':
                _get_worker_func = self.get_worker_for_bundle
            else:
                assert object_type == 'scenarios'
                _get_worker_func = self.get_worker_for_scenario
            for object_name in objects:
                worker_name = _get_worker_func(object_name)
                if worker_name not in worker_map:
                    worker_map[worker_name] = []
                worker_map[worker_name].append(object_name)
            worker_names = worker_map
        else:
            worker_names = self._pyro_worker_list

        was_paused = self.pause_transmit()
        action_handle_data = {}
        for worker_name in worker_names:
            action_handle_data[self._invoke_method_on_worker_pyro(
                worker_name,
                "_solve_objects_for_client",
                method_args=(object_type,
                             worker_map.get(worker_name, None),
                             update_stages,
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

    # TODO: Use these keywords to perform some
    #       validation of fixed variable values in the
    #       results in the following two methods below
    #allow_consistent_values_for_fixed_vars=\
    #   not self._options.preprocess_fixed_variables,
    #comparison_tolerance_for_fixed_vars=\
    #   self._options.comparison_tolerance_for_fixed_variables,

    def _process_bundle_solve_result(self,
                                     bundle_name,
                                     update_stages,
                                     results):

        # Note: update_stages is not used in this function

        auxiliary_values = results['auxiliary_values']
        self._solve_times[bundle_name] = \
            auxiliary_values['solve_time']
        self._pyomo_solve_times[bundle_name] = \
            auxiliary_values['pyomo_solve_time']
        self._gaps[bundle_name] = auxiliary_values['gaps']
        self._termination_condition[bundle_name] = \
            getattr(TerminationCondition,
                    auxiliary_values['termination_condition'])
        self._solution_status[bundle_name] = \
            getattr(SolutionStatus,
                    auxiliary_values['solution_status'])

        solution = results['solution']
        for scenario_name in self._scenario_tree.\
            get_bundle(bundle_name).scenario_names:
            scenario = self._scenario_tree.get_scenario(scenario_name)
            scenario.set_solution(solution[scenario_name])

    def _process_scenario_solve_result(self,
                                       scenario_name,
                                       update_stages,
                                       results):

        # Note: update_stages is not used in this function

        auxiliary_values = results['auxiliary_values']
        self._solve_times[scenario_name] = \
            auxiliary_values['solve_time']
        self._pyomo_solve_times[scenario_name] = \
            auxiliary_values['pyomo_solve_time']
        self._gaps[scenario_name] = auxiliary_values['gaps']
        self._termination_condition[scenario_name] = \
            getattr(TerminationCondition,
                    auxiliary_values['termination_condition'])
        self._solution_status[scenario_name] = \
            getattr(SolutionStatus,
                    auxiliary_values['solution_status'])

        scenario = self._scenario_tree.get_scenario(scenario_name)
        scenario.set_solution(results['solution'])

    def _push_fix_queue_to_instances_impl(self):

        worker_map = {}
        node_count = 0
        for stage in self._scenario_tree.stages:

            for tree_node in stage.nodes:

                if len(tree_node._fix_queue):
                    node_count += 1
                    for scenario in tree_node.scenarios:
                        worker_name = self.get_worker_for_scenario(scenario.name)
                        if worker_name not in worker_map:
                            worker_map[worker_name] = {}
                        if tree_node.name not in worker_map[worker_name]:
                            worker_map[worker_name][tree_node.name] = tree_node._fix_queue

        if node_count > 0:

            was_paused = self.pause_transmit()
            for worker_name in worker_map:
                self._invoke_method_on_worker_pyro(
                    worker_name,
                    "_update_fixed_variables_for_client",
                    method_args=(worker_map[worker_name],),
                    oneway=True)

            if not was_paused:
                self.unpause_transmit()

        return node_count

def ScenarioTreeManagerFactory(options):
    if options.scenario_tree_manager == "serial":
        manager = ScenarioTreeManagerSolverClientSerial(options)
    elif options.scenario_tree_manager == "pyro":
        manager = ScenarioTreeManagerSolverClientPyro(options)
    else:
        raise ValueError("Unrecognized value for option '%s': %s"
                         % ("scenario_tree_manager",
                            options.scenario_tree_manager))
    return manager

def _register_scenario_tree_manager_options(*args, **kwds):
    if len(args) == 0:
        options = PySPConfigBlock()
    else:
        if len(args) != 1:
            raise TypeError(
                "register_options(...) takes at most 1 argument (%s given)"
                % (len(args)))
        options = args[0]
        if not isinstance(options, PySPConfigBlock):
            raise TypeError(
                "register_options(...) argument must be of type PySPConfigBlock, "
                "not %s" % (type(options).__name__))
    safe_register_common_option(options,
                                "scenario_tree_manager",
                                **kwds)
    ScenarioTreeManagerSolverClientSerial.register_options(options,
                                                           **kwds)
    ScenarioTreeManagerSolverClientPyro.register_options(options,
                                                         **kwds)
    return options

ScenarioTreeManagerFactory.register_options = \
    _register_scenario_tree_manager_options
