#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ("ScenarioTreeManagerSolverClientSerial",
           "ScenarioTreeManagerSolverClientPyro",
           "ScenarioTreeManagerSolverFactory")

# TODO: handle pyro as the solver manager when even when the
#       pyro scenario tree manager is used

import math
import time
import sys

from pyutilib.pyro import shutdown_pyro_components
from pyomo.opt import (SolverFactory,
                       SolverStatus,
                       TerminationCondition,
                       SolutionStatus)
from pyomo.opt.base.solvers import OptSolver
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_declare_common_option,
                                    safe_register_common_option)
from pyomo.pysp.util.configured_object import \
    PySPConfiguredObject
from pyomo.pysp.scenariotree.preprocessor import \
    ScenarioTreePreprocessor
from pyomo.pysp.scenariotree.server_pyro \
    import ScenarioTreeServerPyro
from pyomo.pysp.scenariotree.manager import \
    (ScenarioTreeManager,
     _ScenarioTreeManagerWorker,
     ScenarioTreeManagerClientSerial,
     ScenarioTreeManagerClientPyro,
     ScenarioTreeSolveResults)

from six import itervalues, iteritems

#
# The ScenarioTreeManagerSolver interface adds additional
# functionality to the ScenarioTreeManager manager interface
# relating to preprocessing and solving of scenarios and
# bundles.
#

class PySPFailedSolveStatus(RuntimeError):
    """This exception gets raised when one or more
    subproblem solve statuses fail basic status checks when
    processing the results of solve requests by the
    ScenarioTreeManagerSolver."""
    def __init__(self, failures, *args, **kwds):
        super(PySPFailedSolveStatus, self).__init__(*args, **kwds)
        assert type(failures) in (list, tuple)
        self.failures = tuple(failures)

class ScenarioTreeManagerSolver(PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()

        #
        # solve and I/O related
        #
        safe_declare_common_option(options,
                                   "symbolic_solver_labels")
        safe_declare_common_option(options,
                                   "solver_options")
        safe_declare_common_option(options,
                                   "solver")
        safe_declare_common_option(options,
                                   "solver_io")
        safe_declare_common_option(options,
                                   "solver_manager")
        safe_declare_common_option(options,
                                   "disable_warmstart")
        safe_declare_common_option(options,
                                   "disable_advanced_preprocessing")
        safe_declare_common_option(options,
                                   "output_solver_log")
        safe_declare_common_option(options,
                                   "keep_solver_files")
        safe_declare_common_option(options,
                                   "comparison_tolerance_for_fixed_variables")

        return options

    def __init__(self, manager, *args, **kwds):
        if self.__class__ is ScenarioTreeManagerSolver:
            raise NotImplementedError(
                "%s is an abstract class for subclassing" % self.__class__)

        super(ScenarioTreeManagerSolver, self).__init__(*args, **kwds)
        self._manager = manager

    def _solve_objects(self,
                       object_type,
                       objects,
                       ephemeral_solver_options,
                       disable_warmstart,
                       check_status,
                       async_call):

        assert object_type in ('bundles', 'scenarios')

        # queue the solves
        _async_solve_result = self._queue_object_solves(
            object_type,
            objects,
            ephemeral_solver_options,
            disable_warmstart)

        result = self.manager.AsyncResultCallback(
            lambda: self._process_solve_results(
                object_type,
                _async_solve_result.complete(),
                check_status))
        if not async_call:
            result = result.complete()
        return result

    #
    # Interface
    #

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the scenario tree manager solver and any
        associated objects."""
        if self.get_option("verbose"):
            print("Closing "+str(self.__class__.__name__))
        self._close_impl()

    @property
    def manager(self):
        """Return the scenario tree manager that this object
        owns"""
        return self._manager

    def solve_subproblems(self,
                          subproblems=None,
                          **kwds):
        """Solve scenarios or bundles (if they exist).

        Args:
            subproblems (list): The list of subproblem names
                to solve. The default value of :const:`None`
                indicates that all subproblems should be
                solved. Note that if the scenario tree
                contains bundles, this should be a list of
                bundles names; otherwise it should be a list
                of scenario names.
            ephemeral_solver_options (dict): A dictionary of
                solver options to override any persistent
                solver options for this set of solves only.
            disable_warmstart (bool): Disable any warmstart
                functionality available for the selected
                subproblem solvers. Default is
                :const:`False`.
            check_status (bool): Verify that all subproblem
                solves successfully completed (optimal or
                feasible solutions are loaded). Default is
                :const:`True`. This option is meant to help
                users catch errors early on and should be
                set to :const:`False` in situations where
                more advanced status handling is necessary.
            async_call (bool): When set to :const:`True`, an
                async results object is returned that allows
                the solves to be completed
                asynchronously. Default is
                :const:`False`. Note that completing an
                asynchronous solve results in modification
                of the scenario tree state on this manager
                (i.e., solutions are loaded into the
                scenarios).

        Returns:
            A :class:`ScenarioTreeSolveResults` object storing \
            basic status information for each subproblem. If \
            the :attr:`async_call` keyword is set to :const:`True` \
            then an :class:`AsyncResult` object is returned.

        Examples:
            The following lines solve all subproblems
            (automatically validating all solutions are
            optimal or feasible) and then prints a summary
            of the results.

            >>> results = sp.solve_subproblems()
            >>> results.pprint()

            The following lines do the same by first
            initiating an asynchronous solve request.

            >>> job = sp.solve_subproblems(async_call=True)
            >>> # ... do other things ... #
            >>> results = job.complete()
            >>> results.pprint()

        Raises:
            :class:`PySPFailedSolveStatus` if the \
            :attr:`check_status` keyword is :const:`True` \
            and any solves fail.
        """
        ret = None
        if self.manager.scenario_tree.contains_bundles():
            ret = self.solve_bundles(bundles=subproblems,
                                     **kwds)
        else:
            ret = self.solve_scenarios(scenarios=subproblems,
                                       **kwds)
        return ret

    def solve_scenarios(self,
                        scenarios=None,
                        ephemeral_solver_options=None,
                        disable_warmstart=False,
                        check_status=True,
                        async_call=False):
        """Solve scenarios (ignoring bundles even if they exists).

        Args:
            scenarios (list): The list of scenario names to
                solve. The default value of :const:`None`
                indicates that all scenarios should be
                solved.
            ephemeral_solver_options (dict): A dictionary of
                solver options to override any persistent
                solver options for this set of solves only.
            disable_warmstart (bool): Disable any warmstart
                functionality available for the selected
                subproblem solvers. Default is
                :const:`False`.
            check_status (bool): Verify that all subproblem
                solves successfully completed (optimal or
                feasible solutions are loaded). Default is
                :const:`True`. This option is meant to help
                users catch errors early on and should be
                set to :const:`False` in situations where
                more advanced status handling is necessary.
            async_call (bool): When set to :const:`True`, an
                async results object is returned that allows
                the solves to be completed
                asynchronously. Default is
                :const:`False`. Note that completing an
                asynchronous solve results in modification
                of the scenario tree state on this manager
                (i.e., solutions are loaded into the
                scenarios).

        Returns:
            A :class:`ScenarioTreeSolveResults` object storing \
            basic status information for each subproblem. If \
            the :attr:`async_call` keyword is set to :const:`True` \
            then an :class:`AsyncResult` object is returned.

        Examples:
            The following lines solve all scenarios
            (automatically validating all solutions are
            optimal or feasible) and then prints a summary
            of the results.

            >>> results = sp.solve_scenarios()
            >>> results.pprint()

            The following lines do the same by first
            initiating an asynchronous solve request.

            >>> job = sp.solve_scenarios(async_call=True)
            >>> # ... do other things ... #
            >>> results = job.complete()
            >>> results.pprint()

        Raises:
            :class:`PySPFailedSolveStatus` if the \
            :attr:`check_status` keyword is :const:`True` \
            and any solves fail.
        """
        return self._solve_objects('scenarios',
                                   scenarios,
                                   ephemeral_solver_options,
                                   disable_warmstart,
                                   check_status,
                                   async_call)

    def solve_bundles(self,
                      bundles=None,
                      ephemeral_solver_options=None,
                      disable_warmstart=False,
                      check_status=True,
                      async_call=False):
        """Solve bundles (they must exists).

        Args:
            bundles (list): The list of bundle names to
                solve. The default value of :const:`None`
                indicates that all bundles should be
                solved.
            ephemeral_solver_options (dict): A dictionary of
                solver options to override any persistent
                solver options for this set of solves only.
            disable_warmstart (bool): Disable any warmstart
                functionality available for the selected
                subproblem solvers. Default is
                :const:`False`.
            check_status (bool): Verify that all subproblem
                solves successfully completed (optimal or
                feasible solutions are loaded). Default is
                :const:`True`. This option is meant to help
                users catch errors early on and should be
                set to :const:`False` in situations where
                more advanced status handling is necessary.
            async_call (bool): When set to :const:`True`, an
                async results object is returned that allows
                the solves to be completed
                asynchronously. Default is
                :const:`False`. Note that completing an
                asynchronous solve results in modification
                of the scenario tree state on this manager
                (i.e., solutions are loaded into the
                scenarios).

        Returns:
            A :class:`ScenarioTreeSolveResults` object storing \
            basic status information for each subproblem. If \
            the :attr:`async_call` keyword is set to :const:`True` \
            then an :class:`AsyncResult` object is returned.

        Examples:
            The following lines solve all bundles
            (automatically validating all solutions are
            optimal or feasible) and then prints a summary
            of the results.

            >>> results = sp.solve_bundles()
            >>> results.pprint()

            The following lines do the same by first
            initiating an asynchronous solve request.

            >>> job = sp.solve_bundles(async_call=True)
            >>> # ... do other things ... #
            >>> results = job.complete()
            >>> results.pprint()

        Raises:
            :class:`PySPFailedSolveStatus` if the \
            :attr:`check_status` keyword is :const:`True` \
            and any solves fail, or :class:`RuntimeError` if
            the scenario tree was not created with bundles.
        """

        if not self.manager.scenario_tree.contains_bundles():
            raise RuntimeError(
                "Unable to solve bundles. No bundles exist")
        return self._solve_objects('bundles',
                                   bundles,
                                   ephemeral_solver_options,
                                   disable_warmstart,
                                   check_status,
                                   async_call)

    def _process_solve_results(self,
                               object_type,
                               solve_results,
                               check_status):
        """Process and load previously queued solve results."""
        assert object_type in ('bundles', 'scenarios')
        _process_function = self.manager._process_bundle_solve_result \
                            if (object_type == 'bundles') else \
                               self.manager._process_scenario_solve_result

        manager_results = ScenarioTreeSolveResults(object_type)
        failures = []
        for object_name in solve_results:

            results = solve_results[object_name]

            if self.get_option("verbose"):
                print("Processing results for %s=%s"
                      % (object_type[:-1],
                         object_name))

            start_load = time.time()
            #
            # This method is expected to:
            #  - update the dictionaries on the manager_results object
            #    (e.g., objective, gap, solution_status)
            #  - update solutions on scenario tree objects
            _process_function(
                object_name,
                results,
                manager_results=manager_results,
                allow_consistent_values_for_fixed_vars=\
                    not self.get_option("preprocess_fixed_variables"),
                comparison_tolerance_for_fixed_vars=\
                    self.get_option("comparison_tolerance_for_fixed_variables"),
                ignore_fixed_vars=\
                    self.get_option("preprocess_fixed_variables"))

            if check_status:
                if not (((manager_results.solution_status[object_name] ==
                          SolutionStatus.optimal) or \
                         (manager_results.solution_status[object_name] ==
                          SolutionStatus.feasible)) and \
                        ((manager_results.solver_status[object_name] == \
                          SolverStatus.ok) or
                         (manager_results.solver_status[object_name] == \
                          SolverStatus.warning))):
                    failures.append(object_name)
                    if self.get_option("verbose"):
                        print("Solve failed for %s=%s"
                              % (object_type[:-1], object_name))
                else:
                    if self.get_option("verbose"):
                        print("Successfully completed solve for %s=%s"
                              % (object_type[:-1], object_name))
            else:
                if self.get_option("verbose"):
                    print("Solve for %s=%s has completed. "
                          "Skipping status check."
                          % (object_type[:-1], object_name))

            if self.get_option("output_times") or \
               self.get_option("verbose"):
                print("Time loading results for %s %s=%0.2f seconds"
                      % (object_type[:-1],
                         object_name,
                         time.time() - start_load))

        if len(failures) > 0:
            print(" ** At least one of the %s failed to solve! ** "
                  % (object_type))
            print("Reporting statues for failed %s:" % (object_type))
            manager_results.pprint_status(filter_names=lambda x: x in failures)
            raise PySPFailedSolveStatus(
                failures,
                "Solve status check failed for %s %s"
                % (len(failures), object_type))

        if self.get_option("verbose"):
            manager_results.pprint(
                output_times=self.get_option("output_times"))

        if self.get_option("output_times") or \
           self.get_option("verbose"):
            if len(failures) > 0:
                print("Skipping timing statistics due to one or more "
                      "solve failures" % (object_type))
            else:
                manager_results.print_timing_summary()

        return manager_results

    #
    # Methods defined by derived class that are not
    # part of the user interface
    #

    def _close_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _queue_object_solves(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

#
# A partial implementation of the ScenarioTreeManagerSolver
# interface that is common to both the Serial scenario
# tree manager solver as well as the Pyro-based manager solver worker
#

class _ScenarioTreeManagerSolverWorker(ScenarioTreeManagerSolver,
                                       PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()

        # options for controlling the solver manager
        # (not the scenario tree manager)
        safe_declare_common_option(options,
                                   "solver_manager_pyro_host")
        safe_declare_common_option(options,
                                   "solver_manager_pyro_port")
        safe_declare_common_option(options,
                                   "solver_manager_pyro_shutdown")

        ScenarioTreePreprocessor._declare_options(options)

        return options

    @property
    def preprocessor(self):
        return self._preprocessor

    def __init__(self, *args, **kwds):
        if self.__class__ is _ScenarioTreeManagerSolverWorker:
            raise NotImplementedError(
                "%s is an abstract class for subclassing" % self.__class__)

        super(_ScenarioTreeManagerSolverWorker, self).__init__(*args, **kwds)
        # TODO: Does this import need to be delayed because
        #       it is in a plugins subdirectory?
        from pyomo.solvers.plugins.solvers.persistent_solver import \
            PersistentSolver

        assert self.manager is not None

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

        #
        # initialize the preprocessor
        #
        self._preprocessor = None
        if not self.get_option("disable_advanced_preprocessing"):
            self._preprocessor = ScenarioTreePreprocessor(self._options,
                                                          options_prefix=self._options_prefix)
        assert self._manager.preprocessor is None
        self._manager.preprocessor = self._preprocessor

        #
        # initialize the solver manager
        #
        self._solver_manager = SolverManagerFactory(
            self.get_option("solver_manager"),
            host=self.get_option('solver_manager_pyro_host'),
            port=self.get_option('solver_manager_pyro_port'))
        for scenario in self.manager.scenario_tree._scenarios:
            assert scenario._instance is not None
            solver = self._scenario_solvers[scenario.name] = \
                SolverFactory(self.get_option("solver"),
                              solver_io=self.get_option("solver_io"))
            if isinstance(solver, PersistentSolver) and \
               self.get_option("disable_advanced_preprocessing"):
                raise ValueError("Advanced preprocessing can not be disabled "
                                 "when persistent solvers are used")
            if self._preprocessor is not None:
                self._preprocessor.add_scenario(scenario,
                                                scenario._instance,
                                                solver)
        for bundle in self.manager.scenario_tree._scenario_bundles:
            solver = self._bundle_solvers[bundle.name] = \
                SolverFactory(self.get_option("solver"),
                              solver_io=self.get_option("solver_io"))
            if isinstance(solver, PersistentSolver) and \
               self.get_option("disable_advanced_preprocessing"):
                raise ValueError("Advanced preprocessing can not be disabled "
                                 "when persistent solvers are used")
            bundle_instance = \
                self.manager._bundle_binding_instance_map[bundle.name]
            if self._preprocessor is not None:
                self._preprocessor.add_bundle(bundle,
                                              bundle_instance,
                                              solver)

    #
    # Override some methods for ScenarioTreeManager that
    # were implemented by _ScenarioTreeManagerWorker:
    #

    def _close_impl(self):
        if (self._manager is not None) and \
           (self._manager.preprocessor is not None):
            assert self.preprocessor is self._manager.preprocessor
            for bundle in self.manager.scenario_tree._scenario_bundles:
                self._preprocessor.remove_bundle(bundle)
            for scenario in self.manager.scenario_tree._scenarios:
                assert scenario._instance is not None
                self._preprocessor.remove_scenario(scenario)
            self._manager.preprocessor = None
            self._preprocessor = None
        else:
            assert self._preprocessor is None

        if self._solver_manager is not None:
            #self._solver_manager.deactivate()
            self._solver_manager = None
            if self.get_option("solver_manager_pyro_shutdown"):
                print("Shutting down Pyro components for solver manager.")
                shutdown_pyro_components(
                    host=self.get_option("solver_manager_pyro_host"),
                    port=self.get_option("solver_manager_pyro_port"),
                    num_retries=0,
                    caller_name=self.__class__.__name__)
        #for solver in self._scenario_solvers.values():
        #    solver.deactivate()
        self._scenario_solvers = {}
        #for solver in self._bundle_solvers.values():
        #    solver.deactivate()
        self._bundle_solvers = {}
        self._preprocessor = None
        self._objective_sense = None

    #
    # Abstract methods for ScenarioTreeManagerSolver:
    #

    def _queue_object_solves(self,
                             object_type,
                             objects,
                             ephemeral_solver_options,
                             disable_warmstart):

        if self.get_option("verbose"):
            print("Queuing %s solves" % (object_type[:-1]))

        assert object_type in ('bundles', 'scenarios')

        solver_dict = None
        instance_dict = None
        modify_kwds_func = None
        if object_type == 'bundles':
            if objects is None:
                objects = self.manager.scenario_tree._scenario_bundle_map
            solver_dict = self._bundle_solvers
            instance_dict = self.manager._bundle_binding_instance_map
            for bundle_name in objects:
                for scenario_name in self.manager.scenario_tree.\
                    get_bundle(bundle_name).\
                       scenario_names:
                    self.manager.scenario_tree.get_scenario(scenario_name).\
                        _instance_objective.deactivate()
            if self.preprocessor is not None:
                self.preprocessor.preprocess_bundles(bundles=objects)
                modify_kwds_func = self.preprocessor.modify_bundle_solver_keywords
        else:
            if objects is None:
                objects = self.manager.scenario_tree._scenario_map
            solver_dict = self._scenario_solvers
            instance_dict = self.manager._instances
            if self.manager.scenario_tree.contains_bundles():
                for scenario_name in objects:
                    self.manager.scenario_tree.get_scenario(scenario_name).\
                        _instance_objective.activate()
            if self.preprocessor is not None:
                self.preprocessor.preprocess_scenarios(scenarios=objects)
                modify_kwds_func = self.preprocessor.modify_scenario_solver_keywords
        assert solver_dict is not None
        assert instance_dict is not None

        # setup common solve keywords
        common_kwds = {}
        common_kwds['tee'] = self.get_option("output_solver_log")
        common_kwds['keepfiles'] = self.get_option("keep_solver_files")
        common_kwds['symbolic_solver_labels'] = \
            self.get_option("symbolic_solver_labels")
        # we always manually load solutions, so we can
        # control error reporting and such
        common_kwds['load_solutions'] = False

        # Load solver options
        solver_options = {}
        if type(self.get_option("solver_options")) is tuple:
            solver_options.update(
                OptSolver._options_string_to_dict(
                    "".join(self.get_option("solver_options"))))
        else:
            solver_options.update(self.get_option("solver_options"))
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

            if modify_kwds_func is not None:
                # be sure to modify a copy of the kwds
                solve_kwds = modify_kwds_func(object_name, dict(common_kwds))
            else:
                solve_kwds = common_kwds
            opt = solver_dict[object_name]
            instance = instance_dict[object_name]
            if (not self.get_option("disable_warmstart")) and \
               (not disable_warmstart) and \
               opt.warm_start_capable():
                new_action_handle = \
                    self._solver_manager.queue(instance,
                                               opt=opt,
                                               warmstart=True,
                                               **solve_kwds)
            else:
                new_action_handle = \
                    self._solver_manager.queue(instance,
                                               opt=opt,
                                               **solve_kwds)

            action_handle_data[new_action_handle] = object_name

        return self.manager.AsyncResult(
            self._solver_manager,
            action_handle_data=action_handle_data)

class ScenarioTreeManagerSolverClientSerial(
        _ScenarioTreeManagerSolverWorker,
        ScenarioTreeManagerSolver,
        PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()
        return options

    def __init__(self, *args, **kwds):
        super(ScenarioTreeManagerSolverClientSerial, self).\
            __init__(*args, **kwds)

    #
    # Override some methods for ScenarioTreeManager that
    # were implemented by _ScenarioTreeManagerWorkerSolver:
    #

    def _close_impl(self):
        super(ScenarioTreeManagerSolverClientSerial, self)._close_impl()

    #
    # Abstract methods for ScenarioTreeManagerSolver:
    #

    # implemented by _ScenarioTreeManagerSolverWorker
    #def _queue_object_solves(...)

class ScenarioTreeManagerSolverClientPyro(ScenarioTreeManagerSolver,
                                          PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()
        return options

    default_registered_worker_name = 'ScenarioTreeManagerSolverWorkerPyro'

    #
    # Override the PySPConfiguredObject register_options implementation so
    # that the default behavior will be to register this classes default
    # worker type options along with the options for this class
    #

    @classmethod
    def register_options(cls, *args, **kwds):
        registered_worker_name = \
            kwds.pop('registered_worker_name',
                     cls.default_registered_worker_name)
        options = super(ScenarioTreeManagerSolverClientPyro, cls).\
                  register_options(*args, **kwds)
        if registered_worker_name is not None:
            worker_type = ScenarioTreeServerPyro.\
                          get_registered_worker_type(registered_worker_name)
            worker_type.register_options(options, **kwds)

        return options

    def __init__(self, manager, *args, **kwds):
        worker_registered_name = \
            kwds.pop('registered_worker_name',
                     self.default_registered_worker_name)
        super(ScenarioTreeManagerSolverClientPyro, self).\
            __init__(manager, *args, **kwds)

        assert self.manager is not None
        assert self.manager._action_manager is not None
        self._pyro_worker_map = {}
        self._pyro_base_worker_map = {}

        worker_class = ScenarioTreeServerPyro.\
                       get_registered_worker_type(
                           self.default_registered_worker_name)
        try:
            worker_options = worker_class.\
                             extract_user_options_to_dict(
                                 self._options,
                                 source_options_prefix=self._options_prefix,
                                 sparse=True)
        except KeyError:
            raise KeyError(
                "Unable to serialize options for registered worker "
                "name %s (class=%s). The worker options did not "
                "seem to match the registered options on the worker "
                "class. Did you forget to register them? Message: %s"
                % (worker_registered_name,
                   worker_class.__name__,
                   str(sys.exc_info()[1])))

        assert not self.manager._transmission_paused
        if not self.manager.get_option("pyro_handshake_at_startup"):
            self.manager.pause_transmit()
        action_handle_data = {}
        for base_worker_name in self.manager.worker_names:
            server_name = \
                self.manager.get_server_for_worker(base_worker_name)

            worker_name = base_worker_name+"_solver"
            if self.manager.get_option("verbose"):
                print("Initializing worker with name %s on "
                      "scenario tree server %s"
                      % (worker_name, server_name))

            action_handle = self.manager._action_manager.queue(
                queue_name=server_name,
                action="ScenarioTreeServerPyro_initialize",
                worker_type=worker_registered_name,
                worker_name=worker_name,
                init_args=(base_worker_name,),
                init_kwds=worker_options,
                generate_response=True)

            if self.manager.get_option("pyro_handshake_at_startup"):
                action_handle_data[worker_name] =  \
                    self.manager.AsyncResult(
                        self.manager._action_manager,
                        action_handle_data=action_handle).complete()
            else:
                action_handle_data[action_handle] = worker_name

            self._pyro_worker_map[base_worker_name] = worker_name
            self._pyro_base_worker_map[worker_name] = base_worker_name

        if not self.manager.get_option("pyro_handshake_at_startup"):
            self.manager.unpause_transmit()

        if self.manager.get_option("pyro_handshake_at_startup"):
            result = self.manager.AsyncResult(
                None,
                result=action_handle_data)
        else:
            result = self.manager.AsyncResult(
                self.manager._action_manager,
                action_handle_data=action_handle_data)
        result.complete()

    #
    # Abstract methods for ScenarioTreeManagerSolver:
    #

    def _close_impl(self):
        # release the workers created by this manager solver
        if len(self._pyro_worker_map):
            assert len(self._pyro_worker_map) == \
                len(self._pyro_base_worker_map)
            if self.manager._transmission_paused:
                print("Unpausing pyro transmissions in "
                      "preparation for releasing solver workers")
                self.manager.unpause_transmit()
            self.manager.pause_transmit()
            action_handles = []
            for base_worker_name in self._pyro_worker_map:
                worker_name = self._pyro_worker_map[base_worker_name]
                server_name = self.manager.\
                              get_server_for_worker(base_worker_name)
                action_handles.append(
                    self.manager._action_manager.queue(
                    queue_name=server_name,
                    action="ScenarioTreeServerPyro_release",
                    worker_name=worker_name,
                    generate_response=True))
            self.manager.unpause_transmit()
            self.manager._action_manager.wait_all(action_handles)
            for ah in action_handles:
                self.manager._action_manager.get_results(ah)
        self._pyro_worker_map = {}
        self._pyro_base_worker_map = {}

    def _queue_object_solves(self,
                             object_type,
                             objects,
                             ephemeral_solver_options,
                             disable_warmstart):

        assert object_type in ('bundles', 'scenarios')

        if self.get_option("verbose"):
            print("Transmitting solve requests for %s" % (object_type))

        worker_names = None
        worker_map = {}
        if objects is not None:
            if object_type == 'bundles':
                _get_worker_func = self.manager.get_worker_for_bundle
            else:
                assert object_type == 'scenarios'
                _get_worker_func = self.manager.get_worker_for_scenario
            for object_name in objects:
                worker_name = _get_worker_func(object_name)
                if worker_name not in worker_map:
                    worker_map[worker_name] = []
                worker_map[worker_name].append(object_name)
            worker_names = worker_map
        else:
            worker_names = self.manager._pyro_worker_list

        was_paused = self.manager.pause_transmit()
        action_handle_data = {}
        for base_worker_name in worker_names:
            method_args=(object_type,
                         worker_map.get(base_worker_name, None),
                         ephemeral_solver_options,
                         disable_warmstart)
            worker_name = self._pyro_worker_map[base_worker_name]
            server_name = \
                self.manager.get_server_for_worker(base_worker_name)
            action_handle_data[self.manager._action_manager.queue(
                    queue_name=server_name,
                    worker_name=worker_name,
                    action="_solve_objects_for_client",
                    args=method_args,
                    kwds={},
                    generate_response=True)] = worker_name
        if not was_paused:
            self.manager.unpause_transmit()

        return self.manager.AsyncResult(
            self.manager._action_manager,
            action_handle_data=action_handle_data,
            map_result=(lambda ah_to_result: \
                        dict((key, result[key])
                             for result in itervalues(ah_to_result)
                             for key in result)))

def ScenarioTreeManagerSolverFactory(sp, *args, **kwds):
    """Return a scenario tree manager solver appropriate for
    the provided argument.

    Args:
        sp: a serial or pyro client scenario tree manager
        *args: A single additional argument can be provided
            that is a block of registered options used to
            initialize the returned manager solver. The
            block of options can be created by calling
            :attr:`ScenarioTreeManagerSolverFactory.register_options`.
        **kwds: Additional keywords are passed to the
            manager solver that is created.

    Returns: A :class:`ScenarioTreeManagerSolver` object.

    Example:
        The preferred way to use a ScenarioTreeManagerSolver
        object is through a :const:`with` block as it
        modifies the state of the underlying scenario tree
        manager. If used outside a :const:`with` block, the
        manager solver should be shutdown by calling the
        :attr:`close` method.

        >>> with ScenarioTreeManagerSolverFactory(sp) as manager:
        >>>    results = manager.solve_subproblems()
        >>> results.pprint()

        Note that asynchronous solves should be completed
        before the manager solver is closed; otherwise the
        results are undefined.

        >>> with ScenarioTreeManagerSolverFactory(sp) as manager:
        >>>    job = manager.solve_subproblems(async_call=True)
        >>>    reuslts = job.complete()
        >>> results.pprint()
    """
    if isinstance(sp, ScenarioTreeManagerClientSerial):
        manager_class = ScenarioTreeManagerSolverClientSerial
    elif isinstance(sp, ScenarioTreeManagerClientPyro):
        manager_class = ScenarioTreeManagerSolverClientPyro
    else:
        raise ValueError("Unrecognized type for first argument: %s "
                         % (type(sp)))
    if len(args) == 0:
        options = manager_class.register_options()
    elif len(args) == 1:
        options = args[0]
    else:
        raise ValueError("At most 2 arguments allowed "
                         "for function call")
    return manager_class(sp, options, **kwds)

def _register_scenario_tree_manager_solver_options(*args, **kwds):
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

    ScenarioTreeManagerSolverClientSerial.register_options(options,
                                                           **kwds)
    ScenarioTreeManagerSolverClientPyro.register_options(options,
                                                         **kwds)

    return options

ScenarioTreeManagerSolverFactory.register_options = \
    _register_scenario_tree_manager_solver_options
