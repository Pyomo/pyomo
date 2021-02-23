#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ("InvocationType",
           "ScenarioTreeManagerClientSerial",
           "ScenarioTreeManagerClientPyro",
           "ScenarioTreeManagerFactory")

import math
import sys
import time
import itertools
import inspect
import logging
import traceback
from collections import (defaultdict,
                         namedtuple)

import pyutilib.misc
from pyutilib.pyro import (shutdown_pyro_components,
                           using_pyro4)
from pyomo.common.dependencies import dill, dill_available
from pyomo.opt import (UndefinedData,
                       undefined,
                       SolverStatus,
                       TerminationCondition,
                       SolutionStatus)
from pyomo.opt.parallel.manager import ActionHandle
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import (PySPConfigBlock,
                                    safe_declare_common_option,
                                    safe_register_common_option,
                                    _domain_must_be_str,
                                    _domain_tuple_of_str)
from pyomo.pysp.util.misc import load_external_module
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory
from pyomo.pysp.scenariotree.action_manager_pyro \
    import ScenarioTreeActionManagerPyro
from pyomo.pysp.scenariotree.server_pyro \
    import ScenarioTreeServerPyro
from pyomo.pysp.ef import create_ef_instance

import six
from six import (iteritems,
                 itervalues,
                 StringIO,
                 string_types)
from six.moves import xrange

logger = logging.getLogger('pyomo.pysp')

class _InvocationTypeMeta(type):
    def __contains__(cls, obj):
        return isinstance(obj, cls._value)
    def __iter__(cls):
        return iter(
            sorted((obj for obj in cls.__dict__.values()
                    if isinstance(obj, cls._value)),
                   key=lambda _: _.index)
        )

@six.add_metaclass(_InvocationTypeMeta)
class InvocationType(object):
    """Controls execution of function invocations with a scenario tree manager.

    In all cases, the function must accept the process-local scenario
    tree worker as the first argument. Whether or not additional
    arguments are required, depends on the invocation type. For the
    'Single' invocation type, no additional arguments are required.
    Otherwise, the function signature is required to accept a second
    argument representing the worker-local scenario or scenario
    bundle object.

    It is implied that the function invocation takes place on the
    scenario tree worker(s), which is(are) not necessarily the same as
    the scenario tree manager whose method is provided with the
    invocation type. For instance, Pyro-based scenario tree managers
    (e.g., ScenarioTreeManagerClientPyro) must transmit these method
    invocations to their respective scenario tree workers which live
    in separate processes. Any scenario tree worker is itself an
    instance of a ScenarioTreeManager so the same invocation rules
    apply when using this interface in a worker-local context. The
    ScenarioTreeManagerClientSerial implementation is its own scenario
    tree worker, so all function invocations take place locally and on
    the same object whose method is invoked.

    If the worker name is not provided (e.g., when the
    'invoke_function' method is used), then the following behavior is
    implied for each invocation type:

       - Single:
            The function is executed once per scenario tree
            worker. Return value will be in the form of a dict mapping
            worker name to function return value.

       - PerScenario:
            The function is executed once per scenario in the scenario
            tree. Return value will be in the form of a dict mapping
            scenario name to return value.

       - PerScenarioChained:
            The function is executed once per scenario in the scenario
            tree in a sequential call chain. The result from each
            function call is passed into the next function call in
            *arg form after the scenario tree worker and scenario
            arguments (unless no additional function arguments were
            initially provided).  Return value is in the form of a
            tuple (or None), and represents the return value from the
            final call in the chain.

       - PerBundle:
            Identical to the PerScenario invocation type except by
            bundle.

       - PerBundleChained:
            Identical to the PerScenarioChained invocation type except
            by bundle.

     * NOTE: The remaining invocation types listed below should
             initialized with any relevant data before being passed
             into methods that use them. This is done using the
             __call__ method, which returns a matching invocation type
             loaded with the provided data.

             Examples:
                  InvocationType.OnScenario('Scenario1')
                  InvocationType.OnScenarios(['Scenario1', 'Scenario2'])

       - OnScenario(<scenario-name>):
            The function is executed on the named scenario and its
            associated scenario tree worker. Return value corresponds
            exactly to the function return value.

       - OnScenarios([<scenario-names>]):
            The function is executed on the named scenarios and their
            associated scenario tree worker(s). Return value will be
            in the form of a dict mapping scenario name to return
            value.

       - OnScenariosChained([<scenario-names>]):
            Same as PerScenarioChained only executed over the given
            subset of scenarios named. Invocation order is guaranteed
            to correspond exactly to the iteration order of the given
            scenario names.

       - OnBundle(<bundle-name>):
            Identical to the OnScenario invocation type except with a
            bundle.

       - OnBundles([<bundle-names>]):
            Identical to the OnScenarios invocation type except by
            bundle.

       - OnBundlesChained([<bundle-names>]):
            Identical to the OnScenariosChained invocation type except
            by bundle.

    If the scenario tree worker name is provided (e.g., when the
    'invoke_function_on_worker' method is used), then the following
    behaviors change:

       - Single:
            The return value corresponds exactly to the function
            return value (rather than a dict mapping worker_name to
            return value).

       - Per*:
            Function execution takes place only over the scenarios /
            bundles managed by the named scenario tree worker.

       - On*:
            Not necessarily designed for this context, but execution
            behavior remains the same. An exception will be raised if
            the named scenario(s) / bundles(s) are not directly
            managed by the named scenario tree worker.

    """
    class _value(object):
        def __init__(self, key, index):
            self._key = key
            self._index = index
        @property
        def key(self):
            return self._key
        @property
        def index(self):
            return self._index
        def __hash__(self):
            return hash((self.key, self.index))
        def __eq__(self, other):
            return (self.__class__ is other.__class__) and \
                (self.key == other.key) and (self.index == other.index)
        def __ne__(self, other):
            return not self.__eq__(other)
        def __repr__(self):
            return ("InvocationType.%s" % (self.key))
    class _value_with_data(_value):
        def __init__(self, key, id_, domain):
            super(self.__class__, self).__init__(key, id_)
            self._domain = domain
            self._data = None
        @property
        def data(self):
            return self._data
        def __call__(self, data):
            if self.data is not None:
                raise ValueError("Must create from InvocationType class")
            obj = self.__class__(self.key, self.index, self._domain)
            assert obj.data is None
            obj._data = self._domain(data)
            assert obj.data is obj._data
            return obj
    Single =                       _value("Single", 0)
    PerScenario =                  _value("PerScenario", 1)
    PerScenarioChained =           _value("PerScenarioChained", 2)
    PerBundle =                    _value("PerBundle", 3)
    PerBundleChained =             _value("PerBundleChained", 4)
    ### deprecated
    SingleInvocation =             _value("SingleInvocation", 5)
    PerScenarioInvocation =        _value("PerScenarioInvocation", 6)
    PerScenarioChainedInvocation = _value("PerScenarioChainedInvocation", 7)
    PerBundleInvocation =          _value("PerBundleInvocation", 8)
    PerBundleChainedInvocation =   _value("PerBundleChainedInvocation", 9)
    ###
    OnScenario =                   _value_with_data("OnScenario", 10 ,_domain_must_be_str)
    OnScenarios =                  _value_with_data("OnScenarios", 11, _domain_tuple_of_str)
    OnBundle =                     _value_with_data("OnBundle", 12, _domain_must_be_str)
    OnBundles =                    _value_with_data("OnBundles", 13, _domain_tuple_of_str)
    OnScenariosChained =           _value_with_data("OnScenariosChained", 14, _domain_tuple_of_str)
    OnBundlesChained =             _value_with_data("OnBundlesChained", 15, _domain_tuple_of_str)
    def __init__(self, *args, **kwds):
        raise NotImplementedError

_deprecated_invocation_types = \
    {InvocationType.SingleInvocation: InvocationType.Single,
     InvocationType.PerScenarioInvocation: InvocationType.PerScenario,
     InvocationType.PerScenarioChainedInvocation: InvocationType.PerScenarioChained,
     InvocationType.PerBundleInvocation: InvocationType.PerBundle,
     InvocationType.PerBundleChainedInvocation: InvocationType.PerBundleChained}
def _map_deprecated_invocation_type(invocation_type):
    if invocation_type in _deprecated_invocation_types:      #pragma:nocover
        logger.warning("DEPRECATED: %s has been renamed to %s"
                       % (invocation_type, _deprecated_invocation_types[invocation_type]))
        invocation_type = _deprecated_invocation_types[invocation_type]
    return invocation_type

#
# A named tuple that groups together the information required
# to initialize a new worker on a scenario tree server:
#  - type_: "bundles" or "scenarios"
#  - names: A list of names for the scenario tree objects
#           that will be initialized on the worker. The
#           names should represent scenarios or bundles
#           depending on the choice of type_.
#  - data: The data associated with choice of type_.  For
#          'Scenarios', this should be None. For 'Bundles'
#          this should be a dictionary mapping bundle name
#          to a list of scenario names.
#
_WorkerInit = namedtuple('_WorkerInit',
                         ['type_', 'names', 'data'])

#
# A convenience function for populating a _WorkerInit tuple
# for scenario worker initializations. If initializing a single
# scenario, arg should be a scenario name. If initializing a list
# of scenarios, arg should be a list or tuple of scenario names.
#
def _ScenarioWorkerInit(arg):
    if isinstance(arg, string_types):
        # a single scenario
        return _WorkerInit(type_="scenarios",
                           names=(arg,),
                           data=None)
    else:
        # a list of scenarios
        assert type(arg) in (list, tuple)
        for name in arg:
            assert isinstance(name, string_types)
        return _WorkerInit(type_="scenarios",
                           names=arg,
                           data=None)

#
# A convenience function for populating a _WorkerInit tuple
# for bundle worker initializations. If initializing a single
# bundle, arg should be the bundle name and data should be a
# list or tuple. If initializing a list of bundles, arg should
# a list or tuple of bundle names, and data should be a dict
# mapping bundle name to a list or tuple of scenarios.
#
def _BundleWorkerInit(arg, data):
    if isinstance(arg, string_types):
        # a single bundle
        assert type(data) in (list, tuple)
        assert len(data) > 0
        return _WorkerInit(type_="bundles",
                           names=(arg,),
                           data={arg: data})
    else:
        # a list of bundles
        assert type(arg) in (list, tuple)
        assert type(data) is dict
        for name in arg:
            assert isinstance(name, string_types)
            assert type(data[name]) in (list, tuple)
            assert len(data[name]) > 0
        return _WorkerInit(type_="bundles",
                           names=arg,
                           data=data)

class ScenarioTreeSolveResults(object):
    """A container that summarizes the results of solve
    request to a ScenarioTreeManagerSolver. Results will
    be organized by scenario name or bundle name,
    depending on the solve type."""

    def __init__(self, solve_type):
        assert solve_type in ('scenarios','bundles')

        # The type of solve used to generate these results
        # Will always be one of 'scenarios' or 'bundles'
        self._solve_type = solve_type

        # Maps scenario name (or bundle name) to the
        # objective value reported by the corresponding
        # sub-problem.
        self._objective = {}

        # Similar to the above, but calculated from the
        # some of the stage costs for the object, which
        # can be different from the objective when it is
        # augmented by a PySP algorithm.
        self._cost = {}

        # Maps scenario name (or bundle name) to the gap
        # reported by the solver when solving the
        # associated instance. If there is no entry,
        # then there has been no solve. Values can be
        # undefined when the solver plugin does not
        # report a gap.
        self._gap = {}

        # Maps scenario name (or bundle name) to the
        # last solve time reported for the corresponding
        # sub-problem. Presently user time, due to
        # deficiency in solver plugins. Ultimately want
        # wall clock time for reporting purposes.
        self._solve_time = {}

        # Similar to the above, but the time consumed by
        # the invocation of the solve() method on
        # whatever solver plugin was used.
        self._pyomo_solve_time = {}

        # Maps scenario name (or bundle name) to the
        # solver status associated with the solves. If
        # there is no entry or it is undefined, then the
        # object was not solved.
        self._solver_status = {}

        # Maps scenario name (or bundle name) to the
        # solver termination condition associated with
        # the solves. If there is no entry or it is
        # undefined, then the object was not solved
        self._solver_message = {}

        # Maps scenario name (or bundle name) to the
        # solver termination condition associated with
        # the solves. If there is no entry or it is
        # undefined, then the object was not solved.
        self._termination_condition = {}

        # Maps scenario name (or bundle name) to the
        # solution status associated with the solves. If
        # there is no entry or it is undefined, then the
        # object was not solved.
        self._solution_status = {}

    @property
    def solve_type(self):
        """Return a string indicating the type of
        objects associated with these solve results
        ('scenarios' or 'bundles')."""
        return self._solve_type

    @property
    def objective(self):
        """Return a dictionary with the objective values
        for all objects associated with these solve
        results."""
        return self._objective

    @property
    def cost(self):
        """Return a dictionary with the sum of the stage
        costs for all objects associated with these
        solve results."""
        return self._cost

    @property
    def pyomo_solve_time(self):
        """Return a dictionary with the pyomo solve
        times for all objects associated with these
        solve results."""
        return self._pyomo_solve_time

    @property
    def solve_time(self):
        """Return a dictionary with solve times for all
        objects associated with these solve results."""
        return self._solve_time

    @property
    def gap(self):
        """Return a dictionary with solution gaps for
        all objects associated with these solve
        results."""
        return self._gap

    @property
    def solver_status(self):
        """Return a dictionary with solver statuses for
        all objects associated with these solve
        results."""
        return self._solver_status

    @property
    def solver_message(self):
        """Return a dictionary with solver messages for
        all objects associated with these solve
        results."""
        return self._solver_message

    @property
    def termination_condition(self):
        """Return a dictionary with solver termination
        conditions for all objects associated with these
        solve results."""
        return self._termination_condition

    @property
    def solution_status(self):
        """Return a dictionary with solution statuses
        for all objects associated with these solve
        results."""
        return self._solution_status


    def update(self, results):
        assert isinstance(results,
                          ScenarioTreeSolveResults)
        if results.solve_type != self.solve_type:
            raise ValueError(
                "Can not update scenario tree manager solver "
                "results object with solve type '%s' from "
                "another results object with a different solve "
                "type '%s'" % (self.solve_type, results.solve_type))
        for attr_name in ("objective",
                          "cost",
                          "pyomo_solve_time",
                          "solve_time",
                          "gap",
                          "solver_status",
                          "solver_message",
                          "termination_condition",
                          "solution_status"):
            getattr(self, attr_name).update(
                getattr(results, attr_name))

    def results_for(self, object_name):
        """Return a dictionary that summarizes all
        results information for an individual object
        associated with these solve results."""
        if object_name not in self.objective:
            raise KeyError(
                "This results object does not hold any "
                "results for scenario tree object with "
                "name: %s" % (object_name))
        results = {}
        for attr_name in ("objective",
                          "cost",
                          "pyomo_solve_time",
                          "solve_time",
                          "gap",
                          "solver_status",
                          "solver_message",
                          "termination_condition",
                          "solution_status"):
            results[attr_name] = getattr(self, attr_name)[object_name]
        return results

    def pprint(self, output_times=False, filter_names=None):
        """Print a summary of the solve results included in this object."""

        object_names = list(filter(filter_names,
                                   sorted(self.objective.keys())))
        if len(object_names) == 0:
            print("No result data available")
            return

        max_name_len = max(len(str(_object_name)) \
                           for _object_name in object_names)
        if self.solve_type == 'bundles':
            max_name_len = max((len("Bundle Name"), max_name_len))
            line = (("%-"+str(max_name_len)+"s  ") % "Bundle Name")
        else:
            assert self.solve_type == 'scenarios'
            max_name_len = max((len("Scenario Name"), max_name_len))
            line = (("%-"+str(max_name_len)+"s  ") % "Scenario Name")
        line += ("%-16s %-16s %-14s %-14s %-16s"
                 % ("Cost",
                    "Objective",
                    "Objective Gap",
                    "Solver Status",
                    "Term. Condition"))
        if output_times:
            line += (" %-11s" % ("Solve Time"))
            line += (" %-11s" % ("Pyomo Time"))
        print(line)
        for object_name in object_names:
            objective_value = self.objective[object_name]
            cost_value = self.cost[object_name]
            gap = self.gap[object_name]
            solver_status = self.solver_status[object_name]
            term_condition = self.termination_condition[object_name]
            line = ("%-"+str(max_name_len)+"s  ")
            if isinstance(objective_value, UndefinedData):
                line += "%-16s"
            else:
                line += "%-16.7e"
            if isinstance(cost_value, UndefinedData):
                line += " %-16s"
            else:
                line += " %-16.7e"
            if (not isinstance(gap, UndefinedData)) and \
               (gap is not None):
                line += (" %-14.4e")
            else:
                line += (" %-14s")
            line += (" %-14s %-16s")
            line %= (object_name,
                     cost_value,
                     objective_value,
                     gap,
                     solver_status,
                     term_condition)
            if output_times:
                solve_time = self.solve_time.get(object_name)
                if (not isinstance(solve_time, UndefinedData)) and \
                   (solve_time is not None):
                    line += (" %-11.2f")
                else:
                    line += (" %-11s")
                line %= (solve_time,)

                pyomo_solve_time = self.pyomo_solve_time.get(object_name)
                if (not isinstance(pyomo_solve_time, UndefinedData)) and \
                   (pyomo_solve_time is not None):
                    line += (" %-11.2f")
                else:
                    line += (" %-11s")
                line %= (pyomo_solve_time,)
            print(line)
        print("")

    def pprint_status(self, filter_names=None):
        """Print a summary of the solve results included in this object."""

        object_names = list(filter(filter_names,
                                   sorted(self.objective.keys())))
        if len(object_names) == 0:
            print("No result data available")
            return

        max_name_len = max(len(str(_object_name)) \
                           for _object_name in object_names)
        if self.solve_type == 'bundles':
            max_name_len = max((len("Bundle Name"), max_name_len))
            line = (("%-"+str(max_name_len)+"s  ") % "Bundle Name")
        else:
            assert self.solve_type == 'scenarios'
            max_name_len = max((len("Scenario Name"), max_name_len))
            line = (("%-"+str(max_name_len)+"s  ") % "Scenario Name")
        line += ("%-14s %-16s"
                 % ("Solver Status",
                    "Term. Condition"))
        print(line)
        for object_name in object_names:
            solver_status = self.solver_status[object_name]
            term_condition = self.termination_condition[object_name]
            line = ("%-"+str(max_name_len)+"s  ")
            line += ("%-14s %-16s")
            line %= (object_name,
                     solver_status,
                     term_condition)
            print(line)
        print("")

    def print_timing_summary(self, filter_names=None):

        object_names = list(filter(filter_names,
                                   sorted(self.objective.keys())))
        if len(object_names) == 0:
            print("No result data available")
            return

        # if any of the solve times are of type
        # pyomo.opt.results.container.UndefinedData, then don't
        # output timing statistics.
        solve_times = list(self.solve_time[object_name]
                           for object_name in object_names)
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
                  % (self.solve_type,
                     min(solve_times),
                     mean,
                     max(solve_times),
                     std_dev))

        # if any of the solve times are of type
        # pyomo.opt.results.container.UndefinedData, then don't
        # output timing statistics.
        pyomo_solve_times = list(self.pyomo_solve_time[object_name]
                                 for object_name in object_names)
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
                  % (self.solve_type,
                     min(pyomo_solve_times),
                     mean,
                     max(pyomo_solve_times),
                     std_dev))

if using_pyro4:
    import Pyro4
    from Pyro4.util import SerializerBase
    # register hooks for ScenarioTreeSolveResults
    def ScenarioTreeSolveResults_to_dict(obj):
        data = {"__class__": ("pyomo.pysp.scenario_tree.manager_solver."
                              "ScenarioTreeSolveResults")}
        data.update(obj.__dict__)
        # Convert enums to strings to avoid difficult
        # behavior related to certain Pyro serializer
        # settings
        for attr_name in ('_solver_status',
                          '_termination_condition',
                          '_solution_status'):
            data[attr_name] = \
                dict((key, str(val)) for key,val
                     in data[attr_name].items())
        return data
    def dict_to_ScenarioTreeSolveResults(classname, d):
        obj = ScenarioTreeSolveResults(d['_solve_type'])
        assert "__class__" not in d
        obj.__dict__.update(d)
        # Convert status strings back to enums. These are
        # transmitted as strings to avoid difficult behavior
        # related to certain Pyro serializer settings
        for object_name in obj.solver_status:
            obj.solver_status[object_name] = \
                getattr(SolverStatus,
                        obj.solver_status[object_name])
        for object_name in obj.termination_condition:
            obj.termination_condition[object_name] = \
                getattr(TerminationCondition,
                        obj.termination_condition[object_name])
        for object_name in obj.solution_status:
            obj.solution_status[object_name] = \
                getattr(SolutionStatus,
                    obj.solution_status[object_name])
        return obj
    SerializerBase.register_class_to_dict(
        ScenarioTreeSolveResults,
        ScenarioTreeSolveResults_to_dict)
    SerializerBase.register_dict_to_class(
        ("pyomo.pysp.scenario_tree.manager_solver."
         "ScenarioTreeSolveResults"),
        dict_to_ScenarioTreeSolveResults)


#
# A base class and interface that is common to all scenario tree
# client and worker managers.
#

class ScenarioTreeManager(PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()
        return options

    #
    # Note: These Async objects can be cleaned up.
    #       This is a first draft.
    #
    class Async(object):
        def complete(self):
            """Wait for the job request to complete and return the result."""
            raise NotImplementedError                  #pragma:nocover

    class AsyncResult(Async):

        __slots__ = ('_action_manager',
                     '_result',
                     '_action_handle_data',
                     '_invocation_type',
                     '_map_result')

        def __init__(self,
                     action_manager,
                     result=None,
                     action_handle_data=None,
                     map_result=None):
            if result is not None:
                assert action_handle_data is None
            if action_handle_data is not None:
                assert action_manager is not None
            if map_result is not None:
                assert result is None
                assert action_handle_data is not None
            self._action_manager = action_manager
            self._action_handle_data = action_handle_data
            self._result = result
            self._map_result = map_result

        def complete(self):
            """Wait for the job request to complete and return the result."""
            if self._result is not None:
                if isinstance(self._result,
                              ScenarioTreeManager.Async):
                    self._result = self._result.complete()
                return self._result

            if self._action_handle_data is None:
                assert self._result is None
                return None

            result = None
            if isinstance(self._action_handle_data, ActionHandle):
                result = self._action_manager.wait_for(
                    self._action_handle_data)
                if self._map_result is not None:
                    result = self._map_result(self._action_handle_data, result)
            else:
                ah_to_result = {}
                ahs = set(self._action_handle_data)
                while len(ahs) > 0:
                    ah = self._action_manager.wait_any(ahs)
                    ah_to_result[ah] = self._action_manager.get_results(ah)
                    ahs.remove(ah)
                #self._action_manager.wait_all(self._action_handle_data)
                #ah_to_result = dict((ah, self._action_manager.get_results(ah))
                #                    for ah in self._action_handle_data)
                if self._map_result is not None:
                    result = self._map_result(ah_to_result)
                else:
                    result = dict((self._action_handle_data[ah], ah_to_result[ah])
                                  for ah in ah_to_result)
            self._result = result
            return self._result

    # This class ensures that a chain of asynchronous
    # actions are completed in order
    class AsyncResultChain(Async):
        __slots__ = ("_results", "_return_index")

        def __init__(self, results, return_index=-1):
            self._results = results
            self._return_index = return_index

        def complete(self):
            """Wait for the job request to complete and return the result."""
            for i in xrange(len(self._results)):
                assert isinstance(self._results[i],
                                  ScenarioTreeManager.Async)
                self._results[i] = self._results[i].complete()
            result = None
            if self._return_index is not None:
                result = self._results[self._return_index]
            return result

    # This class returns the result of a callback function
    # when completing an asynchronous action
    class AsyncResultCallback(Async):
        __slots__ = ("_result", "_done")

        def __init__(self, result):
            self._result = result
            self._done = False

        def complete(self):
            """Wait for the job request to complete and return the result."""
            if not self._done:
                self._result = self._result()
                self._done = True
            return self._result

    def __init__(self, *args, **kwds):
        if self.__class__ is ScenarioTreeManager:
            raise NotImplementedError(
                "%s is an abstract class for subclassing" % self.__class__)

        super(ScenarioTreeManager, self).__init__(*args, **kwds)

        self._error_shutdown = False
        self._scenario_tree = None
        # bundle info
        self._scenario_to_bundle_map = {}
        # For the users to modify as they please in the aggregate
        # callback as long as the data placed on it can be serialized
        # by Pyro
        self._aggregate_user_data = {}
        # set to true with the __enter__ method is called
        self._inside_with_block = False
        self._initialized = False

        # the objective sense of the subproblems
        self._objective_sense = None


    def _init_bundle(self, bundle_name, scenario_list):
        if self._options.verbose:
            print("Initializing scenario bundle with name %s"
                  % (bundle_name))
        # make sure the bundle was already added to the scenario tree
        assert self._scenario_tree.contains_bundle(bundle_name)
        for scenario_name in scenario_list:
            if scenario_name in self._scenario_to_bundle_map:
                raise ValueError(
                    "Unable to form binding instance for bundle %s. "
                    "Scenario %s already belongs to bundle %s."
                    % (bundle_name,
                       scenario_name,
                       self._scenario_to_bundle_map[scenario_name]))
            self._scenario_to_bundle_map[scenario_name] = bundle_name

    def _release_bundle(self, bundle_name):
        if self._options.verbose:
            print("Releasing scenario bundle with name %s"
                  % (bundle_name))

        # make sure the bundle was already added to the scenario tree
        assert self._scenario_tree.contains_bundle(bundle_name)
        bundle = self._scenario_tree.get_bundle(bundle_name)
        for scenario_name in bundle._scenario_names:
            del self._scenario_to_bundle_map[scenario_name]

    #
    # Interface:
    #

    @property
    def objective_sense(self):
        """Return the objective sense declared for all
        subproblems."""
        return self._objective_sense

    @property
    def modules_imported(self):
        raise NotImplementedError                  #pragma:nocover

    @property
    def scenario_tree(self):
        return self._scenario_tree

    @property
    def initialized(self):
        return self._initialized

    def initialize(self, *args, **kwds):
        """Initialize the scenario tree manager.

        A scenario tree manager must be initialized before using it.
        """

        init_start_time = time.time()
        result = None
        self._initialized = True
        try:
            if self._options.verbose:
                print("Initializing %s with options:"
                      % (type(self).__name__))
                self.display_options()
                print("")
            ############# derived method
            result = self._init(*args, **kwds)
            #############
            if self._options.verbose:
                print("%s is successfully initialized"
                      % (type(self).__name__))

        except:
            if not self._inside_with_block:
                print("Exception encountered. Scenario tree manager "
                      "attempting to shut down.")
                print("Original Exception:")
                traceback.print_exception(*sys.exc_info())
                self.close()
            raise

        if self._options.output_times or \
           self._options.verbose:
            print("Overall initialization time=%.2f seconds"
                  % (time.time() - init_start_time))

        return result

    def __enter__(self):
        self._inside_with_block = True
        return self

    def __exit__(self, *args):
        if args[0] is not None:
            sys.stderr.write("Exception encountered. Scenario tree manager "
                             "attempting to shut down.\n")
            tmp = StringIO()
            _args = list(args) + [None, tmp]
            traceback.print_exception(*_args)
            self._error_shutdown = True
            try:
                self.close()
            except:
                logger.error("Exception encountered during emergency scenario "
                             "tree manager shutdown. Printing original exception "
                             "here:\n")
                logger.error(tmp.getvalue())
                raise
        else:
            self.close()

    def close(self):
        """Close the scenario tree manager and any associated objects."""
        if self._options.verbose:
            print("Closing "+str(self.__class__.__name__))
        self._close_impl()
        if hasattr(self._scenario_tree, "_scenario_instance_factory"):
            self._scenario_tree._scenario_instance_factory.close()
        self._scenario_tree = None
        self._scenario_to_bundle_map = {}
        self._aggregate_user_data = {}
        self._inside_with_block = False
        self._initialized = False
        self._objective_sense = None

    def invoke_function(self,
                        function,
                        module_name=None,
                        invocation_type=InvocationType.Single,
                        function_args=(),
                        function_kwds=None,
                        async_call=False,
                        oneway_call=False):
        """Invokes a function on scenario tree constructs
        managed by this scenario tree manager. The first
        argument accepted by the function must always be the
        process-local scenario tree worker object, which may
        or may not be this object.

        Args:
            function:
                 The function or name of the function to be
                 invoked. If the object is a function, then
                 the manager will attempt to transmit it
                 using the dill package. Otherwise, the
                 argument must be a string and the
                 module_name keyword is required.
            module_name:
                 The name of the module containing the
                 function. This can also be an absolute path
                 to a file that contains the function
                 definition. If this function argument is an
                 actual function, this keyword must be left
                 at its default value of None; otherwise, it
                 is required.
            invocation_type:
                 Controls how the function is invoked. Refer
                 to the doc string for
                 pyomo.pysp.scenariotree.manager.InvocationType
                 for more information.
            function_args:
                 Extra arguments passed to the function when
                 it is invoked. These will always be placed
                 after the initial process-local scenario
                 tree worker object as well as any
                 additional arguments governed by the
                 invocation type.
            function_kwds:
                 Additional keywords to pass to the function
                 when it is invoked.
            async_call:
                 When set to True, the return value will be
                 an asynchronous object. Invocation results
                 can be obtained at any point by calling the
                 complete() method on this object, which
                 will block until all associated action
                 handles are collected.f
            oneway_call:
                 When set to True, it will be assumed no return value
                 is expected from this function (async_call is
                 implied). Setting both async_call and oneway_call to
                 True will result in an exception being raised.

            *Note: The 'oneway_call' and 'async_call' keywords are
                   valid for all scenario tree manager
                   implementations. However, they are
                   designed for use with Pyro-based
                   implementations. Their existence in other
                   implementations is not meant to guarantee
                   asynchronicity, but rather to provide a
                   consistent interface for code to be
                   written around.

        Returns:
            If 'oneway_call' is True, this function will always
            return None. Otherwise, the return value type is
            governed by the 'invocation_type' keyword, which
            will be nested inside an asynchronous object if
            'async_call' is set to True.
        """
        if not self.initialized:
            raise RuntimeError(
                "The scenario tree manager is not initialized.")
        if async_call and oneway_call:
            raise ValueError("async oneway calls do not make sense")
        invocation_type = _map_deprecated_invocation_type(invocation_type)
        if (invocation_type == InvocationType.PerBundle) or \
           (invocation_type == InvocationType.PerBundleChained) or \
           (invocation_type == InvocationType.OnBundle) or \
           (invocation_type == InvocationType.OnBundles) or \
           (invocation_type == InvocationType.OnBundlesChained):
            if not self._scenario_tree.contains_bundles():
                raise ValueError(
                    "Received request for bundle invocation type "
                    "but the scenario tree does not contain bundles.")
        return self._invoke_function_impl(function,
                                          module_name=module_name,
                                          invocation_type=invocation_type,
                                          function_args=function_args,
                                          function_kwds=function_kwds,
                                          async_call=async_call,
                                          oneway_call=oneway_call)

    def invoke_method(self,
                      method_name,
                      method_args=(),
                      method_kwds=None,
                      async_call=False,
                      oneway_call=False):
        """Invokes a method on a scenario tree constructs managed
           by this scenario tree manager client. This may or may not
           take place on this client itself.

        Args:
            method_name:
                 The name of the method to be invoked.
            method_args:
                 Arguments passed to the method when it is invoked.
            method_kwds:
                 Keywords to pass to the method when it is invoked.
            async_call:
                 When set to True, the return value will be an
                 asynchronous object. Invocation results can be
                 obtained at any point by calling the complete()
                 method on this object, which will block until all
                 associated action handles are collected.
            oneway_call:
                 When set to True, it will be assumed no return value
                 is expected from this method (async_call is
                 implied). Setting both async_call and oneway_call to True
                 will result in an exception being raised.

            *Note: The 'oneway_call' and 'async_call' keywords are valid
                   for all scenario tree manager client
                   implementations. However, they are designed for use
                   with Pyro-based implementations. Their existence in
                   other implementations is not meant to guarantee
                   asynchronicity, but rather to provide a consistent
                   interface for code to be written around.

        Returns:
            If 'oneway_call' is True, this function will always return
            None. Otherwise, the return corresponds exactly to the
            method's return value, which will be nested inside an
            asynchronous object if 'async_call' is set to True.
        """
        if not self.initialized:
            raise RuntimeError(
                "The scenario tree manager is not initialized.")
        if async_call and oneway_call:
            raise ValueError("async oneway calls do not make sense")
        return self._invoke_method_impl(method_name,
                                        method_args=method_args,
                                        method_kwds=method_kwds,
                                        async_call=async_call,
                                        oneway_call=oneway_call)

    def push_fix_queue_to_instances(self):
        """Push the fixed queue on the scenario tree nodes onto the
        actual variables on the scenario instances.

        * NOTE: This function is poorly named and this functionality
                will likely be changed in the near future. Ideally, fixing
                would be done through the scenario tree manager, rather
                than through the scenario tree.
        """
        if self.get_option("verbose"):
            print("Synchronizing fixed variable statuses on scenario tree nodes")
            node_count = self._push_fix_queue_to_instances_impl()
            if node_count > 0:
                if self.get_option("verbose"):
                    print("Updated fixed statuses on %s scenario tree nodes"
                          % (node_count))
            else:
                if self.get_option("verbose"):
                    print("No synchronization was needed for scenario tree nodes")

    #
    # Methods defined by derived class that are not
    # part of the user interface
    #

    def _init(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _close_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _invoke_function_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _invoke_method_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _process_bundle_solve_result(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _process_scenario_solve_result(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _push_fix_queue_to_instances_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

#
# A base class and interface that is common to client-side scenario
# tree manager implementations (e.g, both the Pyro and Serial
# versions).
#

class ScenarioTreeManagerClient(ScenarioTreeManager,
                                PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()

        #
        # scenario instance construction
        #
        safe_declare_common_option(options,
                                   "model_location")
        safe_declare_common_option(options,
                                   "scenario_tree_location")
        safe_declare_common_option(options,
                                   "objective_sense_stage_based")
        safe_declare_common_option(options,
                                   "postinit_callback_location")
        safe_declare_common_option(options,
                                   "aggregategetter_callback_location")

        #
        # scenario tree generation
        #
        safe_declare_common_option(options,
                                   "scenario_tree_random_seed")
        safe_declare_common_option(options,
                                   "scenario_tree_downsample_fraction")
        safe_declare_common_option(options,
                                   "scenario_bundle_specification")
        safe_declare_common_option(options,
                                   "create_random_bundles")

        #
        # various
        #
        safe_declare_common_option(options,
                                   "output_times")
        safe_declare_common_option(options,
                                   "verbose")
        safe_declare_common_option(options,
                                   "profile_memory")

        return options

    def __init__(self, *args, **kwds):
        if self.__class__ is ScenarioTreeManagerClient:
            raise NotImplementedError(
                "%s is an abstract class for subclassing" % self.__class__)
        factory = kwds.pop("factory", None)
        super(ScenarioTreeManagerClient, self).__init__(*args, **kwds)

        # callback info
        self._scenario_tree = None
        self._callback_function = {}
        self._callback_mapped_module_name = {}
        self._aggregategetter_keys = []
        self._aggregategetter_names = []
        self._postinit_keys = []
        self._postinit_names = []
        self._modules_imported = {}
        if factory is None:
            self._generate_scenario_tree()
            self._import_callbacks()
        else:
            self._scenario_tree = factory.generate_scenario_tree(
                downsample_fraction=self._options.scenario_tree_downsample_fraction,
                bundles=self._options.scenario_bundle_specification,
                random_bundles=self._options.create_random_bundles,
                random_seed=self._options.scenario_tree_random_seed,
                verbose=self._options.verbose)

    def _generate_scenario_tree(self):

        start_time = time.time()
        if self._options.verbose:
            print("Importing model and scenario tree files")

        scenario_instance_factory = \
            ScenarioTreeInstanceFactory(
                self._options.model_location,
                self._options.scenario_tree_location)

        #
        # Try to prevent unnecessarily re-importing the model module
        # if other callbacks are in the same location. Doing so might
        # have serious consequences.
        #
        if scenario_instance_factory._model_module is not None:
            self.modules_imported[scenario_instance_factory.\
                                  _model_filename] = \
                scenario_instance_factory._model_module
        if scenario_instance_factory._scenario_tree_module is not None:
            self.modules_imported[scenario_instance_factory.\
                                  _scenario_tree_filename] = \
                scenario_instance_factory._scenario_tree_module

        if self._options.output_times or \
           self._options.verbose:
            print("Time to import model and scenario tree "
                  "structure files=%.2f seconds"
                  %(time.time() - start_time))

        try:

            self._scenario_tree = \
                scenario_instance_factory.\
                generate_scenario_tree(
                    downsample_fraction=\
                       self._options.scenario_tree_downsample_fraction,
                    bundles=self._options.scenario_bundle_specification,
                    random_bundles=self._options.create_random_bundles,
                    random_seed=self._options.scenario_tree_random_seed,
                    verbose=self._options.verbose)

            # print the input tree for validation/information
            # purposes.
            if self._options.verbose:
                self._scenario_tree.pprint()

            # validate the tree prior to doing anything serious
            self._scenario_tree.validate()
            if self._options.verbose:
                print("Scenario tree is valid!")

        except:
            print("Failed to generate scenario tree")
            scenario_instance_factory.close()
            raise

    def _import_callbacks(self):

        renamed = {}
        renamed["pysp_aggregategetter_callback"] = \
            "ph_aggregategetter_callback"
        renamed["pysp_postinit_callback"] = \
            "ph_boundsetter_callback"
        for module_names, attr_name, callback_name in (
                (self._options.aggregategetter_callback_location,
                 "_aggregategetter",
                 "pysp_aggregategetter_callback"),
                (self._options.postinit_callback_location,
                 "_postinit",
                 "pysp_postinit_callback")):

            assert callback_name in renamed.keys()
            deprecated_callback_name = renamed[callback_name]
            for module_name in module_names:
                if module_name in self.modules_imported:
                    module = self.modules_imported[module_name]
                    sys_modules_key = module_name
                else:
                    module, sys_modules_key = \
                        load_external_module(module_name,
                                             clear_cache=True,
                                             verbose=self.get_option("verbose"))
                    self.modules_imported[module_name] = module
                callback = None
                for oname, obj in inspect.getmembers(module):
                    if oname == callback_name:
                        callback = obj
                        break
                if callback is None:
                    for oname, obj in inspect.getmembers(module):
                        if oname == deprecated_callback_name:
                            callback = obj
                            break
                    if callback is None:
                        raise ImportError(
                            "PySP callback with name '%s' could "
                            "not be found in module file: %s"
                            % (deprecated_callback_name, module_name))
                    if callback is None:
                        raise ImportError(
                            "PySP callback with name '%s' could "
                            "not be found in module file: %s"
                            % (callback_name, module_name))
                    else:
                        logger.warning(
                            "DEPRECATED: Callback with name '%s' "
                            "has been renamed '%s'"
                            % (deprecated_callback_name,
                               callback_name))
                        callback_name = deprecated_callback_name

                self._callback_function[sys_modules_key] = callback
                getattr(self, attr_name+"_keys").append(sys_modules_key)
                getattr(self, attr_name+"_names").append(callback_name)
                self._callback_mapped_module_name\
                    [sys_modules_key] = module_name

    #
    # Interface
    #

    @property
    def modules_imported(self):
        return self._modules_imported

    # override initialize on ScenarioTreeManager for documentation purposes
    def initialize(self, async_call=False):
        """Initialize the scenario tree manager client.

        Note: Calling complete() on an asynchronous result
              returned from this method will causes changes
              in the state of this object. One should avoid
              using the client until initialization is
              complete.

        Args:
            async_call:
                 When set to True, the return value will be an
                 asynchronous object. Invocation results can be
                 obtained at any point by calling the complete()
                 method on this object, which will block until all
                 associated action handles are collected.

        Returns:
            A dictionary mapping scenario tree worker names to their
            initial return value (True is most cases). If 'async_call'
            is set to True, this return value will be nested inside an
            asynchronous object.
        """
        return super(ScenarioTreeManagerClient, self).initialize(async_call=async_call)

    def invoke_function_on_worker(self,
                                  worker_name,
                                  function,
                                  module_name=None,
                                  invocation_type=InvocationType.Single,
                                  function_args=(),
                                  function_kwds=None,
                                  async_call=False,
                                  oneway_call=False):
        """Invokes a function on a scenario tree worker
        managed by this scenario tree manager client. The
        first argument accepted by the function must always
        be the process-local scenario tree worker object,
        which may or may not be this object.

        Args:
            worker_name:
                 The name of the scenario tree worker. The
                 list of worker names can be found at
                 client.worker_names.
            function:
                 The function or name of the function to be
                 invoked. If the object is a function, then
                 the manager will attempt to transmit it
                 using the dill package. Otherwise, the
                 argument must be a string and the
                 module_name keyword is required.
            module_name:
                 The name of the module containing the
                 function. This can also be an absolute path
                 to a file that contains the function
                 definition. If this function argument is an
                 actual function, this keyword must be left
                 at its default value of None; otherwise, it
                 is required.
            invocation_type:
                 Controls how the function is invoked. Refer
                 to the doc string for
                 pyomo.pysp.scenariotree.manager.InvocationType
                 for more information.
            function_args:
                 Extra arguments passed to the function when
                 it is invoked. These will always be placed
                 after the initial process-local scenario
                 tree worker object as well as any
                 additional arguments governed by the
                 invocation type.
            function_kwds:
                 Additional keywords to pass to the function
                 when it is invoked.
            async_call:
                 When set to True, the return value will be
                 an asynchronous object. Invocation results
                 can be obtained at any point by calling the
                 complete() method on this object, which
                 will block until all associated action
                 handles are collected.
            oneway_call:
                 When set to True, it will be assumed no return value
                 is expected from this function (async_call is
                 implied). Setting both async and oneway_call to True will
                 result in an exception being raised.

            *Note: The 'oneway_call' and 'async_call' keywords are valid
                   for all scenario tree manager
                   implementations. However, they are designed for use
                   with Pyro-based implementations. Their existence in
                   other implementations is not meant to guarantee
                   asynchronicity, but rather to provide a consistent
                   interface for code to be written around.

        Returns:
            If 'oneway_call' is True, this function will always
            return None. Otherwise, the return value type is
            governed by the 'invocation_type' keyword, which
            will be nested inside an asynchronous object if
            'async_call' is set to True.
        """
        if not self.initialized:
            raise RuntimeError(
                "The scenario tree manager is not initialized.")
        if async_call and oneway_call:
            raise ValueError("async oneway calls do not make sense")
        invocation_type = _map_deprecated_invocation_type(invocation_type)
        if (invocation_type == InvocationType.PerBundle) or \
           (invocation_type == InvocationType.PerBundleChained) or \
           (invocation_type == InvocationType.OnBundle) or \
           (invocation_type == InvocationType.OnBundles) or \
           (invocation_type == InvocationType.OnBundlesChained):
            if not self._scenario_tree.contains_bundles():
                raise ValueError(
                    "Received request for bundle invocation type "
                    "but the scenario tree does not contain bundles.")
        return self._invoke_function_on_worker_impl(worker_name,
                                                    function,
                                                    module_name=module_name,
                                                    invocation_type=invocation_type,
                                                    function_args=function_args,
                                                    function_kwds=function_kwds,
                                                    async_call=async_call,
                                                    oneway_call=oneway_call)

    def invoke_method_on_worker(self,
                                worker_name,
                                method_name,
                                method_args=(),
                                method_kwds=None,
                                async_call=False,
                                oneway_call=False):
        """Invokes a method on a scenario tree worker managed
           by this scenario tree manager client. The worker
           may or may not be this client.

        Args:
            worker_name:
                 The name of the scenario tree worker. The list of worker
                 names can be found at client.worker_names.
            method_name:
                 The name of the worker method to be invoked.
            method_args:
                 Arguments passed to the method when it is invoked.
            method_kwds:
                 Keywords to pass to the method when it is invoked.
            async_call:
                 When set to True, the return value will be an
                 asynchronous object. Invocation results can be
                 obtained at any point by calling the complete()
                 method on this object, which will block until all
                 associated action handles are collected.
            oneway_call:
                 When set to True, it will be assumed no return value
                 is expected from this method (async_call is
                 implied). Setting both async and oneway_call to True will
                 result in an exception being raised.

            *Note: The 'oneway_call' and 'async_call' keywords are valid
                   for all scenario tree manager client
                   implementations. However, they are designed for use
                   with Pyro-based implementations. Their existence in
                   other implementations is not meant to guarantee
                   asynchronicity, but rather to provide a consistent
                   interface for code to be written around.

        Returns:
            If 'oneway_call' is True, this function will always return
            None. Otherwise, the return corresponds exactly to the
            method's return value, which will be nested inside an
            asynchronous object if 'async_call' is set to True.
        """
        if not self.initialized:
            raise RuntimeError(
                "The scenario tree manager is not initialized.")
        if async_call and oneway_call:
            raise ValueError("async oneway calls do not make sense")
        return self._invoke_method_on_worker_impl(worker_name,
                                                  method_name,
                                                  method_args=method_args,
                                                  method_kwds=method_kwds,
                                                  async_call=async_call,
                                                  oneway_call=oneway_call)

    @property
    def worker_names(self):
        """The list of worker names managed by this client."""
        return self._worker_names_impl()

    def get_worker_for_scenario(self, scenario_name):
        """Get the worker name assigned to the scenario with the given name."""
        if not self._scenario_tree.contains_scenario(scenario_name):
            raise KeyError("Scenario with name %s does not exist "
                           "in the scenario tree" % (scenario_name))
        return self._get_worker_for_scenario_impl(scenario_name)

    def get_worker_for_bundle(self, bundle_name):
        """Get the worker name assigned to the bundle with the given name."""
        if not self._scenario_tree.contains_bundle(bundle_name):
            raise KeyError("Bundle with name %s does not exist "
                           "in the scenario tree" % (bundle_name))
        return self._get_worker_for_bundle_impl(bundle_name)

    def get_scenarios_for_worker(self, worker_name):
        """Get the list of scenario names assigned to the worker with
        the given name."""
        if worker_name not in self.worker_names:
            raise KeyError("Worker with name %s does not exist under "
                           "in this client" % (worker_name))
        return self._get_scenarios_for_worker_impl(worker_name)

    def get_bundles_for_worker(self, worker_name):
        """Get the list of bundle names assigned to the worker with
        the given name."""
        if worker_name not in self.worker_names:
            raise KeyError("Worker with name %s does not exist under "
                           "in this client" % (worker_name))
        return self._get_bundles_for_worker_impl(worker_name)

    #
    # Partially implement _init for ScenarioTreeManager
    # subclasses are now expected to generate a (possibly
    # dummy) async object during _init_client
    #
    def _init(self, async_call=False):
        async_handle = self._init_client()
        if async_call:
            result = async_handle
        else:
            result = async_handle.complete()
        return result

    #
    # Methods defined by derived class that are not
    # part of the user interface
    #

    def _init_client(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _invoke_function_on_worker_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _invoke_method_on_worker_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _worker_names_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _get_worker_for_scenario_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _get_worker_for_bundle_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _get_scenarios_for_worker_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _get_bundles_for_worker_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

#
# A partial implementation of the ScenarioTreeManager
# interface that is common to both the Serial scenario
# tree manager as well as the Pyro workers used by the
# Pyro scenario tree manager.
#

class _ScenarioTreeManagerWorker(PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()

        safe_declare_common_option(options,
                                   "output_times")
        safe_declare_common_option(options,
                                   "verbose")

        return options

    def __init__(self, *args, **kwds):
        if self.__class__ is _ScenarioTreeManagerWorker:
            raise NotImplementedError(
                "%s is an abstract class for subclassing" % self.__class__)
        super(_ScenarioTreeManagerWorker, self).__init__(*args, **kwds)

        # scenario instance models
        self._instances = None

        # bundle instance models
        self._bundle_binding_instance_map = {}

        # results objects from the most recent call to
        # _process_*_solve_results, may hold more
        # information than just variable values and so can
        # be useful to hold on to until the next round of
        # solves (keys are bundle name or scenario name)
        self._solve_results = {}

        # set by advanced solver managers
        self.preprocessor = None

    #
    # Extension of the manager interface so code can handle
    # cases where multiple workers own a different portions of the
    # scenario tree.
    #

    @property
    def uncompressed_scenario_tree(self):
        raise NotImplementedError                  #pragma:nocover

    def _invoke_function_by_worker(self,
                                   function,
                                   module_name=None,
                                   invocation_type=InvocationType.Single,
                                   function_args=(),
                                   function_kwds=None):

        if function_kwds is None:
            function_kwds = {}

        if not isinstance(function, six.string_types):
            if module_name is not None:
                raise ValueError(
                    "The module_name keyword must be None "
                    "when the function argument is not a string.")
        else:
            if module_name is None:
                raise ValueError(
                    "A module name is required when "
                    "a function name is given")
            elif module_name in self.modules_imported:
                this_module = self.modules_imported[module_name]
            elif module_name in sys.modules:
                this_module = sys.modules[module_name]
            else:
                this_module = pyutilib.misc.import_file(module_name,
                                                        clear_cache=True)
                self.modules_imported[module_name] = this_module
                self.modules_imported[this_module.__file__] = this_module
                if this_module.__file__.endswith(".pyc"):
                    self.modules_imported[this_module.__file__[:-1]] = \
                        this_module

            module_attrname = function
            subname = None
            if not hasattr(this_module, module_attrname):
                if "." in module_attrname:
                    module_attrname, subname = function.split(".",1)
                if not hasattr(this_module, module_attrname):
                    raise RuntimeError(
                        "Function="+function+" is not present "
                        "in module="+module_name)

            function = getattr(this_module, module_attrname)
            if subname is not None:
                function = getattr(function, subname)

        call_objects = None
        if invocation_type == InvocationType.Single:
            pass
        elif (invocation_type == InvocationType.PerScenario) or \
             (invocation_type == InvocationType.PerScenarioChained):
            call_objects = self._scenario_tree.scenarios
        elif (invocation_type == InvocationType.OnScenario):
            call_objects = [self._scenario_tree.get_scenario(invocation_type.data)]
        elif (invocation_type == InvocationType.OnScenarios) or \
             (invocation_type == InvocationType.OnScenariosChained):
            assert len(invocation_type.data) != 0
            call_objects = [self._scenario_tree.get_scenario(scenario_name)
                            for scenario_name in invocation_type.data]
        elif (invocation_type == InvocationType.PerBundle) or \
             (invocation_type == InvocationType.PerBundleChained):
            assert self._scenario_tree.contains_bundles()
            call_objects = self._scenario_tree.bundles
        elif (invocation_type == InvocationType.OnBundle):
            assert self._scenario_tree.contains_bundles()
            call_objects = [self._scenario_tree.get_bundle(invocation_type.data)]
        elif (invocation_type == InvocationType.OnBundles) or \
             (invocation_type == InvocationType.OnBundlesChained):
            assert self._scenario_tree.contains_bundles()
            assert len(invocation_type.data) != 0
            call_objects = [self._scenario_tree.get_bundle(bundle_name)
                            for bundle_name in invocation_type.data]
        else:
            raise ValueError("Unexpected function invocation type '%s'. "
                             "Expected one of %s"
                             % (invocation_type,
                                [str(v) for v in InvocationType]))

        result = None
        if (invocation_type == InvocationType.Single):

            result = function(self,
                              *function_args,
                              **function_kwds)

        elif (invocation_type == InvocationType.OnScenario) or \
             (invocation_type == InvocationType.OnBundle):

            assert len(call_objects) == 1
            result = function(self,
                              call_objects[0],
                              *function_args,
                              **function_kwds)

        elif (invocation_type == InvocationType.PerScenarioChained) or \
             (invocation_type == InvocationType.OnScenariosChained) or \
             (invocation_type == InvocationType.PerBundleChained) or \
             (invocation_type == InvocationType.OnBundlesChained):

            if len(function_args) > 0:
                result = function_args
                for call_object in call_objects:
                    result = function(self,
                                      call_object,
                                      *result,
                                      **function_kwds)
            else:
                result = None
                for call_object in call_objects:
                    result = function(self,
                                      call_object,
                                      **function_kwds)
        else:

            result = dict((call_object.name, function(self,
                                                       call_object,
                                                       *function_args,
                                                       **function_kwds))
                          for call_object in call_objects)

        return result

    # override what is already implemented
    def _init_bundle(self, bundle_name, scenario_list):
        super(_ScenarioTreeManagerWorker, self).\
            _init_bundle(bundle_name, scenario_list)

        if self._options.verbose:
            print("Forming binding instance for scenario bundle %s"
                  % (bundle_name))

        start_time = time.time()

        assert self._scenario_tree.contains_bundle(bundle_name)

        assert bundle_name not in self._bundle_binding_instance_map

        bundle = self._scenario_tree.get_bundle(bundle_name)

        for scenario_name in bundle._scenario_names:
            scenario = self._scenario_tree.get_scenario(scenario_name)
            assert scenario_name in self._scenario_to_bundle_map
            assert self._scenario_to_bundle_map[scenario_name] == bundle_name
            assert scenario._instance is not None
            assert scenario._instance is self._instances[scenario_name]
            assert scenario._instance.parent_block() is None

        # IMPORTANT: The bundle variable IDs must be idential to
        #            those in the parent scenario tree - this is
        #            critical for storing results, which occurs at
        #            the full-scale scenario tree.

        bundle._scenario_tree.linkInInstances(
            self._instances,
            create_variable_ids=False,
            master_scenario_tree=self._scenario_tree,
            initialize_solution_data=False)

        bundle_ef_instance = create_ef_instance(
            bundle._scenario_tree,
            ef_instance_name=bundle.name,
            verbose_output=self._options.verbose)

        self._bundle_binding_instance_map[bundle.name] = \
            bundle_ef_instance

        end_time = time.time()
        if self._options.output_times or \
           self._options.verbose:
            print("Time construct binding instance for scenario bundle "
                  "%s=%.2f seconds" % (bundle_name, end_time - start_time))

    # override what is already implemented
    def _release_bundle(self, bundle_name):

        assert self._scenario_tree.contains_bundle(bundle_name)
        assert bundle_name in self._bundle_binding_instance_map

        bundle_ef_instance = \
            self._bundle_binding_instance_map[bundle_name]

        bundle = self._scenario_tree.get_bundle(bundle_name)

        for scenario_name in bundle._scenario_names:
            scenario = self._scenario_tree.get_scenario(scenario_name)
            bundle_ef_instance.del_component(scenario._instance)
            scenario._instance_objective.activate()

        del self._bundle_binding_instance_map[bundle_name]

        # call the base class method
        super(_ScenarioTreeManagerWorker, self).\
            _release_bundle(bundle_name)

    #
    # Abstract methods for ScenarioTreeManager:
    #

    def _close_impl(self):
        # copy the list of bundle names as the next loop will modify
        # the scenario_tree._scenario_bundles list
        if self._scenario_tree is not None:
            bundle_names = \
                [bundle.name for bundle in self._scenario_tree._scenario_bundles]
            for bundle_name in bundle_names:
                self._release_bundle(bundle_name)
        self._instances = None
        self._bundle_binding_instance_map = None

    def _process_bundle_solve_result(self,
                                     bundle_name,
                                     results,
                                     manager_results=None,
                                     **kwds):

        if manager_results is None:
            manager_results = ScenarioTreeSolveResults('bundles')

        bundle = self.scenario_tree.get_bundle(bundle_name)
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
            manager_results.solve_time[bundle_name] = \
                float(results.solver.user_time)
        elif hasattr(results.solver,"wallclock_time") and \
             (not isinstance(results.solver.wallclock_time,
                             UndefinedData))and \
             (results.solver.wallclock_time is not None):
            manager_results.solve_time[bundle_name] = \
                float(results.solver.wallclock_time)
        elif hasattr(results.solver,"time"):
            solve_time = results.solver.time
            manager_results.solve_time[bundle_name] = \
                float(results.solver.time)
        else:
            manager_results.solve_time[bundle_name] = undefined

        if hasattr(results,"pyomo_solve_time"):
            manager_results.pyomo_solve_time[bundle_name] = \
                results.pyomo_solve_time
        else:
            manager_results.pyomo_solve_time[bundle_name] = undefined

        manager_results.solver_status[bundle_name] = \
            results.solver.status
        manager_results.solver_message[bundle_name] = \
            results.solver.message
        manager_results.termination_condition[bundle_name] = \
            results.solver.termination_condition

        if len(results.solution) > 0:
            assert len(results.solution) == 1

            results_sm = results._smap
            bundle_instance.solutions.load_from(results, **kwds)
            self._solve_results[bundle_name] = (results, results_sm)

            solution0 = results.solution(0)
            if hasattr(solution0, "gap") and \
               (solution0.gap is not None):
                manager_results.gap[bundle_name] = solution0.gap
            else:
                manager_results.gap[bundle_name] = undefined

            manager_results.solution_status[bundle_name] = solution0.status

            bundle_objective_value = 0.0
            bundle_cost_value = 0.0
            for bundle_scenario in bundle._scenario_tree._scenarios:
                scenario = self.scenario_tree.\
                           get_scenario(bundle_scenario.name)
                scenario.update_solution_from_instance()
                # And we need to make sure to use the
                # probabilities assigned to scenarios in the
                # compressed bundle scenario tree
                bundle_objective_value += scenario._objective * \
                                          bundle_scenario.probability
                bundle_cost_value += scenario._cost * \
                                     bundle_scenario.probability

            manager_results.objective[bundle_name] = bundle_objective_value
            manager_results.cost[bundle_name] = bundle_cost_value

        else:

            manager_results.objective[bundle_name] = undefined
            manager_results.cost[bundle_name] = undefined
            manager_results.gap[bundle_name] = undefined
            manager_results.solution_status[bundle_name] = undefined

        return manager_results

    def _process_scenario_solve_result(self,
                                       scenario_name,
                                       results,
                                       manager_results=None,
                                       **kwds):

        if manager_results is None:
            manager_results = ScenarioTreeSolveResults('scenarios')

        scenario = self.scenario_tree.get_scenario(scenario_name)
        scenario_instance = scenario._instance
        if self.scenario_tree.contains_bundles():
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
            manager_results.solve_time[scenario_name] = \
                float(results.solver.user_time)
        elif hasattr(results.solver,"wallclock_time") and \
             (not isinstance(results.solver.wallclock_time,
                             UndefinedData))and \
             (results.solver.wallclock_time is not None):
            manager_results.solve_time[scenario_name] = \
                float(results.solver.wallclock_time)
        elif hasattr(results.solver,"time"):
            manager_results.solve_time[scenario_name] = \
                float(results.solver.time)
        else:
            manager_results.solve_time[scenario_name] = undefined

        if hasattr(results,"pyomo_solve_time"):
            manager_results.pyomo_solve_time[scenario_name] = \
                results.pyomo_solve_time
        else:
            manager_results.pyomo_solve_time[scenario_name] = undefined

        manager_results.solver_status[scenario_name] = \
            results.solver.status
        manager_results.solver_message[scenario_name] = \
            results.solver.message
        manager_results.termination_condition[scenario_name] = \
            results.solver.termination_condition

        if len(results.solution) > 0:
            assert len(results.solution) == 1

            results_sm = results._smap
            scenario_instance.solutions.load_from(results, **kwds)
            self._solve_results[scenario.name] = (results, results_sm)

            scenario.update_solution_from_instance()

            solution0 = results.solution(0)
            if hasattr(solution0, "gap") and \
               (solution0.gap is not None):
                manager_results.gap[scenario_name] = solution0.gap
            else:
                manager_results.gap[scenario_name] = undefined

            manager_results.solution_status[scenario_name] = solution0.status
            manager_results.objective[scenario_name] = scenario._objective
            manager_results.cost[scenario_name] = scenario._cost

        else:

            manager_results.objective[scenario_name] = undefined
            manager_results.cost[scenario_name] = undefined
            manager_results.gap[scenario_name] = undefined
            manager_results.solution_status[scenario_name] = undefined

        return manager_results

    def _push_fix_queue_to_instances_impl(self):

        node_count = 0
        for tree_node in self._scenario_tree._tree_nodes:

            if len(tree_node._fix_queue):
                node_count += 1
                if self.preprocessor is not None:
                    for scenario in tree_node._scenarios:
                        scenario_name = scenario.name
                        for variable_id, (fixed_status, new_value) in \
                              iteritems(tree_node._fix_queue):
                            variable_name, index = \
                                tree_node._variable_ids[variable_id]
                            if fixed_status == tree_node.VARIABLE_FREED:
                                self.preprocessor.\
                                    freed_variables[scenario_name].\
                                    append((variable_name, index))
                            elif fixed_status == tree_node.VARIABLE_FIXED:
                                self.preprocessor.\
                                    fixed_variables[scenario_name].\
                                    append((variable_name, index))

            tree_node.push_fix_queue_to_instances()

        return node_count

#
# The Serial scenario tree manager class. This is a full
# implementation of the ScenarioTreeManager, ScenarioTreeManagerClient
# and _ScenarioTreeManagerWorker interfaces
#

class ScenarioTreeManagerClientSerial(_ScenarioTreeManagerWorker,
                                      ScenarioTreeManagerClient,
                                      PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()

        #
        # scenario instance construction
        #
        safe_declare_common_option(options,
                                   "output_instance_construction_time")
        safe_declare_common_option(options,
                                   "compile_scenario_instances")

        return options

    def __init__(self, *args, **kwds):
        self._worker_name = 'ScenarioTreeManagerClientSerial:MainWorker'
        # good to have to keep deterministic ordering in code
        # rather than loop over the keys of the map on the
        # scenario tree
        self._scenario_names = []
        self._bundle_names = []
        super(ScenarioTreeManagerClientSerial, self).__init__(*args, **kwds)

    # override what is implemented by _ScenarioTreeManagerWorker
    def _init_bundle(self, bundle_name, scenario_list):
        super(ScenarioTreeManagerClientSerial, self).\
            _init_bundle(bundle_name, scenario_list)
        assert bundle_name not in self._bundle_names
        self._bundle_names.append(bundle_name)

    # override what is implemented by _ScenarioTreeManagerWorker
    def _release_bundle(self, bundle_name):
        super(ScenarioTreeManagerClientSerial, self).\
            _release_bundle(bundle_name)
        assert bundle_name in self._bundle_names
        self._bundle_names.remove(bundle_name)

    @property
    def uncompressed_scenario_tree(self):
        return self._scenario_tree

    #
    # Abstract methods for ScenarioTreeManagerClient:
    #

    def _init_client(self):
        assert self._scenario_tree is not None

        #
        # Build scenario instances
        #

        build_start_time = time.time()

        if self._options.verbose:
            print("Constructing scenario tree instances")

        self._instances = \
            self._scenario_tree._scenario_instance_factory.\
            construct_instances_for_scenario_tree(
                self._scenario_tree,
                output_instance_construction_time=\
                   self._options.output_instance_construction_time,
                profile_memory=self._options.profile_memory,
                compile_scenario_instances=\
                    self._options.compile_scenario_instances,
                verbose=self._options.verbose)

        if self._options.output_times or \
           self._options.verbose:
            print("Time to construct scenario instances="
                  "%.2f seconds"
                  % (time.time() - build_start_time))

        if self._options.verbose:
            print("Linking instances into scenario tree")

        build_start_time = time.time()

        # with the scenario instances now available, link the
        # referenced objects directly into the scenario tree.
        self._scenario_tree.linkInInstances(
            self._instances,
            objective_sense=self._options.objective_sense_stage_based,
            create_variable_ids=True)

        self._objective_sense = \
            self.scenario_tree._scenarios[0]._objective_sense
        assert all(_s._objective_sense == self._objective_sense
                   for _s in self.scenario_tree._scenarios)

        self._scenario_names = [_scenario.name for _scenario in
                                self._scenario_tree._scenarios]
        if self._options.output_times or \
           self._options.verbose:
            print("Time link scenario tree with instances="
                  "%.2f seconds" % (time.time() - build_start_time))

        #
        # Create bundle instances if needed
        #
        if self._scenario_tree.contains_bundles():
            start_time = time.time()
            if self._options.verbose:
                print("Construction extensive form instances for all bundles.")

            for bundle in self._scenario_tree._scenario_bundles:
                self._init_bundle(bundle.name, bundle._scenario_names)

            end_time = time.time()
            if self._options.output_times or \
               self._options.verbose:
                print("Scenario bundle construction time=%.2f seconds"
                      % (end_time - start_time))

        if len(self._options.aggregategetter_callback_location):
            # Run the user script to collect aggregate scenario data
            for callback_module_key in self._aggregategetter_keys:
                if self._options.verbose:
                    print("Executing user defined aggregategetter callback function "
                          "defined in module: %s"
                          % (self._callback_mapped_module_name[callback_module_key]))
                for scenario in self._scenario_tree._scenarios:
                    result = self._callback_function[callback_module_key](
                        self,
                        scenario,
                        self._aggregate_user_data)
                    assert len(result) == 1
                    self._aggregate_user_data.update(result[0])

        if len(self._options.postinit_callback_location):
            # run the user script to initialize variable bounds
            for callback_module_key in self._postinit_keys:
                if self._options.verbose:
                    print("Executing user defined posinit callback function "
                          "defined in module: %s"
                          % (self._callback_mapped_module_name[callback_module_key]))
                for scenario in self._scenario_tree._scenarios:
                    self._callback_function[callback_module_key](
                        self,
                        scenario)

        return self.AsyncResult(
            None, result={self._worker_name: True})

    def _invoke_function_on_worker_impl(self,
                                        worker_name,
                                        function,
                                        module_name=None,
                                        invocation_type=InvocationType.Single,
                                        function_args=(),
                                        function_kwds=None,
                                        async_call=False,
                                        oneway_call=False):

        assert worker_name == self._worker_name
        start_time = time.time()

        if self._options.verbose:
            print("Transmitting external function invocation request "
                  "for function=%s in module=%s on worker=%s."
                  % (str(function), module_name, worker_name))

        result = self._invoke_function_by_worker(function,
                                                 module_name=module_name,
                                                 invocation_type=invocation_type,
                                                 function_args=function_args,
                                                 function_kwds=function_kwds)

        if oneway_call:
            result = None
        if async_call:
            result = self.AsyncResult(None, result=result)

        end_time = time.time()
        if self._options.output_times or \
           self._options.verbose:
            print("Function invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    def _invoke_method_on_worker_impl(self,
                                      worker_name,
                                      method_name,
                                      method_args=(),
                                      method_kwds=None,
                                      async_call=False,
                                      oneway_call=False):

        assert worker_name == self._worker_name
        start_time = time.time()

        if self._options.verbose:
            print("Invoking method=%s on worker=%s"
                  % (method_name, self._worker_name))

        if method_kwds is None:
            method_kwds = {}
        result = getattr(self, method_name)(*method_args, **method_kwds)

        if oneway_call:
            result = None
        if async_call:
            result = self.AsyncResult(None, result=result)

        end_time = time.time()
        if self._options.output_times or \
           self._options.verbose:
            print("Method invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    def _worker_names_impl(self):
        return (self._worker_name,)

    def _get_worker_for_scenario_impl(self, scenario_name):
        assert self._scenario_tree.contains_scenario(scenario_name)
        return self._worker_name

    def _get_worker_for_bundle_impl(self, bundle_name):
        assert self._scenario_tree.contains_bundle(bundle_name)
        return self._worker_name

    def _get_scenarios_for_worker_impl(self, worker_name):
        assert worker_name == self._worker_name
        return self._scenario_names

    def _get_bundles_for_worker_impl(self, worker_name):
        assert worker_name == self._worker_name
        return self._bundle_names

    #
    # Abstract methods for ScenarioTreeManager:
    #

    # implemented by _ScenarioTreeManagerWorker
    #def _close_impl(...)

    def _invoke_function_impl(self,
                              function,
                              module_name=None,
                              invocation_type=InvocationType.Single,
                              function_args=(),
                              function_kwds=None,
                              async_call=False,
                              oneway_call=False):
        assert not (async_call and oneway_call)

        result = self._invoke_function_on_worker_impl(
            self._worker_name,
            function,
            module_name=module_name,
            invocation_type=invocation_type,
            function_args=function_args,
            function_kwds=function_kwds,
            async_call=False,
            oneway_call=oneway_call)

        if not oneway_call:
            if invocation_type == InvocationType.Single:
                result = {self._worker_name: result}
        if async_call:
            result = self.AsyncResult(None, result=result)

        return result

    def _invoke_method_impl(self,
                            method_name,
                            method_args=(),
                            method_kwds=None,
                            async_call=False,
                            oneway_call=False):
        assert not (async_call and oneway_call)

        result =  self._invoke_method_on_worker_impl(
            self._worker_name,
            method_name,
            method_args=method_args,
            method_kwds=method_kwds,
            async_call=False,
            oneway_call=oneway_call)

        if not oneway_call:
            result = {self._worker_name: result}
        if async_call:
            result = self.AsyncResult(None, result=result)

        return result

#
# A partial implementation of the ScenarioTreeManager and
# ScenarioTreeManagerClient interfaces for Pyro that may serve some
# future purpose where there is not a one-to-one mapping between
# worker objects and scenarios / bundles in the scenario tree.
#

class _ScenarioTreeManagerClientPyroAdvanced(ScenarioTreeManagerClient,
                                             PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()

        safe_declare_common_option(options,
                                   "pyro_host")
        safe_declare_common_option(options,
                                   "pyro_port")
        safe_declare_common_option(options,
                                   "pyro_shutdown")
        safe_declare_common_option(options,
                                   "pyro_shutdown_workers")

        return options

    def __init__(self, *args, **kwds):
        # distributed worker information
        self._pyro_server_workers_map = {}
        self._pyro_worker_server_map = {}
        # the same as the .keys() of the above map
        # but won't suffer from stochastic iteration
        # order python dictionaries
        self._pyro_worker_list = []
        self._pyro_worker_scenarios_map = {}
        self._pyro_worker_bundles_map = {}
        self._action_manager = None
        self._transmission_paused = False
        super(_ScenarioTreeManagerClientPyroAdvanced, self).__init__(*args, **kwds)

    def _invoke_function_on_worker_pyro(self,
                                        worker_name,
                                        function,
                                        module_name=None,
                                        invocation_type=InvocationType.Single,
                                        function_args=(),
                                        function_kwds=None,
                                        oneway_call=False):

        return self._action_manager.queue(
            queue_name=self.get_server_for_worker(worker_name),
            worker_name=worker_name,
            action="_invoke_function_impl",
            generate_response=not oneway_call,
            args=(function,),
            kwds={'module_name': module_name,
                  'invocation_type': (invocation_type.key,
                                      getattr(invocation_type, 'data', None)),
                  'function_args': function_args,
                  'function_kwds': function_kwds})

    def _invoke_method_on_worker_pyro(
            self,
            worker_name,
            method_name,
            method_args=(),
            method_kwds=None,
            oneway_call=False):

        return self._action_manager.queue(
            queue_name=self.get_server_for_worker(worker_name),
            worker_name=worker_name,
            action="_invoke_method_impl",
            generate_response=not oneway_call,
            args=(method_name,),
            kwds={'method_args': method_args,
                  'method_kwds': method_kwds})

    #
    # Abstract methods for ScenarioTreeManagerClient:
    #

    def _init_client(self):
        assert self._scenario_tree is not None
        return self.AsyncResult(None, result=True)

    def _invoke_function_on_worker_impl(self,
                                        worker_name,
                                        function,
                                        module_name=None,
                                        invocation_type=InvocationType.Single,
                                        function_args=(),
                                        function_kwds=None,
                                        async_call=False,
                                        oneway_call=False):
        assert not (async_call and oneway_call)
        assert self._action_manager is not None
        assert worker_name in self._pyro_worker_list
        start_time = time.time()

        if self._options.verbose:
            print("Invoking external function=%s in module=%s "
                  "on worker=%s"
                  % (str(function), module_name, worker_name))

        if not isinstance(function, six.string_types):
            if not dill_available:
                raise ValueError(
                    "The dill module must be available "
                    "when transmitting function objects")
            if module_name is not None:
                raise ValueError(
                    "The module_name keyword must be None "
                    "when the function argument is not a string.")
            function = dill.dumps(function)
        else:
            if module_name is None:
                raise ValueError(
                    "A module name is required when "
                    "a function name is given")

        action_handle = self._invoke_function_on_worker_pyro(
            worker_name,
            function,
            module_name=module_name,
            invocation_type=invocation_type,
            function_args=function_args,
            function_kwds=function_kwds,
            oneway_call=oneway_call)

        if oneway_call:
            action_handle = None

        result = self.AsyncResult(
            self._action_manager, action_handle_data=action_handle)

        if not async_call:
            result = result.complete()

        end_time = time.time()
        if self._options.output_times or \
           self._options.verbose:
            print("External function invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    def _invoke_method_on_worker_impl(self,
                                      worker_name,
                                      method_name,
                                      method_args=(),
                                      method_kwds=None,
                                      async_call=False,
                                      oneway_call=False):

        assert self._action_manager is not None
        assert worker_name in self._pyro_worker_list
        start_time = time.time()

        if self._options.verbose:
            print("Invoking method=%s on worker=%s"
                  % (method_name, worker_name))

        action_handle = self._invoke_method_on_worker_pyro(
            worker_name,
            method_name,
            method_args=method_args,
            method_kwds=method_kwds,
            oneway_call=oneway_call)

        if oneway_call:
            action_handle = None

        result = self.AsyncResult(
            self._action_manager, action_handle_data=action_handle)

        if not async_call:
            result = result.complete()

        end_time = time.time()
        if self._options.output_times or \
           self._options.verbose:
            print("Method invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    def _worker_names_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _get_worker_for_scenario_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _get_worker_for_bundle_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _get_scenarios_for_worker_impl(self, worker_name):
        assert worker_name in self._pyro_worker_list
        return self._pyro_worker_scenarios_map[worker_name]

    def _get_bundles_for_worker_impl(self, worker_name):
        assert worker_name in self._pyro_worker_list
        return self._pyro_worker_bundles_map[worker_name]

    #
    # Abstract methods for ScenarioTreeManager:
    #

    def _close_impl(self):
        if self._action_manager is not None:
            if self._error_shutdown:
                self.release_scenariotreeservers(ignore_errors=2)
            else:
                self.release_scenariotreeservers()
        if self._options.pyro_shutdown:
            print("Shutting down Pyro components.")
            shutdown_pyro_components(
                host=self._options.pyro_host,
                port=self._options.pyro_port,
                num_retries=0,
                caller_name=self.__class__.__name__)

    def _invoke_function_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    def _invoke_method_impl(self, *args, **kwds):
        raise NotImplementedError                  #pragma:nocover

    #
    # Extended interface for Pyro
    #

    def acquire_scenariotreeservers(self, num_servers, timeout=None):
        """Acquire a pool of scenario tree servers and initialize the
        action manager."""

        assert self._action_manager is None
        self._action_manager = ScenarioTreeActionManagerPyro(
            verbose=self._options.verbose,
            host=self._options.pyro_host,
            port=self._options.pyro_port)
        self._action_manager.acquire_servers(num_servers, timeout=timeout)

        scenario_instance_factory = \
            self._scenario_tree._scenario_instance_factory
        server_init = {}
        if scenario_instance_factory._model_filename is not None:
            server_init['model'] = \
                scenario_instance_factory._model_filename
        elif scenario_instance_factory._model_object is not None:
            # we are pickling a model!
            server_init['model'] = \
                scenario_instance_factory._model_object
        else:
            assert scenario_instance_factory._model_callback is not None
            if dill_available:
                server_init['model_callback'] = \
                    dill.dumps(scenario_instance_factory._model_callback)
            else:
                raise ValueError(
                    "The dill module is required in order to "
                    "initialize the Pyro-based scenario tree "
                    "manager using a model callback function")

        # check if we need to define an MPI subgroup
        if "MPIRank" in self._action_manager.server_pool[0]:
            # extract the MPI rank from the server names
            mpi_group = []
            for server_name in self._action_manager.server_pool:
                items = server_name.split('_')
                assert items[-2] == "MPIRank"
                mpi_group.append(int(items[-1]))
            server_init["mpi_group"] = mpi_group

        # transmit setup requests
        action_handles = []
        # temporarily remove this attribute so that the
        # scenario tree object can be pickled
        instance_factory = self._scenario_tree._scenario_instance_factory
        self._scenario_tree._scenario_instance_factory = None
        server_init['scenario_tree'] = self._scenario_tree
        server_init['data'] = instance_factory.data_directory()
        try:
            self.pause_transmit()
            for server_name in self._action_manager.server_pool:
                action_handles.append(
                    self._action_manager.queue(
                        queue_name=server_name,
                        action="ScenarioTreeServerPyro_setup",
                        options=server_init,
                        generate_response=True))
                self._pyro_server_workers_map[server_name] = []
            self.unpause_transmit()
        finally:
            self._scenario_tree._scenario_instance_factory = instance_factory
        self._action_manager.wait_all(action_handles)
        for ah in action_handles:
            self._action_manager.get_results(ah)

        return len(self._action_manager.server_pool)

    def release_scenariotreeservers(self, ignore_errors=False):
        """Release the pool of scenario tree servers and destroy the
        action manager."""

        assert self._action_manager is not None
        if self._options.verbose:
            print("Releasing %s scenario tree servers"
                  % (len(self._action_manager.server_pool)))

        if self._transmission_paused:
            print("Unpausing pyro transmissions in "
                  "preparation for releasing manager workers")
            self.unpause_transmit()

        self._action_manager.ignore_task_errors = ignore_errors

        self.pause_transmit()
        # copy the keys since the remove_worker function is modifying
        # the dict
        action_handles = []
        for worker_name in list(self._pyro_worker_server_map.keys()):
            action_handles.append(self.remove_worker(worker_name))
        self.unpause_transmit()
        self._action_manager.wait_all(action_handles)
        for ah in action_handles:
            self._action_manager.get_results(ah)
        del action_handles

        generate_response = None
        action_name = None
        if self._options.pyro_shutdown_workers:
            action_name = 'ScenarioTreeServerPyro_shutdown'
            generate_response = False
        else:
            action_name = 'ScenarioTreeServerPyro_reset'
            generate_response = True

        # transmit reset or shutdown requests
        action_handles = []
        self.pause_transmit()
        for server_name in self._action_manager.server_pool:
            action_handles.append(self._action_manager.queue(
                queue_name=server_name,
                action=action_name,
                generate_response=generate_response))
        self.unpause_transmit()
        if generate_response:
            self._action_manager.wait_all(action_handles)
            for ah in action_handles:
                self._action_manager.get_results(ah)
        self._action_manager.close()
        self._action_manager = None
        self._pyro_server_workers_map = {}
        self._pyro_worker_server_map = {}

    def pause_transmit(self):
        """Pause transmission of action requests. Return whether
        transmission was already paused."""
        assert self._action_manager is not None
        self._action_manager.pause()
        was_paused = self._transmission_paused
        self._transmission_paused = True
        return was_paused

    def unpause_transmit(self):
        """Unpause transmission of action requests and bulk transmit
        anything queued."""
        assert self._action_manager is not None
        self._action_manager.unpause()
        self._transmission_paused = False

    def add_worker(self,
                   worker_name,
                   worker_init,
                   worker_options,
                   worker_registered_name,
                   server_name=None,
                   oneway_call=False):

        assert self._action_manager is not None

        if server_name is None:
            # Find a server that currently owns the fewest workers
            server_name = \
                min(self._action_manager.server_pool,
                    key=lambda k: len(self._pyro_server_workers_map.get(k,[])))

        if self._options.verbose:
            print("Initializing worker with name %s on scenario tree server %s"
                  % (worker_name, server_name))

        if isinstance(worker_options, PySPConfigBlock):
            worker_class = ScenarioTreeServerPyro.\
                           get_registered_worker_type(worker_registered_name)
            try:
                worker_options = worker_class.\
                                 extract_user_options_to_dict(worker_options,
                                                              sparse=True)
            except KeyError:
                raise KeyError(
                    "Unable to serialize options for registered worker name %s "
                    "(class=%s). The worker options did not seem to match the "
                    "registered options on the worker class. Did you forget to "
                    "register them? Message: %s" % (worker_registered_name,
                                                    worker_type.__name__,
                                                    str(sys.exc_info()[1])))

        if type(worker_init) is not _WorkerInit:
            raise TypeError("worker_init argument has invalid type %s. "
                            "Must be of type %s" % (type(worker_init),
                                                    _WorkerInit))

        action_handle = self._action_manager.queue(
            queue_name=server_name,
            action="ScenarioTreeServerPyro_initialize",
            worker_type=worker_registered_name,
            worker_name=worker_name,
            init_args=(worker_init,),
            init_kwds=worker_options,
            generate_response=not oneway_call)

        self._pyro_server_workers_map[server_name].append(worker_name)
        self._pyro_worker_server_map[worker_name] = server_name
        self._pyro_worker_list.append(worker_name)

        if worker_init.type_ == "scenarios":
            self._pyro_worker_scenarios_map[worker_name] = worker_init.names
        else:
            assert worker_init.type_ == "bundles"
            self._pyro_worker_bundles_map[worker_name] = worker_init.names
            self._pyro_worker_scenarios_map[worker_name] = []
            for bundle_name in worker_init.names:
                self._pyro_worker_scenarios_map[worker_name].\
                    extend(worker_init.data[bundle_name])

        return action_handle

    def remove_worker(self, worker_name):
        assert self._action_manager is not None
        server_name = self.get_server_for_worker(worker_name)
        ah = self._action_manager.queue(
            queue_name=server_name,
            action="ScenarioTreeServerPyro_release",
            worker_name=worker_name,
            generate_response=True)
        self._pyro_server_workers_map[server_name].remove(worker_name)
        del self._pyro_worker_server_map[worker_name]
        self._pyro_worker_list.remove(worker_name)
        return ah

    def get_server_for_worker(self, worker_name):
        try:
            return self._pyro_worker_server_map[worker_name]
        except KeyError:
            raise KeyError(
                "Scenario tree worker with name %s does not exist on "
                "any scenario tree servers" % (worker_name))

#
# This class extends the initialization process of
# _ScenarioTreeManagerClientPyroAdvanced so that scenario tree servers are
# automatically acquired and assigned worker instantiations that
# manage all scenarios / bundles (thereby completing everything
# necessary to implement the ScenarioTreeManager and
# ScenarioTreeManagerClient interfaces).
#

class ScenarioTreeManagerClientPyro(_ScenarioTreeManagerClientPyroAdvanced,
                                    PySPConfiguredObject):

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()

        safe_declare_common_option(options,
                                   "pyro_required_scenariotreeservers")
        safe_declare_common_option(options,
                                   "pyro_find_scenariotreeservers_timeout")
        safe_declare_common_option(options,
                                   "pyro_handshake_at_startup")

        return options

    default_registered_worker_name = 'ScenarioTreeManagerWorkerPyro'

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
        options = super(ScenarioTreeManagerClientPyro, cls).\
                  register_options(*args, **kwds)
        if registered_worker_name is not None:
            worker_type = ScenarioTreeServerPyro.\
                          get_registered_worker_type(registered_worker_name)
            worker_type.register_options(options, **kwds)

        return options

    def __init__(self, *args, **kwds):
        self._scenario_to_worker_map = {}
        self._bundle_to_worker_map = {}
        self._registered_worker_name = \
            kwds.pop('registered_worker_name',
                     self.default_registered_worker_name)
        super(ScenarioTreeManagerClientPyro, self).__init__(*args, **kwds)

    def _request_scenario_tree_data(self):

        start_time = time.time()

        if self.get_option("verbose"):
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
                        async_call=True)

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
                        async_call=True)

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

        if self.get_option("verbose"):
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

                if self.get_option("verbose"):
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

                if self.get_option("verbose"):
                    print("Successfully loaded scenario tree data for "
                          "scenario="+scenario_name)

        self._objective_sense = \
            self._scenario_tree._scenarios[0]._objective_sense
        assert all(_s._objective_sense == self._objective_sense
                   for _s in self._scenario_tree._scenarios)

        assert all(itervalues(have_node_data))
        assert all(itervalues(have_scenario_data))

        if self.get_option("verbose"):
            print("Scenario tree instance data successfully "
                  "collected")

        if self.get_option("output_times") or \
           self.get_option("verbose"):
            print("Scenario tree data collection time=%.2f seconds"
                  % (time.time() - start_time))

    def _initialize_scenariotree_workers(self):

        start_time = time.time()

        if self._options.verbose:
            print("Transmitting scenario tree worker initializations")

        if len(self._action_manager.server_pool) == 0:
            raise RuntimeError(
                "No scenario tree server processes have been acquired!")

        if self._scenario_tree.contains_bundles():
            jobs = [_BundleWorkerInit(bundle.name,
                                      bundle.scenario_names)
                    for bundle in reversed(self._scenario_tree.bundles)]
        else:
            jobs = [_ScenarioWorkerInit(scenario.name)
                    for scenario in reversed(self._scenario_tree.scenarios)]

        assert len(self._pyro_server_workers_map) == \
            len(self._action_manager.server_pool)
        assert len(self._pyro_worker_server_map) == 0
        assert len(self._pyro_worker_list) == 0

        worker_type = ScenarioTreeServerPyro.\
                      get_registered_worker_type(self._registered_worker_name)
        worker_options = None
        try:
            worker_options = worker_type.\
                             extract_user_options_to_dict(self._options, sparse=True)
        except KeyError:
            raise KeyError(
                "Unable to extract options for registered worker name %s (class=%s). "
                "Did you forget to register the worker options into the options "
                "object passed into this class? Message: %s"
                  % (self._registered_worker_name,
                     worker_type.__name__,
                     str(sys.exc_info()[1])))

        assert worker_options is not None
        worker_initializations = dict((server_name, []) for server_name
                                      in self._action_manager.server_pool)
        # The first loop it just to get the counts
        tmp = defaultdict(int)
        cnt = 0
        for server_name in itertools.cycle(self._action_manager.server_pool):
            if len(jobs) == cnt:
                break
            tmp[server_name] += 1
            cnt += 1
        # We do this in two loops so the scenario / bundle assignment looks
        # contiguous by names listed on the scenario tree
        assert len(tmp) == len(self._action_manager.server_pool)
        for server_name in tmp:
            assert tmp[server_name] > 0
            for _i in xrange(tmp[server_name]):
                worker_initializations[server_name].append(jobs.pop())

        assert not self._transmission_paused
        if not self._options.pyro_handshake_at_startup:
            self.pause_transmit()
        action_handle_data = {}
        for cntr, server_name in enumerate(worker_initializations):

            init_type = worker_initializations[server_name][0].type_
            assert all(init_type == _worker_init.type_ for _worker_init
                       in worker_initializations[server_name])
            assert all(type(_worker_init.names) is tuple
                       for _worker_init in worker_initializations[server_name])
            assert all(len(_worker_init.names) == 1
                       for _worker_init in worker_initializations[server_name])
            worker_name = None
            if init_type == "bundles":
                worker_name = server_name+":Worker_BundleGroup"+str(cntr)
                worker_init = _BundleWorkerInit(
                    [_worker_init.names[0] for _worker_init
                     in worker_initializations[server_name]],
                    dict((_worker_init.names[0],
                          _worker_init.data[_worker_init.names[0]])
                         for _worker_init in worker_initializations[server_name]))
            else:
                assert init_type == "scenarios"
                worker_name = server_name+":Worker_ScenarioGroup"+str(cntr)
                worker_init = _ScenarioWorkerInit(
                    [_worker_init.names[0] for _worker_init
                     in worker_initializations[server_name]])

            action_handle = self.add_worker(
                worker_name,
                worker_init,
                worker_options,
                self._registered_worker_name,
                server_name=server_name)

            if self._options.pyro_handshake_at_startup:
                action_handle_data[worker_name] =  \
                    self.AsyncResult(
                        self._action_manager,
                        action_handle_data=action_handle).complete()
            else:
                action_handle_data[action_handle] = worker_name

            if worker_init.type_ == "bundles":
                for bundle_name in worker_init.names:
                    assert self._scenario_tree.contains_bundle(bundle_name)
                    self._bundle_to_worker_map[bundle_name] = worker_name
                    for scenario_name in worker_init.data[bundle_name]:
                        assert self._scenario_tree.contains_scenario(scenario_name)
                        self._scenario_to_worker_map[scenario_name] = worker_name
            else:
                assert worker_init.type_ == "scenarios"
                for scenario_name in worker_init.names:
                    assert self._scenario_tree.contains_scenario(scenario_name)
                    self._scenario_to_worker_map[scenario_name] = worker_name

        if not self._options.pyro_handshake_at_startup:
            self.unpause_transmit()

        end_time = time.time()

        if self._options.output_times or \
           self._options.verbose:
            print("Initialization transmission time=%.2f seconds"
                  % (end_time - start_time))

        if self._options.pyro_handshake_at_startup:
            return self.AsyncResult(None, result=action_handle_data)
        else:
            return self.AsyncResult(
                self._action_manager, action_handle_data=action_handle_data)

    #
    # Abstract methods for ScenarioTreeManagerClient:
    #

    # Override the implementation on _ScenarioTreeManagerClientPyroAdvanced
    def _init_client(self):
        assert self._scenario_tree is not None
        if self._scenario_tree.contains_bundles():
            for bundle in self._scenario_tree._scenario_bundles:
                self._init_bundle(bundle.name, bundle._scenario_names)
            num_jobs = len(self._scenario_tree._scenario_bundles)
            if self._options.verbose:
                print("Bundle jobs available: %s"
                      % (str(num_jobs)))
        else:
            num_jobs = len(self._scenario_tree._scenarios)
            if self._options.verbose:
                print("Scenario jobs available: %s"
                      % (str(num_jobs)))

        servers_required = self._options.pyro_required_scenariotreeservers
        if servers_required == 0:
            servers_required = num_jobs
        elif servers_required > num_jobs:
            if servers_required > num_jobs:
                print("Value assigned to pyro_required_scenariotreeservers option (%s) "
                      "is greater than the number of available jobs (%s). "
                      "Limiting the number of servers to acquire to %s"
                      % (servers_required, num_jobs, num_jobs))
            servers_required = num_jobs

        timeout = self._options.pyro_find_scenariotreeservers_timeout if \
                  (self._options.pyro_required_scenariotreeservers == 0) else \
                  None

        if self._options.verbose:
            if servers_required == 0:
                assert timeout is not None
                print("Using timeout of %s seconds to acquire up to "
                      "%s servers" % (timeout, num_jobs))
            else:
                print("Waiting to acquire exactly %s servers to distribute "
                      "work over %s jobs" % (servers_required, num_jobs))

        self.acquire_scenariotreeservers(servers_required, timeout=timeout)

        if self._options.verbose:
            print("Broadcasting requests to initialize workers "
                  "on scenario tree servers")

        initialization_handle = self._initialize_scenariotree_workers()

        worker_names = sorted(self._pyro_worker_server_map)

        # run the user script to collect aggregate scenario data. This
        # can slow down initialization as syncronization across all
        # scenario tree servers is required following serial
        # execution
        unpause = False
        if len(self._options.aggregategetter_callback_location):
            assert not self._transmission_paused
            for callback_module_key, callback_name in zip(self._aggregategetter_keys,
                                                          self._aggregategetter_names):
                if self._options.verbose:
                    print("Transmitting invocation of user defined aggregategetter "
                          "callback function defined in module: %s"
                          % (self._callback_mapped_module_name[callback_module_key]))

                result = self.invoke_function(
                    callback_name,
                    self._callback_mapped_module_name[callback_module_key],
                    invocation_type=InvocationType.PerScenarioChained,
                    function_args=(self._aggregate_user_data,))
                self._aggregate_user_data = result[0]

            # Transmit aggregate state to scenario tree servers
            if self._options.verbose:
                print("Broadcasting final aggregate data "
                      "to scenario tree servers")

            self.pause_transmit()
            unpause = True
            self.invoke_method(
                "assign_data",
                method_args=("_aggregate_user_data", self._aggregate_user_data,),
                oneway_call=True)

        # run the user script to initialize variable bounds
        if len(self._options.postinit_callback_location):
            # Note: we pause and unpause around the callback
            #       transmission block to ensure they are
            #       all sent in the same dispatcher call and
            #       their execution order on the workers is
            #       not determined by a race condition
            was_paused = self.pause_transmit()
            # we should not have already been paused unless
            # it happened a few lines above during the
            # aggregategetter execution
            assert (not was_paused) or \
                len(self._options.aggregategetter_callback_location)
            unpause = True
            for callback_module_key, callback_name in zip(self._postinit_keys,
                                                          self._postinit_names):
                if self._options.verbose:
                    print("Transmitting invocation of user defined postinit "
                          "callback function defined in module: %s"
                          % (self._callback_mapped_module_name[callback_module_key]))

                # Transmit invocation to scenario tree workers
                self.invoke_function(
                    callback_name,
                    self._callback_mapped_module_name[callback_module_key],
                    invocation_type=InvocationType.PerScenario,
                    oneway_call=True)

        if unpause:
            self.unpause_transmit()

        async_results = self._request_scenario_tree_data()

        self._initialized = False
        def _complete_init():
            self._gather_scenario_tree_data(async_results)
            self._initialized = True
            return None

        async_callback = self.AsyncResultCallback(_complete_init)
        return self.AsyncResultChain(
            results=[initialization_handle, async_callback],
            return_index=0)

    # implemented by _ScenarioTreeManagerClientPyroAdvanced
    #def _invoke_function_on_worker_impl(...)

    # implemented by _ScenarioTreeManagerClientPyroAdvanced
    #def _invoke_method_on_worker_impl(...)

    def _worker_names_impl(self):
        return self._pyro_worker_list

    def _get_worker_for_scenario_impl(self, scenario_name):
        return self._scenario_to_worker_map[scenario_name]

    def _get_worker_for_bundle_impl(self, bundle_name):
        return self._bundle_to_worker_map[bundle_name]

    #
    # Abstract Methods for ScenarioTreeManager:
    #

    # implemented by _ScenarioTreeManagerClientPyroAdvanced
    #def _close_impl(...)

    def _invoke_function_impl(
            self,
            function,
            module_name=None,
            invocation_type=InvocationType.Single,
            function_args=(),
            function_kwds=None,
            async_call=False,
            oneway_call=False):
        assert not (async_call and oneway_call)
        assert self._action_manager is not None
        start_time = time.time()

        if self._options.verbose:
            print("Transmitting external function invocation request "
                  "for function=%s in module=%s."
                  % (str(function), module_name))

        if not isinstance(function, six.string_types):
            if not dill_available:
                raise ValueError(
                    "This dill module must be available "
                    "when transmitting function objects")
            if module_name is not None:
                raise ValueError(
                    "The module_name keyword must be None "
                    "when the function argument is not a string.")
            function = dill.dumps(function)
        else:
            if module_name is None:
                raise ValueError(
                    "A module name is required when "
                    "a function name is given")

        if self._transmission_paused:
            if (not async_call) and (not oneway_call):
                raise ValueError(
                    "Unable to perform external function invocations. "
                    "Pyro transmissions are currently paused, but the "
                    "function invocation is not one-way and not asynchronous."
                    "This implies action handles be collected within "
                    "this method. Pyro transmissions must be un-paused in order "
                    "for this to take place.")

        action_handle_data = None
        map_result = None
        if (invocation_type == InvocationType.Single) or \
           (invocation_type == InvocationType.PerBundle) or \
           (invocation_type == InvocationType.PerScenario):

            was_paused = self.pause_transmit()
            action_handle_data = {}

            for worker_name in self._pyro_worker_list:
                action_handle_data[self._invoke_function_on_worker_pyro(
                    worker_name,
                    function,
                    module_name=module_name,
                    invocation_type=invocation_type,
                    function_args=function_args,
                    function_kwds=function_kwds,
                    oneway_call=oneway_call)] = worker_name

            if invocation_type != InvocationType.Single:
                map_result = lambda ah_to_result: \
                             dict((key, result[key])
                                  for result in itervalues(ah_to_result)
                                  for key in result)

            if not was_paused:
                self.unpause_transmit()

        elif (invocation_type == InvocationType.OnScenario):

            action_handle_data = self._invoke_function_on_worker_pyro(
                self.get_worker_for_scenario(invocation_type.data),
                function,
                module_name=module_name,
                invocation_type=invocation_type,
                function_args=function_args,
                function_kwds=function_kwds,
                oneway_call=oneway_call)

        elif (invocation_type == InvocationType.OnBundle):

            action_handle_data = self._invoke_function_on_worker_pyro(
                self.get_worker_for_bundle(invocation_type.data),
                function,
                module_name=module_name,
                invocation_type=invocation_type,
                function_args=function_args,
                function_kwds=function_kwds,
                oneway_call=oneway_call)

        elif (invocation_type == InvocationType.OnScenarios) or \
             (invocation_type == InvocationType.OnBundles):

            _get_worker_func = None
            _invocation_type = None
            if invocation_type == InvocationType.OnScenarios:
                _get_worker_func = self.get_worker_for_scenario
                _invocation_type = InvocationType.OnScenarios
            else:
                assert invocation_type == InvocationType.OnBundles
                _get_worker_func = self.get_worker_for_bundle
                _invocation_type = InvocationType.OnBundles

            worker_map = {}
            for object_name in invocation_type.data:
                worker_name = _get_worker_func(object_name)
                if worker_name not in worker_map:
                    worker_map[worker_name] = []
                worker_map[worker_name].append(object_name)

            was_paused = self.pause_transmit()
            action_handle_data = {}
            for worker_name in worker_map:
                action_handle_data[self._invoke_function_on_worker_pyro(
                    worker_name,
                    function,
                    module_name=module_name,
                    invocation_type=_invocation_type(worker_map[worker_name]),
                    function_args=function_args,
                    function_kwds=function_kwds,
                    oneway_call=oneway_call)] = worker_name

            map_result = lambda ah_to_result: \
                         dict((key, result[key])
                              for result in itervalues(ah_to_result)
                              for key in result)

            if not was_paused:
                self.unpause_transmit()

        elif (invocation_type == InvocationType.PerScenarioChained) or \
             (invocation_type == InvocationType.PerBundleChained):

            if self._transmission_paused:
                raise ValueError("Chained invocation type %s cannot be executed "
                                 "when Pyro transmission is paused"
                                 % (invocation_type))

            result = function_args
            for i in xrange(len(self._pyro_worker_list) - 1):
                worker_name = self._pyro_worker_list[i]
                result = self.AsyncResult(
                    self._action_manager,
                    action_handle_data=self._invoke_function_on_worker_pyro(
                        worker_name,
                        function,
                        module_name=module_name,
                        invocation_type=invocation_type,
                        function_args=result,
                        function_kwds=function_kwds,
                        oneway_call=False)).complete()
                if len(function_args) == 0:
                    result = ()

            action_handle_data = self._invoke_function_on_worker_pyro(
                self._pyro_worker_list[-1],
                function,
                module_name=module_name,
                invocation_type=invocation_type,
                function_args=result,
                function_kwds=function_kwds,
                oneway_call=oneway_call)

        elif (invocation_type == InvocationType.OnScenariosChained) or \
             (invocation_type == InvocationType.OnBundlesChained):

            if self._transmission_paused:
                raise ValueError("Chained invocation type %s cannot be executed "
                                 "when Pyro transmission is paused"
                                 % (invocation_type))

            _get_worker_func = None
            _invocation_type = None
            if invocation_type == InvocationType.OnScenariosChained:
                _get_worker_func = self.get_worker_for_scenario
                _invocation_type = InvocationType.OnScenariosChained
            else:
                assert invocation_type == InvocationType.OnBundlesChained
                _get_worker_func = self.get_worker_for_bundle
                _invocation_type = InvocationType.OnBundlesChained

            #
            # We guarantee to execute the chained call in the same
            # order as the list of names on the invocation_type, but
            # we try to be as efficient about this as possible. E.g.,
            # if the order of the chain allows for more than one piece
            # of it to be executed on the worker in a single call, we
            # take advantage of that.
            #
            assert len(invocation_type.data) > 0
            object_names = list(reversed(invocation_type.data))
            object_names_for_worker = []
            result = function_args
            while len(object_names) > 0:
                object_names_for_worker.append(object_names.pop())
                worker_name = _get_worker_func(object_names_for_worker[-1])
                if (len(object_names) == 0) or \
                   (worker_name != _get_worker_func(object_names[-1])):
                    action_handle_data=self._invoke_function_on_worker_pyro(
                        worker_name,
                        function,
                        module_name=module_name,
                        invocation_type=_invocation_type(object_names_for_worker),
                        function_args=result,
                        function_kwds=function_kwds,
                        oneway_call=False)
                    if len(object_names) != 0:
                        result = self.AsyncResult(
                            self._action_manager,
                            action_handle_data=action_handle_data).complete()
                    if len(function_args) == 0:
                        result = ()
                    object_names_for_worker = []

        else:
            raise ValueError("Unexpected function invocation type '%s'. "
                             "Expected one of %s"
                             % (invocation_type,
                                [str(v) for v in InvocationType]))

        if oneway_call:
            action_handle_data = None
            map_result = None

        result = self.AsyncResult(
            self._action_manager,
            action_handle_data=action_handle_data,
            map_result=map_result)

        if not async_call:
            result = result.complete()

        end_time = time.time()

        if self._options.output_times or \
           self._options.verbose:
            print("External function invocation request transmission "
                  "time=%.2f seconds" % (end_time - start_time))

        return result

    def _invoke_method_impl(
            self,
            method_name,
            method_args=(),
            method_kwds=None,
            async_call=False,
            oneway_call=False):
        assert not (async_call and oneway_call)
        assert self._action_manager is not None
        start_time = time.time()

        if self._options.verbose:
            print("Transmitting method invocation request "
                  "to scenario tree workers")

        if self._transmission_paused:
            if (not async_call) and (not oneway_call):
                raise ValueError(
                    "Unable to perform method invocations. "
                    "Pyro transmissions are currently paused, but the "
                    "method invocation is not one-way and not asynchronous."
                    "This implies action handles be collected within "
                    "this method. Pyro transmissions must be un-paused in order "
                    "for this to take place.")

        if method_kwds is None:
            method_kwds = {}

        was_paused = self.pause_transmit()
        action_handle_data = dict(
            (self._action_manager.queue(
                queue_name=self.get_server_for_worker(worker_name),
                worker_name=worker_name,
                action=method_name,
                generate_response=not oneway_call,
                args=method_args,
                kwds=method_kwds),
             worker_name) for worker_name in self._pyro_worker_list)
        if not was_paused:
            self.unpause_transmit()

        if oneway_call:
            action_handle_data = None

        result = self.AsyncResult(
            self._action_manager,
            action_handle_data=action_handle_data)

        if not async_call:
            result = result.complete()

        end_time = time.time()

        if self._options.output_times or \
           self._options.verbose:
            print("Method invocation request transmission "
                  "time=%.2f seconds" % (end_time - start_time))

        return result

    def _process_bundle_solve_result(self,
                                     bundle_name,
                                     results,
                                     manager_results=None,
                                     **kwds):

        if manager_results is None:
            manager_results = ScenarioTreeSolveResults('bundles')

        assert len(results) == 2
        object_results, scenario_tree_results = results

        # update the manager_results object
        for key in object_results:
            getattr(manager_results, key)[bundle_name] = \
                object_results[key]

        # Convert status strings back to enums. These are
        # transmitted as strings to avoid difficult behavior
        # related to certain Pyro serializer settings
        if manager_results.solver_status[bundle_name] is not None:
            manager_results.solver_status[bundle_name] = \
                getattr(SolverStatus,
                        str(manager_results.solver_status[bundle_name]))
        else:
            manager_results.solver_status[bundle_name] = undefined
        if manager_results.termination_condition[bundle_name] is not None:
            manager_results.termination_condition[bundle_name] = \
                getattr(TerminationCondition,
                        str(manager_results.termination_condition[bundle_name]))
        else:
            manager_results.termination_condition[bundle_name] = undefined
        if manager_results.solution_status[bundle_name] is not None:
            manager_results.solution_status[bundle_name] = \
                getattr(SolutionStatus,
                        str(manager_results.solution_status[bundle_name]))
        else:
            manager_results.solution_status[bundle_name] = undefined

        # update the scenario tree solution
        if scenario_tree_results is not None:
            bundle_scenarios = self.scenario_tree.\
                get_bundle(bundle_name).scenario_names
            assert len(bundle_scenarios) == len(scenario_tree_results)
            for scenario_name in bundle_scenarios:
                self.scenario_tree.get_scenario(scenario_name).\
                    set_solution(scenario_tree_results[scenario_name])

        return manager_results

    def _process_scenario_solve_result(self,
                                       scenario_name,
                                       results,
                                       manager_results=None,
                                       **kwds):
        if manager_results is None:
            manager_results = ScenarioTreeSolveResults('scenarios')

        assert len(results) == 2
        object_results, scenario_tree_results = results

        # update the manager_results object
        for key in object_results:
            getattr(manager_results, key)[scenario_name] = \
                object_results[key]

        # Convert status strings back to enums. These are
        # transmitted as strings to avoid difficult behavior
        # related to certain Pyro serializer settings
        if manager_results.solver_status[scenario_name] is not None:
            manager_results.solver_status[scenario_name] = \
                getattr(SolverStatus,
                        str(manager_results.solver_status[scenario_name]))
        else:
            manager_results.solver_status[scenario_name] = undefined
        if manager_results.termination_condition[scenario_name] is not None:
            manager_results.termination_condition[scenario_name] = \
                getattr(TerminationCondition,
                        str(manager_results.termination_condition[scenario_name]))
        else:
            manager_results.termination_condition[scenario_name] = undefined
        if manager_results.solution_status[scenario_name] is not None:
            manager_results.solution_status[scenario_name] = \
                getattr(SolutionStatus,
                        str(manager_results.solution_status[scenario_name]))
        else:
            manager_results.solution_status[scenario_name] = undefined

        # update the scenario tree solution
        if scenario_tree_results is not None:
            self.scenario_tree.get_scenario(scenario_name).\
                set_solution(scenario_tree_results)

        return manager_results

    def _push_fix_queue_to_instances_impl(self):

        worker_map = {}
        node_count = 0
        for stage in self.scenario_tree.stages:

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

            assert not self._transmission_paused
            self.pause_transmit()
            action_handles = []
            for worker_name in worker_map:
                action_handles.append(self._invoke_method_on_worker_pyro(
                    worker_name,
                    "_update_fixed_variables_for_client",
                    method_args=(worker_map[worker_name],),
                    oneway_call=False))
            self.unpause_transmit()
            self._action_manager.wait_all(action_handles)

        return node_count

    #
    # Extended Interface for Pyro
    #

    def get_server_for_scenario(self, scenario_name):
        return self.get_server_for_worker(
            self.get_worker_for_scenario(scenario_name))

    def get_server_for_bundle(self, bundle_name):
        return self.get_server_for_worker(
            self.get_worker_for_bundle(bundle_name))

def ScenarioTreeManagerFactory(options, *args, **kwds):
    type_ = options.scenario_tree_manager
    try:
        manager_type = ScenarioTreeManagerFactory.\
                       registered_types[type_]
    except KeyError:
        raise ValueError("Unrecognized value for option '%s': %s.\n"
                         "Must be one of: %s"
                         % ("scenario_tree_manager",
                            options.scenario_tree_manager,
                            str(sorted(ScenarioTreeManagerFactory.\
                                       registered_types.keys()))))
    return manager_type(options, *args, **kwds)

ScenarioTreeManagerFactory.registered_types = {}
ScenarioTreeManagerFactory.registered_types['serial'] = \
    ScenarioTreeManagerClientSerial
ScenarioTreeManagerFactory.registered_types['pyro'] = \
    ScenarioTreeManagerClientPyro

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
    ScenarioTreeManagerClientSerial.register_options(options,
                                                     **kwds)
    ScenarioTreeManagerClientPyro.register_options(options,
                                                   **kwds)

    return options

ScenarioTreeManagerFactory.register_options = \
    _register_scenario_tree_manager_options
