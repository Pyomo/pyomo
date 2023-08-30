#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import abc
import enum
from datetime import datetime
from typing import Sequence, Dict, Optional, Mapping, NoReturn, List, Tuple
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.block import _BlockData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.config import ConfigDict, ConfigValue, NonNegativeInt, In, NonNegativeFloat
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.errors import ApplicationError
from pyomo.opt.base import SolverFactory as LegacySolverFactory
from pyomo.common.factory import Factory
import os
from pyomo.opt.results.results_ import SolverResults as LegacySolverResults
from pyomo.opt.results.solution import (
    Solution as LegacySolution,
    SolutionStatus as LegacySolutionStatus,
)
from pyomo.opt.results.solver import (
    TerminationCondition as LegacyTerminationCondition,
    SolverStatus as LegacySolverStatus,
)
from pyomo.core.kernel.objective import minimize
from pyomo.core.base import SymbolMap
from pyomo.core.staleflag import StaleFlagManager
from pyomo.solver.config import UpdateConfig
from pyomo.solver.solution import SolutionLoader, SolutionLoaderBase
from pyomo.solver.util import get_objective


class TerminationCondition(enum.Enum):
    """
    An enumeration for checking the termination condition of solvers
    """

    """unknown serves as both a default value, and it is used when no other enum member makes sense"""
    unknown = 42

    """The solver exited because the convergence criteria were satisfied"""
    convergenceCriteriaSatisfied = 0

    """The solver exited due to a time limit"""
    maxTimeLimit = 1

    """The solver exited due to an iteration limit"""
    iterationLimit = 2

    """The solver exited due to an objective limit"""
    objectiveLimit = 3

    """The solver exited due to a minimum step length"""
    minStepLength = 4

    """The solver exited because the problem is unbounded"""
    unbounded = 5

    """The solver exited because the problem is proven infeasible"""
    provenInfeasible = 6

    """The solver exited because the problem was found to be locally infeasible"""
    locallyInfeasible = 7

    """The solver exited because the problem is either infeasible or unbounded"""
    infeasibleOrUnbounded = 8

    """The solver exited due to an error"""
    error = 9

    """The solver exited because it was interrupted"""
    interrupted = 10

    """The solver exited due to licensing problems"""
    licensingProblems = 11


class SolutionStatus(enum.IntEnum):
    """
    An enumeration for interpreting the result of a termination. This describes the designated
    status by the solver to be loaded back into the model.

    For now, we are choosing to use IntEnum such that return values are numerically
    assigned in increasing order.
    """

    """No (single) solution found; possible that a population of solutions was returned"""
    noSolution = 0

    """Solution point does not satisfy some domains and/or constraints"""
    infeasible = 10

    """Feasible solution identified"""
    feasible = 20

    """Optimal solution identified"""
    optimal = 30


class Results(ConfigDict):
    """
    Attributes
    ----------
    termination_condition: TerminationCondition
        The reason the solver exited. This is a member of the
        TerminationCondition enum.
    incumbent_objective: float
        If a feasible solution was found, this is the objective value of
        the best solution found. If no feasible solution was found, this is
        None.
    objective_bound: float
        The best objective bound found. For minimization problems, this is
        the lower bound. For maximization problems, this is the upper bound.
        For solvers that do not provide an objective bound, this should be -inf
        (minimization) or inf (maximization)
    """

    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.declare('solution_loader', ConfigValue(default=SolutionLoader(
            None, None, None, None
        )))
        self.declare('termination_condition', ConfigValue(domain=In(TerminationCondition), default=TerminationCondition.unknown))
        self.declare('solution_status', ConfigValue(domain=In(SolutionStatus), default=SolutionStatus.noSolution))
        self.incumbent_objective: Optional[float] = self.declare('incumbent_objective', ConfigValue(domain=NonNegativeFloat))
        self.objective_bound: Optional[float] = self.declare('objective_bound', ConfigValue(domain=NonNegativeFloat))
        self.declare('solver_name', ConfigValue(domain=str))
        self.declare('solver_version', ConfigValue(domain=tuple))
        self.declare('termination_message', ConfigValue(domain=str))
        self.declare('iteration_count', ConfigValue(domain=NonNegativeInt))
        self.declare('timing_info', ConfigDict())
        self.timing_info.declare('start', ConfigValue=In(datetime))
        self.timing_info.declare('wall_time', ConfigValue(domain=NonNegativeFloat))
        self.timing_info.declare('solver_wall_time', ConfigValue(domain=NonNegativeFloat))
        self.declare('extra_info', ConfigDict(implicit=True))

    def __str__(self):
        s = ''
        s += 'termination_condition: ' + str(self.termination_condition) + '\n'
        s += 'incumbent_objective: ' + str(self.incumbent_objective) + '\n'
        s += 'objective_bound: ' + str(self.objective_bound)
        return s


class SolverBase(abc.ABC):
    class Availability(enum.IntEnum):
        NotFound = 0
        BadVersion = -1
        BadLicense = -2
        FullLicense = 1
        LimitedLicense = 2
        NeedsCompiledExtension = -3

        def __bool__(self):
            return self._value_ > 0

        def __format__(self, format_spec):
            # We want general formatting of this Enum to return the
            # formatted string value and not the int (which is the
            # default implementation from IntEnum)
            return format(self.name, format_spec)

        def __str__(self):
            # Note: Python 3.11 changed the core enums so that the
            # "mixin" type for standard enums overrides the behavior
            # specified in __format__.  We will override str() here to
            # preserve the previous behavior
            return self.name

    @abc.abstractmethod
    def solve(
        self, model: _BlockData, timer: HierarchicalTimer = None, **kwargs
    ) -> Results:
        """
        Solve a Pyomo model.

        Parameters
        ----------
        model: _BlockData
            The Pyomo model to be solved
        timer: HierarchicalTimer
            An option timer for reporting timing
        **kwargs
            Additional keyword arguments (including solver_options - passthrough options; delivered directly to the solver (with no validation))

        Returns
        -------
        results: Results
            A results object
        """
        pass

    @abc.abstractmethod
    def available(self):
        """Test if the solver is available on this system.

        Nominally, this will return True if the solver interface is
        valid and can be used to solve problems and False if it cannot.

        Note that for licensed solvers there are a number of "levels" of
        available: depending on the license, the solver may be available
        with limitations on problem size or runtime (e.g., 'demo'
        vs. 'community' vs. 'full').  In these cases, the solver may
        return a subclass of enum.IntEnum, with members that resolve to
        True if the solver is available (possibly with limitations).
        The Enum may also have multiple members that all resolve to
        False indicating the reason why the interface is not available
        (not found, bad license, unsupported version, etc).

        Returns
        -------
        available: Solver.Availability
            An enum that indicates "how available" the solver is.
            Note that the enum can be cast to bool, which will
            be True if the solver is runable at all and False
            otherwise.
        """
        pass

    @abc.abstractmethod
    def version(self) -> Tuple:
        """
        Returns
        -------
        version: tuple
            A tuple representing the version
        """

    @property
    @abc.abstractmethod
    def config(self):
        """
        An object for configuring solve options.

        Returns
        -------
        InterfaceConfig
            An object for configuring pyomo solve options such as the time limit.
            These options are mostly independent of the solver.
        """
        pass

    def is_persistent(self):
        """
        Returns
        -------
        is_persistent: bool
            True if the solver is a persistent solver.
        """
        return False


class PersistentSolverBase(SolverBase):
    def is_persistent(self):
        return True

    def load_vars(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> NoReturn:
        """
        Load the solution of the primal variables into the value attribute of the variables.

        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose solution should be loaded. If vars_to_load is None, then the solution
            to all primal variables will be loaded.
        """
        for v, val in self.get_primals(vars_to_load=vars_to_load).items():
            v.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    @abc.abstractmethod
    def get_primals(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> Mapping[_GeneralVarData, float]:
        pass

    def get_duals(
        self, cons_to_load: Optional[Sequence[_GeneralConstraintData]] = None
    ) -> Dict[_GeneralConstraintData, float]:
        """
        Declare sign convention in docstring here.

        Parameters
        ----------
        cons_to_load: list
            A list of the constraints whose duals should be loaded. If cons_to_load is None, then the duals for all
            constraints will be loaded.

        Returns
        -------
        duals: dict
            Maps constraints to dual values
        """
        raise NotImplementedError(
            '{0} does not support the get_duals method'.format(type(self))
        )

    def get_slacks(
        self, cons_to_load: Optional[Sequence[_GeneralConstraintData]] = None
    ) -> Dict[_GeneralConstraintData, float]:
        """
        Parameters
        ----------
        cons_to_load: list
            A list of the constraints whose slacks should be loaded. If cons_to_load is None, then the slacks for all
            constraints will be loaded.

        Returns
        -------
        slacks: dict
            Maps constraints to slack values
        """
        raise NotImplementedError(
            '{0} does not support the get_slacks method'.format(type(self))
        )

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> Mapping[_GeneralVarData, float]:
        """
        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose reduced cost should be loaded. If vars_to_load is None, then all reduced costs
            will be loaded.

        Returns
        -------
        reduced_costs: ComponentMap
            Maps variable to reduced cost
        """
        raise NotImplementedError(
            '{0} does not support the get_reduced_costs method'.format(type(self))
        )

    @property
    @abc.abstractmethod
    def update_config(self) -> UpdateConfig:
        pass

    @abc.abstractmethod
    def set_instance(self, model):
        pass

    @abc.abstractmethod
    def add_variables(self, variables: List[_GeneralVarData]):
        pass

    @abc.abstractmethod
    def add_params(self, params: List[_ParamData]):
        pass

    @abc.abstractmethod
    def add_constraints(self, cons: List[_GeneralConstraintData]):
        pass

    @abc.abstractmethod
    def add_block(self, block: _BlockData):
        pass

    @abc.abstractmethod
    def remove_variables(self, variables: List[_GeneralVarData]):
        pass

    @abc.abstractmethod
    def remove_params(self, params: List[_ParamData]):
        pass

    @abc.abstractmethod
    def remove_constraints(self, cons: List[_GeneralConstraintData]):
        pass

    @abc.abstractmethod
    def remove_block(self, block: _BlockData):
        pass

    @abc.abstractmethod
    def set_objective(self, obj: _GeneralObjectiveData):
        pass

    @abc.abstractmethod
    def update_variables(self, variables: List[_GeneralVarData]):
        pass

    @abc.abstractmethod
    def update_params(self):
        pass


# Everything below here preserves backwards compatibility

legacy_termination_condition_map = {
    TerminationCondition.unknown: LegacyTerminationCondition.unknown,
    TerminationCondition.maxTimeLimit: LegacyTerminationCondition.maxTimeLimit,
    TerminationCondition.iterationLimit: LegacyTerminationCondition.maxIterations,
    TerminationCondition.objectiveLimit: LegacyTerminationCondition.minFunctionValue,
    TerminationCondition.minStepLength: LegacyTerminationCondition.minStepLength,
    TerminationCondition.convergenceCriteriaSatisfied: LegacyTerminationCondition.optimal,
    TerminationCondition.unbounded: LegacyTerminationCondition.unbounded,
    TerminationCondition.provenInfeasible: LegacyTerminationCondition.infeasible,
    TerminationCondition.locallyInfeasible: LegacyTerminationCondition.infeasible,
    TerminationCondition.infeasibleOrUnbounded: LegacyTerminationCondition.infeasibleOrUnbounded,
    TerminationCondition.error: LegacyTerminationCondition.error,
    TerminationCondition.interrupted: LegacyTerminationCondition.resourceInterrupt,
    TerminationCondition.licensingProblems: LegacyTerminationCondition.licensingProblems,
}


legacy_solver_status_map = {
    TerminationCondition.unknown: LegacySolverStatus.unknown,
    TerminationCondition.maxTimeLimit: LegacySolverStatus.aborted,
    TerminationCondition.iterationLimit: LegacySolverStatus.aborted,
    TerminationCondition.objectiveLimit: LegacySolverStatus.aborted,
    TerminationCondition.minStepLength: LegacySolverStatus.error,
    TerminationCondition.convergenceCriteriaSatisfied: LegacySolverStatus.ok,
    TerminationCondition.unbounded: LegacySolverStatus.error,
    TerminationCondition.provenInfeasible: LegacySolverStatus.error,
    TerminationCondition.locallyInfeasible: LegacySolverStatus.error,
    TerminationCondition.infeasibleOrUnbounded: LegacySolverStatus.error,
    TerminationCondition.error: LegacySolverStatus.error,
    TerminationCondition.interrupted: LegacySolverStatus.aborted,
    TerminationCondition.licensingProblems: LegacySolverStatus.error,
}


legacy_solution_status_map = {
    TerminationCondition.unknown: LegacySolutionStatus.unknown,
    TerminationCondition.maxTimeLimit: LegacySolutionStatus.stoppedByLimit,
    TerminationCondition.iterationLimit: LegacySolutionStatus.stoppedByLimit,
    TerminationCondition.objectiveLimit: LegacySolutionStatus.stoppedByLimit,
    TerminationCondition.minStepLength: LegacySolutionStatus.error,
    TerminationCondition.convergenceCriteriaSatisfied: LegacySolutionStatus.optimal,
    TerminationCondition.unbounded: LegacySolutionStatus.unbounded,
    TerminationCondition.provenInfeasible: LegacySolutionStatus.infeasible,
    TerminationCondition.locallyInfeasible: LegacySolutionStatus.infeasible,
    TerminationCondition.infeasibleOrUnbounded: LegacySolutionStatus.unsure,
    TerminationCondition.error: LegacySolutionStatus.error,
    TerminationCondition.interrupted: LegacySolutionStatus.error,
    TerminationCondition.licensingProblems: LegacySolutionStatus.error,
}


class LegacySolverInterface:
    def solve(
        self,
        model: _BlockData,
        tee: bool = False,
        load_solutions: bool = True,
        logfile: Optional[str] = None,
        solnfile: Optional[str] = None,
        timelimit: Optional[float] = None,
        report_timing: bool = False,
        solver_io: Optional[str] = None,
        suffixes: Optional[Sequence] = None,
        options: Optional[Dict] = None,
        keepfiles: bool = False,
        symbolic_solver_labels: bool = False,
    ):
        original_config = self.config
        self.config = self.config()
        self.config.tee = tee
        self.config.load_solution = load_solutions
        self.config.symbolic_solver_labels = symbolic_solver_labels
        self.config.time_limit = timelimit
        self.config.report_timing = report_timing
        if solver_io is not None:
            raise NotImplementedError('Still working on this')
        if suffixes is not None:
            raise NotImplementedError('Still working on this')
        if logfile is not None:
            raise NotImplementedError('Still working on this')
        if 'keepfiles' in self.config:
            self.config.keepfiles = keepfiles
        if solnfile is not None:
            if 'filename' in self.config:
                filename = os.path.splitext(solnfile)[0]
                self.config.filename = filename
        original_options = self.options
        if options is not None:
            self.options = options

        results: Results = super().solve(model)

        legacy_results = LegacySolverResults()
        legacy_soln = LegacySolution()
        legacy_results.solver.status = legacy_solver_status_map[
            results.termination_condition
        ]
        legacy_results.solver.termination_condition = legacy_termination_condition_map[
            results.termination_condition
        ]
        legacy_soln.status = legacy_solution_status_map[results.termination_condition]
        legacy_results.solver.termination_message = str(results.termination_condition)

        obj = get_objective(model)
        legacy_results.problem.sense = obj.sense

        if obj.sense == minimize:
            legacy_results.problem.lower_bound = results.objective_bound
            legacy_results.problem.upper_bound = results.incumbent_objective
        else:
            legacy_results.problem.upper_bound = results.objective_bound
            legacy_results.problem.lower_bound = results.incumbent_objective
        if (
            results.incumbent_objective is not None
            and results.objective_bound is not None
        ):
            legacy_soln.gap = abs(
                results.incumbent_objective - results.objective_bound
            )
        else:
            legacy_soln.gap = None

        symbol_map = SymbolMap()
        symbol_map.byObject = dict(self.symbol_map.byObject)
        symbol_map.bySymbol = dict(self.symbol_map.bySymbol)
        symbol_map.aliases = dict(self.symbol_map.aliases)
        symbol_map.default_labeler = self.symbol_map.default_labeler
        model.solutions.add_symbol_map(symbol_map)
        legacy_results._smap_id = id(symbol_map)

        delete_legacy_soln = True
        if load_solutions:
            if hasattr(model, 'dual') and model.dual.import_enabled():
                for c, val in results.solution_loader.get_duals().items():
                    model.dual[c] = val
            if hasattr(model, 'slack') and model.slack.import_enabled():
                for c, val in results.solution_loader.get_slacks().items():
                    model.slack[c] = val
            if hasattr(model, 'rc') and model.rc.import_enabled():
                for v, val in results.solution_loader.get_reduced_costs().items():
                    model.rc[v] = val
        elif results.incumbent_objective is not None:
            delete_legacy_soln = False
            for v, val in results.solution_loader.get_primals().items():
                legacy_soln.variable[symbol_map.getSymbol(v)] = {'Value': val}
            if hasattr(model, 'dual') and model.dual.import_enabled():
                for c, val in results.solution_loader.get_duals().items():
                    legacy_soln.constraint[symbol_map.getSymbol(c)] = {'Dual': val}
            if hasattr(model, 'slack') and model.slack.import_enabled():
                for c, val in results.solution_loader.get_slacks().items():
                    symbol = symbol_map.getSymbol(c)
                    if symbol in legacy_soln.constraint:
                        legacy_soln.constraint[symbol]['Slack'] = val
            if hasattr(model, 'rc') and model.rc.import_enabled():
                for v, val in results.solution_loader.get_reduced_costs().items():
                    legacy_soln.variable['Rc'] = val

        legacy_results.solution.insert(legacy_soln)
        if delete_legacy_soln:
            legacy_results.solution.delete(0)

        self.config = original_config
        self.options = original_options

        return legacy_results

    def available(self, exception_flag=True):
        ans = super().available()
        if exception_flag and not ans:
            raise ApplicationError(f'Solver {self.__class__} is not available ({ans}).')
        return bool(ans)

    def license_is_valid(self) -> bool:
        """Test if the solver license is valid on this system.

        Note that this method is included for compatibility with the
        legacy SolverFactory interface.  Unlicensed or open source
        solvers will return True by definition.  Licensed solvers will
        return True if a valid license is found.

        Returns
        -------
        available: bool
            True if the solver license is valid. Otherwise, False.

        """
        return bool(self.available())

    @property
    def options(self):
        for solver_name in ['gurobi', 'ipopt', 'cplex', 'cbc', 'highs']:
            if hasattr(self, solver_name + '_options'):
                return getattr(self, solver_name + '_options')
        raise NotImplementedError('Could not find the correct options')

    @options.setter
    def options(self, val):
        found = False
        for solver_name in ['gurobi', 'ipopt', 'cplex', 'cbc', 'highs']:
            if hasattr(self, solver_name + '_options'):
                setattr(self, solver_name + '_options', val)
                found = True
        if not found:
            raise NotImplementedError('Could not find the correct options')

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass


class SolverFactoryClass(Factory):
    def register(self, name, doc=None):
        def decorator(cls):
            self._cls[name] = cls
            self._doc[name] = doc

            class LegacySolver(LegacySolverInterface, cls):
                pass

            LegacySolverFactory.register(name, doc)(LegacySolver)

            return cls

        return decorator


SolverFactory = SolverFactoryClass()
