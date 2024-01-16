import abc
import enum
from typing import (
    Sequence,
    Dict,
    Optional,
    Mapping,
    NoReturn,
    List,
    Tuple,
    MutableMapping,
)
from pyomo.core.base.constraint import _GeneralConstraintData, Constraint
from pyomo.core.base.sos import _SOSConstraintData, SOSConstraint
from pyomo.core.base.var import _GeneralVarData, Var
from pyomo.core.base.param import _ParamData, Param
from pyomo.core.base.block import _BlockData, Block
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.collections import ComponentMap
from .utils.get_objective import get_objective
from .utils.collect_vars_and_named_exprs import collect_vars_and_named_exprs
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.config import ConfigDict, ConfigValue, NonNegativeFloat
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
import weakref
from .cmodel import cmodel, cmodel_available
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numvalue import NumericConstant


class TerminationCondition(enum.Enum):
    """
    An enumeration for checking the termination condition of solvers
    """

    unknown = 0
    """unknown serves as both a default value, and it is used when no other enum member makes sense"""

    maxTimeLimit = 1
    """The solver exited due to a time limit"""

    maxIterations = 2
    """The solver exited due to an iteration limit """

    objectiveLimit = 3
    """The solver exited due to an objective limit"""

    minStepLength = 4
    """The solver exited due to a minimum step length"""

    optimal = 5
    """The solver exited with the optimal solution"""

    unbounded = 8
    """The solver exited because the problem is unbounded"""

    infeasible = 9
    """The solver exited because the problem is infeasible"""

    infeasibleOrUnbounded = 10
    """The solver exited because the problem is either infeasible or unbounded"""

    error = 11
    """The solver exited due to an error"""

    interrupted = 12
    """The solver exited because it was interrupted"""

    licensingProblems = 13
    """The solver exited due to licensing problems"""


class SolverConfig(ConfigDict):
    """
    Attributes
    ----------
    time_limit: float
        Time limit for the solver
    stream_solver: bool
        If True, then the solver log goes to stdout
    load_solution: bool
        If False, then the values of the primal variables will not be
        loaded into the model
    symbolic_solver_labels: bool
        If True, the names given to the solver will reflect the names
        of the pyomo components. Cannot be changed after set_instance
        is called.
    report_timing: bool
        If True, then some timing information will be printed at the
        end of the solve.
    """

    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super(SolverConfig, self).__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.declare('time_limit', ConfigValue(domain=NonNegativeFloat))
        self.declare('stream_solver', ConfigValue(domain=bool))
        self.declare('load_solution', ConfigValue(domain=bool))
        self.declare('symbolic_solver_labels', ConfigValue(domain=bool))
        self.declare('report_timing', ConfigValue(domain=bool))

        self.time_limit: Optional[float] = None
        self.stream_solver: bool = False
        self.load_solution: bool = True
        self.symbolic_solver_labels: bool = False
        self.report_timing: bool = False


class MIPSolverConfig(SolverConfig):
    """
    Attributes
    ----------
    mip_gap: float
        Solver will terminate if the mip gap is less than mip_gap
    relax_integrality: bool
        If True, all integer variables will be relaxed to continuous
        variables before solving
    """

    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super(MIPSolverConfig, self).__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.declare('mip_gap', ConfigValue(domain=NonNegativeFloat))
        self.declare('relax_integrality', ConfigValue(domain=bool))

        self.mip_gap: Optional[float] = None
        self.relax_integrality: bool = False


class SolutionLoaderBase(abc.ABC):
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
        """
        Returns a ComponentMap mapping variable to var value.

        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose solution value should be retrieved. If vars_to_load is None,
            then the values for all variables will be retrieved.

        Returns
        -------
        primals: ComponentMap
            Maps variables to solution values
        """
        pass

    def get_duals(
        self, cons_to_load: Optional[Sequence[_GeneralConstraintData]] = None
    ) -> Dict[_GeneralConstraintData, float]:
        """
        Returns a dictionary mapping constraint to dual value.

        Parameters
        ----------
        cons_to_load: list
            A list of the constraints whose duals should be retrieved. If cons_to_load is None, then the duals for all
            constraints will be retrieved.

        Returns
        -------
        duals: dict
            Maps constraints to dual values
        """
        raise NotImplementedError(f'{type(self)} does not support the get_duals method')

    def get_slacks(
        self, cons_to_load: Optional[Sequence[_GeneralConstraintData]] = None
    ) -> Dict[_GeneralConstraintData, float]:
        """
        Returns a dictionary mapping constraint to slack.

        Parameters
        ----------
        cons_to_load: list
            A list of the constraints whose duals should be loaded. If cons_to_load is None, then the duals for all
            constraints will be loaded.

        Returns
        -------
        slacks: dict
            Maps constraints to slacks
        """
        raise NotImplementedError(
            f'{type(self)} does not support the get_slacks method'
        )

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> Mapping[_GeneralVarData, float]:
        """
        Returns a ComponentMap mapping variable to reduced cost.

        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose reduced cost should be retrieved. If vars_to_load is None, then the
            reduced costs for all variables will be loaded.

        Returns
        -------
        reduced_costs: ComponentMap
            Maps variables to reduced costs
        """
        raise NotImplementedError(
            f'{type(self)} does not support the get_reduced_costs method'
        )


class SolutionLoader(SolutionLoaderBase):
    def __init__(
        self,
        primals: Optional[MutableMapping],
        duals: Optional[MutableMapping],
        slacks: Optional[MutableMapping],
        reduced_costs: Optional[MutableMapping],
    ):
        """
        Parameters
        ----------
        primals: dict
            maps id(Var) to (var, value)
        duals: dict
            maps Constraint to dual value
        slacks: dict
            maps Constraint to slack value
        reduced_costs: dict
            maps id(Var) to (var, reduced_cost)
        """
        self._primals = primals
        self._duals = duals
        self._slacks = slacks
        self._reduced_costs = reduced_costs

    def get_primals(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> Mapping[_GeneralVarData, float]:
        if self._primals is None:
            raise RuntimeError(
                'Solution loader does not currently have a valid solution. Please '
                'check the termination condition.'
            )
        if vars_to_load is None:
            return ComponentMap(self._primals.values())
        else:
            primals = ComponentMap()
            for v in vars_to_load:
                primals[v] = self._primals[id(v)][1]
            return primals

    def get_duals(
        self, cons_to_load: Optional[Sequence[_GeneralConstraintData]] = None
    ) -> Dict[_GeneralConstraintData, float]:
        if self._duals is None:
            raise RuntimeError(
                'Solution loader does not currently have valid duals. Please '
                'check the termination condition and ensure the solver returns duals '
                'for the given problem type.'
            )
        if cons_to_load is None:
            duals = dict(self._duals)
        else:
            duals = dict()
            for c in cons_to_load:
                duals[c] = self._duals[c]
        return duals

    def get_slacks(
        self, cons_to_load: Optional[Sequence[_GeneralConstraintData]] = None
    ) -> Dict[_GeneralConstraintData, float]:
        if self._slacks is None:
            raise RuntimeError(
                'Solution loader does not currently have valid slacks. Please '
                'check the termination condition and ensure the solver returns slacks '
                'for the given problem type.'
            )
        if cons_to_load is None:
            slacks = dict(self._slacks)
        else:
            slacks = dict()
            for c in cons_to_load:
                slacks[c] = self._slacks[c]
        return slacks

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> Mapping[_GeneralVarData, float]:
        if self._reduced_costs is None:
            raise RuntimeError(
                'Solution loader does not currently have valid reduced costs. Please '
                'check the termination condition and ensure the solver returns reduced '
                'costs for the given problem type.'
            )
        if vars_to_load is None:
            rc = ComponentMap(self._reduced_costs.values())
        else:
            rc = ComponentMap()
            for v in vars_to_load:
                rc[v] = self._reduced_costs[id(v)][1]
        return rc


class Results(object):
    """
    Attributes
    ----------
    termination_condition: TerminationCondition
        The reason the solver exited. This is a member of the
        TerminationCondition enum.
    best_feasible_objective: float
        If a feasible solution was found, this is the objective value of
        the best solution found. If no feasible solution was found, this is
        None.
    best_objective_bound: float
        The best objective bound found. For minimization problems, this is
        the lower bound. For maximization problems, this is the upper bound.
        For solvers that do not provide an objective bound, this should be -inf
        (minimization) or inf (maximization)

    Here is an example workflow:

        >>> import pyomo.environ as pe
        >>> from pyomo.contrib import appsi
        >>> m = pe.ConcreteModel()
        >>> m.x = pe.Var()
        >>> m.obj = pe.Objective(expr=m.x**2)
        >>> opt = appsi.solvers.Ipopt()
        >>> opt.config.load_solution = False
        >>> results = opt.solve(m) #doctest:+SKIP
        >>> if results.termination_condition == appsi.base.TerminationCondition.optimal: #doctest:+SKIP
        ...     print('optimal solution found: ', results.best_feasible_objective) #doctest:+SKIP
        ...     results.solution_loader.load_vars() #doctest:+SKIP
        ...     print('the optimal value of x is ', m.x.value) #doctest:+SKIP
        ... elif results.best_feasible_objective is not None: #doctest:+SKIP
        ...     print('sub-optimal but feasible solution found: ', results.best_feasible_objective) #doctest:+SKIP
        ...     results.solution_loader.load_vars(vars_to_load=[m.x]) #doctest:+SKIP
        ...     print('The value of x in the feasible solution is ', m.x.value) #doctest:+SKIP
        ... elif results.termination_condition in {appsi.base.TerminationCondition.maxIterations, appsi.base.TerminationCondition.maxTimeLimit}: #doctest:+SKIP
        ...     print('No feasible solution was found. The best lower bound found was ', results.best_objective_bound) #doctest:+SKIP
        ... else: #doctest:+SKIP
        ...     print('The following termination condition was encountered: ', results.termination_condition) #doctest:+SKIP
    """

    def __init__(self):
        self.solution_loader: SolutionLoaderBase = SolutionLoader(
            None, None, None, None
        )
        self.termination_condition: TerminationCondition = TerminationCondition.unknown
        self.best_feasible_objective: Optional[float] = None
        self.best_objective_bound: Optional[float] = None

    def __str__(self):
        s = ''
        s += 'termination_condition: ' + str(self.termination_condition) + '\n'
        s += 'best_feasible_objective: ' + str(self.best_feasible_objective) + '\n'
        s += 'best_objective_bound: ' + str(self.best_objective_bound)
        return s


class UpdateConfig(ConfigDict):
    """
    Attributes
    ----------
    check_for_new_or_removed_constraints: bool
    check_for_new_or_removed_vars: bool
    check_for_new_or_removed_params: bool
    update_constraints: bool
    update_vars: bool
    update_params: bool
    update_named_expressions: bool
    """

    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        if doc is None:
            doc = 'Configuration options to detect changes in model between solves'
        super(UpdateConfig, self).__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.declare(
            'check_for_new_or_removed_constraints',
            ConfigValue(
                domain=bool,
                default=True,
                doc="""
                If False, new/old constraints will not be automatically detected on subsequent
                solves. Use False only when manually updating the solver with opt.add_constraints()
                and opt.remove_constraints() or when you are certain constraints are not being
                added to/removed from the model.""",
            ),
        )
        self.declare(
            'check_for_new_or_removed_vars',
            ConfigValue(
                domain=bool,
                default=True,
                doc="""
                If False, new/old variables will not be automatically detected on subsequent 
                solves. Use False only when manually updating the solver with opt.add_variables() and 
                opt.remove_variables() or when you are certain variables are not being added to /
                removed from the model.""",
            ),
        )
        self.declare(
            'check_for_new_or_removed_params',
            ConfigValue(
                domain=bool,
                default=True,
                doc="""
                If False, new/old parameters will not be automatically detected on subsequent 
                solves. Use False only when manually updating the solver with opt.add_params() and 
                opt.remove_params() or when you are certain parameters are not being added to /
                removed from the model.""",
            ),
        )
        self.declare(
            'check_for_new_objective',
            ConfigValue(
                domain=bool,
                default=True,
                doc="""
                If False, new/old objectives will not be automatically detected on subsequent 
                solves. Use False only when manually updating the solver with opt.set_objective() or 
                when you are certain objectives are not being added to / removed from the model.""",
            ),
        )
        self.declare(
            'update_constraints',
            ConfigValue(
                domain=bool,
                default=True,
                doc="""
                If False, changes to existing constraints will not be automatically detected on 
                subsequent solves. This includes changes to the lower, body, and upper attributes of 
                constraints. Use False only when manually updating the solver with 
                opt.remove_constraints() and opt.add_constraints() or when you are certain constraints 
                are not being modified.""",
            ),
        )
        self.declare(
            'update_vars',
            ConfigValue(
                domain=bool,
                default=True,
                doc="""
                If False, changes to existing variables will not be automatically detected on 
                subsequent solves. This includes changes to the lb, ub, domain, and fixed 
                attributes of variables. Use False only when manually updating the solver with 
                opt.update_variables() or when you are certain variables are not being modified.""",
            ),
        )
        self.declare(
            'update_params',
            ConfigValue(
                domain=bool,
                default=True,
                doc="""
                If False, changes to parameter values will not be automatically detected on 
                subsequent solves. Use False only when manually updating the solver with 
                opt.update_params() or when you are certain parameters are not being modified.""",
            ),
        )
        self.declare(
            'update_named_expressions',
            ConfigValue(
                domain=bool,
                default=True,
                doc="""
                If False, changes to Expressions will not be automatically detected on 
                subsequent solves. Use False only when manually updating the solver with 
                opt.remove_constraints() and opt.add_constraints() or when you are certain 
                Expressions are not being modified.""",
            ),
        )
        self.declare(
            'update_objective',
            ConfigValue(
                domain=bool,
                default=True,
                doc="""
                If False, changes to objectives will not be automatically detected on 
                subsequent solves. This includes the expr and sense attributes of objectives. Use 
                False only when manually updating the solver with opt.set_objective() or when you are 
                certain objectives are not being modified.""",
            ),
        )
        self.declare(
            'treat_fixed_vars_as_params',
            ConfigValue(
                domain=bool,
                default=True,
                doc="""
                This is an advanced option that should only be used in special circumstances. 
                With the default setting of True, fixed variables will be treated like parameters. 
                This means that z == x*y will be linear if x or y is fixed and the constraint 
                can be written to an LP file. If the value of the fixed variable gets changed, we have 
                to completely reprocess all constraints using that variable. If 
                treat_fixed_vars_as_params is False, then constraints will be processed as if fixed 
                variables are not fixed, and the solver will be told the variable is fixed. This means 
                z == x*y could not be written to an LP file even if x and/or y is fixed. However, 
                updating the values of fixed variables is much faster this way.""",
            ),
        )

        self.check_for_new_or_removed_constraints: bool = True
        self.check_for_new_or_removed_vars: bool = True
        self.check_for_new_or_removed_params: bool = True
        self.check_for_new_objective: bool = True
        self.update_constraints: bool = True
        self.update_vars: bool = True
        self.update_params: bool = True
        self.update_named_expressions: bool = True
        self.update_objective: bool = True
        self.treat_fixed_vars_as_params: bool = True


class Solver(abc.ABC):
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
    def solve(self, model: _BlockData, timer: HierarchicalTimer = None) -> Results:
        """
        Solve a Pyomo model.

        Parameters
        ----------
        model: _BlockData
            The Pyomo model to be solved
        timer: HierarchicalTimer
            An option timer for reporting timing

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
        SolverConfig
            An object for configuring pyomo solve options such as the time limit.
            These options are mostly independent of the solver.
        """
        pass

    @property
    @abc.abstractmethod
    def symbol_map(self):
        pass

    def is_persistent(self):
        """
        Returns
        -------
        is_persistent: bool
            True if the solver is a persistent solver.
        """
        return False


class PersistentSolver(Solver):
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


class PersistentSolutionLoader(SolutionLoaderBase):
    def __init__(self, solver: PersistentSolver):
        self._solver = solver
        self._valid = True

    def _assert_solution_still_valid(self):
        if not self._valid:
            raise RuntimeError('The results in the solver are no longer valid.')

    def get_primals(self, vars_to_load=None):
        self._assert_solution_still_valid()
        return self._solver.get_primals(vars_to_load=vars_to_load)

    def get_duals(
        self, cons_to_load: Optional[Sequence[_GeneralConstraintData]] = None
    ) -> Dict[_GeneralConstraintData, float]:
        self._assert_solution_still_valid()
        return self._solver.get_duals(cons_to_load=cons_to_load)

    def get_slacks(
        self, cons_to_load: Optional[Sequence[_GeneralConstraintData]] = None
    ) -> Dict[_GeneralConstraintData, float]:
        self._assert_solution_still_valid()
        return self._solver.get_slacks(cons_to_load=cons_to_load)

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> Mapping[_GeneralVarData, float]:
        self._assert_solution_still_valid()
        return self._solver.get_reduced_costs(vars_to_load=vars_to_load)

    def invalidate(self):
        self._valid = False


"""
What can change in a pyomo model?
- variables added or removed
- constraints added or removed
- objective changed
- objective expr changed
- params added or removed
- variable modified
  - lb
  - ub
  - fixed or unfixed
  - domain
  - value
- constraint modified
  - lower
  - upper
  - body
  - active or not
- named expressions modified
  - expr
- param modified
  - value

Ideas:
- Consider explicitly handling deactivated constraints; favor deactivation over removal
  and activation over addition
  
Notes:
- variable bounds cannot be updated with mutable params; you must call update_variables
"""


class PersistentBase(abc.ABC):
    def __init__(self, only_child_vars=False):
        self._model = None
        self._active_constraints = dict()  # maps constraint to (lower, body, upper)
        self._vars = dict()  # maps var id to (var, lb, ub, fixed, domain, value)
        self._params = dict()  # maps param id to param
        self._objective = None
        self._objective_expr = None
        self._objective_sense = None
        self._named_expressions = (
            dict()
        )  # maps constraint to list of tuples (named_expr, named_expr.expr)
        self._external_functions = ComponentMap()
        self._obj_named_expressions = list()
        self._update_config = UpdateConfig()
        self._referenced_variables = (
            dict()
        )  # var_id: [dict[constraints, None], dict[sos constraints, None], None or objective]
        self._vars_referenced_by_con = dict()
        self._vars_referenced_by_obj = list()
        self._expr_types = None
        self.use_extensions = False
        self._only_child_vars = only_child_vars

    @property
    def update_config(self):
        return self._update_config

    @update_config.setter
    def update_config(self, val: UpdateConfig):
        self._update_config = val

    def set_instance(self, model):
        saved_update_config = self.update_config
        self.__init__(only_child_vars=self._only_child_vars)
        self.update_config = saved_update_config
        self._model = model
        if self.use_extensions and cmodel_available:
            self._expr_types = cmodel.PyomoExprTypes()
        self.add_block(model)
        if self._objective is None:
            self.set_objective(None)

    @abc.abstractmethod
    def _add_variables(self, variables: List[_GeneralVarData]):
        pass

    def add_variables(self, variables: List[_GeneralVarData]):
        for v in variables:
            if id(v) in self._referenced_variables:
                raise ValueError(
                    'variable {name} has already been added'.format(name=v.name)
                )
            self._referenced_variables[id(v)] = [dict(), dict(), None]
            self._vars[id(v)] = (
                v,
                v._lb,
                v._ub,
                v.fixed,
                v.domain.get_interval(),
                v.value,
            )
        self._add_variables(variables)

    @abc.abstractmethod
    def _add_params(self, params: List[_ParamData]):
        pass

    def add_params(self, params: List[_ParamData]):
        for p in params:
            self._params[id(p)] = p
        self._add_params(params)

    @abc.abstractmethod
    def _add_constraints(self, cons: List[_GeneralConstraintData]):
        pass

    def _check_for_new_vars(self, variables: List[_GeneralVarData]):
        new_vars = dict()
        for v in variables:
            v_id = id(v)
            if v_id not in self._referenced_variables:
                new_vars[v_id] = v
        self.add_variables(list(new_vars.values()))

    def _check_to_remove_vars(self, variables: List[_GeneralVarData]):
        vars_to_remove = dict()
        for v in variables:
            v_id = id(v)
            ref_cons, ref_sos, ref_obj = self._referenced_variables[v_id]
            if len(ref_cons) == 0 and len(ref_sos) == 0 and ref_obj is None:
                vars_to_remove[v_id] = v
        self.remove_variables(list(vars_to_remove.values()))

    def add_constraints(self, cons: List[_GeneralConstraintData]):
        all_fixed_vars = dict()
        for con in cons:
            if con in self._named_expressions:
                raise ValueError(
                    'constraint {name} has already been added'.format(name=con.name)
                )
            self._active_constraints[con] = (con.lower, con.body, con.upper)
            if self.use_extensions and cmodel_available:
                tmp = cmodel.prep_for_repn(con.body, self._expr_types)
            else:
                tmp = collect_vars_and_named_exprs(con.body)
            named_exprs, variables, fixed_vars, external_functions = tmp
            if not self._only_child_vars:
                self._check_for_new_vars(variables)
            self._named_expressions[con] = [(e, e.expr) for e in named_exprs]
            if len(external_functions) > 0:
                self._external_functions[con] = external_functions
            self._vars_referenced_by_con[con] = variables
            for v in variables:
                self._referenced_variables[id(v)][0][con] = None
            if not self.update_config.treat_fixed_vars_as_params:
                for v in fixed_vars:
                    v.unfix()
                    all_fixed_vars[id(v)] = v
        self._add_constraints(cons)
        for v in all_fixed_vars.values():
            v.fix()

    @abc.abstractmethod
    def _add_sos_constraints(self, cons: List[_SOSConstraintData]):
        pass

    def add_sos_constraints(self, cons: List[_SOSConstraintData]):
        for con in cons:
            if con in self._vars_referenced_by_con:
                raise ValueError(
                    'constraint {name} has already been added'.format(name=con.name)
                )
            self._active_constraints[con] = tuple()
            variables = con.get_variables()
            if not self._only_child_vars:
                self._check_for_new_vars(variables)
            self._named_expressions[con] = list()
            self._vars_referenced_by_con[con] = variables
            for v in variables:
                self._referenced_variables[id(v)][1][con] = None
        self._add_sos_constraints(cons)

    @abc.abstractmethod
    def _set_objective(self, obj: _GeneralObjectiveData):
        pass

    def set_objective(self, obj: _GeneralObjectiveData):
        if self._objective is not None:
            for v in self._vars_referenced_by_obj:
                self._referenced_variables[id(v)][2] = None
            if not self._only_child_vars:
                self._check_to_remove_vars(self._vars_referenced_by_obj)
            self._external_functions.pop(self._objective, None)
        if obj is not None:
            self._objective = obj
            self._objective_expr = obj.expr
            self._objective_sense = obj.sense
            if self.use_extensions and cmodel_available:
                tmp = cmodel.prep_for_repn(obj.expr, self._expr_types)
            else:
                tmp = collect_vars_and_named_exprs(obj.expr)
            named_exprs, variables, fixed_vars, external_functions = tmp
            if not self._only_child_vars:
                self._check_for_new_vars(variables)
            self._obj_named_expressions = [(i, i.expr) for i in named_exprs]
            if len(external_functions) > 0:
                self._external_functions[obj] = external_functions
            self._vars_referenced_by_obj = variables
            for v in variables:
                self._referenced_variables[id(v)][2] = obj
            if not self.update_config.treat_fixed_vars_as_params:
                for v in fixed_vars:
                    v.unfix()
            self._set_objective(obj)
            for v in fixed_vars:
                v.fix()
        else:
            self._vars_referenced_by_obj = list()
            self._objective = None
            self._objective_expr = None
            self._objective_sense = None
            self._obj_named_expressions = list()
            self._set_objective(obj)

    def add_block(self, block):
        param_dict = dict()
        for p in block.component_objects(Param, descend_into=True):
            if p.mutable:
                for _p in p.values():
                    param_dict[id(_p)] = _p
        self.add_params(list(param_dict.values()))
        if self._only_child_vars:
            self.add_variables(
                list(
                    dict(
                        (id(var), var)
                        for var in block.component_data_objects(Var, descend_into=True)
                    ).values()
                )
            )
        self.add_constraints(
            [
                con
                for con in block.component_data_objects(
                    Constraint, descend_into=True, active=True
                )
            ]
        )
        self.add_sos_constraints(
            [
                con
                for con in block.component_data_objects(
                    SOSConstraint, descend_into=True, active=True
                )
            ]
        )
        obj = get_objective(block)
        if obj is not None:
            self.set_objective(obj)

    @abc.abstractmethod
    def _remove_constraints(self, cons: List[_GeneralConstraintData]):
        pass

    def remove_constraints(self, cons: List[_GeneralConstraintData]):
        self._remove_constraints(cons)
        for con in cons:
            if con not in self._named_expressions:
                raise ValueError(
                    'cannot remove constraint {name} - it was not added'.format(
                        name=con.name
                    )
                )
            for v in self._vars_referenced_by_con[con]:
                self._referenced_variables[id(v)][0].pop(con)
            if not self._only_child_vars:
                self._check_to_remove_vars(self._vars_referenced_by_con[con])
            del self._active_constraints[con]
            del self._named_expressions[con]
            self._external_functions.pop(con, None)
            del self._vars_referenced_by_con[con]

    @abc.abstractmethod
    def _remove_sos_constraints(self, cons: List[_SOSConstraintData]):
        pass

    def remove_sos_constraints(self, cons: List[_SOSConstraintData]):
        self._remove_sos_constraints(cons)
        for con in cons:
            if con not in self._vars_referenced_by_con:
                raise ValueError(
                    'cannot remove constraint {name} - it was not added'.format(
                        name=con.name
                    )
                )
            for v in self._vars_referenced_by_con[con]:
                self._referenced_variables[id(v)][1].pop(con)
            self._check_to_remove_vars(self._vars_referenced_by_con[con])
            del self._active_constraints[con]
            del self._named_expressions[con]
            del self._vars_referenced_by_con[con]

    @abc.abstractmethod
    def _remove_variables(self, variables: List[_GeneralVarData]):
        pass

    def remove_variables(self, variables: List[_GeneralVarData]):
        self._remove_variables(variables)
        for v in variables:
            v_id = id(v)
            if v_id not in self._referenced_variables:
                raise ValueError(
                    'cannot remove variable {name} - it has not been added'.format(
                        name=v.name
                    )
                )
            cons_using, sos_using, obj_using = self._referenced_variables[v_id]
            if cons_using or sos_using or (obj_using is not None):
                raise ValueError(
                    'cannot remove variable {name} - it is still being used by constraints or the objective'.format(
                        name=v.name
                    )
                )
            del self._referenced_variables[v_id]
            del self._vars[v_id]

    @abc.abstractmethod
    def _remove_params(self, params: List[_ParamData]):
        pass

    def remove_params(self, params: List[_ParamData]):
        self._remove_params(params)
        for p in params:
            del self._params[id(p)]

    def remove_block(self, block):
        self.remove_constraints(
            [
                con
                for con in block.component_data_objects(
                    ctype=Constraint, descend_into=True, active=True
                )
            ]
        )
        self.remove_sos_constraints(
            [
                con
                for con in block.component_data_objects(
                    ctype=SOSConstraint, descend_into=True, active=True
                )
            ]
        )
        if self._only_child_vars:
            self.remove_variables(
                list(
                    dict(
                        (id(var), var)
                        for var in block.component_data_objects(
                            ctype=Var, descend_into=True
                        )
                    ).values()
                )
            )
        self.remove_params(
            list(
                dict(
                    (id(p), p)
                    for p in block.component_data_objects(
                        ctype=Param, descend_into=True
                    )
                ).values()
            )
        )

    @abc.abstractmethod
    def _update_variables(self, variables: List[_GeneralVarData]):
        pass

    def update_variables(self, variables: List[_GeneralVarData]):
        for v in variables:
            self._vars[id(v)] = (
                v,
                v._lb,
                v._ub,
                v.fixed,
                v.domain.get_interval(),
                v.value,
            )
        self._update_variables(variables)

    @abc.abstractmethod
    def update_params(self):
        pass

    def update(self, timer: HierarchicalTimer = None):
        if timer is None:
            timer = HierarchicalTimer()
        config = self.update_config
        new_vars = list()
        old_vars = list()
        new_params = list()
        old_params = list()
        new_cons = list()
        old_cons = list()
        old_sos = list()
        new_sos = list()
        current_vars_dict = dict()
        current_cons_dict = dict()
        current_sos_dict = dict()
        timer.start('vars')
        if self._only_child_vars and (
            config.check_for_new_or_removed_vars or config.update_vars
        ):
            current_vars_dict = {
                id(v): v
                for v in self._model.component_data_objects(Var, descend_into=True)
            }
            for v_id, v in current_vars_dict.items():
                if v_id not in self._vars:
                    new_vars.append(v)
            for v_id, v_tuple in self._vars.items():
                if v_id not in current_vars_dict:
                    old_vars.append(v_tuple[0])
        elif config.update_vars:
            start_vars = {v_id: v_tuple[0] for v_id, v_tuple in self._vars.items()}
        timer.stop('vars')
        timer.start('params')
        if config.check_for_new_or_removed_params:
            current_params_dict = dict()
            for p in self._model.component_objects(Param, descend_into=True):
                if p.mutable:
                    for _p in p.values():
                        current_params_dict[id(_p)] = _p
            for p_id, p in current_params_dict.items():
                if p_id not in self._params:
                    new_params.append(p)
            for p_id, p in self._params.items():
                if p_id not in current_params_dict:
                    old_params.append(p)
        timer.stop('params')
        timer.start('cons')
        if config.check_for_new_or_removed_constraints or config.update_constraints:
            current_cons_dict = {
                c: None
                for c in self._model.component_data_objects(
                    Constraint, descend_into=True, active=True
                )
            }
            current_sos_dict = {
                c: None
                for c in self._model.component_data_objects(
                    SOSConstraint, descend_into=True, active=True
                )
            }
            for c in current_cons_dict.keys():
                if c not in self._vars_referenced_by_con:
                    new_cons.append(c)
            for c in current_sos_dict.keys():
                if c not in self._vars_referenced_by_con:
                    new_sos.append(c)
            for c in self._vars_referenced_by_con.keys():
                if c not in current_cons_dict and c not in current_sos_dict:
                    if (c.ctype is Constraint) or (
                        c.ctype is None and isinstance(c, _GeneralConstraintData)
                    ):
                        old_cons.append(c)
                    else:
                        assert (c.ctype is SOSConstraint) or (
                            c.ctype is None and isinstance(c, _SOSConstraintData)
                        )
                        old_sos.append(c)
        self.remove_constraints(old_cons)
        self.remove_sos_constraints(old_sos)
        timer.stop('cons')
        timer.start('params')
        self.remove_params(old_params)

        # sticking this between removal and addition
        # is important so that we don't do unnecessary work
        if config.update_params:
            self.update_params()

        self.add_params(new_params)
        timer.stop('params')
        timer.start('vars')
        self.add_variables(new_vars)
        timer.stop('vars')
        timer.start('cons')
        self.add_constraints(new_cons)
        self.add_sos_constraints(new_sos)
        new_cons_set = set(new_cons)
        new_sos_set = set(new_sos)
        new_vars_set = set(id(v) for v in new_vars)
        cons_to_remove_and_add = dict()
        need_to_set_objective = False
        if config.update_constraints:
            cons_to_update = list()
            sos_to_update = list()
            for c in current_cons_dict.keys():
                if c not in new_cons_set:
                    cons_to_update.append(c)
            for c in current_sos_dict.keys():
                if c not in new_sos_set:
                    sos_to_update.append(c)
            for c in cons_to_update:
                lower, body, upper = self._active_constraints[c]
                new_lower, new_body, new_upper = c.lower, c.body, c.upper
                if new_body is not body:
                    cons_to_remove_and_add[c] = None
                    continue
                if new_lower is not lower:
                    if (
                        type(new_lower) is NumericConstant
                        and type(lower) is NumericConstant
                        and new_lower.value == lower.value
                    ):
                        pass
                    else:
                        cons_to_remove_and_add[c] = None
                        continue
                if new_upper is not upper:
                    if (
                        type(new_upper) is NumericConstant
                        and type(upper) is NumericConstant
                        and new_upper.value == upper.value
                    ):
                        pass
                    else:
                        cons_to_remove_and_add[c] = None
                        continue
            self.remove_sos_constraints(sos_to_update)
            self.add_sos_constraints(sos_to_update)
        timer.stop('cons')
        timer.start('vars')
        if self._only_child_vars and config.update_vars:
            vars_to_check = list()
            for v_id, v in current_vars_dict.items():
                if v_id not in new_vars_set:
                    vars_to_check.append(v)
        elif config.update_vars:
            end_vars = {v_id: v_tuple[0] for v_id, v_tuple in self._vars.items()}
            vars_to_check = [v for v_id, v in end_vars.items() if v_id in start_vars]
        if config.update_vars:
            vars_to_update = list()
            for v in vars_to_check:
                _v, lb, ub, fixed, domain_interval, value = self._vars[id(v)]
                if (fixed != v.fixed) or (fixed and (value != v.value)):
                    vars_to_update.append(v)
                    if self.update_config.treat_fixed_vars_as_params:
                        for c in self._referenced_variables[id(v)][0]:
                            cons_to_remove_and_add[c] = None
                        if self._referenced_variables[id(v)][2] is not None:
                            need_to_set_objective = True
                elif lb is not v._lb:
                    vars_to_update.append(v)
                elif ub is not v._ub:
                    vars_to_update.append(v)
                elif domain_interval != v.domain.get_interval():
                    vars_to_update.append(v)
            self.update_variables(vars_to_update)
        timer.stop('vars')
        timer.start('cons')
        cons_to_remove_and_add = list(cons_to_remove_and_add.keys())
        self.remove_constraints(cons_to_remove_and_add)
        self.add_constraints(cons_to_remove_and_add)
        timer.stop('cons')
        timer.start('named expressions')
        if config.update_named_expressions:
            cons_to_update = list()
            for c, expr_list in self._named_expressions.items():
                if c in new_cons_set:
                    continue
                for named_expr, old_expr in expr_list:
                    if named_expr.expr is not old_expr:
                        cons_to_update.append(c)
                        break
            self.remove_constraints(cons_to_update)
            self.add_constraints(cons_to_update)
            for named_expr, old_expr in self._obj_named_expressions:
                if named_expr.expr is not old_expr:
                    need_to_set_objective = True
                    break
        timer.stop('named expressions')
        timer.start('objective')
        if self.update_config.check_for_new_objective:
            pyomo_obj = get_objective(self._model)
            if pyomo_obj is not self._objective:
                need_to_set_objective = True
        else:
            pyomo_obj = self._objective
        if self.update_config.update_objective:
            if pyomo_obj is not None and pyomo_obj.expr is not self._objective_expr:
                need_to_set_objective = True
            elif pyomo_obj is not None and pyomo_obj.sense is not self._objective_sense:
                # we can definitely do something faster here than resetting the whole objective
                need_to_set_objective = True
        if need_to_set_objective:
            self.set_objective(pyomo_obj)
        timer.stop('objective')

        # this has to be done after the objective and constraints in case the
        # old objective/constraints use old variables
        timer.start('vars')
        self.remove_variables(old_vars)
        timer.stop('vars')


legacy_termination_condition_map = {
    TerminationCondition.unknown: LegacyTerminationCondition.unknown,
    TerminationCondition.maxTimeLimit: LegacyTerminationCondition.maxTimeLimit,
    TerminationCondition.maxIterations: LegacyTerminationCondition.maxIterations,
    TerminationCondition.objectiveLimit: LegacyTerminationCondition.minFunctionValue,
    TerminationCondition.minStepLength: LegacyTerminationCondition.minStepLength,
    TerminationCondition.optimal: LegacyTerminationCondition.optimal,
    TerminationCondition.unbounded: LegacyTerminationCondition.unbounded,
    TerminationCondition.infeasible: LegacyTerminationCondition.infeasible,
    TerminationCondition.infeasibleOrUnbounded: LegacyTerminationCondition.infeasibleOrUnbounded,
    TerminationCondition.error: LegacyTerminationCondition.error,
    TerminationCondition.interrupted: LegacyTerminationCondition.resourceInterrupt,
    TerminationCondition.licensingProblems: LegacyTerminationCondition.licensingProblems,
}


legacy_solver_status_map = {
    TerminationCondition.unknown: LegacySolverStatus.unknown,
    TerminationCondition.maxTimeLimit: LegacySolverStatus.aborted,
    TerminationCondition.maxIterations: LegacySolverStatus.aborted,
    TerminationCondition.objectiveLimit: LegacySolverStatus.aborted,
    TerminationCondition.minStepLength: LegacySolverStatus.error,
    TerminationCondition.optimal: LegacySolverStatus.ok,
    TerminationCondition.unbounded: LegacySolverStatus.error,
    TerminationCondition.infeasible: LegacySolverStatus.error,
    TerminationCondition.infeasibleOrUnbounded: LegacySolverStatus.error,
    TerminationCondition.error: LegacySolverStatus.error,
    TerminationCondition.interrupted: LegacySolverStatus.aborted,
    TerminationCondition.licensingProblems: LegacySolverStatus.error,
}


legacy_solution_status_map = {
    TerminationCondition.unknown: LegacySolutionStatus.unknown,
    TerminationCondition.maxTimeLimit: LegacySolutionStatus.stoppedByLimit,
    TerminationCondition.maxIterations: LegacySolutionStatus.stoppedByLimit,
    TerminationCondition.objectiveLimit: LegacySolutionStatus.stoppedByLimit,
    TerminationCondition.minStepLength: LegacySolutionStatus.error,
    TerminationCondition.optimal: LegacySolutionStatus.optimal,
    TerminationCondition.unbounded: LegacySolutionStatus.unbounded,
    TerminationCondition.infeasible: LegacySolutionStatus.infeasible,
    TerminationCondition.infeasibleOrUnbounded: LegacySolutionStatus.unsure,
    TerminationCondition.error: LegacySolutionStatus.error,
    TerminationCondition.interrupted: LegacySolutionStatus.error,
    TerminationCondition.licensingProblems: LegacySolutionStatus.error,
}


class LegacySolverInterface(object):
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
        self.config.stream_solver = tee
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

        results: Results = super(LegacySolverInterface, self).solve(model)

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
            legacy_results.problem.lower_bound = results.best_objective_bound
            legacy_results.problem.upper_bound = results.best_feasible_objective
        else:
            legacy_results.problem.upper_bound = results.best_objective_bound
            legacy_results.problem.lower_bound = results.best_feasible_objective
        if (
            results.best_feasible_objective is not None
            and results.best_objective_bound is not None
        ):
            legacy_soln.gap = abs(
                results.best_feasible_objective - results.best_objective_bound
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
        elif results.best_feasible_objective is not None:
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
        ans = super(LegacySolverInterface, self).available()
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
