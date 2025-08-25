#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from typing import Sequence, Dict, Optional, Mapping, List, Tuple
import os

from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData
from pyomo.core.base.param import ParamData
from pyomo.core.base.block import BlockData
from pyomo.core.base.objective import Objective, ObjectiveData
from pyomo.common.config import ConfigValue, ConfigDict
from pyomo.common.enums import IntEnum, SolverAPIVersion
from pyomo.common.errors import ApplicationError
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.modeling import NOTSET
from pyomo.opt.results.results_ import SolverResults as LegacySolverResults
from pyomo.opt.results.solution import Solution as LegacySolution
from pyomo.core.kernel.objective import minimize
from pyomo.core.base import SymbolMap
from pyomo.core.base.label import NumericLabeler
from pyomo.core.staleflag import StaleFlagManager
from pyomo.scripting.solve_config import default_config_block
from pyomo.contrib.solver.common.config import SolverConfig, PersistentSolverConfig
from pyomo.contrib.solver.common.util import get_objective
from pyomo.contrib.solver.common.results import (
    Results,
    legacy_solver_status_map,
    legacy_termination_condition_map,
    legacy_solution_status_map,
)


class Availability(IntEnum):
    """
    Class to capture different statuses in which a solver can exist in
    order to record its availability for use.
    """

    FullLicense = 2
    LimitedLicense = 1
    NotFound = 0
    BadVersion = -1
    BadLicense = -2
    NeedsCompiledExtension = -3

    def __bool__(self):
        return self._value_ > 0

    def __format__(self, format_spec):
        return format(self.name, format_spec)

    def __str__(self):
        return self.name


class SolverBase:
    """The base class for "new-style" Pyomo solver interfaces.

    This base class defines the methods all derived solvers are expected
    to implement:

      - :py:meth:`available`
      - :py:meth:`is_persistent`
      - :py:meth:`solve`
      - :py:meth:`version`

    **Class Configuration**

    All derived concrete implementations of this class must define a
    class attribute ``CONFIG`` containing the :class:`ConfigDict` that
    specifies the solver's configuration options.  By convention, the
    ``CONFIG`` should derive from (or implement a superset of the
    options from) one of the following:

    - :class:`~pyomo.contrib.solver.common.config.SolverConfig`
    - :class:`~pyomo.contrib.solver.common.config.BranchAndBoundConfig`
    - :class:`~pyomo.contrib.solver.common.config.PersistentSolverConfig`
    - :class:`~pyomo.contrib.solver.common.config.PersistentBranchAndBoundConfig`

    """

    CONFIG = SolverConfig()

    def __init__(self, **kwds) -> None:
        if "name" in kwds:
            self.name = kwds.pop('name')
        elif not hasattr(self, 'name'):
            self.name = type(self).__name__.lower()

        #: Instance configuration; see CONFIG documentation on derived class
        self.config = self.CONFIG(value=kwds)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit statement - enables `with` statements."""

    @classmethod
    def api_version(self):
        """
        Return the public API supported by this interface.

        Returns
        -------
        ~pyomo.common.enums.SolverAPIVersion
            A solver API enum object
        """
        return SolverAPIVersion.V2

    def solve(self, model: BlockData, **kwargs) -> Results:
        """Solve a Pyomo model.

        Parameters
        ----------
        model: BlockData
            The Pyomo model to be solved
        **kwargs
            Additional keyword arguments (including solver_options - passthrough
            options; delivered directly to the solver (with no validation))

        Returns
        -------
        results: :class:`Results<pyomo.contrib.solver.common.results.Results>`
            A results object

        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'solve'."
        )

    def available(self) -> Availability:
        """Test if the solver is available on this system.

        Nominally, this will return `True` if the solver interface is
        valid and can be used to solve problems and `False` if it cannot.
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
        available: Availability
            An enum that indicates "how available" the solver is.
            Note that the enum can be cast to bool, which will
            be True if the solver is runable at all and False
            otherwise.
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'available'."
        )

    def version(self) -> Tuple:
        """Return the solver version found on the system.

        Returns
        -------
        version: tuple
            A tuple representing the version
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'version'."
        )

    def is_persistent(self) -> bool:
        """``True`` if this supports a persistent interface to the solver.

        Returns
        -------
        is_persistent: bool
            True if the solver is a persistent solver.
        """
        return False


class PersistentSolverBase(SolverBase):
    """
    Base class upon which persistent solvers can be built. This inherits the
    methods from the solver base class and adds those methods that are necessary
    for persistent solvers.

    Example usage can be seen in the Gurobi interface.
    """

    CONFIG = PersistentSolverConfig()

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        self._active_config = self.config

    def solve(self, model: BlockData, **kwargs) -> Results:
        """
        Solve a Pyomo model.

        Parameters
        ----------
        model: BlockData
            The Pyomo model to be solved
        **kwargs
            Additional keyword arguments (including solver_options - passthrough
            options; delivered directly to the solver (with no validation))

        Returns
        -------
        results: :class:`Results<pyomo.contrib.solver.common.results.Results>`
            A results object
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'solve'."
        )

    def is_persistent(self) -> bool:
        """
        Returns
        -------
        is_persistent: bool
            True if the solver is a persistent solver.
        """
        return True

    def _load_vars(self, vars_to_load: Optional[Sequence[VarData]] = None) -> None:
        """
        Load the solution of the primal variables into the value attribute of the variables.

        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose solution should be loaded. If vars_to_load
            is None, then the solution to all primal variables will be loaded.
        """
        for var, val in self._get_primals(vars_to_load=vars_to_load).items():
            var.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def _get_primals(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        """
        Get mapping of variables to primals.

        Parameters
        ----------
        vars_to_load : Optional[Sequence[VarData]], optional
            Which vars to be populated into the map. The default is None.

        Returns
        -------
        Mapping[VarData, float]
            A map of variables to primals.
        """
        raise NotImplementedError(
            f'{type(self)} does not support the get_primals method'
        )

    def _get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> Dict[ConstraintData, float]:
        """
        Declare sign convention in docstring here.

        Parameters
        ----------
        cons_to_load: list
            A list of the constraints whose duals should be loaded. If cons_to_load
            is None, then the duals for all constraints will be loaded.

        Returns
        -------
        duals: dict
            Maps constraints to dual values
        """
        raise NotImplementedError(f'{type(self)} does not support the get_duals method')

    def _get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        """
        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose reduced cost should be loaded. If vars_to_load
            is None, then all reduced costs will be loaded.

        Returns
        -------
        reduced_costs: ComponentMap
            Maps variable to reduced cost
        """
        raise NotImplementedError(
            f'{type(self)} does not support the get_reduced_costs method'
        )

    def set_instance(self, model: BlockData):
        """
        Set an instance of the model.
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'set_instance'."
        )

    def set_objective(self, obj: ObjectiveData):
        """
        Set current objective for the model.
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'set_objective'."
        )

    def add_variables(self, variables: List[VarData]):
        """
        Add variables to the model.
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'add_variables'."
        )

    def add_parameters(self, params: List[ParamData]):
        """
        Add parameters to the model.
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'add_parameters'."
        )

    def add_constraints(self, cons: List[ConstraintData]):
        """
        Add constraints to the model.
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'add_constraints'."
        )

    def add_block(self, block: BlockData):
        """
        Add a block to the model.
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'add_block'."
        )

    def remove_variables(self, variables: List[VarData]):
        """
        Remove variables from the model.
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'remove_variables'."
        )

    def remove_parameters(self, params: List[ParamData]):
        """
        Remove parameters from the model.
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'remove_parameters'."
        )

    def remove_constraints(self, cons: List[ConstraintData]):
        """
        Remove constraints from the model.
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'remove_constraints'."
        )

    def remove_block(self, block: BlockData):
        """
        Remove a block from the model.
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'remove_block'."
        )

    def update_variables(self, variables: List[VarData]):
        """
        Update variables on the model.
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'update_variables'."
        )

    def update_parameters(self):
        """
        Update parameters on the model.
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'update_parameters'."
        )


class LegacySolverWrapper:
    """
    Class to map the new solver interface features into the legacy solver
    interface. Necessary for backwards compatibility.
    """

    class _all_true(object):
        # A mockup of Bunch that returns True for all attribute lookups
        # or containment tests.
        def __getattr__(self, name):
            return True

        def __contains__(self, name):
            return True

    class _UpdatableConfigDict(ConfigDict):
        # The legacy solver interface used Option objects.  we will make
        # the ConfigDict *look* like an Option by supporting update()
        __slots__ = ()

        def update(self, value):
            return self.set_value(value)

    def __init__(self, **kwargs):
        solver_io = kwargs.pop('solver_io', None)
        if solver_io:
            raise NotImplementedError(f'Still working on this ({solver_io=})')
        # There is no reason for a user to be trying to mix both old
        # and new options. That is silly. So we will yell at them.
        _options = kwargs.pop('options', None)
        if 'solver_options' in kwargs:
            if _options is not None:
                raise ValueError(
                    "Both 'options' and 'solver_options' were requested. "
                    "Please use one or the other, not both."
                )
            _options = kwargs.pop('solver_options')
        if _options is not None:
            kwargs['solver_options'] = _options
        super().__init__(**kwargs)
        # Make the legacy 'options' attribute an alias of the new
        # config.solver_options; change its class so it behaves more
        # like an old Bunch/Options object.
        self.options = self.config.solver_options
        self.options.__class__ = LegacySolverWrapper._UpdatableConfigDict
        # We will assume the solver interface supports all capabilities
        # (unless a derived class actually set something
        if not hasattr(self, '_capabilities'):
            self._capabilities = LegacySolverWrapper._all_true()

    #
    # Support "with" statements
    #
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit statement - enables `with` statements."""

    def __setattr__(self, attr, value):
        # 'options' and 'config' are really singleton attributes.  Map
        # any assignment to set_value()
        if attr in ('options', 'config') and attr in self.__dict__:
            getattr(self, attr).set_value(value)
        else:
            super().__setattr__(attr, value)

    def _map_config(
        self,
        tee=NOTSET,
        load_solutions=NOTSET,
        symbolic_solver_labels=NOTSET,
        timelimit=NOTSET,
        report_timing=NOTSET,
        raise_exception_on_nonoptimal_result=NOTSET,
        solver_io=NOTSET,
        suffixes=NOTSET,
        logfile=NOTSET,
        keepfiles=NOTSET,
        solnfile=NOTSET,
        options=NOTSET,
        solver_options=NOTSET,
        writer_config=NOTSET,
    ):
        """Map between legacy and new interface configuration options"""
        if 'report_timing' not in self.config:
            self.config.declare(
                'report_timing', ConfigValue(domain=bool, default=False)
            )
        if tee is not NOTSET:
            self.config.tee = tee
        if load_solutions is not NOTSET:
            self.config.load_solutions = load_solutions
        if symbolic_solver_labels is not NOTSET:
            self.config.symbolic_solver_labels = symbolic_solver_labels
        if timelimit is not NOTSET:
            self.config.time_limit = timelimit
        if report_timing is not NOTSET:
            self.config.report_timing = report_timing
        if (options is not NOTSET) and (solver_options is not NOTSET):
            # There is no reason for a user to be trying to mix both old
            # and new options. That is silly. So we will yell at them.
            # Example that would raise an error:
            # solver.solve(model, options={'foo' : 'bar'}, solver_options={'foo' : 'not_bar'})
            raise ValueError(
                "Both 'options' and 'solver_options' were requested. "
                "Please use one or the other, not both."
            )
        if options is not NOTSET:
            # This block is trying to mimic the existing logic in the legacy
            # interface that allows users to pass initialized options to
            # the solver object and override them in the solve call.
            self.config.solver_options.set_value(options)
        elif solver_options is not NOTSET:
            self.config.solver_options.set_value(solver_options)
        if writer_config is not NOTSET:
            self.config.writer_config.set_value(writer_config)
        # This is a new flag in the interface. To preserve backwards compatibility,
        # its default is set to "False"
        if raise_exception_on_nonoptimal_result is not NOTSET:
            self.config.raise_exception_on_nonoptimal_result = (
                raise_exception_on_nonoptimal_result
            )
        if solver_io is not NOTSET and solver_io is not None:
            raise NotImplementedError('Still working on this')
        if suffixes is not NOTSET and suffixes is not None:
            raise NotImplementedError('Still working on this')
        if logfile is not NOTSET and logfile is not None:
            raise NotImplementedError('Still working on this')
        if keepfiles or 'keepfiles' in self.config:
            cwd = os.getcwd()
            deprecation_warning(
                "`keepfiles` has been deprecated in the new solver interface. "
                "Use `working_dir` instead to designate a directory in which files "
                f"should be generated and saved. Setting `working_dir` to `{cwd}`.",
                version='6.7.1',
            )
            self.config.working_dir = cwd
        # I believe this currently does nothing; however, it is unclear what
        # our desired behavior is for this.
        if solnfile is not NOTSET:
            if 'filename' in self.config:
                filename = os.path.splitext(solnfile)[0]
                self.config.filename = filename

    def _map_results(self, model, results):
        """Map between legacy and new Results objects"""
        legacy_results = LegacySolverResults()
        legacy_soln = LegacySolution()
        legacy_results.solver.status = legacy_solver_status_map[
            results.termination_condition
        ]
        legacy_results.solver.termination_condition = legacy_termination_condition_map[
            results.termination_condition
        ]
        legacy_soln.status = legacy_solution_status_map(results)
        legacy_results.solver.termination_message = str(results.termination_condition)
        legacy_results.problem.number_of_constraints = float('nan')
        legacy_results.problem.number_of_variables = float('nan')
        number_of_objectives = sum(
            1
            for _ in model.component_data_objects(
                Objective, active=True, descend_into=True
            )
        )
        legacy_results.problem.number_of_objectives = number_of_objectives
        if number_of_objectives == 1:
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
            legacy_soln.gap = abs(results.incumbent_objective - results.objective_bound)
        else:
            legacy_soln.gap = None
        return legacy_results, legacy_soln

    def _solution_handler(
        self, load_solutions, model, results, legacy_results, legacy_soln
    ):
        """Method to handle the preferred action for the solution"""
        symbol_map = legacy_soln.symbol_map = SymbolMap()
        symbol_map.default_labeler = NumericLabeler('x')
        if not hasattr(model, 'solutions'):
            # This logic gets around Issue #2130 in which
            # solutions is not an attribute on Blocks
            from pyomo.core.base.PyomoModel import ModelSolutions

            setattr(model, 'solutions', ModelSolutions(model))
        model.solutions.add_symbol_map(symbol_map)
        legacy_results._smap = symbol_map
        legacy_results._smap_id = id(symbol_map)
        delete_legacy_soln = True
        if load_solutions:
            if hasattr(model, 'dual') and model.dual.import_enabled():
                for con, val in results.solution_loader.get_duals().items():
                    model.dual[con] = val
            if hasattr(model, 'rc') and model.rc.import_enabled():
                for var, val in results.solution_loader.get_reduced_costs().items():
                    model.rc[var] = val
        elif results.incumbent_objective is not None:
            delete_legacy_soln = False
            for var, val in results.solution_loader.get_primals().items():
                legacy_soln.variable[symbol_map.getSymbol(var)] = {'Value': val}
            if hasattr(model, 'dual') and model.dual.import_enabled():
                for con, val in results.solution_loader.get_duals().items():
                    legacy_soln.constraint[symbol_map.getSymbol(con)] = {'Dual': val}
            if hasattr(model, 'rc') and model.rc.import_enabled():
                for var, val in results.solution_loader.get_reduced_costs().items():
                    legacy_soln.variable['Rc'] = val

        legacy_results.solution.insert(legacy_soln)
        # Timing info was not originally on the legacy results, but we
        # want to make it accessible to folks who are utilizing the
        # backwards compatible version.  Note that embedding the
        # ConfigDict broke pickling the legacy_results, so we will only
        # return raw nested dicts
        legacy_results.timing_info = results.timing_info.value()
        if delete_legacy_soln:
            legacy_results.solution.delete(0)
        return legacy_results

    def solve(
        self,
        model: BlockData,
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
        # These are for forward-compatibility
        raise_exception_on_nonoptimal_result: bool = False,
        solver_options: Optional[Dict] = None,
        writer_config: Optional[Dict] = None,
    ):
        """
        Solve method: maps new solve method style to backwards compatible version.

        Returns
        -------
        legacy_results
            Legacy results object

        """
        original_config = self.config

        map_args = (
            'tee',
            'load_solutions',
            'symbolic_solver_labels',
            'timelimit',
            'report_timing',
            'raise_exception_on_nonoptimal_result',
            'solver_io',
            'suffixes',
            'logfile',
            'keepfiles',
            'solnfile',
            'options',
            'solver_options',
            'writer_config',
        )
        loc = locals()
        filtered_args = {k: loc[k] for k in map_args if loc.get(k, None) is not None}
        self._map_config(**filtered_args)

        results: Results = super().solve(model)
        legacy_results, legacy_soln = self._map_results(model, results)
        legacy_results = self._solution_handler(
            load_solutions, model, results, legacy_results, legacy_soln
        )

        if self.config.report_timing:
            print(results.timing_info.timer)

        self.config = original_config

        return legacy_results

    def available(self, exception_flag=True):
        """
        Returns a bool determining whether the requested solver is available
        on the system.
        """
        ans = super().available()
        if exception_flag and not ans:
            raise ApplicationError(
                f'Solver "{self.name}" is not available. '
                f'The returned status is: {ans}.'
            )
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

    def config_block(self, init=False):
        """
        Preserves config backwards compatibility; allows new solver interfaces
        to be used in the pyomo solve call
        """
        return default_config_block(self, init)[0]

    def set_options(self, options):
        """
        Method to manually set options; can be called outside of the solve method
        """
        opts = {k: v for k, v in options.value().items() if v is not None}
        if opts:
            self._map_config(**opts)

    def warm_start_capable(self):
        return False

    def default_variable_value(self):
        return None
