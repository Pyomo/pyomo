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

from __future__ import annotations

from typing import Sequence, Dict, Optional, Mapping, List, Any

from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.base.suffix import Suffix
from .util import NoSolutionError
import logging


logger = logging.getLogger(__name__)


def load_import_suffixes(
    pyomo_model, solution_loader: SolutionLoaderBase, solution_id=None
):
    dual_suffix = None
    rc_suffix = None
    for suffix in pyomo_model.component_objects(Suffix, descend_into=True, active=True):
        if not suffix.import_enabled():
            continue
        if suffix.local_name == 'dual':
            dual_suffix = suffix
        elif suffix.local_name == 'rc':
            rc_suffix = suffix
    if dual_suffix is not None:
        duals = solution_loader.get_duals(solution_id=solution_id)
        if duals is NotImplemented:
            logger.warning(f'Cannot load duals into suffix')
        else:
            for k, v in duals.items():
                dual_suffix[k] = v
    if rc_suffix is not None:
        rc = solution_loader.get_reduced_costs(solution_id=solution_id)
        if rc is NotImplemented:
            logger.warning(f'cannot load duals into suffix')
        else:
            for k, v in rc.items():
                rc_suffix[k] = v


class SolutionLoaderBase:
    """
    Base class for all future SolutionLoader classes.

    Intent of this class and its children is to load the solution back into the model.
    """

    def get_solution_ids(self) -> List[Any]:
        """
        If there are multiple solutions available, this will return a
        list of the solution ids which can then be used with other
        methods like `load_soltuion`. If only one solution is
        available, this will return [None]. If no solutions
        are available, this will return None

        Returns
        -------
        solutions_ids: List[Any]
            The identifiers for multiple solutions
        """
        return NotImplemented

    def get_number_of_solutions(self) -> int:
        """
        Returns
        -------
        num_solutions: int
            Indicates the number of solutions found
        """
        return NotImplemented

    def load_solution(self, solution_id=None):
        """
        Load the solution (everything that can be) back into the model

        Parameters
        ----------
        solution_id: Optional[Any]
            If there are multiple solutions, this specifies which solution
            should be loaded. If None, the default solution will be used.
        """
        # this should load everything it can
        self.load_vars(solution_id=solution_id)
        self.load_import_suffixes(solution_id=solution_id)

    def load_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None, solution_id=None
    ) -> None:
        """
        Load the solution of the primal variables into the value attribute
        of the variables.

        Parameters
        ----------
        vars_to_load: list
            The minimum set of variables whose solution should be loaded. If
            vars_to_load is None, then the solution to all primal variables
            will be loaded. Even if vars_to_load is specified, the values of
            other variables may also be loaded depending on the interface.
        solution_id: Optional[Any]
            If there are multiple solutions, this specifies which solution
            should be loaded. If None, the default solution will be used.
        """
        for var, val in self.get_vars(
            vars_to_load=vars_to_load, solution_id=solution_id
        ).items():
            var.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None, solution_id=None
    ) -> Mapping[VarData, float]:
        """
        Returns a ComponentMap mapping variable to var value.

        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose solution value should be retrieved. If vars_to_load
            is None, then the values for all variables will be retrieved.
        solution_id: Optional[Any]
            If there are multiple solutions, this specifies which solution
            should be retrieved. If None, the default solution will be used.

        Returns
        -------
        primals: ComponentMap
            Maps variables to solution values
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'get_vars'."
        )

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None, solution_id=None
    ) -> Dict[ConstraintData, float]:
        """
        Returns a dictionary mapping constraint to dual value.

        Parameters
        ----------
        cons_to_load: list
            A list of the constraints whose duals should be retrieved. If cons_to_load
            is None, then the duals for all constraints will be retrieved.
        solution_id: Optional[Any]
            If there are multiple solutions, this specifies which solution
            should be retrieved. If None, the default solution will be used.

        Returns
        -------
        duals: dict
            Maps constraints to dual values
        """
        return NotImplemented

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None, solution_id=None
    ) -> Mapping[VarData, float]:
        """
        Returns a ComponentMap mapping variable to reduced cost.

        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose reduced cost should be retrieved. If vars_to_load
            is None, then the reduced costs for all variables will be loaded.
        solution_id: Optional[Any]
            If there are multiple solutions, this specifies which solution
            should be retrieved. If None, the default solution will be used.

        Returns
        -------
        reduced_costs: ComponentMap
            Maps variables to reduced costs
        """
        return NotImplemented

    def load_import_suffixes(self, solution_id=None):
        """
        Parameters
        ----------
        solution_id: Optional[Any]
            If there are multiple solutions, this specifies which solution
            should be loaded. If None, the default solution will be used.
        """
        return NotImplemented


class NoSolutionSolutionLoader(SolutionLoaderBase):
    def __init__(self) -> None:
        pass

    def get_solution_ids(self) -> List[Any]:
        return []

    def get_number_of_solutions(self) -> int:
        return 0

    def load_solution(self, solution_id=None):
        raise NoSolutionError()

    def load_vars(
        self, vars_to_load: Sequence[VarData] | None = None, solution_id=None
    ) -> None:
        raise NoSolutionError()

    def get_vars(
        self, vars_to_load: Sequence[VarData] | None = None, solution_id=None
    ) -> Mapping[VarData, float]:
        raise NoSolutionError()

    def get_duals(
        self, cons_to_load: Sequence[ConstraintData] | None = None, solution_id=None
    ) -> Dict[ConstraintData, float]:
        raise NoSolutionError()

    def get_reduced_costs(
        self, vars_to_load: Sequence[VarData] | None = None, solution_id=None
    ) -> Mapping[VarData, float]:
        raise NoSolutionError()

    def load_import_suffixes(self, solution_id=None):
        raise NoSolutionError()


class PersistentSolutionLoader(SolutionLoaderBase):
    """
    Loader for persistent solvers
    """

    def __init__(self, solver, pyomo_model):
        self._solver = solver
        self._valid = True
        self._pyomo_model = pyomo_model

    def _assert_solution_still_valid(self):
        if not self._valid:
            raise RuntimeError('The results in the solver are no longer valid.')

    def get_solution_ids(self) -> List[Any]:
        self._assert_solution_still_valid()
        return super().get_solution_ids()

    def get_number_of_solutions(self) -> int:
        self._assert_solution_still_valid()
        return super().get_number_of_solutions()

    def get_vars(self, vars_to_load=None, solution_id=None):
        self._assert_solution_still_valid()
        return self._solver._get_primals(
            vars_to_load=vars_to_load, solution_id=solution_id
        )

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None, solution_id=None
    ) -> Dict[ConstraintData, float]:
        self._assert_solution_still_valid()
        return self._solver._get_duals(cons_to_load=cons_to_load)

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None, solution_id=None
    ) -> Mapping[VarData, float]:
        self._assert_solution_still_valid()
        return self._solver._get_reduced_costs(vars_to_load=vars_to_load)

    def load_import_suffixes(self, solution_id=None):
        load_import_suffixes(self._pyomo_model, self, solution_id=solution_id)

    def invalidate(self):
        self._valid = False
