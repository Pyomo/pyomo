# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence, Mapping, Any

from pyomo.common.docutils import copy_docstrings
from pyomo.common.flags import NOTSET
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.enums import TraversalStrategy
from pyomo.core.base.var import VarData
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.base.suffix import Suffix


class SolutionLoader:
    """Base class for all Solution Loader classes.

    The intent of this class and its children is to facilitate the
    retrieval of solver results in the context of the Pyomo model,
    either as independent data structures or by loading the data back
    into the original Pyomo model.

    """

    def solution(self, solution_id: Any) -> "SolutionLoaderView":
        """Return a view object that can be used to access a specific solution

        The resulting :class:`SolutionLoaderView` object can be used in
        two ways.  First, as a context manager:

        .. code::

           results = solver.solve(model)
           with results.solution(2) as soln:
               soln.load_vars()
               soln.load_import_suffixes()

        or

        .. code::

           results = solver.solve(model)
           with results.solution(2):
               results.load_vars()
               results.load_import_suffixes()

        Or as if it were a :class:`SolutionLoader`:

        .. code:

           results = solver.solve(model)
           results.solution(2).load_vars()
           results.solution(2).load_import_suffixes()

        Parameters
        ----------
        solution_id : Any
            The solution identifier to "activate" and make available

        """
        return SolutionLoaderView(self, solution_id)

    def _set_solution_id(self, solution_id: Any) -> Any:
        """Activate a solution_id and return the previously active solution_id

        Parameters
        ----------
        solution_id : Any
            The `solution_id` to activate
        """
        # The default implementation assumes the loader only supports a
        # single result, and the result ID is `None`
        if solution_id is not None:
            raise ValueError(
                f"{self.__class__.__name__} does not support multiple solutions"
            )
        return None

    def get_solution_ids(self) -> list[Any]:
        """Return the list of available solution identdiers.

        If there are multiple solutions available, this will return a
        list of the solution identifiers that can be passed to
        :meth:`solution` to activate individual solutions from the
        solver's solution pool. If only one solution is available, this
        will return ``[None]``. If no solutions are available, this will
        return ``[]``

        Returns
        -------
        solutions_ids: list[Any]
            The identifiers for multiple solutions

        """
        # The default implementation assumes the loader only supports a
        # single result, and the result ID is `None`
        if self.get_number_of_solutions():
            return [None]
        return []

    def get_number_of_solutions(self) -> int:
        """The number of solutions available through this :class:`SolutionLoader`

        Returns
        -------
        num_solutions: int
            Indicates the number of solutions found

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} class failed to implement "
            "required method 'get_number_of_solutions'."
        )

    def load_solution(self) -> None:
        """Load the solution (everything that can be) back into the model"""
        # this should load everything it can
        self.load_vars()
        self.load_import_suffixes()

    def load_vars(self, vars_to_load: Sequence[VarData] | None = None) -> None:
        """Load the primal variable values at the solution into the Pyomo model
        :class:`Var` objects

        Parameters
        ----------
        vars_to_load: Sequence[VarData]
            A list of the minimum set of Pyomo variables whose solution
            should be loaded.  If `vars_to_load` is ``None``, then the
            solution to all primal variables will be loaded. Even if
            `vars_to_load` is specified, the values of other variables
            may also be loaded depending on the interface.

        """
        for var, val in self.get_vars(vars_to_load=vars_to_load).items():
            var.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_vars(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> Mapping[VarData, float]:
        """Returns a ComponentMap mapping variable to var value.

        Parameters
        ----------
        vars_to_load: Sequence[VarData]
            A list of the Pyomo variables whose solution value should be
            retrieved. If `vars_to_load` is ``None``, then the values
            for all variables will be retrieved.

        Returns
        -------
        primals: ComponentMap[VarData, float]
            Maps variables to solution values

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} class failed to implement "
            "required method 'get_vars'."
        )

    def get_duals(
        self, cons_to_load: Sequence[ConstraintData] | None = None
    ) -> dict[ConstraintData, float]:
        """Returns a dictionary mapping constraint to dual value.

        Parameters
        ----------
        cons_to_load: Sequence[ConstraintData]
            A list of the constraints whose duals should be
            retrieved. If `cons_to_load` is ``None``, then the duals for all
            constraints will be retrieved.

        Returns
        -------
        duals: dict[ConstraintData, float]
            Maps constraints to dual values

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} class failed to implement "
            "required method 'get_duals'."
        )

    def get_reduced_costs(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> Mapping[VarData, float]:
        """Returns a ComponentMap mapping variable to reduced cost.

        Parameters
        ----------
        vars_to_load: Sequence[VarData]
            A list of the variables whose reduced cost should be
            retrieved. If `vars_to_load` is ``None``, then the reduced
            costs for all variables will be retrieved.

        Returns
        -------
        reduced_costs: ComponentMap[VarData, float]
            Maps variables to reduced costs

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} class failed to implement "
            "required method 'get_reduced_costs'."
        )

    def load_import_suffixes(self) -> None:
        """Clear import suffixes on the model and load data returned by the solver."""
        suffixes = self._collect_and_clear_import_suffixes()
        if 'dual' in suffixes:
            suffixes['dual'].update(self.get_duals())
        if 'rc' in suffixes:
            suffixes['rc'].update(self.get_reduced_costs())

    def _collect_and_clear_import_suffixes(self) -> dict[str, Suffix]:
        """Clear and return all import suffixes on the model.

        This walks the Pyomo model and clears all :class:`Suffix`
        components that are flagged to import values from the solver
        (this includes :attr:`Suffix.IMPORT` and
        :attr:`Suffix.IMPORT_EXPORT`).  It returns a :class:`dict`
        mapping the :attr:`Suffix.local_name` to the :class:`Suffix`
        closest to the root block.

        Returns
        -------
        import_suffixes : dict[str, Suffix]

        """
        import_suffixes = {}
        for suffix in self._pyomo_model.component_objects(
            Suffix,
            active=True,
            descend_into=True,
            descent_order=TraversalStrategy.BreadthFirstSearch,
        ):
            if not suffix.import_enabled():
                continue
            suffix.clear()
            import_suffixes.setdefault(suffix.local_name, suffix)
        return import_suffixes


@copy_docstrings(SolutionLoader)
class SolutionLoaderView:
    """A view onto a specific `solution_id` from a :class:`SolutionLoader`

    This implements :class:`SolutionLoader` API for accessing a
    specific `solution_id` from a :class:`SolutionLoader` instance.
    You can use instances of this class in two ways:

    As a :class:`SolutionLoader` object:
        Accessing the public methods on this view will activate the
        corresponding `solution_id` and return the result from the
        underlying loader object.

    As a context manager:
        If you use this object as a context manager, then the
        `solution_id` is activated upon entry and deactivated upon exit.
        Within the context, you can access either the
        :class:`SolutionLoader` API methods on this context manager,
        or on the underlying loader object to query or access the
        result.

    Parameters
    ----------
    loader : SolutionLoader
        The underlying loader object that this is a view into

    solution_id : Any
        The solution identifier to activate before accessing results.
    """

    def __init__(self, loader: SolutionLoader, solution_id: Any):
        self._loader: SolutionLoader = loader
        self._solution_id: Any = solution_id
        self._previous_id: Any = NOTSET

    def __enter__(self):
        self._previous_id = self._loader._set_solution_id(self._solution_id)
        return self._loader

    def __exit__(self, et, ev, tb):
        assert self._loader._set_solution_id(self._previous_id) == self._solution_id
        self._previous_id = NOTSET

    def get_solution_ids(self) -> list[Any]:
        return self._loader.get_solution_ids()

    def get_number_of_solutions(self) -> int:
        return self._loader.get_number_of_solutions()

    def load_solution(self):
        with self if self._previous_id is NOTSET else nullcontext:
            return self._loader.load_solution()

    def load_vars(self, vars_to_load: Sequence[VarData] | None = None) -> None:
        with self if self._previous_id is NOTSET else nullcontext:
            return self._loader.load_vars(vars_to_load)

    def get_vars(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> Mapping[VarData, float]:
        with self if self._previous_id is NOTSET else nullcontext:
            return self._loader.get_vars(vars_to_load)

    def get_duals(
        self, cons_to_load: Sequence[ConstraintData] | None = None
    ) -> dict[ConstraintData, float]:
        with self if self._previous_id is NOTSET else nullcontext:
            return self._loader.get_duals(cons_to_load)

    def get_reduced_costs(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> Mapping[VarData, float]:
        with self if self._previous_id is NOTSET else nullcontext:
            return self._loader.get_reduced_costs(vars_to_load)

    def load_import_suffixes(self):
        with self if self._previous_id is NOTSET else nullcontext:
            return self._loader.load_import_suffixes()


class PersistentSolutionLoader(SolutionLoader):
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

    def get_solution_ids(self) -> list[Any]:
        self._assert_solution_still_valid()
        return super().get_solution_ids()

    def get_number_of_solutions(self) -> int:
        self._assert_solution_still_valid()
        return super().get_number_of_solutions()

    def get_vars(self, vars_to_load=None):
        self._assert_solution_still_valid()
        return self._solver._get_primals(vars_to_load=vars_to_load)

    def get_duals(
        self, cons_to_load: Sequence[ConstraintData] | None = None
    ) -> dict[ConstraintData, float]:
        self._assert_solution_still_valid()
        return self._solver._get_duals(cons_to_load=cons_to_load)

    def get_reduced_costs(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> Mapping[VarData, float]:
        self._assert_solution_still_valid()
        return self._solver._get_reduced_costs(vars_to_load=vars_to_load)

    def invalidate(self):
        self._valid = False
