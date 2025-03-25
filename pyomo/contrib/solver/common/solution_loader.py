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

from typing import Sequence, Dict, Optional, Mapping, NoReturn

from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData
from pyomo.core.staleflag import StaleFlagManager


class SolutionLoaderBase:
    """
    Base class for all future SolutionLoader classes.

    Intent of this class and its children is to load the solution back into the model.
    """

    def load_vars(self, vars_to_load: Optional[Sequence[VarData]] = None) -> NoReturn:
        """
        Load the solution of the primal variables into the value attribute of the variables.

        Parameters
        ----------
        vars_to_load: list
            The minimum set of variables whose solution should be loaded. If vars_to_load
            is None, then the solution to all primal variables will be loaded. Even if
            vars_to_load is specified, the values of other variables may also be
            loaded depending on the interface.
        """
        for var, val in self.get_primals(vars_to_load=vars_to_load).items():
            var.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_primals(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        """
        Returns a ComponentMap mapping variable to var value.

        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose solution value should be retrieved. If vars_to_load
            is None, then the values for all variables will be retrieved.

        Returns
        -------
        primals: ComponentMap
            Maps variables to solution values
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} failed to implement required method 'get_primals'."
        )

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> Dict[ConstraintData, float]:
        """
        Returns a dictionary mapping constraint to dual value.

        Parameters
        ----------
        cons_to_load: list
            A list of the constraints whose duals should be retrieved. If cons_to_load
            is None, then the duals for all constraints will be retrieved.

        Returns
        -------
        duals: dict
            Maps constraints to dual values
        """
        raise NotImplementedError(f'{type(self)} does not support the get_duals method')

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        """
        Returns a ComponentMap mapping variable to reduced cost.

        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose reduced cost should be retrieved. If vars_to_load
            is None, then the reduced costs for all variables will be loaded.

        Returns
        -------
        reduced_costs: ComponentMap
            Maps variables to reduced costs
        """
        raise NotImplementedError(
            f'{type(self)} does not support the get_reduced_costs method'
        )


class PersistentSolutionLoader(SolutionLoaderBase):
    """
    Loader for persistent solvers
    """

    def __init__(self, solver):
        self._solver = solver
        self._valid = True

    def _assert_solution_still_valid(self):
        if not self._valid:
            raise RuntimeError('The results in the solver are no longer valid.')

    def get_primals(self, vars_to_load=None):
        self._assert_solution_still_valid()
        return self._solver._get_primals(vars_to_load=vars_to_load)

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> Dict[ConstraintData, float]:
        self._assert_solution_still_valid()
        return self._solver._get_duals(cons_to_load=cons_to_load)

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        self._assert_solution_still_valid()
        return self._solver._get_reduced_costs(vars_to_load=vars_to_load)

    def invalidate(self):
        self._valid = False
