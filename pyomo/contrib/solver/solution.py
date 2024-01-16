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
from typing import Sequence, Dict, Optional, Mapping, MutableMapping, NoReturn

from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.var import _GeneralVarData
from pyomo.common.collections import ComponentMap
from pyomo.core.staleflag import StaleFlagManager

# CHANGES:
# - `load` method: should just load the whole thing back into the model; load_solution = True
# - `load_variables`
# - `get_variables`
# - `get_constraints`
# - `get_objective`
# - `get_slacks`
# - `get_reduced_costs`

# duals is how much better you could get if you weren't constrained.
# dual value of 0 means that the constraint isn't actively constraining anything.
# high dual value means that it is costing us a lot in the objective.
# can also be called "shadow price"

# bounds on variables are implied constraints.
# getting a dual on the bound of a variable is the reduced cost.
# IPOPT calls these the bound multipliers (normally they are reduced costs, though). ZL, ZU

# slacks are... something that I don't understand
# but they are necessary somewhere? I guess?


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


# TODO: This is for development uses only; not to be released to the wild
# May turn into documentation someday
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
            duals = {}
            for c in cons_to_load:
                duals[c] = self._duals[c]
        return duals

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


class PersistentSolutionLoader(SolutionLoaderBase):
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
        self, cons_to_load: Optional[Sequence[_GeneralConstraintData]] = None
    ) -> Dict[_GeneralConstraintData, float]:
        self._assert_solution_still_valid()
        return self._solver._get_duals(cons_to_load=cons_to_load)

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> Mapping[_GeneralVarData, float]:
        self._assert_solution_still_valid()
        return self._solver._get_reduced_costs(vars_to_load=vars_to_load)

    def invalidate(self):
        self._valid = False
