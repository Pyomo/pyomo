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

import sys
import heapq
import collections
import dataclasses
import json
import functools

import pyomo.environ as pyo

from .aos_utils import MyMunch, to_dict

nan = float("nan")


def _custom_dict_factory(data):
    return {k: to_dict(v) for k, v in data}


if sys.version_info >= (3, 10):
    dataclass_kwargs = dict(kw_only=True)
else:
    dataclass_kwargs = dict()


@dataclasses.dataclass(**dataclass_kwargs)
class VariableInfo:
    """
    Represents a variable in a solution.

    Attributes
    ----------
    value : float
        The value of the variable.
    fixed : bool
        If True, then the variable was fixed during optimization.
    name : str
        The name of the variable.
    index : int
        The unique identifier for this variable.
    discrete : bool
        If True, then this is a discrete variable
    suffix : dict
        Other information about this variable.
    """

    value: float = nan
    fixed: bool = False
    name: str = None
    repn = None
    index: int = None
    discrete: bool = False
    suffix: MyMunch = dataclasses.field(default_factory=MyMunch)

    def to_dict(self):
        return dataclasses.asdict(self, dict_factory=_custom_dict_factory)


@dataclasses.dataclass(**dataclass_kwargs)
class ObjectiveInfo:
    """
    Represents an objective in a solution.

    Attributes
    ----------
    value : float
        The objective value.
    name : str
        The name of the objective.
    index : int
        The unique identifier for this objective.
    suffix : dict
        Other information about this objective.
    """

    value: float = nan
    name: str = None
    index: int = None
    suffix: MyMunch = dataclasses.field(default_factory=MyMunch)

    def to_dict(self):
        return dataclasses.asdict(self, dict_factory=_custom_dict_factory)


@functools.total_ordering
class Solution:
    """
    An object that describes an optimization solution.

    Parameters
    -----------
    variables : None or list
        A list of :py:class:`VariableInfo` objects. (default is None)
    objective : None or :py:class:`ObjectiveInfo`
        A :py:class:`ObjectiveInfo` object. (default is None)
    objectives : None or list
        A list of :py:class:`ObjectiveInfo` objects. (default is None)
    kwargs : dict
        A dictionary of auxiliary data that is stored with the core solution values.  If the 'suffix'
        keyword is specified, then its value is use to define suffix data.  Otherwise, all
        of the keyword arguments are treated as suffix data.
    """

    def __init__(self, *, variables=None, objective=None, objectives=None, **kwargs):
        self.id = None

        self._variables = []
        self.name_to_variable = {}
        self.fixed_variable_names = set()
        if variables is not None:
            self._variables = variables
            for v in variables:
                if v.name is not None:
                    if v.fixed:
                        self.fixed_variable_names.add(v.name)
                    self.name_to_variable[v.name] = v

        self._objectives = []
        self.name_to_objective = {}
        if objective is not None:
            objectives = [objective]
        if objectives is not None:
            self._objectives = objectives
            for o in objectives:
                if o.name is not None:
                    self.name_to_objective[o.name] = o

        if "suffix" in kwargs:
            self.suffix = MyMunch(kwargs.pop("suffix"))
        else:
            self.suffix = MyMunch(**kwargs)

    def variable(self, index):
        """Returns the specified variable.

        Parameters
        ----------
        index : int or str
            The index or name of the objective. (default is 0)

        Returns
        -------
        VariableInfo
        """
        if type(index) is int:
            return self._variables[index]
        else:
            return self.name_to_variable[index]

    def variables(self):
        """
        Returns
        -------
        list
            The list of variables in the solution.
        """
        return self._variables

    def objective(self, index=0):
        """Returns the specified objective.

        Parameters
        ----------
        index : int or str
            The index or name of the objective. (default is 0)

        Returns
        -------
        :py:class:`ObjectiveInfo`
        """
        if type(index) is int:
            return self._objectives[index]
        else:
            return self.name_to_objective[index]

    def objectives(self):
        """
        Returns
        -------
        list
            The list of objectives in the solution.
        """
        return self._objectives

    def to_dict(self):
        """
        Returns
        -------
        dict
            A dictionary representation of the solution.
        """
        return dict(
            id=self.id,
            variables=[v.to_dict() for v in self.variables()],
            objectives=[o.to_dict() for o in self.objectives()],
            suffix=self.suffix.to_dict(),
        )

    def to_string(self, sort_keys=True, indent=4):
        """
        Returns a string representation of the solution, which is generated
        from a dictionary representation of the solution.

        Parameters
        ----------
        sort_keys : bool
            If True, then sort the keys in the dictionary representation. (default is True)
        indent : int
            Specifies the number of whitespaces to indent each element of the dictionary.

        Returns
        -------
        str
            A string representation of the solution.
        """
        return json.dumps(self.to_dict(), sort_keys=sort_keys, indent=indent)

    def __str__(self):
        return self.to_string()

    __repn__ = __str__

    def _tuple_repn(self):
        """
        Generate a tuple that represents the variables in the model.

        We use string names if possible, because they more explicit than the integer index values.
        """
        if len(self.name_to_variable) == len(self._variables):
            return tuple(
                tuple([k, var.value]) for k, var in self.name_to_variable.items()
            )
        else:
            return tuple(tuple([k, var.value]) for k, var in enumerate(self._variables))

    def __eq__(self, soln):
        if not isinstance(soln, Solution):
            return NotImplemented
        return self._tuple_repn() == soln._tuple_repn()

    def __lt__(self, soln):
        if not isinstance(soln, Solution):
            return NotImplemented
        return self._tuple_repn() <= soln._tuple_repn()


class PyomoSolution(Solution):

    def __init__(self, *, variables=None, objective=None, objectives=None, **kwargs):
        #
        # Q: Do we want to use an index relative to the list of variables specified here?  Or use the Pyomo variable ID?
        # Q: Should this object cache the Pyomo variable object?  Or CUID?
        #
        # TODO: Capture suffix info here.
        #
        vlist = []
        if variables is not None:
            index = 0
            for var in variables:
                vlist.append(
                    VariableInfo(
                        value=(
                            pyo.value(var) if var.is_continuous() else round(pyo.value(var))
                        ),
                        fixed=var.is_fixed(),
                        name=str(var),
                        index=index,
                        discrete=not var.is_continuous(),
                    )
                )
                index += 1

        #
        # TODO: Capture suffix info here.
        #
        if objective is not None:
            objectives = [objective]
        olist = []
        if objectives is not None:
            index = 0
            for obj in objectives:
                olist.append(
                    ObjectiveInfo(value=pyo.value(obj), name=str(obj), index=index)
                )
                index += 1

        super().__init__(variables=vlist, objectives=olist, **kwargs)
