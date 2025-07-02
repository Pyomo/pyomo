import heapq
import collections
import dataclasses
import json
import munch

import pyomo.environ as pyo

from .aos_utils import MyMunch, _to_dict

nan = float("nan")


def _custom_dict_factory(data):
    return {k: _to_dict(v) for k, v in data}


@dataclasses.dataclass
class Variable:
    _: dataclasses.KW_ONLY
    value: float = nan
    fixed: bool = False
    name: str = None
    repn = None
    index: int = None
    discrete: bool = False
    suffix: MyMunch = dataclasses.field(default_factory=MyMunch)

    def to_dict(self):
        return dataclasses.asdict(self, dict_factory=_custom_dict_factory)


@dataclasses.dataclass
class Objective:
    _: dataclasses.KW_ONLY
    value: float = nan
    name: str = None
    index: int = None
    suffix: MyMunch = dataclasses.field(default_factory=MyMunch)

    def to_dict(self):
        return dataclasses.asdict(self, dict_factory=_custom_dict_factory)


class Solution:

    def __init__(self, *, variables=None, objective=None, objectives=None, **kwds):
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

        if "suffix" in kwds:
            self.suffix = MyMunch(kwds.pop("suffix"))
        else:
            self.suffix = MyMunch(**kwds)

    def variable(self, index):
        if type(index) is int:
            return self._variables[index]
        else:
            return self.name_to_variable[index]

    def variables(self):
        return self._variables

    def objective(self, index=0):
        if type(index) is int:
            return self._objectives[index]
        else:
            return self.name_to_objective[index]

    def objectives(self):
        return self._objectives

    def to_dict(self):
        return dict(
            id=self.id,
            variables=[v.to_dict() for v in self.variables()],
            objectives=[o.to_dict() for o in self.objectives()],
            suffix=self.suffix.to_dict(),
        )

    def to_string(self, sort_keys=True, indent=4):
        return json.dumps(
            self.to_dict(),
            sort_keys=sort_keys,
            indent=indent,
        )

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


def PyomoSolution(*, variables=None, objective=None, objectives=None, **kwds):
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
                Variable(
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
            olist.append(Objective(value=pyo.value(obj), name=str(obj), index=index))
            index += 1

    return Solution(variables=vlist, objectives=olist, **kwds)
