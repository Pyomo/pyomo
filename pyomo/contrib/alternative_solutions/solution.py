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
        self.int_to_variable = {}
        self.str_to_variable = {}
        if variables is not None:
            self._variables = variables
            index = 0
            for v in variables:
                self.int_to_variable[index] = v
                if v.name is not None:
                    self.str_to_variable[v.name] = v
                index += 1

        self._objectives = []
        self.int_to_objective = {}
        self.str_to_objective = {}
        if objective is not None:
            objectives = [objective]
        if objectives is not None:
            self._objectives = objectives
            index = 0
            for o in objectives:
                self.int_to_objective[index] = o
                if o.name is not None:
                    self.str_to_objective[o.name] = o
                index += 1

        if "suffix" in kwds:
            self.suffix = MyMunch(kwds.pop("suffix"))
        else:
            self.suffix = MyMunch(**kwds)

    def variable(self, index):
        if type(index) is int:
            return self.int_to_variable[index]
        else:
            return self.str_to_variable[index]

    def variables(self):
        return self._variables

    def tuple_repn(self):
        if len(self.int_to_variable) == len(self._variables):
            return tuple(
                tuple([k, var.value]) for k, var in self.int_to_variable.items()
            )
        elif len(self.str_to_variable) == len(self._variables):
            return tuple(
                tuple([k, var.value]) for k, var in self.str_to_variable.items()
            )
        else:
            return tuple(tuple([k, var.value]) for k, var in enumerate(self._variables))

    def objective(self, index=0):
        if type(index) is int:
            return self.int_to_objective[index]
        else:
            return self.str_to_objective[index]

    def objectives(self):
        return self._objectives

    def to_dict(self):
        return dict(
            id=self.id,
            variables=[v.to_dict() for v in self.variables()],
            objectives=[o.to_dict() for o in self.objectives()],
            suffix=self.suffix.to_dict(),
        )


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
            vlist.append(Variable(value=pyo.value(var), fixed=var.is_fixed(), name=str(var), index=index, discrete=not var.is_continuous()))
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

