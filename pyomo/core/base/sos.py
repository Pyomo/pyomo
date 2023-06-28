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

__all__ = ['SOSConstraint']

import sys
import logging

from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.timing import ConstructionTimer

from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
    ActiveIndexedComponent,
    UnindexedComponent_set,
)
from pyomo.core.base.set_types import PositiveIntegers

logger = logging.getLogger('pyomo.core')


class _SOSConstraintData(ActiveComponentData):
    """
    This class defines the data for a single special ordered set.

    Constructor arguments:
        owner           The Constraint object that owns this data.

    Public class attributes:
        active          A boolean that is true if this objective is active in the model.
        component       The constraint component.

    Private class attributes:
        _variables       SOS variables.
        _weights         SOS variable weights.
        _level           SOS level (Positive Integer)
    """

    __slots__ = ('_variables', '_weights', '_level')

    def __init__(self, owner):
        """Constructor"""
        self._level = None
        self._variables = []
        self._weights = []
        ActiveComponentData.__init__(self, owner)

    def num_variables(self):
        return len(self._variables)

    def items(self):
        return zip(self._variables, self._weights)

    @property
    def level(self):
        """
        Return the SOS level
        """
        return self._level

    @level.setter
    def level(self, level):
        if level not in PositiveIntegers:
            raise ValueError("SOS Constraint level must be a positive integer")
        self._level = level

    @property
    def variables(self):
        """
        Return the variable list for the SOS constraint
        """
        return self._variables

    def get_variables(self):
        for val in self._variables:
            yield val

    def get_items(self):
        assert len(self._variables) == len(self._weights)
        for v, w in zip(self._variables, self._weights):
            yield v, w

    def set_items(self, variables, weights):
        self._variables = []
        self._weights = []
        for v, w in zip(variables, weights):
            self._variables.append(v)
            if w < 0.0:
                raise ValueError(
                    "Cannot set negative weight %f for variable %s" % (w, v.name)
                )
            self._weights.append(w)


@ModelComponentFactory.register("SOS constraint expressions.")
class SOSConstraint(ActiveIndexedComponent):
    """
    Implements constraints for special ordered sets (SOS).

    Parameters
    ----------
    sos : int
        The type of SOS.
    var : pyomo.environ.Var
        The group of variables from which the SOS(s) will be created.
    index : pyomo.environ.Set, list or dict, optional
        A data structure with the indexes for the variables that are to be
        members of the SOS(s). The indexes can be provided as a pyomo Set:
        either indexed, if the SOS is indexed; or non-indexed, otherwise.
        Alternatively, the indexes can be provided as a list, for a non-indexed
        SOS, or as a dict, for indexed SOS(s).
    weights : pyomo.environ.Param or dict, optional
        A data structure with the weights for each member of the SOS(s). These
        can be provided as pyomo Param or as a dict. If not provided, the
        weights will be determined automatically using the var index set.
    rule : optional
        A method returning a 2-tuple with lists of variables and the respective
        weights in the same order, or a list of variables whose weights are
        then determined from their position within the list or, alternatively,
        pyomo.environ.Constraint.Skip if the constraint should be not be
        included in the model/instance. This parameter cannot be used in
        combination with var, index or weights.

    Examples
    -------

    1 - An SOS of type **N** made up of all members of a pyomo Var component:

    >>> # import pyomo
    >>> import pyomo.environ as pyo
    >>> # declare the model
    >>> model = pyo.AbstractModel()
    >>> # define the SOS type
    >>> N = 1 # 2, 3, ...
    >>> # the set that indexes the variables
    >>> model.A = pyo.Set()
    >>> # the variables under consideration
    >>> model.x = pyo.Var(model.A)
    >>> # the sos constraint
    >>> model.mysos = pyo.SOSConstraint(var=model.x, sos=N)

    2 - An SOS of type **N** made up of all members of a pyomo Var component,
    each with a specific weight:

    >>> # declare the model
    >>> model = pyo.AbstractModel()
    >>> # define the SOS type
    >>> N = 1 # 2, 3, ...
    >>> # the set that indexes the variables
    >>> model.A = pyo.Set()
    >>> # the variables under consideration
    >>> model.x = pyo.Var(model.A)
    >>> # the weights for each variable used in the sos constraints
    >>> model.mysosweights = pyo.Param(model.A)
    >>> # the sos constraint
    >>> model.mysos = pyo.SOSConstraint(
    ...     var=model.x,
    ...     sos=N,
    ...     weights=model.mysosweights
    ...     )

    3 - An SOS of type **N** made up of selected members of a Var component:

    >>> # declare the model
    >>> model = pyo.AbstractModel()
    >>> # define the SOS type
    >>> N = 1 # 2, 3, ...
    >>> # the set that indexes the variables
    >>> model.A = pyo.Set()
    >>> # the variables under consideration
    >>> model.x = pyo.Var(model.A)
    >>> # the set that indexes the variables actually used in the constraint
    >>> model.B = pyo.Set(within=model.A)
    >>> # the sos constraint
    >>> model.mysos = pyo.SOSConstraint(var=model.x, sos=N, index=model.B)

    4 - An SOS of type **N** made up of selected members of a Var component,
    each with a specific weight:

    >>> # declare the model
    >>> model = pyo.AbstractModel()
    >>> # define the SOS type
    >>> N = 1 # 2, 3, ...
    >>> # the set that indexes the variables
    >>> model.A = pyo.Set()
    >>> # the variables under consideration
    >>> model.x = pyo.Var(model.A)
    >>> # the set that indexes the variables actually used in the constraint
    >>> model.B = pyo.Set(within=model.A)
    >>> # the weights for each variable used in the sos constraints
    >>> model.mysosweights = pyo.Param(model.B)
    >>> # the sos constraint
    >>> model.mysos = pyo.SOSConstraint(
    ...     var=model.x,
    ...     sos=N,
    ...     index=model.B,
    ...     weights=model.mysosweights
    ...     )

    5 - A set of SOS(s) of type **N** made up of members of a pyomo Var
    component:

    >>> # declare the model
    >>> model = pyo.AbstractModel()
    >>> # define the SOS type
    >>> N = 1 # 2, 3, ...
    >>> # the set that indexes the variables
    >>> model.A = pyo.Set()
    >>> # the variables under consideration
    >>> model.x = pyo.Var(model.A)
    >>> # the set indexing the sos constraints
    >>> model.B = pyo.Set()
    >>> # the sets containing the variable indexes for each constraint
    >>> model.mysosvarindexset = pyo.Set(model.B)
    >>> # the sos constraints
    >>> model.mysos = pyo.SOSConstraint(
    ...     model.B,
    ...     var=model.x,
    ...     sos=N,
    ...     index=model.mysosvarindexset
    ...     )

    6 - A set of SOS(s) of type **N** made up of members of a pyomo Var
    component, each with a specific weight:

    >>> # declare the model
    >>> model = pyo.AbstractModel()
    >>> # define the SOS type
    >>> N = 1 # 2, 3, ...
    >>> # the set that indexes the variables
    >>> model.A = pyo.Set()
    >>> # the variables under consideration
    >>> model.x = pyo.Var(model.A)
    >>> # the set indexing the sos constraints
    >>> model.B = pyo.Set()
    >>> # the sets containing the variable indexes for each constraint
    >>> model.mysosvarindexset = pyo.Set(model.B)
    >>> # the set that indexes the variables used in the sos constraints
    >>> model.C = pyo.Set(within=model.A)
    >>> # the weights for each variable used in the sos constraints
    >>> model.mysosweights = pyo.Param(model.C)
    >>> # the sos constraints
    >>> model.mysos = pyo.SOSConstraint(
    ...     model.B,
    ...     var=model.x,
    ...     sos=N,
    ...     index=model.mysosvarindexset,
    ...     weights=model.mysosweights,
    ...     )

    7 - A simple SOS of type **N** created using the rule parameter:

    >>> # declare the model
    >>> model = pyo.AbstractModel()
    >>> # define the SOS type
    >>> N = 1 # 2, 3, ...
    >>> # the set that indexes the variables
    >>> model.A = pyo.Set()
    >>> # the variables under consideration
    >>> model.x = pyo.Var(model.A, domain=pyo.NonNegativeReals)
    >>> # the rule method creating the constraint
    >>> def rule_mysos(m):
    ...     var_list = [m.x[a] for a in m.x]
    ...     weight_list = [i+1 for i in range(len(var_list))]
    ...     return (var_list, weight_list)
    >>> # the sos constraint(s)
    >>> model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=N)

    8 - A simple SOS of type **N** created using the rule parameter, in which
    the weights are determined automatically:

    >>> # declare the model
    >>> model = pyo.AbstractModel()
    >>> # define the SOS type
    >>> N = 1 # 2, 3, ...
    >>> # the set that indexes the variables
    >>> model.A = pyo.Set()
    >>> # the variables under consideration
    >>> model.x = pyo.Var(model.A, domain=pyo.NonNegativeReals)
    >>> # the rule method creating the constraint
    >>> def rule_mysos(m):
    ...     return [m.x[a] for a in m.x]
    >>> # the sos constraint(s)
    >>> model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=N)

    9 - A set of SOS(s) of type **N** involving members of distinct pyomo Var
    components, each with a specific weight. This requires the rule parameter:

    >>> # declare the model
    >>> model = pyo.AbstractModel()
    >>> # define the SOS type
    >>> N = 1 # 2, 3, ...
    >>> # the set that indexes the x variables
    >>> model.A = pyo.Set()
    >>> # the set that indexes the y variables
    >>> model.B = pyo.Set()
    >>> # the set that indexes the SOS constraints
    >>> model.C = pyo.Set()
    >>> # the x variables, which will be used in the constraints
    >>> model.x = pyo.Var(model.A, domain=pyo.NonNegativeReals)
    >>> # the y variables, which will be used in the constraints
    >>> model.y = pyo.Var(model.B, domain=pyo.NonNegativeReals)
    >>> # the x variable indices for each constraint
    >>> model.mysosindex_x = pyo.Set(model.C)
    >>> # the y variable indices for each constraint
    >>> model.mysosindex_y = pyo.Set(model.C)
    >>> # the weights for the x variable indices
    >>> model.mysosweights_x = pyo.Param(model.A)
    >>> # the weights for the y variable indices
    >>> model.mysosweights_y = pyo.Param(model.B)
    >>> # the rule method with which each constraint c is built
    >>> def rule_mysos(m, c):
    ...     var_list = [m.x[a] for a in m.mysosindex_x[c]]
    ...     var_list.extend([m.y[b] for b in m.mysosindex_y[c]])
    ...     weight_list = [m.mysosweights_x[a] for a in m.mysosindex_x[c]]
    ...     weight_list.extend([m.mysosweights_y[b] for b in m.mysosindex_y[c]])
    ...     return (var_list, weight_list)
    >>> # the sos constraint(s)
    >>> model.mysos = pyo.SOSConstraint(
    ...     model.C,
    ...     rule=rule_mysos,
    ...     sos=N
    ...     )

    """

    Skip = (1000,)

    def __new__(cls, *args, **kwds):
        if cls != SOSConstraint:
            return super(SOSConstraint, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return ScalarSOSConstraint.__new__(ScalarSOSConstraint)
        else:
            return IndexedSOSConstraint.__new__(IndexedSOSConstraint)

    def __init__(self, *args, **kwargs):
        """
        Constructor
        """
        #
        # The 'initialize' or 'rule' argument
        #
        initialize = kwargs.pop('initialize', None)
        initialize = kwargs.pop('rule', initialize)
        if not initialize is None:
            if 'var' in kwargs:
                raise TypeError(
                    "Cannot specify the 'var' argument with the 'rule' or 'initialize' argument"
                )
            if 'index' in kwargs:
                raise TypeError(
                    "Cannot specify the 'index' argument with the 'rule' or 'initialize' argument"
                )
            if 'weights' in kwargs:
                raise TypeError(
                    "Cannot specify the 'weights' argument with the 'rule' or 'initialize' argument"
                )
        #
        # The 'var' argument
        #
        sosVars = kwargs.pop('var', None)
        if sosVars is None and initialize is None:
            raise TypeError(
                "SOSConstraint() requires either the 'var' or 'initialize' arguments"
            )
        #
        # The 'weights' argument
        #
        sosWeights = kwargs.pop('weights', None)
        #
        # The 'index' argument
        #
        sosSet = kwargs.pop('index', None)
        #
        # The 'sos' or 'level' argument
        #
        if 'sos' in kwargs and 'level' in kwargs:
            raise TypeError(
                "Specify only one of 'sos' and 'level' -- "
                "they are equivalent keyword arguments"
            )
        sosLevel = kwargs.pop('sos', None)
        sosLevel = kwargs.pop('level', sosLevel)
        if sosLevel is None:
            raise TypeError(
                "SOSConstraint() requires that either the "
                "'sos' or 'level' keyword arguments be set to indicate "
                "the type of SOS."
            )
        #
        # Set attributes
        #
        self._sosVars = sosVars
        self._sosWeights = sosWeights
        self._sosSet = sosSet
        self._sosLevel = sosLevel
        self._rule = initialize
        #
        # Construct the base class
        #
        kwargs.setdefault('ctype', SOSConstraint)
        ActiveIndexedComponent.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        """
        Construct this component
        """
        assert data is None  # because I don't know why it's an argument
        generate_debug_messages = is_debug_set(logger)
        if self._constructed is True:  # pragma:nocover
            return

        if generate_debug_messages:  # pragma:nocover
            logger.debug("Constructing SOSConstraint %s", self.name)
        timer = ConstructionTimer(self)
        self._constructed = True

        if self._rule is None:
            if self._sosSet is None and self.is_indexed():
                if generate_debug_messages:  # pragma:nocover
                    logger.debug(
                        "  Cannot construct "
                        + self.name
                        + ".  No rule is defined and no SOS sets are defined."
                    )
            else:
                if not self.is_indexed():
                    if self._sosSet is None:
                        if getattr(
                            self._sosVars.index_set(), 'isordered', lambda *x: False
                        )():
                            _sosSet = {None: list(self._sosVars.index_set())}
                        else:
                            _sosSet = {None: set(self._sosVars.index_set())}
                    else:
                        _sosSet = {None: self._sosSet}
                else:
                    _sosSet = self._sosSet

                for index, sosSet in _sosSet.items():
                    if generate_debug_messages:  # pragma:nocover
                        logger.debug(
                            "  Constructing " + self.name + " index " + str(index)
                        )

                    if self._sosLevel == 2:
                        #
                        # Check that the sets are ordered.
                        #
                        ordered = False
                        if (
                            type(sosSet) is list
                            or sosSet is UnindexedComponent_set
                            or len(sosSet) == 1
                        ):
                            ordered = True
                        if hasattr(sosSet, 'isordered') and sosSet.isordered():
                            ordered = True
                        if not ordered:
                            raise ValueError(
                                "Cannot define a SOS over an unordered index."
                            )

                    variables = [self._sosVars[idx] for idx in sosSet]
                    if self._sosWeights is not None:
                        weights = [self._sosWeights[idx] for idx in sosSet]
                    else:
                        weights = None

                    self.add(index, variables, weights)
        else:
            _self_rule = self._rule
            _self_parent = self._parent()
            for index in self._index_set:
                try:
                    tmp = apply_indexed_rule(self, _self_rule, _self_parent, index)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error(
                        "Rule failed when generating expression for "
                        "sos constraint %s with index %s:\n%s: %s"
                        % (self.name, str(index), type(err).__name__, err)
                    )
                    raise
                if tmp is None:
                    raise ValueError(
                        "SOSConstraint rule returned None instead of SOSConstraint.Skip for index %s"
                        % str(index)
                    )
                if type(tmp) is tuple:
                    if tmp is SOSConstraint.Skip:
                        continue
                    # tmp is a tuple of variables, weights
                    self.add(index, tmp[0], tmp[1])
                else:
                    # tmp is a list of variables
                    self.add(index, tmp)
        timer.report()

    def add(self, index, variables, weights=None):
        """
        Add a component data for the specified index.
        """
        if index is None:
            # because ScalarSOSConstraint already makes an _SOSConstraintData instance
            soscondata = self
        else:
            soscondata = _SOSConstraintData(self)
        self._data[index] = soscondata
        soscondata._index = index

        soscondata.level = self._sosLevel

        if weights is None:
            soscondata.set_items(variables, list(range(1, len(variables) + 1)))
        else:
            soscondata.set_items(variables, weights)

    # NOTE: the prefix option is ignored
    def pprint(self, ostream=None, verbose=False, prefix=""):
        """TODO"""
        if ostream is None:
            ostream = sys.stdout
        ostream.write("   " + self.local_name + " : ")
        if not self.doc is None:
            ostream.write(self.doc + '\n')
            ostream.write("  ")
        ostream.write("\tSize=" + str(len(self._data.keys())) + ' ')
        if self.is_indexed():
            ostream.write("\tIndex= " + self._index_set.name + '\n')
        else:
            ostream.write("\n")
        for val in self._data:
            if not val is None:
                ostream.write("\t" + str(val) + '\n')
            ostream.write("\t\tType=" + str(self._data[val].level) + '\n')
            ostream.write("\t\tWeight : Variable\n")
            for var, weight in self._data[val].get_items():
                ostream.write("\t\t" + str(weight) + ' : ' + var.name + '\n')


class ScalarSOSConstraint(SOSConstraint, _SOSConstraintData):
    def __init__(self, *args, **kwd):
        _SOSConstraintData.__init__(self, self)
        SOSConstraint.__init__(self, *args, **kwd)
        self._index = UnindexedComponent_index


class SimpleSOSConstraint(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarSOSConstraint
    __renamed__version__ = '6.0'


class IndexedSOSConstraint(SOSConstraint):
    def __init__(self, *args, **kwds):
        super(IndexedSOSConstraint, self).__init__(*args, **kwds)
