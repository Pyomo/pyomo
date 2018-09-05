#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['SOSConstraint']

import sys
import logging
import six
from six.moves import zip, xrange

from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.core.base.component import ActiveComponentData
from pyomo.core.base.indexed_component import ActiveIndexedComponent, UnindexedComponent_set
from pyomo.core.base.set_types import PositiveIntegers
from pyomo.core.base.sets import Set, _IndexedOrderedSetData

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
        """ Constructor """
        self._level = None
        self._variables = []
        self._weights = []
        ActiveComponentData.__init__(self, owner)

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        result = super(_SOSConstraintData, self).__getstate__()
        for i in _SOSConstraintData.__slots__:
            result[i] = getattr(self, i)
        return result

    # Since this class requires no special processing of the state
    # dictionary, it does not need to implement __setstate__()

    def num_variables(self):
        return len(self._variables)

    @property
    def level(self):
        """
        Return the SOS level
        """
        return self._level

    @level.setter
    def level(self, level):
        if level not in PositiveIntegers:
            raise ValueError("SOS Constraint level must "
                             "be a positive integer")
        self._level = level

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
                raise ValueError("Cannot set negative weight %f "
                                 "for variable %s" % (w, v.name))
            self._weights.append(w)


@ModelComponentFactory.register("SOS constraint expressions.")
class SOSConstraint(ActiveIndexedComponent):
    """
    Represents an SOS-n constraint.

    Usage:
    model.C1 = SOSConstraint(
                             [...],
                             var=VAR,
                             [set=SET OR index=SET],
                             [sos=N OR level=N]
                             [weights=WEIGHTS]
                             )
        [...]   Any number of sets used to index SET
        VAR     The set of variables making up the SOS. Indexed by SET.
        SET     The set used to index VAR. SET is optionally indexed by
                the [...] sets. If SET is not specified, VAR is indexed
                over the set(s) it was defined with.
        N       This constraint is an SOS-N constraint. Defaults to 1.
        WEIGHTS A Param representing the variables weights in the SOS sets.
                A simple counter is used to generate weights when this keyword
                is not used.

    Example:

      model = AbstractModel()
      model.A = Set()
      model.B = Set(A)
      model.X = Set(B)

      model.C1 = SOSConstraint(model.A, var=model.X, set=model.B, sos=1)

    This constraint actually creates one SOS-1 constraint for each
    element of model.A (e.g., if |A| == N, there are N constraints).
    In each constraint, model.X is indexed by the elements of
    model.B[a], where 'a' is the current index of model.A.

      model = AbstractModel()
      model.A = Set()
      model.X = Var(model.A)

      model.C2 = SOSConstraint(var=model.X, sos=2)

    This produces exactly one SOS-2 constraint using all the variables
    in model.X.
    """

    Skip            = (1000,)

    def __new__(cls, *args, **kwds):
        if cls != SOSConstraint:
            return super(SOSConstraint, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            return SimpleSOSConstraint.__new__(SimpleSOSConstraint)
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
                raise TypeError("Cannot specify the 'var' argument with the 'rule' or 'initialize' argument")
            if 'index' in kwargs:
                raise TypeError("Cannot specify the 'index' argument with the 'rule' or 'initialize' argument")
            if 'weights' in kwargs:
                raise TypeError("Cannot specify the 'weights' argument with the 'rule' or 'initialize' argument")
        #
        # The 'var' argument
        #
        sosVars = kwargs.pop('var', None)
        if sosVars is None and initialize is None:
            raise TypeError("SOSConstraint() requires either the 'var' or 'initialize' arguments")
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
            raise TypeError("Specify only one of 'sos' and 'level' -- " \
                  "they are equivalent keyword arguments")
        sosLevel = kwargs.pop('sos', None)
        sosLevel = kwargs.pop('level', sosLevel)
        if sosLevel is None:
            raise TypeError("SOSConstraint() requires that either the " \
                  "'sos' or 'level' keyword arguments be set to indicate " \
                  "the type of SOS.")
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
        assert data is None # because I don't know why it's an argument
        generate_debug_messages \
            = __debug__ and logger.isEnabledFor(logging.DEBUG)
        if self._constructed is True:   #pragma:nocover
            return

        if generate_debug_messages:     #pragma:nocover
            logger.debug("Constructing SOSConstraint %s",self.name)
        timer = ConstructionTimer(self)
        self._constructed = True

        if self._rule is None:
            if self._sosSet is None and self.is_indexed():
                if generate_debug_messages:     #pragma:nocover
                    logger.debug("  Cannot construct "+self.name+".  No rule is defined and no SOS sets are defined.")
            else:
                if not self.is_indexed():
                    if self._sosSet is None:
                        if getattr(self._sosVars.index_set(), 'ordered', False):
                            _sosSet = {None: list(self._sosVars.index_set())}
                        else:
                            _sosSet = {None: set(self._sosVars.index_set())}
                    else:
                        _sosSet = {None: self._sosSet}
                else:
                    _sosSet = self._sosSet

                for index, sosSet in six.iteritems(_sosSet):
                    if generate_debug_messages:     #pragma:nocover
                        logger.debug("  Constructing "+self.name+" index "+str(index))

                    if self._sosLevel == 2:
                        #
                        # Check that the sets are ordered.
                        #
                        ordered=False
                        if type(sosSet) is list or sosSet is UnindexedComponent_set or len(sosSet) == 1:
                            ordered=True
                        if hasattr(sosSet, 'ordered') and sosSet.ordered:
                            ordered=True
                        if type(sosSet) is _IndexedOrderedSetData:
                            ordered=True
                        if not ordered:
                            raise ValueError("Cannot define a SOS over an unordered index.")

                    variables = [self._sosVars[idx] for idx in sosSet]
                    if self._sosWeights is not None:
                        weights = [self._sosWeights[idx] for idx in sosSet]
                    else:
                        weights = None

                    self.add(index, variables, weights)
        else:
            _self_rule = self._rule
            _self_parent = self._parent()
            for index in self._index:
                try:
                    tmp = apply_indexed_rule(self, _self_rule, _self_parent, index)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error(
                        "Rule failed when generating expression for "
                        "sos constraint %s with index %s:\n%s: %s"
                        % ( self.name, str(index), type(err).__name__, err ) )
                    raise
                if tmp is None:
                    raise ValueError("SOSConstraint rule returned None instead of SOSConstraint.Skip for index %s" % str(index))
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
            # because SimpleSOSConstraint already makes an _SOSConstraintData instance
            soscondata = self
        else:
            soscondata = _SOSConstraintData(self)
        self._data[index] = soscondata

        soscondata.level = self._sosLevel

        if weights is None:
            soscondata.set_items(variables, list(xrange(1, len(variables)+1)))
        else:
            soscondata.set_items(variables, weights)

    # NOTE: the prefix option is ignored
    def pprint(self, ostream=None, verbose=False, prefix=""):
        """TODO"""
        if ostream is None:
            ostream = sys.stdout
        ostream.write("   "+self.local_name+" : ")
        if not self.doc is None:
            ostream.write(self.doc+'\n')
            ostream.write("  ")
        ostream.write("\tSize="+str(len(self._data.keys()))+' ')
        if isinstance(self._index,Set):
            ostream.write("\tIndex= "+self._index.name+'\n')
        else:
            ostream.write("\n")
        for val in self._data:
            if not val is None:
                ostream.write("\t"+str(val)+'\n')
            ostream.write("\t\tType="+str(self._data[val].level)+'\n')
            ostream.write("\t\tWeight : Variable\n")
            for var, weight in self._data[val].get_items():
                ostream.write("\t\t"+str(weight)+' : '+var.name+'\n')


# Since this class derives from Component and Component.__getstate__
# just packs up the entire __dict__ into the state dict, there s
# nothing special that we need to do here.  We will just defer to the
# super() get/set state.  Since all of our get/set state methods
# rely on super() to traverse the MRO, this will automatically pick
# up both the Component and Data base classes.

class SimpleSOSConstraint(SOSConstraint, _SOSConstraintData):

    def __init__(self, *args, **kwd):
        _SOSConstraintData.__init__(self, self)
        SOSConstraint.__init__(self, *args, **kwd)


class IndexedSOSConstraint(SOSConstraint):

    def __init__(self, *args, **kwds):
        super(IndexedSOSConstraint,self).__init__(*args, **kwds)

