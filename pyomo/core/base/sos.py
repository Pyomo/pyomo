#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

__all__ = ['SOSConstraint']

import sys
import logging
import weakref
from six.moves import zip

from pyomo.core.base.component import ActiveComponentData, register_component
from pyomo.core.base.sparse_indexed_component import ActiveSparseIndexedComponent
from pyomo.core.base.set_types import PositiveIntegers
from pyomo.core.base.sets import Set


logger = logging.getLogger('pyomo.core')


class _SOSConstraintData(ActiveComponentData): 
    """
    This class defines the data for a single special ordered set.

    Constructor arguments:
        component       The Constraint object that owns this data.

    Public class attributes:
        active          A boolean that is true if this objective is active in the model.
        component       The constraint component.

    Private class attributes:
        _members         SOS member variables.
        _weights         SOS member weights.
        _level           SOS level (Positive Integer)
    """

    __pickle_slots__ = ( '_members', '_weights', '_level')
    __slots__ = __pickle_slots__ + ( '__weakref__', )

    def __init__(self, owner):
        self._level = None
        self._members = []
        self._weights = []
        
        ActiveComponentData.__init__(self, owner)

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        result = super(_SOSConstraintData, self).__getstate__()
        for i in _SOSConstraintData.__pickle_slots__:
            result[i] = getattr(self, i)
        return result

    # Since this class requires no special processing of the state
    # dictionary, it does not need to implement __setstate__()

    def get_weights(self):
        return self._weights

    def get_members(self):
        return self._members

    def get_items(self):
        assert len(self._members) == len(self._weights)
        return list(zip(self._members, self._weights))

    def get_level(self):
        return self._level

    def member_index(self, vardata):
        # it does not suffice to check "if vardata in self._members"
        # because variables hash by .value not by "is"
        member_idx = None
        for idx, member_vardata in enumerate(self._members):
            if vardata is member_vardata:
                member_idx = idx
        return member_idx

    def add_member(self, vardata, weight=None):
        if self.member_index(vardata) is not None:
            raise ValueError("Variable '%s' is already a member of SOSConstraint '%s'" \
                                 % (vardata.cname(True), self.cname(True)))
        self._members.append(vardata)
        if weight is None:
            if len(self._weights) == 0:
                weight = 1
            else:
                weight = max(self._weights)+1
        self._weights.append(weight)

    def remove_member(self, vardata):
        idx = self.member_index(vardata)
        if idx is None:
            raise ValueError("Variable '%s' is not a member of SOSConstraint '%s'" \
                                 % (vardata.cname(True), self.cname(True)))
        del self._members[idx]
        del self._weights[idx]

    def set_weight(self, vardata, weight):
        idx = self.member_index(vardata)
        if idx is None:
            raise ValueError("Variable '%s' is not a member of SOSConstraint '%s'" \
                                 % (vardata.cname(True), self.cname(True)))
        self._weights[idx] = weight
                                 
    def set_level(self, level):
        if level not in PositiveIntegers:
            raise ValueError("SOS Constraint level must be a positive integer")
        self._level = level


class SOSConstraint(ActiveSparseIndexedComponent):
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
    model.D[a], where 'a' is the current index of model.A.

      model = AbstractModel()
      model.A = Set()
      model.X = Var(model.A)

      model.C2 = SOSConstraint(var=model.X, sos=2)

    This produces exactly one SOS-2 constraint using all the variables
    in model.X.
    """

    def __new__(cls, *args, **kwds):
        if cls != SOSConstraint:
            return super(SOSConstraint, cls).__new__(cls)
        if args == ():
            return SimpleSOSConstraint.__new__(SimpleSOSConstraint)
        else:
            return IndexedSOSConstraint.__new__(IndexedSOSConstraint)

    def __init__(self, *args, **kwargs):
        # Get the 'var' parameter
        sosVars = kwargs.pop('var', None)
        # Make sure we have a variable
        if sosVars is None:
            raise TypeError("SOSConstraint() requires the 'var' keyword " \
                  "be specified")

        sosWeights = kwargs.pop('weights', None)

        # Get the 'set' or 'index' parameters
        if 'set' in kwargs and 'index' in kwargs:
            raise TypeError("Specify only one of 'set' and 'index' -- " \
                  "they are equivalent parameters")
        sosSet = kwargs.pop('set', None)
        sosSet = kwargs.pop('index', sosSet)

        # Get the 'sos' or 'level' parameters
        if 'sos' in kwargs and 'level' in kwargs:
            raise TypeError("Specify only one of 'sos' and 'level' -- " \
                  "they are equivalent parameters")
        sosLevel = kwargs.pop('sos', None)
        sosLevel = kwargs.pop('level', sosLevel)

        # Make sure sosLevel has been set
        if sosLevel is None:
            raise TypeError("SOSConstraint() requires that either the " \
                  "'sos' or 'level' keyword arguments be set to indicate " \
                  "the type of SOS.")

        # Set member attributes
        self._sosVars = sosVars
        self._sosWeights = sosWeights
        self._sosSet = sosSet
        self._sosLevel = sosLevel

        kwargs.setdefault('ctype', SOSConstraint)
        ActiveSparseIndexedComponent.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        assert data is None # because I don't know why it's an argument

        generate_debug_messages = __debug__ and logger.isEnabledFor(logging.DEBUG)

        if generate_debug_messages:
            logger.debug("Constructing SOSConstraint %s",self.cname(True))

        if self._constructed is True:
            return
        self._constructed = True

        for index in self._index:
            if generate_debug_messages:
                logger.debug("  Constructing "+self.cname(True)+" index "+str(index))
            self.add(index)

    def add(self, index):
        
        if index is None:
            # because SimpleSOSConstraint already makes an
            # _SOSConstraintData instance
            soscondata = self
        else:
            soscondata = _SOSConstraintData(self)
        soscondata.set_level(self._sosLevel)

        if (self._sosSet is None):
            sosSet = self._sosVars.index_set()
        else:
            if index is None:
                sosSet = self._sosSet
            else:
                sosSet = self._sosSet[index]

        weights = None
        if index is None:
            vars = [self._sosVars[idx] for idx in sosSet]
            if self._sosWeights is not None:
                weights = [self._sosWeights[idx] for idx in sosSet]
            else:
                weights = list(i for i,idx in enumerate(sosSet,1))
        else:
            vars = [self._sosVars[idx] for idx in sosSet]
            if self._sosWeights is not None:
                weights = [self._sosWeights[idx] for idx in sosSet]
            else:
                weights = list(i for i,idx in enumerate(sosSet,1))

        for var, weight in zip(vars,weights):
            soscondata.add_member(var,weight)

        self._data[index] = soscondata

    # NOTE: the prefix option is ignored
    def pprint(self, ostream=None, verbose=False, prefix=""):
        """TODO"""
        if ostream is None:
            ostream = sys.stdout
        ostream.write("   "+self.cname()+" : ")
        if not self.doc is None:
            ostream.write(self.doc+'\n')
            ostream.write("  ")
        ostream.write("\tSize="+str(len(self._data.keys()))+' ')
        if isinstance(self._index,Set):
            ostream.write("\tIndex= "+self._index.cname(True)+'\n')
        else:
            ostream.write("\n")
        for val in self._data:
            if not val is None:
                ostream.write("\t"+str(val)+'\n')
            ostream.write("\t\tType="+str(self._data[val].get_level())+'\n')
            ostream.write("\t\tMembers= (Weight:Variable)\n")
            for var, weight in zip(self._data[val].get_members(), self._data[val].get_weights()):
                ostream.write("\t\t\t"+str(weight)+':'+var.cname(True)+'\n')


class SimpleSOSConstraint(SOSConstraint, _SOSConstraintData):

    def __init__(self, *args, **kwd):
        _SOSConstraintData.__init__(self, self)
        SOSConstraint.__init__(self, *args, **kwd)

    # Since this class derives from Component and Component.__getstate__
    # just packs up the entire __dict__ into the state dict, there s
    # nothng special that we need to do here.  We will just defer to the
    # super() get/set state.  Since all of our get/set state methods
    # rely on super() to traverse the MRO, this will automatically pick
    # up both the Component and Data base classes.


class IndexedSOSConstraint(SOSConstraint):

    def __init__(self, *args, **kwds):
        super(IndexedSOSConstraint,self).__init__(*args, **kwds)


register_component(SOSConstraint, "SOS constraint expressions.")

