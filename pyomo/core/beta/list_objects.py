#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = () #('XVarList', 'XConstraintList', 'XObjectiveList', 'XExpressionList')

import logging
from weakref import ref as weakref_ref
import collections

from pyomo.core.base.set_types import Any
from pyomo.core.base.var import (IndexedVar,
                                 _VarData)
from pyomo.core.base.constraint import (IndexedConstraint,
                                        _ConstraintData)
from pyomo.core.base.objective import (IndexedObjective,
                                       _ObjectiveData)
from pyomo.core.base.expression import (IndexedExpression,
                                        _ExpressionData)

logger = logging.getLogger('pyomo.core')

#
# In the future I think ComponentDict and ComponentList should inherit
# directly from (Active)IndexedComponent, and that class should be
# stripped down to a minimal interface. The abstract interface should
# be implemented on top of these classes.
#

class ComponentList(collections.MutableSequence):

    def __init__(self, interface_datatype, *args):
        self._interface_datatype = interface_datatype
        self._data = []
        if len(args) > 0:
            if len(args) > 1:
                raise TypeError(
                    "ComponentList expected at most 1 arguments, "
                    "got %s" % (len(args)))
            for item in args[0]:
                self.append(item)

    def construct(self, data=None):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug(   #pragma:nocover
                "Constructing ComponentList object, name=%s, from data=%s"
                % (self.name, str(data)))
        if self._constructed:   #pragma:nocover
            return
        self._constructed = True

    #
    # Note: The following methods add a flavor of the dict-interface
    #       to this component. This is simply a hack that allows us to
    #       create working Pyomo examples using this prototype. The
    #       correct way to go would be to remove these methods and to
    #       update pyomo code everywhere to first check whether it is
    #       dealing with a dict or a list interface before
    #       iterating. I don't think that would be difficult to do.
    #

    def keys(self): return range(len(self))
    iterkeys = keys
    def values(self): return list(iter(self))
    itervalues = values
    def items(self): return zip(self.keys(), self.values())
    iteritems = items

    #
    # Define the MutableSequence abstract methods
    #

    # Currently __setitem__ only supports assignment of an explicit
    # instantiation of the interface datatype. Updates to an already
    # existing object inside this container need to be done via that
    # objects interface methods or by directly accessing object
    # attributes.
    #
    # * See notes above the __setitem__ method for ComponentDict
    #   about potentially supporting implicit assignment / update
    #
    def __setitem__(self, i, item):
        if isinstance(item, self._interface_datatype):
            # release the current component (assuming we don't get
            # an index error)
            # * see __delitem__ for explanation
            if item._component is None:
                item._component = weakref_ref(self)
                if hasattr(self, "_active"):
                    self._active |= getattr(item, '_active', True)
                self._data[i]._component = None
                self._data[i] = item
                return
            elif self._data[i] is item:
                # a very special case that makes sense to handle
                # because the implied order should be: (1) delete
                # the object at the current index, (2) insert the
                # the new object. This performs both without any
                # actions, but it is an extremely rare case, so
                # it should go last.
                return
            # see note about allowing components to live in more than
            # one container
            raise ValueError(
                "Invalid component object assignment to ComponentList "
                "%s at index %s. A parent component has already been "
                "assigned the object: %s"
                % (self.name,
                   i,
                   item.parent_component().name))
        # see note about implicit assignment and update
        raise TypeError(
            "ComponentList must be assigned objects "
            "of type %s. Invalid type for key %s: %s"
            % (self._interface_datatype.__name__,
               i,
               type(item)))

    # * Only supports explicit objects. See notes above __setitem__
    #   for more information
    def insert(self, i, item):
        if isinstance(item, self._interface_datatype):
            if item._component is None:
                item._component = weakref_ref(self)
                if hasattr(self, "_active"):
                    self._active |= getattr(item, '_active', True)
                self._data.insert(i, item)
                return
            # see note about allowing components to live in more than
            # one container
            raise ValueError(
                "Invalid component object assignment to ComponentList "
                "%s at index %s. A parent component has already been "
                "assigned the object: %s"
                % (self.name,
                   i,
                   item.parent_component().name))
        # see note about implicit assignment and update
        raise TypeError(
            "ComponentList must be assigned objects "
            "of type %s. Invalid type for key %s: %s"
            % (self._interface_datatype.__name__,
               i,
               type(item)))

    # Since we don't currently allow objects to be assigned when their
    # parent component is already set, it would make sense to reset
    # the parent component back to None for an object being removed
    # from this container; thus, allowing objects to be naturally
    # transferred between containers if one so chooses. This would be
    # more complicated if allowing components to live in more than one
    # container (see note above __setitem__), and maybe this is the
    # reason not to support that.
    def __delitem__(self, i):
        obj = self._data[i]
        obj._component = None
        del self._data[i]

    def __getitem__(self, i): return self._data[i]
    def __len__(self): return self._data.__len__()

    #
    # Override a few default implementations on MutableSequence
    #

    # We want to avoid generating Pyomo expressions by comparing values
    def __contains__(self, item):
        item_id = id(item)
        return any(item_id == id(_v) for _v in self._data)

    # We want to avoid generating Pyomo expressions by comparing values
    def index(self, item, start=0, stop=None):
        '''S.index(value, [start, [stop]]) -> integer -- return first index of value.

           Raises ValueError if the value is not present.
        '''
        if start is not None and start < 0:
            start = max(len(self) + start, 0)
        if stop is not None and stop < 0:
            stop += len(self)

        item_id = id(item)
        i = start
        while stop is None or i < stop:
            try:
                if id(self[i]) == item_id:
                    return i
            except IndexError:
                break
            i += 1
        raise ValueError

    # We want to avoid generating Pyomo expressions by comparing values
    def count(self, item):
        'S.count(value) -> integer -- return number of occurrences of value'
        item_id = id(item)
        cnt = sum(1 for _v in self._data if id(_v) == item_id)
        assert cnt == 1
        return cnt

    # Avoid errors related to calling __setitem__
    # with a component that is already owned
    def reverse(self):
        'S.reverse() -- reverse *IN PLACE*'
        n = len(self)
        data = self._data
        for i in range(n//2):
            data[i], data[n-i-1] = data[n-i-1], data[i]

#
# ComponentList needs to come before IndexedComponent
# (or subclasses of) so we can override certain methods
#

class XVarList(ComponentList, IndexedVar):

    def __init__(self, *args, **kwds):
        IndexedVar.__init__(self, Any, **kwds)
        # Constructor for ComponentList needs to
        # go last in order to handle any initialization
        # iterable as an argument
        ComponentList.__init__(self,
                               _VarData,
                               *args,
                               **kwds)

class XConstraintList(ComponentList, IndexedConstraint):

    def __init__(self, *args, **kwds):
        IndexedConstraint.__init__(self, Any, **kwds)
        # Constructor for ComponentList needs to
        # go last in order to handle any initialization
        # iterable as an argument
        ComponentList.__init__(self,
                               _ConstraintData,
                               *args,
                               **kwds)

class XObjectiveList(ComponentList, IndexedObjective):

    def __init__(self, *args, **kwds):
        IndexedObjective.__init__(self, Any, **kwds)
        # Constructor for ComponentList needs to
        # go last in order to handle any initialization
        # iterable as an argument
        ComponentList.__init__(self,
                               _ObjectiveData,
                               *args,
                               **kwds)

class XExpressionList(ComponentList, IndexedExpression):

    def __init__(self, *args, **kwds):
        IndexedExpression.__init__(self, Any, **kwds)
        # Constructor for ComponentList needs to
        # go last in order to handle any initialization
        # iterable as an argument
        ComponentList.__init__(self,
                               _ExpressionData,
                               *args,
                               **kwds)
