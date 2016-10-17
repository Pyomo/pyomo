#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = () #('VarDict', 'ConstraintDict', 'ObjectiveDict', 'ExpressionDict')

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

class ComponentDict(collections.MutableMapping):

    def __init__(self, interface_datatype, *args):
        self._interface_datatype = interface_datatype
        self._data = {}
        if len(args) > 0:
            if len(args) > 1:
                raise TypeError(
                    "ComponentDict expected at most 1 arguments, "
                    "got %s" % (len(args)))
            self.update(args[0])

    def construct(self, data=None):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug(   #pragma:nocover
                "Constructing ComponentDict object, name=%s, from data=%s"
                % (self.name, str(data)))
        if self._constructed:   #pragma:nocover
            return
        self._constructed = True

    #
    # Define the MutableMapping abstract methods
    #

    #
    # Currently __setitem__ only supports assignment of an explicit
    # instantiation of the interface datatype. Updates to an already
    # existing object inside this container need to be done via that
    # objects interface methods or by directly accessing object
    # attributes.
    #
    # * Note about implicit instantiation / update:
    #    Implicit instantiation or update using assignment of a valid
    #    value type could be supported, but we should more carefully
    #    consider edge cases before guaranteeing such functionality.
    #    The explicit approach provides all functionality one would
    #    need, albeit, with a slightly more verbose syntax (which one
    #    could argue is more clear).
    #
    # * Note about component ownership
    #    Currently this method only supports assignment of datatype
    #    instantiations whose parent component has not been set. This
    #    prevents the current container from "stealing" the component
    #    from a different container, leaving the old container still
    #    thinking all components it contains have their weakref
    #    pointing back to itself.
    #
    #    However, it is not clear that this would actually be an
    #    issue.  For instance, this would seem like a very effective
    #    way of "aliasing" a variable. That is, this component would
    #    only set the parent component on an incoming object
    #    assignment if that object was not already assigned a
    #    parent. All of the problem writers would not be effected by
    #    this because they generate the symbol map by hashing off of
    #    the object id(). This would also seamlessly integrate into
    #    the Suffix API as that component hashes of off id() as well.
    #
    #    Again, I would rather start with more limited functionality
    #    that we could all, without question, agree on, so this is
    #    something to consider later on.
    #
    def __setitem__(self, key, val):
        if isinstance(val, self._interface_datatype):
            if val._component is None:
                val._component = weakref_ref(self)
                if hasattr(self, "_active"):
                    self._active |= getattr(val, '_active', True)
                if key in self._data:
                    # release the current component
                    # * see __delitem__ for explanation
                    self._data[key]._component = None
                self._data[key] = val
                return
            elif (key in self._data) and (self._data[key] is val):
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
                "Invalid component object assignment to ComponentDict "
                "%s at key %s. A parent component has already been "
                "assigned the object: %s"
                % (self.name,
                   key,
                   val.parent_component().name))
        # see note about implicit assignment and update
        raise TypeError(
            "ComponentDict must be assigned objects "
            "of type %s. Invalid type for key %s: %s"
            % (self._interface_datatype.__name__,
               key,
               type(val)))

    # Since we don't currently allow objects to be assigned when their
    # parent component is already set, it would make sense to reset
    # the parent component back to None for an object being removed
    # from this container; thus, allowing objects to be naturally
    # transferred between containers if one so chooses. This would be
    # more complicated if allowing components to live in more than one
    # container (see note above __setitem__), and maybe this is the
    # reason not to support that.
    def __delitem__(self, key):
        obj = self._data[key]
        obj._component = None
        del self._data[key]

    def __getitem__(self, key): return self._data[key]
    def __iter__(self): return self._data.__iter__()
    def __len__(self): return self._data.__len__()

    #
    # Override a few default implementations on MutableMapping
    #

    # We want to avoid generating Pyomo expressions due to
    # comparison of values; thus, we convert this to a plain
    # dictionary mapping key->(type(val), id(val)) and
    # compare that instead.
    def __eq__(self, other):
        if not isinstance(other, collections.Mapping):
            return False
        return dict((key, (type(val), id(val)))
                    for key,val in self.items()) == \
               dict((key, (type(val), id(val)))
                    for key,val in other.items())
    def __ne__(self, other):
        return not (self == other)

#
# ComponentDict needs to come before IndexedComponent
# (or subclasses of) so we can override certain methods
#

class VarDict(ComponentDict, IndexedVar):

    def __init__(self, *args, **kwds):
        IndexedVar.__init__(self, Any, **kwds)
        # Constructor for ComponentDict needs to
        # go last in order to handle any initialization
        # iterable as an argument
        ComponentDict.__init__(self,
                               _VarData,
                               *args,
                               **kwds)

class ConstraintDict(ComponentDict, IndexedConstraint):

    def __init__(self, *args, **kwds):
        IndexedConstraint.__init__(self, Any, **kwds)
        # Constructor for ComponentDict needs to
        # go last in order to handle any initialization
        # iterable as an argument
        ComponentDict.__init__(self,
                               _ConstraintData,
                               *args,
                               **kwds)

class ObjectiveDict(ComponentDict, IndexedObjective):

    def __init__(self, *args, **kwds):
        IndexedObjective.__init__(self, Any, **kwds)
        # Constructor for ComponentDict needs to
        # go last in order to handle any initialization
        # iterable as an argument
        ComponentDict.__init__(self,
                               _ObjectiveData,
                               *args,
                               **kwds)

class ExpressionDict(ComponentDict, IndexedExpression):

    def __init__(self, *args, **kwds):
        IndexedExpression.__init__(self, Any, **kwds)
        # Constructor for ComponentDict needs to
        # go last in order to handle any initialization
        # iterable as an argument
        ComponentDict.__init__(self,
                               _ExpressionData,
                               *args,
                               **kwds)
