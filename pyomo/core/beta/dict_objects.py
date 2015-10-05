
__all__ = () #'ExpressionDict','ConstraintDict','ObjectiveDict')

import logging
from weakref import ref as weakref_ref
import collections

from pyomo.core.base.set_types import Any
from pyomo.core.base.constraint import (IndexedConstraint,
                                        _ConstraintData,
                                        _GeneralConstraintData)
from pyomo.core.base.objective import (IndexedObjective,
                                       _ObjectiveData,
                                       _GeneralObjectiveData)
from pyomo.core.base.expression import (IndexedExpression,
                                        _ExpressionData,
                                        _GeneralExpressionData)

logger = logging.getLogger('pyomo.core')

#
# In the future I think _ComponentDict and _ComponentList should
# inheret directly from (Active)IndexedComponent, and that class
# should be stripped down to a minimal interface.
#

class _ComponentDict(collections.MutableMapping):

    def __init__(self, interface_datatype, default_datatype, *args):
        self._interface_datatype = interface_datatype
        self._default_datatype = default_datatype
        assert issubclass(self._default_datatype,
                          self._interface_datatype)
        self._data = {}
        if len(args) > 0:
            if len(args) > 1:
                raise TypeError(
                    "_ComponentDict expected at most 1 arguments, "
                    "got %s" % (len(args)))
            self.update(args[0])

    def construct(self, data=None):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Constructing _ComponentDict object, name=%s, from data=%s"
                % (self.cname(True), str(data)))
        if self._constructed:
            return
        self._constructed = True

    #
    # Define the MutableMapping abstract methods
    #

    # Currently this method allows implicit instantiation of a
    # component object if the key is missing and assigned something
    # compatible with the component type's constructor.  Additionally,
    # if the key is already present and assignment is in the form of
    # an explicitly instantiated component type, then the current
    # component object at that key will be overwritten rather than
    # updated.
    #
    # It is worth considering disallowing the implicit behavior.
    def __setitem__(self, key, val):
        if not isinstance(val, self._interface_datatype):
            if key in self._data:
                obj = self._data[key]
            else:
                obj = self._default_datatype(None, component=self)
                self._data[key] = obj
                self._active |= getattr(obj, 'active', True)
            obj.set_value(val)
        else:
            val._component = weakref_ref(self)
            self._active |= getattr(val, '_active', True)
            self._data[key] = val
            assert val.parent_component() is self

    def __getitem__(self, key): return self._data[key]
    def __delitem__(self, key): del self._data[key]
    def __iter__(self): return self._data.__iter__()
    def __len__(self): return self._data.__len__()

    #
    # Override a few default implementations on MutableMapping
    #

    # We want to avoid generating Pyomo expressions by comparing values
    # (we could perhaps just return id(self))
    def __eq__(self, other): raise NotImplementedError("_ComponentDict: undefined operation __eq__")

    # The default implementation is slow
    def clear(self):
        'D.clear() -> None.  Remove all items from D.'
        self._data.clear()

#
# _ComponentDict needs to come before IndexedComponent
# (derivatives) so we can override certain methods
#

class ConstraintDict(_ComponentDict, IndexedConstraint):

    def __init__(self, *args, **kwds):
        IndexedConstraint.__init__(self,
                                   Any,
                                   **kwds)
        # Constructor for _ComponentDict needs to
        # go last in order to handle any initialization
        # iterable as an argument
        _ComponentDict.__init__(self,
                                _ConstraintData,
                                _GeneralConstraintData,
                                *args,
                                **kwds)

class ObjectiveDict(_ComponentDict, IndexedObjective):

    def __init__(self, *args, **kwds):
        IndexedObjective.__init__(self,
                                  Any,
                                  **kwds)
        # Constructor for _ComponentDict needs to
        # go last in order to handle any initialization
        # iterable as an argument
        _ComponentDict.__init__(self,
                                _ObjectiveData,
                                _GeneralObjectiveData,
                                *args,
                                **kwds)

class ExpressionDict(_ComponentDict, IndexedExpression):

    def __init__(self, *args, **kwds):
        IndexedExpression.__init__(self, Any)
        # Constructor for _ComponentDict needs to
        # go last in order to handle any initialization
        # iterable as an argument
        _ComponentDict.__init__(self,
                                _ExpressionData,
                                _GeneralExpressionData,
                                *args,
                                **kwds)
