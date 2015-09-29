
__all__ = ('ExpressionDict','ConstraintDict','ObjectiveDict')

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

class _DictComponent(collections.MutableMapping, collections.Hashable):

    def __hash__(self):
        return id(self)

    def __init__(self, interface_datatype, default_datatype, *args, **kwds):
        self._interface_datatype = interface_datatype
        self._default_datatype = default_datatype
        assert issubclass(self._default_datatype,
                          self._interface_datatype)

        if len(args) > 0:
            if len(args) > 1:
                raise TypeError(
                    "_DictComponent expected at most 1 arguments, "
                    "got %s" % (len(args)))
            self.update(args[0])

    def construct(self, data=None):
        """ Apply the rule to construct values in this set """

        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Constructing _DictObject, name=%s, from data=%s"
                % (self.cname(True), str(data)))

        if self._constructed:
            return
        self._constructed = True

    #
    # MutableMapping Abstract Methods
    #

    def __getitem__(self, key):
        return self._data[key]

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

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return self._data.__len__()

    #
    # Overload MutableMapping default implementations
    #

    def __eq__(self, other):
        raise NotImplementedError(
            "_ComponentDict is not comparable")

    def __ne__(self, other):
        raise NotImplementedError(
            "_ComponentDict is not comparable")

    #
    # The remaining methods have slow default implementations for
    # MutableMapping. In particular, they rely KeyError catching
    #
    """
    def __contains__(self, key)
        return key in self._data

    def clear(self):
        'D.clear() -> None.  Remove all items from D.'
        self._data.clear()

    def get(self, key, default=None):
        'D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.'
        if key in self:
            return self[key]
        return default

    def setdefault(self, key, default=None):
        'D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D'
        if key in self:
            return self[key]
        else:
            self[key] = default
        return default
    """

#
# _DictComponent needs to come before IndexedComponent
# (derivatives) so we can override certain methods
#

class ConstraintDict(_DictComponent, IndexedConstraint):

    def __init__(self, *args, **kwds):
        IndexedConstraint.__init__(self,
                                   Any,
                                   **kwds)
        # Constructor for _DictComponent needs to
        # go last in order to handle any initialization
        # iterable as an argument
        _DictComponent.__init__(self,
                                _ConstraintData,
                                _GeneralConstraintData,
                                *args,
                                **kwds)

class ObjectiveDict(_DictComponent, IndexedObjective):

    def __init__(self, *args, **kwds):
        IndexedObjective.__init__(self,
                                  Any,
                                  **kwds)
        # Constructor for _DictComponent needs to
        # go last in order to handle any initialization
        # iterable as an argument
        _DictComponent.__init__(self,
                                _ObjectiveData,
                                _GeneralObjectiveData,
                                *args,
                                **kwds)

class ExpressionDict(_DictComponent, IndexedExpression):

    def __init__(self, *args, **kwds):
        IndexedExpression.__init__(self, Any)
        # Constructor for _DictComponent needs to
        # go last in order to handle any initialization
        # iterable as an argument
        _DictComponent.__init__(self,
                                _ExpressionData,
                                _GeneralExpressionData,
                                *args,
                                **kwds)
