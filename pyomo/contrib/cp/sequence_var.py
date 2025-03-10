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

import logging

from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.contrib.cp import IntervalVar
from pyomo.core import ModelComponentFactory
from pyomo.core.base.component import ActiveComponentData
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import ActiveIndexedComponent
from pyomo.core.base.initializer import Initializer

import sys
from weakref import ref as weakref_ref

logger = logging.getLogger(__name__)


class SequenceVarData(ActiveComponentData):
    """This class defines the abstract interface for a single sequence variable."""

    __slots__ = ('interval_vars',)

    def __init__(self, component=None):
        # in-lining ActiveComponentData and ComponentData constructors, as is
        # traditional:
        self._component = weakref_ref(component) if (component is not None) else None
        self._index = NOTSET
        self._active = True

        # This thing is really just an ordered set of interval vars that we can
        # write constraints over.
        self.interval_vars = []

    def set_value(self, expr):
        # We'll demand expr be a list for now--it needs to be ordered so this
        # doesn't seem like too much to ask
        if not hasattr(expr, '__iter__'):
            raise ValueError(
                "'expr' for SequenceVar must be a list of IntervalVars. "
                "Encountered type '%s' constructing '%s'" % (type(expr), self.name)
            )
        for v in expr:
            if not hasattr(v, 'ctype') or v.ctype is not IntervalVar:
                raise ValueError(
                    "The SequenceVar 'expr' argument must be a list of "
                    "IntervalVars. The 'expr' for SequenceVar '%s' included "
                    "an object of type '%s'" % (self.name, type(v))
                )
            self.interval_vars.append(v)


@ModelComponentFactory.register("Sequences of IntervalVars")
class SequenceVar(ActiveIndexedComponent):
    _ComponentDataClass = SequenceVarData

    def __new__(cls, *args, **kwds):
        if cls != SequenceVar:
            return super(SequenceVar, cls).__new__(cls)
        if args == ():
            return ScalarSequenceVar.__new__(ScalarSequenceVar)
        else:
            return IndexedSequenceVar.__new__(IndexedSequenceVar)

    def __init__(self, *args, **kwargs):
        self._init_rule = Initializer(kwargs.pop('rule', None))
        self._init_expr = kwargs.pop('expr', None)
        kwargs.setdefault('ctype', SequenceVar)
        super(SequenceVar, self).__init__(*args, **kwargs)

        if self._init_expr is not None and self._init_rule is not None:
            raise ValueError(
                "Cannot specify both rule= and expr= for SequenceVar %s" % (self.name,)
            )

    def _getitem_when_not_present(self, index):
        if index is None and not self.is_indexed():
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(component=self)
        parent = self.parent_block()
        obj._index = index

        if self._init_rule is not None:
            obj.set_value(self._init_rule(parent, index))
        if self._init_expr is not None:
            obj.set_value(self._init_expr)

        return obj

    def construct(self, data=None):
        """
        Construct the SequenceVarData objects for this SequenceVar
        """
        if self._constructed:
            return
        self._constructed = True

        if is_debug_set(logger):
            logger.debug("Constructing SequenceVar %s" % self.name)

        # Initialize index in case we hit the exception below
        index = None
        try:
            if not self.is_indexed():
                self._getitem_when_not_present(None)
            if self._init_rule is not None:
                for index in self.index_set():
                    self._getitem_when_not_present(index)
        except Exception:
            err = sys.exc_info()[1]
            logger.error(
                "Rule failed when initializing sequence variable for "
                "SequenceVar %s with index %s:\n%s: %s"
                % (self.name, str(index), type(err).__name__, err)
            )
            raise

    def _pprint(self):
        """Print component information."""
        headers = [
            ("Size", len(self)),
            ("Index", self._index_set if self.is_indexed() else None),
        ]
        return (
            headers,
            self._data.items(),
            ("IntervalVars",),
            lambda k, v: ['[' + ', '.join(iv.name for iv in v.interval_vars) + ']'],
        )


class ScalarSequenceVar(SequenceVarData, SequenceVar):
    def __init__(self, *args, **kwds):
        SequenceVarData.__init__(self, component=self)
        SequenceVar.__init__(self, *args, **kwds)
        self._index = UnindexedComponent_index


class IndexedSequenceVar(SequenceVar):
    pass
