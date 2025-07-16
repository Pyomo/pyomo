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
import sys
from weakref import ref as weakref_ref

from pyomo.common.deprecation import deprecated, RenamedClass
from pyomo.common.formatting import tabular_writer
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.numeric_types import value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import NumericValue
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.misc import apply_indexed_rule

logger = logging.getLogger('pyomo.core')


class ConnectorData(ComponentData, NumericValue):
    """Holds the actual connector information"""

    __slots__ = ('vars', 'aggregators')

    def __init__(self, component=None):
        """Constructor"""
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - ComponentData
        #   - NumericValue
        self._component = weakref_ref(component) if (component is not None) else None
        self._index = NOTSET

        self.vars = {}
        self.aggregators = {}

    def set_value(self, value):
        msg = "Cannot specify the value of a connector '%s'"
        raise ValueError(msg % self.name)

    def is_fixed(self):
        """Return True if all vars/expressions in the Connector are fixed"""
        return all(v.is_fixed() for v in self._iter_vars())

    def is_constant(self):
        """Return False

        Because the expression generation logic will attempt to evaluate
        constant subexpressions, a Connector can never be constant.
        """
        return False

    def is_potentially_variable(self):
        """Return True as connectors may (should!) contain variables"""
        return True

    def polynomial_degree(self):
        ans = 0
        for v in self._iter_vars():
            tmp = v.polynomial_degree()
            if tmp is None:
                return None
            ans = max(ans, tmp)
        return ans

    def is_binary(self):
        return len(self) and all(v.is_binary() for v in self._iter_vars())

    def is_integer(self):
        return len(self) and all(v.is_integer() for v in self._iter_vars())

    def is_continuous(self):
        return len(self) and all(v.is_continuous() for v in self._iter_vars())

    def add(self, var, name=None, aggregate=None):
        if name is None:
            name = var.local_name
        if name in self.vars:
            raise ValueError(
                "Cannot insert duplicate variable name "
                "'%s' into Connector '%s'" % (name, self.name)
            )
        self.vars[name] = var
        if aggregate is not None:
            self.aggregators[name] = aggregate

    def _iter_vars(self):
        for var in self.vars.values():
            if not hasattr(var, 'is_indexed') or not var.is_indexed():
                yield var
            else:
                for v in var.values():
                    yield v


class _ConnectorData(metaclass=RenamedClass):
    __renamed__new_class__ = ConnectorData
    __renamed__version__ = '6.7.2'


@ModelComponentFactory.register(
    "A bundle of variables that can be manipulated together."
)
@deprecated(
    "Use of pyomo.connectors is deprecated. "
    "Its functionality has been replaced by pyomo.network.",
    version='5.6.9',
)
class Connector(IndexedComponent):
    """A collection of variables, which may be defined over a index

    The idea behind a Connector is to create a bundle of variables that
    can be manipulated as a single variable within constraints.  While
    Connectors inherit from variable (mostly so that the expression
    infrastructure can manipulate them), they are not actual variables
    that are exposed to the solver.  Instead, a preprocessor
    (ConnectorExpander) will look for expressions that involve
    connectors and replace the single constraint with a list of
    constraints that involve the original variables contained within the
    Connector.

    Parameters
    ----------
    name : str
        The name of this connector

    index
        The index set that defines the distinct connectors.  By default,
        this is None, indicating that there is a single connector.

    """

    def __new__(cls, *args, **kwds):
        if cls != Connector:
            return super(Connector, cls).__new__(cls)
        if args == ():
            return ScalarConnector.__new__(ScalarConnector)
        else:
            return IndexedConnector.__new__(IndexedConnector)

    # TODO: default keyword is  not used?  Need to talk to Bill ...?
    def __init__(self, *args, **kwd):
        kwd.setdefault('ctype', Connector)
        self._rule = kwd.pop('rule', None)
        self._initialize = kwd.pop('initialize', {})
        self._implicit = kwd.pop('implicit', {})
        self._extends = kwd.pop('extends', None)
        IndexedComponent.__init__(self, *args, **kwd)
        self._conval = {}

    #
    # This method must be defined on subclasses of
    # IndexedComponent
    #
    def _getitem_when_not_present(self, idx):
        _conval = self._data[idx] = ConnectorData(component=self)
        return _conval

    def construct(self, data=None):
        if is_debug_set(logger):  # pragma:nocover
            logger.debug(
                "Constructing Connector, name=%s, from data=%s" % (self.name, data)
            )
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed = True
        #
        # Construct ConnectorData objects for all index values
        #
        if self.is_indexed():
            self._initialize_members(self._index_set)
        else:
            self._data[None] = self
            self._initialize_members([None])
        timer.report()

    def _initialize_members(self, initSet):
        for idx in initSet:
            tmp = self[idx]
            for key in self._implicit:
                tmp.add(None, key)
            if self._extends:
                for key, val in self._extends.vars.items():
                    tmp.add(val, key)
            for key, val in self._initialize.items():
                tmp.add(val, key)
            if self._rule:
                items = apply_indexed_rule(self, self._rule, self._parent(), idx)
                for key, val in items.items():
                    tmp.add(val, key)

    def _pprint(self, ostream=None, verbose=False):
        """Print component information."""

        def _line_generator(k, v):
            for _k, _v in sorted(v.vars.items()):
                if _v is None:
                    _len = '-'
                elif _k in v.aggregators:
                    _len = '*'
                elif hasattr(_v, '__len__'):
                    _len = len(_v)
                else:
                    _len = 1
                yield _k, _len, str(_v)

        return (
            [
                ("Size", len(self)),
                ("Index", self._index_set if self.is_indexed() else None),
            ],
            self._data.items(),
            ("Name", "Size", "Variable"),
            _line_generator,
        )

    def display(self, prefix="", ostream=None):
        """
        Print component state information

        This duplicates logic in Component.pprint()
        """
        if not self.active:
            return
        if ostream is None:
            ostream = sys.stdout
        tab = "    "
        ostream.write(prefix + self.local_name + " : ")
        ostream.write("Size=" + str(len(self)))

        ostream.write("\n")

        def _line_generator(k, v):
            for _k, _v in sorted(v.vars.items()):
                if _v is None:
                    _val = '-'
                elif not hasattr(_v, 'is_indexed') or not _v.is_indexed():
                    _val = str(value(_v))
                else:
                    _val = "{%s}" % (
                        ', '.join(
                            '%r: %r' % (x, value(_v[x])) for x in sorted(_v._data)
                        ),
                    )
                yield _k, _val

        tabular_writer(
            ostream,
            prefix + tab,
            ((k, v) for k, v in self._data.items()),
            ("Name", "Value"),
            _line_generator,
        )


class ScalarConnector(Connector, ConnectorData):
    def __init__(self, *args, **kwd):
        ConnectorData.__init__(self, component=self)
        Connector.__init__(self, *args, **kwd)
        self._index = UnindexedComponent_index


class SimpleConnector(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarConnector
    __renamed__version__ = '6.0'


class IndexedConnector(Connector):
    """An array of connectors"""

    pass
