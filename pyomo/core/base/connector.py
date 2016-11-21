#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = [ 'Connector' ]

import logging
import sys
from six import iteritems, itervalues, iterkeys
from six.moves import xrange
from weakref import ref as weakref_ref

from pyomo.util.plugin import Plugin, implements

from pyomo.core.base.component import ComponentData, register_component
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.misc import apply_indexed_rule, tabular_writer
from pyomo.core.base.numvalue import NumericValue, value
from pyomo.core.base.plugin import IPyomoScriptModifyInstance, TransformationFactory

logger = logging.getLogger('pyomo.core')


class _ConnectorData(ComponentData, NumericValue):
    """Holds the actual connector information"""

    __slots__ = ('vars','aggregators')

    def __init__(self, component=None):
        """Constructor"""
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - ComponentData
        #   - NumericValue
        self._component = weakref_ref(component) if (component is not None) \
                          else None

        self.vars = {}
        self.aggregators = {}
    

    def __getstate__(self):
        state = super(_ConnectorData, self).__getstate__()
        for i in _ConnectorData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: None of the slots on this class need to be edited, so we
    # don't need to implement a specialized __setstate__ method, and
    # can quietly rely on the super() class's implementation.

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

    def _potentially_variable(self):
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
            raise ValueError("Cannot insert duplicate variable name "
                             "'%s' into Connector '%s'" % (name, self.name))
        self.vars[name] = var
        if aggregate is not None:
            self.aggregators[name] = aggregate


    def _iter_vars(self):
        for var in itervalues(self.vars):
            if var.is_expression() or not var.is_indexed():
                yield var
            else:
                for v in itervalues(var):
                    yield v



class Connector(IndexedComponent):
    """A collection of variables, which may be defined over a index

    The idea behind a Connector is to create a bundle of variables that
    can be manipulated as a single variable within constraints.  While
    Connectors inherit from variable (mostly so that the expression
    infrastucture can manipulate them), they are not actual variables
    that are exposed to the solver.  Instead, a preprocessor
    (ConnectorExpander) will look for expressions that involve
    connectors and replace the single constraint with a list of
    constraints that involve the original variables contained within the
    Connector.

    Constructor
        Arguments:
           name         The name of this connector
           index        The index set that defines the distinct connectors.
                          By default, this is None, indicating that there
                          is a single connector.
    """

    def __new__(cls, *args, **kwds):
        if cls != Connector:
            return super(Connector, cls).__new__(cls)
        if args == ():
            return SimpleConnector.__new__(SimpleConnector)
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
    def _default(self, idx):
        _conval = self._data[idx] = _ConnectorData(component=self)
        return _conval


    def construct(self, data=None):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):  #pragma:nocover
            logger.debug( "Constructing Connector, name=%s, from data=%s"
                          % (self.name, data) )
        if self._constructed:
            return
        self._constructed=True
        #
        # Construct _ConnectorData objects for all index values
        #
        if self.is_indexed():
            self._initialize_members(self._index)
        else:
            self._data[None] = self
            self._initialize_members([None])

    def _initialize_members(self, initSet):
        for idx in initSet:
            tmp = self[idx]
            for key in self._implicit:
                tmp.add(None,key)
            if self._extends:
                for key, val in iteritems(self._extends.vars):
                    tmp.add(val,key)
            for key, val in iteritems(self._initialize):
                tmp.add(val,key)
            if self._rule:
                items = apply_indexed_rule(
                    self, self._rule, self._parent(), idx)
                for key, val in iteritems(items):
                    tmp.add(val,key)


    def _pprint(self, ostream=None, verbose=False):
        """Print component information."""
        def _line_generator(k,v):
            for _k, _v in sorted(iteritems(v.vars)):
                if _v is None:
                    _len = '-'
                elif _k in v.aggregators:
                    _len = '*'
                elif _v.is_expression() or not _v.is_indexed():
                    _len = 1
                else:
                    _len = len(_v)
                yield _k, _len, str(_v)
        return ( [("Size", len(self)),
                  ("Index", self._index if self.is_indexed() else None),
                  ],
                 iteritems(self._data),
                 ( "Name","Size", "Variable", ),
                 _line_generator
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
        tab="    "
        ostream.write(prefix+self.local_name+" : ")
        ostream.write("Size="+str(len(self)))

        ostream.write("\n")
        def _line_generator(k,v):
            for _k, _v in sorted(iteritems(v.vars)):
                if _v is None:
                    _val = '-'
                elif _v.is_expression() or not _v.is_indexed():
                    _val = str(value( _v ))
                else:
                    _val = "{%s}" % (', '.join('%r: %r' % (
                        x, value(_v[x])) for x in sorted(_v._data) ),)
                yield _k, _val
        tabular_writer( ostream, prefix+tab,
                        ((k,v) for k,v in iteritems(self._data)),
                        ( "Name","Value" ), _line_generator )


class SimpleConnector(Connector, _ConnectorData):

    def __init__(self, *args, **kwd):
        _ConnectorData.__init__(self, component=self)
        Connector.__init__(self, *args, **kwd)

    #
    # Since this class derives from Component and Component.__getstate__
    # just packs up the entire __dict__ into the state dict, we do not
    # need to define the __getstate__ or __setstate__ methods.
    # We just defer to the super() get/set state.  Since all of our
    # get/set state methods rely on super() to traverse the MRO, this
    # will automatically pick up both the Component and Data base classes.
    #


class IndexedConnector(Connector):
    """An array of connectors"""

    # These methods are normally found on the NumericValue
    # interface, but they are here to trick the expression
    # system into reporting errors about trying to use
    # "indexed NumericValue" objects in expressions
    def as_numeric(self): return self
    def is_expression(self): return False
    def is_relational(self): return False


register_component(
    Connector, "A bundle of variables that can be manipilated together.")


class ConnectorExpander(Plugin):
    implements(IPyomoScriptModifyInstance)

    def apply(self, **kwds):
        instance = kwds.pop('instance')
        xform = TransformationFactory('core.expand_connectors')
        xform.apply_to(instance, **kwds)
        return instance

transform = ConnectorExpander()
