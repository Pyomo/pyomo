#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = [ 'Port' ]

import logging
import sys
from six import iteritems, itervalues, iterkeys
from six.moves import xrange
from weakref import ref as weakref_ref

from pyomo.common.timing import ConstructionTimer
from pyomo.common.plugin import Plugin, implements

from pyomo.core.base.var import VarList
from pyomo.core.base.component import ComponentData
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.misc import apply_indexed_rule, tabular_writer
from pyomo.core.base.numvalue import NumericValue, value
from pyomo.core.base.plugin import register_component, \
    IPyomoScriptModifyInstance, TransformationFactory

logger = logging.getLogger('pyomo.network')


class _PortData(ComponentData, NumericValue):
    """This class defines the data for a single Port."""

    __slots__ = ('vars', 'aggregators', 'extensives', 'extensive_aggregators')

    def __init__(self, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - ComponentData
        #   - NumericValue
        self._component = weakref_ref(component) if (component is not None) \
                          else None

        self.vars = {}
        self.aggregators = {}
        self.extensives = {}
        # default aggregation functions for extensive variables in arcs
        # can't import at top of file because of a circular import
        from pyomo.network.arc import Arc
        self.extensive_aggregators = {"split" : Arc.SplitFrac,
                                      "mix"   : Arc.Balance}

    def __getstate__(self):
        state = super(_PortData, self).__getstate__()
        for i in _PortData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: None of the slots on this class need to be edited, so we
    # don't need to implement a specialized __setstate__ method, and
    # can quietly rely on the super() class's implementation.

    def __getattr__(self, name):
        """Returns self.vars[name] if it exists"""
        if name in self.vars:
            return self.vars[name]
        # Since the base classes don't support getattr, we can just
        # throw the "normal" AttributeError
        raise AttributeError("'%s' object has no attribute '%s'"
                             % (self.__class__.__name__, name))

    def set_value(self, value):
        msg = "Cannot specify the value of a port '%s'"
        raise ValueError(msg % self.name)

    def is_fixed(self):
        """Return True if all vars/expressions in the Port are fixed"""
        return all(v.is_fixed() for v in self._iter_vars())

    def is_constant(self):
        """Return False

        Because the expression generation logic will attempt to evaluate
        constant subexpressions, a Port can never be constant.
        """
        return False

    def is_potentially_variable(self):
        """Return True as ports may (should!) contain variables"""
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

    def add(self, var, name=None, aggregate=None, extensive=None):
        if name is None:
            name = var.local_name
        if name in self.vars:
            raise ValueError("Cannot insert duplicate variable name "
                             "'%s' into Port '%s'" % (name, self.name))
        self.vars[name] = var
        if aggregate is not None:
            if extensive is not None:
                raise ValueError(
                    "Cannot specify aggregator for extensive variable '%s' on "
                    "Port '%s'" % (name, self.name))
            if type(var) is not VarList:
                raise ValueError(
                    "Aggregated variable '%s' must be a VarList "
                    "in Port '%s'" % (name, self.name))
            self.aggregators[name] = aggregate
        elif extensive is not None:
            # avoid name collisions
            if name.endswith("_split") or name.endswith("_equality"):
                raise ValueError(
                    "Extensive variable '%s' on Port '%s' may not end "
                    "with '_split' or '_equality'" % (name, self.name))
            if extensive not in self.extensives:
                # initialize new dict if this is the first of its kind
                self.extensives[extensive] = {}
            self.extensives[extensive][name] = []

    def _iter_vars(self):
        for var in itervalues(self.vars):
            if not hasattr(var, 'is_indexed') or not var.is_indexed():
                yield var
            else:
                for v in itervalues(var):
                    yield v


class Port(IndexedComponent):
    """
    A collection of variables, which may be connected to other ports.

    The idea behind Ports is to create a bundle of variables that can
    be manipulated together by connecting them to other ports via Arcs.
    A preprocess transformation will look for Arcs and expand them into
    a series of constraints that involve the original variables contained
    within the Port. The way these constraints are built can be specified
    for each Port member when adding things to the port, but by default
    the Port members will be equated to each other. Additionally, other
    objects such as expressions can be added to Ports as long as they, or
    their indexed members, can be manipulated within constraint expressions.

    Constructor arguments:
        rule            A function that returns a dictionary of name, var
                            pairs to be initially added to the Port
        doc             A text string describing this component
        name            A name for this component

    Public attributes
        vars            A dictionary mapping added names to variables
    """

    def __new__(cls, *args, **kwds):
        if cls != Port:
            return super(Port, cls).__new__(cls)
        if args == ():
            return SimplePort.__new__(SimplePort)
        else:
            return IndexedPort.__new__(IndexedPort)


    def __init__(self, *args, **kwd):
        kwd.setdefault('ctype', Port)
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
        _conval = self._data[idx] = _PortData(component=self)
        return _conval


    def construct(self, data=None):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):  #pragma:nocover
            logger.debug( "Constructing Port, name=%s, from data=%s"
                          % (self.name, data) )
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed=True
        #
        # Construct _PortData objects for all index values
        #
        if self.is_indexed():
            self._initialize_members(self._index)
        else:
            self._data[None] = self
            self._initialize_members([None])
        timer.report()

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
                elif hasattr(_v,'__len__'):
                    _len = len(_v)
                else:
                    _len = 1
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
                elif not hasattr(_v, 'is_indexed') or not _v.is_indexed():
                    _val = str(value( _v ))
                else:
                    _val = "{%s}" % (', '.join('%r: %r' % (
                        x, value(_v[x])) for x in sorted(_v._data) ),)
                yield _k, _val
        tabular_writer( ostream, prefix+tab,
                        ((k,v) for k,v in iteritems(self._data)),
                        ( "Name","Value" ), _line_generator )


class SimplePort(Port, _PortData):

    def __init__(self, *args, **kwd):
        _PortData.__init__(self, component=self)
        Port.__init__(self, *args, **kwd)


class IndexedPort(Port):
    pass


register_component(
    Port, "A bundle of variables that can be connected to other ports.")


class PortExpander(Plugin):
    implements(IPyomoScriptModifyInstance)

    def apply(self, **kwds):
        instance = kwds.pop('instance')
        xform = TransformationFactory('network.expand_ports')
        xform.apply_to(instance, **kwds)
        return instance

transform = PortExpander()
