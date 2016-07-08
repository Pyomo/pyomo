#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = [ 'Connector' ]

import itertools
import logging
import weakref
import sys
from six import iteritems, itervalues
from six.moves import xrange
from weakref import ref as weakref_ref

from pyomo.util.plugin import Plugin, implements

from pyomo.core.base.component import Component, ComponentData, register_component
from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.expr import _ProductExpression
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule, create_name
from pyomo.core.base.numvalue import NumericValue
from pyomo.core.base.plugin import IPyomoScriptModifyInstance
from pyomo.core.base.var import Var, VarList

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
        # The semantics here are not clear, and given how aggressive
        # Constraint.add() is at simplifying expressions, returning True
        # has undesirable effects.
        # TODO: revisit this after refactoring Constraint.add()
        return False

        if len(self.vars) == 0:
            return False
        for var in itervalues(self.vars):
            if not var.is_fixed():
                return False
        return True

    def is_constant(self):
        # The semantics here are not clear, and given how aggressive
        # Constraint.add() is at simplifying expressions, returning True
        # has undesirable effects.
        # TODO: revisit this after refactoring Constraint.add()
        return False

        for var in itervalues(self.vars):
            if not var.is_constant():
                return False
        return True

    def polynomial_degree(self):
        if self.is_fixed():
            return 0
        return 1

    def is_binary(self):
        for var in itervalues(self.vars):
            if var.is_binary():
                return True
        return False

    def is_integer(self):
        for var in itervalues(self.vars):
            if var.is_integer():
                return True
        return False

    def is_continuous(self):
        for var in itervalues(self.vars):
            if var.is_continuous():
                return True
        return False


    def add(self, var, name=None, aggregate=None):
        if name is None:
            name = var.cname()
        if name in self.vars:
            raise ValueError(
                "Cannot insert duplicate variable name "
                "'%s' into Connector '%s'" % (name, self.cname()) )
        self.vars[name] = var
        if aggregate is not None:
            self.aggregators[var] = aggregate


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

    def as_numeric(self):
        if None in self._conval:
            return self._conval[None]
        return self

    def is_expression(self):
        return False

    def is_relational(self):
        return False

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
            for _k, _v in iteritems(v.vars):
                _len = 1 if _v.is_expression() else len(_v)
                yield _k, _len, str(_v)
        return ( [("Size", len(self)),
                  ("Index", self._index \
                       if self._index != UnindexedComponent_set else None),
                  ],
                 iteritems(self._data),
                 ( "Name","Size", "Variable", ),
                 _line_generator
             )


    def display(self, prefix="", ostream=None):
        if ostream is None:
            ostream = sys.stdout
        ostream.write(prefix+"Connector "+self.name+" :")
        ostream.write("  Size="+str(len(self)))
        if None in self._data:
            ostream.write(prefix+"  : {"+\
                ', '.join(sorted(self._data[None].keys()))+"}"+'\n')
        else:
            for key in sorted(self._data.keys()):
                ostream.write(prefix+"  "+str(key)+" : {"+\
                  ', '.join(sorted(self._data[key].keys()))+"}"+'\n')


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
    pass


register_component(Connector, "A bundle of variables that can be manipilated together.")


class ConnectorExpander(Plugin):
    implements(IPyomoScriptModifyInstance)

    def apply(self, **kwds):
        instance = kwds.pop('instance')
        xform = TransformationFactory('core.expand_connectors')
        xform.apply_to(instance, **kwds)
        return instance

transform = ConnectorExpander()
