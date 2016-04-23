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

from pyomo.core.base.component import Component, register_component
from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.expr import _ProductExpression
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule, create_name
from pyomo.core.base.numvalue import NumericValue
from pyomo.core.base.plugin import IPyomoScriptModifyInstance
from pyomo.core.base.var import Var, VarList

logger = logging.getLogger('pyomo.core')


class _ConnectorData(NumericValue):
    """Holds the actual connector information"""

    __slots__ = ('connector','index','vars','aggregators')

    def __init__(self, name):
        """Constructor"""

        # IMPT: The following three lines are equivalent to calling the
        #       basic NumericValue constructor, i.e., as follows:
        #       NumericValue.__init__(self, name, domain, None, False)
        #       That particular constructor call takes a lot of time
        #       for big models, and is unnecessary because we're not
        #       validating any values.
        self.name = name
        self.domain = None
        self.value = None

        # NOTE: the "name" attribute (part of the base NumericValue class) is
        #       typically something like some_var[x,y,z] - an easy-to-read
        #       representation of the variable/index pair.

        # NOTE: both of the following are presently set by the parent. arguably, we
        #       should provide keywords to streamline initialization.
        self.connector = None # the "parent" variable.
        self.index = None # the index of this variable within the "parent"
        self.vars = {}
        self.aggregators = {}
    
    def __str__(self):
        # the name can be None, in which case simply return "".
        if self.name is None:
            return ""
        else:
            return self.name

    def __getstate__(self):
        result = NumericValue.__getstate__(self)
        for i in _ConnectorData.__slots__:
            result[i] = getattr(self, i)
        if type(result['connector']) is weakref.ref:
            result['connector'] = result['connector']()
        return result

    def __setstate__(self, state):
        for (slot_name, value) in iteritems(state):
            self.__dict__[slot_name] = value
        if 'connector' in self.__dict__ and self.connector is not None:
            self.connector = weakref.ref(self.connector)

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
            name = var.name
        if name in self.vars:
            raise ValueError("Cannot insert duplicate variable name "
                             "'%s' into Connector '%s'" % ( name, self.name ))
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
        self._initialize = kwd.pop('initialize', None)
        self._implicit = kwd.pop('implicit', None)
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

    def keys(self):
        return self._conval.keys()

    def __iter__(self):
        return self._conval.keys().__iter__()

    def iteritems(self):
        return iteritems(self._conval)

    def __contains__(self,ndx):
        return ndx in self._conval

    def __len__(self):
        return len(self._conval)

    def __getitem__(self,ndx):
        """This method returns a _ConnectorData object.
        """
        try:
            return self._conval[ndx]
        except KeyError: # thrown if the supplied index is hashable, but not defined.
            msg = "Unknown index '%s' in connector %s;" % (str(ndx), self.name)
            if (isinstance(ndx, (tuple, list)) and len(ndx) != self.dim()):
                msg += "    Expecting %i-dimensional indices" % self.dim()
            else:
                msg += "    Make sure the correct index sets were used.\n"
                msg += "    Is the ordering of the indices correct?"
            raise KeyError(msg)
        except TypeError: # thrown if the supplied index is not hashable
            msg = sys.exc_info()[1]
            msg2 = "Unable to index connector %s using supplied index with " % self.name
            msg2 += str(msg)
            raise TypeError(msg2)

    def _add_indexed_member(self, ndx):
        new_conval = _ConnectorData(create_name(self.name,ndx))
        new_conval.component = weakref.ref(self)
        new_conval.index = ndx
        
        self._conval[ndx] = new_conval

    def construct(self, data=None):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            logger.debug("Constructing Connector, name=%s, from data=%s", self.name, str(data))
        if self._constructed:
            return
        self._constructed=True
        #
        # Construct _ConnectorData objects for all index values
        #
        rule = self._rule is not None
        extend = self._extends is not None
        init = self._initialize is not None
        if self.is_indexed():
            for ndx in self._index:
                self._add_indexed_member(ndx)
                tmp = self[ndx]
                if self._implicit:
                    for key in self._implicit:
                        self.add(None,key)
                if extend:
                    for key, val in self._extends.vars:
                        tmp.add(val,key)
                if init:
                    for key, val in iteritems(self._initialize):
                        tmp.add(val,key)
                if rule:
                    items = apply_indexed_rule(
                        self, self._rule, self._parent(), ndx)
                    for key, val in iteritems(items):
                        tmp.add(val,key)
        else:
            # if the dimension is a scalar (i.e., we're dealing
            # with a _ConObject), and the _data is already initialized.
            if self._implicit:
                for key in self._implicit:
                    self.add(None,key)
            if extend:
                for key, val in self._extends.vars:
                    self.add(val,key)
            if init:
                for key, val in iteritems(dict(self._initialize)):
                    self.add(val,key)
            if rule:
                items = apply_indexed_rule(
                    self, self._rule, self._parent(), ())
                for key, val in iteritems(items):
                    self.add(val,key)
            

    def _pprint(self, ostream=None, verbose=False):
        """Print component information."""
        def _line_generator(k,v):
            for _k, _v in iteritems(self._conval[k]):
                yield _k, len(_v), _v
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
        if None in self._conval:
            ostream.write(prefix+"  : {"+\
                ', '.join(sorted(self._conval[None].keys()))+"}"+'\n')
        else:
            for key in sorted(self._conval.keys()):
                ostream.write(prefix+"  "+str(key)+" : {"+\
                  ', '.join(sorted(self._conval[key].keys()))+"}"+'\n')


class SimpleConnector(Connector, _ConnectorData):

    def __init__(self, *args, **kwd):

        _ConnectorData.__init__(self, kwd.get('name', None) )
        Connector.__init__(self, *args, **kwd)
        self._conval[None] = self
        self._conval[None].component = weakref.ref(self)
        self._conval[None].index = None

    def __getstate__(self):
        result = _ConnectorData.__getstate__(self)
        for key,value in iteritems(self.__dict__):
            result[key]=value
        if type(result['_conval'][None].component) is weakref.ref:
            result['_conval'][None].component = None
        return result

    def __setstate__(self, dict):
        for key in dict:
            setattr(self, key, dict[key])
        self._conval[None].component = weakref.ref(self)

    def is_constant(self):
        return _ConnectorData.is_constant(self)


# a IndexedConnector is the implementation representing an indexed connector.

class IndexedConnector(Connector):
    
    def __init__(self, *args, **kwds):

        Connector.__init__(self, *args, **kwds)
        self._dummy_val = _ConnectorData(kwds.get('name', None))

    def __float__(self):
        raise TypeError("Cannot access the value of array connector "+self.name)

    def __int__(self):
        raise TypeError("Cannot access the value of array connector "+self.name)

    def set_value(self, value):
        msg = "Cannot specify the value of a connector '%s'"
        raise ValueError(msg % self.name)

    def __str__(self):
        return self.name

    #def construct(self, data=None):
    #    Connector.construct(self, data)



register_component(Connector, "A bundle of variables that can be manipilated together.")


class ConnectorExpander(Plugin):
    implements(IPyomoScriptModifyInstance)

    def apply(self, **kwds):
        instance = kwds.pop('instance')
        xform = TransformationFactory('core.expand_connectors')
        xform.apply_to(instance, **kwds)
        return instance

transform = ConnectorExpander()
