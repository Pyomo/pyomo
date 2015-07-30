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

from pyomo.util.plugin import Plugin, implements

from pyomo.core.base.component import Component, register_component
from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.expr import _ProductExpression
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.misc import apply_indexed_rule, create_name
from pyomo.core.base.numvalue import NumericValue
from pyomo.core.base.plugin import IPyomoScriptModifyInstance
from pyomo.core.base.var import Var, VarList

logger = logging.getLogger('pyomo.core')


class _ConnectorValue(NumericValue):
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
        for i in _ConnectorValue.__slots__:
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

    def pprint(self, ostream=None, verbose=False):
        if ostream is None:
            ostream = sys.stdout
        ostream.write(str(self))

    def add(self, var, name=None, aggregate=None):
        if name is None:
            name = var.name
        if name in self.vars:
            raise ValueError("Cannot insert duplicate variable name "
                             "'%s' into Connector '%s'" % ( name, self.name ))
        self.vars[name] = var
        if aggregate is not None:
            self.aggregators[var] = aggregate


class SimpleConnectorBase(IndexedComponent):
    """A collection of variables, which may be defined over a index"""

    """ Constructor
        Arguments:
           name         The name of this connector
           index        The index set that defines the distinct connectors.
                          By default, this is None, indicating that there
                          is a single connector.
    """

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
        """This method returns a _ConnectorValue object.
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
        new_conval = _ConnectorValue(create_name(self.name,ndx))
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
        # Construct _ConnectorValue objects for all index values
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
            

    def pprint(self, ostream=None, verbose=False):
        if ostream is None:
            ostream = sys.stdout
        ostream.write("  "+self.name+" :")
        ostream.write("\tSize="+str(len(self)))
        if self._index_set is not None:
            ostream.write("\tIndicies: ")
            for idx in self._index_set:
                ostream.write(str(idx.name)+", ")
            print("")
        if None in self._conval:
            ostream.write("\tName : Variable\n")
            for item in iteritems(self._conval[None]):
                ostream.write("\t %s : %s\n" % item)
        else:
            ostream.write("\tKey : Name : Variable\n")
            tmp=self._conval.keys()
            tmp.sort()
            for key in tmp:
                for name, var in iteritems(self._conval[key]):
                    ostream.write("\t %s : %s : %s\n" % ( key, name, var ))


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


class SimpleConnector(SimpleConnectorBase, _ConnectorValue):

    def __init__(self, *args, **kwd):

        _ConnectorValue.__init__(self, kwd.get('name', None) )
        SimpleConnectorBase.__init__(self, *args, **kwd)
        self._conval[None] = self
        self._conval[None].component = weakref.ref(self)
        self._conval[None].index = None

    def __getstate__(self):
        result = _ConnectorValue.__getstate__(self)
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
        return _ConnectorValue.is_constant(self)


# a IndexedConnector is the implementation representing an indexed connector.

class IndexedConnector(SimpleConnectorBase):
    
    def __init__(self, *args, **kwds):

        SimpleConnectorBase.__init__(self, *args, **kwds)
        self._dummy_val = _ConnectorValue(kwds.get('name', None))

    def __float__(self):
        raise TypeError("Cannot access the value of array connector "+self.name)

    def __int__(self):
        raise TypeError("Cannot access the value of array connector "+self.name)

    def set_value(self, value):
        msg = "Cannot specify the value of a connector '%s'"
        raise ValueError(msg % self.name)

    def __str__(self):
        return self.name

    def construct(self, data=None):
        SimpleConnectorBase.construct(self, data)


class Connector(Component):
    """A 'bundle' of variables that can be manipulated together"""

    @classmethod
    def conserved_quantity():
        pass

    # The idea behind a Connector is to create a bundle of variables
    # that can be manipulated as a single variable within constraints.
    # While Connectors inherit from variable (mostly so that the
    # expression infrastucture can manipulate them), they are not actual
    # variables that are exposed to the solver.  Instead, a preprocessor
    # (ConnectorExpander) will look for expressions that involve
    # connectors and replace the single constraint with a list of
    # constraints that involve the original variables contained within
    # the Connector.

    def __new__(cls, *args, **kwds):
        if args == ():
            self = SimpleConnector(*args, **kwds)
        else:
            self = IndexedConnector(*args, **kwds)
        return self




class ConnectorExpander(Plugin):
    implements(IPyomoScriptModifyInstance)

    def apply(self, **kwds):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            logger.debug("Calling ConnectorExpander")
                
        instance = kwds['instance']
        blockList = list(instance.block_data_objects(active=True))
        noConnectors = True
        for b in blockList:
            if b.component_map(Connector):
                noConnectors = False
                break
        if noConnectors:
            return

        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            logger.debug("   Connectors found!")

        #
        # At this point, there are connectors in the model, so we must
        # look for constraints that involve connectors and expand them.
        #
        #options = kwds['options']
        #model = kwds['model']

        # In general, blocks should be relatively self-contained, so we
        # should build the connectors from the "bottom up":
        blockList.reverse()

        # Expand each constraint involving a connector
        for block in blockList:
            if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
                logger.debug("   block: " + block.cname())

            CCC = {}
            for name, constraint in itertools.chain\
                    ( iteritems(block.component_map(Constraint)), 
                      iteritems(block.component_map(ConstraintList)) ):
                cList = []
                CCC[name+'.expanded'] = cList
                for idx, c in iteritems(constraint._data):
                    if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
                        logger.debug("   (looking at constraint %s[%s])", name, idx)
                    connectors = []
                    self._gather_connectors(c.body, connectors)
                    if len(connectors) == 0:
                        continue
                    if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
                        logger.debug("   (found connectors in constraint)")
                    
                    # Validate that all connectors match
                    errors, ref, skip = self._validate_connectors(connectors)
                    if errors:
                        logger.error(
                            ( "Connector mismatch: errors detected when "
                              "constructing constraint %s\n    " %
                              (name + (idx and '[%s]' % idx or '')) ) +
                            '\n    '.join(reversed(errors)) )
                        raise ValueError(
                            "Connector mismatch in constraint %s" % \
                            name + (idx and '[%s]' % idx or ''))
                    
                    if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
                        logger.debug("   (connectors valid)")

                    # Fill in any empty connectors
                    for conn in connectors:
                        if conn.vars:
                            continue
                        for var in ref.vars:
                            if var in skip:
                                continue
                            v = Var()
                            block.add_component(conn.cname() + '.auto.' + var, v)
                            conn.vars[var] = v
                            v.construct()
                    
                    # OK - expand this constraint
                    self._expand_constraint(block, name, idx, c, ref, skip, cList)
                    # Now deactivate the original constraint
                    c.deactivate()
            for name, exprs in iteritems(CCC):
                cList = ConstraintList()
                block.add_component( name, cList )
                cList.construct()
                for expr in exprs:
                    cList.add(expr)
                

        # Now, go back and implement VarList aggregators
        for block in blockList:
            for conn in itervalues(block.component_map(Connector)):
                for var, aggregator in iteritems(conn.aggregators):
                    c = Constraint(expr=aggregator(block, var))
                    block.add_component(
                        conn.cname() + '.' + var.cname() + '.aggregate', c)
                    c.construct()

    def _gather_connectors(self, expr, connectors):
        if expr.is_expression():
            if expr.__class__ is _ProductExpression:
                for e in expr._numerator:
                    self._gather_connectors(e, connectors)
                for e in expr._denominator:
                    self._gather_connectors(e, connectors)
            else:
                for e in expr._args:
                    self._gather_connectors(e, connectors)
        elif isinstance(expr, _ConnectorValue):
            connectors.append(expr)

    def _validate_connectors(self, connectors):
        errors = []
        ref = None
        skip = set()
        for idx in xrange(len(connectors)):
            if connectors[idx].vars.keys():
                ref = connectors.pop(idx)
                break
        if ref is None:
            errors.append(
                "Cannot identify a reference connector: no connectors "
                "have assigned variables" )
            return errors, ref, skip

        a = set(ref.vars.keys())
        for key, val in iteritems(ref.vars):
            if val is None:
                skip.add(key)
        for tmp in connectors:
            b = set(tmp.vars.keys())
            if not b:
                continue
            for key, val in iteritems(tmp.vars):
                if val is None:
                    skip.add(key)
            for var in a - b:
                # TODO: add a fq_name so we can easily get
                # the full model.block.connector name
                errors.append(
                    "Connector '%s' missing variable '%s' "
                    "(appearing in reference connector '%s')" %
                    ( tmp.cname(), var, ref.cname() ) )
            for var in b - a:
                errors.append(
                    "Reference connector '%s' missing variable '%s' "
                    "(appearing in connector '%s')" %
                    ( ref.cname(), var, tmp.cname() ) )
        return errors, ref, skip

    def _expand_constraint(self, block, name, idx, constraint, ref, skip, cList):
        def _substitute_var(arg, var):
            if arg.is_expression():
                if arg.__class__ is _ProductExpression:
                    _substitute_vars(arg._numerator, var)
                    _substitute_vars(arg._denominator, var)
                else:
                    _substitute_vars(arg._args, var)
                return arg
            elif isinstance(arg, _ConnectorValue):
                v = arg.vars[var]
                if v.is_expression():
                    v = v.clone()
                return _substitute_var(v, var) 
            elif isinstance(arg, VarList):
                return arg.add()
            return arg

        def _substitute_vars(args, var):
            for idx, arg in enumerate(args):
                if arg.is_expression():
                    if arg.__class__ is _ProductExpression:
                        _substitute_vars(arg._numerator, var)
                        _substitute_vars(arg._denominator, var)
                    else:
                        _substitute_vars(arg._args, var)
                elif isinstance(arg, _ConnectorValue):
                    v = arg.vars[var]
                    if v.is_expression():
                        v = v.clone()
                    args[idx] = _substitute_var(v, var) 
                elif isinstance(arg, VarList):
                    args[idx] = arg.add()

        for var in ref.vars.iterkeys():
            if var in skip:
                continue
            if constraint.body.is_expression():
                c = _substitute_var(constraint.body.clone(), var)
            else:
                c = _substitute_var(constraint.body, var)
            if constraint.equality:
                cList.append( ( c, constraint.upper ) )
            else:
                cList.append( ( constraint.lower, c, constraint.upper ) )

transform = ConnectorExpander()


register_component(Connector, "A bundle of variables that can be manipilated together.")

