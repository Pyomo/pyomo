#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = [ 'Connection' ]

from pyomo.core.base.component import ActiveComponentData
from pyomo.core.base.indexed_component import (ActiveIndexedComponent,
    UnindexedComponent_set)
from pyomo.core.base.connector import (SimpleConnector, IndexedConnector,
    _ConnectorData)
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.plugin import (register_component,
    IPyomoScriptModifyInstance, TransformationFactory)
from pyomo.common.plugin import Plugin, implements
from pyomo.common.timing import ConstructionTimer
from six import iteritems
from weakref import ref as weakref_ref
import logging, sys

logger = logging.getLogger('pyomo.core')


class _ConnectionData(ActiveComponentData):

    def __init__(self, component=None, **kwds):

        self._component = weakref_ref(component) if (component is not None) \
                          else None
        self._active = True
        self._source = None
        self._destination = None
        self._connectors = None
        self._directed = None
        if len(kwds):
            self.set_value(**kwds)

    @property
    def source(self):
        return self._source

    @property
    def destination(self):
        return self._destination

    @property
    def connectors(self):
        return self._connectors

    @property
    def directed(self):
        return self._directed

    def set_value(self, **kwds):
        """Set the connector attributes on this connection."""
        # allow user to change source without having to repass destination
        source = kwds.pop("source", self._source)
        destination = kwds.pop("destination", self._destination)
        connectors = kwds.pop("connectors", self._connectors)

        if len(kwds):
            raise ValueError(
                "set_args passed unrecognized keyword arguments:\n\t" +
                "\n\t".join("%s = %s" % (k,v) for k,v in iteritems(kwds)))

        self._validate_conns(source, destination, connectors)

        # bypass add_component
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_destination", destination)
        # tuples do not go through add_component
        self._connectors = tuple(connectors) if connectors is not None else None
        self._directed = self._source is not None

    def _validate_conns(self, source, destination, connectors):
        connector_types = set([SimpleConnector, _ConnectorData])
        if connectors is not None:
            if source is not None or destination is not None:
                raise ValueError("Cannot specify 'source' or 'destination' "
                                 "when using 'connectors' argument.")
            if (type(connectors) not in (list, tuple) or
                len(connectors) != 2):
                raise ValueError("Argument 'connectors' must be list or tuple "
                                 "containing exactly 2 Connectors.")
            for c in connectors:
                if type(c) not in connector_types:
                    if type(c) is IndexedConnector:
                        raise ValueError(
                            "Found IndexedConnector '%s' in 'connectors', "
                            "must use single Connectors for Connection." % c)
                    raise ValueError("Found object '%s' in 'connectors' not "
                                     "of type Connector." % str(c))
        else:
            if source is None or destination is None:
                raise ValueError("Must specify both 'source' and "
                                 "'destination' for directed Connection.")
            if type(source) not in connector_types:
                if type(source) is IndexedConnector:
                    raise ValueError(
                        "Found IndexedConnector '%s' as source, must use "
                        "single Connectors for Connection." % source)
                raise ValueError("Source object '%s' not of type "
                                 "Connector." % str(source))
            if type(destination) not in connector_types:
                if type(destination) is IndexedConnector:
                    raise ValueError(
                        "Found IndexedConnector '%s' as destination, must use "
                        "single Connectors for Connection." % destination)
                raise ValueError("Destination object '%s' not of type "
                                 "Connector." % str(destination))


class Connection(ActiveIndexedComponent):
    """
    Component used for equating the members of two Connector objects.

    Constructor arguments:
        source          A single Connector for a directed connection
        destination     A single Connector for a directed connection
        connectors      A two-member list or tuple of single Connectors
                            for an undirected connection
        directed        True if directed. Use along with rule to be able to
                            return an implied (source, destination) tuple
        rule            A function that returns either a dictionary of the
                            connection arguments or a two-member iterable
        doc             A text string describing this component
        name            A name for this component
    """

    _ComponentDataClass = _ConnectionData

    def __new__(cls, *args, **kwds):
        if cls != Connection:
            return super(Connection, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            return SimpleConnection.__new__(SimpleConnection)
        else:
            return IndexedConnection.__new__(IndexedConnection)

    def __init__(self, *args, **kwds):
        source = kwds.pop("source", None)
        destination = kwds.pop("destination", None)
        connectors = kwds.pop("connectors", None)
        if source is destination is connectors is None:
            self._init_vals = None
        else:
            self._init_vals = dict(
                source=source, destination=destination, connectors=connectors)
        self._directed = kwds.pop("directed", None)
        self._rule = kwds.pop('rule', None)
        kwds.setdefault("ctype", Connection)
        super(Connection, self).__init__(*args, **kwds)

    def construct(self, data=None):
        """
        Initialize the Connection
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing Connection %s" % (self.name))

        if self._constructed:
            return

        timer = ConstructionTimer(self)
        self._constructed = True

        if self._rule is None and self._init_vals is None:
            # No construction rule or values specified
            return
        elif self._rule is not None and self._init_vals is not None:
            raise ValueError(
                "Cannot specify rule along with source/destination/connectors "
                "keywords for connection '%s'" % self.name)

        self_parent = self._parent()

        if not self.is_indexed():
            if self._rule is None:
                tmp = self._init_vals
            else:
                try:
                    tmp = self._rule(self_parent)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error(
                        "Rule failed when generating values for "
                        "connection %s:\n%s: %s"
                        % (self.name, type(err).__name__, err))
                    raise
            tmp = self._validate_init_vals(tmp)
            self._setitem_when_not_present(None, tmp)
        else:
            if self._init_vals is not None:
                raise IndexError(
                    "Connection '%s': Cannot initialize multiple indices "
                    "of a connection with a single connectors" %
                    (self.name,) )
            for idx in self._index:
                try:
                    tmp = apply_indexed_rule(self, self._rule, self_parent, idx)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error(
                        "Rule failed when generating values for "
                        "connection %s with index %s:\n%s: %s"
                        % (self.name, str(idx), type(err).__name__, err))
                    raise
                tmp = self._validate_init_vals(tmp)
                self._setitem_when_not_present(idx, tmp)
        timer.report()

    def _validate_init_vals(self, vals):
        # returns dict version of vals if not already dict
        if type(vals) is not dict:
            # check that it's a two-member iterable
            conns = None
            if hasattr(vals, "__iter__"):
                conns = tuple(vals)
            if conns is None or len(conns) != 2:
                raise ValueError(
                    "Connection rule for '%s' did not return either a "
                    "dict or a two-member iterable." % self.name)
            if self._directed is True:
                vals = {"source": vals[0], "destination": vals[1]}
            else:
                vals = {"connectors": vals}
        elif self._directed is not None:
            # if for some reason they specified directed, check it
            if (((vals["source"] is not None or vals["destination"] is not None)
                 and self._directed is False) or
                (vals["connectors"] is not None and self._directed is True)):
                raise ValueError(
                    "Passed incorrect value for 'directed' for connection "
                    "'%s'. Value is set automatically when using keywords."
                    % self.name)
        return vals

    def _setitem_when_not_present(self, index, value):
        # value should be a dict which gets unpacked and passed to set_value
        if index is None and not self.is_indexed():
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(component=self)
        try:
            obj.set_value(**value)
            return obj
        except:
            del self._data[index]
            raise

class SimpleConnection(_ConnectionData, Connection):

    def __init__(self, *args, **kwds):
        _ConnectionData.__init__(self, self)
        Connection.__init__(self, *args, **kwds)

    def set_value(self, *args, **kwds):
        """Set the connection attributes on this connection."""
        if not self._constructed:
            raise ValueError("Setting the value of connection '%s' before "
                             "the Connection has been constructed (there "
                             "is currently no object to set)." % (self.name))
        return super(SimpleConnection, self).set_value(*args, **kwds)

    def pprint(self, filename=None, ostream=None, verbose=False, prefix=""):
        Connection.pprint(self, filename=filename, ostream=ostream,
                          verbose=verbose, prefix=prefix)


class IndexedConnection(Connection):
    pass


register_component(Connection, "Connection used for equating two Connectors.")


class ConnectionExpander(Plugin):
    implements(IPyomoScriptModifyInstance)

    def apply(self, **kwds):
        instance = kwds.pop('instance')
        xform = TransformationFactory('core.expand_connections')
        xform.apply_to(instance, **kwds)
        return instance

transform = ConnectionExpander()
