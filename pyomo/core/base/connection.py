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
from pyomo.core.base.plugin import (register_component,
    IPyomoScriptModifyInstance, TransformationFactory)
from pyomo.common.plugin import Plugin, implements
from six import iteritems


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
        """Set the connection attributes on this connection."""

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
        self._connectors = connectors # lists do not go through add_component
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
        connectors      A two-member iterable of single Connectors for an
                            undirected connection
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
            return SimpleConstraint.__new__(SimpleConstraint)
        else:
            return IndexedConstraint.__new__(IndexedConstraint)

    def __init__(self, *args, **kwds):
        tmp = dict()
        tmp["source"] = kwds.pop("source", None)
        tmp["destination"] = kwds.pop("destination", None)
        tmp["connectors"] = kwds.pop("connectors", None)
        self._directed = kwds.pop("directed", None)
        self._init_vals = tmp
        self._rule = kwargs.pop('rule', None)
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
        init_rule = self._rule
        init_vals = self._init_vals
        self_parent = self._parent()

        if not self.is_indexed():
            if init_rule is None:
                tmp = init_vals
            else:
                try:
                    tmp = init_rule(self_parent)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error(
                        "Rule failed when generating expression for "
                        "connection %s:\n%s: %s"
                        % (self.name,
                           type(err).__name__,
                           err))
                    raise
                if type(tmp) is not dict:
                    if len(tmp) != 2:
                        raise ValueError("")
                    if self._directed is True:
                        tmp = {"source": tmp[0], "destination": tmp[1]}
                    else:
                        tmp = {"connectors": tmp}
            self.set_value(**tmp)

        else:
            if _init_expr is not None:
                raise IndexError(
                    "Constraint '%s': Cannot initialize multiple indices "
                    "of a constraint with a single expression" %
                    (self.name,) )

            for ndx in self._index:
                try:
                    tmp = apply_indexed_rule(self,
                                             init_rule,
                                             self_parent,
                                             ndx)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error(
                        "Rule failed when generating expression for "
                        "constraint %s with index %s:\n%s: %s"
                        % (self.name,
                           str(ndx),
                           type(err).__name__,
                           err))
                    raise
                self._setitem_when_not_present(ndx, tmp)
        timer.report()


class SimpleConnection(_ConnectionData, Connection):

    def __init__(self, *args, **kwds):
        _ConnectionData.__init__(self, self)
        Connection.__init__(self, *args, **kwds)

    def set_value(self, *args, **kwds):
        """Set the connection attributes on this connection."""
        if not self._constructed:
            raise ValueError(
                "Setting the value of connection '%s' before "
                "the Connection has been constructed (there "
                "is currently no object to set)." % (self.name))
        return super(SimpleConnection, self).set_value(*args, **kwds)

    def pprint(self, filename=None, ostream=None, verbose=False, prefix=""):
        Connection.pprint(self, filename=filename, ostream=ostream,
                          verbose=verbose, prefix=prefix)


class IndexedConnection(Connection):

    def __init__(self, *args, **kwds):
        source = kwds.pop("source", None)
        destination = kwds.pop("destination", None)
        connectors = kwds.pop("connectors", None)

        self._validate_iconns(source, destination, connectors)

        # these attributes need different names than _ConnectionData
        # for the Block rule generating routine to copy them over correctly
        object.__setattr__(self, "_isource", source)
        object.__setattr__(self, "_idestination", destination)
        self._iconnectors = connectors
        self._idirected = self.source is not None

        kwds.setdefault("rule", self._indexed_connection_rule())

        super(IndexedConnection, self).__init__(*args, **kwds)

    @property
    def source(self):
        return self._isource

    @property
    def destination(self):
        return self._idestination

    @property
    def connectors(self):
        return self._iconnectors

    @property
    def directed(self):
        return self._idirected

    def _validate_iconns(self, source, destination, connectors):
        if connectors is not None:
            if source is not None or destination is not None:
                raise ValueError("Cannot specify 'source' or 'destination' "
                                 "when using 'connectors' argument.")
            if (type(connectors) not in (list, tuple) or
                len(connectors) != 2):
                raise ValueError("Argument 'connectors' must be list or tuple "
                                 "containing exactly 2 IndexedConnectors.")
            for c in connectors:
                if type(c) is not IndexedConnector:
                    raise ValueError("Found object '%s' in 'connectors' not "
                                     "of type IndexedConnector." % str(c))
        else:
            if source is None or destination is None:
                raise ValueError("Must specify both 'source' and "
                                 "'destination' for directed Connection.")
            if type(source) is not IndexedConnector:
                raise ValueError("Source object '%s' not of type "
                                 "IndexedConnector." % str(source))
            if type(destination) is not IndexedConnector:
                raise ValueError("Destination object '%s' not of type "
                                 "IndexedConnector." % str(destination))

    def _indexed_connection_rule(self):
        if self.directed:
            def generate_connections(m, *args):
                return _ConnectionData(self, source=self.source[args],
                                       destination=self.destination[args])
        else:
            def generate_connections(m, *args):
                connectors = [conn[args] for conn in self.connectors]
                return _ConnectionData(self, connectors=connectors)
        return generate_connections


register_component(Connection, "Connection used for equating two Connectors.")


class ConnectionExpander(Plugin):
    implements(IPyomoScriptModifyInstance)

    def apply(self, **kwds):
        instance = kwds.pop('instance')
        xform = TransformationFactory('core.expand_connections')
        xform.apply_to(instance, **kwds)
        return instance

transform = ConnectionExpander()
