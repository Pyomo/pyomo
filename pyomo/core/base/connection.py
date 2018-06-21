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

from pyomo.core.base.block import Block, _BlockData
from pyomo.core.base.connector import (SimpleConnector, IndexedConnector,
    _ConnectorData)
from pyomo.core.base.plugin import (register_component,
    IPyomoScriptModifyInstance, TransformationFactory)
from pyomo.common.plugin import Plugin, implements
from six import iteritems


class _ConnectionData(_BlockData):

    def __init__(self, component, **kwds):

        super(_ConnectionData, self).__init__(component)

        # We have to be ok with not setting these attributes when initialized,
        # because IndexedComponent._setitem_when_not_present creates this with
        # no arguments except component. Thus we must rely on always using
        # set_connection to set these attributes so that _validate_conns is
        # called, especially if it is after instantiating the class. There is a
        # check in the ConnectionExpander to make sure each Connection is
        # initialized before it is expanded, but this would probably only
        # happen if the user passed a bad custom IndexedConnection rule.
        if len(kwds):
            self.set_connection(**kwds)

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

    def set_connection(self, **kwds):
        if not hasattr(self, "_source"):
            # if not yet defined, use None for default
            source = kwds.pop("source", None)
            destination = kwds.pop("destination", None)
            connectors = kwds.pop("connectors", None)
        else:
            # if this is resetting them, use previous setting as default so you
            # can change only source and not have to pass destination again
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
        self._directed = self.source is not None

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


class Connection(Block):

    _ComponentDataClass = _ConnectionData

    def __new__(cls, *args, **kwds):
        if cls != Connection:
            return super(Connection, cls).__new__(cls)
        if args == ():
            return SimpleConnection.__new__(SimpleConnection)
        else:
            return IndexedConnection.__new__(IndexedConnection)

    def __init__(self, *args, **kwds):
        kwds.setdefault("ctype", Connection)
        super(Connection, self).__init__(*args, **kwds)


class SimpleConnection(_ConnectionData, Connection):

    def __init__(self, *args, **kwds):
        tmp = dict()
        tmp["source"] = kwds.pop("source", None)
        tmp["destination"] = kwds.pop("destination", None)
        tmp["connectors"] = kwds.pop("connectors", None)

        _ConnectionData.__init__(self, self, **tmp)
        Connection.__init__(self, *args, **kwds)
        self._data[None] = self

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


register_component(Connection, "Connection block equating two Connectors.")


class ConnectionExpander(Plugin):
    implements(IPyomoScriptModifyInstance)

    def apply(self, **kwds):
        instance = kwds.pop('instance')
        xform = TransformationFactory('core.expand_connections')
        xform.apply_to(instance, **kwds)
        return instance

transform = ConnectionExpander()
