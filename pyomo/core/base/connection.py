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


class _ConnectionData(_BlockData):
    def __init__(self, component, **kwds):

        super(_ConnectionData, self).__init__(component)

        # We have to be ok with not setting these attributes when initialized,
        # because IndexedComponent._setitem_when_not_present creates this with
        # no arguments except component. Thus we must rely on always using
        # _set_args to set these attributes so that _validate_conns is called,
        # especially if it is after instantiating the class.
        if len(kwds):
            source = kwds.pop("source", None)
            destination = kwds.pop("destination", None)
            connectors = kwds.pop("connectors", None)
            self._set_args(source, destination, connectors)

    def _set_args(self, source=None, destination=None, connectors=None):
        _setattr = object.__setattr__ # bypass add_component
        _setattr(self, "source", source)
        _setattr(self, "destination", destination)
        _setattr(self, "connectors", connectors)
        self._validate_conns()

    def _validate_conns(self):
        connector_types = set([SimpleConnector, _ConnectorData])
        if self.connectors is not None:
            if self.source is not None or self.destination is not None:
                raise ValueError("Cannot specify 'source' or 'destination' "
                                 "when using 'connectors' argument.")
            if (type(self.connectors) not in (list, tuple) or
                len(self.connectors) != 2):
                raise ValueError("Argument 'connectors' must be list or tuple "
                                 "containing exactly 2 Connectors.")
            for c in self.connectors:
                if type(c) not in connector_types:
                    if type(c) is IndexedConnector:
                        raise ValueError(
                            "Found IndexedConnector '%s' in 'connectors', "
                            "must use single Connectors for Connection." % c)
                    raise ValueError("Found object '%s' in 'connectors' not "
                                     "of type Connector." % str(c))
        else:
            if self.source is None or self.destination is None:
                raise ValueError("Must specify both 'source' and "
                                 "'destination' for directed Connection.")
            if type(self.source) not in connector_types:
                if type(self.source) is IndexedConnector:
                    raise ValueError(
                        "Found IndexedConnector '%s' as source, must use "
                        "single Connectors for Connection." % self.source)
                raise ValueError("Source object '%s' not of type "
                                 "Connector." % str(self.source))
            if type(self.destination) not in connector_types:
                if type(self.destination) is IndexedConnector:
                    raise ValueError(
                        "Found IndexedConnector '%s' as destination, must use "
                        "single Connectors for Connection." % self.destination)
                raise ValueError("Destination object '%s' not of type "
                                 "Connector." % str(self.destination))
        self.directed = self.source is not None


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
        object.__setattr__(self, "isource", kwds.pop("source", None))
        object.__setattr__(self, "idestination", kwds.pop("destination", None))
        object.__setattr__(self, "iconnectors", kwds.pop("connectors", None))
        self._validate_iconns()
        kwds.setdefault("rule", self._indexed_connection_rule())

        super(IndexedConnection, self).__init__(*args, **kwds)

    def _validate_iconns(self):
        if self.iconnectors is not None:
            if self.isource is not None or self.idestination is not None:
                raise ValueError("Cannot specify 'source' or 'destination' "
                                 "when using 'connectors' argument.")
            if (type(self.iconnectors) not in (list, tuple) or
                len(self.iconnectors) != 2):
                raise ValueError("Argument 'connectors' must be list or tuple "
                                 "containing exactly 2 IndexedConnectors.")
            for c in self.iconnectors:
                if type(c) is not IndexedConnector:
                    raise ValueError("Found object '%s' in 'connectors' not "
                                     "of type IndexedConnector." % str(c))
        else:
            if self.isource is None or self.idestination is None:
                raise ValueError("Must specify both 'source' and "
                                 "'destination' for directed Connection.")
            if type(self.isource) is not IndexedConnector:
                raise ValueError("Source object '%s' not of type "
                                 "IndexedConnector." % str(self.isource))
            if type(self.idestination) is not IndexedConnector:
                raise ValueError("Destination object '%s' not of type "
                                 "IndexedConnector." % str(self.destination))
        self.idirected = self.isource is not None

    def _indexed_connection_rule(self):
        if self.idirected:
            def generate_connections(m, *args):
                return _ConnectionData(self, source=self.isource[args],
                                       destination=self.idestination[args])
        else:
            def generate_connections(m, *args):
                connectors = [conn[args] for conn in self.iconnectors]
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
