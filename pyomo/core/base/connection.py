#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import register_component, Block, Connector
from pyomo.core.base.block import _BlockData

class _ConnectionData(_BlockData):
    pass

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
        # bypass add_component, just want a reference to these connectors
        object.__setattr__(self, "source", kwds.pop("source", None))
        object.__setattr__(self, "destination", kwds.pop("destination", None))
        object.__setattr__(self, "connectors", kwds.pop("connectors", None))
        kwds.setdefault('ctype', Connection)

        Block.__init__(self, *args, **kwds)

        self._validate_conns()
        self.directed = self.source is not None

    def _validate_conns(self):
        if self.connectors is not None:
            if self.source is not None or self.destination is not None:
                raise ValueError("Cannot specify 'source' or 'destination' "
                                 "when using 'connectors' argument.")
            if (type(self.connectors) not in (list, tuple) or
                len(self.connectors) != 2):
                raise ValueError("Argument 'connectors' must be a list or "
                                 "tuple containing exactly 2 Connectors.")
            for c in self.connectors:
                if c.type() is not Connector:
                    raise ValueError("Found object '%s' in 'connectors' not "
                                     "of type() Connector." % c)
        else:
            if self.source is None or self.destination is None:
                raise ValueError("Must specify both 'source' and "
                                 "'destination' for directed Connection.")
            try:
                if self.source.type() is not Connector:
                    raise ValueError("Source object '%s' not of "
                                     "type Connector." % self.source)
                if self.destination.type() is not Connector:
                    raise ValueError("Destination object '%s' not of "
                                     "type Connector." % self.destination)
            except AttributeError:
                raise ValueError("Both 'source' and 'destination' objects "
                                 "must be of type Connector")


class SimpleConnection(_ConnectionData, Connection):

    def __init__(self, *args, **kwds):
        _ConnectionData.__init__(self, self)
        Connection.__init__(self, *args, **kwds)
        self._data[None] = self

    def pprint(self, filename=None, ostream=None, verbose=False, prefix=""):
        Connection.pprint(self, filename=filename, ostream=ostream,
                          verbose=verbose, prefix=prefix)


class IndexedConnection(Connection):
    pass


register_component(Connection, "Connection block equating two Connectors.")
