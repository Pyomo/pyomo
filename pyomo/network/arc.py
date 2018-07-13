#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = [ 'Arc' ]

from pyomo.network.port import (SimplePort, IndexedPort, _PortData)
from pyomo.core.base.component import ActiveComponentData
from pyomo.core.base.indexed_component import (ActiveIndexedComponent,
    UnindexedComponent_set)
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.plugin import (register_component,
    IPyomoScriptModifyInstance, TransformationFactory)
from pyomo.common.plugin import Plugin, implements
from pyomo.common.timing import ConstructionTimer
from six import iteritems
from weakref import ref as weakref_ref
import logging, sys

logger = logging.getLogger('pyomo.network')


class _ArcData(ActiveComponentData):
    """This class defines the data for a single Arc."""

    __slots__ = ('_ports', '_directed', '_expanded_block')

    def __init__(self, component=None, **kwds):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - ActiveComponentData
        #   - ComponentData
        self._component = weakref_ref(component) if (component is not None) \
                          else None
        self._active = True

        self._ports = None
        self._directed = None
        self._expanded_block = None
        if len(kwds):
            self.set_value(kwds)

    def __getstate__(self):
        state = super(_ArcData, self).__getstate__()
        for i in _ArcData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: None of the slots on this class need to be edited, so we
    # don't need to implement a specialized __setstate__ method, and
    # can quietly rely on the super() class's implementation.

    def __getattr__(self, name):
        """Returns self.expanded_block.name if it exists"""
        eb = self.expanded_block
        if eb is not None:
            try:
                return getattr(eb, name)
            except AttributeError:
                # if it didn't work, throw our own error below
                pass
        # Since the base classes don't support getattr, we can just
        # throw the "normal" AttributeError
        raise AttributeError("'%s' object has no attribute '%s'"
                             % (self.__class__.__name__, name))

    @property
    def source(self):
        # directed can be true before construction
        # so make sure ports is not None
        return self._ports[0] if (
            self._directed is True and self._ports is not None
            ) else None

    src = source

    @property
    def destination(self):
        return self._ports[1] if (
            self._directed is True and self._ports is not None
            ) else None

    dest = destination

    @property
    def ports(self):
        return self._ports

    @property
    def directed(self):
        return self._directed

    @property
    def expanded_block(self):
        return self._expanded_block

    def set_value(self, vals):
        """
        Set the port attributes on this arc.

        If these values are being reassigned, note that the defaults
        are still None, so you may need to repass some attributes.
        """
        source = vals.pop("source", None)
        destination = vals.pop("destination", None)
        ports = vals.pop("ports", None)

        if len(vals):
            raise ValueError(
                "set_value passed unrecognized keyword arguments:\n\t" +
                "\n\t".join("%s = %s" % (k,v) for k,v in iteritems(vals)))

        self._validate_ports(source, destination, ports)

        if self.ports is not None:
            # we are reassigning this arc's values, clean up port lists
            for port in self.ports:
                port._arcs.remove(self)
            if self._directed:
                self.source._dests.remove(self)
                self.destination._sources.remove(self)

        self._ports = tuple(ports) if ports is not None \
            else (source, destination)
        self._directed = source is not None
        for port in self._ports:
            port._arcs.append(self)
        if self._directed:
            source._dests.append(self)
            destination._sources.append(self)

    def _validate_ports(self, source, destination, ports):
        port_types = {SimplePort, _PortData}
        msg = "Arc %s: " % self.name
        if ports is not None:
            if source is not None or destination is not None:
                raise ValueError(msg +
                    "cannot specify 'source' or 'destination' "
                    "when using 'ports' argument.")
            if (type(ports) not in (list, tuple) or
                len(ports) != 2):
                raise ValueError(msg +
                    "argument 'ports' must be list or tuple "
                    "containing exactly 2 Ports.")
            for c in ports:
                if type(c) not in port_types:
                    if type(c) is IndexedPort:
                        raise ValueError(msg +
                            "found IndexedPort '%s' in 'ports', must "
                            "use single Ports for Arc." % c.name)
                    raise ValueError(msg +
                        "found object '%s' in 'ports' not "
                        "of type Port." % str(c))
        else:
            if source is None or destination is None:
                raise ValueError(msg +
                    "must specify both 'source' and 'destination' "
                    "for directed Arc.")
            if type(source) not in port_types:
                if type(source) is IndexedPort:
                    raise ValueError(msg +
                        "found IndexedPort '%s' as source, must use "
                        "single Ports for Arc." % source.name)
                raise ValueError(msg +
                    "source object '%s' not of type Port." % str(source))
            if type(destination) not in port_types:
                if type(destination) is IndexedPort:
                    raise ValueError(msg +
                        "found IndexedPort '%s' as destination, must use "
                        "single Ports for Arc." % destination.name)
                raise ValueError(msg +
                    "destination object '%s' not of type Port."
                    % str(destination))


class Arc(ActiveIndexedComponent):
    """
    Component used for connecting the members of two Port objects.

    Constructor arguments:
        source          A single Port for a directed arc
        destination     A single Port for a directed arc
        ports           A two-member list or tuple of single Ports
                            for an undirected arc
        directed        True if directed. Use along with rule to be able to
                            return an implied (source, destination) tuple
        rule            A function that returns either a dictionary of the
                            arc arguments or a two-member iterable of ports
        doc             A text string describing this component
        name            A name for this component

    Public attributes
        source          The source Port when directed, else None
        destination     The destination Port when directed, else None
        ports           A tuple containing both ports. If directed, this
                            is in the order (source, destination)
        directed        True if directed, False if not
    """

    _ComponentDataClass = _ArcData

    def __new__(cls, *args, **kwds):
        if cls != Arc:
            return super(Arc, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return SimpleArc.__new__(SimpleArc)
        else:
            return IndexedArc.__new__(IndexedArc)

    def __init__(self, *args, **kwds):
        source = kwds.pop("source", kwds.pop("src", None))
        destination = kwds.pop("destination", kwds.pop("dest", None))
        ports = kwds.pop("ports", None)
        self._directed = kwds.pop("directed", None)
        self._rule = kwds.pop('rule', None)
        kwds.setdefault("ctype", Arc)

        super(Arc, self).__init__(*args, **kwds)

        if source is destination is ports is None:
            self._init_vals = None
        else:
            self._init_vals = dict(
                source=source, destination=destination, ports=ports)

    def construct(self, data=None):
        """Initialize the Arc"""
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing Arc %s" % self.name)

        if self._constructed:
            return

        timer = ConstructionTimer(self)
        self._constructed = True

        if self._rule is None and self._init_vals is None:
            # No construction rule or values specified
            return
        elif self._rule is not None and self._init_vals is not None:
            raise ValueError(
                "Cannot specify rule along with source/destination/ports "
                "keywords for arc '%s'" % self.name)

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
                        "arc %s:\n%s: %s"
                        % (self.name, type(err).__name__, err))
                    raise
            tmp = self._validate_init_vals(tmp)
            self._setitem_when_not_present(None, tmp)
        else:
            if self._init_vals is not None:
                raise IndexError(
                    "Arc '%s': Cannot initialize multiple indices "
                    "of an arc with single ports" % self.name)
            for idx in self._index:
                try:
                    tmp = apply_indexed_rule(self, self._rule, self_parent, idx)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error(
                        "Rule failed when generating values for "
                        "arc %s with index %s:\n%s: %s"
                        % (self.name, str(idx), type(err).__name__, err))
                    raise
                tmp = self._validate_init_vals(tmp)
                self._setitem_when_not_present(idx, tmp)
        timer.report()

    def _validate_init_vals(self, vals):
        # returns dict version of vals if not already dict
        if type(vals) is not dict:
            # check that it's a two-member iterable
            ports = None
            if hasattr(vals, "__iter__"):
                # note: every iterable except strings has an __iter__ attribute
                # but strings are not valid anyway
                ports = tuple(vals)
            if ports is None or len(ports) != 2:
                raise ValueError(
                    "Arc rule for '%s' did not return either a "
                    "dict or a two-member iterable." % self.name)
            if self._directed is True:
                vals = {"source": ports[0], "destination": ports[1]}
            else:
                vals = {"ports": ports}
        elif self._directed is not None:
            # if for some reason they specified directed, check it
            s = vals.get("source", None)
            d = vals.get("destination", None)
            c = vals.get("ports", None)
            if (((s is not None or d is not None) and self._directed is False)
                or (c is not None and self._directed is True)):
                raise ValueError(
                    "Passed incorrect value for 'directed' for arc "
                    "'%s'. Value is set automatically when using keywords."
                    % self.name)
        return vals

    def _pprint(self):
        """Return data that will be printed for this component."""
        return (
            [("Size", len(self)),
             ("Index", self._index if self.is_indexed() else None),
             ("Active", self.active)],
            iteritems(self),
            ("Ports", "Directed", "Active"),
            lambda k, v: ["(%s, %s)" % v.ports if v.ports is not None else None,
                          v.directed,
                          v.active])


class SimpleArc(_ArcData, Arc):

    def __init__(self, *args, **kwds):
        _ArcData.__init__(self, self)
        Arc.__init__(self, *args, **kwds)

    def set_value(self, vals):
        """
        Set the port attributes on this arc.

        If these values are being reassigned, note that the defaults
        are still None, so you may need to repass some attributes.
        """
        if not self._constructed:
            raise ValueError("Setting the value of arc '%s' before "
                             "the Arc has been constructed (there "
                             "is currently no object to set)." % self.name)
        if len(self._data) == 0:
            self._data[None] = self
        try:
            super(SimpleArc, self).set_value(vals)
        except:
            # don't allow model walker to find poorly constructed arcs
            del self._data[None]
            raise

    def pprint(self, ostream=None, verbose=False, prefix=""):
        Arc.pprint(self, ostream=ostream, verbose=verbose, prefix=prefix)


class IndexedArc(Arc):
    def __init__(self, *args, **kwds):
        self._expanded_block = None
        super(IndexedArc, self).__init__(*args, **kwds)

    @property
    def expanded_block(self):
        # indexed block that contains all the blocks for this arc
        return self._expanded_block


register_component(Arc, "Component used for connecting two Ports.")
