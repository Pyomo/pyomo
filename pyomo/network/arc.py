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

from pyomo.network.port import Port
from pyomo.core.base.component import ActiveComponentData
from pyomo.core.base.indexed_component import (ActiveIndexedComponent,
    UnindexedComponent_set)
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.common.log import is_debug_set
from pyomo.common.timing import ConstructionTimer
from six import iteritems
from weakref import ref as weakref_ref
import logging, sys

logger = logging.getLogger('pyomo.network')


def _iterable_to_dict(vals, directed, name):
    if type(vals) is not dict:
        # check that it's a two-member iterable
        try:
            ports = tuple(vals)
        except TypeError:
            ports = None
        if ports is None or len(ports) != 2:
            raise ValueError(
                "Value for arc '%s' is not either a "
                "dict or a two-member iterable." % name)
        if directed:
            source, destination = ports
            ports = None
        else:
            source = destination = None
        vals = dict(source=source, destination=destination,
                    ports=ports, directed=directed)
    elif "directed" not in vals:
        vals["directed"] = directed
    return vals


class _ArcData(ActiveComponentData):
    """
    This class defines the data for a single Arc

    Attributes
    ----------
        source: `Port`
            The source Port when directed, else None. Aliases to src.
        destination: `Port`
            The destination Port when directed, else None. Aliases to dest.
        ports: `tuple`
            A tuple containing both ports. If directed, this is in the
            order (source, destination).
        directed: `bool`
            True if directed, False if not
        expanded_block: `Block`
            A reference to the block on which expanded constraints for this
            arc were placed
    """

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
        """Returns `self.expanded_block.name` if it exists"""
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
            self._directed and self._ports is not None
            ) else None

    src = source

    @property
    def destination(self):
        return self._ports[1] if (
            self._directed and self._ports is not None
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
        """Set the port attributes on this arc"""
        # the following allows m.a = Arc(directed=True); m.a = (m.p, m.q)
        # and m.a will be directed
        d = self._directed if self._directed is not None else \
            self.parent_component()._init_directed

        vals = _iterable_to_dict(vals, d, self.name)

        source = vals.pop("source", None)
        destination = vals.pop("destination", None)
        ports = vals.pop("ports", None)
        directed = vals.pop("directed", None)

        if len(vals):
            raise ValueError(
                "set_value passed unrecognized keywords in val:\n\t" +
                "\n\t".join("%s = %s" % (k, v) for k, v in iteritems(vals)))

        if directed is not None:
            if source is None and destination is None:
                if directed and ports is not None:
                    # implicitly directed ports tuple, transfer to src and dest
                    try:
                        source, destination = ports
                        ports = None
                    except:
                        raise ValueError(
                            "Failed to unpack 'ports' argument of arc '%s'. "
                            "Argument must be a 2-member tuple or list."
                            % self.name)
            elif not directed:
                # throw an error if they gave an inconsistent directed value
                raise ValueError(
                    "Passed False value for 'directed' for arc '%s', but "
                    "specified source or destination." % self.name)

        self._validate_ports(source, destination, ports)

        if self.ports is not None:
            # we are reassigning this arc's values, clean up port lists
            weakref_self = weakref_ref(self)
            for port in self.ports:
                port._arcs.remove(weakref_self)
            if self._directed:
                self.source._dests.remove(weakref_self)
                self.destination._sources.remove(weakref_self)

        self._ports = tuple(ports) if ports is not None \
            else (source, destination)
        self._directed = source is not None
        weakref_self = weakref_ref(self)
        for port in self._ports:
            port._arcs.append(weakref_self)
        if self._directed:
            source._dests.append(weakref_self)
            destination._sources.append(weakref_self)

    def _validate_ports(self, source, destination, ports):
        msg = "Arc %s: " % self.name
        if ports is not None:
            if source is not None or destination is not None:
                raise ValueError(msg +
                    "cannot specify 'source' or 'destination' "
                    "when using 'ports' argument.")
            if (type(ports) not in (list, tuple) or len(ports) != 2):
                raise ValueError(msg +
                    "argument 'ports' must be list or tuple "
                    "containing exactly 2 Ports.")
            for p in ports:
                try:
                    if p.ctype is not Port:
                        raise ValueError(msg +
                            "found object '%s' in 'ports' not "
                            "of type Port." % p.name)
                    elif p.is_indexed():
                        raise ValueError(msg +
                            "found indexed Port '%s' in 'ports', must "
                            "use single Ports for Arc." % p.name)
                except AttributeError:
                    raise ValueError(msg +
                        "found object '%s' in 'ports' not "
                        "of type Port." % str(p))
        else:
            if source is None or destination is None:
                raise ValueError(msg +
                    "must specify both 'source' and 'destination' "
                    "for directed Arc.")
            for p, side in [(source, "source"), (destination, "destination")]:
                try:
                    if p.ctype is not Port:
                        raise ValueError(msg +
                            "%s object '%s' not of type Port." % (p.name, side))
                    elif p.is_indexed():
                        raise ValueError(msg +
                            "found indexed Port '%s' as %s, must use "
                            "single Ports for Arc." % (source.name, side))
                except AttributeError:
                    raise ValueError(msg +
                        "%s object '%s' not of type Port." % (str(p), side))


@ModelComponentFactory.register("Component used for connecting two Ports.")
class Arc(ActiveIndexedComponent):
    """
    Component used for connecting the members of two Port objects

    Parameters
    ----------
        source: `Port`
            A single Port for a directed arc. Aliases to src.
        destination: `Port`
            A single`Port for a directed arc. Aliases to dest.
        ports
            A two-member list or tuple of single Ports for an undirected arc
        directed: `bool`
            Set True for directed. Use along with `rule` to be able to
            return an implied (source, destination) tuple.
        rule: `function`
            A function that returns either a dictionary of the arc arguments
            or a two-member iterable of ports
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
        self._init_directed = kwds.pop("directed", None)
        self._rule = kwds.pop('rule', None)
        kwds.setdefault("ctype", Arc)

        super(Arc, self).__init__(*args, **kwds)

        if source is None and destination is None and ports is None:
            self._init_vals = None
        else:
            self._init_vals = dict(
                source=source, destination=destination, ports=ports)

    def construct(self, data=None):
        """Initialize the Arc"""
        if is_debug_set(logger):
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
                tmp["directed"] = self._init_directed
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
                tmp = _iterable_to_dict(tmp, self._init_directed, self.name)
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
                tmp = _iterable_to_dict(tmp, self._init_directed, self.name)
                self._setitem_when_not_present(idx, tmp)
        timer.report()

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


class IndexedArc(Arc):
    def __init__(self, *args, **kwds):
        self._expanded_block = None
        super(IndexedArc, self).__init__(*args, **kwds)

    @property
    def expanded_block(self):
        # indexed block that contains all the blocks for this arc
        return self._expanded_block


