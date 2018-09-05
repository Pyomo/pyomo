#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
logger = logging.getLogger('pyomo.network')

from six import next, iteritems, itervalues

from pyomo.common.modeling import unique_component_name

from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.core.base.indexed_component import UnindexedComponent_set
from pyomo.core.base import Transformation, Var, Block, SortComponents, TransformationFactory

from pyomo.network import Arc
from pyomo.network.util import replicate_var

# keyword arguments for component_objects and component_data_objects
obj_iter_kwds = dict(ctype=Arc, active=True, sort=SortComponents.deterministic)


@TransformationFactory.register('network.expand_arcs',
          doc="Expand all Arcs in the model to simple constraints")
class ExpandArcs(Transformation):

    def _apply_to(self, instance, **kwds):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Calling ArcExpander")

        # need to collect all ports to see every port each
        # is related to so that we can expand empty ports
        port_list, known_port_sets, matched_ports = \
            self._collect_ports(instance)

        self._add_blocks(instance)

        for port in port_list:
            # iterate over ref so that the index set is the same
            # for all occurences of this member in related ports
            # and so we iterate over members deterministically
            ref = known_port_sets[id(matched_ports[port])]
            for k, v in sorted(iteritems(ref)):
                rule, kwds = port._rules[k]
                if v[1] >= 0:
                    index_set = v[0].index_set()
                else:
                    index_set = UnindexedComponent_set
                rule(port, k, index_set, **kwds)

        for arc in instance.component_objects(**obj_iter_kwds):
            arc.deactivate()

    def _collect_ports(self, instance):
        self._name_buffer = {}
        # List of the ports in the order in which we found them
        # (this should be deterministic, provided that the user's model
        # is deterministic)
        port_list = []
        # ID of the next port group (set of matched ports)
        groupID = 0
        # port_groups stars out as a dict of {id(set): (groupID, set)}
        # If you sort by the groupID, then this will be deterministic.
        port_groups = dict()
        # map of port to the set of ports that must match it
        matched_ports = ComponentMap()

        for arc in instance.component_data_objects(**obj_iter_kwds):
            ports = ComponentSet(arc.ports)
            ref = None

            for p in arc.ports:
                if p in matched_ports:
                    if ref is None:
                        # The first port in this arc has
                        # already been seen. We will use that Set as
                        # the reference
                        ref = matched_ports[p]
                    elif ref is not matched_ports[p]:
                        # We already have a reference group; merge this
                        # new group into it.

                        # Optimization: this merge is linear in the size
                        # of the src set. If the reference set is
                        # smaller, save time by switching to a new
                        # reference set.
                        src = matched_ports[p]
                        if len(ref) < len(src):
                            ref, src = src, ref
                        ref.update(src)
                        for i in src:
                            matched_ports[i] = ref
                        del port_groups[id(src)]
                    # else: pass
                    #   The new group *is* the reference group;
                    #   there is nothing to do.
                else:
                    # The port has not been seen before.
                    port_list.append(p)
                    if ref is None:
                        # This is the first port in the arc:
                        # start a new reference set.
                        ref = ComponentSet()
                        port_groups[id(ref)] = (groupID, ref)
                        groupID += 1
                    # This port hasn't been seen. Record it.
                    ref.add(p)
                    matched_ports[p] = ref

        # Validate all port sets and expand the empty ones
        known_port_sets = {}
        for groupID, port_set in sorted(itervalues(port_groups)):
            known_port_sets[id(port_set)] \
                = self._validate_and_expand_port_set(port_set)

        return port_list, known_port_sets, matched_ports

    def _validate_and_expand_port_set(self, ports):
        ref = {}
        # First, go through the ports and get the superset of all fields
        for p in ports:
            for k, v in iteritems(p.vars):
                if k in ref:
                    # We have already seen this var
                    continue
                if v is None:
                    # This is an implicit var
                    continue
                # OK: New var, so add it to the reference list
                _len = (
                    -1 if not v.is_indexed()
                    else len(v))
                ref[k] = (v, _len, p, p.rule_for(k))

        if not ref:
            logger.warning(
                "Cannot identify a reference port: no ports "
                "in the port set have assigned variables:\n\t(%s)"
                % ', '.join(sorted(p.name for p in itervalues(ports))))
            return ref

        # Now make sure that ports match
        empty_or_partial = []
        for p in ports:
            p_is_partial = False
            if not p.vars:
                # This is an empty port and should be defined with
                # "auto" vars
                empty_or_partial.append(p)
                continue

            for k, v in iteritems(ref):
                if k not in p.vars:
                    raise ValueError(
                        "Port mismatch: Port '%s' missing variable "
                        "'%s' (appearing in reference port '%s')" %
                        (p.name, k, v[2].name))
                _v = p.vars[k]
                if _v is None:
                    if not p_is_partial:
                        empty_or_partial.append(p)
                        p_is_partial = True
                    continue
                _len = (
                    -1 if not _v.is_indexed()
                    else len(_v))
                if (_len >= 0) ^ (v[1] >= 0):
                    raise ValueError(
                        "Port mismatch: Port variable '%s' mixing "
                        "indexed and non-indexed targets on ports '%s' "
                        "and '%s'" %
                        (k, v[2].name, p.name))
                if _len >= 0 and _len != v[1]:
                    raise ValueError(
                        "Port mismatch: Port variable '%s' index "
                        "mismatch (%s elements in reference port '%s', "
                        "but %s elements in port '%s')" %
                        (k, v[1], v[2].name, _len, p.name))
                if v[1] >= 0 and len(v[0].index_set() ^ _v.index_set()):
                    raise ValueError(
                        "Port mismatch: Port variable '%s' has "
                        "mismatched indices on ports '%s' and '%s'" %
                        (k, v[2].name, p.name))
                if p.rule_for(k) is not v[3]:
                    raise ValueError(
                        "Port mismatch: Port variable '%s' has "
                        "different rules on ports '%s' and '%s'" %
                        (k, v[2].name, p.name))

        # as we are adding things to the model, sort by key so that
        # the order things are added is deterministic
        sorted_refs = sorted(iteritems(ref))
        if len(empty_or_partial) > 1:
            # This is expensive (names aren't cheap), but does result in
            # a deterministic ordering
            empty_or_partial.sort(key=lambda x: x.getname(
                fully_qualified=True, name_buffer=self._name_buffer))

        # Fill in any empty ports
        for p in empty_or_partial:
            block = p.parent_block()
            for k, v in sorted_refs:
                if k in p.vars and p.vars[k] is not None:
                    continue

                vname = unique_component_name(
                    block, '%s_auto_%s' % (p.getname(
                        fully_qualified=True, name_buffer=self._name_buffer),k))

                new_var = replicate_var(v[0], vname, block)

                # add this new variable to the port so that it has a rule
                p.add(new_var, k, rule=v[3])

        return ref

    def _add_blocks(self, instance):
        # iterate over component_objects so we can make indexed blocks
        for arc in instance.component_objects(**obj_iter_kwds):
            blk = Block(arc.index_set())
            bname = unique_component_name(
                arc.parent_block(), "%s_expanded" % arc.local_name)
            arc.parent_block().add_component(bname, blk)
            arc._expanded_block = blk
            if arc.is_indexed():
                for i in arc:
                    arc[i]._expanded_block = blk[i]
