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
from collections import OrderedDict

from pyomo.core.expr import current as EXPR
from pyomo.core.kernel import ComponentMap, ComponentSet
from pyomo.core.base.plugin import alias
from pyomo.core.base import Transformation, Constraint, ConstraintList, \
    Var, VarList, Block, SortComponents
from pyomo.network.port import Port, _PortData, SimplePort
from pyomo.network.arc import Arc, SimpleArc
from pyomo.common.modeling import unique_component_name


class _PortExpansion(Transformation):
    def _collect_ports(self, instance, ctype):
        self._name_buffer = {}
        # List of the ports in the order in which we found them
        # (this should be deterministic, provided that the user's model
        # is deterministic)
        port_list = []
        # list of constraints with ports: tuple(constraint, port_set)
        # (this should be deterministic, provided that the user's model
        # is deterministic)
        constraint_list = []
        # analogous to constraint_list
        arc_list = []
        # ID of the next port group (set of matched ports)
        groupID = 0
        # port_groups stars out as a dict of {id(set): (groupID, set)}
        # If you sort by the groupID, then this will be deterministic.
        port_groups = dict()
        # map of port to the set of ports that must match it
        matched_ports = ComponentMap()
        # The set of ports found in the current component
        found = ComponentSet()

        port_types = set([SimplePort, _PortData])
        for comp in instance.component_data_objects(
                ctype, sort=SortComponents.deterministic, active=True):
            if comp.type() is Constraint:
                itr = EXPR.identify_components(comp.body, port_types)
            else: # Arc
                itr = comp.ports
            ref = None
            for c in itr:
                found.add(c)
                if c in matched_ports:
                    if ref is None:
                        # The first port in this comp has
                        # already been seen.  We will use that Set as
                        # the reference
                        ref = matched_ports[c]
                    elif ref is not matched_ports[c]:
                        # We already have a reference group; merge this
                        # new group into it.

                        # Optimization: this merge is linear in the size
                        # of the src set.  If the reference set is
                        # smaller, save time by switching to a new
                        # reference set.
                        src = matched_ports[c]
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
                    port_list.append(c)
                    if ref is None:
                        # This is the first port in the comp:
                        # start a new reference set.
                        ref = ComponentSet()
                        port_groups[id(ref)] = (groupID, ref)
                        groupID += 1
                    # This port hasn't been seen.  Record it.
                    ref.add(c)
                    matched_ports[c] = ref
            if ref is not None:
                if comp.type() is Constraint:
                    constraint_list.append( (comp, found) )
                else:
                    arc_list.append( (comp, found) )
                found = ComponentSet()

        # Validate all port sets and expand the empty ones
        known_port_sets = {}
        for groupID, port_set in sorted(itervalues(port_groups)):
            known_port_sets[id(port_set)] \
                = self._validate_and_expand_port_set(port_set)

        return (constraint_list, arc_list, port_list,
                matched_ports, known_port_sets)

    def _validate_and_expand_port_set(self, ports):
        ref = {}
        # First, go through the ports and get the superset of all fields
        for c in ports:
            for k,v in iteritems(c.vars):
                if k in ref:
                    # We have already seen this var
                    continue
                if v is None:
                    # This is an implicit var
                    continue
                # OK: New var, so add it to the reference list
                _len = (
                    #-3 if v is None else
                    -2 if k in c.aggregators else
                    -1 if not hasattr(v, 'is_indexed') or not v.is_indexed()
                    else len(v) )
                ref[k] = ( v, _len, c )

        if not ref:
            logger.warning(
                "Cannot identify a reference port: no ports "
                "in the port set have assigned variables:\n\t(%s)"
                % ', '.join(sorted(c.name for c in itervalues(ports))))
            return ref

        # Now make sure that ports match
        empty_or_partial = []
        for c in ports:
            c_is_partial = False
            if not c.vars:
                # This is an empty port and should be defined with
                # "auto" vars
                empty_or_partial.append(c)
                continue

            for k,v in iteritems(ref):
                if k not in c.vars:
                    raise ValueError(
                        "Port mismatch: Port '%s' missing variable "
                        "'%s' (appearing in reference port '%s')" %
                        ( c.name, k, v[2].name ) )
                _v = c.vars[k]
                if _v is None:
                    if not c_is_partial:
                        empty_or_partial.append(c)
                        c_is_partial = True
                    continue
                _len = (
                    -3 if _v is None else
                    -2 if k in c.aggregators else
                    -1 if not hasattr(_v, 'is_indexed') or not _v.is_indexed()
                    else len(_v) )
                if (_len >= 0) ^ (v[1] >= 0):
                    raise ValueError(
                        "Port mismatch: Port variable '%s' mixing "
                        "indexed and non-indexed targets on ports '%s' "
                        "and '%s'" %
                        ( k, v[2].name, c.name ))
                if _len >= 0 and _len != v[1]:
                    raise ValueError(
                        "Port mismatch: Port variable '%s' index "
                        "mismatch (%s elements in reference port '%s', "
                        "but %s elements in port '%s')" %
                        ( k, v[1], v[2].name, _len, c.name ))
                if v[1] >= 0 and len(v[0].index_set() ^ _v.index_set()):
                    raise ValueError(
                        "Port mismatch: Port variable '%s' has "
                        "mismatched indices on ports '%s' and '%s'" %
                        ( k, v[2].name, c.name ))


        # as we are adding things to the model, sort by key so that
        # the order things are added is deterministic
        sorted_refs = sorted(iteritems(ref))
        if len(empty_or_partial) > 1:
            # This is expensive (names aren't cheap), but does result in
            # a deterministic ordering
            empty_or_partial.sort(key=lambda x: x.getname(
                fully_qualified=True, name_buffer=self._name_buffer))

        # Fill in any empty ports
        for c in empty_or_partial:
            block = c.parent_block()
            for k, v in sorted_refs:
                if k in c.vars and c.vars[k] is not None:
                    continue

                if v[1] >= 0:
                    idx = ( v[0].index_set(), )
                else:
                    idx = ()
                var_args = {}
                try:
                    var_args['domain'] = v[0].domain
                except AttributeError:
                    pass
                try:
                    var_args['bounds'] = v[0].bounds
                except AttributeError:
                    pass
                new_var = Var( *idx, **var_args )
                vname = '%s.auto.%s' % (c.getname(
                    fully_qualified=True, name_buffer=self._name_buffer), k)
                block.add_component(vname, new_var)
                if idx:
                    for i in idx[0]:
                        new_var[i].domain = v[0][i].domain
                        new_var[i].setlb( v[0][i].lb )
                        new_var[i].setub( v[0][i].ub )
                c.vars[k] = new_var

        return ref

    def _build_arcs(self, arc_list, matched_ports,
                           known_port_sets):
        indexed_ctns = OrderedDict() # maintain deterministic order we have
        for ctn, port_set in arc_list:
            if not isinstance(ctn, SimpleArc):
                # create indexed blocks later for indexed arcs
                lst = indexed_ctns.get(ctn.parent_component(), [])
                lst.append( (ctn, port_set) )
                indexed_ctns[ctn.parent_component()] = lst
                continue
            blk = Block()
            bname = unique_component_name(
                ctn.parent_block(), "%s_expanded" % ctn.getname(
                    fully_qualified=False, name_buffer=self._name_buffer))
            ctn.parent_block().add_component(bname, blk)
            # add reference to this block onto the Arc object
            ctn._expanded_block = blk
            self._add_arcs(
                blk, port_set, matched_ports, known_port_sets)
            ctn.deactivate()

        for ictn in indexed_ctns:
            blk = Block(ictn.index_set())
            bname = unique_component_name(
                ictn.parent_block(), "%s_expanded" % ictn.getname(
                    fully_qualified=False, name_buffer=self._name_buffer))
            ictn.parent_block().add_component(bname, blk)
            ictn._expanded_block = blk
            for ctn, port_set in indexed_ctns[ictn]:
                i = ctn.index()
                self._add_arcs(
                    blk[i], port_set, matched_ports, known_port_sets)
                ctn._expanded_block = blk
            ictn.deactivate()

    def _add_arcs(self, blk, port_set, matched_ports,
                         known_port_sets):
        if len(port_set) == 1:
            # possible to have a arc equating a port to itself
            # emit the trivial constraint, as opposed to skipping it
            # port_set is a set, so make a list that contains itself repeated
            port_set = [k for k in port_set] * 2
        port = next(iter(port_set))
        ref = known_port_sets[id(matched_ports[port])]
        for k, v in sorted(iteritems(ref)):
            # if one of them is extensive, make the new variable
            # if both are, skip the constraint since both use the same var
            # name is k, conflicts are prevented by a check in add function
            # the new var will mirror the original var and have same index set
            cont = once = False
            for c in port_set:
                for etype in c.extensives:
                    if k in c.extensives[etype]:
                        if once:
                            cont = True
                            c.extensives[etype][k].append(evar)
                        else:
                            once = True
                            evar = Var(c.vars[k].index_set())
                            blk.add_component(k, evar)
                            c.extensives[etype][k].append(evar)
                        break
            if cont:
                continue

            cname = k + "_equality"
            if v[1] >= 0:
                # v[0] is an indexed var
                def rule(m, *args):
                    tmp = []
                    for c in port_set:
                        e = False
                        for etype in c.extensives:
                            if k in c.extensives[etype]:
                                e = True
                                tmp.append(evar[args])
                                break
                        if not e:
                            tmp.append(c.vars[k][args])
                    return tmp[0] == tmp[1]
                con = Constraint(v[0].index_set(), rule=rule)
            else:
                tmp = []
                for c in port_set:
                    if k in c.aggregators:
                        tmp.append(c.vars[k].add())
                    elif k in c.extensives:
                        tmp.append(evar)
                    else:
                        tmp.append(c.vars[k])
                con = Constraint(expr=tmp[0] == tmp[1])
            blk.add_component(cname, con)

    def _implement_aggregators(self, port_list):
        for port in port_list:
            block = port.parent_block()
            for var, aggregator in iteritems(port.aggregators):
                c = Constraint(expr=aggregator(block, port.vars[var]))
                cname = '%s.%s.aggregate' % (port.getname(
                    fully_qualified=True, name_buffer=self._name_buffer), var)
                block.add_component(cname, c)

    def _implement_extensives(self, port_list):
        for ctr in port_list:
            unit = ctr.parent_block()
            for etype in ctr.extensives:
                if etype not in ctr.extensive_aggregators:
                    raise KeyError(
                        "No aggregator in extensive_aggregators for extensive "
                        "type '%s' in Port '%s'" % (etype, ctr.name))
                fcn = ctr.extensive_aggregators[etype]
                # build list of arcs using the parent blocks of all
                # the evars in one of the lists in ctr.extensives[etype]
                ctns = [evar.parent_block() for evar in
                        next(itervalues(ctr.extensives[etype]))]
                fcn(unit, ctns, ctr, etype)


class ExpandPorts(_PortExpansion):
    alias('network.expand_ports',
          doc="Expand all ports in the model to simple constraints")

    def _apply_to(self, instance, **kwds):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            logger.debug("Calling PortExpander")

        portsFound = False
        for c in instance.component_data_objects(Port):
            portsFound = True
            break
        if not portsFound:
            return

        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            logger.debug("   Ports found!")

        #
        # At this point, there are ports in the model, so we must
        # look for constraints that involve ports and expand them.
        #
        (constraint_list, arc_list, port_list, matched_ports,
            known_port_sets) = self._collect_ports(instance,
            (Constraint, Arc))

        # Expand each constraint
        for constraint, port_set in constraint_list:
            cList = ConstraintList()
            cname = '%s.expanded' % constraint.getname(
                fully_qualified=False, name_buffer=self._name_buffer)
            constraint.parent_block().add_component(cname, cList)
            portId = next(iter(port_set))
            ref = known_port_sets[id(matched_ports[portId])]
            for k,v in sorted(iteritems(ref)):
                if v[1] >= 0:
                    _iter = v[0]
                else:
                    _iter = (v[0],)
                for idx in _iter:
                    substitution = {}
                    for c in port_set:
                        if v[1] >= 0:
                            new_v = c.vars[k][idx]
                        elif k in c.aggregators:
                            new_v = c.vars[k].add()
                        else:
                            new_v = c.vars[k]
                        substitution[id(c)] = new_v
                    cList.add((
                        constraint.lower,
                        EXPR.clone_expression( constraint.body, substitution ),
                        constraint.upper ))
            constraint.deactivate()

        self._build_arcs(arc_list, matched_ports,
            known_port_sets)

        # Now, go back and implement VarList aggregators
        self._implement_aggregators(port_list)


class ExpandArcs(_PortExpansion):
    alias('network.expand_arcs',
          doc="Expand all Arcs in the model to simple constraints")

    def _apply_to(self, instance, **kwds):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            logger.debug("Calling ArcExpander")

        # need to collect all ports to see every port each
        # is related to so that we can expand empty ports
        (_, arc_list, port_list, matched_ports,
            known_port_sets) = self._collect_ports(instance, Arc)

        self._build_arcs(arc_list, matched_ports,
            known_port_sets)

        # Now, go back and implement aggregators
        self._implement_aggregators(port_list)
        self._implement_extensives(port_list)
