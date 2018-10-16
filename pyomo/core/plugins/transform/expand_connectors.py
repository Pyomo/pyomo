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
logger = logging.getLogger('pyomo.core')

from six import next, iteritems, itervalues

from pyomo.core.expr import current as EXPR
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.core.base import Transformation, TransformationFactory, Connector, Constraint, \
    ConstraintList, Var, VarList, TraversalStrategy, SortComponents
from pyomo.core.base.connector import _ConnectorData, SimpleConnector


@TransformationFactory.register('core.expand_connectors', 
          doc="Expand all connectors in the model to simple constraints")
class ExpandConnectors(Transformation):

    def _apply_to(self, instance, **kwds):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            logger.debug("Calling ConnectorExpander")

        connectorsFound = False
        for c in instance.component_data_objects(Connector):
            connectorsFound = True
            break
        if not connectorsFound:
            return

        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            logger.debug("   Connectors found!")

        self._name_buffer = {}

        #
        # At this point, there are connectors in the model, so we must
        # look for constraints that involve connectors and expand them.
        #
        # List of the connectors in the order in which we found them
        # (this should be deterministic, provided that the user's model
        # is deterministic)
        connector_list = []
        # list of constraints with connectors: tuple(constraint, connector_set)
        # (this should be deterministic, provided that the user's model
        # is deterministic)
        constraint_list = []
        # ID of the next connector group (set of matched connectors)
        groupID = 0
        # connector_groups stars out as a dict of {id(set): (groupID, set)}
        # If you sort by the groupID, then this will be deterministic.
        connector_groups = dict()
        # map of connector to the set of connectors that must match it
        matched_connectors = ComponentMap()
        # The set of connectors found in the current constraint
        found = ComponentSet()

        connector_types = set([SimpleConnector, _ConnectorData])
        for constraint in instance.component_data_objects(
                Constraint, sort=SortComponents.deterministic):
            ref = None
            for c in EXPR.identify_components(constraint.body, connector_types):
                found.add(c)
                if c in matched_connectors:
                    if ref is None:
                        # The first connector in this constraint has
                        # already been seen.  We will use that Set as
                        # the reference
                        ref = matched_connectors[c]
                    elif ref is not matched_connectors[c]:
                        # We already have a reference group; merge this
                        # new group into it.
                        #
                        # Optimization: this merge is linear in the size
                        # of the src set.  If the reference set is
                        # smaller, save time by switching to a new
                        # reference set.
                        src = matched_connectors[c]
                        if len(ref) < len(src):
                            ref, src = src, ref
                        ref.update(src)
                        for _ in src:
                            matched_connectors[_] = ref
                        del connector_groups[id(src)]
                    # else: pass
                    #   The new group *is* the reference group;
                    #   there is nothing to do.
                else:
                    # The connector has not been seen before.
                    connector_list.append(c)
                    if ref is None:
                        # This is the first connector in the constraint:
                        # start a new reference set.
                        ref = ComponentSet()
                        connector_groups[id(ref)] = (groupID, ref)
                        groupID += 1
                    # This connector hasn't been seen.  Record it.
                    ref.add(c)
                    matched_connectors[c] = ref
            if ref is not None:
                constraint_list.append((constraint, found))
                found = ComponentSet()

        # Validate all connector sets and expand the empty ones
        known_conn_sets = {}
        for groupID, conn_set in sorted(itervalues(connector_groups)):
            known_conn_sets[id(conn_set)] \
                = self._validate_and_expand_connector_set(conn_set)

        # Expand each constraint
        for constraint, conn_set in constraint_list:
            cList = ConstraintList()
            constraint.parent_block().add_component(
                '%s.expanded' % ( constraint.getname(
                    fully_qualified=False, name_buffer=self._name_buffer), ),
                cList )
            connId = next(iter(conn_set))
            ref = known_conn_sets[id(matched_connectors[connId])]
            for k,v in sorted(iteritems(ref)):
                if v[1] >= 0:
                    _iter = v[0]
                else:
                    _iter = (v[0],)
                for idx in _iter:
                    substitution = {}
                    for c in conn_set:
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

        # Now, go back and implement VarList aggregators
        for conn in connector_list:
            block = conn.parent_block()
            for var, aggregator in iteritems(conn.aggregators):
                c = Constraint(expr=aggregator(block, conn.vars[var]))
                block.add_component(
                    '%s.%s.aggregate' % (
                        conn.getname(
                            fully_qualified=True,
                            name_buffer=self._name_buffer),
                        var), c )


    def _validate_and_expand_connector_set(self, connectors):
        ref = {}
        # First, go through the connectors and get the superset of all fields
        for c in connectors:
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
                "Cannot identify a reference connector: no connectors "
                "in the connector set have assigned variables:\n\t(%s)"
                % ( ', '.join(sorted(c.name for c in connectors)), ))
            return ref

        # Now make sure that connectors match
        empty_or_partial = []
        for c in connectors:
            c_is_partial = False
            if not c.vars:
                # This is an empty connector and should be defined with
                # "auto" vars
                empty_or_partial.append(c)
                continue

            for k,v in iteritems(ref):
                if k not in c.vars:
                    raise ValueError(
                        "Connector mismatch: Connector '%s' missing variable "
                        "'%s' (appearing in reference connector '%s')" %
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
                        "Connector mismatch: Connector variable '%s' mixing "
                        "indexed and non-indexed targets on connectors '%s' "
                        "and '%s'" %
                        ( k, v[2].name, c.name ))
                if _len >= 0 and _len != v[1]:
                    raise ValueError(
                        "Connector mismatch: Connector variable '%s' index "
                        "mismatch (%s elements in reference connector '%s', "
                        "but %s elements in connector '%s')" %
                        ( k, v[1], v[2].name, _len, c.name ))
                if v[1] >= 0 and len(v[0].index_set() ^ _v.index_set()):
                    raise ValueError(
                        "Connector mismatch: Connector variable '%s' has "
                        "mismatched indices on connectors '%s' and '%s'" %
                        ( k, v[2].name, c.name ))


        # as we are adding things to the model, sort by key so that
        # the order things are added is deterministic
        sorted_refs = sorted(iteritems(ref))
        if len(empty_or_partial) > 1:
            # This is expensive (names aren't cheap), but does result in
            # a deterministic ordering
            empty_or_partial.sort(key=lambda x: x.getname(
                fully_qualified=True, name_buffer=self._name_buffer))

        # Fill in any empty connectors
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
                block.add_component(
                    '%s.auto.%s' % (
                        c.getname(fully_qualified=True,
                                  name_buffer=self._name_buffer), k ),
                    new_var)
                if idx:
                    for i in idx[0]:
                        new_var[i].domain = v[0][i].domain
                        new_var[i].setlb( v[0][i].lb )
                        new_var[i].setub( v[0][i].ub )
                c.vars[k] = new_var

        return ref
