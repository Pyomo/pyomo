#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import logging
logger = logging.getLogger('pyomo.core')

from six import next, iteritems, iterkeys, itervalues

from pyomo.core.base.plugin import alias
from pyomo.core.base import Transformation, Connector, Constraint, \
    ConstraintList, Var, VarList, TraversalStrategy
from pyomo.core.base.connector import _ConnectorData, SimpleConnector
from pyomo.core.base import expr

class ExpandConnectors(Transformation):
    alias('core.expand_connectors', 
          doc="Expand all connectors in the model to simple constraints")

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

        #
        # At this point, there are connectors in the model, so we must
        # look for constraints that involve connectors and expand them.
        #
        connector_types = set([SimpleConnector, _ConnectorData])
        constraint_list = []
        connector_list = []
        matched_connectors = {}
        found = dict()
        for constraint in instance.component_data_objects(Constraint):
            for c in expr.identify_variables(
                    constraint.body, include_potentially_variable=True):
                if c.__class__ in connector_types:
                    found[id(c)] = c
            if not found:
                continue

            # Note that it is important to copy the set of found
            # connectors, since the matching routine below will
            # manipulate sets in place.
            found_this_constraint = dict(found)
            constraint_list.append( (constraint, found_this_constraint) )

            # Find all the connectors that are used in the constraint,
            # so we know which connectors to validate against each
            # other.  Note that the validation must be transitive (that
            # is, if con1 has a & b and con2 has b & c, then a,b, and c
            # must all validate against each other.
            for cId, c in iteritems(found_this_constraint):
                if cId in matched_connectors:
                    oldSet = matched_connectors[cId]
                    found.update( oldSet )
                    for _cId in oldSet:
                        matched_connectors[_cId] = found
                else:
                    connector_list.append(c)
                matched_connectors[cId] = found

            # Reset found back to empty (this is more efficient as the
            # bulk of the constraints in the model will not have
            # connectors - so if we did this at the top of the loop, we
            # would spend a lot of time clearing empty sets
            found = {}

        # Validate all connector sets and expand the empty ones
        known_conn_sets = {}
        for connector in connector_list:
            conn_set = matched_connectors[id(connector)]
            if id(conn_set) in known_conn_sets:
                continue
            known_conn_sets[id(conn_set)] \
                = self._validate_and_expand_connector_set(conn_set)

        # Expand each constraint
        for constraint, conn_set in constraint_list:
            cList = ConstraintList()
            constraint.parent_block().add_component(
                '%s.expanded' % ( constraint.local_name, ), cList )
            connId = next(iterkeys(conn_set))
            ref = known_conn_sets[id(matched_connectors[connId])]
            for k,v in sorted(iteritems(ref)):
                if v[1] >= 0:
                    _iter = v[0]
                else:
                    _iter = (v[0],)
                for idx in _iter:
                    substitution = {}
                    for c in itervalues(conn_set):
                        if v[1] >= 0:
                            new_v = c.vars[k][idx]
                        elif k in c.aggregators:
                            new_v = c.vars[k].add()
                        else:
                            new_v = c.vars[k]
                        substitution[id(c)] = new_v
                    cList.add((
                        constraint.lower,
                        expr.clone_expression( constraint.body, substitution ),
                        constraint.upper ))
            constraint.deactivate()

        # Now, go back and implement VarList aggregators
        for conn in connector_list:
            block = conn.parent_block()
            for var, aggregator in iteritems(conn.aggregators):
                c = Constraint(expr=aggregator(block, conn.vars[var]))
                block.add_component(
                    '%s.%s.aggregate' % (conn.local_name, var), c )


    def _validate_and_expand_connector_set(self, connectors):
        ref = {}
        # First, go through the connectors and get the superset of all fields
        for c in itervalues(connectors):
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
                    -1 if v.is_expression() or not v.is_indexed() else
                    len(v) )
                ref[k] = ( v, _len, c )

        if not ref:
            logger.warning(
                "Cannot identify a reference connector: no connectors "
                "in the connector set have assigned variables:\n\t(%s)"
                % ( ', '.join(sorted(c.name for c in connectors)), ))
            return ref

        # Now make sure that connectors match
        empty_or_partial = []
        for c in itervalues(connectors):
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
                    -1 if _v.is_expression() or not _v.is_indexed() else
                    len(_v) )
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
            empty_or_partial.sort(key=lambda x: x.name)

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
                block.add_component('%s.auto.%s' % ( c.local_name, k ), new_var)
                if idx:
                    for i in idx[0]:
                        new_var[i].domain = v[0][i].domain
                        new_var[i].setlb( v[0][i].lb )
                        new_var[i].setub( v[0][i].ub )
                c.vars[k] = new_var

        return ref
