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

from six import iteritems, iterkeys, itervalues

from pyomo.core.base.plugin import alias
from pyomo.core.base import Transformation, Connector, Constraint, ConstraintList, VarList, TraversalStrategy
from pyomo.core.base.connector import _ConnectorData
from pyomo.core.base.expr import clone_expression

class ExpandConnectors(Transformation):
    alias('core.expand_connectors', 
          doc="Expand all connectors in the model to simple constraints")

    def __init__(self):
        super(ExpandConnectors, self).__init__()

    def _apply_to(self, instance, **kwds):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            logger.debug("Calling ConnectorExpander")
                
        noConnectors = True
        for b in instance.block_data_objects(active=True):
            if b.component_map(Connector):
                noConnectors = False
                break
        if noConnectors:
            return

        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            logger.debug("   Connectors found!")

        #
        # At this point, there are connectors in the model, so we must
        # look for constraints that involve connectors and expand them.
        #
        #options = kwds['options']
        #model = kwds['model']

        # In general, blocks should be relatively self-contained, so we
        # should build the connectors from the "bottom up":
        blockList = list(instance.block_data_objects(
            active=True, 
            descent_order=TraversalStrategy.PostfixDepthFirstSearch ))

        # Expand each constraint involving a connector
        for block in blockList:
            if __debug__ and logger.isEnabledFor(logging.DEBUG): #pragma:nocover
                logger.debug("   block: " + block.cname())

            CCC = []
            for constraint in block.component_objects(
                    (Constraint, ConstraintList)):
                name = constraint.name
                cList = []
                CCC.append((name+'.expanded', cList))
                for idx, c in iteritems(constraint):
                    if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
                        logger.debug("   (looking at constraint %s[%s])", name, idx)
                    connectors = []
                    self._gather_connectors(c.body, connectors)
                    if len(connectors) == 0:
                        continue
                    if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
                        logger.debug("   (found connectors in constraint)")
                    
                    # Validate that all connectors match
                    errors, ref, skip = self._validate_connectors(connectors)
                    if errors:
                        logger.error(
                            ( "Connector mismatch: errors detected when "
                              "constructing constraint %s\n    " %
                              (name + (idx and '[%s]' % idx or '')) ) +
                            '\n    '.join(reversed(errors)) )
                        raise ValueError(
                            "Connector mismatch in constraint %s" % \
                            name + (idx and '[%s]' % idx or ''))
                    
                    if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
                        logger.debug("   (connectors valid)")

                    # Fill in any empty connectors
                    for conn in connectors:
                        if conn.vars:
                            continue
                        for var in ref.vars:
                            if var in skip:
                                continue
                            v = Var()
                            block.add_component(conn.cname() + '.auto.' + var, v)
                            conn.vars[var] = v
                            v.construct()
                    
                    # OK - expand this constraint
                    self._expand_constraint(block, name, idx, c, ref, skip, cList, connectors)
                    # Now deactivate the original constraint
                    c.deactivate()
            for name, exprs in CCC:
                if not exprs:
                    continue
                cList = ConstraintList()
                block.add_component( name, cList )
                #cList.construct()
                for expr in exprs:
                    cList.add(expr)
                

        # Now, go back and implement VarList aggregators
        for block in blockList:
            for conn in itervalues(block.component_map(Connector)):
                for var, aggregator in iteritems(conn.aggregators):
                    c = Constraint(expr=aggregator(block, var))
                    block.add_component(
                        conn.cname() + '.' + var.cname() + '.aggregate', c)
                    c.construct()

    def _gather_connectors(self, expr, connectors):
        if expr.is_expression():
            if expr.__class__ is _ProductExpression:
                for e in expr._numerator:
                    self._gather_connectors(e, connectors)
                for e in expr._denominator:
                    self._gather_connectors(e, connectors)
            else:
                for e in expr._args:
                    self._gather_connectors(e, connectors)
        elif isinstance(expr, _ConnectorData):
            connectors.append(expr)

    def _validate_connectors(self, connectors):
        errors = []
        ref = None
        skip = set()
        for idx in xrange(len(connectors)):
            if connectors[idx].vars.keys():
                ref = connectors.pop(idx)
                break
        if ref is None:
            errors.append(
                "Cannot identify a reference connector: no connectors "
                "have assigned variables" )
            return errors, ref, skip

        a = set(ref.vars.keys())
        for key, val in iteritems(ref.vars):
            if val is None:
                skip.add(key)
        for tmp in connectors:
            b = set(tmp.vars.keys())
            if not b:
                continue
            for key, val in iteritems(tmp.vars):
                if val is None:
                    skip.add(key)
            for var in a - b:
                # TODO: add a fq_name so we can easily get
                # the full model.block.connector name
                errors.append(
                    "Connector '%s' missing variable '%s' "
                    "(appearing in reference connector '%s')" %
                    ( tmp.cname(), var, ref.cname() ) )
            for var in b - a:
                errors.append(
                    "Reference connector '%s' missing variable '%s' "
                    "(appearing in connector '%s')" %
                    ( ref.cname(), var, tmp.cname() ) )
        return errors, ref, skip

    def _expand_constraint(self, block, name, idx, constraint, ref, skip, cList, connectors):
        def _substitute_var(arg, var):
            if arg.is_expression():
                if arg.__class__ is _ProductExpression:
                    _substitute_vars(arg._numerator, var)
                    _substitute_vars(arg._denominator, var)
                else:
                    _substitute_vars(arg._args, var)
                return arg
            elif isinstance(arg, _ConnectorData):
                v = arg.vars[var]
                if v.is_expression():
                    v = v.clone()
                return _substitute_var(v, var) 
            elif isinstance(arg, VarList):
                return arg.add()
            return arg

        def _substitute_vars(args, var):
            for idx, arg in enumerate(args):
                if arg.is_expression():
                    if arg.__class__ is _ProductExpression:
                        _substitute_vars(arg._numerator, var)
                        _substitute_vars(arg._denominator, var)
                    else:
                        _substitute_vars(arg._args, var)
                elif isinstance(arg, _ConnectorData):
                    v = arg.vars[var]
                    if v.is_expression():
                        v = v.clone()
                    args[idx] = _substitute_var(v, var) 
                elif isinstance(arg, VarList):
                    args[idx] = arg.add()

        for var in sorted(ref.vars.iterkeys()):
            if var in skip:
                continue
            #vMap = dict((id(c),var) for c in connectors)
            #if constraint.equality:
            #    cList.append( ( clone_expression(constraint.body, substitute=vMap),
            #                    constraint.upper ) )
            #else:
            #    cList.append( ( constraint.lower,
            #                    clone_expression(constraint.body, substitute=vMap),
            #                    constraint.upper ) )
            #return

            if constraint.body.is_expression():
                c = _substitute_var(constraint.body.clone(), var)
            else:
                c = _substitute_var(constraint.body, var)
            if constraint.equality:
                cList.append( ( c, constraint.upper ) )
            else:
                cList.append( ( constraint.lower, c, constraint.upper ) )
