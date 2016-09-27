#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import weakref
import logging

from pyomo.util.plugin import alias
from pyomo.core import *
from pyomo.core.base import expr as EXPR, Transformation
from pyomo.core.base.block import SortComponents
from pyomo.core.base import _ExpressionData
from pyomo.core.base.var import _VarData
from pyomo.repn import generate_canonical_repn, LinearCanonicalRepn
from pyomo.gdp import *

from six import iteritems, iterkeys

logger = logging.getLogger('pyomo.core')

# NL_Mode_GrossmannLee is the original NL convex hull from Grossmann &
# Lee (2003), which substitutes nonlinear constraints
#     h_ik(x) <= 0
# with
#     x_k = sum( nu_ik )
#     y_ik * h_ik( nu_ik/y_ik ) <= 0
#
NL_Mode_GrossmannLee = 1
#
# NL_Mode_GrossmannLee is the original NL convex hull from Grossmann &
# Lee (2003), which substitutes nonlinear constraints
#     h_ik(x) <= 0
# with
#     x_k = sum( nu_ik )
#     (y_ik + eps) * h_ik( nu_ik/(y_ik + eps) ) <= 0
#
NL_Mode_LeeGrossmann = 2
#
# NL_Mode_Sawaya is an improved relaxation that avoids numerical issues
# from the Lee & Grossmann formulation by using:
#     x_k = sum( nu_ik )
#     ((1-eps)*y_ik + eps) * h_ik( nu_ik/((1-eps)*y_ik + eps) ) \
#        - eps * r_ki(0) * ( 1-y_ik )
#
NL_Mode_Sawaya = 3

EPS = 1e-2

class ConvexHull_Transformation(Transformation):

    alias('gdp.chull', doc="Relaxes a disjunctive model into an algebraic model by forming the convex hull relaxation of each disjunction.")

    def __init__(self):
        super(ConvexHull_Transformation, self).__init__()
        self.handlers = {
            Constraint : self._xform_constraint,
            Var : self._xform_var,
            Connector : self._xform_skip,
            Param : self._xform_skip,
            Set : self._xform_skip,
            Suffix : self._xform_skip,
            }
        self._promote_vars = []
        #self._mode = NL_Mode_GrossmannLee
        #self._mode = NL_Mode_LeeGrossmann
        self._mode = NL_Mode_Sawaya

    def _apply_to(self, instance, **kwds):
        options = kwds.pop('options', {})

        targets = kwds.pop('targets', None)

        if kwds:
            logger.warning("GDP(CHull): unrecognized keyword arguments:\n\t%s"
                           % ( '\n\t'.join(iterkeys(kwds)), ))

        if targets is None:
            for block in instance.block_data_objects(
                    active=True, sort=SortComponents.deterministic ):
                self._transformBlock(block)
        else:
            if isinstance(targets, Component):
                targets = (targets, )
            for _t in targets:
                if not _t.active:
                    continue
                if _t.parent_component() is _t:
                    _name = _t.local_name
                    for _idx, _obj in _t.iteritems():
                        if _obj.active:
                            self._transformDisjunction(_name, _idx, _obj)
                else:
                    self._transformDisjunction(
                        _t.parent_component().local_name, _t.index(), _t)

    def _transformBlock(self, block):
        # For every (active) disjunction in the block, convert it to a
        # simple constraint and then relax the individual (active)
        # disjuncts
        #
        # Note: we need to make a copy of the list because singletons
        # are going to be reclassified, which could foul up the
        # iteration
        for (name, idx), obj in block.component_data_iterindex(Disjunction,
                                                               active=True,
                                                               sort=SortComponents.deterministic):
            self._transformDisjunction(name, idx, obj)

    def _transformDisjunction(self, name, idx, obj):
        disaggregatedVars = {}

        for disjunct in obj.parent_component()._disjuncts[idx]:
            self._transform_disjunct(disjunct, obj.parent_block(), disaggregatedVars)

        def _generate_name(idx):
            if type(idx) in (tuple, list):
                if len(idx) == 0:
                    return ''
                else:
                    return '['+','.join([_generate_name(x) for x in idx])+']'
            else:
                return str(idx)

        # Correlate the disaggregated variables across the disjunctions
        #disjunctions = block.component_map(Disjunction)
        #for Disj in disjunctions.itervalues():
        #    for disjuncts in obj.parent_component()._disjuncts[idx]:
        Disj = obj
        block = obj.parent_block()
        if True:  # hack for indentation
                disjuncts = [ d for d in obj.parent_component()._disjuncts[idx] if d.active ]
                localVars = {}
                cName = _generate_name(idx)
                cName = Disj.parent_component().local_name + (".%s"%(cName,) if cName else "")
                for d in disjuncts:
                    for eid, e in iteritems(disaggregatedVars.get(id(d), ['',{}])[1]):
                        localVars.setdefault(eid, (e[0],[]))[1].append(e[2])
                for d in disjuncts:
                    for eid, v in iteritems(localVars):
                        if eid not in disaggregatedVars.get(id(d), ['',{}])[1]:
                            tmp = Var(domain=v[0].domain,
                                      bounds=(min(0,value(v[0].lb)),
                                              max(0,value(v[0].ub))))
                            disaggregatedVars[id(d)][1][eid] = (v[0], d.indicator_var, tmp)
                            v[1].append(tmp)
                for v in sorted(localVars.values(), key=lambda x: x[0].name):
                    newC = Constraint( expr = v[0] == sum(v[1]) )
                    block.add_component( "%s.%s" % (cName, v[0].name), newC )
                    newC.construct()

        # Promote the local disaggregated variables and add BigM
        # constraints to force them to 0 when not active.
        for d_data in sorted(disaggregatedVars.values(), key=lambda x: x[0]):
            for e in sorted(d_data[1].values(), key=lambda x: x[0].local_name):
                v_name = "%s%s" % (d_data[0],e[0].local_name)
                # add the disaggregated variable
                block.add_component( v_name, e[2] )
                e[2].construct()
                # add Big-M constraints on disaggregated variable to
                # force to 0 if not active
                if e[0].lb is not None and value(e[0].lb) != 0:
                    newC = Constraint(expr=value(e[0].lb) * e[1] <= e[2])
                    block.add_component( v_name+"_lo", newC )
                    newC.construct()
                if e[0].ub is not None and value(e[0].ub) != 0:
                    newC = Constraint(expr=e[2] <= value(e[0].ub) * e[1])
                    block.add_component( v_name+"_hi", newC )
                    newC.construct()

        # Recreate each Disjunction as a simple constraint
        #
        # Note: we do this at the end because the "disjunctions" opject
        # is a lightweight reference to the underlying component data:
        # replacing Disjunctions with Constraints results in this
        # PseudoMap being *empty* after this block!
        #for name, obj in disjunctions.iteritems():
        #    def _cGenerator(block, *idx):
        #        if idx == None:
        #            cData = obj._data[None]
        #        else:
        #            cData = obj._data[idx]
        #        if cData.equality:
        #            return (cData.body, cData.upper)
        #        else:
        #            return (cData.lower, cData.body, cData.upper)
        #    newC = Constraint(obj._index, rule=_cGenerator)
        #    block.del_component(name)
        #    block.add_component(name, newC)
        #    newC.construct()
        _tmp = obj.parent_block().component('_gdp_relax_chull')
        if _tmp is None:
            _tmp = Block()
            obj.parent_block().add_component('_gdp_relax_chull', _tmp)

        if obj.parent_component().dim() == 0:
            # Since there can't be more than one Disjunction in a
            # SimpleDisjunction, then we can just reclassify the entire
            # component in place
            obj.parent_block().del_component(obj.local_name)
            _tmp.add_component(name, obj)
            _tmp.reclassify_component_type(obj, Constraint)
        else:
            # Look for a constraint in our transformation workspace
            # where we can "move" this disjunction so that the writers
            # will see it.
            _constr = _tmp.component(name)
            if _constr is None:
                _constr = Constraint(
                    obj.parent_component().index_set())
                _tmp.add_component(name, _constr)
            # Move this disjunction over to the Constraint
            _constr._data[idx] = obj.parent_component()._data.pop(idx)
            _constr._data[idx]._component = weakref.ref(_constr)

        # Promote the indicator variables up into the model
        for var, block, name in self._promote_vars:
            var.parent_block().del_component(var.local_name)
            block.add_component(name, var)

    def _transform_disjunct(self, disjunct, block, disaggregatedVars):
        if not disjunct.active:
            disjunct.indicator_var.fix(0)
            return
        if disjunct.parent_block().local_name.startswith('_gdp_relax'):
            # Do not transform a block more than once
            return

        # Calculate a unique name by concatenating all parent block names
        fullName = disjunct.name

        varMap = disaggregatedVars.setdefault(id(disjunct), [fullName,{}])[1]

        # Transform each component within this disjunct
        for name, obj in disjunct.component_map().iteritems():
            handler = self.handlers.get(obj.type(), None)
            if handler is None:
                raise GDP_Error(
                    "No cHull transformation handler registered "
                    "for modeling components of type %s" % obj.type() )
            handler(fullName+name, obj, varMap, disjunct, block)

    def _xform_skip(self, _name, var, varMap, disjunct, block):
        pass

    def _xform_var(self, name, var, varMap, disjunct, block):
        # "Promote" the local variables up to the main model
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("GDP(cHull): Promoting local variable '%s' as '%s'",
                         var.local_name, name)
        # This is a bit of a hack until we can re-think the chull
        # transformation in the context of Pyomo fully-supporting nested
        # block models
        #var.parent_block().del_component(var.local_name)
        #block.add_component(name, var)
        var.construct()
        self._promote_vars.append((var,block,name))

    def _xform_constraint(self, _name, constraint, varMap, disjunct, block):
        lin_body_map = getattr(block,"lin_body",None)
        for cname, c in iteritems(constraint._data):
            name = _name + ('.%s' % (cname,) if cname else '')

            if (not lin_body_map is None) and (not lin_body_map.get(c) is None):
                raise GDP_Error('GDP(cHull) cannot process linear ' \
                      'constraint bodies (yet) (found at ' + name + ').')

            constant = 0
            try:
                cannonical = generate_canonical_repn(c.body)
                if isinstance(cannonical, LinearCanonicalRepn):
                    NL = False
                else:
                    NL = canonical_is_nonlinear(cannonical)
            except:
                NL = True

            # We need to evaluate teh expression at the origin *before*
            # we substitute the expression variables with the
            # disaggregated variables
            if NL and self._mode == NL_Mode_Sawaya:
                h_0 = value( self._eval_at_origin(
                    NL, c.body.clone(), disjunct.indicator_var, varMap ) )

            expr = self._var_subst(NL, c.body, disjunct.indicator_var, varMap)
            if NL:
                y = disjunct.indicator_var
                if self._mode == NL_Mode_GrossmannLee:
                    expr = expr * y
                elif self._mode == NL_Mode_LeeGrossmann:
                    expr = (y + EPS) * expr
                elif self._mode == NL_Mode_Sawaya:
                    expr = ((1-EPS)*y + EPS)*expr - EPS*h_0*(1-y)
                else:
                    raise RuntimeError("Unknown NL CHull mode")
            else:
                # We need to make sure to pull out the constant terms
                # from the expression and put them into the lb/ub
                if cannonical.constant == None:
                    constant = 0
                else:
                    constant = cannonical.constant

            if c.lower is not None:
                if __debug__ and logger.isEnabledFor(logging.DEBUG):
                    logger.debug("GDP(cHull): Promoting constraint " +
                                 "'%s' as '%s_lo'", name, name)
                bound = c.lower() - constant
                if bound != 0:
                    newC = Constraint( expr = bound*disjunct.indicator_var \
                                       <= expr - constant )
                else:
                    newC = Constraint( expr = bound <= expr - constant )
                block.add_component( name+"_lo", newC )
                newC.construct()
            if c.upper is not None:
                if __debug__ and logger.isEnabledFor(logging.DEBUG):
                    logger.debug("GDP(cHull): Promoting constraint " +
                                 "'%s' as '%s_hi'", name, name)
                bound = c.upper() - constant
                if bound != 0:
                    newC = Constraint( expr = expr - constant <= \
                                       bound*disjunct.indicator_var )
                else:
                    newC = Constraint( expr = expr - constant <= bound )
                block.add_component( name+"_hi", newC )
                newC.construct()

    def _var_subst(self, NL, expr, y, varMap):
        # Recursively traverse the S-expression and substitute all model
        # variables with disaggregated local disjunct variables (logic
        # stolen from collect_cannonical_repn())

        #
        # Expression
        #
        if isinstance(expr,EXPR._ExpressionBase):
            if isinstance(expr,EXPR._ProductExpression):
                expr._numerator = [self._var_subst(NL, e, y, varMap) for e in expr._numerator]
                expr._denominator = [self._var_subst(NL, e, y, varMap) for e in expr._denominator]
            elif isinstance(expr, _ExpressionData) or \
                     isinstance(expr,EXPR._SumExpression) or \
                     isinstance(expr,EXPR._AbsExpression) or \
                     isinstance(expr,EXPR._IntrinsicFunctionExpression) or \
                     isinstance(expr,EXPR._PowExpression):
                expr._args = [self._var_subst(NL, e, y, varMap) for e in expr._args]
            else:
                raise ValueError("Unsupported expression type: "+str(expr))
        #
        # Constant
        #
        elif expr.is_fixed():
            pass
        #
        # Variable
        #
        elif isinstance(expr, _VarData):
            # Do not transform fixed variables
            if expr.fixed:
                return expr
            # Check if this disjunct has used this variable before...
            if id(expr) not in varMap:
                # create a new variable
                if expr.lb is None or expr.ub is None:
                    raise GDP_Error(
                        "Disjunct constraint referenced unbounded model "
                        "variable.\nAll variables must be bounded to use "
                        "the Convex Hull transformation.\n\t"
                        "Variable: %s" % (expr.name,) )
                v = Var( domain=expr.domain,
                         bounds=(min(0,value(expr.lb)),
                                 max(0,value(expr.ub))))
                varMap[id(expr)] = (expr, y, v)
            if NL:
                if self._mode == NL_Mode_GrossmannLee:
                    return varMap[id(expr)][2] / y
                elif self._mode == NL_Mode_LeeGrossmann:
                    return varMap[id(expr)][2] / (y+EPS)
                elif self._mode == NL_Mode_Sawaya:
                    return varMap[id(expr)][2] / ( (1-EPS)*y + EPS )
                else:
                    raise RuntimeError("Unknown NL CHull mode")
            else:
                return varMap[id(expr)][2]
        elif expr.type() is Var:
            raise GDP_Error("Unexprected Var encoundered in expression")
        #
        # ERROR
        #
        else:
            raise ValueError("Unexpected expression type: "+str(expr))

        return expr


    def _eval_at_origin(self, NL, expr, y, varMap):
        # Recursively traverse the S-expression and substitute all free
        # model variables with 0.  This is a "poor-man's" approach to
        # evaluating the expression at the origin.
        #
        # TODO: we ahould probably make this more efficient by
        # traversing the expression, identifying the variables,
        # preserving their current value, setting them to 0, evaluate
        # the expression, and restore the variable values -- instead of
        # making a copy of the expression like we are doing here.

        #
        # Expression
        #
        if isinstance(expr,EXPR._ExpressionBase):
            if isinstance(expr,EXPR._ProductExpression):
                expr._numerator = [ self._eval_at_origin(NL, e, y, varMap)
                                   for e in expr._numerator ]
                expr._denominator = [ self._eval_at_origin(NL, e, y, varMap)
                                     for e in expr._denominator ]
            elif isinstance(expr, _ExpressionData) or \
                     isinstance(expr,EXPR._SumExpression) or \
                     isinstance(expr,EXPR._AbsExpression) or \
                     isinstance(expr,EXPR._IntrinsicFunctionExpression) or \
                     isinstance(expr,EXPR._PowExpression):
                expr._args = [ self._eval_at_origin(NL, e, y, varMap)
                              for e in expr._args ]
            else:
                raise ValueError("Unsupported expression type: "+str(expr))
        #
        # Constant
        #
        elif expr.is_fixed():
            pass
        #
        # Variable
        #
        elif isinstance(expr, _VarData):
            # Do not substitute fixed variables
            if not expr.fixed:
                return 0
        else:
            raise ValueError("Unexpected expression type: "+str(expr))

        return expr

