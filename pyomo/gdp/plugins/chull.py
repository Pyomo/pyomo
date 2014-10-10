#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

from six import itervalues, iteritems

from pyomo.misc.plugin import alias
from pyomo.core import *
from pyomo.core.base import expr, Transformation
from pyomo.core.base.block import SortComponents
from pyomo.core.base import _ExpressionData
from pyomo.core.base.var import _VarData
from pyomo.gdp import *

import weakref
import logging
logger = logging.getLogger('pyomo.core')

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
            }
        self._promote_vars = []

    def apply(self, instance, **kwds):
        options = kwds.pop('options', {})

        targets = kwds.pop('targets', None)
        if targets is None:
            for block in instance.all_blocks(active=True, sort_by_keys=True):
                self._transformBlock(block)
        else:
            if isinstance(targets, Component):
                targets = (targets, )
            for _t in targets:
                if not _t.active:
                    continue
                if _t.parent_component() is _t:
                    _name = _t.cname()
                    for _idx, _obj in _t.iteritems():
                        if _obj.active:
                            self._transformDisjunction(_name, _idx, _obj)
                else:
                    self._transformDisjunction(
                        _t.parent_component().cname(), _t.index(), _t )

        # REQUIRED: re-call preprocess()
        instance.preprocess()
        return instance

    def _transformBlock(self, block):
        # For every (active) disjunction in the block, convert it to a
        # simple constraint and then relax the individual (active)
        # disjuncts
        #
        # Note: we need to make a copy of the list because singletons
        # are going to be reclassified, which could foul up the
        # iteration
        for name, idx, obj in list(block.active_component_data(Disjunction,sort=SortComponents.sorter(sort_by_keys=True))):
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
        #disjunctions = block.components(Disjunction)
        #for Disj in disjunctions.itervalues():
        #    for disjuncts in obj.parent_component()._disjuncts[idx]:
        Disj = obj
        block = obj.parent_block()
        if True:  # hack for indentation
                disjuncts = [ d for d in obj.parent_component()._disjuncts[idx] if d.active ]
                localVars = {}
                cName = _generate_name(idx)
                cName = Disj.parent_component().cname() + (".%s"%(cName,) if cName else "")
                for d in disjuncts:
                    for eid, e in iteritems(disaggregatedVars.get(id(d), ['',{}])[1]):
                        localVars.setdefault(eid, (e[0],[]))[1].append(e[2])
                for d in disjuncts:
                    for eid, v in iteritems(localVars):
                        if eid not in disaggregatedVars.get(id(d), ['',{}])[1]:
                            tmp = Var(domain=v[0].domain)
                            tmp.setlb(min(0,value(v[0].lb)))
                            tmp.setub(max(0,value(v[0].ub)))
                            disaggregatedVars[id(d)][1][eid] = (v[0], d.indicator_var, tmp)
                            v[1].append(tmp)
                for v in sorted(localVars.values(), key=lambda x: x[0].cname(True)):
                    newC = Constraint( expr = v[0] == sum(v[1]) )
                    block.add_component( "%s.%s" % (cName, v[0].cname(True)), newC )
                    newC.construct()

        # Promote the local disaggregated variables and add BigM
        # constraints to force them to 0 when not active.
        for d_data in sorted(disaggregatedVars.values(), key=lambda x: x[0]):
            for e in sorted(d_data[1].values(), key=lambda x: x[0].cname()):
                v_name = "%s%s" % (d_data[0],e[0].cname())
                # add the disaggregated variable
                block.add_component( v_name, e[2] )
                e[2].construct()
                # add Big-M constraints on disaggregated variable to
                # force to 0 if not active
                if e[2].lb is not None and value(e[2].lb) != 0:
                    newC = Constraint(expr=value(e[2].lb) * e[1] <= e[2])
                    block.add_component( v_name+"_lo", newC )
                    newC.construct()
                if e[2].ub is not None and value(e[2].ub) != 0:
                    newC = Constraint(expr=e[2] <= value(e[2].ub) * e[1])
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
        #        if cData._equality:
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
            obj.parent_block().del_component(obj.name)
            _tmp.add_component(name, obj)
            _tmp.reclassify_component_type(obj, Constraint)
        else:
            # Look for a constraint in our transformation workspace
            # where we can "move" this disjunction so that the writers
            # will see it.
            _constr = _tmp.component(name)
            if _constr is None:
                _constr = Constraint(
                    obj.parent_component().index_set(), noruleinit=True )
                _tmp.add_component(name, _constr)
            # Move this disjunction over to the Constraint
            _constr._data[idx] = obj.parent_component()._data.pop(idx)
            _constr._data[idx]._component = weakref.ref(_constr)
        
        # Promote the indicator variables up into the model
        for var, block, name in self._promote_vars:
            var.parent_block().del_component(var.cname())
            block.add_component(name, var)
            

    def _transform_disjunct(self, disjunct, block, disaggregatedVars):
        if not disjunct.active:
            disjunct.indicator_var.fix(0)
            return
        if disjunct.parent_block().cname().startswith('_gdp_relax'):
            # Do not transform a block more than once
            return

        # Calculate a unique name by concatenating all parent block names
        fullName = disjunct.cname(True)

        varMap = disaggregatedVars.setdefault(id(disjunct), [fullName,{}])[1]

        # Transform each component within this disjunct
        for name, obj in disjunct.components().iteritems():
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
                         var.cname(), name)
        # This is a bit of a hack until we can re-think the chull
        # transformation in the context of Pyomo fully-supporting nested
        # block models
        #var.parent_block().del_component(var.cname())
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

            exp = self._var_subst(NL, c.body, disjunct.indicator_var, varMap)
            if NL:
                exp = exp * disjunct.indicator_var
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
                                       <= exp - constant )
                else:
                    newC = Constraint( expr = bound <= exp - constant )
                block.add_component( name+"_lo", newC )
                newC.construct()
            if c.upper is not None:
                if __debug__ and logger.isEnabledFor(logging.DEBUG):
                    logger.debug("GDP(cHull): Promoting constraint " +
                                 "'%s' as '%s_hi'", name, name)
                bound = c.upper() - constant
                if bound != 0:
                    newC = Constraint( expr = exp - constant <= \
                                       bound*disjunct.indicator_var )
                else:
                    newC = Constraint( expr = exp - constant <= bound )
                block.add_component( name+"_hi", newC )
                newC.construct()

    def _var_subst(self, NL, exp, y, varMap):
        # Recursively traverse the S-expression and substitute all model
        # variables with disaggregated local disjunct variables (logic
        # stolen from collect_cannonical_repn())

        #
        # Expression
        #
        if isinstance(exp,expr._ExpressionBase):
            if isinstance(exp,expr._ProductExpression):
                exp._numerator = [self._var_subst(NL, e, y, varMap) for e in exp._numerator]
                exp._denominator = [self._var_subst(NL, e, y, varMap) for e in exp._denominator]
            elif isinstance(exp, _ExpressionData) or \
                     isinstance(exp,expr._SumExpression) or \
                     isinstance(exp,expr._AbsExpression) or \
                     isinstance(exp,expr._PowExpression):
                exp._args = [self._var_subst(NL, e, y, varMap) for e in exp._args]
            else:
                raise ValueError("Unsupported expression type: "+str(exp))
        #
        # Constant
        #
        elif exp.is_fixed():
            pass
        #
        # Variable
        #
        elif isinstance(exp, _VarData):
            # Do not transform fixed variables
            if exp.fixed:
                return exp
            # Check if this disjunct has used this variable before...
            if id(exp) not in varMap:
                # create a new variable
                if exp.lb is None or exp.ub is None:
                    raise GDP_Error(
                        "Disjunct constraint referenced unbounded model "
                        "variable.\nAll variables must be bounded to use "
                        "the Convex Hull transformation.\n\t"
                        "Variable: %s" % (exp.cname(True),) )
                v = Var(domain=exp.domain)
                v.setlb(min(0,value(exp.lb)))
                v.setub(max(0,value(exp.ub)))
                varMap[id(exp)] = (exp, y, v)
            if NL:
                return varMap[id(exp)][2] / y
            else:
                return varMap[id(exp)][2]
        elif exp.type() is Var:
            raise GDP_Error("Unexpected Var encoundered in expression")
        #
        # ERROR
        #
        else:
            raise ValueError("Unexpected expression type: "+str(exp))

        return exp

