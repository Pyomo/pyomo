#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from six import iteritems

from pyomo.util.plugin import alias
from pyomo.core import *
from pyomo.repn import *
from pyomo.core.base import Transformation
from pyomo.core.base.block import SortComponents
from pyomo.repn.canonical_repn import LinearCanonicalRepn
from pyomo.gdp import *

import weakref
import logging
logger = logging.getLogger('pyomo.core')


class BigM_Transformation(Transformation):

    alias('gdp.bigm', doc="Relaxes a disjunctive model into an algebraic model by adding Big-M terms to all disjunctive constraints.")

    def __init__(self):
        super(BigM_Transformation, self).__init__()
        self.handlers = {
            Constraint : self._xform_constraint,
            Var : self._xform_var,
            Connector : self._xform_skip,
            Suffix : self._xform_skip,
            Param : self._xform_skip,
            }

    def apply(self, instance, **kwds):
        options = kwds.pop('options', {})

        inplace = kwds.pop('inplace', None)
        if 'inplace' in options:
            if bool(options['inplace']) != inplace and inplace is not None:
                raise RuntimeError(
                    "conflicting inplace options: apply(inplace=%s) with "
                    "options['inplace']==%s" % (inplace, options['inplace']) )
            inplace = options['inplace']
        elif inplace is None:
            inplace = True

        if not inplace:
            instance = instance.clone()

        targets = kwds.pop('targets', None)
        if targets is None:
            for block in instance.all_blocks(
                    active=True, sort=SortComponents.deterministic ):
                self._transformBlock(block)
        else:
            if isinstance(targets, Component):
                targets = (targets, )
            for _t in target:
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
        for name, idx, obj in list(block.active_component_data_iter(Disjunction,sort=SortComponents.deterministic)):
            self._transformDisjunction(name, idx, obj)

    def _transformDisjunction(self, name, idx, obj):
        # For the time being, we need to relax the disjuncts *before* we
        # move the disjunction constraint over (otherwise we wouldn't be
        # able to get to the _disjuncts map from the other component).
        #
        # FIXME: the disjuncts list should just be on the _DisjunctData
        if obj.parent_block().cname().startswith('_gdp_relax'):
            # Do not transform a block more than once
            return

        for disjunct in obj.parent_component()._disjuncts[idx]:
            self._bigM_relax_disjunct(disjunct)

        _tmp = obj.parent_block().component('_gdp_relax_bigm')
        if _tmp is None:
            _tmp = Block()
            obj.parent_block().add_component('_gdp_relax_bigm', _tmp)

        if obj.parent_component().dim() == 0:
            # Since there can't be more than one Disjunction in a
            # SimpleDisjunction, then we can just reclassify the entire
            # component in place
            obj.parent_block().del_component(obj)
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


    def _bigM_relax_disjunct(self, disjunct):
        #
        if not disjunct.active:
            disjunct.indicator_var.fix(0)
            return
        if disjunct.parent_block().cname().startswith('_gdp_relax'):
            # Do not transform a block more than once
            return
        
        _tmp = disjunct.parent_block().component('_gdp_relax_bigm')
        if _tmp is None:
            _tmp = Block()
            disjunct.parent_block().add_component('_gdp_relax_bigm', _tmp)

        # Move this disjunct over to a Block component (so the writers
        # will pick it up)
        if disjunct.parent_component().dim() == 0:
            # Since there can't be more than one Disjunct in a
            # SimpleDisjunct, then we can just reclassify the entire
            # component into our scratch space
            disjunct.parent_block().del_component(disjunct)
            _tmp.add_component(disjunct.name, disjunct)
            _tmp.reclassify_component_type(disjunct, Block)
        else:
            _block = _tmp.component(disjunct.parent_component().cname())
            if _block is None:
                _block = Block(disjunct.parent_component().index_set())
                _tmp.add_component(disjunct.parent_component().cname(), _block)
            # Move this disjunction over to the Constraint
            idx = disjunct.index()
            _block._data[idx] = disjunct.parent_component()._data.pop(idx)
            _block._data[idx]._component = weakref.ref(_block)
       

        # Transform each component within this disjunct
        for name, obj in list(disjunct.components().iteritems()):
            handler = self.handlers.get(obj.type(), None)
            if handler is None:
                raise GDP_Error(
                    "No BigM transformation handler registered "
                    "for modeling components of type %s" % obj.type() )
            handler(name, obj, disjunct)


    def _xform_skip(self, _name, constraint, disjunct):
        pass

    def _xform_constraint(self, _name, constraint, disjunct):
        if 'BigM' in disjunct.components(Suffix):
            M = disjunct.component('BigM').get(constraint)
        else:
            M = disjunct.next_M()
        lin_body_map = getattr(disjunct.model(),"lin_body",None)
        for cname, c in iteritems(constraint._data):
            if not c.active:
                continue
            c.deactivate()

            name = _name + (cname and '.'+cname or '')

            if (not lin_body_map is None) and (not lin_body_map.get(c) is None):
                raise GDP_Error('GDP(BigM) cannot process linear ' \
                      'constraint bodies (yet) (found at ' + name + ').')

            if isinstance(M, list):
                if len(M):
                    m = M.pop(0)
                else:
                    m = (None,None)
            else:
                m = M
            if not isinstance(m, tuple):
                if m is None:
                    m = (None, None)
                else:
                    m = (-1*m,m)

            # If we need an M (either for upper and/or lower bounding of
            # the expression, then try and estimate it
            if ( c.lower is not None and m[0] is None ) or \
                   ( c.upper is not None and m[1] is None ):
                m = self._estimate_M(c.body, name, m)

            bounds = (c.lower, c.upper)
            for i in (0,1):
                if bounds[i] is None:
                    continue
                if m[i] is None:
                    raise GDP_Error("Cannot relax disjunctive " + \
                          "constraint %s because M is not defined." % name)
                n = name;
                if bounds[1-i] is None:
                    n += '_eq'
                else:
                    n += ('_lo','_hi')[i]

                if __debug__ and logger.isEnabledFor(logging.DEBUG):
                    logger.debug("GDP(BigM): Promoting local constraint "
                                 "'%s' as '%s'", constraint.name, n)
                M_expr = (m[i]-bounds[i])*(1-disjunct.indicator_var)
                if i == 0:
                    newC = Constraint(expr=c.lower <= c.body - M_expr)
                else:
                    newC = Constraint(expr=c.body - M_expr <= c.upper)
                disjunct.add_component(n, newC)
                newC.construct()


    def _xform_var(self, name, var, disjunct):
        pass


    def _estimate_M(self, expr, name, m):
        # Calculate a best guess at M
        repn = generate_canonical_repn(expr)
        M = [0,0]

        if not isinstance(repn, LinearCanonicalRepn):
            logger.error("GDP(BigM): cannot estimate M for nonlinear "
                         "expressions.\n\t(found while processing %s)",
                         name)
            return m

        if repn.constant != None:
            for i in (0,1):
                if M[i] is not None:
                    M[i] += repn.constant

        for i in xrange(0,len(repn.linear)):
            var = repn.variables[i]
            coef = repn.linear[i]
            bounds = (value(var.lb), value(var.ub))
            for i in (0,1):
                # reverse the bounds if the coefficient is negative
                if coef > 0:
                    j = i
                else:
                    j = 1-i

                try:
                    M[j] += value(bounds[i]) * coef
                except:
                    M[j] = None


        # Allow user-defined M values to override the estimates
        for i in (0,1):
            if m[i] is not None:
                M[i] = m[i]
        return tuple(M)

