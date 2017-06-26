# -*- coding: UTF-8 -*-
#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from six.moves import xrange as range
from six import iteritems, iterkeys

from pyomo.util.plugin import alias
from pyomo.core import (Constraint, Var, Connector, Suffix, Param, Set,
                        Block, value)
from pyomo.repn import generate_canonical_repn
from pyomo.core.base import Transformation
from pyomo.core.base.block import SortComponents
from pyomo.repn import LinearCanonicalRepn
from pyomo.gdp import Disjunction, GDP_Error

import weakref
import logging
logger = logging.getLogger('pyomo.core')


class BigM_Transformation(Transformation):
    """Relax disjunctive model using big-M terms.

    Relaxes a disjunctive model into an algebraic model by adding Big-M terms
    to all disjunctive constraints.
    """

    alias('gdp.bigm', doc=__doc__)

    def __init__(self):
        """Initialize transformation object."""
        super(BigM_Transformation, self).__init__()
        self.handlers = {
            Constraint: self._xform_constraint,
            Var: self._xform_skip,
            Connector: self._xform_skip,
            Suffix: self._xform_skip,
            Param: self._xform_skip,
            Set: self._xform_skip
        }

    def _apply_to(self, instance, **kwds):
        options = kwds.pop('options', {})

        bigM = options.get('default_bigM', None)
        bigM = kwds.pop('default_bigM', bigM)
        if bigM is not None:
            #
            # Test for the suffix - this test will (correctly) generate
            # a warning if the component is already declared, but is a
            # different ctype (e.g., a constraint or block)
            #
            if 'BigM' not in instance.component_map(Suffix):
                instance.BigM = Suffix(direction=Suffix.LOCAL)
            #
            # Note: this will implicitly change the model default BigM
            # value so that the argument overrides the option, which
            # overrides any default specified on the model.
            #
            instance.BigM[None] = bigM

        # targets = kwds.pop('targets', None)
        targets = None

        if kwds:
            logger.warning("GDP(BigM): unrecognized keyword arguments:\n%s"
                           % ('\n'.join(iterkeys(kwds)), ))

        self._tmp_disjunct_to_check = set()

        if targets is None:
            for block in instance.block_data_objects(
                    active=True, sort=SortComponents.deterministic):
                self._transformBlock(block)
        # else:
        #     if isinstance(targets, Component):
        #         targets = (targets, )
        #     for _t in targets:
        #         if not _t.active:
        #             continue
        #         if _t.parent_component() is _t:
        #             _name = _t.local_name
        #             for _idx, _obj in _t.iteritems():
        #                 if _obj.active:
        #                     self._transformDisjunction(_name, _idx, _obj)
        #         else:
        #             self._transformDisjunction(
        #                 _t.parent_component().local_name, _t.index(), _t)

        for indexed_disjunct in self._tmp_disjunct_to_check:
            if all(getattr(indexed_disjunct[indx],
                           '_GDP_disjunct_is_relaxed', False)
                   for indx in indexed_disjunct):

                blk = indexed_disjunct.parent_block()
                blk.reclassify_component_type(indexed_disjunct, Block)

    def _transformBlock(self, block):
        # For every (active) disjunction in the block, convert it to a
        # simple constraint and then relax the individual (active)
        # disjuncts
        #
        # Note: we need to make a copy of the list because singletons
        # are going to be reclassified, which could foul up the
        # iteration
        for (name, idx), obj in block.component_data_iterindex(
                ctype=Disjunction, active=True,
                sort=SortComponents.deterministic):
            self._transformDisjunction(name, idx, obj)

    def _transformDisjunction(self, name, idx, obj):
        # For the time being, we need to relax the disjuncts *before* we
        # move the disjunction constraint over (otherwise we wouldn't be
        # able to get to the _disjuncts map from the other component).
        #
        # FIXME: the disjuncts list should just be on the _DisjunctData
        if obj.parent_block().local_name.startswith('_gdp_relax'):
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
                    obj.parent_component().index_set())
                _tmp.add_component(name, _constr)
            # Move this disjunction over to the Constraint
            _constr._data[idx] = obj.parent_component()._data.pop(idx)
            _constr._data[idx]._component = weakref.ref(_constr)

    def _bigM_relax_disjunct(self, disjunct):
        #
        if not disjunct.active:
            disjunct.indicator_var.fix(0)
            return

        #: The IndexedDisjunct or SimpleDisjunct holding this disjunct
        disj_obj = disjunct.parent_component()
        #: Suffix determining if a child _DisjunctData is relaxed
        is_relaxed = getattr(disjunct, '_GDP_disjunct_is_relaxed', False)
        if is_relaxed:
            return  # Disjunct already relaxed.
        else:
            disjunct._GDP_disjunct_is_relaxed = True

        # Turn this disjunct into a Block component (so the writers will pick
        # it up)
        if disjunct.parent_component().dim() == 0:
            # Since there can't be more than one Disjunct in a
            # SimpleDisjunct, then we can just reclassify the entire
            # component
            disjunct.parent_block().reclassify_component_type(disjunct, Block)
        else:
            # At the end, we need to check to see if this IndexedDisjunct has
            # been fully relaxed, so add to a set.
            self._tmp_disjunct_to_check.add(disj_obj)

        # Transform each component within this disjunct
        for name, obj in list(disjunct.component_map().iteritems()):
            handler = self.handlers.get(obj.type(), None)
            if handler is None:
                raise GDP_Error(
                    "No BigM transformation handler registered "
                    "for modeling components of type %s" % obj.type())
            handler(name, obj, disjunct)

    def _xform_skip(self, _name, constraint, disjunct):
        pass

    def _xform_constraint(self, _name, constraint, disjunct):
        if 'BigM' in disjunct.component_map(Suffix):
            M = disjunct.component('BigM').get(constraint)
        else:
            M = disjunct.next_M()
        # Qi: no idea what lin_body is, and no documentation so ¯\_(ツ)_/¯
        lin_body_map = getattr(disjunct.model(), "lin_body", None)

        for cname, c in iteritems(constraint._data):
            if not c.active:
                continue
            c.deactivate()

            name = _name + ('.' + str(cname) if cname is not None else '')

            if (lin_body_map is not None and
                    lin_body_map.get(c) is not None):
                raise GDP_Error('GDP(BigM) cannot process linear '
                                'constraint bodies (yet) (found at {}).'
                                .format(name))

            if isinstance(M, list):
                if len(M):
                    m = M.pop(0)
                else:
                    m = (None, None)
            else:
                m = M
            if not isinstance(m, tuple):
                if m is None:
                    m = (None, None)
                else:
                    m = (-1 * m, m)

            # If we need an M (either for upper and/or lower bounding of
            # the expression, then try and estimate it
            if (c.lower is not None and m[0] is None) or \
                    (c.upper is not None and m[1] is None):
                m = self._estimate_M(c.body, name, m, disjunct)

            bounds = (c.lower, c.upper)
            for i in (0, 1):
                if bounds[i] is None:
                    continue
                if m[i] is None:
                    raise GDP_Error("Cannot relax disjunctive "
                                    "constraint {} because M is not defined."
                                    .format(name))
                n = name
                if bounds[1 - i] is None:
                    n += '_eq'
                else:
                    n += ('_lo', '_hi')[i]

                if __debug__ and logger.isEnabledFor(logging.DEBUG):
                    logger.debug("GDP(BigM): Promoting local constraint "
                                 "'%s' as '%s'", constraint.local_name, n)
                M_expr = (m[i] - bounds[i]) * (1 - disjunct.indicator_var)
                if i == 0:
                    newC = Constraint(expr=c.lower <= c.body - M_expr)
                else:
                    newC = Constraint(expr=c.body - M_expr <= c.upper)
                disjunct.add_component(n, newC)
                newC.construct()

    def _estimate_M(self, expr, name, m, disjunct):
        # Calculate a best guess at M
        repn = generate_canonical_repn(expr)
        M = [0, 0]

        if isinstance(repn, LinearCanonicalRepn):
            if repn.constant is not None:
                for i in (0, 1):
                    if M[i] is not None:
                        M[i] += repn.constant

            for i, coef in enumerate(repn.linear or []):
                var = repn.variables[i]
                coef = repn.linear[i]
                bounds = (value(var.lb), value(var.ub))
                for i in (0, 1):
                    # reverse the bounds if the coefficient is negative
                    if coef > 0:
                        j = i
                    else:
                        j = 1 - i

                    try:
                        M[j] += value(bounds[i]) * coef
                    except:
                        M[j] = None
        else:
            logger.info("GDP(BigM): cannot estimate M for nonlinear "
                        "expressions.\n\t(found while processing %s)",
                        name)
            M = [None, None]

        # Allow user-defined M values to override the estimates
        for i in (0, 1):
            if m[i] is not None:
                M[i] = m[i]

        # Search for global BigM values: if there are still undefined
        # M's, then search up the block hierarchy for the first block
        # that contains a BigM Suffix with a non-None value for the
        # "None" component.
        if None in M:
            m = None
            while m is None and disjunct is not None:
                if 'BigM' in disjunct.component_map(Suffix):
                    m = disjunct.component('BigM').get(None)
                disjunct = disjunct.parent_block()
            if m is not None:
                try:
                    # We always allow M values to be specified as pairs
                    # (for lower / upper bounding)
                    M = [m[i] if x is None else x for i, x in enumerate(M)]
                except:
                    # We assume the default M is positive (so we need to
                    # invert it for the lower-bound M)
                    M = [(2 * i - 1) * m if x is None else x
                         for i, x in enumerate(M)]

        return tuple(M)
