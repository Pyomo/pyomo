#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import copy
import itertools
from six import iteritems
from six.moves import xrange

from pyomo.core import Block, ConstraintList, Set, Constraint
from pyomo.core.base import Reference
from pyomo.common.modeling import unique_component_name
from pyomo.gdp.disjunct import Disjunct, Disjunction

import logging
logger = logging.getLogger('pyomo.gdp')


def _pseudo_clone(self):
    """Clone everything in a Disjunct except for the indicator_var"""
    memo = {
        '__block_scope__': {id(self): True, id(None): False},
        id(self.indicator_var): self.indicator_var,
    }
    new_block = copy.deepcopy(self, memo)
    new_block._parent = None
    return new_block


def _squish_singletons(tuple_iter):
    """Squish all singleton tuples into their non-tuple values."""
    for tup in tuple_iter:
        if len(tup) == 1:
            yield tup[0]
        else:
            yield tup


def apply_basic_step(disjunctions_or_constraints):
    #
    # Basic steps only apply to XOR'd disjunctions
    #
    disjunctions = list(obj for obj in disjunctions_or_constraints
                        if obj.ctype == Disjunction)
    constraints = list(obj for obj in disjunctions_or_constraints
                       if obj.ctype == Constraint)
    for d in disjunctions:
        if not d.xor:
            raise ValueError(
                "Basic steps can only be applied to XOR'd disjunctions\n\t"
                "(raised by disjunction %s)" % (d.name,))
        if not d.active:
            logger.warning("Warning: applying basic step to a previously "
                           "deactivated disjunction (%s)" % (d.name,))

    ans = Block(concrete=True)
    ans.DISJUNCTIONS = Set(initialize=xrange(len(disjunctions)))
    ans.INDEX = Set(
        dimen=len(disjunctions),
        initialize=_squish_singletons(itertools.product(
            *tuple( xrange(len(d.disjuncts)) for d in disjunctions ))))

    #
    # Form the individual disjuncts for the new basic step
    #
    ans.disjuncts = Disjunct(ans.INDEX)
    for idx in ans.INDEX:
        #
        # Each source disjunct will be copied (cloned) into its own
        # subblock
        #
        ans.disjuncts[idx].src = Block(ans.DISJUNCTIONS)
        for i in ans.DISJUNCTIONS:
            tmp = _pseudo_clone(disjunctions[i].disjuncts[
                idx[i] if isinstance(idx, tuple) else idx])
            for k,v in list(iteritems( tmp.component_map() )):
                if k == 'indicator_var':
                    continue
                tmp.del_component(k)
                ans.disjuncts[idx].src[i].add_component(k,v)
        # Copy in the constraints corresponding to the improper disjunctions
        ans.disjuncts[idx].improper_constraints = ConstraintList()
        for constr in constraints:
            if constr.is_indexed():
                for indx in constr:
                    ans.disjuncts[idx].improper_constraints.add(
                        (constr[indx].lower, constr[indx].body, constr[indx].upper)
                    )
                    constr[indx].deactivate()
            # need this so that we can take an improper basic step with a
            # ConstraintData
            else:
                ans.disjuncts[idx].improper_constraints.add(
                    (constr.lower, constr.body, constr.upper)
                )
                constr.deactivate()

    #
    # Link the new disjunct indicator_var's to the original
    # indicator_var's.  Since only one of the new
    #
    NAME_BUFFER = {}
    ans.indicator_links = ConstraintList()
    for i in ans.DISJUNCTIONS:
        for j in xrange(len(disjunctions[i].disjuncts)):
            orig_var = disjunctions[i].disjuncts[j].indicator_var
            ans.indicator_links.add(
                orig_var ==
                sum( ans.disjuncts[idx].indicator_var for idx in ans.INDEX
                     if (idx[i] if isinstance(idx, tuple) else idx) == j ))
            # and throw on a Reference to original on the block
            name_base = orig_var.getname(fully_qualified=True,
                                         name_buffer=NAME_BUFFER)
            ans.add_component(unique_component_name( ans, name_base),
                              Reference(orig_var))

    # Form the new disjunction
    ans.disjunction = Disjunction(expr=[ans.disjuncts[i] for i in ans.INDEX])

    #
    # Deactivate the old disjunctions / disjuncts
    #
    for i in ans.DISJUNCTIONS:
        disjunctions[i].deactivate()
        for d in disjunctions[i].disjuncts:
            d._deactivate_without_fixing_indicator()

    return ans


if __name__ == '__main__':
    from pyomo.environ import ConcreteModel, Constraint, Var
    m = ConcreteModel()
    def _d(d, i):
        d.x = Var(xrange(i))
        d.silly = Constraint(expr=d.indicator_var == i)
    m.d = Disjunct([1,2], rule=_d)
    def _e(e, i):
        e.y = Var(xrange(2,i))
    m.e = Disjunct([3,4,5], rule=_e)

    m.dd = Disjunction(expr=[m.d[1], m.d[2]])
    m.ee = Disjunction(expr=[m.e[3], m.e[4], m.e[5]])
    m.Z = apply_basic_step([m.dd, m.ee])

    m.pprint()
