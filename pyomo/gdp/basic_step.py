#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import copy
import itertools

from pyomo.core import Block, ConstraintList, Set, Constraint
from pyomo.core.base import Reference
from pyomo.common.modeling import unique_component_name
from pyomo.gdp.disjunct import Disjunct, Disjunction

import logging

logger = logging.getLogger('pyomo.gdp')


def _clone_all_but_indicator_vars(self):
    """Clone everything in a Disjunct except for the indicator_vars"""
    return self.clone(
        {
            id(self.indicator_var): self.indicator_var,
            id(self.binary_indicator_var): self.binary_indicator_var,
        }
    )


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
    disjunctions = list(
        obj for obj in disjunctions_or_constraints if obj.ctype is Disjunction
    )
    constraints = list(
        obj for obj in disjunctions_or_constraints if obj.ctype is Constraint
    )
    if len(disjunctions) + len(constraints) != len(disjunctions_or_constraints):
        raise ValueError(
            'apply_basic_step only accepts a list containing '
            'Disjunctions or Constraints'
        )
    if not disjunctions:
        raise ValueError(
            'apply_basic_step: argument list must contain at least one Disjunction'
        )
    for d in disjunctions:
        if not d.xor:
            raise ValueError(
                "Basic steps can only be applied to XOR'd disjunctions\n\t"
                "(raised by disjunction %s)" % (d.name,)
            )
        if not d.active:
            logger.warning(
                "Warning: applying basic step to a previously "
                "deactivated disjunction (%s)" % (d.name,)
            )

    ans = Block(concrete=True)
    ans.DISJUNCTIONS = Set(initialize=range(len(disjunctions)))
    ans.INDEX = Set(
        dimen=len(disjunctions),
        initialize=_squish_singletons(
            itertools.product(*tuple(range(len(d.disjuncts)) for d in disjunctions))
        ),
    )

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
            src_disj = disjunctions[i].disjuncts[
                idx[i] if isinstance(idx, tuple) else idx
            ]
            tmp = _clone_all_but_indicator_vars(src_disj)
            for k, v in list(tmp.component_map().items()):
                if v.parent_block() is not tmp:
                    # Skip indicator_var and binary_indicator_var
                    continue
                tmp.del_component(k)
                ans.disjuncts[idx].src[i].add_component(k, v)
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
    ans.indicator_links = ConstraintList()
    for i in ans.DISJUNCTIONS:
        for j in range(len(disjunctions[i].disjuncts)):
            orig_var = disjunctions[i].disjuncts[j].indicator_var
            orig_binary_var = orig_var.get_associated_binary()
            ans.indicator_links.add(
                orig_binary_var
                == sum(
                    ans.disjuncts[idx].binary_indicator_var
                    for idx in ans.INDEX
                    if (idx[i] if isinstance(idx, tuple) else idx) == j
                )
            )
            # and throw on a Reference to original on the block
            for v in (orig_var, orig_binary_var):
                name_base = v.getname(fully_qualified=True)
                ans.add_component(unique_component_name(ans, name_base), Reference(v))

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
        d.x = Var(range(i))
        d.silly = Constraint(expr=d.indicator_var == i)

    m.d = Disjunct([1, 2], rule=_d)

    def _e(e, i):
        e.y = Var(range(2, i))

    m.e = Disjunct([3, 4, 5], rule=_e)

    m.dd = Disjunction(expr=[m.d[1], m.d[2]])
    m.ee = Disjunction(expr=[m.e[3], m.e[4], m.e[5]])
    m.Z = apply_basic_step([m.dd, m.ee])

    m.pprint()
