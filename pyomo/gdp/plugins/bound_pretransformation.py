#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
    Any,
    Block,
    Constraint,
    NonNegativeIntegers,
    SortComponents,
    value,
    Var,
)
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.expr import identify_variables
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import is_child_of, get_gdp_tree
from pyomo.repn.standard_repn import generate_standard_repn
import logging

logger = logging.getLogger(__name__)


@TransformationFactory.register(
    'gdp.bound_pretransformation',
    doc="Partially transforms a GDP to a MIP by finding all disjunctive "
    "constraints with common left-hand sides and transforming them according "
    "to the formulation in Balas 1988",
)
class BoundPretransformation(Transformation):
    """
    Implements a special case of the transformation mentioned in [1] for
    handling disjunctive constraints with common left-hand sides (i.e.,
    Constraint bodies). Automatically detects univariate disjunctive
    Constraints (bounds or equalities involving one variable), and
    transforms them according to [1]. The transformed Constraints are
    deactivated, but the remainder of the GDP is untouched. That is,
    to completely transform the GDP, a GDP-to-MIP transformation is
    needed that will transform the remaining disjunctive constraints as
    well as any LogicalConstraints and the logic of the disjunctions
    themselves.

    NOTE: Because this transformation allows tighter bound values higher in
    the GDP hierarchy to supersede looser ones that are lower, the transformed
    model will not necessarily still be valid in the case that there are
    mutable Params in disjunctive variable bounds or in the transformed
    Constraints and the values of those mutable Params are later changed.
    Similarly, if this transformation is called when Vars are fixed, it will
    only be guaranteed to be valid when those Vars remain fixed to the same
    values.

    [1] Egon Balas, "On the convex hull of the union of certain polyhedra,"
        Operations Research Letters, vol. 7, 1988, pp. 279-283
    """

    CONFIG = ConfigDict('gdp.bound_pretransformation')
    CONFIG.declare(
        'targets',
        ConfigValue(
            default=None,
            domain=target_list,
            description="target or list of targets to transform",
            doc="""
            This specifies the list of Disjunctions or Blocks to be (partially)
            transformed. If None (default), the entire model is transformed. 
            Note that if the transformation is done out of place, the list of
            targets should be attached to the model before it is cloned, and
            the list will specify the targets on the cloned instance.
            """,
        ),
    )
    transformation_name = 'bound_pretransformation'

    def __init__(self):
        super().__init__()
        self.logger = logger

    def _apply_to(self, instance, **kwds):
        if not instance.ctype in (Block, Disjunct):
            raise GDP_Error(
                "Transformation called on %s of type %s. 'instance'"
                " must be a ConcreteModel, Block, or Disjunct (in "
                "the case of nested disjunctions)." % (instance.name, instance.ctype)
            )

        self._config = self.CONFIG(kwds.pop('options', {}))
        self._config.set_value(kwds)

        targets = self._config.targets
        if targets is None:
            targets = (instance,)

        transformation_blocks = {}
        bound_dict = ComponentMap()
        self._update_bounds_from_constraints(instance, bound_dict, None, is_root=True)
        # [ESJ 05/04/23]: In the future, I should think about getting my little
        # trees from this tree, or asking for leaves rooted somewhere specific
        # or something. Because this transformation currently does the work of
        # getting the GDP tree twice...
        whole_tree = get_gdp_tree(targets, instance)
        for t in whole_tree.topological_sort():
            if t.ctype is Disjunction and whole_tree.in_degree(t) == 0:
                self._transform_disjunction(
                    t, instance, bound_dict, transformation_blocks
                )

    def _transform_disjunction(
        self, disjunction, instance, bound_dict, transformation_blocks
    ):
        # We go from root to leaves so that whenever we hit a variable, we can
        # ask if the bounds we're seeing on its parent Disjunct (or if we're at
        # the root, in the global scope) are looser or tighter than the bounds
        # on it, and we pass the tightest ones down. For sane models, the bounds
        # will tighten as we go down the tree, but that's of course not
        # guaranteed since not all models are sane...
        disjunctions_to_transform = set()
        gdp_forest = get_gdp_tree((disjunction,), instance)
        for d in gdp_forest.topological_sort():
            if d.ctype is Disjunct:
                self._update_bounds_from_constraints(d, bound_dict, gdp_forest)
        self._create_transformation_constraints(
            disjunction, bound_dict, gdp_forest, transformation_blocks
        )

    def _get_bound_dict_for_var(self, bound_dict, v):
        v_bounds = bound_dict.get(v)
        if v_bounds is None:
            v_bounds = bound_dict[v] = {
                None: (v.lb, v.ub),
                'to_deactivate': ComponentMap(),
            }
        return v_bounds

    def _update_bounds_from_constraints(
        self, disjunct, bound_dict, gdp_forest, is_root=False
    ):
        for constraint in disjunct.component_data_objects(
            Constraint,
            active=True,
            descend_into=Block,
            sort=SortComponents.deterministic,
        ):
            # Avoid walking the whole expression tree if we have more than one
            # variable by just trying to get two. If we succeed at one but not
            # two, then the constraint is a bound or equality constraint and we
            # save it. Otherwise, we just keep going to the next constraint.
            var_gen = identify_variables(constraint.body, include_fixed=False)
            try:
                next(var_gen)
            except StopIteration:
                # No variables
                continue
            try:
                next(var_gen)
            except StopIteration:
                # There was one but not two: This is what we want.
                repn = generate_standard_repn(constraint.body)
                # If this is a trivial constraint, repn could actually be empty,
                # so we check that we really do have one linear var now
                if not repn.is_linear() or len(repn.linear_vars) != 1:
                    continue
                v = repn.linear_vars[0]
                coef = repn.linear_coefs[0]
                constant = repn.constant
                lower = (
                    (value(constraint.lower) - constant) / coef
                    if constraint.lower is not None
                    else None
                )
                upper = (
                    (value(constraint.upper) - constant) / coef
                    if constraint.upper is not None
                    else None
                )
                if coef < 0:
                    # we divided by a negative coef above, so flip the constraint
                    (lower, upper) = (upper, lower)
                v_bounds = self._get_bound_dict_for_var(bound_dict, v)
                self._update_bounds_dict(
                    v_bounds,
                    lower,
                    upper,
                    disjunct if not is_root else None,
                    gdp_forest,
                )
                if not is_root:
                    if disjunct in v_bounds['to_deactivate']:
                        v_bounds['to_deactivate'][disjunct].add(constraint)
                    else:
                        v_bounds['to_deactivate'][disjunct] = ComponentSet([constraint])

    def _get_tightest_ancestral_bounds(self, v_bounds, disjunct, gdp_forest):
        lb = None
        ub = None
        parent = disjunct
        while lb is None or ub is None:
            if parent in v_bounds:
                l, u = v_bounds[parent]
                if lb is None and l is not None:
                    lb = l
                if ub is None and u is not None:
                    ub = u
            if parent is None:
                break
            parent = gdp_forest.parent_disjunct(parent)
        return lb, ub

    def _update_bounds_dict(self, v_bounds, lower, upper, disjunct, gdp_forest):
        (lb, ub) = self._get_tightest_ancestral_bounds(v_bounds, disjunct, gdp_forest)
        if lower is not None:
            if lb is None or lower > lb:
                # This GDP is more constrained here than it was in the parent
                # Disjunct (what we would expect, usually. If it's looser, we're
                # essentially just ignoring it...)
                lb = lower
        if upper is not None:
            if ub is None or upper < ub:
                # Same case as above in the UB
                ub = upper
        # In all other cases, there is nothing to do... The parent gives more
        # information, so we just propagate that down
        v_bounds[disjunct] = (lb, ub)

    def _create_transformation_constraints(
        self, root_disjunction, bound_dict, gdp_forest, transformation_blocks
    ):
        to_transform = ComponentSet([root_disjunction])

        while to_transform:
            disjunction = to_transform.pop()

            trans_block = self._add_transformation_block(
                disjunction, transformation_blocks
            )
            if self.transformation_name not in disjunction._transformation_map:
                disjunction._transformation_map[self.transformation_name] = (
                    ComponentMap()
                )
            trans_map = disjunction._transformation_map[self.transformation_name]

            for disj in disjunction.disjuncts:
                to_transform.update(gdp_forest.children(disj))

            for v, v_bounds in bound_dict.items():
                unique_id = len(trans_block.transformed_bound_constraints)
                if not any(disj in v_bounds for disj in disjunction.disjuncts):
                    # There are no bound Constraints on this Disjunction. We
                    # don't want to create a bunch of empty Blocks and things as
                    # if there were, so we continue to the next.
                    continue
                all_lbs = True
                all_ubs = True
                lb_expr = 0
                ub_expr = 0
                deactivate_lower = ComponentSet()
                deactivate_upper = ComponentSet()
                for disj in disjunction.disjuncts:
                    (lb, ub) = self._get_tightest_ancestral_bounds(
                        v_bounds, disj, gdp_forest
                    )
                    if lb is None:
                        all_lbs = False
                        if not all_ubs:
                            # We're not going to get all of either: we're done.
                            break
                    if ub is None:
                        all_ubs = False
                        if not all_lbs:
                            break
                    if all_lbs:
                        lb_expr += lb * disj.binary_indicator_var
                        # If these bounds came from above here in the GDP
                        # hierarchy, this disjunct might not actually have
                        # constraints to deactivate. If it does, we add them to
                        # our list to take care of if we end up being able to
                        # make a constraint
                        if disj in v_bounds['to_deactivate']:
                            deactivate_lower.update(v_bounds['to_deactivate'][disj])
                    if all_ubs:
                        ub_expr += ub * disj.binary_indicator_var
                        if disj in v_bounds['to_deactivate']:
                            deactivate_upper.update(v_bounds['to_deactivate'][disj])

                # actually make the constraint(s) now
                if all_lbs:
                    idx = (v.local_name + '_lb', unique_id)
                    trans_block.transformed_bound_constraints[idx] = lb_expr <= v
                    trans_map[v] = [trans_block.transformed_bound_constraints[idx]]
                    for c in deactivate_lower:
                        if c.lower is None:
                            # There's nothing to do
                            continue
                        if c.upper is None or (all_ubs and c in deactivate_upper):
                            c.deactivate()
                        else:
                            c.deactivate()
                            c.parent_block().add_component(
                                unique_component_name(
                                    c.parent_block(), c.local_name + '_ub'
                                ),
                                Constraint(expr=v <= c.upper),
                            )
                if all_ubs:
                    idx = (v.local_name + '_ub', unique_id + 1)
                    trans_block.transformed_bound_constraints[idx] = ub_expr >= v
                    if v in trans_map:
                        trans_map[v].append(
                            trans_block.transformed_bound_constraints[idx]
                        )
                    else:
                        trans_map[v] = [trans_block.transformed_bound_constraints[idx]]
                    for c in deactivate_upper:
                        if c.upper is None:
                            # There's nothing to do
                            continue
                        if c.lower is None or (all_lbs and c in deactivate_lower):
                            c.deactivate()
                        else:
                            c.deactivate()
                            c.parent_block().add_component(
                                unique_component_name(
                                    c.parent_block(), c.local_name + '_lb'
                                ),
                                Constraint(expr=v >= c.lower),
                            )

    def _add_transformation_block(self, disjunction, transformation_blocks):
        to_block = disjunction.parent_block()
        if to_block in transformation_blocks:
            return transformation_blocks[to_block]

        trans_block_name = unique_component_name(
            to_block, '_pyomo_gdp_%s_reformulation' % self.transformation_name
        )
        transformation_blocks[to_block] = trans_block = Block()
        to_block.add_component(trans_block_name, trans_block)

        trans_block.transformed_bound_constraints = Constraint(
            Any * NonNegativeIntegers
        )

        return trans_block

    def get_transformed_constraints(self, v, disjunction):
        if self.transformation_name not in disjunction._transformation_map:
            logger.debug(
                "No variable on Disjunction '%s' was transformed with the "
                "gdp.%s transformation" % (disjunction.name, self.transformation_name)
            )
            return []
        trans_map = disjunction._transformation_map[self.transformation_name]
        if v not in trans_map:
            logger.debug(
                "Constraint bounding variable '%s' on Disjunction '%s' was "
                "not transformed by the 'gdp.%s' transformation"
                % (v.name, disjunction.name, self.transformation_name)
            )
            return []
        return trans_map[v]
