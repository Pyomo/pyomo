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

from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import Block, Constraint, SortComponents
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import get_gdp_tree
import logging

## debug
from pytest import set_trace

logger = logging.getLogger('pyomo.gdp.common_constraint_body')

@TransformationFactory.register(
    'gdp.common_constraint_body',
    doc="Partially transforms a GDP to a MIP by finding all disjunctive "
    "constraints with common left-hand sides and transforming them according "
    "to Balas 1988, Blair, and Jeroslow (TODO: I think)")
class CommonLHSTransformation(Transformation):
    """
    Implements the special transformation mentioned in [1], [2], and [3] for
    handling disjunctive constraints with common left-hand sides (i.e., 
    Constraint bodies).

    [1] Egon Balas, "On the convex hull of the union of certain polyhedra,"
        Operations Research Letters, vol. 7, 1988, pp. 279-283
    [2] TODO: Blair 1990
    [3] TODO: Jeroslow 1988
    """

    CONFIG = ConfigDict('gdp.common_constraint_body')
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
            """
        )
    )
    transformation_name = 'common_constraint_body'
    
    def __init__(self):
        super().__init__()
        self.logger = logger
        
    def _apply_to(self, instance, **kwds):
        if not instance.ctype in (Block, Disjunct):
            raise GDP_Error("Transformation called on %s of type %s. 'instance'"
                            " must be a ConcreteModel, Block, or Disjunct (in "
                            "the case of nested disjunctions)." %
                            (instance.name, instance.ctype))

        self._config = self.CONFIG(kwds.pop('options', {}))
        self._config.set_value(kwds)

        targets = self._config.targets
        if targets is None:
            targets = (instance,)

        for t in targets:
            gdp_forest = get_gdp_tree(t, instance)
            # we have to go from leaf to root because we pass bound information
            # upwards--the innermost disjuncts should restrict it the most. If
            # that's not true, they're useless, and if there are contradictions,
            # we'll catch them.
            bound_dict = ComponentMap()
            for d in gdp_forest.reverse_topological_sort():
                if d.ctype is Disjunct:
                    self._update_bounds_from_constraints(d, bound_dict)
                elif d.ctype is Disjunction and gdp_tree.parent(d) is None:
                    # we're at a root, finish the transformation
                    self._transform_disjunction(bound_dict)

    def _update_bounds_from_constraints(self, disjunct, bound_dict):
        for constraint in disjunct.component_data_objects(
                Constraint,
                active=True,
                descend_into=Block,
                order=SortComponents.deterministic):
            if (hasattr(constraint.body, 'ctype')
                and constraint.body.ctype is Var):
                v = constraint.body
                # Then this is a bound or an equality 
                current_bounds = bound_dict.get(v)
                if current_bounds is None:
                    # TODO: These need to be values, I think. We can guarantee
                    # this is right when mutable stuff changes...
                    bound_dict[v] = (constraint.lower, constraint.upper,
                                     disjunct.binary_indicator_var,
                                     disjunct.binary_indicator_var)
                else:
                    self._update_bounds_dict(bound_dict, v, constraint,
                                             disjunct.binary_indicator_var,
                                             current_bounds)

    def _update_bounds_dict(self, bound_dict, variable, cons, indicator_var,
                            current_bounds):
        (lb, ub, lb_indicator_var, ub_indicator_var) = current_bounds
        if value(cons.lower) <= lb:
            # This GDP is more constrained higher in the tree. This is a bit
            # surprising since it means the descendent constraint is useless,
            # but we just replace what we will put in the final constraint:
            lb = cons.lower
            lb_indicator_var = indicator_var
        if value(cons.upper) <= ub:
            # Same surprise as above in the UB
            ub = current_ub
            ub_indicator_var = indicator_var
        # In all other cases, there is nothing to do... The lower gives more
        # information and all the logical stuff will be filled in later by a
        # "full" transformation to MIP.
        bounds_dict[v] = (lb, ub, lb_indicator_var, ub_indicator_var)

    def _transform_disjunction(self, disjunction, bound_dict,
                               transformation_blocks):
        trans_block = self._add_transformation_block(disjunction,
                                                     transformation_blocks)

    def _add_transformation_block(self, disjunction, transformation_blocks):
        to_block = disjunction.parent_block()
        if to_block in transformation_blocks:
            return transformation_blocks[disjunction]

        trans_block_name = unique_component_name(
            to_block,
            '_pyomo_gdp_common_constraint_body_reformulation'
        )
        transformation_blocks[to_block] = trans_block = Block()
        to_block.add_component(trans_block_name, trans_block)

        trans_block.transformed_bound_constraints = Constraint(
            NonNegativeIntegers)
        
        return trans_block
