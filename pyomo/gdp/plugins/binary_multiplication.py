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

from .gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core.base import TransformationFactory
from pyomo.core.util import target_list
from pyomo.gdp import Disjunction
from weakref import ref as weakref_ref
import logging


logger = logging.getLogger(__name__)


@TransformationFactory.register(
    'gdp.binary_multiplication',
    doc="Reformulate the GDP as an MINLP by multiplying f(x) <= 0 by y to get "
    "f(x) * y <= 0 where y is the binary corresponding to the Boolean indicator "
    "var of the Disjunct containing f(x) <= 0.",
)
class GDPBinaryMultiplicationTransformation(GDP_to_MIP_Transformation):
    CONFIG = ConfigDict("gdp.binary_multiplication")
    CONFIG.declare(
        'targets',
        ConfigValue(
            default=None,
            domain=target_list,
            description="target or list of targets that will be transformed",
            doc="""

        This specifies the list of components to transform. If None (default), the
        entire model is transformed. Note that if the transformation is done out
        of place, the list of targets should be attached to the model before it
        is cloned, and the list will specify the targets on the cloned
        instance.""",
        ),
    )

    transformation_name = 'binary_multiplication'

    def __init__(self):
        super().__init__(logger)

    def _apply_to(self, instance, **kwds):
        try:
            self._apply_to_impl(instance, **kwds)
        finally:
            self._restore_state()

    def _apply_to_impl(self, instance, **kwds):
        self._process_arguments(instance, **kwds)

        # filter out inactive targets and handle case where targets aren't
        # specified.
        targets = self._filter_targets(instance)
        # transform logical constraints based on targets
        self._transform_logical_constraints(instance, targets)
        # we need to preprocess targets to make sure that if there are any
        # disjunctions in targets that their disjuncts appear before them in
        # the list.
        gdp_tree = self._get_gdp_tree_from_targets(instance, targets)
        preprocessed_targets = gdp_tree.reverse_topological_sort()

        for t in preprocessed_targets:
            if t.ctype is Disjunction:
                self._transform_disjunctionData(
                    t,
                    t.index(),
                    parent_disjunct=gdp_tree.parent(t),
                    root_disjunct=gdp_tree.root_disjunct(t),
                )

    def _transform_disjunctionData(
        self, obj, index, parent_disjunct=None, root_disjunct=None
    ):
        (transBlock, xorConstraint) = self._setup_transform_disjunctionData(
            obj, root_disjunct
        )

        # add or (or xor) constraint
        or_expr = 0
        for disjunct in obj.disjuncts:
            or_expr += disjunct.binary_indicator_var
            self._transform_disjunct(disjunct, transBlock)

        if obj.xor:
            xorConstraint[index] = or_expr == 1
        else:
            xorConstraint[index] = or_expr >= 1
        # Mark the DisjunctionData as transformed by mapping it to its XOR
        # constraint.
        obj._algebraic_constraint = weakref_ref(xorConstraint[index])

        # and deactivate for the writers
        obj.deactivate()

    def _transform_disjunct(self, obj, transBlock):
        # We're not using the preprocessed list here, so this could be
        # inactive. We've already done the error checking in preprocessing, so
        # we just skip it here.
        if not obj.active:
            return

        relaxationBlock = self._get_disjunct_transformation_block(obj, transBlock)

        # Transform each component within this disjunct
        self._transform_block_components(obj, obj)

        # deactivate disjunct to keep the writers happy
        obj._deactivate_without_fixing_indicator()

    def _transform_constraint(self, obj, disjunct):
        # add constraint to the transformation block, we'll transform it there.
        transBlock = disjunct._transformation_block()
        constraint_map = transBlock.private_data('pyomo.gdp')

        disjunctionRelaxationBlock = transBlock.parent_block()

        # We will make indexes from ({obj.local_name} x obj.index_set() x ['lb',
        # 'ub']), but don't bother construct that set here, as taking Cartesian
        # products is kind of expensive (and redundant since we have the
        # original model)
        newConstraint = transBlock.transformedConstraints

        for i in sorted(obj.keys()):
            c = obj[i]
            if not c.active:
                continue

            self._add_constraint_expressions(
                c, i, disjunct.binary_indicator_var, newConstraint, constraint_map
            )

            # deactivate because we relaxed
            c.deactivate()

    def _add_constraint_expressions(
        self, c, i, indicator_var, newConstraint, constraint_map
    ):
        # Since we are both combining components from multiple blocks and using
        # local names, we need to make sure that the first index for
        # transformedConstraints is guaranteed to be unique. We just grab the
        # current length of the list here since that will be monotonically
        # increasing and hence unique. We'll append it to the
        # slightly-more-human-readable constraint name for something familiar
        # but unique. (Note that we really could do this outside of the loop
        # over the constraint indices, but I don't think it matters a lot.)
        unique = len(newConstraint)
        name = c.local_name + "_%s" % unique
        transformed = constraint_map.transformed_constraints[c]

        lb, ub = c.lower, c.upper
        if (c.equality or lb is ub) and lb is not None:
            # equality
            newConstraint.add((name, i, 'eq'), (c.body - lb) * indicator_var == 0)
            transformed.append(newConstraint[name, i, 'eq'])
            constraint_map.src_constraint[newConstraint[name, i, 'eq']] = c
        else:
            # inequality
            if lb is not None:
                newConstraint.add((name, i, 'lb'), 0 <= (c.body - lb) * indicator_var)
                transformed.append(newConstraint[name, i, 'lb'])
                constraint_map.src_constraint[newConstraint[name, i, 'lb']] = c
            if ub is not None:
                newConstraint.add((name, i, 'ub'), (c.body - ub) * indicator_var <= 0)
                transformed.append(newConstraint[name, i, 'ub'])
                constraint_map.src_constraint[newConstraint[name, i, 'ub']] = c
