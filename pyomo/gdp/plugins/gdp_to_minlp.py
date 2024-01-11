from .gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core.base import TransformationFactory
from pyomo.core.util import target_list
from pyomo.gdp import Disjunction
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.core.util import target_list
from weakref import ref as weakref_ref
import logging


logger = logging.getLogger('pyomo.gdp.gdp_to_minlp')


@TransformationFactory.register(
    'gdp.gdp_to_minlp', doc="Reformulate the GDP as an MINLP."
)
class GDPToMINLPTransformation(GDP_to_MIP_Transformation):
    CONFIG = ConfigDict("gdp.gdp_to_minlp")
    CONFIG.declare(
        'targets',
        ConfigValue(
            default=None,
            domain=target_list,
            description="target or list of targets that will be relaxed",
            doc="""

        This specifies the list of components to relax. If None (default), the
        entire model is transformed. Note that if the transformation is done out
        of place, the list of targets should be attached to the model before it
        is cloned, and the list will specify the targets on the cloned
        instance.""",
        ),
    )

    transformation_name = 'gdp_to_minlp'

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

        rhs = 1 if parent_disjunct is None else parent_disjunct.binary_indicator_var
        if obj.xor:
            xorConstraint[index] = or_expr == rhs
        else:
            xorConstraint[index] = or_expr >= rhs
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
        constraintMap = transBlock._constraintMap

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
                c, i, disjunct.binary_indicator_var, newConstraint, constraintMap
            )

            # deactivate because we relaxed
            c.deactivate()

    def _add_constraint_expressions(
        self, c, i, indicator_var, newConstraint, constraintMap
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

        lb, ub = c.lower, c.upper
        if (c.equality or lb is ub) and lb is not None:
            # equality
            newConstraint.add(
                (name, i, 'eq'), (c.body - lb) * indicator_var == 0
            )
            constraintMap['transformedConstraints'][c] = [newConstraint[name, i, 'eq']]
            constraintMap['srcConstraints'][newConstraint[name, i, 'eq']] = c
        else:
            # inequality
            if lb is not None:
                newConstraint.add(
                    (name, i, 'lb'), 0 <= (c.body - lb) * indicator_var
                )
                constraintMap['transformedConstraints'][c] = [
                    newConstraint[name, i, 'lb']
                ]
                constraintMap['srcConstraints'][newConstraint[name, i, 'lb']] = c
            if ub is not None:
                newConstraint.add(
                    (name, i, 'ub'), (c.body - ub) * indicator_var <= 0
                )
                transformed = constraintMap['transformedConstraints'].get(c)
                if transformed is not None:
                    constraintMap['transformedConstraints'][c].append(
                        newConstraint[name, i, 'ub']
                    )
                else:
                    constraintMap['transformedConstraints'][c] = [
                        newConstraint[name, i, 'ub']
                    ]
                constraintMap['srcConstraints'][newConstraint[name, i, 'ub']] = c
