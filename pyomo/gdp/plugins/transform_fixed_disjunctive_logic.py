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

from pyomo.common.collections import ComponentSet
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.gdp import Disjunct, Disjunction


@TransformationFactory.register(
    'gdp.transform_fixed_disjunctive_logic',
    doc="""Convert fixed disjunctive logic (fixed Disjunct indicator_vars) to
    MI(N)LP form in cases where enough of the logic is fixed to determine 
    the logical solution.""",
)
class TransformFixedDisjunctiveLogic(Transformation):
    """This transformation finds fixed disjunctive logic (fixed indicator_vars)
    in the given GDP, and, for any Disjunction, when the fixed logic fully
    determines what Disjunct will be selected it reclassifies all the Disjuncts
    of the Disjunction as Blocks, and activates or deactivates them according
    to whether they are fixed (or implied to be fixed) as 'True' or 'False'
    (respectively).

    Note that this transformation does not necessarily return a MI(N)LP since
    it will not transform Disjunctions that are not fully determined by the
    fixed logic.

    If using 'apply_to' rather than 'create_using', this transformation is
    reversible. Calling apply_to returns a token to reverse the transformation
    In order to reverse the transformation, pass this token back to the
    transformation as the 'reverse' argument.
    """

    CONFIG = ConfigDict('gdp.transform_fixed_disjunctive_logic')
    CONFIG.declare(
        "reverse",
        ConfigValue(
            default=None,
            description="The token returned by a (forward) call to this "
            "transformation, if you wish to reverse the transformation.",
            doc="""
            This argument should be the reverse transformation token
            returned by a previous call to this transformation to transform
            fixed disjunctive logic in the given model.

            If this argument is specified, this call to the transformation
            will reverse what the transformation did in the call that returned
            the token. Note that if there are intermediate changes to the model
            in between the forward and the backward calls to the transformation,
            the behavior could be unexpected.
            """,
        ),
    )
    CONFIG.declare(
        "targets",
        ConfigValue(
            default=None,
            description="The target component or list of target "
            "components to transform.",
            doc="""
            This specifies the list of components to relax. If None (default),
            the entire model is transformed. These are expected to be Blocks or
            Disjunctions.

            Note that if the transformation is done out of place, the list of
            targets should be attached to the model before it is cloned, and the
            list will specify the targets on the cloned instance.""",
        ),
    )

    def _apply_to(self, model, **kwds):
        config = self.config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        if config.reverse is not None and config.targets is None:
            # we can do this super efficiently
            self._reverse_transformation(model, config.reverse)
            return

        targets = (model,) if config.targets is None else config.targets

        reverse_token = {'_disjunctions': {}, '_disjuncts': {}}
        for t in targets:
            if isinstance(t, Block):
                blocks = t.values() if t.is_indexed() else (t,)
                for block in blocks:
                    self._transform_block(block, reverse_token, reverse=config.reverse)
            elif t.ctype is Disjunction:
                disjunctions = t.values() if t.is_indexed() else (t,)
                for disj in disjunctions:
                    self._transform_disjunction(
                        disj, reverse_token, reverse=config.reverse
                    )
        if not config.reverse:
            return reverse_token

    def _transform_block(self, block, reverse_token, reverse):
        if reverse is not None:
            transform = self._transform_disjunction
        else:
            transform = self._reverse_transform_disjunction
            reverse_token = reverse

        for disjunction in block.component_data_objects(
            Disjunction,
            active=True,
            descend_into=(Block, Disjunct),
            sort=SortComponents.deterministic,
        ):
            transform(disjunction, reverse_token)

    def _transform_disjunction(self, disjunction, reverse_token):
        no_val = ComponentSet()
        true_val = ComponentSet()
        false_val = ComponentSet()
        num_disjuncts = len(disjunction.disjuncts)
        for disjunct in disjunction.disjuncts:
            ind_var = disjunct.indicator_var
            if value(ind_var) is None:
                no_val.add(disjunct)
            elif value(ind_var):
                true_val.add(disjunct)
            else:
                false_val.add(disjunct)
        if disjunction.xor:
            if len(true_val) > 1:
                raise InfeasibleConstraintException(
                    "Exactly-one constraint for Disjunction "
                    "'%s' is violated. The following Disjuncts "
                    "are all selected: %s"
                    % (disjunction.name, ', '.join([d.name for d in true_val]))
                )
            elif len(true_val) == 1:
                # We can fix everything
                self._reclassify_disjuncts(
                    true_val, no_val.union(false_val), reverse_token['_disjuncts']
                )
                # This disjunction is fully transformed
                reverse_token['_disjunctions'].add(disjunction)
                disjunction.deactivate()
            elif len(false_val) == num_disjuncts - 1:
                # We can fix everything. Since we didn't hit the case above, we
                # know that the non-False value is a None.
                self._reclassify_disjuncts(
                    no_val, false_val, reverse_token['_disjuncts']
                )
                reverse_token['_disjunctions'].add(disjunction)
                disjunction.deactivate()
        # It's only an 'at-least' not an 'exactly'...
        elif len(false_val) == num_disjuncts:
            raise InfeasibleConstraintException(
                "Atleast-one constraint for Disjunction "
                "'%s' is violated. That is, all the Disjunct "
                "indicator_vars are 'False'." % disjunction.name
            )
        elif len(no_val) == 0:
            self._reclassify_disjuncts(true_val, false_val, reverse_token['_disjuncts'])
            reverse_token['_disjunctions'].add(disjunction)
            disjunction.deactivate()

    def _reclassify_disjuncts(
        self, true_disjunctions, false_disjunctions, reverse_dict
    ):
        for disj in true_disjunctions:
            reverse_dict[disj] = (disj.indicator_var.fixed, value(disj.indicator_var))
            disj.indicator_var.fix(True)
            disj.parent_block().reclassify_component_type(disj, Block)
        for disj in false_disjunctions:
            reverse_dict[disj] = (disj.indicator_var.fixed, value(disj.indicator_var))
            # Deactivating fixes the indicator_var to False
            disj.deactivate()

    def _reverse_transformation(self, model, reverse_token, targets):
        if targets is None:
            for disjunction in reverse_token['_disjunctions']:
                disjunction.activate()
            for disjunct, (fixed, val) in reverse_token['_disjuncts']:
                disjunct.parent_component().reclassify_component_type(
                    disjunct, Disjunct
                )
                disjunct.indicator_var = val
                disjunct.indivaro_var.fixed = fixed
            return

        for t in targets:
            if isinstance(t, Block):
                blocks = t.values() if t.is_indexed() else (t,)
                for block in blocks:
                    self._reverse_transform_block(block, reverse_token)
            elif t.ctype is Disjunction:
                disjunctions = t.values() if t.is_indexed() else (t,)
                for disj in disjunctions:
                    self._reverse_transform_disjunction(disj, reverse_token)

    def _reverse_transform_disjunction(disjunction, reverse_token):
        if disjunction in reverse_token['_disjunctions']:
            disjunction.activate()
        for disjunt in disjunction.disjuncts:
            if disjunct in reverse_token['_disjuncts']:
                disjunct.parent_component().reclassify_component_type(
                    disjunct, Disjunct
                )
                (val, fixed) = reverse_token['_disjuncts'][disjunct]
                disjunct.indicator_var = val
                disjunct.indicator_var.fixed = fixed
