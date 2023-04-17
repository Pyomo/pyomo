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
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.core.base import (
    Block,
    SortComponents,
    ReversibleTransformation,
    TransformationFactory,
)
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction


@TransformationFactory.register(
    'gdp.transform_current_disjunctive_logic',
    doc="""Convert current disjunctive logic (values of Disjunct
    indicator_vars) to MI(N)LP form in cases where enough of the 
    logic is fixed to determine which Disjunct(s) will be selected.
    """,
)
class TransformCurrentDisjunctiveLogic(ReversibleTransformation):
    """This transformation finds disjunctive logic (indicator_vars values)
    in the given GDP, and, for any Disjunction, when the logic fully
    determines what Disjunct will be selected, it reclassifies all the Disjuncts
    of the Disjunction as Blocks, and activates or deactivates them according
    to whether their indicator_vars are set (or implied to be set) as
    'True' or 'False' (respectively).

    Note that this transformation does not necessarily return a MI(N)LP since
    it will not transform Disjunctions that are not fully determined by the
    current logic. Be careful in partially-transformed states to remember that
    if even one DisjunctData in an IndexedDisjunct is reclassified as a Block,
    all of the DisjunctDatas will be as well. It is strongly recommended to not use
    DisjunctDatas from a single IndexedDisjunction in multiple Disjunctions
    if you will be working with the partially-transformed model.

    If using 'apply_to' rather than 'create_using', this transformation is
    reversible. Calling apply_to returns a token to reverse the transformation
    In order to reverse the transformation, pass this token back to the
    transformation as the 'reverse' argument.
    """

    CONFIG = ConfigDict('gdp.transform_current_disjunctive_logic')
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
            domain=target_list,
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
            # we can do this super efficiently and be done
            self._reverse_transformation(model, config.reverse)
            return

        targets = (model,) if config.targets is None else config.targets

        if config.reverse is None:
            reverse_token = {'_disjunctions': set(), '_disjuncts': {}}
            disjunction_transform = self._transform_disjunction
        else:
            reverse_token = config.reverse
            disjunction_transform = self._reverse_transform_disjunction

        for t in targets:
            if isinstance(t, Block):
                blocks = t.values() if t.is_indexed() else (t,)
                for block in blocks:
                    self._transform_block(block, reverse_token, disjunction_transform)
            elif t.ctype is Disjunction:
                disjunctions = t.values() if t.is_indexed() else (t,)
                for disj in disjunctions:
                    disjunction_transform(disj, reverse_token)
        if config.reverse is None:
            return reverse_token

    def _transform_block(self, block, reverse_token, disjunction_transform):
        for disjunction in block.component_data_objects(
            Disjunction,
            active=True,
            descend_into=(Block, Disjunct),
            sort=SortComponents.deterministic,
        ):
            disjunction_transform(disjunction, reverse_token)

    def _transform_disjunction(self, disjunction, reverse_token):
        no_val = set()
        true_val = set()
        false_val = set()
        num_disjuncts = 0
        for disjunct in disjunction.disjuncts:
            if not disjunct.active:
                continue
            num_disjuncts += 1
            ind_var = disjunct.indicator_var
            if ind_var.value is None:
                no_val.add(disjunct)
            elif ind_var.value:
                true_val.add(disjunct)
            else:
                false_val.add(disjunct)
        if disjunction.xor:
            if len(true_val) > 1 or len(false_val) == num_disjuncts:
                raise InfeasibleConstraintException(
                    "Exactly-one constraint for Disjunction "
                    "'%s' is violated. The following Disjuncts "
                    "are selected: %s"
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
            reverse_dict[disj] = (disj.indicator_var.fixed, disj.indicator_var.value)
            disj.indicator_var.fix(True)
            disj.parent_block().reclassify_component_type(disj, Block)
        for disj in false_disjunctions:
            reverse_dict[disj] = (disj.indicator_var.fixed, disj.indicator_var.value)
            # Deactivating fixes the indicator_var to False
            disj.deactivate()

    def _reverse_transformation(self, model, reverse_token):
        for disjunction in reverse_token['_disjunctions']:
            disjunction.activate()
        for disjunct, (fixed, val) in reverse_token['_disjuncts'].items():
            disjunct.parent_block().reclassify_component_type(disjunct, Disjunct)
            disjunct.activate()
            disjunct.indicator_var = val
            disjunct.indicator_var.fixed = fixed

    def _reverse_transform_disjunction(self, disjunction, reverse_token):
        if disjunction in reverse_token['_disjunctions']:
            disjunction.activate()
        for disjunct in disjunction.disjuncts:
            if disjunct in reverse_token['_disjuncts']:
                disjunct.parent_block().reclassify_component_type(disjunct, Disjunct)
                disjunct.activate()
                (fixed, val) = reverse_token['_disjuncts'][disjunct]
                disjunct.indicator_var = val
                disjunct.indicator_var.fixed = fixed
