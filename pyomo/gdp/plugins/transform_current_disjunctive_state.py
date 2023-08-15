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
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (
    Any,
    Block,
    SortComponents,
    Transformation,
    TransformationFactory,
    ReverseTransformationToken,
)
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error


@TransformationFactory.register(
    'gdp.transform_current_disjunctive_state',
    doc="""Convert current disjunctive state (values of Disjunct
    indicator_vars) to MI(N)LP form in cases where enough of the 
    logic is fixed to determine which Disjunct(s) will be selected.
    """,
)
class TransformCurrentDisjunctiveState(Transformation):
    """This transformation finds disjunctive state (indicator_var values)
    in the given GDP, and, for any Disjunction, when the state fully
    determines what Disjunct will be selected, it reclassifies all the Disjuncts
    of the Disjunction as Blocks, and activates or deactivates them according
    to whether their indicator_vars are set (or implied to be set) as
    'True' or 'False' (respectively).

    Note that this transformation does not necessarily return a MI(N)LP since
    it will not transform Disjunctions that are not fully determined by the
    current state. Be careful in partially-transformed states to remember that
    if even one DisjunctData in an IndexedDisjunct is reclassified as a Block,
    all of the DisjunctDatas will be as well. It is strongly recommended to not use
    DisjunctDatas from a single IndexedDisjunction in multiple Disjunctions
    if you will be working with the partially-transformed model.

    If using 'apply_to' rather than 'create_using', this transformation is
    reversible. Calling apply_to returns a token to reverse the transformation
    In order to reverse the transformation, pass this token back to the
    transformation as the 'reverse' argument.
    """

    CONFIG = ConfigDict('gdp.transform_current_disjunctive_state')
    CONFIG.declare(
        "reverse",
        ConfigValue(
            default=None,
            description="The token returned by a (forward) call to this "
            "transformation, if you wish to reverse the transformation.",
            doc="""
            This argument should be the reverse transformation token
            returned by a previous call to this transformation to transform
            fixed disjunctive state in the given model.

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
            This specifies the list of components to transform. If None (default),
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

        if config.reverse is not None and config.targets is not None:
            raise ValueError(
                "The 'gdp.transform_current_disjunctive_state' transformation "
                "cannot be called with both targets and a reverse token "
                "specified. If reversing the transformation, do not include "
                "targets: The reverse transformation will restore all the "
                "components the original transformation call transformed."
            )

        targets = (model,) if config.targets is None else config.targets

        if config.reverse is None:
            disjunct_containers = set()
            disjunct_set = set()
            reverse_dict = {'_disjunctions': set(), '_disjuncts': {}}
            reverse_token = ReverseTransformationToken(
                self.__class__, model, targets, reverse_dict
            )
            disjunction_transform = self._transform_disjunction
        else:
            reverse_token = config.reverse
            reverse_token.check_token_valid(self.__class__, model, targets)
            reverse_dict = reverse_token.reverse_dict
            disjunct_set = None
            disjunct_containers = None
            disjunction_transform = self._reverse_transform_disjunction

        for t in targets:
            if isinstance(t, Block):
                blocks = t.values() if t.is_indexed() else (t,)
                for block in blocks:
                    self._transform_block(
                        block,
                        reverse_dict,
                        disjunction_transform,
                        disjunct_set,
                        disjunct_containers,
                    )
            elif t.ctype is Disjunction:
                disjunctions = t.values() if t.is_indexed() else (t,)
                for disj in disjunctions:
                    disjunction_transform(
                        disj, reverse_dict, disjunct_set, disjunct_containers
                    )
        if config.reverse is None:
            if len(disjunct_set) > 0:
                raise GDP_Error(
                    "Found active Disjuncts on the model that "
                    "were not included in any Disjunctions:\n%s\nPlease "
                    "deactivate them or include them in a Disjunction."
                    % ', '.join([d.name for d in disjunct_set])
                )
            return reverse_token

    def _transform_block(
        self,
        block,
        reverse_dict,
        disjunction_transform,
        disjunct_set,
        disjunct_containers,
    ):
        # We iterate through inactive ones as well in case this is a reverse
        # transformation. We will check for active status in the
        # disjunction_transform function.
        for disjunction in block.component_data_objects(
            Disjunction,
            active=None,
            descend_into=(Block, Disjunct),
            sort=SortComponents.deterministic,
        ):
            disjunction_transform(
                disjunction, reverse_dict, disjunct_set, disjunct_containers
            )

    def _transform_disjunction(
        self, disjunction, reverse_dict, disjunct_set, disjunct_containers
    ):
        if not disjunction.active:
            return
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
        can_transform = True
        if len(false_val) == num_disjuncts:
            raise InfeasibleConstraintException(
                "Logical constraint for Disjunction "
                "'%s' is violated: All the Disjunct "
                "indicator_vars are 'False.'" % disjunction.name
            )
        elif disjunction.xor:
            if len(true_val) > 1:
                raise InfeasibleConstraintException(
                    "Exactly-one constraint for Disjunction "
                    "'%s' is violated. The following Disjuncts "
                    "are selected: %s"
                    % (disjunction.name, ', '.join([d.name for d in true_val]))
                )
            elif len(true_val) == 1:
                # We can fix everything
                self._reclassify_disjuncts(
                    true_val,
                    no_val.union(false_val),
                    reverse_dict['_disjuncts'],
                    disjunct_set,
                    disjunct_containers,
                )
                # This disjunction is fully transformed
                reverse_dict['_disjunctions'].add(disjunction)
                disjunction.deactivate()
            elif len(false_val) == num_disjuncts - 1:
                # We can fix everything. Since we didn't hit the case above, we
                # know that the non-False value is a None.
                self._reclassify_disjuncts(
                    no_val,
                    false_val,
                    reverse_dict['_disjuncts'],
                    disjunct_set,
                    disjunct_containers,
                )
                reverse_dict['_disjunctions'].add(disjunction)
                disjunction.deactivate()
            else:
                can_transform = False
        # It's only an 'at-least' not an 'exactly', so if everything has a value
        # we can transform it
        elif len(no_val) == 0:
            self._reclassify_disjuncts(
                true_val,
                false_val,
                reverse_dict['_disjuncts'],
                disjunct_set,
                disjunct_containers,
            )
            reverse_dict['_disjunctions'].add(disjunction)
            disjunction.deactivate()
        else:
            can_transform = False
        if not can_transform:
            raise GDP_Error(
                "Disjunction '%s' does not contain enough Disjuncts with "
                "values in their indicator_vars to specify which Disjuncts "
                "are True. Cannot fully transform model." % disjunction.name
            )

    def _update_transformed_disjuncts(self, disj, disjunct_set, disjunct_containers):
        parent = disj.parent_component()
        if parent.is_indexed():
            if parent not in disjunct_containers:
                disjunct_set.update(d for d in parent.values() if d.active)
                disjunct_containers.add(parent)
            disjunct_set.remove(disj)

    def _reclassify_disjuncts(
        self,
        true_disjuncts,
        false_disjuncts,
        reverse_dict,
        disjunct_set,
        disjunct_containers,
    ):
        for disj in true_disjuncts:
            reverse_dict[disj] = (disj.indicator_var.fixed, disj.indicator_var.value)
            self._update_transformed_disjuncts(disj, disjunct_set, disjunct_containers)
            parent_block = disj.parent_block()
            parent_block.reclassify_component_type(disj, Block)
            disj.indicator_var.fix(True)

        for disj in false_disjuncts:
            reverse_dict[disj] = (disj.indicator_var.fixed, disj.indicator_var.value)
            # Deactivating fixes the indicator_var to False
            self._update_transformed_disjuncts(disj, disjunct_set, disjunct_containers)
            disj.deactivate()

    def _reverse_transform_disjunction(
        self, disjunction, reverse_token, disjunct_set, disjunct_containers
    ):
        if disjunction in reverse_token['_disjunctions']:
            disjunction.activate()
        for disjunct in disjunction.disjuncts:
            if disjunct in reverse_token['_disjuncts']:
                (fixed, val) = reverse_token['_disjuncts'][disjunct]
                disjunct.parent_block().reclassify_component_type(disjunct, Disjunct)
                disjunct.activate()
                disjunct.indicator_var = val
                disjunct.indicator_var.fixed = fixed
