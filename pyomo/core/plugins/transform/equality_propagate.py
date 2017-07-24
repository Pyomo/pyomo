"""Transformation to propagate state through an equality set."""
from pyomo.core.base.block import generate_cuid_names
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.expr import _SumExpression
from pyomo.core.base.var import Var, _GeneralVarData
from pyomo.core.kernel.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.util.plugin import alias
from six import iteritems

__author__ = "Qi Chen <qichen at andrew.cmu.edu>"


class FixedVarPropagator(IsomorphicTransformation):
    """Propagates variable fixing for equalities of type x = y.

    If x is fixed and y is not fixed, then this function will fix y to the
    value of x.

    This transformation can also be performed as a temporary transformation,
    whereby the transformed variables are saved and can be later unfixed.

    """

    alias('core.propagate_fixed_vars', doc=__doc__)

    def __init__(self):
        """Initialize the transformation."""
        super(FixedVarPropagator, self).__init__()
        self._tmp_propagate_fixed = set()
        self._transformed_instance = None

    def _apply_to(self, instance, tmp=False):
        """Apply the transformation.

        Args:
            instance (Block): the block on which to search for x == y
                constraints. Note that variables may be located anywhere in
                the model.
            tmp (bool, optional): Whether the variable fixing will be temporary

        Returns:
            None

        Raises:
            ValueError: if two fixed variables x = y have different values.

        """
        if tmp:
            self._transformed_instance = instance
        #: dict: Mapping of variable to its UID
        var_to_id = self.var_to_id = generate_cuid_names(
            instance.model(), ctype=Var, descend_into=True)
        id_to_var = self.id_to_var = {
            cuid: obj for obj, cuid in iteritems(var_to_id)}
        #: set of UIDs: The set of all fixed variables
        fixed_vars = set()
        #: dict: map of var UID to the set of all equality-linked var UIDs
        eq_var_map = {}
        for constr in instance.component_data_objects(ctype=Constraint,
                                                      active=True,
                                                      descend_into=True):
            # Check to make sure the constraint is of form v1 - v2 == 0
            if constr.lower == 0 and constr.upper == 0 \
                    and isinstance(constr.body, _SumExpression) \
                    and len(constr.body._args) == 2 \
                    and 1 in constr.body._coef \
                    and -1 in constr.body._coef \
                    and isinstance(constr.body._args[0], _GeneralVarData) \
                    and isinstance(constr.body._args[1], _GeneralVarData) \
                    and constr.body._const == 0:
                v1 = constr.body._args[0]
                v2 = constr.body._args[1]
                if v1.fixed:
                    fixed_vars.add(var_to_id[v1])
                if v2.fixed:
                    fixed_vars.add(var_to_id[v2])
                set1 = eq_var_map.get(var_to_id[v1], set([var_to_id[v1]]))
                set2 = eq_var_map.get(var_to_id[v2], set([var_to_id[v2]]))
                union = set1.union(set2)
                for vID in union:
                    eq_var_map[vID] = union
        processed = set()
        # Go through each fixed variable to propagate the 'fixed' status to all
        # equality-linked variabes.
        for v1ID in fixed_vars:
            # If we have already processed the variable, skip it.
            if v1ID not in processed:
                var_val = value(id_to_var[v1ID])
                eq_set = eq_var_map[v1ID]
                for v2ID in eq_set:
                    if (id_to_var[v2ID].fixed and
                            value(id_to_var[v2ID]) != var_val):
                        raise ValueError(
                            'Variables {} and {} have conflicting fixed '
                            'values of {} and {}, but are linked by '
                            'equality constraints.'
                            .format(id_to_var[v1ID].name,
                                    id_to_var[v2ID].name,
                                    value(id_to_var[v1ID]),
                                    value(id_to_var[v2ID])))
                    elif not id_to_var[v2ID].fixed:
                        id_to_var[v2ID].fix(var_val)
                        if tmp:
                            self._tmp_propagate_fixed.add(v2ID)
                # Add all variables in the equality set to the set of processed
                # variables.
                processed |= eq_set

    def revert(self):
        """Revert variables fixed by the transformation."""
        for varUID in self._tmp_propagate_fixed:
            var = self.id_to_var[varUID]
            var.unfix()
        self._tmp_propagate_fixed.clear()
