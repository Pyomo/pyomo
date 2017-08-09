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


def _build_equality_set(m, var_to_id, id_to_var):
    """Construct an equality set map.

    Maps all variables to the set of variables that are linked to them by
    equality. Mapping takes place using ComponentUID. That is, if you have x =
    y, then you would have xUID -> set([xUID, yUID]) and yUID -> set([xUID,
    yUID]) in the mapping.

    var_to_id and id_to_var are the model symbol map and reverse map,
    respectively.

    """
    #: dict: map of var UID to the set of all equality-linked var UIDs
    eq_var_map = {}
    for constr in m.component_data_objects(ctype=Constraint,
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
            set1 = eq_var_map.get(var_to_id[v1], set([var_to_id[v1]]))
            set2 = eq_var_map.get(var_to_id[v2], set([var_to_id[v2]]))
            union = set1.union(set2)
            for vID in union:
                eq_var_map[vID] = union

    return eq_var_map


class FixedVarPropagator(IsomorphicTransformation):
    """Propagates variable fixing for equalities of type x = y.

    If x is fixed and y is not fixed, then this transformation will fix y to
    the value of x.

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
        #: dict: Mapping of UIDs to variables
        id_to_var = self.id_to_var = dict(
            (cuid, obj) for obj, cuid in iteritems(var_to_id))
        eq_var_map = _build_equality_set(instance, var_to_id, id_to_var)
        #: set of UIDs: The set of all fixed variables
        fixed_vars = set(vID for vID in eq_var_map
                         if self.id_to_var[vID].fixed)
        processed = set()
        # Go through each fixed variable to propagate the 'fixed' status to all
        # equality-linked variabes.
        for v1ID in fixed_vars:
            # If we have already processed the variable, skip it.
            if v1ID in processed:
                continue

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


class VarBoundPropagator(IsomorphicTransformation):
    """Propagates variable bounds for equalities of type x = y.

    If x has a tighter bound then y, then this transformation will adjust the
    bounds on y to match those of x.

    """

    alias('core.propagate_eq_var_bounds', doc=__doc__)

    def __init__(self):
        """Initialize the transformation."""
        super(VarBoundPropagator, self).__init__()
        self._old_bounds = {}

    def _apply_to(self, instance, tmp=False):
        """Apply the transformation.

        Args:
            instance (Block): the block on which to search for x == y
                constraints. Note that variables may be located anywhere in
                the model.
            tmp (bool, optional): Whether the bound modifications will be
                temporary

        Returns:
            None

        """
        if tmp:
            self._transformed_instance = instance
        #: dict: Mapping of variable to its UID
        var_to_id = self.var_to_id = generate_cuid_names(
            instance.model(), ctype=Var, descend_into=True)
        #: dict: Mapping of UIDs to variables
        id_to_var = self.id_to_var = dict(
            (cuid, obj) for obj, cuid in iteritems(var_to_id))
        eq_var_map = _build_equality_set(instance, var_to_id, id_to_var)
        processed = set()
        # Go through each variable in an equality set to propagate the variable
        # bounds to all equality-linked variables.
        for varID in eq_var_map:
            # If we have already processed the variable, skip it.
            if varID in processed:
                continue

            lbs = [id_to_var[vID].lb
                   for vID in eq_var_map[varID]
                   if id_to_var[vID].lb is not None]
            max_lb = max(lbs) if len(lbs) > 0 else None
            ubs = [id_to_var[vID].ub
                   for vID in eq_var_map[varID]
                   if id_to_var[vID].ub is not None]
            min_ub = min(ubs) if len(ubs) > 0 else None

            if max_lb > min_ub:
                # the lower bound is above the upper bound. Raise a ValueError.
                # get variable with the highest lower bound
                v1ID = next(vID for vID in eq_var_map[varID]
                            if id_to_var[vID].lb == max_lb)
                # get variable with the lowest upper bound
                v2ID = next(vID for vID in eq_var_map[varID]
                            if id_to_var[vID].ub == min_ub)
                raise ValueError(
                    'Variable {} has a lower bound {} '
                    ' > the upper bound {} of variable {}, '
                    'but they are linked by equality constraints.'
                    .format(id_to_var[v1ID].name,
                            value(id_to_var[v1ID].lb),
                            value(id_to_var[v2ID].ub),
                            id_to_var[v2ID].name))
            else:
                for vID in eq_var_map[varID]:
                    self._old_bounds[vID] = (id_to_var[vID].lb,
                                             id_to_var[vID].ub)
                    id_to_var[vID].setlb(max_lb)
                    id_to_var[vID].setub(min_ub)

                processed |= eq_var_map[varID]

    def revert(self):
        """Revert variable bounds."""
        for varID in self._old_bounds:
            old_LB, old_UB = self._old_bounds[varID]
            self.id_to_var[varID].setlb(old_LB)
            self.id_to_var[varID].setub(old_UB)
        self._old_bounds = {}
