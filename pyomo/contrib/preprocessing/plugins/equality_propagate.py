"""Transformation to propagate state through an equality set."""
import textwrap

from pyomo.core.base.constraint import Constraint
from pyomo.core.base.suffix import Suffix
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn.canonical_repn import generate_canonical_repn
from pyomo.util.plugin import alias


def _build_equality_set(m):
    """Construct an equality set map.

    Maps all variables to the set of variables that are linked to them by
    equality. Mapping takes place using id(). That is, if you have x = y, then
    you would have id(x) -> ComponentSet([x, y]) and id(y) -> ComponentSet([x,
    y]) in the mapping.

    """
    #: dict: map of var UID to the set of all equality-linked var UIDs
    eq_var_map = ComponentMap()
    relevant_vars = ComponentSet()
    for constr in m.component_data_objects(ctype=Constraint,
                                           active=True,
                                           descend_into=True):
        # Check to make sure the constraint is of form v1 - v2 == 0
        if (value(constr.lower) == 0 and value(constr.upper) == 0 and
                constr.body.polynomial_degree() == 1):
            repn = generate_canonical_repn(constr.body)
            # only take the variables with nonzero coefficients
            vars_ = [v for i, v in enumerate(repn.variables) if repn.linear[i]]
            if (len(vars_) == 2 and
                    sorted(l for l in repn.linear if l) == [-1, 1]):
                # this is an a == b constraint.
                v1 = vars_[0]
                v2 = vars_[1]
                set1 = eq_var_map.get(v1, ComponentSet([v1]))
                set2 = eq_var_map.get(v2, ComponentSet([v2]))
                relevant_vars.update([v1, v2])
                set1.update(set2)  # set1 is now the union
                for v in set1:
                    eq_var_map[v] = set1

    return eq_var_map, relevant_vars


def _detect_fixed_variables(m):
    """Detect fixed variables due to constraints of form var = const."""
    new_fixed_vars = ComponentSet()
    for constr in m.component_data_objects(ctype=Constraint,
                                           active=True,
                                           descend_into=True):
        if constr.equality and constr.body.polynomial_degree() == 1:
            repn = generate_canonical_repn(constr.body)
            if len(repn.variables) == 1 and repn.linear[0]:
                var = repn.variables[0]
                coef = float(repn.linear[0])
                const = repn.constant if repn.constant is not None else 0
                var_val = (value(constr.lower) - value(const)) / coef
                var.fix(var_val)
                new_fixed_vars.add(var)
    return new_fixed_vars


class FixedVarPropagator(IsomorphicTransformation):
    """Propagates variable fixing for equalities of type x = y.

    If x is fixed and y is not fixed, then this transformation will fix y to
    the value of x.

    This transformation can also be performed as a temporary transformation,
    whereby the transformed variables are saved and can be later unfixed.

    """

    alias('contrib.propagate_fixed_vars',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def __init__(self):
        """Initialize the transformation."""
        super(FixedVarPropagator, self).__init__()

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
        if tmp and not hasattr(instance, '_tmp_propagate_fixed'):
            instance._tmp_propagate_fixed = ComponentSet()
        eq_var_map, relevant_vars = _build_equality_set(instance)
        #: ComponentSet: The set of all fixed variables
        fixed_vars = ComponentSet((v for v in relevant_vars if v.fixed))
        newly_fixed = _detect_fixed_variables(instance)
        if tmp:
            instance._tmp_propagate_fixed.update(newly_fixed)
        fixed_vars.update(newly_fixed)
        processed = ComponentSet()
        # Go through each fixed variable to propagate the 'fixed' status to all
        # equality-linked variabes.
        for v1 in fixed_vars:
            # If we have already processed the variable, skip it.
            if v1 in processed:
                continue

            eq_set = eq_var_map.get(v1, ComponentSet([v1]))
            for v2 in eq_set:
                if (v2.fixed and value(v1) != value(v2)):
                    raise ValueError(
                        'Variables {} and {} have conflicting fixed '
                        'values of {} and {}, but are linked by '
                        'equality constraints.'
                        .format(v1.name,
                                v2.name,
                                value(v1),
                                value(v2)))
                elif not v2.fixed:
                    v2.fix(value(v1))
                    if tmp:
                        instance._tmp_propagate_fixed.add(v2)
            # Add all variables in the equality set to the set of processed
            # variables.
            processed.update(eq_set)

    def revert(self, instance):
        """Revert variables fixed by the transformation."""
        for var in instance._tmp_propagate_fixed:
            var.unfix()
        del instance._tmp_propagate_fixed


class VarBoundPropagator(IsomorphicTransformation):
    """Propagates variable bounds for equalities of type x = y.

    If x has a tighter bound then y, then this transformation will adjust the
    bounds on y to match those of x.

    """

    alias('contrib.propagate_eq_var_bounds',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def __init__(self):
        """Initialize the transformation."""
        super(VarBoundPropagator, self).__init__()

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
        if tmp and not hasattr(instance, '_tmp_propagate_original_bounds'):
            instance._tmp_propagate_original_bounds = Suffix(
                direction=Suffix.LOCAL)
        eq_var_map, relevant_vars = _build_equality_set(instance)
        processed = ComponentSet()
        # Go through each variable in an equality set to propagate the variable
        # bounds to all equality-linked variables.
        for var in relevant_vars:
            # If we have already processed the variable, skip it.
            if var in processed:
                continue

            var_equality_set = eq_var_map.get(var, ComponentSet([var]))

            #: variable lower bounds in the equality set
            lbs = [v.lb for v in var_equality_set if v.has_lb()]
            max_lb = max(lbs) if len(lbs) > 0 else None
            #: variable upper bounds in the equality set
            ubs = [v.ub for v in var_equality_set if v.has_ub()]
            min_ub = min(ubs) if len(ubs) > 0 else None

            # Check  for error due to bound cross-over
            if max_lb is not None and min_ub is not None and max_lb > min_ub:
                # the lower bound is above the upper bound. Raise a ValueError.
                # get variable with the highest lower bound
                v1 = next(v for v in var_equality_set if v.lb == max_lb)
                # get variable with the lowest upper bound
                v2 = next(v for v in var_equality_set if v.ub == min_ub)
                raise ValueError(
                    'Variable {} has a lower bound {} '
                    ' > the upper bound {} of variable {}, '
                    'but they are linked by equality constraints.'
                    .format(v1.name, value(v1.lb), value(v2.ub), v2.name))

            for v in var_equality_set:
                if tmp:
                    # TODO warn if overwriting
                    instance._tmp_propagate_original_bounds[v] = (
                        v.lb, v.ub)
                v.setlb(max_lb)
                v.setub(min_ub)

            processed.update(var_equality_set)

    def revert(self, instance):
        """Revert variable bounds."""
        for v in instance._tmp_propagate_original_bounds:
            old_LB, old_UB = instance._tmp_propagate_original_bounds[v]
            v.setlb(old_LB)
            v.setub(old_UB)
        del instance._tmp_propagate_original_bounds
