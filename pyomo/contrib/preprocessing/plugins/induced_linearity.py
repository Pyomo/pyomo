"""Transformation to reformulate nonlinear models with linearity induced from
discrete variables.

Ref: Grossmann, IE; Voudouris, VT; Ghattas, O. Mixed integer linear
reformulations for some nonlinear discrete design optimization problems.

"""

from __future__ import division

import logging
import textwrap
from math import fabs

from pyomo.common.modeling import unique_component_name
from pyomo.common.plugin import alias
from pyomo.contrib.preprocessing.util import SuppressConstantObjectiveWarning
from pyomo.core import (Binary, Block, Constraint, Objective, Set,
                        TransformationFactory, Var, summation, value)
from pyomo.core.kernel import ComponentMap, ComponentSet
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.gdp import Disjunct
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn

logger = logging.getLogger('pyomo.contrib.preprocessing')


class InducedLinearity(IsomorphicTransformation):
    """Reformulate nonlinear constraints with induced linearity.

    Finds continuous variables v where v = d1 + d2 + d3, where d's are discrete
    variables. These continuous variables may participate nonlinearly in other
    expressions, which may then be induced to be linear.

    The overall algorithm flow can be summarized as:
    1. Detect effectively discrete variables and the constraints that
    imply discreteness.
    2. Determine the set of valid values for each effectively discrete variable
        - NOTE: 1, 2 must incorporate scoping considerations (Disjuncts)
    3. Find nonlinear expressions in which effectively discrete variables
    participate.
    4. Reformulate nonlinear expressions appropriately.

    """

    alias('contrib.induced_linearity',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def _apply_to(self, model):
        """Apply the transformation to the given model."""
        _process_container(model)
        _process_subcontainers(model)


def _process_subcontainers(blk):
    for disj in blk.component_data_objects(
            Disjunct, active=True, descend_into=True):
        _process_container(disj)
        _process_subcontainers(disj)


def _process_container(blk):
    equality_tolerance = 1E-6
    if not hasattr(blk, '_induced_linearity_info'):
        blk._induced_linearity_info = Block()
    eff_discr_vars = detect_effectively_discrete_vars(
        blk, equality_tolerance)
    # TODO will need to go through this for each disjunct, since it does
    # not (should not) descend into Disjuncts.

    # Determine the valid values for the effectively discrete variables
    possible_var_values = determine_valid_values(blk, eff_discr_vars)

    # Collect find bilinear expressions that can be reformulated using
    # knowledge of effectively discrete variables
    bilinear_map = _bilinear_expressions(blk)

    # Relevant constraints are those with bilinear terms that involve
    # effectively_discrete_vars
    processed_pairs = ComponentSet()
    for v1, var_values in possible_var_values.items():
        v1_pairs = bilinear_map.get(v1, ())
        for v2, bilinear_constrs in v1_pairs.items():
            if (v1, v2) in processed_pairs:
                continue
            _process_bilinear_constraints(
                blk, v1, v2, var_values, bilinear_constrs)
            processed_pairs.add((v2, v1))
            # processed_pairs.add((v1, v2))  # TODO is this necessary?


def determine_valid_values(block, discr_var_to_constrs_map):
    """Calculate valid values for each effectively discrete variable.

    We need the set of possible values for the effectively discrete variable in
    order to do the reformulations.

    Right now, we select a naive approach where we look for variables in the
    discreteness-inducing constraints. We then adjust their values and see if
    things are stil feasible. Based on their coefficient values, we can infer a
    set of allowable values for the effectively discrete variable.

    Args:
        block: The model or a disjunct on the model.

    """
    possible_values = ComponentMap()

    for eff_discr_var, constrs in discr_var_to_constrs_map.items():
        # get the superset of possible values by looking through the
        # constraints
        for constr in constrs:
            repn = generate_standard_repn(constr.body)
            var_coef = sum(coef for i, coef in enumerate(repn.linear_coefs)
                           if repn.linear_vars[i] is eff_discr_var)
            const = -(repn.constant - constr.upper) / var_coef
            possible_vals = frozenset((const,))
            for i, var in enumerate(repn.linear_vars):
                if var is eff_discr_var:
                    continue
                coef = -repn.linear_coefs[i] / var_coef
                if var.is_binary():
                    var_values = (0, coef)
                elif var.is_integer():
                    var_values = range(var.lb, var.ub + 1)
                else:
                    raise ValueError(
                        '%s has unacceptable variable domain: %s' %
                        (var.name, var.domain))
                possible_vals = frozenset(
                    (v1 + v2 for v1 in possible_vals for v2 in var_values))
            old_possible_vals = possible_values.get(eff_discr_var, None)
            if old_possible_vals is not None:
                possible_values[eff_discr_var] = old_possible_vals & possible_vals
            else:
                possible_values[eff_discr_var] = possible_vals

    possible_values = prune_possible_values(block, possible_values)

    return possible_values


def prune_possible_values(block_scope, possible_values):
    # Prune the set of possible values by solving a series of feasibility
    # problems
    top_level_scope = block_scope.model()
    top_level_scope._possible_values = possible_values
    top_level_scope._possible_value_vars = list(v for v in possible_values)
    top_level_scope._tmp_block_scope = (block_scope,)
    model = top_level_scope.clone()
    for obj in model.component_data_objects(Objective, active=True):
        obj.deactivate()
    for constr in model.component_data_objects(
            Constraint, active=True, descend_into=(Block, Disjunct)):
        if constr.body.polynomial_degree() not in (1, 0):
            constr.deactivate()
    if block_scope.type() == Disjunct:
        disj = model._tmp_block_scope[0]
        disj.indicator_var.fix(1)
        TransformationFactory('gdp.bigm').apply_to(model)
    model.test_feasible = Constraint()
    model._obj = Objective(expr=1)
    for eff_discr_var, vals in model._possible_values.items():
        val_feasible = {}
        for val in vals:
            model.test_feasible.set_value(eff_discr_var == val)
            with SuppressConstantObjectiveWarning():
                res = SolverFactory('glpk').solve(model)
            if res.solver.termination_condition is tc.infeasible:
                val_feasible[val] = False
        model._possible_values[eff_discr_var] = frozenset(
            v for v in model._possible_values[eff_discr_var]
            if val_feasible.get(v, True))
    for i, var in enumerate(top_level_scope._possible_value_vars):
        possible_values[var] = model._possible_values[model._possible_value_vars[i]]

    return possible_values


def _process_bilinear_constraints(block, v1, v2, var_values, bilinear_constrs):
    # TODO check that the appropriate variable bounds exist.
    if not (v2.has_lb() and v2.has_lb()):
        logger.warning(textwrap.dedent("""\
            Attempting to transform bilinear term {v1} * {v2} using effectively
            discrete variable {v1}, but {v2} is missing a lower or upper bound:
            ({v2lb}, {v2ub}).
            """.format(v1=v1, v2=v2, v2lb=v2.lb, v2ub=v2.ub)).strip())
        return False
    blk = Block()
    unique_name = unique_component_name(
        block, ("%s_%s_bilinear" % (v1.local_name, v2.local_name))
        .translate({ord(c): '' for c in "[]"}))
    block._induced_linearity_info.add_component(unique_name, blk)
    # TODO think about not using floats as indices in a set
    blk.valid_values = Set(initialize=var_values)
    blk.x_active = Var(blk.valid_values, domain=Binary, initialize=1)
    blk.v_increment = Var(
        blk.valid_values, domain=v2.domain,
        bounds=(v2.lb, v2.ub), initialize=v2.value)
    blk.v_defn = Constraint(expr=v2 == summation(blk.v_increment))

    @blk.Constraint(blk.valid_values)
    def v_lb(blk, val):
        return v2.lb * blk.x_active[val] <= blk.v_increment[val]

    @blk.Constraint(blk.valid_values)
    def v_ub(blk, val):
        return blk.v_increment[val] <= v2.ub * blk.x_active[val]
    blk.select_one_value = Constraint(expr=summation(blk.x_active) == 1)
    # Categorize as case 1 or case 2
    for bilinear_constr in bilinear_constrs:
        # repn = generate_standard_repn(bilinear_constr.body)

        # Case 1: no other variables besides bilinear term in constraint. v1
        # (effectively discrete variable) is positive.
        # if (len(repn.quadratic_vars) == 1 and len(repn.linear_vars) == 0
        #         and repn.nonlinear_expr is None):
        #     _reformulate_case_1(v1, v2, discrete_constr, bilinear_constr)

        # NOTE: Case 1 is left unimplemented for now, because it involves some
        # messier logic with respect to how the transformation needs to happen.

        # Case 2: this is everything else, but do we want to have a special
        # case if there are nonlinear expressions involved with the constraint?
        pass
        _reformulate_case_2(blk, v1, v2, bilinear_constr)
    pass


def _reformulate_case_2(blk, v1, v2, bilinear_constr):
    repn = generate_standard_repn(bilinear_constr.body)
    replace_index = next(
        i for i, var_tup in enumerate(repn.quadratic_vars)
        if (var_tup[0] is v1 and var_tup[1] is v2) or
           (var_tup[0] is v2 and var_tup[1] is v1))
    bilinear_constr.set_value((
        bilinear_constr.lower,
        sum(coef * repn.linear_vars[i]
            for i, coef in enumerate(repn.linear_coefs)) +
        repn.quadratic_coefs[replace_index] * sum(
            val * blk.v_increment[val] for val in blk.valid_values) +
        sum(repn.quadratic_coefs[i] * var_tup[0] * var_tup[1]
            for i, var_tup in enumerate(repn.quadratic_vars)
            if not i == replace_index) +
        repn.constant +
        zero_if_None(repn.nonlinear_expr),
        bilinear_constr.upper
    ))


def zero_if_None(val):
    return val if val is not None else 0


def _bilinear_expressions(model):
    # TODO for now, we look for only expressions where the bilinearities are
    # exposed on the root level SumExpression, and thus accessible via
    # generate_standard_repn. This will not detect exp(x*y). We require a
    # factorization transformation to be applied beforehand in order to pick
    # these constraints up.
    pass
    # Bilinear map will be stored in the format:
    # x --> (y --> [constr1, constr2, ...], z --> [constr2, constr3])
    bilinear_map = ComponentMap()
    for constr in model.component_data_objects(
            Constraint, active=True, descend_into=(Block, Disjunct)):
        if constr.body.polynomial_degree() in (1, 0):
            continue  # Skip trivial and linear constraints
        repn = generate_standard_repn(constr.body)
        for pair in repn.quadratic_vars:
            v1, v2 = pair
            v1_pairs = bilinear_map.get(v1, ComponentMap())
            if v2 in v1_pairs:
                # bilinear term has been found before. Simply add constraint to
                # the set associated with the bilinear term.
                v1_pairs[v2].add(constr)
            else:
                # We encounter the bilinear term for the first time.
                bilinear_map[v1] = v1_pairs
                bilinear_map[v2] = bilinear_map.get(v2, ComponentMap())
                constraints_with_bilinear_pair = ComponentSet([constr])
                bilinear_map[v1][v2] = constraints_with_bilinear_pair
                bilinear_map[v2][v1] = constraints_with_bilinear_pair
    return bilinear_map


def detect_effectively_discrete_vars(block, equality_tolerance):
    """Detect effectively discrete variables.

    These continuous variables are the sum of discrete variables.

    """
    # Map of effectively_discrete var --> inducing constraints
    effectively_discrete = ComponentMap()

    for constr in block.component_data_objects(Constraint, active=True):
        if constr.lower is None or constr.upper is None:
            continue  # skip inequality constraints
        if fabs(value(constr.lower) - value(constr.upper)
                ) > equality_tolerance:
            continue  # not equality constriant. Skip.
        if constr.body.polynomial_degree() not in (1, 0):
            continue  # skip nonlinear expressions
        repn = generate_standard_repn(constr.body)
        if len(repn.linear_vars) < 2:
            # TODO should this be < 2 or < 1?
            # TODO we should make sure that trivial equality relations are
            # preprocessed before this, or we will end up reformulating
            # expressions that we do not need to here.
            continue
        non_discrete_vars = list(v for v in repn.linear_vars
                                 if v.is_continuous())
        if len(non_discrete_vars) == 1:
            # We know that this is an effectively discrete continuous
            # variable. Add it to our identified variable list.
            var = non_discrete_vars[0]
            inducing_constraints = effectively_discrete.get(var, [])
            inducing_constraints.append(constr)
            effectively_discrete[var] = inducing_constraints
        # TODO we should eventually also look at cases where all other
        # non_discrete_vars are effectively_discrete_vars

    return effectively_discrete
