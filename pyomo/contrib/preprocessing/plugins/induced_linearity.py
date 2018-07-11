"""Transformation to reformulate nonlinear models with linearity induced from
discrete variables.

Ref: Grossmann, IE; Voudouris, VT; Ghattas, O. Mixed integer linear
reformulations for some nonlinear discrete design optimization problems.

"""

from __future__ import division

import textwrap
from math import fabs

from pyomo.core.base import Block, Constraint, VarList, Objective
from pyomo.core.expr.current import ExpressionReplacementVisitor
from pyomo.core.expr.numvalue import value
from pyomo.core.kernel import ComponentMap, ComponentSet
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn import generate_standard_repn
from pyomo.common.plugin import alias
from pyomo.gdp import Disjunct
import logging

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
        constraint_bound_tolerance = 1E-6
        effectively_discrete_vars = ComponentSet(
            _effectively_discrete_vars(model, constraint_bound_tolerance))
        # TODO will need to go through this for each disjunct, since it does
        # not (should not) descend into Disjuncts.

        # Collect find bilinear expressions that can be reformulated using
        # knowledge of effectively discrete variables
        bilinear_map = _bilinear_expressions(model)

        # Relevant constraints are those with bilinear terms that involve
        # effectively_discrete_vars
        processed_pairs = ComponentSet()
        for v1, discrete_constr in effectively_discrete_vars:
            v1_pairs = bilinear_map.get(v1, ())
            for v2, bilinear_constrs in v1_pairs.items():
                if (v1, v2) in processed_pairs:
                    continue
                _process_bilinear_constraints(
                    v1, v2, discrete_constr, bilinear_constrs)
                processed_pairs.add((v2, v1))
                processed_pairs.add((v1, v2))  # TODO is this necessary?

        # Reformulate the bilinear terms
        pass


def _process_bilinear_constraints(v1, v2, discrete_constr, bilinear_constrs):
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
        if True:
            _reformulate_case_2(v1, v2, discrete_constr, bilinear_constr)
    pass


def _reformulate_case_1(v1, v2, discrete_constr, bilinear_constr):
    raise NotImplementedError()


def _reformulate_case_2(v1, v2, discrete_constr, bilinear_constr):
    pass


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


def _detect_effectively_discrete_vars(block, equality_tolerance):
    """Detect effectively discrete variables and their set of valid values.

    These continuous variables are the sum of discrete variables.

    """
    # Map of effectively_discrete var --> set(possible values)
    effectively_discrete = ComponentMap()

    # Find SOS sets
    for constr in block.component_data_objects(Constraint, active=True):
        pass

    for constr in block.component_data_objects(Constraint, active=True):
        if constr.lower is None or constr.upper is None:
            continue  # skip inequality constraints
        if fabs(value(constr.lower) - value(constr.upper)
                ) > equality_tolerance:
            continue  # not equality constriant. Skip.
        if constr.body.polynomial_degree() not in (1, 0):
            continue  # skip nonlinear expressions
        repn = generate_standard_repn(constr.body)
        constant_term = NoneToZero(repn.constant) - NoneToZero(constr.lower)
        if len(repn.linear_vars) < 2:
            # TODO should this be < 2 or < 1?
            # TODO we should make sure that trivial equality relations are
            # preprocessed before this, or we will end up reformulating
            # expressions that we do not need to here.
            continue
        non_discrete_vars = list(v for v in repn.linear_vars
                                 if v.is_continuous())
        discrete_vars = list(v for v in repn.linear_vars
                             if v.is_binary() or v.is_integer())
        if len(non_discrete_vars) == 1:
            # We know that this is an effectively discrete continuous
            # variable. Add it to our identified variable list.
            yield non_discrete_vars[0], constr
        # TODO we should eventually also look at cases where all other
        # non_discrete_vars are effectively_discrete_vars


def NoneToZero(val):
    return 0 if val is None else val


def _effectively_discrete_vars(block, constraint_bound_tolerance):
    """Yield variables that are effectively discrete, with relevant constraint.

    These continuous variables are the sum of discrete variables. The
    constraint is the one which links the variables together.

    """
    for constr in block.component_data_objects(
            Constraint, active=True, descend_into=True):
        if constr.lower is None or constr.upper is None:
            continue  # skip inequality constraints
        if fabs(value(constr.lower) - value(constr.upper)
                ) > constraint_bound_tolerance:
            continue  # not equality constriant. Skip.
        if constr.body.polynomial_degree() not in (1, 0):
            continue  # skip nonlinear expressions
        repn = generate_standard_repn(constr.body)
        if len(repn.linear_vars) < 2:
            # TODO we should make sure that trivial equality relations are
            # preprocessed before this, or we will end up reformulating
            # expressions that we do not need to here.
            continue
        non_discrete_vars = list(v for v in repn.linear_vars
                                 if v.is_continuous())
        if len(non_discrete_vars) == 1:
            # We know that this is an effectively discrete continuous
            # variable. Add it to our identified variable list.
            yield non_discrete_vars[0], constr
        # TODO we should eventually also look at cases where all other
        # non_discrete_vars are effectively_discrete_vars
