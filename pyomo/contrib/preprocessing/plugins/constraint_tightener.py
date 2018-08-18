import logging
import textwrap

from pyomo.core import Constraint, value, TransformationFactory
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn.standard_repn import generate_standard_repn

logger = logging.getLogger('pyomo.contrib.preprocessing')


@TransformationFactory.register('core.tighten_constraints_from_vars',
          doc="Tightens upper and lower bound on linear constraints.")
class TightenContraintFromVars(IsomorphicTransformation):
    """Tightens upper and lower bound on constraints based on variable bounds.

    Iterates through each variable and tightens the constraint bounds using
    the inferred values from the variable bounds.

    For now, this only operates on linear constraints.

    """

    def _apply_to(self, instance):
        for constr in instance.component_data_objects(
                ctype=Constraint, active=True, descend_into=True):
            repn = generate_standard_repn(constr.body)
            if not repn.is_linear():
                continue

            # tighten the constraint bound as much as possible
            LB = UB = 0
            if repn.constant:
                LB = UB = repn.constant
            # loop through each coefficent and variable pair
            for i, coef in enumerate(repn.linear_coefs):
                # TODO: Rounding issues
                # Calculate bounds using interval arithmetic
                if coef >= 0:
                    if repn.linear_vars[i].has_ub():
                        UB = UB + coef * value(repn.linear_vars[i].ub)
                    else:
                        UB = float('Inf')
                    if repn.linear_vars[i].has_lb():
                        LB = LB + coef * value(repn.linear_vars[i].lb)
                    else:
                        LB = float('-Inf')
                else:
                    # coef is negative, so signs switch
                    if repn.linear_vars[i].has_lb():
                        UB = UB + coef * value(repn.linear_vars[i].lb)
                    else:
                        LB = float('-Inf')
                    if repn.linear_vars[i].has_ub():
                        LB = LB + coef * value(repn.linear_vars[i].ub)
                    else:
                        UB = float('Inf')

            # if inferred bound is tighter, replace bound
            new_ub = min(value(constr.upper), UB) if constr.has_ub() else UB
            new_lb = max(value(constr.lower), LB) if constr.has_lb() else LB
            constr.set_value((new_lb, constr.body, new_ub))

            if UB < LB:
                logger.error(
                    "Infeasible variable bounds: "
                    "Constraint %s has inferred LB %s > UB %s" %
                    (constr.name, new_lb, new_ub))
