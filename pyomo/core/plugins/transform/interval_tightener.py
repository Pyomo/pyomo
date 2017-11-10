from pyomo.core.base.constraint import Constraint
from pyomo.core.kernel.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn.canonical_repn import generate_canonical_repn
from pyomo.util.plugin import alias
import logging
import textwrap
logger=logging.getLogger('pyomo.core')
class BoundsTightener(IsomorphicTransformation):
    """Tightens bound on constraint, by iterating through each variable and
    tightening the constraint bounds accordingly"""


    alias('core.interval_tightener',doc=textwrap.fill(textwrap.dedent(__doc__.strip())))
    def __init__(self):
        super(BoundsTightener, self).__init__()

    def _apply_to(self,instance):

        for constr in instance.component_data_objects(ctype = Constraint,
                                                        active = True,
                                                        descend_into=True):
            if not constr.body.polynomial_degree() == 1:
                continue

            #tighten the constraint bound as much as possible
            repn = generate_canonical_repn(constr.body)
            UB = 0
            if repn.constant:
                UB = -1 * repn.constant
            LB = UB
            #loop through each coefficent and variable pair
            for i, coef in enumerate(repn.linear):
                #TODO: ROunding issues
                #Calculate bounds using interval arithmetic
                if coef >= 0:
                    if repn.variables[i].ub:
                        UB = UB + coef * value(repn.variables[i].ub)
                    else:
                        UB = float('Inf')
                    if repn.variables[i].lb:
                        LB = LB + coef * value(repn.variables[i].lb)
                    else:
                        LB = -float('Inf')
                else:
                    if repn.variables[i].ub:
                        UB = UB + coef * value(repn.variables[i].lb)
                    else:
                        LB = -float('Inf')
                    if repn.variables[i].lb:
                        LB = LB + coef * value(repn.variables[i].ub)
                    else:
                        UB = float('Inf')
            #if bound is tighter, replace bound
            if(constr.has_ub() and value(constr.upper) > UB) or (not constr.has_ub()):
                if UB == float('Inf'):
                    constr._upper = None
                else:
                    constr._upper = float(UB)
            if(constr.has_lb() and value(constr.lower) < LB) or (not constr.has_lb()):
                if LB == -float('Inf'):
                    constr._lower = None
                else:
                    constr._lower = float(LB)

            if UB < LB:
                logger.error("Infeasible variable bounds")
