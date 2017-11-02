from pyomo.core.base.constraint import Constraint
from pyomo.core.kernel.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn.canonical_repn import generate_canonical_repn
from pyomo.util.plugin import alias

import textwrap

class BoundsTightener(IsomorphicTransformation):
    """Tightens bound on constraint, by iterating through each variable and
    tightening the constraint bounds accordingly"""


    alias('core.bounds_tightener',doc=textwrap.fill(textwrap.dedent(__doc__.strip())))
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
            for i, coef in enumerate(repn.linear):
                #TODO: handle unbounded vars
                #TODO: ROunding issues

                if coef >= 0:
                    if repn.variables[i].ub and repn.variables[i].lb:
                        UB = UB + coef * value(repn.variables[i].ub)
                        LB = LB + coef * value(repn.variables[i].lb)
                    else:
                        if not repn.variables[i].ub:
                            UB = float('Inf')
                        if not repn.variables[i].lb:
                            LB = -float('Inf')

                else:
                    if repn.variables[i].ub and repn.variables[i].lb:
                        UB = UB + coef * value(repn.variables[i].lb)
                        LB = LB + coef * value(repn.variables[i].ub)
                    else:
                        if not repn.variables[i].ub:
                            LB = -float('Inf')
                        if not repn.variables[i].lb:
                            UB = float('Inf')

            if(constr.has_ub() and value(constr.upper) > UB) or (not constr.has_ub()):
                constr._upper = float(UB)
            if(constr.has_lb() and value(constr.lower) < LB) or (not constr.has_lb()):
                constr._lower = float(LB)
            if UB < LB:
                print "Infeasible variable bounds"
