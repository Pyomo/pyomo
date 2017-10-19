from pyomo.core.base.constraint import Constraint
from pyomo.core.kernel.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn.canonical_repn import generate_canonical_repn
from pyomo.util.plugin import alias

import textwrap

class ConstraintBoundTightener(IsomorphicTransformation):
    """Tightens bound on constraint, as well as each variable"""


    alias('core.bound_tightener',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))
	def __init__(self):
		super(ConstraintBoundTightener, self).__init__()

	def _apply_to(self,instance):

		for constr in instance.component_data_objects(ctype = Constraint,
														active = True,
														descend_into=True):
			if not constr.body.polynomial_degree() == 1:
				continue

            #tighten the constraint bound as much as possible
			repn = generate_canonical_repn(constr.body)
			UB = repn.constant
			LB = UB
			for i, coef in enumerate(repn.linear):
                #TODO:handle case when var has no bound
				UB = UB + coef * value(repn.variables[i].ub)
				LB = LB + coef * value(repn.variables[i].lb)
			if(constr.has_ub() and value(constr.upper) > UB):
				value(constr.upper) = UB
			if(constr.has_lb() and value(constr.lower) < LB):
				value(constr.lower) = LB

            #tighten the coefficient bounds as much as possible
            UB = None
            LB = None
            for i, coefi in enumerate(repn.linear):
                if(constr.has_ub()):
                    UB = value(constr.upper)
                if(constr.has_lb()):
                    LB = value(constr.lower)
                for j,coefj in enumerate(repn.linear):
                    if not i = j:
                        if UB and repn.variables[j].lb:
                            UB = UB - coefj*repn.variables[j].lb
                        if LB and repn.variables[j].ub
                            LB = LB - coefj*repn.variables[j].ub
                repn.variables[i].lb = LB/coefi
                repn.variables[i].ub = UB/coefi
                if repn.variables[i].value < LB/coefi:
                    repn.variables[i].value = repn.variables[i].lb
                if repn.variables[i].value > UB/coefi:
                    repn.variables[i].value = repn.variables[i].ub
