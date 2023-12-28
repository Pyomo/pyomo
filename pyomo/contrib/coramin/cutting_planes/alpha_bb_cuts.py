from pyomo.core.base.block import _BlockData
from .base import CutGenerator
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.numeric_expr import NumericExpression
from pyomo.core.expr.visitor import identify_variables
from typing import List, Optional
from pyomo.contrib.coramin.relaxations.hessian import Hessian
from pyomo.contrib.coramin.utils.coramin_enums import EigenValueBounder
from pyomo.contrib.appsi.base import Solver
from pyomo.core.expr.visitor import value
from pyomo.core.expr.relational_expr import RelationalExpression
from pyomo.core.expr.taylor_series import taylor_series_expansion


class AlphaBBCutGenerator(CutGenerator):
    def __init__(self, lhs: _GeneralVarData, rhs: NumericExpression, eigenvalue_opt: Optional[Solver] = None, method: EigenValueBounder = EigenValueBounder.GershgorinWithSimplification) -> None:
        self.lhs = lhs
        self.rhs = rhs
        self.xlist: List[_GeneralVarData] = list(identify_variables(rhs, include_fixed=False))
        self.hessian = Hessian(expr=rhs, opt=eigenvalue_opt, method=method)

    def generate(self, model: _BlockData, solver: Solver | None = None) -> Optional[RelationalExpression]:
        lhs_val = value(self.lhs)
        if lhs_val >= value(self.rhs):
            return None
        
        alpha = max(0, -0.5 * self.hessian.get_minimum_eigenvalue())
        alpha_sum = 0
        for ndx, v in enumerate(self.xlist):
            lb, ub = v.bounds
            alpha_sum += (v - lb) * (v - ub)
        alpha_bb_rhs = self.rhs + alpha * alpha_sum
        if lhs_val >= value(alpha_bb_rhs):
            return None
        
        return self.lhs >= taylor_series_expansion(alpha_bb_rhs)
