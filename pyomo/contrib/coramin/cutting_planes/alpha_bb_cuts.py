from pyomo.core.base.block import _BlockData
from .base import CutGenerator
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.numeric_expr import NumericExpression, NumericValue
from pyomo.core.expr.visitor import identify_variables
from typing import List, Optional, Union
from pyomo.contrib.coramin.relaxations.hessian import Hessian
from pyomo.contrib.coramin.utils.coramin_enums import EigenValueBounder
from pyomo.contrib.appsi.base import Solver
from pyomo.core.expr.visitor import value
from pyomo.core.expr.relational_expr import RelationalExpression
from pyomo.core.expr.taylor_series import taylor_series_expansion
import pybnb


class AlphaBBCutGenerator(CutGenerator):
    def __init__(
        self,
        lhs: Union[float, int, NumericValue],
        rhs: NumericExpression,
        eigenvalue_opt: Optional[Solver] = None,
        method: EigenValueBounder = EigenValueBounder.GershgorinWithSimplification,
        feasibility_tol: float = 1e-6,
    ) -> None:
        self.lhs = lhs
        self.rhs = rhs
        self.xlist: List[_GeneralVarData] = list(
            identify_variables(rhs, include_fixed=False)
        )
        self.hessian = Hessian(expr=rhs, opt=eigenvalue_opt, method=method)
        self.feasibility_tol = feasibility_tol
        self._proven_convex = dict()

    def _most_recent_ancestor(self, node: pybnb.Node):
        res = None
        while res is None:
            p = node.state.parent
            if p is None:
                break
            if p in self._proven_convex:
                res = p
                break
            node = p
        return res

    def generate(self, node: Optional[pybnb.Node]) -> Optional[RelationalExpression]:
        try:
            lhs_val = value(self.lhs)
        except ValueError:
            return None
        if lhs_val + self.feasibility_tol >= value(self.rhs):
            return None

        if node is None:
            mra = None
        else:
            mra = self._most_recent_ancestor(node)
        if mra in self._proven_convex and self._proven_convex[mra]:
            alpha_bb_rhs = self.rhs
        else:
            alpha = max(0, -0.5 * self.hessian.get_minimum_eigenvalue())
            if alpha == 0:
                self._proven_convex[node] = True
                alpha_bb_rhs = self.rhs
            else:
                self._proven_convex[node] = False
                alpha_sum = 0
                for ndx, v in enumerate(self.xlist):
                    lb, ub = v.bounds
                    alpha_sum += (v - lb) * (v - ub)
                alpha_bb_rhs = self.rhs + alpha * alpha_sum
                if lhs_val + self.feasibility_tol >= value(alpha_bb_rhs):
                    return None

        return self.lhs >= taylor_series_expansion(alpha_bb_rhs)
