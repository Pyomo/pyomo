from pyomo.core.base.block import _BlockData
from pyomo.repn.standard_repn import generate_standard_repn, StandardRepn
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.base.block import _BlockData
from pyomo.contrib.coramin.cutting_planes.alpha_bb_cuts import AlphaBBCutGenerator
from pyomo.contrib.coramin.cutting_planes.base import CutGenerator
from pyomo.contrib.coramin.utils.coramin_enums import EigenValueBounder
from typing import List
from pyomo.common.config import ConfigDict, ConfigValue


class AlphaBBConfig(ConfigDict):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(description, doc, implicit, implicit_domain, visibility)
        self.max_num_vars: int = self.declare("max_num_vars", ConfigValue(default=4))
        self.method: EigenValueBounder = self.declare(
            "method",
            ConfigValue(default=EigenValueBounder.GershgorinWithSimplification),
        )


def find_cut_generators(m: _BlockData, config: AlphaBBConfig) -> List[CutGenerator]:
    cut_generators = list()
    for c in m.nonlinear.cons.values():
        repn: StandardRepn = generate_standard_repn(
            c.body, quadratic=False, compute_values=True
        )
        if repn.nonlinear_expr is None:
            continue
        if len(repn.nonlinear_vars) > config.max_num_vars:
            continue

        if len(repn.linear_coefs) > 0:
            lhs = LinearExpression(
                constant=repn.constant,
                linear_coefs=repn.linear_coefs,
                linear_vars=repn.linear_vars,
            )
        else:
            lhs = repn.constant

        # alpha bb convention is lhs >= rhs
        if c.lb is not None:
            cg = AlphaBBCutGenerator(
                lhs=lhs - c.lb,
                rhs=-repn.nonlinear_expr,
                eigenvalue_opt=None,
                method=config.method,
            )
            cut_generators.append(cg)
        if c.ub is not None:
            cg = AlphaBBCutGenerator(
                lhs=c.ub - lhs,
                rhs=repn.nonlinear_expr,
                eigenvalue_opt=None,
                method=config.method,
            )
            cut_generators.append(cg)

    return cut_generators
