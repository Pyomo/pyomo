# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from typing import Any

from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.var import VarData

from pyomo.contrib.solver.common.config import BranchAndBoundConfig
from pyomo.contrib.solver.common.util import IncompatibleModelError
from .xpress_base import (
    XpressSolverMixin,
    XpressSolutionLoaderBase,
    EntityMaps,
    XpressExpressionWalker,
    _OBJ_SENSE_MAP,
    _register_variable,
    _register_pool_collector,
    xp,
)


class XpressDirectSolutionLoader(XpressSolutionLoaderBase):
    """Solution loader for the non-persistent XpressDirect solver."""


class XpressDirectConfig(BranchAndBoundConfig):
    """Configuration for XpressDirect."""

    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        self.pool_solutions: int = self.declare(
            'pool_solutions',
            ConfigValue(
                default=0,
                domain=NonNegativeInt,
                description=(
                    'MIP solution pool size (0 = disabled). '
                    'N > 0: keep a rolling window of the last N solutions found.'
                ),
            ),
        )


class XpressDirect(XpressSolverMixin):
    CONFIG = XpressDirectConfig()

    def _create_xpress_model(
        self, model: BlockData, config: BranchAndBoundConfig, timer: HierarchicalTimer
    ) -> tuple[Any, XpressDirectSolutionLoader, bool]:
        timer.start('compile_model')
        pyo_objs = list(model.component_data_objects(Objective, active=True))
        pyo_cons = list(model.component_data_objects(Constraint, active=True))
        pyo_sos = list(model.component_data_objects(SOSConstraint, active=True))
        if len(pyo_objs) > 1:
            raise IncompatibleModelError(
                f'Xpress supports at most one objective (received {len(pyo_objs)}).'
            )
        timer.stop('compile_model')

        timer.start('load_model')
        xp_prob = xp.problem()
        use_names = config.symbolic_solver_labels
        var_map: dict[int, Any] = {}
        pyo_vars: list[VarData] = []
        walker = XpressExpressionWalker(
            var_map, prob=xp_prob, use_names=use_names, registered_vars=pyo_vars
        )
        xp_cons = self._add_cons_impl(xp_prob, pyo_cons, walker, use_names)
        cons_map = dict(zip(pyo_cons, xp_cons))
        if len(pyo_objs) > 0:
            obj_result = walker.walk_expression(pyo_objs[0].expr)
            sense = _OBJ_SENSE_MAP[pyo_objs[0].sense]
            xp_prob.setObjective(obj_result, sense=sense)
        # SOS vars may not appear in constraints/objective -- register if missing.
        for sos_con in pyo_sos:
            for var in sos_con.variables:
                _register_variable(walker, var)
        xp_sos = self._add_sos_impl(xp_prob, pyo_sos, var_map, use_names)
        sos_map = dict(zip(pyo_sos, xp_sos))
        timer.stop('load_model')

        maps = EntityMaps(vars=var_map, cons=cons_map, sos=sos_map)
        if xp_prob.attributes.mipents > 0 or xp_prob.attributes.sets > 0:
            entind = [i for i, var in enumerate(pyo_vars) if not var.is_continuous()]
            self._warmstart(xp_prob, pyo_vars, entind)

        pool: list = []
        if config.pool_solutions > 0:
            pool = _register_pool_collector(xp_prob, config.pool_solutions)

        return (
            xp_prob,
            XpressDirectSolutionLoader(
                xp_prob, model, pyo_vars, maps, pool_solutions=pool
            ),
            len(pyo_objs) > 0,
        )
