# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________


from typing import Any, Sequence, Optional, Mapping, NoReturn

from pyomo.core.base import Var
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData
from pyomo.common.collections import ComponentMap
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn.plugins.gams_writer_v2 import GAMSWriterInfo
from pyomo.contrib.solver.common.solution_loader import SolutionLoader
from pyomo.contrib.solver.common.util import (
    NoDualsError,
    NoSolutionError,
    NoReducedCostsError,
)


class GMSSolutionLoader(SolutionLoader):
    """
    Loader for solvers that create .gms files (e.g., gams)
    """

    def __init__(
        self,
        pyomo_model,
        gdx_data: dict[str, tuple[float, float]],
        gms_info: GAMSWriterInfo,
    ) -> None:
        self._gdx_data = gdx_data
        self._gms_info = gms_info
        self._pyomo_model = pyomo_model

    def get_number_of_solutions(self) -> int:
        if self._gms_info is None:
            return 0
        return 1

    def load_vars(self, vars_to_load: Optional[Sequence[VarData]] = None) -> None:
        if self._gms_info is None:
            raise NoSolutionError()
        if self._gdx_data is None:
            assert len(self._gms_info.var_symbol_map.bySymbol) == 0
        else:
            for sym, obj in self._gms_info.var_symbol_map.bySymbol.items():
                level = self._gdx_data[sym][0]
                obj.set_value(level, skip_validation=True)

        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        if self._gms_info is None:
            raise NoSolutionError()
        val_map = {}
        if self._gdx_data is None:
            assert len(self._gms_info.var_symbol_map.bySymbol) == 0
        else:
            for sym, obj in self._gms_info.var_symbol_map.bySymbol.items():
                val_map[id(obj)] = self._gdx_data[sym][0]

        res = ComponentMap()
        if vars_to_load is None:
            vars_to_load = self._gms_info.var_symbol_map.bySymbol.values()

        for obj in vars_to_load:
            res[obj] = val_map[id(obj)]

        return res

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> dict[ConstraintData, float]:
        if self._gms_info is None:
            raise NoDualsError()
        if self._gdx_data is None:
            raise NoDualsError()

        con_map = {}
        for sym, obj in self._gms_info.con_symbol_map.bySymbol.items():
            con_map[id(obj)] = self._gdx_data[sym][1]

        for sym, obj in self._gms_info.con_symbol_map.aliases.items():
            if self._gdx_data[sym][1] != 0:
                con_map[id(obj)] = self._gdx_data[sym][1]

        res = ComponentMap()
        if cons_to_load is None:
            cons_to_load = self._gms_info.con_symbol_map.bySymbol.values()

        for obj in cons_to_load:
            res[obj] = con_map[id(obj)]

        return res

    def get_reduced_costs(self, vars_to_load=None):
        if self._gms_info is None:
            raise NoReducedCostsError()
        if self._gdx_data is None:
            raise NoReducedCostsError()

        var_map = {}
        for sym, obj in self._gms_info.var_symbol_map.bySymbol.items():
            var_map[id(obj)] = self._gdx_data[sym][1]

        res = ComponentMap()
        if vars_to_load is None:
            vars_to_load = self._gms_info.var_symbol_map.bySymbol.values()

        for obj in vars_to_load:
            res[obj] = var_map[id(obj)]

        return res
