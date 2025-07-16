#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.appsi.base import PersistentBase
from pyomo.common.config import (
    ConfigDict,
    ConfigValue,
    NonNegativeFloat,
    NonNegativeInt,
)
from .cmodel import cmodel, cmodel_available
from typing import List, Optional
from pyomo.core.base.var import VarData
from pyomo.core.base.param import ParamData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.sos import SOSConstraintData
from pyomo.core.base.objective import ObjectiveData, minimize, maximize
from pyomo.core.base.block import BlockData
from pyomo.core.base import SymbolMap, TextLabeler
from pyomo.common.errors import InfeasibleConstraintException


class IntervalConfig(ConfigDict):
    """
    Configuration options for the FBBT IntervalTightener

    Attributes
    ----------
    feasibility_tol: float
    integer_tol: float
    improvement_tol: float
    max_iter: int
    """

    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super(IntervalConfig, self).__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.feasibility_tol: float = self.declare(
            'feasibility_tol', ConfigValue(domain=NonNegativeFloat, default=1e-8)
        )
        self.integer_tol: float = self.declare(
            'integer_tol', ConfigValue(domain=NonNegativeFloat, default=1e-5)
        )
        self.improvement_tol: float = self.declare(
            'improvement_tol', ConfigValue(domain=NonNegativeFloat, default=1e-4)
        )
        self.max_iter: int = self.declare(
            'max_iter', ConfigValue(domain=NonNegativeInt, default=10)
        )
        self.deactivate_satisfied_constraints: bool = self.declare(
            'deactivate_satisfied_constraints', ConfigValue(domain=bool, default=False)
        )


class IntervalTightener(PersistentBase):
    def __init__(self):
        super(IntervalTightener, self).__init__()
        self._config = IntervalConfig()
        self._cmodel = None
        self._var_map = dict()
        self._con_map = dict()
        self._param_map = dict()
        self._rvar_map = dict()
        self._rcon_map = dict()
        self._pyomo_expr_types = cmodel.PyomoExprTypes()
        self._symbolic_solver_labels: bool = False
        self._symbol_map = SymbolMap()
        self._var_labeler = None
        self._con_labeler = None
        self._param_labeler = None
        self._obj_labeler = None
        self._objective = None

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, val: IntervalConfig):
        self._config = val

    def set_instance(self, model, symbolic_solver_labels: Optional[bool] = None):
        saved_config = self.config
        saved_update_config = self.update_config

        self.__init__()
        self.config = saved_config
        self.update_config = saved_update_config
        self._expr_types = cmodel.PyomoExprTypes()

        if symbolic_solver_labels is not None:
            self._symbolic_solver_labels = symbolic_solver_labels
        if self._symbolic_solver_labels:
            self._var_labeler = TextLabeler()
            self._con_labeler = TextLabeler()
            self._param_labeler = TextLabeler()
            self._obj_labeler = TextLabeler()

        self._model = model
        self._cmodel = cmodel.FBBTModel()
        self.add_block(model)
        if self._objective is None:
            self.set_objective(None)

    def _add_variables(self, variables: List[VarData]):
        if self._symbolic_solver_labels:
            set_name = True
            symbol_map = self._symbol_map
            labeler = self._var_labeler
        else:
            set_name = False
            symbol_map = None
            labeler = None
        cmodel.process_pyomo_vars(
            self._pyomo_expr_types,
            variables,
            self._var_map,
            self._param_map,
            self._vars,
            self._rvar_map,
            set_name,
            symbol_map,
            labeler,
            False,
        )

    def _add_params(self, params: List[ParamData]):
        cparams = cmodel.create_params(len(params))
        for ndx, p in enumerate(params):
            cp = cparams[ndx]
            cp.value = p.value
            self._param_map[id(p)] = cp
        if self._symbolic_solver_labels:
            for ndx, p in enumerate(params):
                cp = cparams[ndx]
                cp.name = self._symbol_map.getSymbol(p, self._param_labeler)

    def _add_constraints(self, cons: List[ConstraintData]):
        cmodel.process_fbbt_constraints(
            self._cmodel,
            self._pyomo_expr_types,
            cons,
            self._var_map,
            self._param_map,
            self._active_constraints,
            self._con_map,
            self._rcon_map,
        )
        if self._symbolic_solver_labels:
            for c, cc in self._con_map.items():
                cc.name = self._symbol_map.getSymbol(c, self._con_labeler)

    def _add_sos_constraints(self, cons: List[SOSConstraintData]):
        if len(cons) != 0:
            raise NotImplementedError(
                'IntervalTightener does not support SOS constraints'
            )

    def _remove_constraints(self, cons: List[ConstraintData]):
        if self._symbolic_solver_labels:
            for c in cons:
                self._symbol_map.removeSymbol(c)
        for c in cons:
            cc = self._con_map.pop(c)
            self._cmodel.remove_constraint(cc)
            del self._rcon_map[cc]

    def _remove_sos_constraints(self, cons: List[SOSConstraintData]):
        if len(cons) != 0:
            raise NotImplementedError(
                'IntervalTightener does not support SOS constraints'
            )

    def _remove_variables(self, variables: List[VarData]):
        if self._symbolic_solver_labels:
            for v in variables:
                self._symbol_map.removeSymbol(v)
        for v in variables:
            cvar = self._var_map.pop(id(v))
            del self._rvar_map[cvar]

    def _remove_params(self, params: List[ParamData]):
        if self._symbolic_solver_labels:
            for p in params:
                self._symbol_map.removeSymbol(p)
        for p in params:
            del self._param_map[id(p)]

    def _update_variables(self, variables: List[VarData]):
        cmodel.process_pyomo_vars(
            self._pyomo_expr_types,
            variables,
            self._var_map,
            self._param_map,
            self._vars,
            self._rvar_map,
            False,
            None,
            None,
            True,
        )

    def update_params(self):
        for p_id, p in self._params.items():
            cp = self._param_map[p_id]
            cp.value = p.value

    def set_objective(self, obj: ObjectiveData):
        if self._symbolic_solver_labels:
            if self._objective is not None:
                self._symbol_map.removeSymbol(self._objective)
        super().set_objective(obj)

    def _set_objective(self, obj: ObjectiveData):
        if obj is None:
            ce = cmodel.Constant(0)
            sense = 0
        else:
            ce = cmodel.appsi_expr_from_pyomo_expr(
                obj.expr, self._var_map, self._param_map, self._pyomo_expr_types
            )
            if obj.sense is minimize:
                sense = 0
            else:
                sense = 1
        cobj = cmodel.FBBTObjective(ce)
        cobj.sense = sense
        self._cmodel.objective = cobj
        self._objective = obj
        if self._symbolic_solver_labels and obj is not None:
            cobj.name = self._symbol_map.getSymbol(obj, self._obj_labeler)

    def _update_pyomo_var_bounds(self):
        for cv, v in self._rvar_map.items():
            cv_lb = cv.get_lb()
            cv_ub = cv.get_ub()
            if -cmodel.inf < cv_lb:
                v.setlb(cv_lb)
                v_id = id(v)
                _v, _lb, _ub, _fixed, _domain, _value = self._vars[v_id]
                self._vars[v_id] = (_v, cv_lb, _ub, _fixed, _domain, _value)
            if cv_ub < cmodel.inf:
                v.setub(cv_ub)
                v_id = id(v)
                _v, _lb, _ub, _fixed, _domain, _value = self._vars[v_id]
                self._vars[v_id] = (_v, _lb, cv_ub, _fixed, _domain, _value)

    def _deactivate_satisfied_cons(self):
        cons_to_deactivate = list()
        if self.config.deactivate_satisfied_constraints:
            for c, cc in self._con_map.items():
                if not cc.active:
                    cons_to_deactivate.append(c)
        self.remove_constraints(cons_to_deactivate)
        for c in cons_to_deactivate:
            c.deactivate()

    def perform_fbbt(
        self, model: BlockData, symbolic_solver_labels: Optional[bool] = None
    ):
        if model is not self._model:
            self.set_instance(model, symbolic_solver_labels=symbolic_solver_labels)
        else:
            if (
                symbolic_solver_labels is not None
                and symbolic_solver_labels != self._symbolic_solver_labels
            ):
                raise RuntimeError(
                    'symbolic_solver_labels can only be changed through the set_instance method. '
                    'Please either use set_instance or create a new instance of IntervalTightener.'
                )
            self.update()
        try:
            n_iter = self._cmodel.perform_fbbt(
                self.config.feasibility_tol,
                self.config.integer_tol,
                self.config.improvement_tol,
                self.config.max_iter,
                self.config.deactivate_satisfied_constraints,
            )
        finally:
            # we want to make sure the pyomo model and cmodel stay in sync
            # even if an exception is raised and caught
            self._update_pyomo_var_bounds()
            self._deactivate_satisfied_cons()
        return n_iter

    def perform_fbbt_with_seed(self, model: BlockData, seed_var: VarData):
        if model is not self._model:
            self.set_instance(model)
        else:
            self.update()
        try:
            n_iter = self._cmodel.perform_fbbt_with_seed(
                self._var_map[id(seed_var)],
                self.config.feasibility_tol,
                self.config.integer_tol,
                self.config.improvement_tol,
                self.config.max_iter,
                self.config.deactivate_satisfied_constraints,
            )
        finally:
            # we want to make sure the pyomo model and cmodel stay in sync
            # even if an exception is raised and caught
            self._update_pyomo_var_bounds()
            self._deactivate_satisfied_cons()
        return n_iter
