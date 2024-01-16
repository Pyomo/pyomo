from typing import List
from pyomo.core.base.param import _ParamData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.block import _BlockData
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.numvalue import value
from pyomo.contrib.appsi.base import PersistentBase
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.kernel.objective import minimize
from .config import WriterConfig
from pyomo.common.collections import OrderedSet
import os
from ..cmodel import cmodel, cmodel_available
from pyomo.repn.plugins.ampl.ampl_ import set_pyomo_amplfunc_env


class NLWriter(PersistentBase):
    def __init__(self, only_child_vars=False):
        super(NLWriter, self).__init__(only_child_vars=only_child_vars)
        self._config = WriterConfig()
        self._writer = None
        self._symbol_map = SymbolMap()
        self._var_labeler = None
        self._con_labeler = None
        self._param_labeler = None
        self._pyomo_var_to_solver_var_map = dict()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_var_to_pyomo_var_map = dict()
        self._solver_con_to_pyomo_con_map = dict()
        self._pyomo_param_to_solver_param_map = dict()
        self._expr_types = None

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, val: WriterConfig):
        self._config = val

    @property
    def symbol_map(self):
        return self._symbol_map

    def set_instance(self, model):
        saved_config = self.config
        saved_update_config = self.update_config
        self.__init__(only_child_vars=self._only_child_vars)
        self.config = saved_config
        self.update_config = saved_update_config
        self._model = model
        self._expr_types = cmodel.PyomoExprTypes()

        if self.config.symbolic_solver_labels:
            self._var_labeler = TextLabeler()
            self._con_labeler = TextLabeler()
            self._param_labeler = TextLabeler()

        self._writer = cmodel.NLWriter()

        self.add_block(model)
        if self._objective is None:
            self.set_objective(None)
        self._set_pyomo_amplfunc_env()

    def _add_variables(self, variables: List[_GeneralVarData]):
        if self.config.symbolic_solver_labels:
            set_name = True
            symbol_map = self._symbol_map
            labeler = self._var_labeler
        else:
            set_name = False
            symbol_map = None
            labeler = None
        cmodel.process_pyomo_vars(
            self._expr_types,
            variables,
            self._pyomo_var_to_solver_var_map,
            self._pyomo_param_to_solver_param_map,
            self._vars,
            self._solver_var_to_pyomo_var_map,
            set_name,
            symbol_map,
            labeler,
            False,
        )

    def _add_params(self, params: List[_ParamData]):
        cparams = cmodel.create_params(len(params))
        for ndx, p in enumerate(params):
            cp = cparams[ndx]
            cp.value = p.value
            self._pyomo_param_to_solver_param_map[id(p)] = cp
        if self.config.symbolic_solver_labels:
            for ndx, p in enumerate(params):
                cp = cparams[ndx]
                cp.name = self._symbol_map.getSymbol(p, self._param_labeler)

    def _add_constraints(self, cons: List[_GeneralConstraintData]):
        cmodel.process_nl_constraints(
            self._writer,
            self._expr_types,
            cons,
            self._pyomo_var_to_solver_var_map,
            self._pyomo_param_to_solver_param_map,
            self._active_constraints,
            self._pyomo_con_to_solver_con_map,
            self._solver_con_to_pyomo_con_map,
        )
        if self.config.symbolic_solver_labels:
            for c, cc in self._pyomo_con_to_solver_con_map.items():
                cc.name = self._symbol_map.getSymbol(c, self._con_labeler)

    def _add_sos_constraints(self, cons: List[_SOSConstraintData]):
        if len(cons) != 0:
            raise NotImplementedError('NL writer does not support SOS constraints')

    def _remove_constraints(self, cons: List[_GeneralConstraintData]):
        if self.config.symbolic_solver_labels:
            for c in cons:
                self._symbol_map.removeSymbol(c)
                self._con_labeler.remove_obj(c)
        for c in cons:
            cc = self._pyomo_con_to_solver_con_map.pop(c)
            self._writer.remove_constraint(cc)
            del self._solver_con_to_pyomo_con_map[cc]

    def _remove_sos_constraints(self, cons: List[_SOSConstraintData]):
        if len(cons) != 0:
            raise NotImplementedError('NL writer does not support SOS constraints')

    def _remove_variables(self, variables: List[_GeneralVarData]):
        if self.config.symbolic_solver_labels:
            for v in variables:
                self._symbol_map.removeSymbol(v)
                self._var_labeler.remove_obj(v)
        for v in variables:
            cvar = self._pyomo_var_to_solver_var_map.pop(id(v))
            del self._solver_var_to_pyomo_var_map[cvar]

    def _remove_params(self, params: List[_ParamData]):
        if self.config.symbolic_solver_labels:
            for p in params:
                self._symbol_map.removeSymbol(p)
                self._param_labeler.remove_obj(p)
        for p in params:
            del self._pyomo_param_to_solver_param_map[id(p)]

    def _update_variables(self, variables: List[_GeneralVarData]):
        cmodel.process_pyomo_vars(
            self._expr_types,
            variables,
            self._pyomo_var_to_solver_var_map,
            self._pyomo_param_to_solver_param_map,
            self._vars,
            self._solver_var_to_pyomo_var_map,
            False,
            None,
            None,
            True,
        )

    def update_params(self):
        for p_id, p in self._params.items():
            cp = self._pyomo_param_to_solver_param_map[p_id]
            cp.value = p.value

    def _set_objective(self, obj: _GeneralObjectiveData):
        if obj is None:
            const = cmodel.Constant(0)
            lin_vars = list()
            lin_coef = list()
            nonlin = cmodel.Constant(0)
            sense = 0
        else:
            pyomo_expr_types = cmodel.PyomoExprTypes()
            repn = generate_standard_repn(
                obj.expr, compute_values=False, quadratic=False
            )
            const = cmodel.appsi_expr_from_pyomo_expr(
                repn.constant,
                self._pyomo_var_to_solver_var_map,
                self._pyomo_param_to_solver_param_map,
                pyomo_expr_types,
            )
            lin_vars = [
                self._pyomo_var_to_solver_var_map[id(i)] for i in repn.linear_vars
            ]
            lin_coef = [
                cmodel.appsi_expr_from_pyomo_expr(
                    i,
                    self._pyomo_var_to_solver_var_map,
                    self._pyomo_param_to_solver_param_map,
                    pyomo_expr_types,
                )
                for i in repn.linear_coefs
            ]
            if repn.nonlinear_expr is None:
                nonlin = cmodel.appsi_expr_from_pyomo_expr(
                    0,
                    self._pyomo_var_to_solver_var_map,
                    self._pyomo_param_to_solver_param_map,
                    pyomo_expr_types,
                )
            else:
                nonlin = cmodel.appsi_expr_from_pyomo_expr(
                    repn.nonlinear_expr,
                    self._pyomo_var_to_solver_var_map,
                    self._pyomo_param_to_solver_param_map,
                    pyomo_expr_types,
                )
            if obj.sense is minimize:
                sense = 0
            else:
                sense = 1
        cobj = cmodel.NLObjective(const, lin_coef, lin_vars, nonlin)
        cobj.sense = sense
        self._writer.objective = cobj

    def write(self, model: _BlockData, filename: str, timer: HierarchicalTimer = None):
        if timer is None:
            timer = HierarchicalTimer()
        if model is not self._model:
            timer.start('set_instance')
            self.set_instance(model)
            timer.stop('set_instance')
        else:
            timer.start('update')
            self.update(timer=timer)
            for cv, v in self._solver_var_to_pyomo_var_map.items():
                if v.value is not None:
                    cv.value = v.value
            timer.stop('update')
        timer.start('write file')
        self._writer.write(filename)
        timer.stop('write file')

    def update(self, timer: HierarchicalTimer = None):
        super(NLWriter, self).update(timer=timer)
        self._set_pyomo_amplfunc_env()

    def get_ordered_vars(self):
        return [
            self._solver_var_to_pyomo_var_map[i] for i in self._writer.get_solve_vars()
        ]

    def get_ordered_cons(self):
        return [
            self._solver_con_to_pyomo_con_map[i] for i in self._writer.get_solve_cons()
        ]

    def get_active_objective(self):
        return self._objective

    def _set_pyomo_amplfunc_env(self):
        if self._external_functions:
            external_Libs = OrderedSet()
            for con, ext_funcs in self._external_functions.items():
                external_Libs.update([i._fcn._library for i in ext_funcs])
            set_pyomo_amplfunc_env(external_Libs)
        elif "PYOMO_AMPLFUNC" in os.environ:
            del os.environ["PYOMO_AMPLFUNC"]
