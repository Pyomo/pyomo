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
from .config import WriterConfig
from .cmodel_converter import PyomoToCModelWalker
from pyomo.common.dependencies import attempt_import
from pyomo.core.kernel.objective import minimize


cmodel, cmodel_available = attempt_import('pyomo.contrib.appsi.cmodel.cmodel',
                                          'Appsi requires building a small c++ extension. '
                                          'Please use thye "pyomo build-extensions" command')


class NLWriter(PersistentBase):
    def __init__(self):
        super(NLWriter, self).__init__()
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
        self._walker = PyomoToCModelWalker(self._pyomo_var_to_solver_var_map, self._pyomo_param_to_solver_param_map)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, val: WriterConfig):
        self._config = val

    def set_instance(self, model):
        saved_config = self.config
        saved_update_config = self.update_config
        self.__init__()
        self.config = saved_config
        self.update_config = saved_update_config
        self._model = model

        if self.config.symbolic_solver_labels:
            self._var_labeler = TextLabeler()
            self._con_labeler = TextLabeler()
            self._param_labeler = TextLabeler()
        else:
            self._var_labeler = NumericLabeler('x')
            self._con_labeler = NumericLabeler('c')
            self._param_labeler = NumericLabeler('p')

        self._writer = cmodel.NLWriter()

        self.add_block(model)
        if self._objective is None:
            self.set_objective(None)

    def _add_variables(self, variables: List[_GeneralVarData]):
        cvars = cmodel.create_vars(len(variables))
        for ndx, v in enumerate(variables):
            cv = cvars[ndx]
            cv.name = self._symbol_map.getSymbol(v, self._var_labeler)
            if not v.is_continuous():
                raise NotImplementedError('NLWriter currently only supports continuous variables')
            lb = value(v.lb)
            ub = value(v.ub)
            if lb is not None:
                cv.lb = lb
            if ub is not None:
                cv.ub = ub
            if v.value is not None:
                cv.value = v.value
            if v.is_fixed():
                cv.fixed = True
            self._pyomo_var_to_solver_var_map[id(v)] = cv
            self._solver_var_to_pyomo_var_map[cv] = v

    def _add_params(self, params: List[_ParamData]):
        cparams = cmodel.create_params(len(params))
        for ndx, p in enumerate(params):
            cp = cparams[ndx]
            cp.name = self._symbol_map.getSymbol(p, self._param_labeler)
            cp.value = p.value
            self._pyomo_param_to_solver_param_map[id(p)] = cp

    def _add_constraints(self, cons: List[_GeneralConstraintData]):
        for c in cons:
            cname = self._symbol_map.getSymbol(c, self._con_labeler)
            repn = generate_standard_repn(c.body, compute_values=False, quadratic=False)
            const = self._walker.dfs_postorder_stack(repn.constant)
            lin_vars = [self._pyomo_var_to_solver_var_map[id(i)] for i in repn.linear_vars]
            lin_coef = [self._walker.dfs_postorder_stack(i) for i in repn.linear_coefs]
            if repn.nonlinear_expr is None:
                nonlin = self._walker.dfs_postorder_stack(0)
            else:
                nonlin = self._walker.dfs_postorder_stack(repn.nonlinear_expr)
            cc = cmodel.NLConstraint(const, lin_coef, lin_vars, nonlin)
            lb = c.lower
            ub = c.upper
            if lb is not None:
                cc.lb = self._walker.dfs_postorder_stack(lb)
            if ub is not None:
                cc.ub = self._walker.dfs_postorder_stack(ub)
            self._writer.add_constraint(cc)
            self._pyomo_con_to_solver_con_map[c] = cc
            self._solver_con_to_pyomo_con_map[cc] = c

    def _add_sos_constraints(self, cons: List[_SOSConstraintData]):
        if len(cons) != 0:
            raise NotImplementedError('NL writer does not support SOS constraints')

    def _remove_constraints(self, cons: List[_GeneralConstraintData]):
        for c in cons:
            cc = self._pyomo_con_to_solver_con_map.pop(c)
            self._writer.remove_constraint(cc)
            self._symbol_map.removeSymbol(c)
            self._con_labeler.remove_obj(c)
            del self._solver_con_to_pyomo_con_map[cc]

    def _remove_sos_constraints(self, cons: List[_SOSConstraintData]):
        if len(cons) != 0:
            raise NotImplementedError('NL writer does not support SOS constraints')

    def _remove_variables(self, variables: List[_GeneralVarData]):
        for v in variables:
            cvar = self._pyomo_var_to_solver_var_map.pop(id(v))
            del self._solver_var_to_pyomo_var_map[cvar]
            self._symbol_map.removeSymbol(v)
            self._var_labeler.remove_obj(v)

    def _remove_params(self, params: List[_ParamData]):
        for p in params:
            del self._pyomo_param_to_solver_param_map[id(p)]
            self._symbol_map.removeSymbol(p)
            self._param_labeler.remove_obj(p)

    def update_variables(self, variables: List[_GeneralVarData]):
        for v in variables:
            cv = self._pyomo_var_to_solver_var_map[id(v)]
            if not v.is_continuous():
                raise NotImplementedError('NLWriter currently only supports continuous variables')
            lb = value(v.lb)
            ub = value(v.ub)
            if lb is None:
                cv.lb = -cmodel.inf
            else:
                cv.lb = lb
            if ub is None:
                cv.ub = cmodel.inf
            else:
                cv.ub = ub
            if v.value is not None:
                cv.value = v.value
            if v.is_fixed():
                cv.fixed = True
            else:
                cv.fixed = False

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
            repn = generate_standard_repn(obj.expr, compute_values=False, quadratic=False)
            const = self._walker.dfs_postorder_stack(repn.constant)
            lin_vars = [self._pyomo_var_to_solver_var_map[id(i)] for i in repn.linear_vars]
            lin_coef = [self._walker.dfs_postorder_stack(i) for i in repn.linear_coefs]
            if repn.nonlinear_expr is None:
                nonlin = cmodel.Constant(0)
            else:
                nonlin = self._walker.dfs_postorder_stack(repn.nonlinear_expr)
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

    def get_ordered_vars(self):
        return [self._solver_var_to_pyomo_var_map[i] for i in self._writer.get_solve_vars()]

    def get_ordered_cons(self):
        return [self._solver_con_to_pyomo_con_map[i] for i in self._writer.get_solve_cons()]

    def get_active_objective(self):
        return self._objective
