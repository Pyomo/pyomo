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
from pyomo.core.kernel.objective import minimize, maximize
from .config import WriterConfig
from ..cmodel import cmodel, cmodel_available


class LPWriter(PersistentBase):
    def __init__(self, only_child_vars=False):
        super(LPWriter, self).__init__(only_child_vars=only_child_vars)
        self._config = WriterConfig()
        self._writer = None
        self._symbol_map = SymbolMap()
        self._var_labeler = None
        self._con_labeler = None
        self._param_labeler = None
        self._obj_labeler = None
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
            self._obj_labeler = TextLabeler()
        else:
            self._var_labeler = NumericLabeler('x')
            self._con_labeler = NumericLabeler('c')
            self._param_labeler = NumericLabeler('p')
            self._obj_labeler = NumericLabeler('obj')

        self._writer = cmodel.LPWriter()

        self.add_block(model)
        if self._objective is None:
            self.set_objective(None)

    def _add_variables(self, variables: List[_GeneralVarData]):
        cmodel.process_pyomo_vars(
            self._expr_types,
            variables,
            self._pyomo_var_to_solver_var_map,
            self._pyomo_param_to_solver_param_map,
            self._vars,
            self._solver_var_to_pyomo_var_map,
            True,
            self._symbol_map,
            self._var_labeler,
            False,
        )

    def _add_params(self, params: List[_ParamData]):
        cparams = cmodel.create_params(len(params))
        for ndx, p in enumerate(params):
            cp = cparams[ndx]
            cp.name = self._symbol_map.getSymbol(p, self._param_labeler)
            cp.value = p.value
            self._pyomo_param_to_solver_param_map[id(p)] = cp

    def _add_constraints(self, cons: List[_GeneralConstraintData]):
        cmodel.process_lp_constraints(cons, self)

    def _add_sos_constraints(self, cons: List[_SOSConstraintData]):
        if len(cons) != 0:
            raise NotImplementedError('LP writer does not yet support SOS constraints')

    def _remove_constraints(self, cons: List[_GeneralConstraintData]):
        for c in cons:
            cc = self._pyomo_con_to_solver_con_map.pop(c)
            self._writer.remove_constraint(cc)
            self._symbol_map.removeSymbol(c)
            del self._solver_con_to_pyomo_con_map[cc]

    def _remove_sos_constraints(self, cons: List[_SOSConstraintData]):
        if len(cons) != 0:
            raise NotImplementedError('LP writer does not yet support SOS constraints')

    def _remove_variables(self, variables: List[_GeneralVarData]):
        for v in variables:
            cvar = self._pyomo_var_to_solver_var_map.pop(id(v))
            del self._solver_var_to_pyomo_var_map[cvar]
            self._symbol_map.removeSymbol(v)

    def _remove_params(self, params: List[_ParamData]):
        for p in params:
            del self._pyomo_param_to_solver_param_map[id(p)]
            self._symbol_map.removeSymbol(p)

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
        cobj = cmodel.process_lp_objective(
            self._expr_types,
            obj,
            self._pyomo_var_to_solver_var_map,
            self._pyomo_param_to_solver_param_map,
        )
        if obj is None:
            sense = 0
            cname = 'objective'
        else:
            cname = self._symbol_map.getSymbol(obj, self._obj_labeler)
            if obj.sense is minimize:
                sense = 0
            else:
                sense = 1
        cobj.sense = sense
        cobj.name = cname
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
            timer.stop('update')
        timer.start('write file')
        self._writer.write(filename)
        timer.stop('write file')

    def get_vars(self):
        return [
            self._solver_var_to_pyomo_var_map[i] for i in self._writer.get_solve_vars()
        ]

    def get_ordered_cons(self):
        return [
            self._solver_con_to_pyomo_con_map[i] for i in self._writer.get_solve_cons()
        ]

    def get_active_objective(self):
        return self._objective

    @property
    def symbol_map(self):
        return self._symbol_map
