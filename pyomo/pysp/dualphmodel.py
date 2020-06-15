#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import copy

from pyomo.core import ConcreteModel, Suffix, Var, Block, Set, RangeSet, Objective, Constraint, NonNegativeReals, sum_product, value
from pyomo.opt import SolverFactory
from pyomo.core.expr.current import ExpressionBase
from pyomo.pysp.phutils import update_all_rhos, find_active_objective

from six import iteritems

class DualPHModel(object):

    def _create_model(self):
        model = ConcreteModel()
        model.dual = Suffix(direction=Suffix.IMPORT,default=0.0)
        for stage in self._ph._scenario_tree._stages[:-1]: # all blended stages
            for tree_node in stage._tree_nodes:
                setattr(model,tree_node._name,Block())
                block = getattr(model,tree_node._name)
                block.var_to_id = {}
                block.id_to_var = []
                for cntr, (var,index) in enumerate((name,idx) for name, indices in iteritems(tree_node._variable_indices) for idx in indices):
                    block.var_to_id[var,index] = cntr
                    block.id_to_var.append((var,index))
                block.var_index = RangeSet(0,len(block.id_to_var)-1)
        self._model = model
        
    def __init__(self,ph):
        self._ph = ph
        self._model = None
        self._solved = False
        self._alphas = {}
        self._wprod = {}
        self._wbars = {}
        self._iter = -1
        self._history = []
        self._create_model()

    def add_cut(self,first=False):
        self._iter += 1
        model = self._model

        self._wprod[self._iter] = self._compute_weight_weight_inner_product()
        if first is True:
            self._alphas[self._iter] = -( self._compute_objective_term() + (self._ph._rho/2.0)*self._wprod[self._iter] )
        else:
            self._alphas[self._iter] = -(self._compute_objective_term()) + self._compute_xbar_weight_inner_product()

        if self._solved is True:
            if self._compute_convergence() is True:
                return True

        model.del_component('cuts')
        model.cuts = Set(initialize=sorted(self._alphas.keys()))
        model.del_component('beta')
        model.beta = Var(model.cuts,within=NonNegativeReals)
        model.del_component('beta_sum_one')
        model.beta_sum_one = Constraint(expr=sum_product(model.beta)==1)
        model.del_component('obj')
        model.obj = Objective(expr=sum(self._alphas[i]*model.beta[i] for i in model.cuts))

        self._wbars[self._iter] = {}
        for stage in self._ph._scenario_tree._stages[:-1]: # all blended stages
            for tree_node in stage._tree_nodes:
                self._wbars[self._iter][tree_node._name] = copy.deepcopy(tree_node._wbars)
                block = getattr(model,tree_node._name)
                def _c_rule(block,i):
                    lhs = sum(model.beta[k]*self._wbars[k][tree_node._name][block.id_to_var[i][0]][block.id_to_var[i][1]] for k in model.beta.index_set())
                    if not isinstance(lhs,ExpressionBase):
                        return Constraint.Skip
                    return lhs == 0
                block.del_component('con')
                block.con = Constraint(block.var_index, rule=_c_rule)
        return False

    def solve(self):
        opt = SolverFactory("cplex")
        model = self._model
        model.dual.clearValue()
        model.load(opt.solve(model))#,keepfiles=True,symbolic_solver_labels=True,tee=True))
        self._solved = True
        self._update_tree_node_xbars()

    def _compute_convergence(self):
        model = self._model
        #print(model.dual[model.beta_sum_one])
        #print(self._compute_xbar_wbar_inner_product())
        #print(self._alphas[self._iter])
        dual_convergence = model.dual[model.beta_sum_one]+self._compute_xbar_wbar_inner_product()-self._alphas[self._iter]
        print("**** Convergence Metric: "+str(dual_convergence))
        self._history.append(dual_convergence)
        if self._iter > 1:
            if (dual_convergence < 1e-4) or all(self._history[-1] == x for x in self._history[-5:-1]):
                rho_old = self._ph._rho
                #(RHS-k-a_old)(2/w) + 1/rold = 1/r_new
                rho_new = self._ph._rho = rho_old*1.5 # 1.0/( (1.0/rho_old) + (dual_convergence-1)*(2.0/self._wprod[self._iter]) )
                if not (raw_input("Set rho to "+str(rho_new)+" ?") == 'n'):
                    update_all_rhos(self._ph._instances, self._ph._scenario_tree, rho_value=self._ph._rho)
                    for i in list(self._alphas.keys()):
                        self._alphas[i] = self._alphas[i] + (0.5/rho_new-0.5/rho_old)*self._wprod[i]
                    return False
                return True
        return False

    def _compute_objective_term(self):
        total = 0.0
        for scenario in self._ph._scenario_tree._scenarios:
            instance = self._ph._instances[scenario._name]
            total += -scenario._probability*value(find_active_objective(instance)) 
        return total

    def _compute_weight_weight_inner_product(self):
        w_inner_prod = 0.0
        for stage in self._ph._scenario_tree._stages[:-1]:  # all blended stages
            for tree_node in stage._tree_nodes:
                for variable_name, variable_indices in iteritems(tree_node._variable_indices):
                    weight_parameter_name = "PHWEIGHT_"+variable_name
                    rho_parameter_name = "PHRHO_"+variable_name
                    for scenario in tree_node._scenarios:
                        instance = self._ph._instances[scenario._name]
                        weight_parameter = getattr(instance, weight_parameter_name)
                        rho_parameter = getattr(instance, rho_parameter_name)
                        for index in variable_indices:
                            w_inner_prod += scenario._probability*weight_parameter[index].value*weight_parameter[index].value
        return w_inner_prod

    def _compute_xbar_weight_inner_product(self):
        inner_prod = 0.0
        for stage in self._ph._scenario_tree._stages[:-1]:  # all blended stages
            for tree_node in stage._tree_nodes:
                tree_node_xbars = tree_node._xbars
                for variable_name, variable_indices in iteritems(tree_node._variable_indices):
                    weight_parameter_name = "PHWEIGHT_"+variable_name
                    var_xbar = tree_node_xbars[variable_name]
                    for scenario in tree_node._scenarios:
                        instance = self._ph._instances[scenario._name]
                        weight_parameter = getattr(instance, weight_parameter_name)
                        for index in variable_indices:
                            inner_prod += scenario._probability*var_xbar[index]*weight_parameter[index].value
        return inner_prod

    def _compute_xbar_wbar_inner_product(self):
        inner_prod = 0.0
        for stage in self._ph._scenario_tree._stages[:-1]:  # all blended stages
            for tree_node in stage._tree_nodes:
                tree_node_xbars = tree_node._xbars
                tree_node_wbars = tree_node._wbars
                for variable_name, variable_indices in iteritems(tree_node._variable_indices):
                    var_xbar = tree_node_xbars[variable_name]
                    var_wbar = tree_node_wbars[variable_name]
                    for index in variable_indices:
                        inner_prod += var_xbar[index]*var_wbar[index]
        return inner_prod

    # This function updates the xbars dictionary on the ph scenario tree
    # to match the duals of the associated constraints in this model
    def _update_tree_node_xbars(self):
        # update the xbars
        model = self._model
        for stage in self._ph._scenario_tree._stages[:-1]: # all blended stages
            for tree_node in stage._tree_nodes:
                block = getattr(model,tree_node._name)
                for idx, con_data in iteritems(block.con):
                    var_name = block.id_to_var[idx][0]
                    var_index = block.id_to_var[idx][1]
                    tree_node._xbars[var_name][var_index] = model.dual[con_data]
            print(stage._tree_nodes[0]._xbars)
