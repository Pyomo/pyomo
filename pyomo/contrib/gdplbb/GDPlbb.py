import logging
from copy import deepcopy
from math import copysign, fabs

from six import iteritems

import pyomo.util.plugin
from pyomo.core.base import expr as EXPR
from pyomo.core.base import (Block, Constraint, ConstraintList, Expression,
                             Objective, Set, Suffix, TransformationFactory,
                             Var, maximize, minimize, value)
from pyomo.core.base.block import generate_cuid_names
from pyomo.core.base.symbolic import differentiate
from pyomo.core.kernel import (ComponentMap, ComponentSet, NonNegativeReals,
                               Reals)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolutionStatus, SolverFactory, SolverStatus
from pyomo.opt.base import IOptSolver
from pyomo.opt.results import ProblemSense, SolverResults

import heapq

@SolverFactory.register('gdplbb',doc='Branch and Bound based GDP Solver')


class GDPlbbSolver(opject):
    """A branch and bound-based GDP solver."""

    def solve(self, model, **kwds):
        """
        PSEUDOCODE
        Initialize minheap h ordered by objective value
        root = model.clone
        root.init_active_disj = list of currently active disjunctions
        root.curr_active_disj = []
        for each disj in root.init_active_disj
        	Deactivate disj
        Apply Sat Solver to root
        if infeasible
        	Return no-solution EXIT
        solve root
        push (root,root.obj.value()) onto minheap h

        while not heap.empty()
        	pop (m,v) from heap
        	if len(m.init_active_disj == 0):
        		copy m to model
        		return good-solution EXIT
        	find disj D in m.init_active_disj

        	for each disjunct d in D
        		set d false
        	for each disjunct d in D
        		set d true
        		mnew = m.clone
        		Apply Sat Solver to mnew
        		if mnew infeasible
        			Return no-solution EXIT
        		solve(mnew)
        		push (mnew,menw.obj.value()) onto minheap h
        		set d false
        """

        #Validate model to be used with gdplbb
        validate_model(model)
        #Set solver as an MINLP
        solver = SolverFactory('baron')

        indicator_list_name = unique_compoinent_name(model,"_indicator_list")
        indicator_vars = []
        for disjunction in model.component_data_objects(
            ctype = Disjunction, active=True):
            for disjunct in disjunction.disjuncts:
                indicator_vars.append(disjunction.disjunct.indicator_var)
        setattr(model, indicator_list_name, indicator_vars)


        heap = []
        root = model.clone()
        root.init_active_disjunctions = list(model.component_data_objects(
            ctype = Disjunction, active=True))
        root.curr_active_disjunctions = []
        for djn in root.init_active_disjunctions:
            djn.deactivate()
        #Satisfiability check would go here

        obj_value = minlp_solve(root,solver)#fix

        heapq.heappush(heap,(obj_value,root))
        while len(heap)>0:
            mdl = heapq.heappop(h)[1]
            if(len(mdl.init_active_disjunctions) ==  0):
                orig_var_list = getattr(model, indicator_list_name)
                best_soln_var_list = getattr(mdl, indicator_list_name)
                for orig_var, new_var in zip(orig_var_list,best_soln_var_list):
                    if not orig_var.is_fixed():
                        orig_var.value = new_var.value
                TransformationFactory('gdp.fix_disjuncts').apply_to(model)
                return solver.solve(model)

            disjunction = mdl.init_active_disjunctions.pop(0)
            disjunction.activate()
            mdl.curr_active_disjunctions.append(disjunction)
            for disj in list(disjunction.disjuncts):
                disj.indicator_var = 0
            for disj in list(disjunction.disjuncts):
                disj.indicator_var = 1
                mnew = model.clone()
                disj.indicator_var = 0
                obj_value = minlp_solve(mnew,solver) #fix
                heapq.heappush(heap,(obj_value,mnew))



    def validate_model(self,model):
        #Validates that model has only exclusive disjunctions
        for d in model.component_data_objects(
            ctype = Disjunction, active=True):
            if(not d.xor):
                raise ValueError('GDPlbb unable to handle '
                                'non-exclusive disjunctions')

    def minlp_solve(self,gdp,solver):
        minlp = gdp.clone()
        TransformationFactory('gdp.fix_disjuncts').apply_to(minlp)
        solver.solve(minlp)
        return value(minlp.obj.expr)
        #TO FINISH
