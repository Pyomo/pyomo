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
        	if len(m.init_active_disj == len(m.curr_active_disj):
        		copy m to model
        		return good-solution EXIT
        	find disj D in m.init_active_disj such that disj is not in m.curr_active_disj

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

        solver = SolverFactory('(SOME MINLP SOLVER)')

        heap = []
        root = model.clone()
        root.init_active_disjunctions = model.component_data_objects(
            ctype = Disjunction, active=True):
        root.curr_active_disjunctions = []
        for djn in root.init_active_disjunctions
            djn.deactivate()
        #Satisfiability check would go here

        solver.solve(root)

        heapq.heappush(heap,(value(root.obj.expr),root))

        while len(heap)>0:
            mdl = heapq.heappop(h)[1]
            if(len(mdl.init_active_disjunctions) ==  0):
                ASSIGN VALS FROM mdl TO model
                return
-----------------------------------------------------------------------

            disjunction = inactive_disjunctions[0]
            activate(disjunction)
            for each clause in disjunction:
                new = current.clone()
                set clause True and fix
                ##convert to MINLP
                minlp_solve(new)
                heapq.heappush(h,(new.obj,new))
        return incumbent

    def deactivate_disjunctions(self,model):
        for d in model.component_data_objects(ctype = Disjunction,active = True):
            #deactivate disjunctions




    def validate_model(self,model):
        for d in model.component_data_objects(
            ctype = Disjunction, active=True):
            if(not d.xor):
                raise ValueError('GDPlbb unable to handle '
                                'non-exclusive disjunctions')
