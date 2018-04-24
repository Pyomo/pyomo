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
class GDPlbbSolver(pyomo.util.plugin.Plugin):
    """A branch and bound-based GDP solver."""

    pyomo.util.plugin.implements(IOptSolver)
    pyomo.util.plugin.alias('gdplbb',
                            doc='The GDPlbb logic-based Branch and Bound GDP solver')


    def solve(self, model, **kwds):
        heap = []
        root = model.clone()
        incumbent = root
        initial_inactive_disjunctions = model.component_data_objects(
            ctype = Disjunction, active=false): #ComponentSet() from contrib preprocessing plugins equality propogate

        deactivate(all disjunctions)
        num_inactive disjunct
        #Solve root as MINLP subproblems
        #See fix_disjuncts.py
        minlp_solve(root) #some epsilon
        heapq.heappush(h,(root.obj,root))
        while len(h)>0:
            current = heapq.heappop(h)[1]
            if(len(inactive_disjunctions) == ): #TO BEGIN WITH
                incumbent = current
                break
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
