# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:12:23 2022

@author: jlgearh
"""
import sys

from numpy.random import normal
from numpy.linalg import norm

from pyomo.common.modeling import unique_component_name
import pyomo.environ as pe
from pyomo.opt import SolverFactory
from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.contrib import appsi

def _is_concrete_model(model):
    assert isinstance(model, ConcreteModel), \
        "Parameter 'model' must be an instance of a Pyomo ConcreteModel"

def _get_solver(solver, solver_options={}, use_persistent_solver=False):
    if use_persistent_solver:
        assert solver == 'gurobi', \
            "Persistent solver option requires the use of Gurobi."
        opt = appsi.solvers.Gurobi()
        opt.config.stream_solver = True
        for parameter, value in solver_options.items():
            opt.set_gurobi_param(parameter, value)
    else:
        opt = SolverFactory(solver)
        for parameter, value in solver_options.items():
            opt.options[parameter] = value
    return opt

def _get_active_objective(model):
    '''
    Finds and returns the active objective function for a model. Assumes there 
    is exactly one active objective.
    '''
    active_objs = [o for o in model.component_data_objects(pe.Objective, 
                                                           active=True)]
    assert len(active_objs) == 1, \
        "Model has zero or more than one active objective function"
    
    return active_objs[0]

def _add_aos_block(model, name='_aos_block'):
    '''Adds an alternative optimal solution block with a unique name.'''
    aos_block = pe.Block()
    model.add_component(unique_component_name(model, name), aos_block)
    return aos_block

def _add_objective_constraint(aos_block, objective, objective_value, 
                             rel_opt_gap, abs_gap):
    '''
    Adds a relative and/or absolute objective function constraint to the 
    specified block.
    '''
    if rel_opt_gap is not None or abs_gap is not None:
        objective_is_min = objective.is_minimizing()
        objective_expr = objective.expr
    
        objective_sense = -1
        if objective_is_min:
            objective_sense = 1
            
        if rel_opt_gap is not None:
            objective_cutoff = objective_value * \
                                (1 + objective_sense * rel_opt_gap)
    
            if objective_is_min:
                aos_block.optimality_tol_rel = \
                    pe.Constraint(expr=objective_expr <= \
                                  objective_cutoff)
            else:
                aos_block.optimality_tol_rel = \
                    pe.Constraint(expr=objective_expr >= \
                                  objective_cutoff)
        
        if abs_gap is not None:
            objective_cutoff = objective_value + objective_sense \
                * abs_gap
    
            if objective_is_min:
                aos_block.optimality_tol_abs = \
                    pe.Constraint(expr=objective_expr <= \
                                  objective_cutoff)
            else:
                aos_block.optimality_tol_abs = \
                    pe.Constraint(expr=objective_expr >= \
                                  objective_cutoff)

def _get_max_solutions(max_solutions):
    assert isinstance(max_solutions, (int, type(None))), \
        'max_solutions parameter must be an integer or None'
    if isinstance(max_solutions, int):
        assert max_solutions >= 1, \
            ('max_solutions parameter must be an integer greater than or equal'
             ' to 1' 
            )
    num_solutions = max_solutions
    if max_solutions is None:
        num_solutions = sys.maxsize
    return num_solutions

def _get_random_direction(num_dimensions):
    idx = 0
    while idx < 100:
        samples = normal(size=num_dimensions)
        samples_norm = norm(samples)
        if samples_norm > 1e-4:
            return samples / samples_norm
        idx += 1
    raise Exception
