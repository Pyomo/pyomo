#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# Hybrid GDP Reformulation Algorithm  (HyRA)
#
# This is a basic implementation of the hybrid GDP relaxation algorithms
# proposed by Trespalacios and Grossmann in:
#
#    Trespalacios, F. and I.E. Grossmann.  "Algorithmic approach for
#    improved mixed-integer reformulations of convex Generalized
#    Disjunctive Programs" INFORMS Journal on Computing, 27(1) 2015.
#       preprint: http://egon.cheme.cmu.edu/Papers/Trespalacios_Grossmann_improving_GDP.pdf
#
#    Trespalacios, F. and I.E. Grossmann.  "Cutting planes algorithm for
#    convex Generalized Disjunctive Programs," submitted for
#    publication (2014).  
#       preprint: http://egon.cheme.cmu.edu/Papers/Trespalacio_gdp_cutting_planes_v2.pdf
#
#  The HyRA is implemented in Pyomo using the following Pseudocode:
#   1) For each disjunction in the model
#       a) Deactivate the disjunction and the corresponding disjuncts
#          (*TODO: make the reformulations ignore deactivated blocks!)
#       b) Out-of place reformulation (chull) 
#       c) In-place relaxation
#       d) For each disjunct in the "deactivated" disjunction,
#               i) "promote it" up to the model (reactivate it)
#               ii) Solve, and record the characteristic value and infeasibility
#   2) Pick the characteristic (Key) disjunction (on the original model
#      instance)
#       a) calculate W for all disjunctions in the model
#       b) key disjunction (k*) is the largest W using the largest
#          characteristic value to break ties
#   3) Apply basic steps
#       a) Select another disjunction to apply basic step (using alg from (2)),
#       b) Apply basic step
#       c) analyse & eliminate redundant / infeasible disjunct (#4 in [Ref.])
#       d) Repeat until any of the termination criteria are satisfied
#   4) Apply improper basic step of global constraints with key disjunction
#   5) Reformulate
#       a) Key Disjunction with the chull (*TODO: support only
#          reformulating parts of a model!)
#       b) any remaining disjunctions with BigM
#

from pyomo.util.plugin import alias
from pyomo.core import *
from pyomo.core.base import Transformation
from pyomo.core.base.block import SortComponents
from pyomo.core.base.expr import identify_variables, clone_expression
from pyomo.gdp import *

from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

from six import iterkeys, itervalues, iteritems

import sys
import weakref
import logging
logger = logging.getLogger('pyomo.gdp')
# Uncomment the following to get HYRA to output logging info
#logger.setLevel(logging.INFO)
import time 


_domain_relaxation_map = {
    Integers : Reals, 
    PositiveIntegers : PositiveReals, 
    NonPositiveIntegers : NonPositiveReals, 
    NegativeIntegers : NegativeReals,
    NonNegativeIntegers : NonNegativeReals,
    IntegerInterval : RealInterval,
}

_acceptable_termination_conditions = set([
    TerminationCondition.optimal,
    TerminationCondition.globallyOptimal,
    TerminationCondition.locallyOptimal,
])
_infeasible_termination_conditions = set([
    TerminationCondition.infeasible,
    TerminationCondition.invalidProblem,
])

def relax_discrete_vars(model):
    # Relax the Big-M model
    relaxed_vars = {}
    _base_model_vars = model.component_data_objects(
        Var, active=True, descend_into=(Block, Disjunct) )
    for var in _base_model_vars:
        if var.domain is Binary or var.domain is Boolean:
            relaxed_vars[id(var)] = (var, var.domain)
            var.domain = NonNegativeReals
            var.setlb(0)
            var.setub(1)
        elif var.domain in _domain_relaxation_map:
            relaxed_vars[id(var)] = (var, var.domain)
            var.domain = _domain_relaxation_map[var.domain]
    return relaxed_vars


class HybridReformulationAlgorithm(Transformation):

    alias('gdp.hyra', doc="")

    def __init__(self):
        super(HybridReformulationAlgorithm, self).__init__()
        self.timeInSolver = 0
        self.initialTime = 0
        self.info = False

    def _solve_model(self, m):

        results = self.solver.solve(m)
        ss = results.solver.status 
        tc = results.solver.termination_condition

        self.timeInSolver += results['Solver'][0]['Time']

        if ss == SolverStatus.ok and tc in _acceptable_termination_conditions:
            #m.load(results)
            return None, None
        elif tc in _infeasible_termination_conditions:
            return 'INFEASIBLE', results
        else:
            return 'NONOPTIMAL', results
    
    def _evaluate_disjunct_lp_relaxation(self, model):
        _infeasible_disjuncts = []
        _LP_values = {}
        _characteristic_value = {}

        all_disjunctions = model.component_data_objects(
            Disjunction, active=True, descend_into=(Block, Disjunct), 
            descent_order=TraversalStrategy.PostfixDepthFirstSearch )

        for _single_disjunction in all_disjunctions:
            # (1.b)
            tmp_model = model.clone()
            _tmp_single_disjunction = ComponentUID(
                _single_disjunction ).find_component(tmp_model)

            # (1.a) Deactivate the disjunction and the corresponding
            # disjuncts (*TODO: make the reformulations ignore
            # deactivated blocks!)
            _tmp_single_disjunction.deactivate()
            if self.info: 
                logger.info( 'GDP(hyra) Disjunction %s formed across:' % 
                             _single_disjunction.cname(True) )

            _active_disjuncts = []
            for _disjunct in _tmp_single_disjunction.parent_component()._disjuncts[_single_disjunction.index()]:
                if _disjunct.active:
                    _disjunct.deactivate()
                    _disjunct.indicator_var.fix(0)
                    _active_disjuncts.append(_disjunct)
                    if self.info: 
                        logger.info("GDP(hyra) " + ' '*3 + _disjunct.cname(True))
                else:
                    if self.info: 
                        logger.info("GDP(hyra) " + ' '*3 + _disjunct.cname(True) + ' [inactive]')

            TransformationFactory('gdp.chull').apply_to(tmp_model)

            # (1.c) TODO: reimplement as a call to a relaxation transformation
            # TransformationFactory('relax_binary').apply_to(tmp_model)
            relax_discrete_vars(tmp_model)

            # (1.d)
            for _disjunct in _active_disjuncts:
                tmp_model._tmp_basic_step = Block()

                # models may directly reference the indicator variable
                # outside the disjunct.  We need to remember it here so
                # that we can fix it to 1 here, and then at the end fix
                # it back to 0.
                tmp_indicator_var = _disjunct.indicator_var

                # (1.d.i)
                tmp_indicator_var.fix(1)
                for _name, _comp in _disjunct.component_map(active=True).iteritems():
                    _disjunct.del_component(_name)
                    tmp_model._tmp_basic_step.add_component(_name, _comp)

                # (1.d.ii)
                err, results = self._solve_model(tmp_model)
                if err:
                    _src_disjunct = ComponentUID(_disjunct).find_component(model)
                    if self.DETERMINISTIC_ALGORITHM:
                        _infeasible_disjuncts.append( _src_disjunct )
                    else:
                        _src_disjunct.deactivate()
                    _obj = None
                    if self.info: 
                        logger.info("GDP(hyra) " + ' '*6 + "%s: %s" % (_disjunct.cname(True), err) )
                    if err != 'INFEASIBLE':
                        logger.warning("GDP(hyra) subsolve returned error:\n%s" % ( results.solver, ))
                else:
                    _obj = value( list(tmp_model.component_data_objects(Objective, active=True))[-1] )
                    _LP_values[id( ComponentUID(_disjunct).find_component_on(
                        model) )] = _obj
                    _sing_disj_id = id(_single_disjunction)
                    _characteristic_value[ _sing_disj_id ] = min( 
                        _characteristic_value.get(_sing_disj_id, _obj ), 
                        _obj ) 
                    if self.info: 
                        logger.info("GDP(hyra) " + ' '*6 + "%s: %s" % (_disjunct.cname(True), _obj) )


                tmp_indicator_var.fix(0)
                tmp_model.del_component('_tmp_basic_step')

            if self.info: 
                logger.info("GDP(hyra) characteristic value %s" % _characteristic_value.get(
                    id(_single_disjunction), None ) )
            tmp_model = None 
            # suggestion move this inside the loop - by Mahdi
            #
            # [JDS 7/27] The above is fine inside the loop, although not
            # strictly necessary (the previous model will be implicitly
            # released by the next iteration
        return _characteristic_value, _LP_values, _infeasible_disjuncts



    def _apply_to(self, model, **kwds): 
        self.initialTime = time.time()
        self.timeInSolver = 0
        self.info = bool( logger.isEnabledFor(logging.INFO) )

        solver_name = kwds.get('options',{}).get('solver', 'bonmin')
        self.solver = SolverFactory(solver_name)#, solver_io='python')
        if self.solver is None:
            raise RuntimeError("Unknown solver %s specified for the gdp.hyra transformation" % solver_name)

        # If set to "TRUE", the algorithm first calculates all the
        # characteristic values, then return and remove the infeasible
        # disjuncts
        #
        # If set to "FALSE", the algorithm removes the infeasible terms
        # as characteristic values are being calculated; as a result,
        # the values of characteristic values depend on the order that
        # disjuncts are visited, and is path dependant
        self.DETERMINISTIC_ALGORITHM = False

        _characteristic_value, _LP_values, _disjuncts_to_deactivate \
            = self._evaluate_disjunct_lp_relaxation(model)

        # If we defer deactivating infeasible disjuncts until after all
        # characteristic values are calculated, then we need to go back and
        # deactivate them now
        for _disjunct in _disjuncts_to_deactivate:
            _disjunct.deactivate()


        #=====================================================================
        # Measuring the size of the problem, needed for terminations criteria  
        # ConstraintNumberoriginal need to be calculated before Basic Steps
        ConstraintNumberoriginal=0
        # DisjunctNumberoriginal need to be calculated before Basic Steps
        DisjunctNumberoriginal=0

        for _idx, _single_disjunction in model.component_data_iterindex(Disjunction, active=True):
            for _disjunct in _single_disjunction.parent_component()._disjuncts[_idx[1]]:
                if not _disjunct.active:
                    continue
                DisjunctNumberoriginal +=1
                for c in _disjunct.component_data_objects(Constraint, active=True, descend_into=(Block, Disjunct)):
                    ConstraintNumberoriginal +=1
        #=====================================================================

        _disjunction_by_id = {}
        _vars_by_disjunction = {} 
        _W_by_disjunction = {}

        for _idx, _single_disjunction in model.component_data_iterindex(Disjunction, active=True):
            _disjunction_by_id[id(_single_disjunction)] = _single_disjunction
            _vars_by_disjunction[id(_single_disjunction)] = set()
            _W_by_disjunction[id(_single_disjunction)] = 0
            for _disjunct in _single_disjunction.parent_component()._disjuncts[_idx[1]]:
                if not _disjunct.active:
                    continue
                _all_con = _disjunct.component_data_objects(Constraint, active=True)
                for _single_constraint in _all_con:
                    _vars_by_disjunction[id(_single_disjunction)].update(
                        id(x) for x in identify_variables(
                            _single_constraint.body, include_fixed=False ) )

        # (2.a)
        _all_disjunctions = model.component_data_objects(Disjunction, active=True)
        for _single_disjunction in _all_disjunctions:
            _self = id(_single_disjunction)
            for _other in _vars_by_disjunction.iterkeys():
                if _other >= _self:
                    continue
                _common_vars = _vars_by_disjunction[_self].intersection(
                _vars_by_disjunction[_other] )
                if len(_common_vars) > 0:
                    _delta_w = 1. / (
                        len(_vars_by_disjunction[_other]) * 
                        len(_vars_by_disjunction[_self]) )
                    _W_by_disjunction[_other] += _delta_w
                    _W_by_disjunction[_self]  += _delta_w        

        # (2.b)
        _max_W = max(_W_by_disjunction.values())
        _max_W_id = [ k for k,v in _W_by_disjunction.iteritems() if v == _max_W ]
        _max_char_vals = max( _characteristic_value[k] for k in _max_W_id )
        key_disjunction_id = [ k for k in _max_W_id 
                               if _characteristic_value[k] == _max_char_vals ][0]
        key_disjunction = _disjunction_by_id[key_disjunction_id]


        # declare a block to hold the disjunctions and constraints that
        # we are about to create
        model._basic_step_hybrid = Block()
        model._basic_step_hybrid.indicator_var_maps = ConstraintList()

        if self.info: 
            logger.info("GDP(hyra) Calculating basic steps...")

        # (3)
        BasicStepIteration= 0
        BasicStepObjectiveValue=[_characteristic_value[key_disjunction_id]]

        while True:
            BasicStepIteration += 1
            if self.info: 
                logger.info("GDP(hyra)  ...basic step iteration %s" % BasicStepIteration)
            # (3.a)
            for k in _W_by_disjunction:
                _W_by_disjunction[k] = 0
            _self = key_disjunction_id  
            for _other in _vars_by_disjunction:
                if _other == _self:
                    continue
                _common_vars = _vars_by_disjunction[_self].intersection(
                        _vars_by_disjunction[_other] )
                if len(_common_vars) > 0:
                    _delta_w = 1. / (
                        len(_vars_by_disjunction[_other]) * 
                        len(_vars_by_disjunction[_self]) )
                    _W_by_disjunction[_other] += _delta_w

            _max_W = max(_W_by_disjunction.values())
            _max_W_id = [ k for k,v in _W_by_disjunction.iteritems() if v == _max_W ]
            _max_char_vals = max( _characteristic_value[k] for k in _max_W_id if k is not key_disjunction_id)
            target_disjunction_id = [ k for k in _max_W_id if _characteristic_value[k] == _max_char_vals and k is not key_disjunction_id ][0]
            target_disjunction = _disjunction_by_id[target_disjunction_id]


            # (3.b)
            _key_disjunction_index = None if key_disjunction.parent_component() is key_disjunction else key_disjunction.index()
            _key_disjuncts = [x for x in key_disjunction.parent_component()._disjuncts[_key_disjunction_index] if x.active]
            _target_disjunction_index = None if target_disjunction.parent_component() is target_disjunction else target_disjunction.index()
            _target_disjuncts = [x for x in target_disjunction.parent_component()._disjuncts[_target_disjunction_index] if x.active]


            i = 0
            key_disjunction.deactivate()
            for _disjunct in key_disjunction.parent_component()._disjuncts[_key_disjunction_index]:
                _disjunct.deactivate()
                _iv = _disjunct.indicator_var
                _disjunct.del_component(_iv)
                _disjunct.indicator_var_name = 'basic_step_%i_key_iv_%s' % ( BasicStepIteration, i )
                model._basic_step_hybrid.add_component(_disjunct.indicator_var_name, _iv)
                i += 1
            i = 0
            target_disjunction.deactivate()
            for _disjunct in target_disjunction.parent_component()._disjuncts[_target_disjunction_index]:
                _disjunct.deactivate()
                _iv = _disjunct.indicator_var
                _disjunct.del_component(_iv)
                _disjunct.indicator_var_name = 'basic_step_%i_target_iv_%s' % ( BasicStepIteration, i )
                model._basic_step_hybrid.add_component(_disjunct.indicator_var_name, _iv)
                i += 1
            #target_disjunction = None


            if model._basic_step_hybrid.component('key_disjunct') is not None:
                model._basic_step_hybrid.del_component('key_disjunct')
                model._basic_step_hybrid.del_component('key_disjunct_index')
                model._basic_step_hybrid.del_component('key_disjunction')
            model._basic_step_hybrid.key_disjunct = Disjunct(range(len(_key_disjuncts)*len(_target_disjuncts)))
            for k in range(len(_key_disjuncts)):
                _k_iv = getattr( model._basic_step_hybrid, _key_disjuncts[k].indicator_var_name )
                for t in range(len(_target_disjuncts)):
                    _t_iv = getattr( model._basic_step_hybrid, _target_disjuncts[t].indicator_var_name )
                    if self.info: 
                        logger.info("GDP(hyra) constructing key disjunct %s x %s" % (k, t))
                    new_disjunct =  model._basic_step_hybrid.key_disjunct[k*len(_target_disjuncts)+t]
                    tmp = _key_disjuncts[k].clone()
                    for name,comp in list( tmp.component_map(active=True).iteritems() ):
                        #if name is 'indicator_var':
                        #    continue
                        tmp.del_component(name)
                        new_disjunct.add_component(name+'_k', comp)
                    tmp = _target_disjuncts[t].clone()
                    for name,comp in list( tmp.component_map(active=True).iteritems() ):
                        #if name is 'indicator_var':
                        #    continue
                        tmp.del_component(name)
                        new_disjunct.add_component(name+'_t', comp)

                    model._basic_step_hybrid.indicator_var_maps.add(
                        new_disjunct.indicator_var <= _k_iv )
                    model._basic_step_hybrid.indicator_var_maps.add(
                        new_disjunct.indicator_var <= _t_iv )
                    model._basic_step_hybrid.indicator_var_maps.add(
                        new_disjunct.indicator_var >= _k_iv + _t_iv - 1 )

            key_disjunction \
                = model._basic_step_hybrid.key_disjunction \
                = Disjunction(expr=model._basic_step_hybrid.key_disjunct.values())

            # add the new key disjunction (after basic step) to our data structures (for next pass)
            key_disjunction_id = id(key_disjunction)
            _disjunction_by_id[key_disjunction_id] = key_disjunction
            _vars_by_disjunction[key_disjunction_id] = _vars_by_disjunction[_self].union(
                    _vars_by_disjunction[target_disjunction_id])
            del _vars_by_disjunction[_self]
            _W_by_disjunction[key_disjunction_id] = _W_by_disjunction.pop(_self)
    
            del _vars_by_disjunction[target_disjunction_id]
            del _W_by_disjunction[target_disjunction_id]
            del _disjunction_by_id[target_disjunction_id]

            # (3.c; F.T.#4)

            ### Testing/removing infeasible Disjuncts in the new Key Disjunction  
            tmp_model = model.clone()
            _single_disjunction = key_disjunction
            _tmp_single_disjunction = ComponentUID(_single_disjunction).find_component_on(tmp_model)
            # Deactivate the disjunction and the corresponding disjuncts (*TODO: make the reformulations ignore deactivated blocks!)
            _tmp_single_disjunction.deactivate()
            if self.info: 
                logger.info("GDP(hyra) Disjunction %s formed across:" % _tmp_single_disjunction.cname(True))
            # I am really not sure why _tmp_single_disjunction.index isn't defined!
            _active_disjuncts = [d for d in _tmp_single_disjunction.parent_component()._disjuncts.values()[0] if d.active ]
            for _disjunct in _active_disjuncts:
                if self.info: 
                    logger.info("GDP(hyra)    %s" % ( _disjunct.cname(True), ))
                _disjunct.deactivate()
                _disjunct.indicator_var.fix(0)

            TransformationFactory('gdp.chull').apply_to(tmp_model)

            # (To-Do)
            # TransformationFactory('relax_binary').apply_to(tmp_model)
            relax_discrete_vars(tmp_model)

            _char_values=[]
            tmp_model._tmp_basic_step = Block(tmp_model._basic_step_hybrid.key_disjunct.index_set())
            # I am really not sure why _tmp_single_disjunction.index isn't defined!
            for _disjunct in _active_disjuncts:
                _main_disjunct = ComponentUID(_disjunct).find_component(model)
                idx = _disjunct.index()
                tmp_indicator_var = _disjunct.indicator_var
                tmp_indicator_var.fix(1)
                tmp_model._tmp_basic_step._data[idx] = _disjunct.parent_component()._data.pop(idx)
                _disjunct._component = weakref.ref(tmp_model._tmp_basic_step)
                _disjunct.activate()

                err, results = self._solve_model(tmp_model)
                if err:
                    _main_disjunct.deactivate()
                    _main_disjunct.indicator_var.fix(0)
                    if self.info: 
                        logger.info("GDP(hyra) " + ' '*6 + "Testing feasibility for disjunct %s: %s" 
                                    % (_disjunct.cname(True), err) )
                else:
                    _objectives = tmp_model.component_map(Objective, active=True).values()
                    if len(_objectives) != 1:
                        raise RuntimeError("I am confused: I couldn't find exactly one active objective")
                    obj_value = value(_objectives[0])
                    if self.info: 
                        logger.info("GDP(hyra) " + ' '*6 + "Testing feasibility for disjunct %s: %s" 
                                    % (_disjunct.cname(True), obj_value) )

                    if id(_single_disjunction) not in _characteristic_value:
                        _characteristic_value[id(_single_disjunction)] = obj_value
                    else:
                        _characteristic_value[id(_single_disjunction)] = min(_characteristic_value[id(_single_disjunction)], obj_value)
                    _char_values.append(obj_value)

                # (cleanup)
                tmp_indicator_var.fix(0)
                del tmp_model._tmp_basic_step._data[idx]

            BasicStepObjectiveValue.append(min(_char_values))
            if self.info: 
                logger.info("GDP(hyra) Basic Steps Objective Value %s" % BasicStepObjectiveValue)

            tmp_model = None


            #=====================================================================
            # Updating the size of the problem, needed for terminations criteria  
            # ConstraintNumberoriginal is calculated earlier
            ConstraintCounter=0
            # DisjunctNumberoriginal is calculated earlier
            KeyDisjunctCounter=0
            for _idx, _single_disjunction in model.component_data_iterindex(Disjunction, active=True):
                for _disjunct in _single_disjunction.parent_component()._disjuncts[_idx[1]]:
                    for _c in _disjunct.component_data_objects(Constraint, active=True):
                        ConstraintCounter +=1

            # FIXME: I think this is fragile: it assumes the key disjunction is a singleton.
            for _disjunct in key_disjunction._disjuncts.values()[0]:
                if _disjunct.active:
                    KeyDisjunctCounter +=1
            #=====================================================================

            # (3.d) basic step termination criteria                                       
            if abs(BasicStepObjectiveValue[-2]-BasicStepObjectiveValue[-1]) <= 0.01:
                if self.info: 
                    logger.info("GDP(hyra) Terminating due to minimum improvement")
                break
        
            elif BasicStepIteration >= 100:   
                if self.info: 
                    logger.info("GDP(hyra) Terminating Basic Steps: Iteration count >= 100")
                break
            elif ConstraintCounter > 2*ConstraintNumberoriginal: 
                if self.info: 
                    logger.info("GDP(hyra) Terminating Basic Steps: # Constraints > 2 * # Original")
                break  
            elif KeyDisjunctCounter>0.5*DisjunctNumberoriginal:
                if self.info: 
                    logger.info("GDP(hyra) Terminating Basic Steps: # Key Disjuncts > 1/2 # original disjuncts")
                break

        # (4)
        #
        # loop through the key disjuncts and add a "_global_constraints"
        # ConstraintList to each one
        #
        # loop through all (global) constraints in the model.  for each
        # constraint, if the constraint shares a variable with the key
        # disjunction, deactivate it and copy it into ALL of the key
        # disjuncts of the key disjunction by adding it to each of the
        # _global_constraints ConstraintLists

        for _disjunct in model._basic_step_hybrid.key_disjunct.itervalues():
            _disjunct._global_constraints = ConstraintList()
        
        for _single_glob_constraint in model.component_data_objects(Constraint, active=True):
            _constraint_vars = set([id(x) for x in identify_variables(_single_glob_constraint.body, include_fixed=False)])
            _common_vars = _vars_by_disjunction[key_disjunction_id].intersection(_constraint_vars )
            if not _common_vars:
                continue
            for _disjunct in model._basic_step_hybrid.key_disjunct.itervalues():
                if _single_glob_constraint.body.is_expression():
                    _body = _single_glob_constraint.body.clone()
                else:
                    _body = _single_glob_constraint.body
                    
                _disjunct._global_constraints.add(
                    ( _single_glob_constraint.lower,
                      _body,
                      _single_glob_constraint.upper ) )
                _single_glob_constraint.deactivate()

        # (5)
        TransformationFactory('gdp.chull').apply_to(model, targets=key_disjunction)
        TransformationFactory('gdp.bigm').apply_to(model)

        #print 'The presolve time'
        if self.info: 
            logger.info("GDP(hyra) total transformation time (seconds):  %0.2f"
                        % ( time.time() - self.initialTime, ))
            logger.info("GDP(hyra) total time in solver calls (seconds): %0.2f"
                        % ( self.timeInSolver, ))
        return model



class HybridReformulationAlgorithm_CuttingPlanes(Transformation):

    alias('gdp.hyra-cp', doc="")

    def __init__(self):
        super(HybridReformulationAlgorithm_CuttingPlanes, self).__init__()
        self.timeInSolver = 0
        self.initialTime = 0
        self.info = False
        self.WITH_CHARACTERISTIC_VALUES = False
        self.AbsoluteImprovement = 0.001
        self.RelativeImprovement = 0.1
        self.NumberOfBasicSteps = 1

    def _solve_model(self, m):

        results = self.solver.solve(m)
        ss = results.solver.status 
        tc = results.solver.termination_condition

        self.timeInSolver += results['Solver'][0]['Time']

        if ss == SolverStatus.ok and tc in _acceptable_termination_conditions:
            #m.load(results)
            return None, None
        elif tc in _infeasible_termination_conditions:
            return 'INFEASIBLE', results
        else:
            return 'NONOPTIMAL', results
    
    def _evaluate_disjunct_lp_relaxation(self, model):
        _infeasible_disjuncts = []
        _LP_values = {}
        _characteristic_value = {}

        all_disjunctions = model.component_data_objects(
            Disjunction, active=True, descend_into=(Block, Disjunct), 
            descent_order=TraversalStrategy.PostfixDepthFirstSearch )

        for _single_disjunction in all_disjunctions:
            # (1.b)
            tmp_model = model.clone()
            _tmp_single_disjunction = ComponentUID(
                _single_disjunction ).find_component(tmp_model)

            # (1.a) Deactivate the disjunction and the corresponding
            # disjuncts (*TODO: make the reformulations ignore
            # deactivated blocks!)
            _tmp_single_disjunction.deactivate()
            if self.info: 
                logger.info( 'GDP(hyra) Disjunction %s formed across:' % 
                             _single_disjunction.cname(True) )

            _active_disjuncts = []
            for _disjunct in _tmp_single_disjunction.parent_component()._disjuncts[_single_disjunction.index()]:
                if _disjunct.active:
                    _disjunct.deactivate()
                    _disjunct.indicator_var.fix(0)
                    _active_disjuncts.append(_disjunct)
                    if self.info: 
                        logger.info("GDP(hyra) " + ' '*3 + _disjunct.cname(True))
                else:
                    if self.info: 
                        logger.info("GDP(hyra) " + ' '*3 + _disjunct.cname(True) + ' [inactive]')

            TransformationFactory('gdp.chull').apply_to(tmp_model)

            # (1.c) TODO: reimplement as a call to a relaxation transformation
            # TransformationFactory('relax_binary').apply(tmp_model,
            #     in_place=True)
            relax_discrete_vars(tmp_model)

            # (1.d)
            for _disjunct in _active_disjuncts:
                tmp_model._tmp_basic_step = Block()

                # models may directly reference the indicator variable
                # outside the disjunct.  We need to remember it here so
                # that we can fix it to 1 here, and then at the end fix
                # it back to 0.
                tmp_indicator_var = _disjunct.indicator_var

                # (1.d.i)
                tmp_indicator_var.fix(1)
                for _name, _comp in _disjunct.component_map(active=True).iteritems():
                    _disjunct.del_component(_name)
                    tmp_model._tmp_basic_step.add_component(_name, _comp)

                # (1.d.ii)
                err, results = self._solve_model(tmp_model)
                if err:
                    _src_disjunct = ComponentUID(_disjunct).find_component(model)
                    if self.DETERMINISTIC_ALGORITHM:
                        _infeasible_disjuncts.append( _src_disjunct )
                    else:
                        _src_disjunct.deactivate()
                    _obj = None
                    if self.info: 
                        logger.info("GDP(hyra) " + ' '*6 + "%s: %s" % (_disjunct.cname(True), err) )
                    if err != 'INFEASIBLE':
                        logger.warning("GDP(hyra) subsolve returned error:\n%s" % ( results.solver, ))
                else:
                    _obj = value( list(tmp_model.component_data_objects(Objective, active=True))[-1] )
                    _LP_values[id( ComponentUID(_disjunct).find_component_on(
                        model) )] = _obj
                    _characteristic_value[ id(_single_disjunction) ] = \
                        min( _characteristic_value.get(
                            id(_single_disjunction), _obj ), _obj ) 
                    if self.info: 
                        logger.info("GDP(hyra) " + ' '*6 + "%s: %s" % (_disjunct.cname(True), _obj) )


                tmp_indicator_var.fix(0)
                tmp_model.del_component('_tmp_basic_step')

            if self.info: 
                logger.info("GDP(hyra) characteristic value %s" % _characteristic_value.get(
                    id(_single_disjunction), None ) )
            tmp_model = None 
            # suggestion move this inside the loop - by Mahdi
            #
            # [JDS 7/27] The above is fine inside the loop, although not
            # strictly necessary (the previous model will be implicitly
            # released by the next iteration
        return _characteristic_value, _LP_values, _infeasible_disjuncts


    def _apply_basic_step( self, model, scratch, _W_by_disjunction, 
                           _characteristic_value, _vars_by_disjunction, _disjunction_by_id,
                           disallowed ):
        # (2.b)
        _max_W = max( v for k,v in iteritems(_W_by_disjunction)
                      if k not in disallowed )
        _max_W_id = [ k for k,v in _W_by_disjunction.iteritems() if v == _max_W and k not in disallowed ]
        if self.WITH_CHARACTERISTIC_VALUES:
            _max_char_vals = max( _characteristic_value[k] for k in _max_W_id )
            key_disjunction_id = [ k for k in _max_W_id 
                                   if _characteristic_value[k] == _max_char_vals ][0]
        else:
            key_disjunction_id = _max_W_id[0]
        key_disjunction = _disjunction_by_id[key_disjunction_id]
        disallowed.add( key_disjunction_id )

        if self.info: 
            logger.info("GDP(hyra) Calculating basic steps...")

        # (3)
        BasicStepIteration= 0
        if self.WITH_CHARACTERISTIC_VALUES:
            BasicStepObjectiveValue=[_characteristic_value[key_disjunction_id]]

        #while True:
        BasicStepIteration += 1
        if self.info: 
            logger.info("GDP(hyra)  ...basic step iteration %s" % BasicStepIteration)
        # (3.a)
        for k in _W_by_disjunction:
            _W_by_disjunction[k] = 0
        _self = key_disjunction_id  
        for _other in _vars_by_disjunction:
            if _other == _self:
                continue
            _common_vars = _vars_by_disjunction[_self].intersection(
                    _vars_by_disjunction[_other] )
            if len(_common_vars) > 0:
                _delta_w = 1. / (
                    len(_vars_by_disjunction[_other]) * 
                    len(_vars_by_disjunction[_self]) )
                _W_by_disjunction[_other] += _delta_w

        _max_W = max( v for k,v in iteritems(_W_by_disjunction) 
                      if k not in disallowed )
        _max_W_id = [ k for k,v in _W_by_disjunction.iteritems() 
                      if v == _max_W and k not in disallowed ]
        if self.WITH_CHARACTERISTIC_VALUES:
            _max_char_vals = max( _characteristic_value[k] for k in _max_W_id )
            target_disjunction_id = [ k for k in _max_W_id if _characteristic_value[k] == _max_char_vals ][0]
        else:
            target_disjunction_id = [ k for k in _max_W_id ][0]
        target_disjunction = _disjunction_by_id[target_disjunction_id]
        disallowed.add( target_disjunction_id )

        # (3.b)
        _key_disjunction_index = None if key_disjunction.parent_component() is key_disjunction else key_disjunction.index()
        _key_disjuncts = [x for x in key_disjunction.parent_component()._disjuncts[_key_disjunction_index] if x.active]
        _target_disjunction_index = None if target_disjunction.parent_component() is target_disjunction else target_disjunction.index()
        _target_disjuncts = [x for x in target_disjunction.parent_component()._disjuncts[_target_disjunction_index] if x.active]


        i = 0
        key_disjunction.deactivate()
        for _disjunct in key_disjunction.parent_component()._disjuncts[_key_disjunction_index]:
            _disjunct.deactivate()
            _iv = _disjunct.indicator_var
            _disjunct.del_component(_iv)
            _disjunct.indicator_var_name = 'basic_step_%i_key_iv_%s' % ( BasicStepIteration, i )
            scratch.add_component(_disjunct.indicator_var_name, _iv)
            i += 1
        i = 0
        target_disjunction.deactivate()
        for _disjunct in target_disjunction.parent_component()._disjuncts[_target_disjunction_index]:
            _disjunct.deactivate()
            _iv = _disjunct.indicator_var
            _disjunct.del_component(_iv)
            _disjunct.indicator_var_name = 'basic_step_%i_target_iv_%s' % ( BasicStepIteration, i )
            scratch.add_component(_disjunct.indicator_var_name, _iv)
            i += 1
        #target_disjunction = None


        if scratch.component('key_disjunct') is not None:
            scratch.del_component('key_disjunct')
            scratch.del_component('key_disjunct_index')
            scratch.del_component('key_disjunction')
        scratch.key_disjunct = Disjunct(range(len(_key_disjuncts)*len(_target_disjuncts)))
        for k in range(len(_key_disjuncts)):
            _k_iv = getattr( scratch, _key_disjuncts[k].indicator_var_name )
            for t in range(len(_target_disjuncts)):
                _t_iv = getattr( scratch, _target_disjuncts[t].indicator_var_name )
                if self.info: 
                    logger.info("GDP(hyra) constructing key disjunct %s x %s" % (k, t))
                new_disjunct =  scratch.key_disjunct[k*len(_target_disjuncts)+t]
                tmp = _key_disjuncts[k].clone()
                for name,comp in list( tmp.component_map(active=True).iteritems() ):
                    #if name is 'indicator_var':
                    #    continue
                    tmp.del_component(name)
                    new_disjunct.add_component(name+'_k', comp)
                tmp = _target_disjuncts[t].clone()
                for name,comp in list( tmp.component_map(active=True).iteritems() ):
                    #if name is 'indicator_var':
                    #    continue
                    tmp.del_component(name)
                    new_disjunct.add_component(name+'_t', comp)

                scratch.indicator_var_maps.add(
                    new_disjunct.indicator_var <= _k_iv )
                scratch.indicator_var_maps.add(
                    new_disjunct.indicator_var <= _t_iv )
                scratch.indicator_var_maps.add(
                    new_disjunct.indicator_var >= _k_iv + _t_iv - 1 )

        key_disjunction \
            = scratch.key_disjunction \
            = Disjunction(expr=scratch.key_disjunct.values())

        # add the new key disjunction (after basic step) to our data structures (for next pass)
        key_disjunction_id = id(key_disjunction)
        _disjunction_by_id[key_disjunction_id] = key_disjunction
        _vars_by_disjunction[key_disjunction_id] = _vars_by_disjunction[_self].union(
                _vars_by_disjunction[target_disjunction_id])
        del _vars_by_disjunction[_self]
        _W_by_disjunction[key_disjunction_id] = _W_by_disjunction.pop(_self)

        del _vars_by_disjunction[target_disjunction_id]
        del _W_by_disjunction[target_disjunction_id]
        del _disjunction_by_id[target_disjunction_id]

        # (3.c; F.T.#4)

        ### Testing/removing infeasible Disjuncts in the new Key Disjunction  
        tmp_model = model.clone()
        _single_disjunction = key_disjunction
        _tmp_single_disjunction = ComponentUID(_single_disjunction).find_component_on(tmp_model)
        # Deactivate the disjunction and the corresponding disjuncts (*TODO: make the reformulations ignore deactivated blocks!)
        _tmp_single_disjunction.deactivate()
        if self.info: 
            logger.info("GDP(hyra) Disjunction %s formed across:" % _tmp_single_disjunction.cname(True))
        # I am really not sure why _tmp_single_disjunction.index isn't defined!
        _active_disjuncts = [d for d in _tmp_single_disjunction.parent_component()._disjuncts.values()[0] if d.active ]
        for _disjunct in _active_disjuncts:
            if self.info: 
                logger.info("GDP(hyra)    %s" % ( _disjunct.cname(True), ))
            _disjunct.deactivate()
            _disjunct.indicator_var.fix(0)

        TransformationFactory('gdp.chull').apply_to(tmp_model)

        # (To-Do)
        # TransformationFactory('relax_binary').apply(tmp_model, in_place=True)
        relax_discrete_vars(tmp_model)

        if self.WITH_CHARACTERISTIC_VALUES:
            _char_values=[]
            tmp_model._tmp_basic_step = Block(ComponentUID(scratch).find_component_on(tmp_model).key_disjunct.index_set())
            # I am really not sure why _tmp_single_disjunction.index isn't defined!
            for _disjunct in _active_disjuncts:
                _main_disjunct = ComponentUID(_disjunct).find_component(model)
                idx = _disjunct.index()
                tmp_indicator_var = _disjunct.indicator_var
                tmp_indicator_var.fix(1)
                tmp_model._tmp_basic_step._data[idx] = _disjunct.parent_component()._data.pop(idx)
                _disjunct._component = weakref.ref(tmp_model._tmp_basic_step)
                _disjunct.activate()

                err, results = self._solve_model(tmp_model)
                if err:
                    _main_disjunct.deactivate()
                    _main_disjunct.indicator_var.fix(0)
                    if self.info: 
                        logger.info("GDP(hyra) " + ' '*6 + "Testing feasibility for disjunct %s: %s" 
                                    % (_disjunct.cname(True), err) )
                else:
                    _objectives = tmp_model.component_map(Objective, active=True).values()
                    if len(_objectives) != 1:
                        raise RuntimeError("I am confused: I couldn't find exactly one active objective")
                    obj_value = value(_objectives[0])
                    if self.info: 
                        logger.info("GDP(hyra) " + ' '*6 + "Testing feasibility for disjunct %s: %s" 
                                    % (_disjunct.cname(True), obj_value) )

                    if self.WITH_CHARACTERISTIC_VALUES:
                        if id(_single_disjunction) not in _characteristic_value:
                            _characteristic_value[id(_single_disjunction)] = obj_value
                        else:
                            _characteristic_value[id(_single_disjunction)] = min(_characteristic_value[id(_single_disjunction)], obj_value)
                        _char_values.append(obj_value)

                # (cleanup)
                tmp_indicator_var.fix(0)
                del tmp_model._tmp_basic_step._data[idx]

            BasicStepObjectiveValue.append(min(_char_values))
            if self.info: 
                logger.info("GDP(hyra) Basic Steps Objective Value %s" % BasicStepObjectiveValue)

        tmp_model = None


        #=====================================================================
        # Updating the size of the problem, needed for terminations criteria  
        # ConstraintNumberoriginal is calculated earlier
        ConstraintCounter=0
        # DisjunctNumberoriginal is calculated earlier
        KeyDisjunctCounter=0
        for _idx, _single_disjunction in model.component_data_iterindex(Disjunction, active=True):
            for _disjunct in _single_disjunction.parent_component()._disjuncts[_idx[1]]:
                for _c in _disjunct.component_data_objects(Constraint, active=True):
                    ConstraintCounter +=1

        # FIXME: I think this is fragile: it assumes the key disjunction is a singleton.
        for _disjunct in key_disjunction._disjuncts.values()[0]:
            if _disjunct.active:
                KeyDisjunctCounter +=1
        #=====================================================================


        # (4)
        #
        # loop through the key disjuncts and add a "_global_constraints"
        # ConstraintList to each one
        #
        # loop through all (global) constraints in the model.  for each
        # constraint, if the constraint shares a variable with the key
        # disjunction, deactivate it and copy it into ALL of the key
        # disjuncts of the key disjunction by adding it to each of the
        # _global_constraints ConstraintLists

        for _disjunct in scratch.key_disjunct.itervalues():
            _disjunct._global_constraints = ConstraintList()
        
        for _single_glob_constraint in model.component_data_objects(Constraint, active=True):
            _constraint_vars = dict((id(x),x) for x in identify_variables(_single_glob_constraint.body, include_fixed=False))
            _common_vars = _vars_by_disjunction[key_disjunction_id].intersection( list(iterkeys(_constraint_vars)) )
            if not _common_vars:
                continue
            for _disjunct in scratch.key_disjunct.itervalues():
                if _single_glob_constraint.body.is_expression():
                    _body = _single_glob_constraint.body.clone()
                else:
                    _body = _single_glob_constraint.body
                    
                _disjunct._global_constraints.add(
                    ( _single_glob_constraint.lower,
                      _body,
                      _single_glob_constraint.upper ) )
                _single_glob_constraint.deactivate()


    def _apply_to(self, base_model, **kwds): 
        self.initialTime = time.time()
        self.timeInSolver = 0
        self.info = bool( logger.isEnabledFor(logging.INFO) )

        solver_name = kwds.get('options',{}).get('solver', 'bonmin')
        self.solver = SolverFactory(solver_name)#, solver_io='python')
        if self.solver is None:
            raise RuntimeError("Unknown solver %s specified for the gdp.hyra transformation" % solver_name)

        model = base_model.clone()

        base_separation_vars = {}
        # HACK: explicitly grab all indicator variables
        for _single_disjunct in base_model.component_data_objects(Disjunct, descend_into=(Disjunct,Block), active=True):
            base_separation_vars[id(_single_disjunct.indicator_var)] = _single_disjunct.indicator_var
        for _single_glob_constraint in base_model.component_data_objects(Constraint, descend_into=(Disjunct,Block), active=True):
            base_separation_vars.update(
                dict((id(x),x) for x in identify_variables(_single_glob_constraint.body, include_fixed=False))
            )
        base_separation_vars = list( itervalues(base_separation_vars) )
        hull_separation_vars = [ model.find_component(x) for x in base_separation_vars ]
        for i,v in enumerate(hull_separation_vars):
            if v is None:
                print("Missing: %s" % base_separation_vars[i].cname(True))
        #print hull_separation_vars
        #print base_separation_vars


        # If set to "TRUE", the algorithm first calculates all the
        # characteristic values, then return and remove the infeasible
        # disjuncts
        #
        # If set to "FALSE", the algorithm removes the infeasible terms
        # as characteristic values are being calculated; as a result,
        # the values of characteristic values depend on the order that
        # disjuncts are visited, and is path dependant
        self.DETERMINISTIC_ALGORITHM = False

        if self.WITH_CHARACTERISTIC_VALUES:
            _characteristic_value, _LP_values, _disjuncts_to_deactivate \
                = self._evaluate_disjunct_lp_relaxation(model)

            # If we defer deactivating infeasible disjuncts until after all
            # characteristic values are calculated, then we need to go back and
            # deactivate them now
            for _disjunct in _disjuncts_to_deactivate:
                _disjunct.deactivate()
        else:
            _characteristic_value = {}

        #=====================================================================
        # Measuring the size of the problem, needed for terminations criteria  
        # ConstraintNumberoriginal need to be calculated before Basic Steps
        ConstraintNumberoriginal=0
        # DisjunctNumberoriginal need to be calculated before Basic Steps
        DisjunctNumberoriginal=0

        for _idx, _single_disjunction in model.component_data_iterindex(Disjunction, active=True):
            for _disjunct in _single_disjunction.parent_component()._disjuncts[_idx[1]]:
                if not _disjunct.active:
                    continue
                DisjunctNumberoriginal +=1
                for c in _disjunct.component_data_objects(Constraint, active=True, descend_into=(Block, Disjunct)):
                    ConstraintNumberoriginal +=1
        #=====================================================================

        _disjunction_by_id = {}
        _vars_by_disjunction = {} 
        _W_by_disjunction = {}

        for _idx, _single_disjunction in model.component_data_iterindex(Disjunction, active=True):
            _disjunction_by_id[id(_single_disjunction)] = _single_disjunction
            _vars_by_disjunction[id(_single_disjunction)] = set()
            _W_by_disjunction[id(_single_disjunction)] = 0
            for _disjunct in _single_disjunction.parent_component()._disjuncts[_idx[1]]:
                if not _disjunct.active:
                    continue
                _all_con = _disjunct.component_data_objects(Constraint, active=True)
                for _single_constraint in _all_con:
                    _vars_by_disjunction[id(_single_disjunction)].update(
                        id(x) for x in identify_variables(
                            _single_constraint.body, include_fixed=False ) )

        # (2.a)
        _all_disjunctions = model.component_data_objects(Disjunction, active=True)
        for _single_disjunction in _all_disjunctions:
            _self = id(_single_disjunction)
            for _other in _vars_by_disjunction.iterkeys():
                if _other >= _self:
                    continue
                _common_vars = _vars_by_disjunction[_self].intersection(
                _vars_by_disjunction[_other] )
                if len(_common_vars) > 0:
                    _delta_w = 1. / (
                        len(_vars_by_disjunction[_other]) * 
                        len(_vars_by_disjunction[_self]) )
                    _W_by_disjunction[_other] += _delta_w
                    _W_by_disjunction[_self]  += _delta_w        

        # declare a block to hold the disjunctions and constraints that
        # we are about to create
        model._basic_step_hybrid = Block(range(self.NumberOfBasicSteps))
        selected_disjuncts = set()
        for i in range(self.NumberOfBasicSteps):
            model._basic_step_hybrid[i].indicator_var_maps = ConstraintList()
            self._apply_basic_step( model, model._basic_step_hybrid[i], 
                                    _W_by_disjunction, _characteristic_value, _vars_by_disjunction, _disjunction_by_id,
                                    selected_disjuncts )


        
        #
        # START CUTTING PLANES ALGORITHM HERE
        #

        # (a) Make BM reformulation of the original model
        #    - add place holder for future cuts
        TransformationFactory('gdp.bigm').apply_to(base_model)
        base_model.hyra_cutting_planes = ConstraintList()

        # (b) Make HR of the transformed model (after basic steps)
        #    - replace objective in HR with the separation problem objective
        #       -- note: minimum distance over ORIGINAL variables and not the HR disaggregated variables
        model.hull_separation_params = Param(range(len(hull_separation_vars)),
                                             mutable=True, initialize=0)
        TransformationFactory('gdp.chull').apply_to(model)
        for obj in model.component_data_objects(Objective, active=True, descend_into=(Block, Disjunct)):
            obj.deactivate()
        model.hyra_separation_objective = Objective(
            expr= sum( (model.hull_separation_params[i]-hull_separation_vars[i])**2
                       for i in range(len(hull_separation_vars)) ) )
        relax_discrete_vars(model)

        # Relax the Big-M model
        relaxed_vars = relax_discrete_vars(base_model)

        base_obj = list(base_model.component_data_objects(Objective, active=True, descend_into=(Block, Disjunct)))[0]
        while 1:
            # (c) solve the BM for LP relaxation value
            state, base_results = self._solve_model(base_model)
            if state:
                print("Relaxed Big-M solve failed: " + state)
            # TODO: check result for feasibility / error

            # (d) solve the separation problem from the solution of the BM
            #    - add a cut based on the sep problem to the BM model
            for i,v in enumerate(base_separation_vars):
                model.hull_separation_params[i] = value(v)
            state, sep_results = self._solve_model(model)
            if state:
                print("Separation problem solve failed: " + state)
                
            cut = sum( 2*( value(v) - value(model.hull_separation_params[i]) ) 
                       * ( base_separation_vars[i] - value(v) ) 
                       for i,v in enumerate(hull_separation_vars) ) >= 0
            base_model.hyra_cutting_planes.add(cut)
            #print "CUT", cut

            # (e) repeat (c)-(d) until "enough" cuts.
            sep = value(model.hyra_separation_objective)
            if sep < self.AbsoluteImprovement or sep/max(self.AbsoluteImprovement,abs(value(base_obj))) < self.RelativeImprovement:
                break

        # UNDO LP relaxation of the base_model
        for var, domain in itervalues(relaxed_vars):
            var.domain = domain

        if self.info: 
            logger.info("GDP(hyra) total transformation time (seconds):  %0.2f"
                        % ( time.time() - self.initialTime, ))
            logger.info("GDP(hyra) total time in solver calls (seconds): %0.2f"
                        % ( self.timeInSolver, ))
        return base_model

