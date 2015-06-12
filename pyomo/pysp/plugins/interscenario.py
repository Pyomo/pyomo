#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from six import iteritems
from six.moves import xrange

from pyomo.core import (
    minimize, value, TransformationFactory,
    ComponentUID, Block, Constraint, ConstraintList,
    Param, Var, Set, Objective, Suffix )
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.pysp import phextension
from pyomo.solvers.plugins.smanager.phpyro import SolverManager_PHPyro
from pyomo.util.plugin import SingletonPlugin, implements

from pyomo.pysp.phsolverserverutils import \
    transmit_external_function_invocation_to_worker

import logging
logger = logging.getLogger('pyomo.pysp')

_acceptable_termination_conditions = set([
    TerminationCondition.optimal,
    TerminationCondition.globallyOptimal,
    TerminationCondition.locallyOptimal,
])
_infeasible_termination_conditions = set([
    TerminationCondition.infeasible,
    TerminationCondition.invalidProblem,
])

def get_modified_instance(ph, scenario_tree, scenario_or_bundle):
    if scenario_or_bundle._name in get_modified_instance.data:
        return get_modified_instance.data[scenario_or_bundle._name]

    # Note: the var_ids are on the ORIGINAL scenario models 
    rootNode = scenario_tree.findRootNode()
    var_ids = list(rootNode._variable_datas.keys())

    # Find the model
    if scenario_tree.contains_bundles():
        base_model = ph._bundle_binding_instance_map[scenario_or_bundle._name]
    else:
        base_model = ph._instances[scenario_or_bundle._name]
    base_model._interscenario_plugin_cutlist = ConstraintList()

    # Now, make a copy for us to play with
    model = base_model.clone()
    get_modified_instance.data[scenario_or_bundle._name] = model

    model._interscenario_plugin = Block()

    # Right now, this is hard-coded for 2-stage problems - so we only
    # need to worry about the variables from the root node.  These
    # variables should exist on all scenarios.  Set up a (trivial)
    # equality constraint for each variable:
    #    var == current_value{param} + separation_variable{var, fixed=0}
    _cuid_buffer = {}
    model._interscenario_plugin.STAGE1VAR = _S1V = Set(initialize=var_ids)
    model._interscenario_plugin.separation_variables = _sep = Var( _S1V )
    model._interscenario_plugin.fixed_variable_values \
        = _param = Param( _S1V, mutable=True, initialize=0 )
    _sep.fix(0)

    def _set_var_value(b, i):
        # Note indexing: for each 1st stage var, pick an arbitrary
        # (first) scenario and return the variable (and not it's
        # probability)
        #print "looking for ", rootNode._variable_datas[i][0][0].cname(True)
        return _param[i] + _sep[i] == ComponentUID(
            rootNode._variable_datas[i][0][0], _cuid_buffer).find_component_on(
                b.model())
    model._interscenario_plugin.fixed_variables_constraint \
        = _con = Constraint( _S1V, rule=_set_var_value )

    # Move the objective to a standardized place so we can easily find it later
    _orig_objective = list( model.component_data_objects(
        Objective, active=True, descend_into=True ) )
    assert(len(_orig_objective) == 1)
    _orig_objective = _orig_objective[0]
    _orig_objective.parent_block().del_component(_orig_objective)
    model._interscenario_plugin.original_obj = _orig_objective
    # add (and deactivate) the objective for the infeasibility
    # separation problem
    model._interscenario_plugin.separation_obj = Objective(
        expr= sum( _sep[i]**2 for i in var_ids ) )
    model._interscenario_plugin.separation_obj.deactivate()

    # Make sure we get dual information
    if 'dual' not in model:
        # Export and import floating point data
        model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    if 'rc' not in model:
        model.rc = Suffix(direction=Suffix.IMPORT_EXPORT)

    return model

get_modified_instance.data = {}

def get_dual_values(solver, model):
    if id(model) not in get_dual_values.discrete_stage2_vars:
        # 1st attempt to get duals: we need to see if the model has
        # discrete variables (solvers won't give duals if there are
        # still active discrete variables)
        try:
            get_dual_values.discrete_stage2_vars[id(model)] = False
            return get_dual_values(solver, model)
        except:
            get_dual_values.discrete_stage2_vars[id(model)] = True
            # Find the discrete variables to populate the list
            return get_dual_values(solver, model)

    duals = {}
    _con = model._interscenario_plugin.fixed_variables_constraint

    if get_dual_values.discrete_stage2_vars[id(model)]:
        # Fix all discrete variables
        xfrm = TransformationFactory('core.fix_discrete')
        xfrm.apply_to(model)

        #SOLVE
        results = solver.solve(model)
        ss = results.solver.status 
        tc = results.solver.termination_condition
        #self.timeInSolver += results['Solver'][0]['Time']
        if ss == SolverStatus.ok and tc in _acceptable_termination_conditions:
            state = ''
        elif tc in _infeasible_termination_conditions:
            state = 'INFEASIBLE'
        else:
            state = 'NONOPTIMAL'
        if state:
            logger.warning("Resolving subproblem model with fixed second-stage "
                           "discrete variables failed (%s).  "
                           "Dual values not available." % (state,) )

        # Get the duals
        for varid in model._interscenario_plugin.STAGE1VAR:
            duals[varid] = model.dual[_con[varid]]
        # Free the discrete second-stage variables
        xfrm.apply_to(model, undo=True)
        
    else:
        # return the duals
        for varid in model._interscenario_plugin.STAGE1VAR:
            duals[varid] = model.dual.get(_con[varid], None)

    return duals
    
get_dual_values.discrete_stage2_vars = {}


def reset_modified_instance(ph, scenario_tree, scenario_or_bundle):
    get_modified_instance.data = {}
    get_dual_values.discrete_stage2_vars = {}


def solve_separation_problem(solver, model):
    model._interscenario_plugin.original_obj.deactivate()
    model._interscenario_plugin.separation_obj.activate()
    model._interscenario_plugin.separation_variables.unfix()

    #SOLVE
    results = solver.solve(model)
    ss = results.solver.status 
    tc = results.solver.termination_condition
    #self.timeInSolver += results['Solver'][0]['Time']
    if ss == SolverStatus.ok and tc in _acceptable_termination_conditions:
        state = ''
    elif tc in _infeasible_termination_conditions:
        state = 'INFEASIBLE'
    else:
        state = 'NONOPTIMAL'
    if state:
        logger.warning("Solving the interscenario cut separation subproblem "
                       "failed (%s)." % (state,) )

    _sep = model._interscenario_plugin.separation_variables
    _par = model._interscenario_plugin.fixed_variable_values
    cut = dict((vid, (2*value(_sep[vid])-value(_par[vid]), value(_sep[vid])))
               for vid in model._interscenario_plugin.STAGE1VAR)

    model._interscenario_plugin.original_obj.activate()
    model._interscenario_plugin.separation_obj.deactivate()
    model._interscenario_plugin.separation_variables.fix(0)
    return cut


def add_new_cuts(ph, scenario_tree, scenario_or_bundle, cutlist):
    # Find the model
    if scenario_tree.contains_bundles():
        base_model = ph._bundle_binding_instance_map[scenario_or_bundle._name]
    else:
        base_model = ph._instances[scenario_or_bundle._name]

    # Add the cuts to the ConstraintList
    cl = base_model._interscenario_plugin_cutlist
    for cut in cutlist:
        cl.add( sum(
            cutinfo[0]*(scenario_tree._variable_datas[i][0] - cutinfo[1]) 
            for i, cutinfo in iteritems(cut) ) >= 0 )


def solve_fixed_scenario_solutions( 
        ph, scenario_tree, scenario_or_bundle, scenario_solutions ):

    model = get_modified_instance(ph, scenario_tree, scenario_or_bundle)
    _param = model._interscenario_plugin.fixed_variable_values
    _sep = model._interscenario_plugin.separation_variables

    # We need to know which scenarios are local to this instance ... so
    # we don't waste time repeating work.
    if scenario_tree.contains_bundles():
        local_scenarios = scenario_or_bundle._scenario_names
    else:
        local_scenarios = [ scenario_or_bundle._name ]
    local_probability = scenario_or_bundle._probability

    # Solve each solution here and cache the resulting objective
    cutlist = []
    obj_values = []
    dual_values = []
    for var_values, scenario_list in scenario_solutions:
        local = False
        for scenario in local_scenarios:
            if scenario in scenario_list:
                local = True
                break
        if local:
            # Here is where we could save some time and not repeat work
            # ... for now I am being lazy and re-solving so that we get
            # the dual values, etc for this scenario as well.  If nothing
            # else, i makes averaging easier.
            pass

        assert( len(var_values) == len(_param) )
        for var_id, var_value in iteritems(var_values):
            _param[var_id] = var_value
        
        results = ph._solver.solve(model, tee=False)
        ss = results.solver.status 
        tc = results.solver.termination_condition
        #self.timeInSolver += results['Solver'][0]['Time']
        if ss == SolverStatus.ok and tc in _acceptable_termination_conditions:
            state = 0 #'FEASIBLE'
            #print "\nFEASIBLE", len(model.dual), len(model.rc), var_values
            obj_values.append( value(model._interscenario_plugin.original_obj) )
            dual_values.append( get_dual_values(ph._solver, model) )
        elif tc in _infeasible_termination_conditions:
            state = 1 #'INFEASIBLE'
            obj_values.append(None)
            dual_values.append(None)
            cut = solve_separation_problem(ph._solver, model)
            if cut is not None:
                cutlist.append(cut)
        else:
            state = 2 #'NONOPTIMAL'
            obj_values.append(None)
            dual_values.append(None)

    return obj_values, dual_values, local_probability, cutlist



class InterScenarioPlugin(SingletonPlugin):

    implements(phextension.IPHExtension) 

    def __init__(self):
        self.incumbent = None

    def pre_ph_initialization(self,ph):
        pass

    def post_instance_creation(self,ph):
        pass

    def post_ph_initialization(self, ph):
        if len(ph._scenario_tree._stages) > 2:
            raise RuntimeError(
                "InterScenarioPlugin only works with 2-stage problems" )
        self._sense_to_min = 1 if ph._objective_sense == minimize else -1

    def post_iteration_0_solves(self, ph):
        self._interscenario_plugin(ph)

    def post_iteration_0(self, ph):
        pass

    def pre_iteration_k_solves(self, ph):
        pass

    def post_iteration_k_solves(self, ph):
        pass

    def post_iteration_k(self, ph):
        pass

    def post_ph_execution(self, ph):
        pass

    def _interscenario_plugin(self,ph):
        # (1) Collect all scenario (first) stage variables
        unique_solutions = self._collect_unique_scenario_solutions(ph)

        # (2) Filter them to find a set we want to distribute
        pass

        # (3) Distribute (some) of the variable sets out to the
        # scenarios, fix, and resolve; Collect and return the
        # objectives, duals, and any cuts
        partial_obj_values, dual_values, probability, cuts \
            = self._solve_interscenario_solutions( ph, unique_solutions )

        # (4) distribute any cuts
        if cuts:
            self._distribute_cuts(ph, cuts)

        # (5) compute updated rho estimates
        pass

        # (6) set the new rho values
        pass

        # (7) compute and publish the new incumbent
        self._update_incumbent(partial_obj_values, probability, unique_solutions)

    def _collect_unique_scenario_solutions(self, ph):
        # list of (varmap, scenario_list) tuples
        unique_solutions = []

        # See ph.py:update_variable_statistics for a multistage version...
        rootNode = ph._scenario_tree.findRootNode()
        for scenario in rootNode._scenarios:
            found = False
            # Note: because we are looking for unique variable values,
            # then if the user is bundling, this will implicitly re-form
            # the bundles
            for _sol in unique_solutions:
                if scenario._x[rootNode._name] == _sol[0]:
                    _sol[1].append(scenario)
                    found = True
                    break
            if not found:
                unique_solutions.append( 
                    ( scenario._x[rootNode._name], [scenario] ) )           

        return unique_solutions


    def _solve_interscenario_solutions(self, ph, scenario_solutions):
        results = ([],[],[])
        cutlist = []
        if not isinstance( ph._solver_manager, SolverManager_PHPyro ):

            if ph._scenario_tree.contains_bundles():
                subproblems = ph._scenario_tree._scenario_bundles
            else:
                subproblems = ph._scenario_tree._scenarios

            for problem in subproblems:
                _tmp = solve_fixed_scenario_solutions(
                    ph, ph._scenario_tree, problem, scenario_solutions )
                for i,r in enumerate(results):
                    r.append(_tmp[i])
                cutlist.extend(_tmp[-1])
        else:
            action_handles = transmit_external_function_invocation(
                ph,
                'pyomo.pysp.plugins.interscenario',
                'solve_fixed_scenario_solutions',
                return_action_handles = True,
                function_args=scenario_solutions )

            num_results_so_far = 0
            num_results = len(action_handles)
            while (num_results_so_far < num_results):
                _ah = ph._solver_manager.wait_any()
                _tmp = ph._solver_manager.get_results(_ah)
                for i,r in enumerate(results):
                    r.append(_tmp[i])
                cutlist.extend(_tmp[-1])
                num_results_so_far += 1

        return results + (cutlist,)


    def _distribute_cuts(self, ph, cutlist):
        if not isinstance( ph._solver_manager, SolverManager_PHPyro ):

            if ph._scenario_tree.contains_bundles():
                subproblems = ph._scenario_tree._scenario_bundles
            else:
                subproblems = ph._scenario_tree._scenarios

            for problem in subproblems:
                add_new_cuts( ph, ph._scenario_tree, problem, cutlist )
        else:
            action_handles = transmit_external_function_invocation(
                ph,
                'pyomo.pysp.plugins.interscenario',
                'add_new_cuts',
                return_action_handles = True,
                function_args=cutlist )

            solver_manager.wait_all(action_handles)


    def _update_incumbent(self, partial_obj_values, probability, unique_solns):
        obj_values = []
        for soln_id in xrange(len( unique_solns )):
            obj = 0.
            for scen_or_bundle_id, p in enumerate(probability):
                obj += p * partial_obj_values[scen_or_bundle_id][soln_id]
            obj_values.append(obj * self._sense_to_min)

        best_obj = min(obj_values)
        if self.incumbent is not None and \
           self.incumbent[0] * self._sense_to_min > best_obj:
            return

        # New incumbent!
        _id = obj_values.index(best_obj)
        self.incumbent = ( best_obj * self._sense_to_min, unique_solns[_id] )
        logger.info("New incumbent: %s" % (self.incumbent[0],))
