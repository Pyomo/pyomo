#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import math
from six import iteritems
from six.moves import xrange
import weakref

from pyomo.core import (
    minimize, value, TransformationFactory,
    ComponentUID, Block, Constraint, ConstraintList,
    Param, Var, Set, Objective, Suffix )
from pyomo.opt import (
    SolverFactory, SolverStatus, TerminationCondition, ProblemFormat )
from pyomo.pysp import phextension
from pyomo.solvers.plugins.smanager.phpyro import SolverManager_PHPyro
from pyomo.util.plugin import SingletonPlugin, implements

from pyomo.repn.compute_ampl_repn import preprocess_block_constraints \
    as ampl_preprocess_block_constraints
from pyomo.repn.compute_canonical_repn import preprocess_block_constraints \
    as canonical_preprocess_block_constraints

from pyomo.pysp.phsolverserverutils import \
    transmit_external_function_invocation_to_worker
from pyomo.pysp.convergence import TermDiffConvergence

import logging
logger = logging.getLogger('pyomo.pysp')

ALLOW_VARIABLE_SLACK = False

_acceptable_termination_conditions = set([
    TerminationCondition.optimal,
    TerminationCondition.globallyOptimal,
    TerminationCondition.locallyOptimal,
])
_infeasible_termination_conditions = set([
    TerminationCondition.infeasible,
    TerminationCondition.invalidProblem,
])

def get_modified_instance(ph, scenario_tree, scenario_or_bundle, epsilon=None):
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
    base_model._interscenario_plugin = Block()
    base_model._interscenario_plugin.cutlist = ConstraintList()

    # Now, make a copy for us to play with
    model = base_model.clone()
    get_modified_instance.data[scenario_or_bundle._name] = model

    # Right now, this is hard-coded for 2-stage problems - so we only
    # need to worry about the variables from the root node.  These
    # variables should exist on all scenarios.  Set up a (trivial)
    # equality constraint for each variable:
    #    var == current_value{param} + separation_variable{var, fixed=0}
    model._interscenario_plugin.epsilon = Param(initialize=epsilon)
    model._interscenario_plugin.STAGE1VAR = _S1V = Set(initialize=var_ids)
    model._interscenario_plugin.separation_variables \
        = _sep = Var( _S1V )
    model._interscenario_plugin.fixed_variable_values \
        = _param = Param( _S1V, mutable=True, initialize=0 )

    if ALLOW_VARIABLE_SLACK:
        for idx in _sep:
            _sep[idx].setlb(-epsilon)
            _sep[idx].setub(epsilon)
    else:
        _sep.fix(0)

    _cuid_buffer = {}
    _base_src = base_model._interscenario_plugin.local_stage1_varmap = {}
    _src = model._interscenario_plugin.local_stage1_varmap = {}
    for i in _S1V:
        # Note indexing: for each 1st stage var, pick an arbitrary
        # (first) scenario and return the variable (and not it's
        # probability)
        src_VAR = ComponentUID(
            rootNode._variable_datas[i][0][0], _cuid_buffer).find_component_on(
                model)
        _src[i] = weakref.ref(src_VAR)

        src_VAR = ComponentUID(
            rootNode._variable_datas[i][0][0], _cuid_buffer).find_component_on(
                base_model)
        _base_src[i] = weakref.ref(src_VAR)

    def _set_var_value(b, i):
        return _param[i] + _sep[i] - _src[i]() == 0
    model._interscenario_plugin.fixed_variables_constraint \
        = _con = Constraint( _S1V, rule=_set_var_value )

    #
    # Note: while the objective has already been modified to include
    # placeholders for the proximal terms, they are set a "0" values -
    # so by cloning the model now, we are in effect getting a clone of
    # the original unmodified deterministic scenario objective.
    #
    # TODO: When we get the duals of the first-stage variables, do we
    # want the dual WRT the original objective, or the dual WRT the
    # augmented objective?
    #
    # Move the objective to a standardized place so we can easily find it later
    _orig_objective = list( model.component_data_objects(
        Objective, active=True, descend_into=True ) )
    assert(len(_orig_objective) == 1)
    _orig_objective = _orig_objective[0]
    _orig_objective.parent_block().del_component(_orig_objective)
    model._interscenario_plugin.original_obj = _orig_objective
    # add (and deactivate) the objective for the infeasibility
    # separation problem.
    model._interscenario_plugin.separation_obj = Objective(
        expr= sum( _sep[i]**2 for i in var_ids ),
        sense = minimize )
    model._interscenario_plugin.separation_obj.deactivate()

    # Make sure we get dual information
    if 'dual' not in model:
        # Export and import floating point data
        model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    #if 'rc' not in model:
    #    model.rc = Suffix(direction=Suffix.IMPORT_EXPORT)

    model.preprocess()
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
        xfrm = TransformationFactory('core.relax_discrete')
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
        else:
            # Get the duals
            model.solutions.load_from(results)
            #model.dual.pprint()
            for varid in model._interscenario_plugin.STAGE1VAR:
                duals[varid] = model.dual[_con[varid]]
        # Free the discrete second-stage variables
        xfrm.apply_to(model, undo=True)
        
    else:
        # return the duals
        for varid in model._interscenario_plugin.STAGE1VAR:
            duals[varid] = model.dual[_con[varid]]

    return duals
    
get_dual_values.discrete_stage2_vars = {}


def reset_modified_instance(ph, scenario_tree, scenario_or_bundle):
    get_modified_instance.data = {}
    get_dual_values.discrete_stage2_vars = {}


def solve_separation_problem(solver, model):
    xfrm = TransformationFactory('core.relax_discrete')
    xfrm.apply_to(model)

    model._interscenario_plugin.original_obj.deactivate()
    model._interscenario_plugin.separation_obj.activate()
    #model._interscenario_plugin.separation_variables.unfix
    _par = model._interscenario_plugin.fixed_variable_values
    _sep = model._interscenario_plugin.separation_variables
    if ALLOW_VARIABLE_SLACK:
        epsilon = value(model._interscenario_plugin.epsilon)
        for idx in _sep:
            _sep[idx].setlb(None)
            _sep[idx].setub(None)
    else:
        _sep.unfix()

    #SOLVE
    results = solver.solve(model)
    ss = results.solver.status 
    tc = results.solver.termination_condition
    #self.timeInSolver += results['Solver'][0]['Time']
    if ss == SolverStatus.ok and tc in _acceptable_termination_conditions:
        state = ''
        model.solutions.load_from(results)
    elif tc in _infeasible_termination_conditions:
        state = 'INFEASIBLE'
    else:
        state = 'NONOPTIMAL'
    if state:
        logger.warning("Solving the interscenario cut separation subproblem "
                       "failed (%s)." % (state,) )
        ans = None
    else:
        cut = dict((vid, (value(_sep[vid]), value(_par[vid])))
                   for vid in model._interscenario_plugin.STAGE1VAR)
        obj = value(model._interscenario_plugin.separation_obj)
        ans = (obj, cut)

    model._interscenario_plugin.original_obj.activate()
    model._interscenario_plugin.separation_obj.deactivate()
    #model._interscenario_plugin.separation_variables.fix(0)
    if ALLOW_VARIABLE_SLACK:
        for idx in _sep:
            _sep[idx].setlb(-epsilon)
            _sep[idx].setub(epsilon)
    else:
        _sep.fix(0)

    xfrm.apply_to(model, undo=True)
    return ans


def add_new_cuts(ph, scenario_tree, scenario_or_bundle, cutlist):
    # Find the model
    if scenario_tree.contains_bundles():
        base_model = ph._bundle_binding_instance_map[scenario_or_bundle._name]
    else:
        base_model = ph._instances[scenario_or_bundle._name]

    model = get_modified_instance(ph, scenario_tree, scenario_or_bundle)
    epsilon = value(model._interscenario_plugin.epsilon)

    # Add the cuts to the ConstraintList on the original and modified models
    for m in (base_model, model):
        _cl = m._interscenario_plugin.cutlist
        _src = m._interscenario_plugin.local_stage1_varmap
        for cut_obj, cut in cutlist:
            _cl.add( sum(
                2 * (_sep*(1.-epsilon))
                  * (_src[i]() - (_par+_sep*(1.-epsilon)))
                for i, (_sep, _par) in iteritems(cut) 
                if abs(_sep) > epsilon
            ) >= 0 )

        #m.preprocess()
        if ph._solver.problem_format() == ProblemFormat.nl:
            ampl_preprocess_block_constraints(m._interscenario_plugin)
        else:
            _map = {}
            canonical_preprocess_block_constraints(m._interscenario_plugin,_map)


def solve_fixed_scenario_solutions( 
        ph, scenario_tree, scenario_or_bundle, scenario_solutions, epsilon ):

    model = get_modified_instance(ph, scenario_tree, scenario_or_bundle, epsilon)
    _block = model._interscenario_plugin
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
        
        # TODO: We only need to update the CanonicalRepn for the binding
        # constraints ... so we could save a LOT of time by not
        # preprocessing the whole model.
        #
        #model.preprocess()
        if ph._solver.problem_format() == ProblemFormat.nl:
            ampl_preprocess_block_constraints(_block)
        else:
            var_id_map = {}
            canonical_preprocess_block_constraints(_block, var_id_map)

        results = ph._solver.solve(model)
        ss = results.solver.status 
        tc = results.solver.termination_condition
        #self.timeInSolver += results['Solver'][0]['Time']
        if ss == SolverStatus.ok and tc in _acceptable_termination_conditions:
            state = 0 #'FEASIBLE'
            model.solutions.load_from(results)
            obj_values.append( value(model._interscenario_plugin.original_obj) )
            dual_values.append( get_dual_values(ph._solver, model) )
            cutlist.append(None)
        elif tc in _infeasible_termination_conditions:
            state = 1 #'INFEASIBLE'
            obj_values.append(None)
            dual_values.append(None)
            cutlist.append( solve_separation_problem(ph._solver, model) )
        else:
            state = 2 #'NONOPTIMAL'
            obj_values.append(None)
            dual_values.append(None)
            cutlist.append(None)

    return obj_values, dual_values, local_probability, cutlist



class InterScenarioPlugin(SingletonPlugin):

    implements(phextension.IPHExtension) 

    def __init__(self):
        self.incumbent = None
        self.rho = None
        self.x_deviation = None
        self.lastConvergenceMetric = None
        self.cutlist = []
        self.lastRun = 0

        self.epsilon = 1e-6
        self.convergenceRelativeDegredation = 0.33
        self.convergenceAbsoluteDegredation = 0.001
        self.iterationInterval = 10
        # multiplier on computed rho values
        self.rhoScale = 0.99
        # How quickly rho moves to new values [0-1: 0-never, 1-instantaneous]
        self.rhoDamping = 0.1

        self.converger = TermDiffConvergence()

    def pre_ph_initialization(self,ph):
        pass

    def post_instance_creation(self,ph):
        pass

    def post_ph_initialization(self, ph):
        if len(ph._scenario_tree._stages) > 2:
            raise RuntimeError(
                "InterScenario plugin only works with 2-stage problems" )
        self._sense_to_min = 1 if ph._objective_sense == minimize else -1
        self.rho = dict((v,ph._rho) for v in ph._scenario_tree.findRootNode()._xbars)

    def post_iteration_0_solves(self, ph):
        self._interscenario_plugin(ph)

    def post_iteration_0(self, ph):
        self.converger.update( ph._current_iteration,
                               ph,
                               ph._scenario_tree,
                               ph._instances )
        self.lastConvergenceMetric = self.converger.lastMetric()
        pass

    def pre_iteration_k_solves(self, ph):
        if self.cutlist:
            self._distribute_cuts(ph, self.cutlist)
            self.cutlist = []
        pass

    def post_iteration_k_solves(self, ph):
        self.converger.update( ph._current_iteration,
                               ph,
                               ph._scenario_tree,
                               ph._instances )
        curr = self.converger.lastMetric()
        last = self.lastConvergenceMetric
        delta = curr - last
        if ( ( delta > last * self.convergenceRelativeDegredation and
               delta > self.convergenceAbsoluteDegredation )
             or ph._current_iteration-self.lastRun >= self.iterationInterval ):
            self.lastRun = ph._current_iteration
            self._interscenario_plugin(ph)
        self.lastConvergenceMetric = curr
        pass

    def post_iteration_k(self, ph):
        pass

    def post_ph_execution(self, ph):
        self._interscenario_plugin(ph)
        pass


    def _interscenario_plugin(self,ph):
        print("InterScenario plugin: analyzing scenario dual information")

        # (1) Collect all scenario (first) stage variables
        unique_solutions = self._collect_unique_scenario_solutions(ph)

        # (2) Filter them to find a set we want to distribute
        pass

        # (3) Distribute (some) of the variable sets out to the
        # scenarios, fix, and resolve; Collect and return the
        # objectives, duals, and any cuts
        partial_obj_values, dual_values, probability, cuts \
            = self._solve_interscenario_solutions( ph, unique_solutions )

        # Compute the non-anticipative objective values for each
        # scenario solution
        obj_values = self._compute_objective( partial_obj_values, probability,
                                              unique_solutions )

        for id, soln in enumerate(unique_solutions):
            print("  Scenario %2d: generated %2d cuts, "
                  "cut by %2d other scenarios; objective %10s, "
                  "scenario cost [%s], cut obj [%s]" % (
                id,
                sum(1 for c in cuts[id] if c is not None),
                sum(1 for c in cuts if c[id] is not None),
                "None" if obj_values[id] is None else "%10.2f" % obj_values[id],
                ", ".join("%10.2f" % x._cost for x in soln[1]),
                ", ".join(" None" if x is None else "%5.2f" % x[0] for x in cuts[id]),
            ))

        # (4) save any cuts for distribution before the next solve
        self.cutlist = []
        for c in cuts:
            self.cutlist.extend(x for x in c if x is not None)

        # (5) compute updated rho estimates
        new_rho = self._process_dual_information(
            ph, dual_values, probability, unique_solutions)
        if ph._current_iteration == 0:
            for v,r in iteritems(new_rho):
                self.rho[v] = self.rhoScale*r
        else:
            for v,r in iteritems(new_rho):
                self.rho[v] += self.rhoDamping*(self.rhoScale*r - self.rho[v])

        # (6) compute and publish the new incumbent
        self._update_incumbent( obj_values, unique_solutions )

        # (7) set the new rho values
        if len(self.cutlist) >= 0.1 * sum(len(c) for c in cuts):
            # Bypass RHO updates and check for more cuts
            self.lastRun = ph._current_iteration - self.iterationInterval
            return

        #print("SETTING SELF.RHO", self.rho)
        rootNode = ph._scenario_tree.findRootNode()
        for v, r in iteritems(self.rho):
            ph.setRhoAllScenarios(rootNode, v, r)
            #if self.incumbent is None:
            #    continue
            #for incumbent_scenario in self.incumbent[1][1]:
            #    ph.setRhoOneScenario(rootNode, incumbent_scenario, v, 0)


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
        results = ([],[],[], [])
        #cutlist = []
        if not isinstance( ph._solver_manager, SolverManager_PHPyro ):

            if ph._scenario_tree.contains_bundles():
                subproblems = ph._scenario_tree._scenario_bundles
            else:
                subproblems = ph._scenario_tree._scenarios

            for problem in subproblems:
                _tmp = solve_fixed_scenario_solutions(
                    ph, ph._scenario_tree, problem, scenario_solutions,
                    self.epsilon )
                for i,r in enumerate(results):
                    r.append(_tmp[i])
                #cutlist.extend(_tmp[-1])
        else:
            action_handles = transmit_external_function_invocation(
                ph,
                'pyomo.pysp.plugins.interscenario',
                'solve_fixed_scenario_solutions',
                return_action_handles = True,
                function_args=(scenario_solutions, self.epsilon) )

            num_results_so_far = 0
            num_results = len(action_handles)
            while (num_results_so_far < num_results):
                _ah = ph._solver_manager.wait_any()
                _tmp = ph._solver_manager.get_results(_ah)
                for i,r in enumerate(results):
                    r.append(_tmp[i])
                #cutlist.extend(_tmp[-1])
                num_results_so_far += 1

        return results# + (cutlist,)


    def _distribute_cuts(self, ph, cutlist):
        print( "InterScenario plugin: adding %d feasibility cuts" % 
               (len(cutlist),) )

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


    def _compute_objective(self, partial_obj_values, probability, unique_solns):
        obj_values = []
        for soln_id in xrange(len( unique_solns )):
            obj = 0.
            for scen_or_bundle_id, p in enumerate(probability):
                if partial_obj_values[scen_or_bundle_id][soln_id] is None:
                    obj = None
                    break
                obj += p * partial_obj_values[scen_or_bundle_id][soln_id]
            obj_values.append(obj)
        return obj_values

    def _update_incumbent(self, obj_values, unique_solns):
        feasible_obj = [o for o in obj_values if o is not None]
        if not feasible_obj:
            print("InterScenario plugin: No scenario solutions are globally feasible")
            return

        print("InterScenario plugin: Feasible objectives: %s" % ( feasible_obj, ))

        best_obj = min(self._sense_to_min*x for x in feasible_obj)
        if self.incumbent is not None and \
           self.incumbent[0] * self._sense_to_min <= best_obj:
            return

        # New incumbent!
        _id = obj_values.index(best_obj * self._sense_to_min)
        self.incumbent = ( best_obj*self._sense_to_min, unique_solns[_id], _id )
        print("New incumbent: %s = %s, %s" % self.incumbent)
        logger.info("New incumbent: %s" % (self.incumbent[0],))


    def _process_dual_information(self, ph, dual_values, probability, solutions):
        # Notes:
        #  dual_values: [ [ { var_id: dual } ] ]
        #    - list of list of maps of variable id to dual value.  The
        #      outer list is returned by each subproblem (corresponds to
        #      a bundle or scenario).  The order in this list matches
        #      the order in the probability list.  The inner list holds
        #      the dual values for each solution the scenario/bundle was
        #      asked to evaluate.  This inner list is in the same order
        #      as the solutions list.
        #  probability: [ scenario/bundle probility ]
        #    - list of the scenario or bundle probability for the
        #      submodel that returned the corresponding objective/dual
        #      values
        #   solutions: [ {var_id:var_value}, [ scenarios ] ]
        #    - list of candidate solutions holding the 1st stage
        #      variable values (in a map) and the list of scenarios
        #      that had that solution as the optimal solution in this
        #      iteration

        # soln_prob: the total probability of all scenarios that have
        # this solution as their locally-optimal solution
        soln_prob = [0.] * len(solutions)
        for soln_id, soln_info in enumerate(solutions):
            for src_scen in soln_info[1]:
                soln_prob[soln_id] += src_scen._probability
        total_soln_prob = sum(soln_prob)

        # xbar: { var_id : xbar }
        #   - this has the average first stage variable values.  We
        #     should really get this from the scenario tree, as we
        #     cannot guarantee that we will see all the current values
        #     here (they can be filtered)
        #xbar = dict( (
        #    k,
        #    sum(v*soln_prob[i] for i,v in enumerate(vv))/total_soln_prob )
        #             for k, vv in iteritems(var_info) )
        xbar = ph._scenario_tree.findRootNode()._xbars
        if self.x_deviation is None:
            self.x_deviation = dict(
                ( v,
                  max(s[0][v] for s in solutions)
                  - min(s[0][v] for s in solutions) ) for v in xbar )

        weighted_rho = dict((v,0.) for v in xbar)
        for soln_id, soln_p in enumerate(soln_prob):
            avg_dual = dict((v,0.) for v in xbar)
            p_total = 0.
            for scen_id, p in enumerate(probability):
                if dual_values[scen_id][soln_id] is None:
                    continue
                for v,d in iteritems(dual_values[scen_id][soln_id]):
                    avg_dual[v] += d * p
                p_total += p
            if p_total:
                for v in avg_dual:
                    avg_dual[v] /= p_total
            #x_deviation = dict( (v, abs(xbar[v]-solutions[soln_id][0][v]))
            #                     for v in xbar )
            for v,x_dev in iteritems(self.x_deviation):
                weighted_rho[v] += soln_prob[soln_id]*avg_dual[v]/(x_dev+1.)
        for v in xbar:
            weighted_rho[v] = abs(weighted_rho[v]) #/ total_soln_prob

        #return weighted_rho

        # var_info: { var_id : [ scenario values ] }
        #   - this has the list of all values for a single 1st stage
        #     variable, in the same order as the solutions list (and the
        #     soln_prob list)
        var_info = {}
        for soln_id, soln_info in enumerate(solutions):
            for k,v in iteritems(soln_info[0]):
                try:
                    var_info[k].append(v)
                except:
                    var_info[k] = [v]

        dual_info = {}
        for sid, scenario_results in enumerate(dual_values):
            for solution in scenario_results:
                if solution is None:
                    continue
                for k,v in iteritems(solution):
                    try:
                        dual_info[k].append(v)
                    except:
                        dual_info[k] = [v]

        for k, duals in iteritems(dual_info):
            # DISABLE!
            break

            d_min = min(duals)
            d_max = max(duals)
            _sum = sum(abs(x) for x in duals)
            _sumsq = sum(x**2 for x in duals)
            n = float(len(duals))
            d_avg = _sum/n
            d_stdev = math.sqrt(_sumsq/n - d_avg**2)

            x_min = min(var_info[k])
            x_max = max(var_info[k])
            _sum = sum(abs(x) for x in var_info[k])
            _sumsq = sum(x**2 for x in var_info[k])
            n = float(len(var_info[k]))
            x_avg = _sum/n
            x_stdev = math.sqrt(_sumsq/n - x_avg**2 + 1e-6)
            print(" %d: %6.1f [ %6.1f, %6.1f ] %5.1f --- "
                  "%6.1f [ %6.1f, %6.1f ] %5.1f  RHO %f" % (
                      k, 
                      d_avg, d_min, d_max, d_stdev,
                      x_avg, x_min, x_max, x_stdev,
                      weighted_rho[k]))

        return weighted_rho
