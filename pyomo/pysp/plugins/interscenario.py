#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import operator
import math
from six import iterkeys, iteritems, StringIO
from six.moves import xrange
import weakref

from pyutilib.misc import reset_redirect, setup_redirect
from pyomo.common.timing import toc

from pyomo.core import (
    minimize, value, TransformationFactory,
    ComponentUID, Block, Constraint, ConstraintList,
    Param, Var, VarList, Set, Objective, Suffix,
    Binary, NonNegativeIntegers,
)
from pyomo.opt import (
    SolverFactory, SolverStatus, TerminationCondition, ProblemFormat )
from pyomo.pysp import phextension
from pyomo.solvers.plugins.smanager.phpyro import SolverManager_PHPyro
from pyomo.common.plugin import SingletonPlugin, implements

from pyomo.repn.standard_repn import (preprocess_block_constraints,
                                      preprocess_block_objectives)
from pyomo.pysp.phsolverserverutils import InvocationType
from pyomo.pysp.convergence import NormalizedTermDiffConvergence

import logging
logger = logging.getLogger('pyomo.pysp')

import pyomo.version
PYOMO_4_0 = pyomo.version.version_info[:2] < (4,1)

FALLBACK_ON_BRUTE_FORCE_PREPROCESS = False

_acceptable_termination_conditions = set([
    TerminationCondition.optimal,
    TerminationCondition.globallyOptimal,
    TerminationCondition.locallyOptimal,
])
_infeasible_termination_conditions = set([
    TerminationCondition.infeasible,
    TerminationCondition.invalidProblem,
])

def get_modified_instance( ph, scenario_tree, scenario_or_bundle, **options):
    # Find the model
    if scenario_tree.contains_bundles():
        model = ph._bundle_binding_instance_map[scenario_or_bundle._name]
    else:
        model = ph._instances[scenario_or_bundle._name]
    b = model.component('_interscenario_plugin')
    if b is not None:
        return model

    #
    # We need to add the interscenario information to this model
    #
    model._interscenario_plugin = b = Block()

    # Save our options
    #
    b.epsilon     = options.pop('epsilon')
    b.cut_scale   = options.pop('cut_scale')
    b.allow_slack = options.pop('allow_slack')
    b.enable_rho  = options.pop('enable_rho')
    b.enable_cuts = options.pop('enable_cuts')
    assert( len(options) == 0 )

    # Information for generating cuts
    #
    b.cutlist = ConstraintList()
    b.abs_int_vars = VarList(within=NonNegativeIntegers)
    b.abs_binary_vars = VarList(within=Binary)

    # Note: the var_ids are on the ORIGINAL scenario models
    rootNode = scenario_tree.findRootNode()
    var_ids = list(iterkeys(rootNode._variable_datas))

    # Right now, this is hard-coded for 2-stage problems - so we only
    # need to worry about the variables from the root node.  These
    # variables should exist on all scenarios.  Set up a (trivial)
    # equality constraint for each variable:
    #    var == current_value{param} + separation_variable{var, fixed=0}
    b.STAGE1VAR = _S1V = Set(initialize=var_ids)
    b.separation_variables = _sep = Var( _S1V, dense=True )
    b.fixed_variable_values = _param = Param(_S1V, mutable=True, initialize=0)

    b.rho = weakref.ref(model.component('PHRHO_%s' % rootNode._name))
    b.weights = weakref.ref(model.component('PHWEIGHT_%s' % rootNode._name))

    if b.allow_slack:
        for idx in _sep:
            _sep[idx].setlb(-b.epsilon)
            _sep[idx].setub(b.epsilon)
    else:
        _sep.fix(0)

    _cuidBuffer = {}
    _src = b.local_stage1_varmap = {}
    for i in _S1V:
        # Note indexing: for each 1st stage var, pick an arbitrary
        # (first) scenario and return the variable (and not it's
        # probability)
        _cuid = ComponentUID(rootNode._variable_datas[i][0][0], _cuidBuffer)
        _src[i] = weakref.ref(_cuid.find_component_on(model))
        #_base_src[i] = weakref.ref(_cuid.find_component_on(base_model))

    def _set_var_value(b, i):
        return _param[i] + _sep[i] - _src[i]() == 0
    b.fixed_variables_constraint \
        = _con = Constraint( _S1V, rule=_set_var_value )

    #
    # TODO: When we get the duals of the first-stage variables, do we
    # want the dual WRT the original objective, or the dual WRT the
    # augmented objective?
    #
    # Move the objective to a standardized place so we can easily find it later
    if PYOMO_4_0:
        _orig_objective = list( x[2] for x in model.all_component_data(
                Objective, active=True, descend_into=True ) )
    else:
        _orig_objective = list( model.component_data_objects(
                Objective, active=True, descend_into=True ) )
    assert(len(_orig_objective) == 1)
    _orig_objective = _orig_objective[0]
    b.original_obj = weakref.ref(_orig_objective)

    # add (and deactivate) the objective for the infeasibility
    # separation problem.
    b.separation_obj = Objective(
        expr= sum( _sep[i]**2 for i in var_ids ),
        sense = minimize )

    # Make sure we get dual information
    if 'dual' not in model:
        # Export and import floating point data
        model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    #if 'rc' not in model:
    #    model.rc = Suffix(direction=Suffix.IMPORT_EXPORT)

    if FALLBACK_ON_BRUTE_FORCE_PREPROCESS:
        model.preprocess()
    else:
        _map = {}
        preprocess_block_constraints(b, idMap=_map)

    # Note: we wait to deactivate the objective until after we
    # preprocess so that the obective is correctly processed.
    b.separation_obj.deactivate()
    # (temporarily) deactivate the fixed stage-1 variables
    _con.deactivate()

    toc("InterScenario plugin: generated modified problem instance")
    return model

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
        if PYOMO_4_0:
            xfrm.apply(model, inplace=True)
        else:
            xfrm.apply_to(model)

        # Note: preprocessing is only necessary if we are changing a
        # fixed/freed variable.
        if FALLBACK_ON_BRUTE_FORCE_PREPROCESS:
            model.preprocess()
        else:
            _map = {}
            preprocess_block_constraints(
                model._interscenario_plugin, idMap=_map)

        #SOLVE
        results = solver.solve(model, warmstart=True)
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
            logger.warning(
                "Resolving subproblem model with relaxed second-stage "
                "discrete variables failed (%s).  "
                "Dual values not available." % (state,) )
        else:
            # Get the duals
            if PYOMO_4_0:
                model.load(results)
            else:
                model.solutions.load_from(results)
            #model.dual.pprint()
            for varid in model._interscenario_plugin.STAGE1VAR:
                duals[varid] = model.dual[_con[varid]]
        # Free the discrete second-stage variables
        if PYOMO_4_0:
            xfrm.apply(model, inplace=True, undo=True)
        else:
            xfrm.apply_to(model, undo=True)

    else:
        # return the duals
        for varid in model._interscenario_plugin.STAGE1VAR:
            duals[varid] = model.dual[_con[varid]]

    return duals

get_dual_values.discrete_stage2_vars = {}


def solve_separation_problem(solver, model, fallback):
    xfrm = TransformationFactory('core.relax_discrete')
    if PYOMO_4_0:
        xfrm.apply(model, inplace=True)
    else:
        xfrm.apply_to(model)

    _block = model._interscenario_plugin

    # Switch objectives
    _block.original_obj().deactivate()
    _block.separation_obj.activate()

    #_block.separation_variables.unfix()
    _par = _block.fixed_variable_values
    _sep = _block.separation_variables
    allow_slack = _block.allow_slack
    if allow_slack:
        epsilon = _block.epsilon
        for idx in _sep:
            _sep[idx].setlb(None)
            _sep[idx].setub(None)
    else:
        _sep.unfix()

    # Note: preprocessing is only necessary if we are changing a
    # fixed/freed variable.
    if FALLBACK_ON_BRUTE_FORCE_PREPROCESS:
        model.preprocess()
    else:
        _map = {}
        preprocess_block_objectives(_block, idMap=_map)
        preprocess_block_constraints(_block, idMap=_map)

    #SOLVE
    output_buffer = StringIO()
    setup_redirect(output_buffer)
    try:
        results = solver.solve(model, tee=True)
    except:
        logger.warning("Exception raised solving the interscenario "
                       "evaluation subproblem")
        logger.warning("Solver log:\n%s" % output_buffer.getvalue())
        raise
    finally:
        reset_redirect()

    ss = results.solver.status
    tc = results.solver.termination_condition
    #self.timeInSolver += results['Solver'][0]['Time']
    if ss == SolverStatus.ok and tc in _acceptable_termination_conditions:
        state = ''
        if PYOMO_4_0:
            model.load(results)
        else:
            model.solutions.load_from(results)
    elif tc in _infeasible_termination_conditions:
        state = 'INFEASIBLE'
        ans = "!!!!"
    else:
        state = 'NONOPTIMAL'
        ans = "????"
    if state:
        if fallback:
            #logger.warning("Initial attempt to solve the interscenario cut "
            #               "separation subproblem failed with the default "
            #               "solver (%s)." % (state,) )
            pass
        else:
            logger.warning("Solving the interscenario cut separation "
                           "subproblem failed (%s)." % (state,) )
            logger.warning("Solver log:\n%s" % output_buffer.getvalue())
    else:
        cut = dict((vid, (value(_sep[vid]), value(_par[vid])))
                   for vid in _block.STAGE1VAR)
        obj = value(_block.separation_obj)
        ans = (math.sqrt(obj), cut)

    output_buffer.close()

    # Restore the objective
    _block.original_obj().activate()
    _block.separation_obj.deactivate()

    # Turn off the separation variables
    if allow_slack:
        for idx in _sep:
            _sep[idx].setlb(-epsilon)
            _sep[idx].setub(epsilon)
    else:
        _sep.fix(0)

    if PYOMO_4_0:
        xfrm.apply(model, inplace=True, undo=True)
    else:
        xfrm.apply_to(model, undo=True)

    if FALLBACK_ON_BRUTE_FORCE_PREPROCESS:
        pass
    else:
        _map = {}
        preprocess_block_objectives(_block, idMap=_map)
    return ans


def add_new_cuts( ph, scenario_tree, scenario_or_bundle,
                  feasibility_cuts, incumbent_cuts, resolve ):
    # Find the model
    m = get_modified_instance(ph, scenario_tree, scenario_or_bundle)
    _block = m._interscenario_plugin
    epsilon = _block.epsilon
    cut_scale = _block.cut_scale

    # Add the cuts to the ConstraintList on the model
    _cutlist = _block.cutlist
    _src = _block.local_stage1_varmap
    for cut_obj, cut in feasibility_cuts:
        expr = sum(
            2 * (_sep*(1-cut_scale))
              * (_src[i]() - (_par+_sep*(1-cut_scale)))
            for i, (_sep, _par) in iteritems(cut)
            if abs(_sep) > epsilon*max(1,_par)
        )
        if expr != 0:
            _cutlist.add( expr >= 0 )

    for cut in incumbent_cuts:
        _int_binaries = []
        for vid, val in iteritems(cut[1]):
            # Deal with integer variables
            # b + c >= z
            # b <= M*y
            # c <= M*(1-y)
            # x - val = c - b
            # b,c >= 0
            b = _block.abs_int_vars.add()
            c = _block.abs_int_vars.add()
            z = _block.abs_binary_vars.add()
            y = _block.abs_binary_vars.add()
            _cutlist.add( b + c >= z )
            _cutlist.add( b <= _src[vid]().ub * y )
            _cutlist.add( c <= _src[vid]().ub * (1-y) )
            _cutlist.add( _src[vid]() - val == c - b )
            _int_binaries.append( z )

        _cutlist.add( sum(_int_binaries) + sum(
            _src[vid]() if val<0.5 else (1-_src[vid]())
            for vid,val in iteritems(cut[0]) ) >= 1 )

    if FALLBACK_ON_BRUTE_FORCE_PREPROCESS:
        m.preprocess()
    else:
        _map = {}
        preprocess_block_constraints(_block, idMap=_map)

    if resolve:
        results = ph._solver.solve(m, warmstart=True)
        ss = results.solver.status
        tc = results.solver.termination_condition
        #self.timeInSolver += results['Solver'][0]['Time']
        if ss == SolverStatus.ok and tc in _acceptable_termination_conditions:
            if PYOMO_4_0:
                m.load(results)
            else:
                m.solutions.load_from(results)
            _src = _block.local_stage1_varmap

            return (
                # Note: _src is {id: weakref} and original_obj is a
                # weakref; so dereference weakref before computing value
                dict((_id, value(_var())) for _id, _var in iteritems(_src)),
                value(_block.original_obj()) )
        else:
            return None


def solve_fixed_scenario_solutions(
        ph, scenario_tree, scenario_or_bundle,
        scenario_solutions, **model_options ):

    model = get_modified_instance(
        ph, scenario_tree, scenario_or_bundle, **model_options )
    _block = model._interscenario_plugin
    _param = _block.fixed_variable_values
    _sep = _block.separation_variables
    _con = _block.fixed_variables_constraint

    # We need to know which scenarios are local to this instance ... so
    # we don't waste time repeating work.
    if scenario_tree.contains_bundles():
        local_scenarios = scenario_or_bundle._scenario_names
    else:
        local_scenarios = [ scenario_or_bundle._name ]

    ipopt = SolverFactory("ipopt")

    #
    # Turn off RHO!
    #
    _saved_rho_values = _block.rho().extract_values()
    _block.rho().store_values(0)

    # Enable the constraints to fix the Stage 1 variables:
    _con.activate()

    # Solve each solution here and cache the resulting objective
    cutlist = []
    obj_values = []
    dual_values = []
    for var_values, scenario_name_list in scenario_solutions:
        local = False
        for scenario in local_scenarios:
            if scenario in scenario_name_list:
                local = True
                break
        if local:
            # Here is where we could save some time and not repeat work
            # ... for now I am being lazy and re-solving so that we get
            # the dual values, etc for this scenario as well.  If nothing
            # else, it makes averaging easier.
            pass

        assert( len(var_values) == len(_param) )
        for var_id, var_value in iteritems(var_values):
            _param[var_id] = var_value

        # TODO: We only need to update the StandardRepn for the binding
        # constraints ... so we could save a LOT of time by not
        # preprocessing the whole model.
        #
        if FALLBACK_ON_BRUTE_FORCE_PREPROCESS:
            model.preprocess()
        else:
            var_id_map = {}
            preprocess_block_constraints(_block, idMap=var_id_map)

        toc("preprocessed scenario %s" % ( scenario_or_bundle._name, ))
        output_buffer = StringIO()
        setup_redirect(output_buffer)
        try:
            results = ph._solver.solve(model, tee=True) # warmstart=True)
        except:
            logger.warning("Exception raised solving the interscenario "
                           "evaluation subproblem")
            logger.warning("Solver log:\n%s" % output_buffer.getvalue())
            raise
        finally:
            reset_redirect()
        toc("solved solution from scenario set %s on scenario %s" %
            ( scenario_name_list, scenario_or_bundle._name, ))

        ss = results.solver.status
        tc = results.solver.termination_condition
        #self.timeInSolver += results['Solver'][0]['Time']
        if ss == SolverStatus.ok and tc in _acceptable_termination_conditions:
            state = 0 #'FEASIBLE'
            if PYOMO_4_0:
                model.load(results)
            else:
                model.solutions.load_from(results)
            #
            # Turn off W, recompute the objective
            #
            _saved_w_values = _block.weights().extract_values()
            _block.weights().store_values(0)
            obj_values.append(value(_block.original_obj()))
            _block.weights().store_values(_saved_w_values)

            # NOTE: Getting the dual values resolves the model
            # (potentially relaxing second state variables.
            if _block.enable_rho:
                dual_values.append( get_dual_values(ph._solver, model) )
            else:
                dual_values.append(None)
            cutlist.append(".  ")
        elif True or tc in _infeasible_termination_conditions:
            state = 1 #'INFEASIBLE'
            obj_values.append(None)
            dual_values.append(None)
            if _block.enable_cuts:
                cut = solve_separation_problem(ph._solver, model, True)
                if cut == '????':
                    if ph._solver.problem_format() != ProblemFormat.nl:
                        model.preprocess()
                        #preprocess_block_objectives(_block)
                        #preprocess_block_constraints(_block)
                    cut = solve_separation_problem(ipopt, model, False)
            else:
                cut = "X  "
            cutlist.append( cut )
            toc("solved separation problem for solution from scenario set "
                "%s on scenario %s" %
                ( scenario_name_list, scenario_or_bundle._name, ))
        else:
            state = 2 #'NONOPTIMAL'
            obj_values.append(None)
            dual_values.append(None)
            cutlist.append("?  ")
            logger.warning("Solving the interscenario evaluation "
                           "subproblem failed (%s)." % (state,) )
            logger.warning("Solver log:\n%s" % output_buffer.getvalue())


    #
    # Turn RHO, W back on!
    #
    _block.weights().store_values(_saved_w_values)
    _block.rho().store_values(_saved_rho_values)

    # Disable the constraints to fix the Stage 1 variables:
    _con.deactivate()

    return obj_values, dual_values, cutlist


class InterScenarioPlugin(SingletonPlugin):

    implements(phextension.IPHExtension)

    def __init__(self):
        self.enableRhoUpdates = True
        self.enableFeasibilityCuts = True
        self.enableIncumbentCuts = True
        self.epsilon = 1e-7
        self.cut_scale = 0#1e-4
        self.allow_variable_slack = False
        # Force this plugin to run every N iterations
        self.iterationInterval = 100
        # Alternative methods to trigger the plugin:
        #
        # If the convergence metric degrades by either a relative or
        # absolute amount
        self.convergenceRelativeDegredation = 10.33
        self.convergenceAbsoluteDegredation = 10.001
        # If at least recutThreshold fraction of all-to-all scenario
        # tests produced feasibility cuts
        self.recutThreshold = 0.33
        # If at least this fraction of unique solutions are preserved
        # from one iteration to the next
        self.repeated_solution_threshhold = 0.90

        # multiplier on computed rho values
        self.rhoScale = 0.75
        # How quickly rho moves to new values [0..1]
        #   0: no damping (jump to calculated rho)
        #   1: complete damping (do not change current value of rho)
        self.rhoDamping = 0.1
        # Minimum difference in objective to include a cut, and minimum
        # difference in variable values to include that term in a cut
        self.cutThreshold_minDiff = 0.0001
        # Fraction of the cut library to use for cross-scenario
        # (all-to-all) cuts
        self.cutThreshold_crossCut = 0
        # Force the InterScenario plugin to re-run while the improvement
        # in the Lagrangean bound is at least this much:
        self.iteration0RecutBoundImprovement = 0.0025

    def reset(self, ph):
        self.incumbent = None
        self.rho = None
        self.x_deviation = None
        self.lastConvergenceMetric = None
        self.feasibility_cuts = []
        self.incumbent_cuts = []
        self.lastRun = 0
        self.average_solution = None
        self.converger = NormalizedTermDiffConvergence()
        self.unique_scenario_solutions = []

    def pre_ph_initialization(self,ph):
        self.reset(ph)
        pass

    def post_instance_creation(self,ph):
        if self.enableRhoUpdates:
            rootNode = ph._scenario_tree.findRootNode()
            for v in rootNode._xbars:
                ph.setRhoAllScenarios(rootNode, v, 0)
        pass

    def post_ph_initialization(self, ph):
        if len(ph._scenario_tree._stages) > 2:
            raise RuntimeError(
                "InterScenario plugin only works with 2-stage problems" )

        self._sense_to_min = 1 if ph._objective_sense == minimize else -1

        # We are going to manage RHO here.  So, we want to turn it off
        # until we finish the initial round of interscenario feasibility
        # cuts.
        if self.enableRhoUpdates:
            rootNode = ph._scenario_tree.findRootNode()
            for v in rootNode._xbars:
                ph.setRhoAllScenarios(rootNode, v, 0)
        #self.rho = dict((v,ph._rho) for v in ph._scenario_tree.findRootNode()._xbars)

    def post_iteration_0_solves(self, ph):
        self._collect_unique_scenario_solutions(ph)
        self._interscenario_plugin(ph)
        count = 0
        while self.rho is None and self.feasibility_cuts:
            count += 1
            toc( "InterScenario plugin: PH iteration 0 re-solve pass %s"
                   % (count,) )
            _stale_scenarios = []
            for _id, _soln in enumerate(self.unique_scenario_solutions):
                _was_cut = sum(
                    1 for c in self.feasibility_cuts if type(c[_id]) is tuple)
                if _was_cut:
                    _stale_scenarios.extend(_soln[1])

            self._distribute_cuts(ph, True)
            toc("InterScenario plugin: distributed cuts to scenarios")
            self._collect_unique_scenario_solutions(ph)
            self._interscenario_plugin(ph)
        self.lastRun = 0

    def post_iteration_0(self, ph):
        self.converger.update( ph._current_iteration,
                               ph,
                               ph._scenario_tree,
                               ph._instances )
        self.lastConvergenceMetric = self.converger.lastMetric()
        pass

    def pre_iteration_k_solves(self, ph):
        if self.feasibility_cuts or self.incumbent_cuts:
            self._distribute_cuts(ph)
        pass

    def post_iteration_k_solves(self, ph):
        self.converger.update( ph._current_iteration,
                               ph,
                               ph._scenario_tree,
                               ph._instances )
        curr = self.converger.lastMetric()
        last = self.lastConvergenceMetric
        delta = curr - last
        #print("InterScenario convergence:", last, curr, delta)
        run = False

        if ( self._collect_unique_scenario_solutions(ph) >=
             self.repeated_solution_threshhold ):
            print("InterScenario plugin: triggered by no change in "
                  "scenario solutions")
            run = True

        if ( delta > last * self.convergenceRelativeDegredation and
             delta > self.convergenceAbsoluteDegredation ):
            print( "InterScenario plugin: triggered by convergence degredation "
                   "(%0.4f; %+0.4f)" % (curr, delta) )
            run = True

        if ph._current_iteration-self.lastRun >= self.iterationInterval:
            print("InterScenario plugin: triggered by iteration limit")
            run = True

        if self.rho is None:
            print( "InterScenario plugin: triggered to initialize rho")
            run = True
        elif self.enableRhoUpdates:
            rootNode = ph._scenario_tree.findRootNode()
            for _id, rho in iteritems(self.rho):
                _max = rootNode._maximums[_id]
                _min = rootNode._minimums[_id]
                if rho < self.epsilon and _max - _min > self.epsilon:
                    print( "InterScenario plugin: triggered by variable "
                           "divergence with rho==0 (%s: %s; [%s, %s])"
                           % (_id, rho, _max, _min))
                    run = True
                    break

        if run:
            self.lastRun = ph._current_iteration
            self._interscenario_plugin(ph)

        self.lastConvergenceMetric = curr
        pass

    def post_iteration_k(self, ph):
        pass

    def post_ph_execution(self, ph):
        self._collect_unique_scenario_solutions(ph)
        self._interscenario_plugin(ph)
        pass


    def _interscenario_plugin(self,ph):
        toc("InterScenario plugin: analyzing scenario dual information")

        # (1) Collect all scenario (first) stage variables
        #self._collect_unique_scenario_solutions(ph)

        # (2) Filter them to find a set we want to distribute
        pass

        # (3) Distribute (some) of the variable sets out to the
        # scenarios, fix, and resolve; Collect and return the
        # objectives, duals, and any cuts
        partial_obj_values, dual_values, cuts, probability \
            = self._solve_interscenario_solutions( ph )

        # Compute the non-anticipative objective values for each
        # scenario solution
        self.feasible_objectives = self._compute_objective(
            partial_obj_values, probability )

        for _id, soln in enumerate(self.unique_scenario_solutions):
            _scenarios = [ph._scenario_tree.get_scenario(x) for x in soln[1]]
            print(
                "  Solution %2d: generated %2d cuts, "
                "cut by %2d other scenarios; objective %10s, "
                "scenario cost [%s], cut obj [%s] [generated by %s]" % (
                    _id,
                    sum(1 for c in cuts[_id] if type(c) is tuple),
                    sum(1 for c in cuts if type(c[_id]) is tuple),
                    "None" if self.feasible_objectives[_id] is None
                    else "%10.2f" % self.feasible_objectives[_id],
                    ", ".join("%10.2f" % x._cost for x in _scenarios),
                    " ".join("%5.2f" % x[0] if type(x) is tuple else "%5s" % x
                             for x in cuts[_id]),
                    ','.join(soln[1])
                ))

        scenarioCosts = [ ph._scenario_tree.get_scenario(x)._cost
                          for s in self.unique_scenario_solutions
                          for x in s[1] ]
        scenarioProb =  [ ph._scenario_tree.get_scenario(x)._probability
                          for s in self.unique_scenario_solutions
                          for x in s[1] ]
        _avg = sum( scenarioProb[i]*c for i,c in enumerate(scenarioCosts) )
        _max = max( scenarioCosts )
        _min = min( scenarioCosts )
        if self.average_solution is None:
            _del_avg = None
            _del_avg_str = "-----%"
        else:
            _prev = self.average_solution
            _del_avg = (_avg-_prev) / max(abs(_avg),abs(_prev))
            _del_avg_str = "%+.2f%%" % ( 100*_del_avg, )
        self.average_solution = _avg
        print("  Average scenario cost: %f (%s) Max-min: %f  (%0.2f%%)" % (
            _avg, _del_avg_str, _max-_min, abs(100.*(_max-_min)/_avg) ))

        # (4) save any cuts for distribution before the next solve
        #self.feasibility_cuts = []
        #for c in cuts:
        #    self.feasibility_cuts.extend(
        #        x for x in c if type(x) is tuple and x[0] > self.cutThreshold )
        #cutCount = len(self.feasibility_cuts)
        if self.enableFeasibilityCuts:
            self.feasibility_cuts = cuts
        cutCount = sum( sum( 1 for x in c if type(x) is tuple
                             and  x[0]>self.cutThreshold_minDiff )
                        for c in cuts )
        subProblemCount = sum(len(c) for c in cuts)

        # (5) compute and publish the new incumbent
        self._update_incumbent(ph)

        # (6a) If this is iteration 0, and we have feasibility cuts, and
        # they are (sufficiently) helping the Lagrangean bound, then
        # skip setting rho and do another round oc cuts
        if ph._current_iteration == 0:
            # Tell ph that we may have a good opter bound
            ph._update_reported_bounds(outer=self.average_solution)

            if ( cutCount > self.recutThreshold*(subProblemCount-len(cuts))
                 and ( _del_avg is None or
                       _del_avg > self.iteration0RecutBoundImprovement )):
                # Bypass RHO updates and check for more cuts
                #self.lastRun = ph._current_iteration - self.iterationInterval
                return

        # (6b) compute updated rho estimates
        new_rho, loginfo = self._process_dual_information(
            ph, dual_values, probability )
        _scale = self.rhoScale
        if self.rho is None:
            print("InterScenario plugin: initializing rho")
            self.rho = {}
            for v,r in iteritems(new_rho):
                self.rho[v] = _scale*r
        else:
            _damping = self.rhoDamping
            for v,r in iteritems(new_rho):
                if self.rho[v]:
                    self.rho[v] += (1-_damping)*(_scale*r - self.rho[v])
                    #self.rho[v] = max(_scale*r,  self.rho[v]) - \
                    #    _damping*abs(_scale*r - self.rho[v])
                else:
                    self.rho[v] = _scale*r

        for v,l in sorted(iteritems(loginfo)):
            if v is None:
                print(l)
            else:
                print(l % (self.rho[v],))

        #print("SETTING SELF.RHO", self.rho)
        rootNode = ph._scenario_tree.findRootNode()
        if self.enableRhoUpdates:
            for v, r in iteritems(self.rho):
                ph.setRhoAllScenarios(rootNode, v, r)


    def _collect_unique_scenario_solutions(self, ph):
        # list of (varmap, scenario_list) tuples
        _old_unique_scenario_solutions = self.unique_scenario_solutions
        self.unique_scenario_solutions = []

        # See ph.py:update_variable_statistics for a multistage version...
        rootNode = ph._scenario_tree.findRootNode()
        for scenario in rootNode._scenarios:
            _this_sol = dict(scenario._x[rootNode._name])
            for _id, _val in iteritems(scenario._x[rootNode._name]):
                #if rootNode.is_variable_fixed(_id):
                #    continue
                if rootNode.is_variable_binary(_id) or \
                        rootNode.is_variable_integer(_id):
                    _this_sol[_id] = int(round(_val))

            found = False
            # Note: because we are looking for unique variable values,
            # then if the user is bundling, this will implicitly re-form
            # the bundles
            for _sol in self.unique_scenario_solutions:
                if _this_sol == _sol[0]:
                    _sol[1].append(scenario._name)
                    found = True
                    break
            if not found:
                self.unique_scenario_solutions.append(
                    ( _this_sol, [scenario._name] ) )

        _unchanged = 0
        for _old_soln, _old_scen in _old_unique_scenario_solutions:
            for _soln, _scen in self.unique_scenario_solutions:
                if _old_soln == _soln:
                    _unchanged += 1
                    break
        print( "Interscenario plugin: %s unchanged scenario solutions "
               "(out of %s)" %
               ( _unchanged, len(self.unique_scenario_solutions) ))
        return float(_unchanged) / len(self.unique_scenario_solutions)

    def _solve_interscenario_solutions(self, ph):
        results = ([],[],[],)
        probability = []
        #cutlist = []
        distributed = isinstance( ph._solver_manager, SolverManager_PHPyro )
        action_handles = []

        if ph._scenario_tree.contains_bundles():
            subproblems = ph._scenario_tree._scenario_bundles
        else:
            subproblems = ph._scenario_tree._scenarios

        for problem in subproblems:
            probability.append(problem._probability)
            options=( self.unique_scenario_solutions, )
            kwd_options = {
                'epsilon':     self.epsilon,
                'cut_scale':   self.cut_scale,
                'allow_slack': self.allow_variable_slack,
                'enable_rho':  self.enableRhoUpdates,
                'enable_cuts': self.enableFeasibilityCuts
            }
            if distributed:
                action_handles.append(
                    ph._solver_manager.queue(
                        action="invoke_external_function",
                        name=problem._name,
                        queue_name=ph._phpyro_job_worker_map[problem._name],
                        invocation_type=InvocationType.SingleInvocation.key,
                        generateResponse=True,
                        module_name='pyomo.pysp.plugins.interscenario',
                        function_name='solve_fixed_scenario_solutions',
                        function_kwds=kwd_options,
                        function_args=options,
                    ) )
            else:
                _tmp = solve_fixed_scenario_solutions(
                    ph, ph._scenario_tree, problem,
                    *options, **kwd_options )
                for i,r in enumerate(results):
                    r.append(_tmp[i])
                #cutlist.extend(_tmp[-1])

        if distributed:
            num_results_so_far = 0
            num_results = len(action_handles)
            for r in results:
                r.extend([None]*num_results)

            while (num_results_so_far < num_results):
                _ah = ph._solver_manager.wait_any()
                _ah_id = action_handles.index(_ah)
                _tmp = ph._solver_manager.get_results(_ah)
                for i,r in enumerate(results):
                    r[_ah_id] = _tmp[i]
                #cutlist.extend(_tmp[-1])
                num_results_so_far += 1

        return results + (probability,) # + (cutlist,)


    def _distribute_cuts(self, ph, resolve=False):
        totalCuts = 0
        cutObj = sorted( c[0] for x in self.feasibility_cuts for c in x
                         if type(c) is tuple
                         and c[0] > self.cutThreshold_minDiff )
        if cutObj:
            allCutThreshold = cutObj[
                min( int((1-self.cutThreshold_crossCut)*len(cutObj)),
                     len(cutObj)-1 ) ]
        else:
            allCutThreshold = 1

        distributed = isinstance( ph._solver_manager, SolverManager_PHPyro )

        if ph._scenario_tree.contains_bundles():
            subproblems = ph._scenario_tree._scenario_bundles
            get_scenarios = lambda x: x._scenario_names
        else:
            subproblems = ph._scenario_tree._scenarios
            get_scenarios = lambda x: [x]

        resolves = []
        for problem in subproblems:
            cuts = []
            for id, (x, s) in enumerate(self.unique_scenario_solutions):
                found = False
                for scenario in get_scenarios(problem):
                    if scenario._name in s:
                        found = True
                        break
                if found:
                    cuts.extend( c[id] for c in self.feasibility_cuts
                                 if type(c[id]) is tuple
                                 and c[id][0] > self.cutThreshold_minDiff )
                elif self.feasible_objectives[id] is None:
                    # We only add cuts generated by other scenarios to
                    # scenarios that are not currently feasible (as
                    # these are feassibility cuts, they should not
                    # impact feasible scenarios)
                    cuts.extend( c[id] for c in self.feasibility_cuts
                                 if type(c[id]) is tuple
                                 and c[id][0] > allCutThreshold )

            if not cuts and not self.incumbent_cuts:
                resolves.append(None)
                continue

            totalCuts += len(cuts)
            if distributed:
                resolves.append(
                    ph._solver_manager.queue(
                        action="invoke_external_function",
                        name=problem._name,
                        queue_name=ph._phpyro_job_worker_map[problem._name],
                        invocation_type=InvocationType.SingleInvocation.key,
                        generateResponse=True,
                        module_name='pyomo.pysp.plugins.interscenario',
                        function_name='add_new_cuts',
                        function_kwds=None,
                        function_args=( cuts,
                                        self.incumbent_cuts,
                                        resolve ),
                    ) )
            else:
                ans = add_new_cuts( ph, ph._scenario_tree, problem,
                                    cuts, self.incumbent_cuts, resolve )
                resolves.append(ans)
                toc("distributed cuts to scenario %s%s" %
                    ( problem._name,
                      ' and resolved scenario' if resolve else '' ))

        toc( "InterScenario plugin: added %d feasibility cuts from a "
               "library of %s cuts" % (totalCuts, len(cutObj)) )
        self.feasibility_cuts = []

        if self.incumbent_cuts:
            print( "InterScenario plugin: added %d incumbent cuts" %
                   (len(self.incumbent_cuts), ) )
            self.incumbent_cuts = []


        if distributed:
            num_results_so_far = sum(1 for x in resolves if x is None)
            num_results = len(resolves)

            while (num_results_so_far < num_results):
                _ah = ph._solver_manager.wait_any()
                _ah_idx = resolves.index(_ah)
                resolves[_ah_idx] = ph._solver_manager.get_results(_ah)
                num_results_so_far += 1

        if resolve:
            # Transfer the first stage values and cost back to PH and
            # recompute xbar
            rootNode = ph._scenario_tree.findRootNode()
            for _id, problem in enumerate(subproblems):
                ans = resolves[_id]
                if ans is None:
                    continue
                for scenario in get_scenarios(problem):
                    scenario._cost = ans[1]
                    assert( sorted(ans[0]) ==
                            sorted(scenario._x[rootNode._name]) )
                    scenario._x[rootNode._name] = ans[0] #[_vid] = _vval
            ph.update_variable_statistics()

    def _compute_objective(self, partial_obj_values, probability):
        obj_values = []
        for soln_id in xrange(len( self.unique_scenario_solutions )):
            obj = 0.
            for scen_or_bundle_id, p in enumerate(probability):
                if partial_obj_values[scen_or_bundle_id][soln_id] is None:
                    obj = None
                    break
                obj += p * partial_obj_values[scen_or_bundle_id][soln_id]
            obj_values.append(obj)
        return obj_values

    def _update_incumbent(self, ph):
        feasible_obj = [ o for o in enumerate(self.feasible_objectives)
                         if o[1] is not None ]
        if not feasible_obj:
            print( "InterScenario plugin: No scenario solutions are "
                   "globally feasible" )
            return

        print( "InterScenario plugin: Feasible objectives: %s" %
               ( sorted(o[1] for o in feasible_obj), ) )

        best_id, best_obj = min(
            ((x[0], self._sense_to_min*x[1]) for x in feasible_obj),
            key=operator.itemgetter(1) )

        binary_vars = []
        integer_vars = []
        continuous_vars = []
        rootNode = ph._scenario_tree.findRootNode()
        for _id in rootNode._scenarios[0]._x[rootNode._name]:
            if rootNode.is_variable_fixed(_id):
                continue
            if rootNode.is_variable_binary(_id):
                binary_vars.append(_id)
            elif rootNode.is_variable_integer(_id):
                integer_vars.append(_id)
            elif rootNode.is_variable_semicontinuous(_id):
                assert False, "FIXME"
            else:
                # we can not add incumbent cuts for continuous domains
                continuous_vars.append(_id)

        if self.incumbent is None or \
           self.incumbent[0] * self._sense_to_min > best_obj + self.epsilon:
            # Cut the old incumbent
            if self.enableIncumbentCuts and self.incumbent and not continuous_vars:
                _x = self.incumbent[1][0]
                self.incumbent_cuts.append(
                    ( dict((vid, round(_x[vid])) for vid in binary_vars),
                      dict((vid, round(_x[vid])) for vid in integer_vars),
                ) )
            # New incumbent!
            self.incumbent = ( best_obj*self._sense_to_min,
                               self.unique_scenario_solutions[best_id],
                               best_id )

            # Tell PH (that we have a good inner bound)
            ph._update_reported_bounds(inner=best_obj)

            msg = "InterScenario plugin: NEW incumbent: %s = %s, %s" \
                  % self.incumbent
            print(msg)
            logger.info(msg)
        elif self.incumbent[0]*self._sense_to_min < best_obj - self.epsilon:
            # Keep existing incumbent... so the best thing here can be cut
            msg = "InterScenario plugin: incumbent: %s = %s, %s" \
                  % self.incumbent
            print(msg)
            best_id = -1

        if continuous_vars or not self.enableIncumbentCuts:
            return

        for _id, obj in feasible_obj:
            if _id == best_id:
                continue
            _x = self.unique_scenario_solutions[_id][0]
            self.incumbent_cuts.append(
                ( dict((vid, round(_x[vid])) for vid in binary_vars),
                  dict((vid, round(_x[vid])) for vid in integer_vars),
                  ) )


    def _process_dual_information(self, ph, dual_values, probability):
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
        #  unique_scenario_solutions: [ {var_id:var_value}, [ scenario_names ] ]
        #    - list of candidate solutions holding the 1st stage
        #      variable values (in a map) and the list of scenarios
        #      that had that solution as the optimal solution in this
        #      iteration

        # soln_prob: the total probability of all scenarios that have
        # this solution as their locally-optimal solution
        soln_prob = [0.] * len(self.unique_scenario_solutions)
        for soln_id, soln_info in enumerate(self.unique_scenario_solutions):
            for src_scen_name in soln_info[1]:
                src_scen = ph._scenario_tree.get_scenario(src_scen_name)
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
                  max(s[0][v] for s in self.unique_scenario_solutions)
                  - min(s[0][v] for s in self.unique_scenario_solutions) )
                for v in xbar )

        max_dual = dict((v,0.) for v in xbar)
        weighted_rho = dict((v,0.) for v in xbar)
        for soln_id, soln_p in enumerate(soln_prob):
            x = self.unique_scenario_solutions[soln_id][0]
            avg_dual = dict((v,0.) for v in xbar)
            p_total = 0.
            for scen_id, p in enumerate(probability):
                if dual_values[scen_id][soln_id] is None:
                    continue
                for v,d in iteritems(dual_values[scen_id][soln_id]):
                    avg_dual[v] += math.copysign(d, xbar[v]-x[v]) * p
                    max_dual[v] = max(max_dual[v], abs(d))
                p_total += p
            if p_total:
                for v in avg_dual:
                    avg_dual[v] /= p_total
            #x_deviation = dict( (v, abs(xbar[v]-self.unique_scenario_solutions[soln_id][0][v]))
            #                     for v in xbar )
            for v,x_dev in iteritems(self.x_deviation):
                weighted_rho[v] += soln_prob[soln_id]*avg_dual[v]/(x_dev+1.)

        if False: # MAX dual (not average)
            for v,x_dev in iteritems(self.x_deviation):
                weighted_rho[v] += max_dual[v]/(x_dev+1.)

        # var_info: { var_id : [ scenario values ] }
        #   - this has the list of all values for a single 1st stage
        #     variable, in the same order as the solutions list (and the
        #     soln_prob list)
        var_info = {}
        for soln_id, soln_info in enumerate(self.unique_scenario_solutions):
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

        # (optionally) scale the weighted rho
        #for v in xbar:
        #    weighted_rho[v] = weighted_rho[v] / total_soln_prob

        # Check for rho == 0
        _min_rho = min(_rho for _rho in weighted_rho if _rho > 0)
        for v in xbar:
            if weighted_rho[v] <= 0:
                # If there is variable disagreement, but no objective
                # pressure to price the disagreement, the best thing we
                # can do is guess and let later iterations sort it out.
                #
                #if max(var_info[v]) - min(var_info[v]) > 0:
                #    weighted_rho[v] = 1.
                #
                # Actually, we will just set all 0 rho values to the
                # smallest non-zero dual
                weighted_rho[v] = _min_rho


        loginfo = {
            None:
            "%4s: %6s [%7s, %7s] %7s;  "
            "%6s [%6s, %6s] %6s;  RHO %7s : %7s" % (
                '---',
                'Dual', 'min', 'max', 'stdev',
                'Var', 'min', 'max', 'stdev',
                'computed', 'final' )
        }
        for k, duals in iteritems(dual_info):
            # DISABLE!
            #break

            d_min = min(duals)
            d_max = max(duals)
            _sum = sum(abs(x) for x in duals)
            _sumsq = sum(x**2 for x in duals)
            n = float(len(duals))
            d_avg = _sum/n
            d_stdev = math.sqrt(abs(_sumsq/n - d_avg**2))

            x_min = min(var_info[k])
            x_max = max(var_info[k])
            _sum = sum(abs(x) for x in var_info[k])
            _sumsq = sum(x**2 for x in var_info[k])
            n = float(len(var_info[k]))
            x_avg = _sum/n
            x_stdev = math.sqrt(abs(_sumsq/n - x_avg**2 + 1e-6))
            loginfo[k] = \
                "%4d: %6.1f [%7.1f, %7.1f] %7.1f;  " \
                "%6.1f [%6.1f, %6.1f] %6.1f;  RHO %7.2f : %%7.2f" % (
                k,
                d_avg, d_min, d_max, d_stdev,
                x_avg, x_min, x_max, x_stdev,
                weighted_rho[k] )

        return weighted_rho, loginfo
