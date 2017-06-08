
import sys
import pyutilib.math
import pyomo.opt


def lp_bounds(results, data, options):
    name='unknown'
    for key in ['lower bound', 'upper bound', 'solution gap']:
        data[key] = False, 'Error: No results data available (%s)' % key
    if results is None:
        return
    #
    #
    value = results.problem[0].lower_bound
    data['lower bound'] = pyutilib.math.approx_equal(value, options._baseline.problem[0].lower_bound, options.abstol, options.reltol), "'{0}' lower bound for {2} LP (baseline={1})".format(value, options._baseline.problem[0].lower_bound, name)
    #
    value = results.problem[0].upper_bound
    data['upper bound'] = pyutilib.math.approx_equal(value, options._baseline.problem[0].upper_bound, options.abstol, options.reltol), "'{0}' upper bound for {2} LP (baseline={1})".format(value, options._baseline.problem[0].upper_bound, name)
    #
    value = results.solution[0].gap
    if value != pyomo.opt.undefined:
        data['solution gap'] = pyutilib.math.approx_equal(value, 0.0, options.abstol, options.reltol), "'{0}' solution gap for {1} LP solution".format(value, name)
    #

#
# Verify the problem name?
# Verify that we get reduced cost and duals (and others?)
# Verify problem stats?  (# constraints, # nonzeros, etc)
# Distinguish between feasible and optimal SolutionStatus values?
#
def lp_feasible_solution(results, data, options, name):
    for key in ['single objective', 'sense', 'solver status', 'termination condition', 'single solution', 'solution status', 'objective name', 'objective name', 'objective value']:
        data[key] = False, 'Error: No results data available (%s)' % key
    if results is None:
        return
    #
    #
    value = results.problem[0].number_of_objectives
    data['single objective'] =  value==1, '{0} objectives found for {1} LP'.format(value, name)
    #
    value = str(results.problem[0].sense)
    data['sense'] = value==options._baseline.problem[0].sense, "'{0}' sense for {2} LP (baseline={1})".format(value, options._baseline.problem[0].sense, name)
    #
    #
    value = results.solver.status
    data['solver status'] = value==pyomo.opt.SolverStatus.ok, "'{0}' solver status for {1} LP".format(value, name)
    #
    value = results.solver.termination_condition
    data['termination condition'] = value==pyomo.opt.TerminationCondition.optimal, "'{0}' termination condition for {1} LP".format(value, name)
    #
    value = results.solver.get('error_rc',None)
    data['error rc'] = value==0 or value==None, "'{0}' error rc for {1} LP".format(value, name)
    #
    #
    value = len(results.solution)
    data['single solution'] = value==1, '{0} solutions found for {1} LP'.format(value, name)
    #
    value = results.solution[0].status
    data['solution status'] = value==pyomo.opt.SolutionStatus.optimal or value==pyomo.opt.SolutionStatus.feasible, "'{0}' solution status for {1} LP solution".format(value, name)
    #
    if len(results.solution[0].objective) > 0:
        objname = results.solution[0].objective.keys()[0]
        _objname = options._baseline.solution[0].objective.keys()[0]
        data['objective name'] = objname == _objname, "'{0}' objective name for {2} LP solution (baseline={1})".format(objname, _objname, name)
        #
        value = results.solution[0].objective[objname].value
        data['objective value'] = pyutilib.math.approx_equal(value, options._baseline.solution[0].objective[_objname].value, options.abstol, options.reltol), "'{0}' objective value for {2} LP solution (baseline={1})".format(value, options._baseline.solution[0].objective[_objname].value, name)


def lp_feasible(results, data, options):
    lp_feasible_solution(results, data, options, 'feasible')

def lp_constant(results, data, options):
    lp_feasible_solution(results, data, options, 'constant')


def lp_unbounded(results, data, options):
    for key in ['single objective', 'no solution', 'solver status', 'termination condition']:
        data[key] = False, 'Error: No results data available (%s)' % key
    if results is None:
        return
    #
    value = results.problem[0].number_of_objectives
    data['single objective'] =  value==1, '{0} objectives found for unbounded LP'.format(value)
    #
    value = len(results.solution)
    data['no solution'] = value==0, '{0} solutions found for unbounded LP'.format(value)
    #
    value = results.solver.status
    data['solver status'] = value==pyomo.opt.SolverStatus.ok, "'{0}' solver status for unbounded LP".format(value)
    #
    value = results.solver.termination_condition
    data['termination condition'] = value==pyomo.opt.TerminationCondition.unbounded, "'{0}' termination condition for unbounded LP".format(value)


def lp_infeasible(results, data, options):
    for key in ['single objective', 'solver status', 'termination condition']:
        data[key] = False, 'Error: No results data available (%s)' % key
    if results is None:
        return
    #
    value = results.problem[0].number_of_objectives
    data['single objective'] =  value==1, '{0} objectives found for infeasible LP'.format(value)
    #
    value = results.solver.status
    data['solver status'] = value==pyomo.opt.SolverStatus.ok, "'{0}' solver status for infeasible LP".format(value)
    #
    value = results.solver.termination_condition
    data['termination condition'] = value==pyomo.opt.TerminationCondition.infeasible, "'{0}' termination condition for infeasible LP".format(value)


def lp_unique_dual(results, data, options):
    for key in ['lower bound reduced cost suffix', 'upper bound reduced cost suffix', 'reduced cost suffix', 'dual suffix', 'slack suffix']:
        data[key] = False, 'Error: No results data available (%s)' % key
    if results is None:
        return
    if len(results.solution) == 0:
        return
    #
    soln = results.solution[0]
    for var in soln.variable:
        v = soln.variable[var]
        if 'Lrc' in v:
            data['lower bound reduced cost suffix'] =  True, '"lrc" suffix provided by solver'
        if 'Urc' in v:
            data['upper bound reduced cost suffix'] =  True, '"urc" suffix provided by solver'
        if 'Rc' in v:
            data['reduced cost suffix'] =  True, '"rc" suffix provided by solver'
    #
    for con in soln.constraint:
        c = soln.constraint[con]
        if 'Dual' in c:
            data['dual suffix'] =  True, '"dual" suffix provided by solver'
        if 'Slack' in c:
            data['slack suffix'] =  True, '"slack" suffix provided by solver'

