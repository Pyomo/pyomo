import sys
import os
from six import iteritems
import pyutilib.th as unittest
from pyutilib.misc.pyyaml_util import *
import pyomo.core.scripting.util as util
from pyomo.core.base import Var, active_components_data
import pyomo.environ

from pyomo.core.base.objective import minimize, maximize
from pyomo.core.base.piecewise import Bound, PWRepn

from os.path import dirname, abspath, join

yaml_available=False
try:
    import yaml
    yaml_available = True
except: 
    pass

currdir = dirname( abspath(__file__) )

smoke_problems = ['convex_var','step_var','step_vararray']

nightly_problems = ['convex_vararray', 'concave_vararray', \
                'concave_var','piecewise_var', 'piecewise_vararray']

expensive_problems = ['piecewise_multi_vararray', \
                'convex_multi_vararray1','concave_multi_vararray1', \
                'convex_multi_vararray2','concave_multi_vararray2']


def module_available(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False

def has_gurobi_lp():
    try:
        gurobi = pyomo.plugins.solvers.GUROBI(keepfiles=True)
        available = (not gurobi.executable() is None) and gurobi.available(False)
        return available
    except pyutilib.common.ApplicationError:
        return False

def has_gurobi_nl():
    try:
        gurobi = pyomo.plugins.solvers.GUROBI(keepfiles=True)
        available = (not gurobi.executable() is None) and gurobi.available(False)
        asl = pyomo.plugins.solvers.ASL(keepfiles=True, options={'solver':'gurobi_ampl'})
        return available and (not asl.executable() is None) and asl.available(False)
    except pyutilib.common.ApplicationError:
        return False

def has_gurobi_python():
    if module_available('gurobipy'):
        return True
    return False

def has_cplex_lp():
    try:
        cplex = pyomo.plugins.solvers.CPLEX(keepfiles=True)
        available = (not cplex.executable() is None) and cplex.available(False)
        return available
    except pyutilib.common.ApplicationError:
        return False

def has_cplex_nl():
    try:
        cplex = pyomo.plugins.solvers.CPLEX(keepfiles=True)
        available = (not cplex.executable() is None) and cplex.available(False)
        asl = pyomo.plugins.solvers.ASL(keepfiles=True, options={'solver':'cplexamp'})
        return available and (not asl.executable() is None) and asl.available(False)
    except pyutilib.common.ApplicationError:
        return False

def has_cplex_python():
    if module_available('cplex'):
        return True
    return False

def has_glpk_python():
    if module_available('glpk'):
        return True
    return False

def has_glpk_lp():
    try:
        glpk = pyomo.plugins.solvers.GLPK(keepfiles=True)
        available = (not glpk.executable() is None) and glpk.available(False)
        return available
    except pyutilib.common.ApplicationError:
        return False

writer_solver = []
#if has_cplex_python():
#    writer_solver.append(('python','cplex'))
#if has_gurobi_python():
#    writer_solver.append(('python','gurobi'))
#if has_cplex_lp():
#    writer_solver.append(('lp','cplex'))
#if has_gurobi_lp():
#    writer_solver.append(('lp','gurobi'))
if has_cplex_nl():
    writer_solver.append(('nl','cplexamp'))
#if has_gurobi_nl():
#    writer_solver.append(('nl','gurobi_ampl'))
#if has_glpk_lp():
#    writer_solver.append(('lp','glpk'))
#if has_glpk_python():
#    writer_solver.append(('python','glpk'))


def createTestMethod(pName,problem,solver,writer,kwds):
    
    def testMethod(obj):
        from pyutilib.misc import Options
        # WEH - This logic is dangerous.  Explicit imports should be used.
        #os.sys.path.append(join(currdir,'problems'))
        m = __import__(problem)
        #os.sys.path.pop()
        options = Options()
        options.solver = solver
        options.solver_io = writer
        options.quiet = True
        options.debug = True
        data = Options(options=options)
        
        model = m.define_model(**kwds)
        instance = model.create()

        opt_data = util.apply_optimizer(data, instance=instance)
        
        instance.load(opt_data.results)

        new_results = ( (var.cname(),var.value) for var in active_components_data(instance,Var) ) # non-recursive
        baseline_results = getattr(obj,problem+'_results')
        for name, value in new_results:
            if abs(baseline_results[name]-value) > 0.00001:
                raise IOError("Difference in baseline solution values and current solution values using:\n" + \
                "Solver: "+solver+"\n" + \
                "Writer: "+writer+"\n" + \
                "Variable: "+name+"\n" + \
                "Solution: "+str(value)+"\n" + \
                "Baseline: "+str(baseline_results[name])+"\n")

    return testMethod


def assignTests(cls, problem_list):
    for writer,solver in writer_solver:
        for PROBLEM in problem_list:
            aux_list = ['','force_pw']
            for AUX in aux_list:
                for REPN in PWRepn:
                    for BOUND_TYPE in Bound:
                        for SENSE in [maximize,minimize]:
                            if not( ((BOUND_TYPE == Bound.Lower) and (SENSE == maximize)) or \
                                    ((BOUND_TYPE == Bound.Upper) and (SENSE == minimize)) or \
                                    ((REPN in [PWRepn.BIGM_BIN,PWRepn.BIGM_SOS1,PWRepn.MC]) and ('step' in PROBLEM)) ):
                                kwds = {}
                                kwds['sense'] = SENSE
                                kwds['pw_repn'] = REPN
                                kwds['pw_constr_type'] = BOUND_TYPE
                                if SENSE == maximize:
                                    attrName = "test_{0}_{1}_{2}_{3}_{4}_{5}".format(PROBLEM,REPN,BOUND_TYPE,'maximize',solver,writer)
                                elif SENSE == minimize:
                                    attrName = "test_{0}_{1}_{2}_{3}_{4}_{5}".format(PROBLEM,REPN,BOUND_TYPE,'minimize',solver,writer)
                                if AUX != '':
                                    kwds[AUX] = True 
                                    attrName += '_'+AUX
                                setattr(cls,attrName,createTestMethod(attrName,PROBLEM,solver,writer,kwds))
                                if yaml_available:
                                    with open(join(currdir,'baselines',PROBLEM+'_baseline_results.yml'),'r') as f:
                                        baseline_results = yaml.load(f)
                                        setattr(cls,PROBLEM+'_results',baseline_results)

@unittest.skipIf(writer_solver==[], "Can't find a solver.")
@unittest.skipUnless(yaml_available, "PyYAML module is not available.")
class PW_Tests(unittest.TestCase): pass

@unittest.category('nightly', 'expensive')
class PiecewiseLinearTest_Nightly(PW_Tests): pass
assignTests(PiecewiseLinearTest_Nightly, nightly_problems)

@unittest.category('smoke', 'nightly', 'expensive')
class PiecewiseLinearTest_Smoke(PW_Tests): pass
assignTests(PiecewiseLinearTest_Smoke, smoke_problems)

@unittest.category('expensive')
class PiecewiseLinearTest_Expensive(PW_Tests): pass
assignTests(PiecewiseLinearTest_Expensive, expensive_problems)

if __name__ == "__main__":
    unittest.main()

