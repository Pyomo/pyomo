import os
from os.path import join
import yaml

import pyutilib.misc
import pyomo.environ as pe
from pyomo.contrib.alternative_solutions.solnpool import \
    gurobi_generate_solutions
from pyomo.common.fileutils import this_file_dir
from pyomo.contrib.alternative_solutions.comparison import consensus

currdir = this_file_dir()


def run(testname, model, N, debug=False):
    solutions = gurobi_generate_solutions(model=model, max_solutions=N)
    print(solutions)
    # Verify final results

    results = [soln.get_variable_name_values() for soln in solutions]
    output = yaml.dump(results, default_flow_style=None)
    outputfile = join(currdir, "{}_results.yaml".format(testname))
    with open(outputfile, "w") as OUTPUT:
        OUTPUT.write(output)

    baselinefile = join(currdir, "{}_baseline.yaml".format(testname))
    tmp = pyutilib.misc.compare_file(outputfile, baselinefile, tolerance=1e-7)
    assert tmp[0] == False, "Files differ:  diff {} {}".format(outputfile, baselinefile)
    os.remove(outputfile)

    if N>1:
        # Verify consensus pattern

        comp = consensus(results)
        output = yaml.dump(comp, default_flow_style=None)
        outputfile = join(currdir, "{}_comp_results.yaml".format(testname))
        with open(outputfile, "w") as OUTPUT:
            OUTPUT.write(output)
        
        baselinefile = join(currdir, "{}_comp_baseline.yaml".format(testname))
        tmp = pyutilib.misc.compare_file(outputfile, baselinefile, tolerance=1e-7)
        assert tmp[0] == False, "Files differ:  diff {} {}".format(outputfile, baselinefile)
        os.remove(outputfile)



import test_cases

model = test_cases.knapsack(10)

ast = '*'*10

# print(ast,'Start APPSI',ast)
# from pyomo.contrib import appsi
# opt = appsi.solvers.Gurobi()
# opt.config.stream_solver = True
# #opt.set_instance(model)
# opt.gurobi_options['PoolSolutions'] = 10
# opt.gurobi_options['PoolSearchMode'] = 2
# #opt.set_gurobi_param('PoolSolutions', 10)
# #opt.set_gurobi_param('PoolSearchMode', 2)
# results = opt.solve(model)
# print(ast,'END APPSI',ast)

# print(ast,'Start Solve Factory',ast)
# from pyomo.opt import SolverFactory
# opt2 = SolverFactory('gurobi')
# opt.gurobi_options['PoolSolutions'] = 10
# opt.gurobi_options['PoolSearchMode'] = 2
# opt2.solve(model, tee=True)
# print(ast,'End Solve Factory',ast)

print(ast,'Start Solve Factory',ast)
from pyomo.opt import SolverFactory
opt3 = SolverFactory('appsi_gurobi')
opt3.gurobi_options['PoolSolutions'] = 10
opt3.gurobi_options['PoolSearchMode'] = 2
opt3.solve(model, tee=True)
print(ast,'End Solve Factory',ast)