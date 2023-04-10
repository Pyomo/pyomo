import os
from os.path import join
import yaml
import pytest
import random

import pyutilib.misc
import pyomo.environ as pe
from pyomo.contrib.alternative_solutions.solnpool import \
    gurobi_generate_solutions
from pyomo.common.fileutils import this_file_dir
from pyomo.contrib.alternative_solutions.comparison import consensus

currdir = this_file_dir()


def knapsack(N):
    random.seed(1000)

    N = N
    W = N/10.0


    model = pe.ConcreteModel()

    model.INDEX = pe.RangeSet(1,N)

    model.w = pe.Param(model.INDEX, initialize=lambda model, i : random.uniform(0.0,1.0), within=pe.Reals)

    model.v = pe.Param(model.INDEX, initialize=lambda model, i : random.uniform(0.0,1.0), within=pe.Reals)

    model.x = pe.Var(model.INDEX, within=pe.Boolean)

    model.o = pe.Objective(expr=sum(model.v[i]*model.x[i] for i in model.INDEX), sense=pe.maximize)

    model.c = pe.Constraint(expr=sum(model.w[i]*model.x[i] for i in model.INDEX) <= W)

    return model


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



def test_knapsack_100_1():
    run('knapsack_100_1', knapsack(100), 1)

def test_knapsack_100_10():
    run('knapsack_100_10', knapsack(100), 10)

def test_knapsack_100_100():
    run('knapsack_100_100', knapsack(100), 100)

