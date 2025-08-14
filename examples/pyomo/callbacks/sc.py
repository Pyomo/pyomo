#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.collections import Bunch
import pyomo.environ as pyo
import math
import random


def print_model_stats(options, model):
    print("-" * 40)
    if options is None:
        print("DEFAULT")
    else:
        print(options.type)
    rowc = {}
    for i in model.I:
        rowc[i] = 0
    colc = {}
    for i in model.J:
        colc[i] = 0
    for i, j in model.S:
        rowc[i] += 1
        colc[j] += 1
    print("Row Counts")
    s = 0.0
    for i in sorted(rowc):
        s += rowc[i]
    print("Average: %s" % str(s / len(rowc)))
    print("Col Counts")
    s = 0.0
    for i in sorted(colc):
        s += colc[i]
    print("Average: %s" % str(s / len(colc)))
    print("I %d" % len(model.I))
    print("J %d" % len(model.J))
    print("-" * 40)


def pyomo_create_model(options=None, model_options=None):
    if model_options is None:
        model_options = Bunch()
    if model_options.type is None:
        model_options.type = 'fixed_set_size'
    #
    # m - number of elements
    #
    m = 100 if model_options.m is None else model_options.m
    #
    # n - number of sets
    #
    n = 200 if model_options.n is None else model_options.n
    seed = 9090 if model_options.seed is None else model_options.seed
    random.seed(9090)
    #
    if model_options.type == 'fixed_set_size':
        #
        # p   - fixed number elements per set
        # rho - fixed fraction of elements per set
        #
        p = model_options.p
        if p is None:
            if model_options.rho is None:
                p = int(math.ceil(m * 0.7))
            else:
                p = int(math.ceil(m * model_options.rho))

        #
        def S_rule(model):
            ans = set()
            for j in range(1, n + 1):
                tmp = list(range(1, m + 1))
                random.shuffle(tmp)
                for i in range(0, p):
                    ans.add((tmp[i], j))
            return ans

    elif model_options.type == 'fixed_element_coverage':
        #
        # p   - fixed number of sets that cover each element
        # rho - fixed fraction of sets that cover each element
        #
        p = model_options.p
        if p is None:
            if model_options.rho is None:
                p = int(math.ceil(n * 0.4))
            else:
                p = int(math.ceil(n * model_options.rho))

        #
        def S_rule(model):
            ans = set()
            for i in range(1, m + 1):
                tmp = list(range(1, n + 1))
                random.shuffle(tmp)
                for j in range(0, p):
                    ans.add((i, tmp[j]))
            return ans

    elif model_options.type == 'fixed_probability':
        #
        # rho - probability of selecting element for a set
        #
        rho = 0.3 if model_options.rho is None else model_options.rho

        #
        def S_rule(model):
            ans = set()
            for j in range(1, n + 1):
                for i in range(1, m + 1):
                    if random.uniform(0, 1) < rho:
                        ans.add((i, j))
            return ans

    elif model_options.type == 'fixed_fill':
        #
        # rho - |S|/(I*J)
        #
        rho = 0.3 if model_options.rho is None else model_options.rho

        #
        def S_rule(model):
            ans = set()
            for j in range(1, n + 1):
                for i in range(1, m + 1):
                    if random.uniform(0, 1) < rho:
                        ans.add((i, j))
            return ans

    #
    # CREATE MODEL
    #
    model = pyo.ConcreteModel()
    #
    # (i,j) in S if element i in set j
    #
    model.S = pyo.Set(dimen=2, initialize=S_rule)

    #
    # Dynamically create the I and J index sets, since
    # some rows or columns of S may not be populated.
    #
    def I_rule(model):
        return set((i for (i, j) in model.S))

    model.I = pyo.Set(initialize=I_rule)

    def J_rule(model):
        return set((j for (i, j) in model.S))

    model.J = pyo.Set(initialize=J_rule)
    #
    # Weights
    #
    model.w = pyo.Param(model.J, within=pyo.NonNegativeReals, initialize=1.0)
    #
    # Set selection binary variables
    #
    model.x = pyo.Var(model.J, within=pyo.Binary)

    #
    # Objective
    #
    def cost_rule(model):
        return pyo.sum_product(model.w, model.x)

    model.cost = pyo.Objective(rule=cost_rule)

    #
    # Constraint
    #
    def cover_rule(model, i):
        expr = 0
        for j in model.x:
            if (i, j) in model.S:
                expr += model.x[j]
        #
        # WEH - this check is not needed, since I is constructed dynamically
        #
        # if expr is 0:
        # return Constraint.Skip
        return expr >= 1

    model.cover = pyo.Constraint(model.I, rule=cover_rule)

    #
    print_model_stats(model_options, model)
    return model


def test_model(options=None):
    model = pyomo_create_model(model_options=options)
    # print_model_stats(options, model)


if __name__ == '__main__':
    test_model()
    #
    options = Bunch()
    options.type = 'fixed_set_size'
    options.m = 11
    options.n = 21
    options.rho = 0.3
    test_model(options)
    #
    options = Bunch()
    options.type = 'fixed_element_coverage'
    test_model(options)
    #
    options = Bunch()
    options.m = 100
    options.n = 200
    options.type = 'fixed_probability'
    test_model(options)
    #
    options = Bunch()
    options.type = 'fixed_element_coverage'
    options.m = 10
    options.n = 100
    options.rho = 0.1
    test_model(options)
    #
