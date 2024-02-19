#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

'''
One dimensional cutting-stock example using column generation techniques.
Method from Section 12.8 of

Bradley, S.P., A.C. Hax, and T.L. Magnanti. 1977. Applied Mathematical Programming,
Addison-Wesley, Reading, MA. Available: http://web.mit.edu/15.053/www/AMP.htm.

Data from https://en.wikipedia.org/wiki/Cutting_stock_problem 
'''

import pyomo.environ as pyo

# width: number
demand = {
    1380: 22,
    1520: 25,
    1560: 12,
    1710: 14,
    1820: 18,
    1880: 18,
    1930: 20,
    2000: 10,
    2050: 12,
    2100: 14,
    2140: 16,
    2150: 18,
    2200: 20,
}

# master roll size
W = 5600


def create_base_cutting_stock(demand, W):
    initial_patterns = dict()

    ## cutting stock base problem
    cs = pyo.ConcreteModel()

    cs.pattern = pyo.VarList(domain=pyo.NonNegativeReals)

    # add initial columns for each
    # demanded width
    for i, width in enumerate(demand):
        cs.pattern.add()
        initial_patterns[i + 1] = {width: int(W // width)}

    # add the demand constraints; supply initial identity columns;
    # filling in as many of a single width on a pattern as possible
    cs.demand = pyo.Constraint(demand.keys())
    for i, (width, quantity) in enumerate(demand.items()):
        cs.demand[width] = (
            initial_patterns[i + 1][width] * cs.pattern[i + 1] >= quantity
        )

    cs.obj = pyo.Objective(expr=pyo.quicksum(cs.pattern.values()), sense=pyo.minimize)

    cs.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    ## knapsack cut generator
    ks = pyo.ConcreteModel()

    ks.widths = pyo.Var(demand.keys(), within=pyo.NonNegativeIntegers)

    ks.knapsack = pyo.Constraint(
        expr=pyo.quicksum(width * ks.widths[width] for width in demand) <= W
    )

    # blank objective, set by the dual values of cs
    ks.obj = pyo.Objective(expr=0, sense=pyo.maximize)

    return cs, ks, initial_patterns


def solve_cutting_stock(demand, W, solver, iterations=30):
    cs, ks, patterns = create_base_cutting_stock(demand, W)

    if '_persistent' not in solver:
        raise RuntimeError(
            'solver must be a string for pyo.SolverFactory and persistent'
        )

    cs_s = pyo.SolverFactory(solver)
    ks_s = pyo.SolverFactory(solver)

    cs_s.set_instance(cs)
    ks_s.set_instance(ks)

    for _ in range(iterations):
        cs_s.solve()

        duals = {width: cs.dual[cs.demand[width]] for width in demand}

        ks.obj.expr = sum(duals[width] * ks.widths[width] for width in demand)

        ks_s.set_objective(ks.obj)

        ks_s.solve()

        if pyo.value(ks.obj) <= 1:
            # no better column
            break

        # else we'll add the column from ks
        new_pattern_var = cs.pattern.add()
        np_widths = []
        np_constraints = []
        pattern = dict()

        for width, var in ks.widths.items():
            cut_number = int(round(pyo.value(var)))
            if cut_number == 0:
                continue
            np_widths.append(cut_number)
            np_constraints.append(cs.demand[width])

            pattern[width] = cut_number

        patterns[len(cs.pattern)] = pattern

        cs_s.add_column(cs, new_pattern_var, 1.0, np_constraints, np_widths)

    # heuristically solve the cutting stock problem with integer restrictions
    # to get an integer feasible solution

    cs.pattern.domain = pyo.NonNegativeIntegers
    for var in cs.pattern.values():
        cs_s.update_var(var)
    del cs.dual

    cs_s.solve()

    return cs, patterns


if __name__ == '__main__':
    import sys

    solver = sys.argv[1]

    cs, patterns = solve_cutting_stock(demand, W, solver)

    print('Sheets Required: ' + str(int(pyo.value(cs.obj))))
    print('Repetition\tPattern')
    for idx, var in cs.pattern.items():
        quantity = int(pyo.value(var))
        if quantity > 0:
            print_str = str(quantity) + '\t\t'
            for width, number in patterns[idx].items():
                print_str += str(int(number)) + ':' + str(int(width)) + ', '
            print(print_str[:-2])
