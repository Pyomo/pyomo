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

import csv


def pyomo_postprocess(options=None, instance=None, results=None):
    #
    # Collect the data
    #
    vars = set()
    data = {}
    f = {}
    for i in range(len(results.solution)):
        data[i] = {}
        for var in results.solution[i].variable:
            vars.add(var)
            data[i][var] = results.solution[i].variable[var]['Value']
        for obj in results.solution[i].objective:
            f[i] = results.solution[i].objective[obj]['Value']
            break
    #
    # Write a CSV file, with one row per solution.
    # The first column is the function value, the remaining
    # columns are the values of nonzero variables.
    #
    rows = []
    vars = list(vars)
    vars.sort()
    rows.append(['obj'] + vars)
    for i in range(len(results.solution)):
        row = [f[i]]
        for var in vars:
            row.append(data[i].get(var, None))
        rows.append(row)
    print("Creating results file results.csv")
    OUTPUT = open('results.csv', 'w')
    writer = csv.writer(OUTPUT)
    writer.writerows(rows)
    OUTPUT.close()
