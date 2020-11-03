#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import glob, six

# build obj gradient and constraint Jacobian
# from a gjh file written by the ASL gjh 'solver'
# gjh solver may be called using pyomo, use keepfiles = True to write the file

# Use symbolic_solver_labels=True option in pyomo allong with keepfiles = True
# to write the .row file and .col file to get variable mappings


def readgjh(fname=None):
    if fname is None:
        files = list(glob.glob("*.gjh"))
        fname = files.pop(0)
        if len(files) > 1:
            print("**** WARNING **** More than one gjh file in current directory")
            print("  Processing: %s\nIgnoring: %s" % (
                fname, '\n         '.join(files)))

    f = open(fname,"r")

    data = "dummy_str"
    while data != "param g :=\n":
        data = f.readline()

    data = f.readline()
    g = []
    while data[0] != ';':
        # gradient entry (sparse)
        entry = [int(data.split()[0]) - 1, float(data.split()[1])] # subtract 1 to index from 0
        g.append(entry) # build obj gradient in sparse format
        data = f.readline()


    while data != "param J :=\n":
        data = f.readline()

    data = f.readline()
    J = []
    while data[0] != ';':
        if data[0] == '[':
            # Jacobian row index
            #
            # The following replaces int(filter(str.isdigit,data)),
            # which only works in 2.x
            data_as_int = int(''.join(six.moves.filter(str.isdigit, data)))
            row = data_as_int - 1  # subtract 1 to index from 0
            data = f.readline()

        entry = [row, int(data.split()[0]) - 1, float(data.split()[1])]  # subtract 1 to index from 0
        J.append(entry) # Jacobian entries, sparse format
        data = f.readline()
    f.close()

    #
    # TODO: Parse the Hessian information
    #

    f = open(fname[:-3]+'col',"r")
    data = f.read()
    varlist = data.split()
    f.close()

    f = open(fname[:-3]+'row',"r")
    data = f.read()
    conlist = data.split()
    f.close()

    return g,J,varlist,conlist


