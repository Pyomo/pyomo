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

# functions.py


# @all:
def Apply(f, a):
    r = []
    for i in range(len(a)):
        r.append(f(a[i]))
    return r


def SqifOdd(x):
    # if x is odd, 2*int(x/2) is not x
    # due to integer divide of x/2
    if 2 * int(x / 2) == x:
        return x
    else:
        return x * x


ShortList = range(4)
B = Apply(SqifOdd, ShortList)
print(B)
# @:all
