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

# iterate.py

# @all:
D = {'Mary': 231}
D['Bob'] = 123
D['Alice'] = 331
D['Ted'] = 987

for i in sorted(D):
    if i == 'Alice':
        continue
    if i == 'John':
        print("Loop ends. Cleese alert!")
        break
    print(i + " " + str(D[i]))
else:
    print("Cleese is not in the list.")
# @:all
