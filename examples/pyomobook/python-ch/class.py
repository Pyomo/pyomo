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

# class.py


# @all:
class IntLocker:
    sint = None

    def __init__(self, i):
        self.set_value(i)

    def set_value(self, i):
        if type(i) is not int:
            print("Error: %d is not integer." % i)
        else:
            self.sint = i

    def pprint(self):
        print("The Int Locker has " + str(self.sint))


a = IntLocker(3)
a.pprint()  # prints: The Int Locker has 3
a.set_value(5)
a.pprint()  # prints: The Int Locker has 5
# @:all
