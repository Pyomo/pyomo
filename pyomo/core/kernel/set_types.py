#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging


logger = logging.getLogger('pyomo.core')

_virtual_sets = []

#
# Dummy types used by Kernel as domain flags
#
class RealSet(object):
    @staticmethod
    def get_interval():
        return (None, None, 0)

    @staticmethod
    def is_continuous():
        return True

    @staticmethod
    def is_integer():
        return False

    @staticmethod
    def is_binary():
        return False


class IntegerSet(object):
    @staticmethod
    def get_interval():
        return (None, None, 1)

    @staticmethod
    def is_continuous():
        return False

    @staticmethod
    def is_integer():
        return True

    @staticmethod
    def is_binary():
        return False


class BinarySet(object):
    @staticmethod
    def get_interval():
        return (0, 1, 1)

    @staticmethod
    def is_continuous():
        return False

    @staticmethod
    def is_integer():
        return True

    @staticmethod
    def is_binary():
        return True

#TODO: Deprecate BooleanSet (that will soon be replaced by a true BooleanSet
#      admitting {True, False})
BooleanSet = BinarySet
