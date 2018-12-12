#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


# ToDo: make this an abstract class
class SparseBase(object):

    def __init__(self):
        self._symmetric = False
        self._name = None

    @property
    def is_symmetric(self):
        return self._symmetric

    @is_symmetric.setter
    def is_symmetric(self, value):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, mname):
        self._name = mname

    def set_shape(self, shape):
        raise RuntimeError('set shape is not supported')

    def tofullmatrix(self):
        return NotImplemented
