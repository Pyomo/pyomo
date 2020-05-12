#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

def validate_index(i, array_len, array_name=''):
    if not isinstance(i, int):
        raise TypeError(
            'Index into %s array must be an integer. Got %s'
            % (array_name, type(i)))
    if i < 1 or i > array_len:
        # NOTE: Use the FORTRAN indexing (same as documentation) to
        # set and access info/cntl arrays from Python, whereas C
        # functions use C indexing. Maybe this is too confusing.
        raise IndexError(
            'Index %s is out of range for %s array of length %s'
            % (i, array_name, array_len))

def validate_value(val, dtype, array_name=''):
    if not isinstance(val, dtype):
        raise ValueError(
            'Members of %s array must have type %s. Got %s'
            % (array_name, dtype, type(val)))

class _NotSet:
    pass

