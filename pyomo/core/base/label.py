#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['CounterLabeler', 'NumericLabeler', 'TextLabeler',
           'AlphaNumTextLabeler','NameLabeler']

try:
    import string as _string
except:
    _string = str

# This module provides some basic functionality for generating labels
# from pyomo names, which often contain characters such as "[" and "]"
# (e.g., in my_var[1]).  These characters generally cause issues with
# optimization input file formats, e.g., CPLEX LP files.  The purpose of
# this module is to provide a simple remap function, that will satisfy
# broadly problematic symbols. if solver-specific remaps are required,
# they should be handled either in the corresponding solver plugin.

# NOTE: Simple single-character substitutions should be handled by adding
#       to the translation table constructed below - first argument is the
#       "from" characters, second argument is the "to" characters.
simple_translation_table = _string.maketrans("[]{} -#$%&*+.,/;<=>?@^!~':",
                                             "()()______________________")
def label_from_name(name):

    if name is None:
        raise RuntimeError("Illegal name=None supplied to "
                           "label_from_name function")

    return _string.translate(name, simple_translation_table)

alphanum_translation_table = _string.maketrans("()[]{} -#$%&*+.,/;<=>?@^!~':",
                                               "____________________________")
def alphanum_label_from_name(name):

    if name is None:
        raise RuntimeError("Illegal name=None supplied to "
                           "alphanum_label_from_name function")

    return _string.translate(name, alphanum_translation_table)

class CounterLabeler(object):
    def __init__(self, start=0):
        self._id = start

    def __call__(self, obj=None):
        self._id += 1
        return self._id

class NumericLabeler(object):
    def __init__(self, prefix, start=0):
        self.id = start
        self.prefix = prefix

    def __call__(self, obj=None):
        self.id += 1
        return self.prefix + str(self.id)

class TextLabeler(object):
    def __init__(self):
        self.name_buffer = {}

    def __call__(self, obj):
        return label_from_name(obj.cname(True, self.name_buffer))

class AlphaNumTextLabeler(object):
    def __init__(self):
        self.name_buffer = {}

    def __call__(self, obj):
        return alphanum_label_from_name(obj.cname(True, self.name_buffer))

class NameLabeler(object):
    def __init__(self):
        self.name_buffer = {}

    def __call__(self, obj):
        return obj.cname(True, self.name_buffer)
