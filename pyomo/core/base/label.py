#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['CounterLabeler', 'NumericLabeler', 'CNameLabeler', 'TextLabeler',
           'AlphaNumTextLabeler','NameLabeler', 'CuidLabeler']

import six
if six.PY3:
    _string = str
else:
    import string as _string
from pyomo.core.base.component import ComponentUID

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
cpxlp_translation_table = _string.maketrans("[]{} -#$%&*+.,/;<=>?@^!~':",
                                             "()()______________________")
def cpxlp_label_from_name(name):

    if name is None:
        raise RuntimeError("Illegal name=None supplied to "
                           "cpxlp_label_from_name function")

    return _string.translate(name, cpxlp_translation_table)

alphanum_translation_table = _string.maketrans("()[]{} -#$%&*+.,/;<=>?@^!~':",
                                               "____________________________")
def alphanum_label_from_name(name):

    if name is None:
        raise RuntimeError("Illegal name=None supplied to "
                           "alphanum_label_from_name function")

    return _string.translate(name, alphanum_translation_table)

class CuidLabeler(object):

    def __call__(self, obj=None):
        return ComponentUID(obj)

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

#
# TODO: [JDS] I would like to rename TextLabeler to LPLabeler - as it
# generated LP-file-compliant labels - and make the CNameLabeler the
# TextLabeler.  This makes sense as the name() is the closest thing we
# have to a human-readable canonical text naming convention (the
# ComponentUID strings are actually unique, but not meant to be human
# readable).  Unfortunately, the TextLabeler is used all over the place
# (particularly PySP), and I don't know how much depends on the labels
# actually being LP-compliant.
#
class CNameLabeler(object):
    def __init__(self):
        self.name_buffer = {}

    def __call__(self, obj):
        return obj.getname(True, self.name_buffer)

class TextLabeler(object):
    def __init__(self):
        self.name_buffer = {}

    def __call__(self, obj):
        return cpxlp_label_from_name(obj.getname(True, self.name_buffer))

class AlphaNumTextLabeler(object):
    def __init__(self):
        self.name_buffer = {}

    def __call__(self, obj):
        return alphanum_label_from_name(obj.getname(True, self.name_buffer))

class NameLabeler(object):
    def __init__(self):
        self.name_buffer = {}

    def __call__(self, obj):
        return obj.getname(True, self.name_buffer)
