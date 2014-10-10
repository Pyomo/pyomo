#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2010 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

__all__ = ['label_from_name', 'nl_label_from_name']

try:
    long
    import string as str
except:
    pass


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
simple_translation_table = str.maketrans("[]{} -#$%&*+.,/;<=>?@^!~':",
                                            "()()______________________")

nl_translation_table = str.maketrans("{} -#$%&*+./;<=>?@^!~'",
                                            "()____________________")

def label_from_name(name):

    global simple_translation_table

    if name is None:
        raise RuntimeError(
            "Illegal name=None supplied to label_from_name function")

    return str.translate(name, simple_translation_table)

def nl_label_from_name(name):
    global nl_translation_table
    
    if name is None:
        raise RuntimeError("Illegal name=None supplied to nl_label_from_name function")
    
    return str.translate(name, nl_translation_table)
