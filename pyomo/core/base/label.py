#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['CounterLabeler', 'NumericLabeler', 'CNameLabeler', 'TextLabeler',
           'AlphaNumericTextLabeler','NameLabeler', 'CuidLabeler',
           'ShortNameLabeler']

import re
import six
if six.PY3:
    _translate = str.translate
else:
    import string
    _translate = string.translate

from pyomo.core.base.componentuid import ComponentUID

# This module provides some basic functionality for generating labels
# from pyomo names, which often contain characters such as "[" and "]"
# (e.g., in my_var[1]).  These characters generally cause issues with
# optimization input file formats, e.g., CPLEX LP files.  The purpose of
# this module is to provide a simple remap function, that will satisfy
# broadly problematic symbols. if solver-specific remaps are required,
# they should be handled in the corresponding solver plugin.

class _CharMapper(object):
    def __init__(self, preserve, translate, other):
        """
        Arguments::
           preserve: a string of characters to preserve
           translate: a dict or key/value list of characters to translate
           other: the character to return for all characters not in
                  preserve or translate
        """
        self.table = {k if isinstance(k, int) else ord(k): v
            for k,v in six.iteritems(dict(translate)) }
        for c in preserve:
            _c = ord(c)
            if _c in self.table and self.table[_c] != c:
                raise RuntimeError("Duplicate character '%s' appears in both "
                                   "translate table and preserve list" % (c,))
            self.table[_c] = c
        self.other = other

    def __getitem__(self, c):
        # Most of the time c should be known.  For the rare cases we
        # encounter a new character, remember it by adding a new entry
        # into the translation table and return the default character
        try:
            return self.table[c]
        except:
            self.table[c] = self.other
            return self.other

    def make_table(self):
        return ''.join(self[i] for i in range(256))

_alpha = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJLKMNOPQRSTUVWXYZ'
_digit = '1234567890'
_cpxlp_translation_table = _CharMapper( preserve=_alpha+_digit+'()_',
                                        translate = zip('[]{}', '()()'),
                                        other='_' ).make_table()
def cpxlp_label_from_name(name):

    if name is None:
        raise RuntimeError("Illegal name=None supplied to "
                           "cpxlp_label_from_name function")

    return _translate(name, _cpxlp_translation_table)

_alphanum_translation_table = _CharMapper( preserve=_alpha+_digit+'_',
                                       translate = {},
                                       other='_' ).make_table()
def alphanum_label_from_name(name):

    if name is None:
        raise RuntimeError("Illegal name=None supplied to "
                           "alphanum_label_from_name function")

    return _translate(name, _alphanum_translation_table)

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

    def remove_obj(self, obj):
        pass

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

    def remove_obj(self, obj):
        self.name_buffer.pop(id(obj))

class AlphaNumericTextLabeler(object):
    def __init__(self):
        self.name_buffer = {}

    def __call__(self, obj):
        return alphanum_label_from_name(obj.getname(True, self.name_buffer))

class NameLabeler(object):
    def __init__(self):
        self.name_buffer = {}

    def __call__(self, obj):
        return obj.getname(True, self.name_buffer)

class ShortNameLabeler(object):
    def __init__(self, limit, suffix, start=0, labeler=None,
                 prefix="", caseInsensitive=False, legalRegex=None):
        self.id = start
        self.prefix = prefix
        self.suffix = suffix
        self.limit = limit
        if labeler is not None:
            self.labeler = labeler
        else:
            self.labeler = AlphaNumericTextLabeler()
        self.known_labels = set() if caseInsensitive else None
        if isinstance(legalRegex, six.string_types):
            self.legalRegex = re.compile(legalRegex)
        else:
            self.legalRegex = legalRegex

    def __call__(self, obj=None):
        lbl = self.labeler(obj)
        lbl_len = len(lbl)
        shorten = False
        if lbl_len > self.limit:
            shorten = True
        elif lbl_len == self.limit and lbl.startswith(self.prefix) \
             and lbl.endswith(self.suffix):
            shorten = True
        elif self.known_labels is not None and lbl.upper() in self.known_labels:
            shorten = True
        elif self.legalRegex and not self.legalRegex.match(lbl):
            shorten = True
        if shorten:
            self.id += 1
            suffix = "%s%d%s" % (self.suffix, self.id, self.suffix)
            tail = -self.limit + len(suffix) + len(self.prefix)
            if tail >= 0:
                raise RuntimeError(
                    "Too many identifiers.\n\t"
                    "The ShortNameLabeler cannot generate a guaranteed unique "
                    "label limited to %d characters" % (self.limit,)) 
            lbl = self.prefix + lbl[tail:] + suffix
        if self.known_labels is not None:
            self.known_labels.add(lbl.upper())
        return lbl
