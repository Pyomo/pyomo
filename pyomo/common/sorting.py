#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

class _robust_sort_keyfcn(object):
    """Class for robustly generating sortable keys for arbitrary data.

    Generates keys (for use with Python `sorted()` that are
    (str(type_name), val), where val is the actual value (if the type
    is comparable), otherwise is the string representation of the value.
    If str() also fails, we fall back on id().

    This allows sorting lists with mixed types in Python3

    We implement this as a callable object so that we can store the
    _typemap without resorting to global variables.

    """
    _typemap = {
        bool: (1, bool.__name__),
        int: (1, float.__name__),
        float: (1, float.__name__),
        tuple: (4, tuple.__name__),
    }

    def __init__(self, key=None):
        self._key = key

    def __call__(self, val):
        """Generate a tuple ( str(type_name), val ) for sorting the value.

        `key=` expects a function.  We are generating a functor so we
        have a convenient place to store the _typemap, which converts
        the type-specific functions for converting a value to the second
        argument of the sort key.

        """
        if self._key is not None:
            val = self._key(val)

        return self._generate_sort_key(val)

    def _classify_type(self, val):
        _type = val.__class__
        _typename = _type.__name__
        try:
            # 1: Check if the type is comparable.  In Python 3, sorted()
            #    uses "<" to compare objects.
            val < val
            i = 1
            # 1a: Check if the value is comparable to a float.  If
            # it is, sort it as if it were a float.
            try:
                val < 1.
                _typename = float.__name__
            except:
                pass
        except:
            try:
                # 2: try converting the value to string
                str(val)
                i = 2
            except:
                # 3: fallback on id().  Not deterministic
                #    (run-to-run), but at least is consistent within
                #    this run.
                i = 3
        self._typemap[_type] = i, _typename
        return i, _typename

    def _generate_sort_key(self, val):
        try:
            i, _typename = self._typemap[val.__class__]
        except KeyError:
            # If this is not a type we have seen before, determine what
            # to use for the second value in the tuple.
            i, _typename = self._classify_type(val)
        if i == 1:
            # value type is directly comparable
            return _typename, val
        elif i == 4:
            # nested tuple: recurse into it (so that the tuple is comparable)
            return _typename, tuple(self._generate_sort_key(v) for v in val)
        elif i == 2:
            # value type is convertible to string
            return _typename, str(val)
        else:
            # everything else (incuding i==3), fall back on id()
            return _typename, id(val)


def sorted_robust(arg, key=None, reverse=False):
    """Utility to sort an arbitrary iterable.

    This returns the sorted(arg) in a consistent order by first tring
    the standard sorted() function, and if that fails (for example with
    mixed-type Sets in Python3), use the _robust_sort_keyfcn utility
    (above) to generate sortable keys.

    """
    # It is possible that arg is a generator.  We need to cache the
    # elements returned by the generator in case 'sort' raises an
    # exception (this ensures we don't lose any elements).  Further, as
    # we will use the in-place `list.sort()`, we want to copy any
    # incoming lists so we do not accidentally have any
    # side-effects.
    arg = list(arg)
    try:
        arg.sort(key=key, reverse=reverse)
    except:
        arg.sort(key=_robust_sort_keyfcn(key), reverse=reverse)
    return arg
