#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


class _robust_sort_keyfcn(object):
    """Class for robustly generating sortable keys for arbitrary data.

    Generates keys (for use with Python `sorted()` that are
    (str(type_name), val), where val is the actual value (if the type
    is comparable), otherwise the string representation of the value.
    If str() also fails, we fall back on id().

    This allows sorting lists with mixed types in Python3

    We implement this as a callable object so that we can store the
    user's original key function, if provided

    """

    _typemap = {
        int: (1, float.__name__),
        float: (1, float.__name__),
        str: (1, str.__name__),
        tuple: (4, tuple.__name__),
    }

    def __init__(self, key=None):
        self._key = key

    def __call__(self, val):
        """Generate a tuple ( str(type_name), val ) for sorting the value.

        `key=` expects a function.  We are generating a functor so we
        have a convenient place to store the user-provided key and the
        (singleton) _typemap, which maps types to the type-specific
        functions for converting a value to the second argument of the
        sort key.

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
                # Extra check that the comparison returns a meaningful result
                if bool(val < 1.0) != bool(1.0 < val or 1.0 == val):
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

    def _generate_sort_key(self, val):
        if val.__class__ not in self._typemap:
            # If this is not a type we have seen before, determine what
            # to use for the second value in the sorting tuple.
            self._classify_type(val)
        i, _typename = self._typemap[val.__class__]
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
            # everything else (including i==3), fall back on id()
            return _typename, id(val)


def sorted_robust(iterable, key=None, reverse=False):
    """Utility to sort an arbitrary iterable.

    This returns the sorted(arg) in a consistent order by first trying
    the standard sort() function, and if that fails (for example with
    mixed-type Sets in Python3), use the _robust_sort_keyfcn utility
    (above) to generate sortable keys.

    Parameters
    ----------
    iterable: iterable
        the source of items to sort
    key: function
        a function of one argument that is used to extract the
        comparison key from each element in `iterable`
    reverse: bool
        if True, the iterable is sorted as if each comparison was reversed.

    Returns
    -------
    list
    """
    # Because we implement this as a "try a normal (fast) sort, then
    # fall back on our slow, but robust sort", we will need to cache the
    # incoming arg: it may be a generator, in which case we would need
    # to exhaust it so we can cache all the values for the case that the
    # first sort attempt fails.  Given that, it is simpler / easier to
    # take *all* incoming args and create a new list, then use list's
    # in-place sort().  By copying *all* incoming data, we avoid
    # possible side effects in the case that the user provided a list.
    ans = list(iterable)
    try:
        ans.sort(key=key, reverse=reverse)
    except:
        ans.sort(key=_robust_sort_keyfcn(key), reverse=reverse)
    return ans
