#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['display']

import logging
import sys
import types

from six import itervalues, string_types

logger = logging.getLogger('pyomo.core')


def display(obj, ostream=None):
    """ Display data in a Pyomo object"""
    if ostream is None:
        ostream = sys.stdout
    try:
        display_fcn = obj.display
    except AttributeError:
        raise TypeError(
            "Error trying to display values for object of type %s:\n"
            "\tObject does not support the 'display()' method"
            % (type(obj), ) )
    try:
        display_fcn(ostream=ostream)
    except Exception:
        err = sys.exc_info()[1]
        logger.error(
            "Error trying to display values for object of type %s:\n\t%s"
            % (type(obj), err) )
        raise


def create_name(name, ndx):
    """Create a canonical name for a component using the given index"""
    if ndx is None:
        return name
    if type(ndx) is tuple:
        tmp = str(ndx).replace(', ',',')
        return name+"["+tmp[1:-1]+"]"
    return name+"["+str(ndx)+"]"


def apply_indexed_rule(obj, rule, model, index, options=None):
    try:
        if options is None:
            if index.__class__ is tuple:
                return rule(model, *index)
            elif index is None and not obj.is_indexed():
                return rule(model)
            else:
                return rule(model, index)
        else:
            if index.__class__ is tuple:
                return rule(model, *index, **options)
            elif index is None and not obj.is_indexed():
                return rule(model, **options)
            else:
                return rule(model, index, **options)
    except TypeError:
        try:
            if options is None:
                return rule(model)
            else:
                return rule(model, **options)
        except:
            # Nothing appears to have matched... re-trigger the original
            # TypeError
            if options is None:
                if index.__class__ is tuple:
                    return rule(model, *index)
                elif index is None and not obj.is_indexed():
                    return rule(model)
                else:
                    return rule(model, index)
            else:
                if index.__class__ is tuple:
                    return rule(model, *index, **options)
                elif index is None and not obj.is_indexed():
                    return rule(model, **options)
                else:
                    return rule(model, index, **options)

def apply_parameterized_indexed_rule(obj, rule, model, param, index):
    if index.__class__ is tuple:
        return rule(model, param, *index)
    if index is None:
        return rule(model, param)
    return rule(model, param, index)


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
    def __init__(self):
        self._typemap = {}

    def __call__(self, val):
        """Generate a tuple ( str(type_name), val ) for sorting the value.

        `key=` expects a function.  We are generating a functor so we
        have a convenient place to store the _typemap, which converts
        the type-specific functions for converting a value to the second
        argument of the sort key.

        """
        try:
            i, _typename = self._typemap[val.__class__]
        except KeyError:
            # If this is not a type we have seen before, determine what
            # to use for the second value in the tuple.
            _type = val.__class__
            _typename = _type.__name__
            try:
                # 1: Check if the type is comparable.  In Python 3, sorted()
                #    uses "<" to compare objects.
                val < val
                i = 1
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
        if i == 1:
            return _typename, val
        elif i == 2:
            return _typename, str(val)
        else:
            return _typename, id(val)


def sorted_robust(arg):
    """Utility to sort an arbitrary iterable.

    This returns the sorted(arg) in a consistent order by first tring
    the standard sorted() function, and if that fails (for example with
    mixed-type Sets in Python3), use the _robust_sort_keyfcn utility
    (above) to generate sortable keys.

    """
    # It is possible that arg is a generator.  We need to cache the
    # elements returned by the generator in case 'sorted' raises an
    # exception (this ensures we don't loose any elements).  Howevver,
    # if we were passed a list, we do not want to make an unnecessary
    # copy.  Tuples are OK because tuple(a) will not copy a if it is
    # already a tuple.
    if type(arg) is not list:
        arg = tuple(arg)
    try:
        return sorted(arg)
    except:
        return sorted(arg, key=_robust_sort_keyfcn())


def _to_ustr(obj):
    if not isinstance(obj, string_types):
        try:
            obj = str(obj)
        except:
            return u"None"
    # If this is a Python 2.x string, then we want to decode it to a
    # proper unicode string so that len() counts embedded multibyte
    # characters as a single codepoint (so the resulting tabular
    # alignment is correct)
    if hasattr(obj, 'decode'):
        return obj.decode('utf-8')
    return obj

def tabular_writer(ostream, prefix, data, header, row_generator):
    """Output data in tabular form

    Parameters:
    - ostream: the stream to write to
    - prefix:  prefix each line with this string
    - data:    a generator returning (key, value) pairs (e.g., from iteritems())
    - header:  list of column header strings
    - row_generator: a generator that returns tuples of values for each
      line of the table
    """

    prefix = _to_ustr(prefix)

    _rows = {}
    #_header = ("Key","Initial Value","Lower Bound","Upper Bound",
    #           "Current Value","Fixed","Stale")
    # NB: _width is a list because we will change these values
    if header:
        header = (u"Key",) + tuple(_to_ustr(x) for x in header)
        _width = [len(x) for x in header]
    else:
        _width = None

    for _key, _val in data:
        try:
            _rowSet = row_generator(_key, _val)
        except ValueError:
            _rows[_key] = None
            continue

        if isinstance(_rowSet, types.GeneratorType):
            _rows[_key] = [
                ((_to_ustr("" if i else _key),) if header else ()) +
                tuple( _to_ustr(x) for x in _r )
                for i,_r in enumerate(_rowSet) ]
        else:
            _rows[_key] = [
                ((_to_ustr(_key),) if header else ()) +
                tuple( _to_ustr(x) for x in _rowSet) ]

        for _row in _rows[_key]:
            if not _width:
                _width = [0]*len(_row)
            for _id, x in enumerate(_row):
                _width[_id] = max(_width[_id], len(x))

    # NB: left-justify header
    if header:
        # Note: do not right-pad the last header with unnecessary spaces
        tmp = _width[-1]
        _width[-1] = 0
        ostream.write(prefix
                      + " : ".join( "%%-%ds" % _width[i] % x
                                    for i,x in enumerate(header) )
                      + "\n")
        _width[-1] = tmp

    # If there is no data, we are done...
    if not _rows:
        return

    # right-justify data, except for the last column if there are spaces
    # in the data (probably an expression or vector)
    _width = ["%"+str(i)+"s" for i in _width]

    if any( ' ' in r[-1]
            for x in itervalues(_rows) if x is not None
            for r in x  ):
        _width[-1] = '%s'
    for _key in sorted_robust(_rows):
        _rowSet = _rows[_key]
        if not _rowSet:
            _rowSet = [ [_key] + [None]*(len(_width)-1) ]
        for _data in _rowSet:
            ostream.write(
                prefix
                + " : ".join( _width[i] % x for i,x in enumerate(_data) )
                + "\n")

