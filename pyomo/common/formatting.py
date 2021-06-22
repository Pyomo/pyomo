#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import types
from pyomo.common.sorting import sorted_robust

def tostr(value, quote_str=False):
    """Convert a value to a string

    
    """
    if isinstance(value, list):
        # Override the generation of str(list), but only if the object
        # is using the default implementation of list.__str__
        if value.__class__.__str__ is list.__str__:
            return "[%s]" % (', '.join(tostr(v, True) for v in value))
    elif isinstance(value, tuple):
        # Override the generation of str(list), but only if the object
        # is using the default implementation of list.__str__
        if value.__class__.__str__ is tuple.__str__:
            if len(value) == 1:
                return "(%s,)" % (tostr(value[0], True),)
            return "(%s)" % (', '.join(tostr(v, True) for v in value))
    elif isinstance(value, dict):
        # Override the generation of str(), but only if the object
        # is using the default implementation of list.__str__
        if value.__class__.__str__ is dict.__str__:
            return "{%s}" % (', '.join(
                '%s: %s' % (tostr(k, True), tostr(v, True))
                for k, v in value.items()
            ))
    elif isinstance(value, str):
        if quote_str:
            return repr(value)
        else:
            return value

    return str(value)
    

def _to_ustr(obj):
    if not isinstance(obj, str):
        try:
            obj = tostr(obj)
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
            for x in _rows.values() if x is not None
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

