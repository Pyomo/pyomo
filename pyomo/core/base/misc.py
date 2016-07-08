#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['display']

import logging
import sys
import types

from six import itervalues

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
            elif index is None:
                return rule(model)
            else:
                return rule(model, index)
        else:
            if index.__class__ is tuple:
                return rule(model, *index, **options)
            elif index is None:
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
                elif index is None:
                    return rule(model)
                else:
                    return rule(model, index)
            else:
                if index.__class__ is tuple:
                    return rule(model, *index, **options)
                elif index is None:
                    return rule(model, **options)
                else:
                    return rule(model, index, **options)

def apply_parameterized_indexed_rule(obj, rule, model, param, index):
    if index.__class__ is tuple:
        return rule(model, param, *index)
    if index is None:
        return rule(model, param)
    return rule(model, param, index)

def _safe_to_str(obj):
    try:
        return str(obj)
    except:
        return "None"

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

    _rows = {}
    #_header = ("Key","Initial Value","Lower Bound","Upper Bound",
    #           "Current Value","Fixed","Stale")
    # NB: _width is a list because we will change these values
    if header:
        header = ('Key',) + tuple(header)
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
                ((_safe_to_str("" if i else _key),) if header else ()) +
                tuple( _safe_to_str(x) for x in _r )
                for i,_r in enumerate(_rowSet) ]
        else:
            _rows[_key] = [
                ((_safe_to_str(_key),) if header else ()) +
                tuple( _safe_to_str(x) for x in _rowSet) ]

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
    for _key in sorted(_rows):
        _rowSet = _rows[_key]
        if not _rowSet:
            _rowSet = [ [_key] + [None]*(len(_width)-1) ]
        for _data in _rowSet:
            ostream.write(
                prefix
                + " : ".join( _width[i] % x for i,x in enumerate(_data) )
                + "\n")

