#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['convert_problem']

import copy
import os

from pyomo.opt.base.formats import guess_format
from pyomo.opt.base.error import ConverterError
from pyomo.common import Factory

try:
    unicode
except NameError:
    basestring = unicode = str


# WEH - Should we treat these as singleton objects?  Not for now, since
# I can't think of a case where that would impact performance
ProblemConverterFactory = Factory('problem converter')


def convert_problem(args,
                    target_problem_type,
                    valid_problem_types,
                    has_capability=lambda x: False,
                    **kwds):
    """
    Convert a problem, defined by the 'args' tuple, into another
    problem.
    """

    if len(valid_problem_types) == 0:
        raise ConverterError("No valid problem types")

    if not (target_problem_type is None or \
             target_problem_type in valid_problem_types):
        msg = "Problem type '%s' is not valid"
        raise ConverterError(msg % str( target_problem_type ))

    if len(args) == 0:
        raise ConverterError("Empty argument list")

    #
    # Setup list of source problem types
    #
    tmp = args[0]
    if isinstance(tmp,basestring):
        fname = tmp.split(os.sep)[-1]
        if os.sep in fname:   #pragma:nocover
            fname = tmp.split(os.sep)[-1]
        source_ptype = [guess_format(fname)]
        if source_ptype is [None]:
            raise ConverterError("Unknown suffix type: "+tmp)
    else:
        source_ptype = args[0].valid_problem_types()

    #
    # Setup list of valid problem types
    #
    valid_ptypes = copy.copy(valid_problem_types)
    if target_problem_type is not None:
        valid_ptypes.remove(target_problem_type)
        valid_ptypes = [target_problem_type]  + valid_ptypes
    if source_ptype[0] in valid_ptypes:
        valid_ptypes.remove(source_ptype[0])
        valid_ptypes = [source_ptype[0]]  + valid_ptypes

    #
    # Iterate over the valid problem types, starting with the target type
    #
    # Apply conversion and return for first match
    #
    for ptype in valid_ptypes:

        for s_ptype in source_ptype:

        #
        # If the source and target types are equal, then simply the return
        # the args (return just the first element of the tuple if it has length
        # one.
        #
            if s_ptype == ptype:
                return (args,ptype,None)
            #
            # Otherwise, try to convert
            #
            for name in ProblemConverterFactory:

                converter = ProblemConverterFactory(name)
                if converter.can_convert(s_ptype, ptype):
                    tmp = [s_ptype,ptype] + list(args)
                    tmp = tuple(tmp)
                    # propagate input keywords to the converter
                    tmpkw = kwds
                    tmpkw['capabilities'] = has_capability
                    problem_files, symbol_map = converter.apply(*tmp, **tmpkw)
                    return problem_files, ptype, symbol_map

    msg = 'No conversion possible.  Source problem type: %s.  Valid target '  \
          'types: %s'
    raise ConverterError(msg % (str(source_ptype[0]), list(map(str, valid_ptypes))))
