#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________


__all__ = ['PyomoMIPConverter']

import os
from six import iteritems, PY3
using_py3 = PY3

import pyutilib.services

from pyomo.misc.plugin import *
from pyomo.opt.base import *
from pyomo.solvers.plugins.converter.pico import PicoMIPConverter


class PyomoMIPConverter(SingletonPlugin):

    implements(IProblemConverter)

    pico_converter = PicoMIPConverter()

    def __init__(self,**kwds):
        SingletonPlugin.__init__(self,**kwds)

    def can_convert(self, from_type, to_type):
        """Returns true if this object supports the specified conversion"""
        if from_type != ProblemFormat.pyomo:
            return False
        #
        # Return True for specific from/to pairs
        #
        if to_type in ( ProblemFormat.nl, ProblemFormat.cpxlp,
                        ProblemFormat.osil ):
            return True

        if to_type == ProblemFormat.mps and self.pico_converter.available():
            return True

        return False

    def apply(self, *args, **kwds):
        """
        Generate a NL or LP file from Pyomo, and then do subsequent
        conversions.
        """

        import pyomo.core.scripting.convert

        capabilities = kwds.pop("capabilities", None)

        # all non-consumed keywords are assumed to be options 
        # that should be passed to the writer.
        io_options = {}
        for kwd, value in iteritems(kwds):
            io_options[kwd] = value
        kwds.clear()

        # basestring is gone in Python 3.x, merged with str.
        if using_py3:
            compare_type = str
        else:
            compare_type = basestring

        if isinstance(args[2], compare_type):
            instance = None
        else:
            instance = args[2]

        if args[1] is ProblemFormat.cpxlp:
            problem_filename = pyutilib.services.TempfileManager.create_tempfile(suffix = '.pyomo.lp')
            if instance is not None:
                (problem_filename, varmap) = instance.write(filename=problem_filename,
                                                            format=ProblemFormat.cpxlp,
                                                            solver_capability=capabilities,
                                                            io_options=io_options)
                return (problem_filename,),varmap # no map file is necessary
            else:
                ans = pyomo.core.scripting.convert.pyomo2lp(['--save-model',problem_filename,args[2]])
                if ans.errorcode:
                    raise RuntimeError("pyomo2lp conversion returned nonzero error code (%s)" % ans.errorcode)
                model = ans.retval
                problem_filename = model.filename
                varmap = model.symbol_map
                return (problem_filename,),varmap

        elif args[1] in [ProblemFormat.mps, ProblemFormat.nl]:
            problem_filename = pyutilib.services.TempfileManager.create_tempfile(suffix = '.pyomo.nl')
            if not instance is None:
                (problem_filename, varmap) = instance.write(filename=problem_filename, 
                                                            format=ProblemFormat.nl, 
                                                            solver_capability=capabilities,
                                                            io_options=io_options)
            else:
                ans = pyomo.core.scripting.convert.pyomo2nl(['--save-model',problem_filename,args[2]])
                if ans.errorcode:
                    raise RuntimeError("pyomo2lp conversion returned nonzero error code (%s)" % ans.errorcode)
                model = ans.retval
                problem_filename = model.filename
                varmap = model.symbol_map
            if args[1] is ProblemFormat.nl:
                return (problem_filename,),varmap
            #
            # Convert from NL to MPS
            #
            # TBD: We don't support a variable map file when going from NL to MPS within the PICO converter.
            # NOTE: this is a problem with the MPS writer that is provided by COIN-OR
            # NOTE: we should generalize this so it doesn't strictly depend on the PICO converter utility.
            #
            ans = self.pico_converter.apply(ProblemFormat.nl,ProblemFormat.mps,problem_filename)
            os.remove(problem_filename)
            return ans

        elif args[1] is ProblemFormat.osil:
            problem_filename = pyutilib.services.TempfileManager.create_tempfile(suffix='pyomo.osil')
            if instance:
                (problem_filename, varmap) = instance.write(filename=problem_filename,
                                                            format=ProblemFormat.osil,
                                                            solver_capability=capabilities,
                                                            io_options=io_options)
                return (problem_filename,), None
            else:
                msg = 'There is currently no script conversion available from\n'
                msg += 'Pyomo to OSiL format.'

                raise NotImplementedError(msg)
