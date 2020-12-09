#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


import os
from six import iteritems, PY3

from pyomo.common.tempfiles import TempfileManager
from pyomo.opt.base import ProblemFormat
from pyomo.opt.base.convert import ProblemConverterFactory
from pyomo.solvers.plugins.converter.pico import PicoMIPConverter
from pyomo.core.kernel.block import IBlock


@ProblemConverterFactory.register('pyomo')
class PyomoMIPConverter(object):

    pico_converter = PicoMIPConverter()

    def can_convert(self, from_type, to_type):
        """Returns true if this object supports the specified conversion"""
        if from_type != ProblemFormat.pyomo:
            return False
        #
        # Return True for specific from/to pairs
        #
        if to_type in (ProblemFormat.nl,
                       ProblemFormat.cpxlp,
                       ProblemFormat.osil,
                       ProblemFormat.bar,
                       ProblemFormat.mps):
            return True

        return False

    def apply(self, *args, **kwds):
        """
        Generate a NL or LP file from Pyomo, and then do subsequent
        conversions.
        """

        import pyomo.scripting.convert

        capabilities = kwds.pop("capabilities", None)

        # all non-consumed keywords are assumed to be options
        # that should be passed to the writer.
        io_options = {}
        for kwd, value in iteritems(kwds):
            io_options[kwd] = value
        kwds.clear()

        # basestring is gone in Python 3.x, merged with str.
        if PY3:
            compare_type = str
        else:
            compare_type = basestring

        if isinstance(args[2], compare_type):
            instance = None
        else:
            instance = args[2]

        if args[1] == ProblemFormat.cpxlp:
            problem_filename = TempfileManager.\
                               create_tempfile(suffix = '.pyomo.lp')
            if instance is not None:
                if isinstance(instance, IBlock):
                    symbol_map_id = instance.write(
                        problem_filename,
                        format=ProblemFormat.cpxlp,
                        _solver_capability=capabilities,
                        _called_by_solver=True,
                        **io_options)
                else:
                    (problem_filename, symbol_map_id) = \
                        instance.write(
                            filename=problem_filename,
                            format=ProblemFormat.cpxlp,
                            solver_capability=capabilities,
                            io_options=io_options)
                return (problem_filename,), symbol_map_id
            else:

                #
                # I'm simply exposing a fatal issue with
                # this code path. How would we convert the
                # collected keywords into command-line
                # arguments that can be sent to the writer?
                #
                if len(io_options):
                    raise ValueError(
                        "The following io_options will be ignored "
                        "(please create a bug report):\n\t" +
                        "\n\t".join("%s = %s" % (k,v)
                                    for k,v in iteritems(io_options)))

                ans = pyomo.scripting.convert.\
                      pyomo2lp(['--output',problem_filename,args[2]])
                if ans.errorcode:
                    raise RuntimeError("pyomo2lp conversion "
                                       "returned nonzero error code "
                                       "(%s)" % ans.errorcode)

                model = ans.retval
                problem_filename = model.filename
                symbol_map = model.symbol_map
                return (problem_filename,),symbol_map

        elif args[1] == ProblemFormat.bar:
            problem_filename = TempfileManager.\
                               create_tempfile(suffix = '.pyomo.bar')
            if instance is not None:
                if isinstance(instance, IBlock):
                    symbol_map_id = instance.write(
                        problem_filename,
                        format=ProblemFormat.bar,
                        _solver_capability=capabilities,
                        _called_by_solver=True,
                        **io_options)
                else:
                    (problem_filename, symbol_map_id) = \
                        instance.write(
                            filename=problem_filename,
                            format=ProblemFormat.bar,
                            solver_capability=capabilities,
                            io_options=io_options)
                return (problem_filename,), symbol_map_id
            else:

                #
                # I'm simply exposing a fatal issue with
                # this code path. How would we convert the
                # collected keywords into command-line
                # arguments that can be sent to the writer?
                #
                if len(io_options):
                    raise ValueError(
                        "The following io_options will be ignored "
                        "(please create a bug report):\n\t" +
                        "\n\t".join("%s = %s" % (k,v)
                                    for k,v in iteritems(io_options)))

                ans = pyomo.scripting.convert.\
                      pyomo2bar(['--output',problem_filename,args[2]])
                if ans.errorcode:
                    raise RuntimeError("pyomo2bar conversion "
                                       "returned nonzero error code "
                                       "(%s)" % ans.errorcode)
                model = ans.retval
                problem_filename = model.filename
                symbol_map = model.symbol_map
                return (problem_filename,),symbol_map

        elif args[1] in [ProblemFormat.mps, ProblemFormat.nl]:
            if args[1] == ProblemFormat.nl:
                problem_filename = TempfileManager.\
                                   create_tempfile(suffix = '.pyomo.nl')
                if io_options.get("symbolic_solver_labels", False):
                    TempfileManager.add_tempfile(
                        problem_filename[:-3]+".row",
                        exists=False)
                    TempfileManager.add_tempfile(
                        problem_filename[:-3]+".col",
                        exists=False)
            else:
                assert args[1] == ProblemFormat.mps
                problem_filename = TempfileManager.\
                                   create_tempfile(suffix = '.pyomo.mps')
            if instance is not None:
                if isinstance(instance, IBlock):
                    symbol_map_id = instance.write(
                        problem_filename,
                        format=args[1],
                        _solver_capability=capabilities,
                        _called_by_solver=True,
                        **io_options)
                else:
                    (problem_filename, symbol_map_id) = \
                        instance.write(
                            filename=problem_filename,
                            format=args[1],
                            solver_capability=capabilities,
                            io_options=io_options)
                return (problem_filename,), symbol_map_id
            else:

                #
                # I'm simply exposing a fatal issue with
                # this code path. How would we convert the
                # collected keywords into command-line
                # arguments that can be sent to the writer?
                #
                if len(io_options):
                    raise ValueError(
                        "The following io_options will be ignored "
                        "(please create a bug report):\n\t" +
                        "\n\t".join("%s = %s" % (k,v)
                                    for k,v in iteritems(io_options)))

                ans = pyomo.scripting.convert.\
                      pyomo2nl(['--output',problem_filename,args[2]])
                if ans.errorcode:
                    raise RuntimeError("pyomo2nl conversion "
                                       "returned nonzero error "
                                       "code (%s)" % ans.errorcode)
                model = ans.retval
                problem_filename = model.filename
                symbol_map = model.symbol_map

                if args[1] == ProblemFormat.nl:
                    return (problem_filename,),symbol_map
                #
                # Convert from NL to MPS
                #
                # TBD: We don't support a variable map file when going
                #      from NL to MPS within the PICO converter.
                # NOTE: this is a problem with the MPS writer that is
                #       provided by COIN-OR
                # NOTE: we should generalize this so it doesn't strictly
                #       depend on the PICO converter utility.
                #
                ans = self.pico_converter.apply(ProblemFormat.nl,
                                                ProblemFormat.mps,
                                                problem_filename)
                os.remove(problem_filename)
                return ans

        elif args[1] == ProblemFormat.osil:
            if False:
                problem_filename = TempfileManager.\
                               create_tempfile(suffix='pyomo.osil')
                if instance:
                    if isinstance(instance, IBlock):
                        symbol_map_id = instance.write(
                            problem_filename,
                            format=ProblemFormat.osil,
                            _solver_capability=capabilities,
                            _called_by_solver=True,
                            **io_options)
                    else:
                        (problem_filename, symbol_map_id) = \
                            instance.write(
                                filename=problem_filename,
                                format=ProblemFormat.osil,
                                solver_capability=capabilities,
                                io_options=io_options)
                    return (problem_filename,), None
            else:
                raise NotImplementedError(
                    "There is currently no "
                    "script conversion available from "
                    "Pyomo to OSiL format.")
