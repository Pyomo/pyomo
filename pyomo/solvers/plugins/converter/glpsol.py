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
import six

import pyutilib.subprocess
import pyomo.common
from pyomo.common.errors import ApplicationError
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt.base import ProblemFormat, ConverterError
from pyomo.opt.base.convert import ProblemConverterFactory


@ProblemConverterFactory.register('glpsol')
class GlpsolMIPConverter(object):

    def can_convert(self, from_type, to_type):
        """Returns true if this object supports the specified conversion"""
        #
        # Test if the glpsol executable is available
        #
        if not pyomo.common.Executable("glpsol"):
            return False
        #
        # Return True for specific from/to pairs
        #
        if from_type == ProblemFormat.mod and to_type == ProblemFormat.cpxlp:
            return True
        if from_type == ProblemFormat.mod and to_type == ProblemFormat.mps:
            if not pyomo.common.Executable("ampl"):
                #
                # Only convert mod->mps with ampl
                #
                return True
        return False

    def apply(self, *args, **kwargs):
        """Convert an instance of one type into another"""
        if not isinstance(args[2],six.string_types):
            raise ConverterError("Can only apply glpsol to convert file data")
        _exec = pyomo.common.Executable("glpsol")
        if not _exec:
            raise ConverterError("The 'glpsol' executable cannot be found")
        cmd = [_exec.path(), "--math"]
        #
        # MPS->LP conversion is ignored in coverage because it's not being
        #   used; instead, we're using pico_convert for this conversion
        #
        modfile=''
        if args[1] == ProblemFormat.mps: #pragma:nocover
            ofile = TempfileManager.create_tempfile(suffix = '.glpsol.mps')
            cmd.extend([
                "--check",
                "--name", "MPS model derived from "+os.path.basename(args[2]),
                "--wfreemps", ofile
            ])
        elif args[1] == ProblemFormat.cpxlp:
            ofile = TempfileManager.create_tempfile(suffix = '.glpsol.lp')
            cmd.extend([
                "--check",
                "--name","MPS model derived from "+os.path.basename(args[2]),
                "--wcpxlp", ofile
            ])
        if len(args[2:]) == 1:
            cmd.append(args[2])
        else:
            #
            # Create a temporary model file, since GLPSOL can only
            # handle one input file
            #
            modfile = TempfileManager.create_tempfile(suffix = '.glpsol.mod')
            OUTPUT=open(modfile,"w")
            flag=False
            #
            # Read the model file
            #
            INPUT= open(args[2])
            for line in INPUT:
                line = line.strip()
                if line == "data;":
                    raise ConverterError("Problem composing mathprog model and data files - mathprog file already has data in it!")
                if line != "end;":
                    OUTPUT.write(line+'\n')
            INPUT.close()
            OUTPUT.write("data;\n")
            #
            # Read the data files
            #
            for file in args[3:]:
                INPUT= open(file)
                for line in INPUT:
                    line = line.strip()
                    if line != "end;" and line != "data;":
                        OUTPUT.write(line+'\n')
                INPUT.close()
                OUTPUT.write("end;\n")
            OUTPUT.close()
            cmd.append(modfile)
        pyutilib.subprocess.run(cmd)
        if not os.path.exists(ofile):       #pragma:nocover
            raise ApplicationError("Problem launching 'glpsol' to create "+ofile)
        if os.path.exists(modfile):
            os.remove(modfile)
        return (ofile,),None # empty variable map
