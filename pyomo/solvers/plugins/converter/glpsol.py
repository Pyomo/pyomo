#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['GlpsolMIPConverter']

import os
import six

import pyutilib.subprocess
import pyutilib.common
import pyutilib.services
from pyomo.opt.base import *
from pyomo.util.plugin import *


class GlpsolMIPConverter(SingletonPlugin):

    implements(IProblemConverter)

    def __init__(self,**kwds):
        SingletonPlugin.__init__(self,**kwds)

    def can_convert(self, from_type, to_type):
        """Returns true if this object supports the specified conversion"""
        #
        # Test if the glpsol executable is available
        #
        if pyutilib.services.registered_executable("glpsol") is None:
            return False
        #
        # Return True for specific from/to pairs
        #
        if from_type == ProblemFormat.mod and to_type == ProblemFormat.cpxlp:
            return True
        if from_type == ProblemFormat.mod and to_type == ProblemFormat.mps:
            if pyutilib.services.registered_executable("ampl") is None:
                #
                # Only convert mod->mps with ampl
                #
                return True
        return False

    def apply(self, *args, **kwargs):
        """Convert an instance of one type into another"""
        if not isinstance(args[2],six.string_types):
            raise ConverterError("Can only apply glpsol to convert file data")
        cmd = pyutilib.services.registered_executable("glpsol").get_path()
        if cmd is None:
            raise ConverterError("The 'glpsol' executable cannot be found")
        cmd = cmd +" --math"
        #
        # MPS->LP conversion is ignored in coverage because it's not being
        #   used; instead, we're using pico_convert for this conversion
        #
        modfile=''
        if args[1] == ProblemFormat.mps: #pragma:nocover
            ofile = pyutilib.services.TempfileManager.create_tempfile(suffix = '.glpsol.mps')
            cmd = cmd + " --check --name 'MPS model derived from "+os.path.basename(args[2])+"' --wfreemps "+ofile
        elif args[1] == ProblemFormat.cpxlp:
            ofile = pyutilib.services.TempfileManager.create_tempfile(suffix = '.glpsol.lp')
            cmd = cmd + " --check --name 'MPS model derived from "+os.path.basename(args[2])+"' --wcpxlp "+ofile
        if len(args[2:]) == 1:
            cmd = cmd+" "+args[2]
        else:
            #
            # Create a temporary model file, since GLPSOL can only
            # handle one input file
            #
            modfile = pyutilib.services.TempfileManager.create_tempfile(suffix = '.glpsol.mod')
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
            cmd = cmd+" "+modfile
        pyutilib.subprocess.run(cmd)
        if not os.path.exists(ofile):       #pragma:nocover
            raise pyutilib.common.ApplicationError("Problem launching 'glpsol' to create "+ofile)
        if os.path.exists(modfile):
            os.remove(modfile)
        return (ofile,),None # empty variable map
