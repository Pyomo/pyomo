#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os.path
import pyutilib.subprocess
import pyutilib.common
import pyomo.common

from pyomo.opt.base import *
from pyomo.opt.base.convert import ProblemConverterFactory

try:
    unicode
except:
    basestring = unicode = str


@ProblemConverterFactory.register('ampl')
class AmplMIPConverter(object):

    def can_convert(self, from_type, to_type):
        """Returns true if this object supports the specified conversion"""
        #
        # Test if the ampl executable is available
        #
        if pyomo.common.registered_executable("ampl") is None:
            return False
        #
        # Return True for specific from/to pairs
        #
        if from_type == ProblemFormat.mod and to_type == ProblemFormat.nl:
            return True
        if from_type == ProblemFormat.mod and to_type == ProblemFormat.mps:
            return True
        return False

    def apply(self, *args, **kwargs):
        """Convert an instance of one type into another"""
        if not isinstance(args[2],basestring):
            raise ConverterError("Can only apply ampl to convert file data")
        cmd = pyomo.common.registered_executable("ampl").get_path()
        if cmd is None:
            raise ConverterError("The 'ampl' executable cannot be found")
        script_filename = pyutilib.services.TempfileManager.create_tempfile(suffix = '.ampl')

        if args[1] == ProblemFormat.nl:
            output_filename = pyutilib.services.TempfileManager.create_tempfile(suffix = '.nl')
        else:
            output_filename = pyutilib.services.TempfileManager.create_tempfile(suffix = '.mps')

        cmd += " " + script_filename
        #
        # Create the AMPL script
        #
        OUTPUT = open(script_filename, 'w')
        OUTPUT.write("#\n")
        OUTPUT.write("# AMPL script for converting the following files\n")
        OUTPUT.write("#\n")
        if len(args[2:]) == 1:
            OUTPUT.write('model '+args[2]+";\n")
        else:
            OUTPUT.write('model '+args[2]+";\n")
            OUTPUT.write('data '+args[3]+";\n")
        abs_ofile = os.path.abspath(output_filename)
        if args[1] == ProblemFormat.nl:
            OUTPUT.write('write g'+abs_ofile[:-3]+";\n")
        else:
            OUTPUT.write('write m'+abs_ofile[:-4]+";\n")
        OUTPUT.close()
        #
        # Execute command and cleanup
        #
        output = pyutilib.subprocess.run(cmd)
        if not os.path.exists(output_filename):       #pragma:nocover
            raise pyutilib.common.ApplicationError("Problem launching 'ampl' to create '%s': %s" % (output_filename, output))
        return (output_filename,),None # empty variable map
