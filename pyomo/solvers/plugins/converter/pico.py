#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import six

import os.path

import pyutilib.common
import pyutilib.subprocess
import pyomo.common

from pyomo.opt.base import *


class PicoMIPConverter(object):

    def can_convert(self, from_type, to_type):
        """Returns true if this object supports the specified conversion"""
        #
        # Test if the glpsol executable is available
        #
        if pyomo.common.registered_executable("pico_convert") is None:
            return False
        #
        # Return True for specific from/to pairs
        #
        if from_type == ProblemFormat.nl and to_type == ProblemFormat.cpxlp:
            return True
        if from_type == ProblemFormat.nl and to_type == ProblemFormat.mps:
            return True
        if from_type == ProblemFormat.mps and to_type == ProblemFormat.cpxlp:
            return True
        if from_type == ProblemFormat.cpxlp and to_type == ProblemFormat.mps:
            return True
        return False

    def available(self):
        cmd = pyomo.common.registered_executable("pico_convert")
        return not cmd is None

    def apply(self, *args, **kwargs):
        """
        Run the external pico_convert utility
        """
        if len(args) != 3:
            raise ConverterError("Cannot apply pico_convert with more than one filename or model")
        cmd = pyomo.common.registered_executable("pico_convert")
        if cmd is None:
            raise ConverterError("The 'pico_convert' application cannot be found")

        pico_convert_cmd = cmd.get_path()
        target=str(args[1])
        if target=="cpxlp":
            target="lp"
        # NOTE: if you have an extra "." in the suffix, the pico_convert program fails to output to the correct filename.
        output_filename = pyutilib.services.TempfileManager.create_tempfile(suffix = 'pico_convert.' + target)
        if not isinstance(args[2],six.string_types):
            fname= pyutilib.services.TempfileManager.create_tempfile(suffix= 'pico_convert.' +str(args[0]))
            args[2].write(filename=fname, format=args[1])
            cmd = pico_convert_cmd +" --output="+output_filename+" "+target+" "+fname
        else:
            cmd = pico_convert_cmd +" --output="+output_filename+" "+target
            for item in args[2:]:
                if not os.path.exists(item):
                    raise ConverterError("File "+item+" does not exist!")
                cmd = cmd + " "+item
        print("Running command: "+cmd)
        pyutilib.subprocess.run(cmd)
        if not os.path.exists(output_filename):       #pragma:nocover
            raise pyutilib.common.ApplicationError(\
                    "Problem launching 'pico_convert' to create "+output_filename)
        return (output_filename,),None # no variable map at the moment
