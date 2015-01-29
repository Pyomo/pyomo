#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for pyomo.opt.opt_config
#

import os
from os.path import abspath, dirname
pyomodir = dirname(abspath(__file__))+os.sep+".."+os.sep+".."+os.sep
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.services
import pyutilib.misc

old_tempdir = pyutilib.services.TempfileManager.tempdir

class OptConfigDebug(unittest.TestCase):

    def setUp(self):
        pyutilib.services.TempfileManager.tempdir = currdir

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()
        pyutilib.services.TempfileManager.tempdir = old_tempdir

    def Xtest_config1(self):
        """
        Read in config file opt1.cfg
        """
        import pyutilib.component.app
        app = pyutilib.component.app.SimpleApplication("testapp")
        #app.config.summarize()
        app.save_configuration(currdir+"opt1-out.cfg")
        app.configure(currdir+"opt1.cfg")
        if pyutilib.services.registered_executable("pico_convert"):
            self.assertEqual( pyutilib.services.registered_executable("pico_convert").get_path(), pyutilib.misc.search_file("pico_convert"))
        if pyutilib.services.registered_executable("glpsol"):
            self.assertEqual( pyutilib.services.registered_executable("glpsol").get_path(), pyutilib.misc.search_file("glpsol"))
        if pyutilib.services.registered_executable("ampl"):
            self.assertEqual( pyutilib.services.registered_executable("ampl").get_path(), pyutilib.misc.search_file("ampl"))
        if pyutilib.services.registered_executable("timer"):
            self.assertEqual( pyutilib.services.registered_executable("timer").get_path(), pyutilib.misc.search_file("timer"))


if __name__ == "__main__":
    unittest.main()
