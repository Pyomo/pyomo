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

from pyutilib.misc import Options

from pyomo.dataportal.factory import DataManagerFactory
from pyomo.dataportal.process_data import _process_include


@DataManagerFactory.register("dat", "Pyomo data command file interface")
class PyomoDataCommands(object):

    def __init__(self):
        self._info = []
        self.options = Options()

    def available(self):
        return True

    def initialize(self, **kwds):
        self.filename = kwds.pop('filename')
        self.add_options(**kwds)

    def add_options(self, **kwds):
        self.options.update(kwds)

    def open(self):
        if self.filename is None:               #pragma:nocover
            raise IOError("No filename specified")
        if not os.path.exists(self.filename):   #pragma:nocover
            raise IOError("Cannot find file '%s'" % self.filename)

    def close(self):
        pass

    def read(self):
        """
        This function does nothing, since executing Pyomo data commands
        both reads and processes the data all at once.
        """
        pass

    def write(self, data):                      #pragma:nocover
        """
        This function does nothing, because we cannot write to a *.dat file.
        """
        pass

    def process(self, model, data, default):
        """
        Read Pyomo data commands and process the data.
        """
        _process_include(['include', self.filename], model, data, default, self.options)

    def clear(self):
        self._info = []
