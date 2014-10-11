#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

import os.path
import re

from pyutilib.misc import Options
from pyomo.util.plugin import alias, Plugin, implements

from pyomo.core.base.plugin import IDataManager
from pyomo.core.data.process_data import _process_include


class AmplDataCommands(Plugin):

    alias("dat", "Import data with AMPL data commands.")

    implements(IDataManager, service=False)

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
        if self.filename is None:
            raise IOError("No filename specified")
        if not os.path.exists(self.filename):
            raise IOError("Cannot find file '%s'" % self.filename)

    def close(self):
        pass

    def read(self):
        """
        This function does nothing, since executing AMPL data commands
        both reads and processes the data all at once.
        """
        return True

    def write(self, data):
        """
        Return False, because we cannot write to a *.dat file.
        """
        return False

    def process(self, model, data, default):
        """
        Read AMPL data commands and process the data.
        """
        _process_include(['include', self.filename], model, data, default, self.options)

    def clear(self):
        self._info = []
