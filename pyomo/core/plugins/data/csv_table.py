#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os.path
import csv

from pyomo.util.plugin import alias

from pyomo.core.base.param import Param
from pyomo.core.data.TableData import TableData


class CSVTable(TableData):

    alias("csv", "CSV file interface")

    def __init__(self):
        TableData.__init__(self)

    def open(self):
        if self.filename is None:                       #pragma:nocover
            raise IOError("No filename specified")

    def close(self):
        self.FILE.close()

    def read(self):
        if not os.path.exists(self.filename):           #pragma:nocover
            raise IOError("Cannot find file '%s'" % self.filename)
        self.FILE = open(self.filename, 'r')
        tmp=[]
        for tokens in csv.reader(self.FILE):
            if tokens != ['']:
                tmp.append(tokens)
        self.FILE.close()
        if len(tmp) == 0:
            raise IOError("Empty *.csv file")
        elif len(tmp) == 1:
            if not self.options.param is None:
                if type(self.options.param) in (list, tuple):
                    p = self.options.param[0]
                else:
                    p = self.options.param
                if isinstance(p, Param):
                    self.options.model = p.model()
                    p = p.local_name
                self._info = ["param",p,":=",tmp[0][0]]
            elif len(self.options.symbol_map) == 1:
                self._info = ["param",self.options.symbol_map[self.options.symbol_map.keys()[0]],":=",tmp[0][0]]
            else:
                raise IOError("Data looks like a parameter, but multiple parameter names have been specified: %s" % str(self.options.symbol_map))
        else:
            self._set_data(tmp[0], tmp[1:])

    def write(self, data):
        if self.options.set is None and self.options.param is None:
            raise IOError("Unspecified model component")
        self.FILE = open(self.filename, 'w')
        table = self.get_table()
        writer = csv.writer(self.FILE)
        writer.writerows(table)
        self.FILE.close()

