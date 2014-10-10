#  _________________________________________________________________________
#
#  Coopr: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Coopr README.txt file.
#  _________________________________________________________________________

import sys
import os.path
try:
    import win32com
    win32com_available=True
except ImportError:
    win32com_available=False

from pyutilib.excel import ExcelSpreadsheet
import pyutilib.common

from coopr.core.plugin import alias
from coopr.pyomo.data.TableData import TableData

from coopr.pyomo.plugins.data.db_table import pyodbc_db_Table


class SheetTable(TableData):

    alias("win32com", "Manage IO with Excel XLS files.")

    def __init__(self):
        TableData.__init__(self)

    def open(self):
        if self.filename is None:
            raise IOError("No filename specified")
        if not os.path.exists(self.filename):
            raise IOError("Cannot find file '%s'" % self.filename)
        self.sheet = None
        if self._data is not None:
            self.sheet = self._data
        else:
            try:
                self.sheet = ExcelSpreadsheet(self.filename)
            except pyutilib.common.ApplicationError:
                raise

    def read(self):
        if self.sheet is None:
            return
        tmp = self.sheet.get_range(self.options.range, raw=True)
        if type(tmp) in (int,long,float):
            if not self.options.param is None:
                self._info = ["param",self.options.param.name,":=",tmp]
            elif len(self.options.symbol_map) == 1:
                self._info = ["param",self.options.symbol_map[self.options.symbol_map.keys()[0]],":=",tmp]
            else:
                raise IOError("Data looks like a parameter, but multiple parameter names have been specified: %s" % str(self.options.symbol_map))
        elif len(tmp) == 0:
            raise IOError("Empty range '%s'" % self.options.range)
        else:
            tmp = [list(x) for x in tmp]
            self._set_data(tmp[0], tmp[1:])
        return True

    def close(self):
        if self._data is None and not self.sheet is None:
            del self.sheet




if win32com_available:

    class SheetTable_xls(SheetTable):

        alias("xls", "Manage IO with Excel XLS files.")

        def __init__(self):
            SheetTable.__init__(self)

        def available(self):
            return win32com_available

        def requirements(self):
            return "win32com"

else:

    class pyodbc_xls(pyodbc_db_Table):

        alias("xls", "Manage IO with Excel XLS files.")

        def __init__(self):
            pyodbc_db_Table.__init__(self)

        def requirements(self):
            return "win32com or pyodbc"


class SheetTable_xlsx(pyodbc_db_Table):

    alias("xlsx", "Manage IO with Excel XLSX files.")

    def __init__(self):
        pyodbc_db_Table.__init__(self)

    def requirements(self):
        return "win32com or pyodbc"


class SheetTable_xlsb(pyodbc_db_Table):

    alias("xlsb", "Manage IO with Excel XLSB files.")

    def __init__(self):
        pyodbc_db_Table.__init__(self)

    def requirements(self):
        return "win32com or pyodbc"


class SheetTable_xlsm(pyodbc_db_Table):

    alias("xlsm", "Manage IO with Excel XLSM files.")

    def __init__(self):
        pyodbc_db_Table.__init__(self)

    def requirements(self):
        return "win32com or pyodbc"


