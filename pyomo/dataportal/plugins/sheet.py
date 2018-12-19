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
import six
from pyutilib.excel import ExcelSpreadsheet
import pyutilib.common

try:
    import win32com
    win32com_available=True
except ImportError:
    win32com_available=False
_excel_available = False  #pragma:nocover
if win32com_available:
    from pyutilib.excel.spreadsheet_win32com import ExcelSpreadsheet_win32com
    tmp = ExcelSpreadsheet_win32com()
    try:
        tmp._excel_dispatch()
        tmp._excel_quit()
        _excel_available = True
    except:
        pass
try:
    import openpyxl
    openpyxl_available=True
except (ImportError, SyntaxError):
    # Some versions of openpyxl contain python2.6-incompatible syntax
    openpyxl_available=False
try:
    import xlrd
    xlrd_available=True
except ImportError:
    xlrd_available=False

from pyomo.dataportal import TableData
from pyomo.dataportal.plugins.db_table import pyodbc_available, pyodbc_db_Table, pypyodbc_available, pypyodbc_db_Table
from pyomo.dataportal.factory import DataManagerFactory


class SheetTable(TableData):

    def __init__(self, ctype=None):
        TableData.__init__(self)
        self.ctype=ctype

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
                self.sheet = ExcelSpreadsheet(self.filename, ctype=self.ctype)
            except pyutilib.common.ApplicationError:
                raise

    def read(self):
        if self.sheet is None:
            return
        tmp = self.sheet.get_range(self.options.range, raw=True)
        if type(tmp) is float or type(tmp) in six.integer_types:
            if not self.options.param is None:
                self._info = ["param"] + list(self.options.param) + [":=",tmp]
            elif len(self.options.symbol_map) == 1:
                self._info = ["param",self.options.symbol_map[self.options.symbol_map.keys()[0]],":=",tmp]
            else:
                raise IOError("Data looks like a parameter, but multiple parameter names have been specified: %s" % str(self.options.symbol_map))
        elif len(tmp) == 0:
            raise IOError("Empty range '%s'" % self.options.range)
        else:
            if type(tmp[1]) in (list,tuple):
                tmp_ = tmp[1:]
            else:
                tmp_ = [[x] for x in tmp[1:]]
            self._set_data(tmp[0], tmp_)

    def close(self):
        if self._data is None and not self.sheet is None:
            del self.sheet




if pyodbc_available or not pypyodbc_available:
    pyodbc_db_base = pyodbc_db_Table
else:
    pyodbc_db_base = pypyodbc_db_Table

#
# FIXME: The pyodbc interface doesn't work right now.  We will disable it.
#
pyodbc_available = False

if (win32com_available and _excel_available) or xlrd_available:

    @DataManagerFactory.register("xls", "Excel XLS file interface")
    class SheetTable_xls(SheetTable):

        def __init__(self):
            if win32com_available and _excel_available:
                SheetTable.__init__(self, ctype='win32com')
            else:
                SheetTable.__init__(self, ctype='xlrd')

        def available(self):
            return win32com_available or xlrd_available

        def requirements(self):
            return "win32com or xlrd"

elif pyodbc_available:

    @DataManagerFactory.register("xls", "Excel XLS file interface")
    class pyodbc_xls(pyodbc_db_base):

        def __init__(self):
            pyodbc_db_base.__init__(self)

        def requirements(self):
            return "pyodbc or pypyodbc"

        def open(self):
            if self.filename is None:
                raise IOError("No filename specified")
            if not os.path.exists(self.filename):
                raise IOError("Cannot find file '%s'" % self.filename)
            return pyodbc_db_base.open(self)


if (win32com_available and _excel_available) or openpyxl_available:

    @DataManagerFactory.register("xlsx", "Excel XLSX file interface")
    class SheetTable_xlsx(SheetTable):

        def __init__(self):
            if win32com_available and _excel_available:
                SheetTable.__init__(self, ctype='win32com')
            else:
                SheetTable.__init__(self, ctype='openpyxl')

        def available(self):
            return win32com_available or openpyxl_available

        def requirements(self):
            return "win32com or openpyxl"

elif pyodbc_available:
    #
    # This class is OK, but the pyodbc interface doesn't work right now.
    #

    @DataManagerFactory.register("xlsx", "Excel XLSX file interface")
    class SheetTable_xlsx(pyodbc_db_base):

        def __init__(self):
            pyodbc_db_base.__init__(self)

        def requirements(self):
            return "pyodbc or pypyodbc"

        def open(self):
            if self.filename is None:
                raise IOError("No filename specified")
            if not os.path.exists(self.filename):
                raise IOError("Cannot find file '%s'" % self.filename)
            return pyodbc_db_base.open(self)


if pyodbc_available:

    @DataManagerFactory.register("xlsb", "Excel XLSB file interface")
    class SheetTable_xlsb(pyodbc_db_base):

        def __init__(self):
            pyodbc_db_base.__init__(self)

        def requirements(self):
            return "pyodbc or pypyodbc"

        def open(self):
            if self.filename is None:
                raise IOError("No filename specified")
            if not os.path.exists(self.filename):
                raise IOError("Cannot find file '%s'" % self.filename)
            return pyodbc_db_base.open(self)


if (win32com_available and _excel_available) or openpyxl_available:

    @DataManagerFactory.register("xlsm", "Excel XLSM file interface")
    class SheetTable_xlsm(SheetTable):

        def __init__(self):
            if win32com_available and _excel_available:
                SheetTable.__init__(self, ctype='win32com')
            else:
                SheetTable.__init__(self, ctype='openpyxl')

        def available(self):
            return win32com_available or openpyxl_available

        def requirements(self):
            return "win32com or openpyxl"

elif pyodbc_available:

    @DataManagerFactory.register("xlsm", "Excel XLSM file interface")
    class SheetTable_xlsm(pyodbc_db_base):

        def __init__(self):
            pyodbc_db_base.__init__(self)

        def requirements(self):
            return "pyodbc or pypyodbc"

        def open(self):
            if self.filename is None:
                raise IOError("No filename specified")
            if not os.path.exists(self.filename):
                raise IOError("Cannot find file '%s'" % self.filename)
            return pyodbc_db_base.open(self)

