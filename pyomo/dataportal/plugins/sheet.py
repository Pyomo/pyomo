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
from pyutilib.excel.spreadsheet import ExcelSpreadsheet, Interfaces

from pyomo.dataportal import TableData
# from pyomo.dataportal.plugins.db_table import (
#     pyodbc_available, pyodbc_db_Table, pypyodbc_available, pypyodbc_db_Table
# )
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.common.errors import ApplicationError


def _attempt_open_excel():
    if _attempt_open_excel.result is None:
        from pyutilib.excel.spreadsheet_win32com import (
            ExcelSpreadsheet_win32com
        )
        try:
            tmp = ExcelSpreadsheet_win32com()
            tmp._excel_dispatch()
            tmp._excel_quit()
            _attempt_open_excel.result = True
        except:
            _attempt_open_excel.result = False
    return _attempt_open_excel.result

_attempt_open_excel.result = None


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
            except ApplicationError:
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




@DataManagerFactory.register("xls", "Excel XLS file interface")
class SheetTable_xls(SheetTable):

    def __init__(self):
        if Interfaces()['win32com'].available and _attempt_open_excel():
            SheetTable.__init__(self, ctype='win32com')
        elif Interfaces()['xlrd'].available:
            SheetTable.__init__(self, ctype='xlrd')
        else:
            raise RuntimeError("No excel interface is available; install %s"
                               % self.requirements())

    def available(self):
        _inter = Interfaces()
        return (_inter['win32com'].available and _attempt_open_excel()) \
            or _inter['xlrd'].available

    def requirements(self):
        return "win32com or xlrd"


# @DataManagerFactory.register("xls", "Excel XLS file interface")
# class pyodbc_xls(pyodbc_db_base):

#     def __init__(self):
#         pyodbc_db_base.__init__(self)

#     def requirements(self):
#         return "pyodbc or pypyodbc"

#     def open(self):
#         if self.filename is None:
#             raise IOError("No filename specified")
#         if not os.path.exists(self.filename):
#             raise IOError("Cannot find file '%s'" % self.filename)
#         return pyodbc_db_base.open(self)


@DataManagerFactory.register("xlsx", "Excel XLSX file interface")
class SheetTable_xlsx(SheetTable):

    def __init__(self):
        if Interfaces()['win32com'].available and _attempt_open_excel():
            SheetTable.__init__(self, ctype='win32com')
        elif Interfaces()['openpyxl'].available:
            SheetTable.__init__(self, ctype='openpyxl')
        else:
            raise RuntimeError("No excel interface is available; install %s"
                               % self.requirements())

    def available(self):
        _inter = Interfaces()
        return (_inter['win32com'].available and _attempt_open_excel()) \
            or _inter['openpyxl'].available

    def requirements(self):
        return "win32com or openpyxl"

#
# This class is OK, but the pyodbc interface doesn't work right now.
#

# @DataManagerFactory.register("xlsx", "Excel XLSX file interface")
# class SheetTable_xlsx(pyodbc_db_base):
#
#     def __init__(self):
#         pyodbc_db_base.__init__(self)
#
#     def requirements(self):
#         return "pyodbc or pypyodbc"
#
#     def open(self):
#         if self.filename is None:
#             raise IOError("No filename specified")
#         if not os.path.exists(self.filename):
#             raise IOError("Cannot find file '%s'" % self.filename)
#         return pyodbc_db_base.open(self)


# @DataManagerFactory.register("xlsb", "Excel XLSB file interface")
# class SheetTable_xlsb(pyodbc_db_base):
#
#     def __init__(self):
#         pyodbc_db_base.__init__(self)
#
#     def requirements(self):
#         return "pyodbc or pypyodbc"
#
#     def open(self):
#         if self.filename is None:
#             raise IOError("No filename specified")
#         if not os.path.exists(self.filename):
#             raise IOError("Cannot find file '%s'" % self.filename)
#         return pyodbc_db_base.open(self)


@DataManagerFactory.register("xlsm", "Excel XLSM file interface")
class SheetTable_xlsm(SheetTable):

    def __init__(self):
        if Interfaces()['win32com'].available and _attempt_open_excel():
            SheetTable.__init__(self, ctype='win32com')
        elif Interfaces()['openpyxl'].available:
            SheetTable.__init__(self, ctype='openpyxl')
        else:
            raise RuntimeError("No excel interface is available; install %s"
                               % self.requirements())

    def available(self):
        _inter = Interfaces()
        return (_inter['win32com'].available and _attempt_open_excel()) \
            or _inter['openpyxl'].available

    def requirements(self):
        return "win32com or openpyxl"

# @DataManagerFactory.register("xlsm", "Excel XLSM file interface")
# class SheetTable_xlsm(pyodbc_db_base):
#
#     def __init__(self):
#         pyodbc_db_base.__init__(self)
#
#     def requirements(self):
#         return "pyodbc or pypyodbc"
#
#     def open(self):
#         if self.filename is None:
#             raise IOError("No filename specified")
#         if not os.path.exists(self.filename):
#             raise IOError("Cannot find file '%s'" % self.filename)
#         return pyodbc_db_base.open(self)

