#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for odbc.ini file handler
#

import os

import pyutilib.th as unittest

from pyomo.environ import *

try:
    import pyodbc
    pyodbc_available = True

    from pyomo.core.data.db_table import ODBCConfig, ODBCError
except ImportError:
    pyodbc_available = False

@unittest.skipIf(not pyodbc_available, "PyODBC is not installed.")
class TestODBCIni(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)


        self.ACCESS_CONFIGSTR = "Microsoft Access Driver (*.mdb)"
        self.EXCEL_CONFIGSTR = "Microsoft Excel Driver (*.xls)"

        self.simple_data = """[ODBC Data Sources]
testdb = Microsoft Access Driver (*.mdb)

[testdb]
Database = testdb.mdb
"""

        self.complex_data = """Discard me = true

[ODBC Data Sources]
test1 = Microsoft Access Driver (*.mdb)
test2 = Microsoft Excel Driver (*.xls)

[test1]
Database = test1.db
LogonID = Admin
pwd = secret_pass

[test2]
Database = test2.xls

[ODBC]
UNICODE = UTF-8
"""
    
    def test_create(self):
        config = ODBCConfig()
        self.assertIsNone(config.file)

    def test_init_empty_data(self):
        config = ODBCConfig()
        self.assertEquals({}, config.sources)
        self.assertEquals({}, config.source_specs)
        self.assertEquals({}, config.odbc_info)

    def test_init_simple_data(self):
        config = ODBCConfig(data=self.simple_data)
        self.assertEquals({'testdb' : self.ACCESS_CONFIGSTR}, config.sources)
        self.assertEquals({'testdb' : {'Database' : "testdb.mdb"}}, config.source_specs)
        self.assertEquals({}, config.odbc_info)
    
    def test_init_complex_data(self):
        config = ODBCConfig(data=self.complex_data)
        self.assertEquals({'test1' : self.ACCESS_CONFIGSTR, 'test2' : self.EXCEL_CONFIGSTR}, config.sources)
        self.assertEquals({'test1' : {'Database' : "test1.db", 'LogonID' : "Admin", 'pwd' : "secret_pass"}, 'test2' : {'Database' : "test2.xls"}}, config.source_specs)
        self.assertEquals({'UNICODE' : "UTF-8"}, config.odbc_info)

    def test_add_source(self):
        config = ODBCConfig()
        config.add_source("testdb", self.ACCESS_CONFIGSTR)
        self.assertEquals({'testdb' : self.ACCESS_CONFIGSTR}, config.sources)
        self.assertEquals({}, config.source_specs)
        self.assertEquals({}, config.odbc_info)

    def test_del_source(self):
        config = ODBCConfig(data=self.simple_data)
        config.del_source('testdb')
        self.assertEquals({}, config.sources)

    def test_add_source_reserved(self):
        config = ODBCConfig()
        with self.assertRaises(ODBCError):
            config.add_source("ODBC Data Sources", self.ACCESS_CONFIGSTR)
        with self.assertRaises(ODBCError):
            config.add_source("ODBC", self.ACCESS_CONFIGSTR)

    def test_add_source_spec(self):
        config = ODBCConfig()
        config.add_source("testdb", self.ACCESS_CONFIGSTR)
        config.add_source_spec("testdb", {'Database' : "testdb.mdb"})
        self.assertEquals({'testdb' : {'Database' : "testdb.mdb"}}, config.source_specs)
    
    def test_add_spec_bad(self):
        config = ODBCConfig()
        with self.assertRaises(ODBCError):
            config.add_source_spec("testdb", {'Database' : "testdb.mdb"})

    def test_del_source_dependent(self):
        config = ODBCConfig()
        config.add_source("testdb", self.ACCESS_CONFIGSTR)
        config.add_source_spec("testdb", {'Database' : "testdb.mdb"})
        config.del_source("testdb")
        self.assertEquals({}, config.sources)
        self.assertEquals({}, config.source_specs)

    def test_set_odbc_info(self):
        config = ODBCConfig()
        config.set_odbc_info("UNICODE", "UTF-8")
        self.assertEquals({'UNICODE' : "UTF-8"}, config.odbc_info)

    def test_odbc_repr(self):
        config = ODBCConfig(data=self.simple_data)
        self.assertMultiLineEqual(config.odbc_repr(), self.simple_data)

    def test_baselines(self):
        filenames = ['simple_odbc', 'diet']
        basePath = os.path.split(os.path.abspath(__file__))[0]
        for fn in filenames:
            iniPath = os.path.join(basePath, 'baselines', '{0}.ini'.format(fn))
            outPath = os.path.join(basePath, 'baselines', '{0}.out'.format(fn))

            config = ODBCConfig(filename=iniPath)
            config.write(outPath)

            written = ODBCConfig(filename = outPath)
            self.assertEquals(config, written)

            try:
                os.remove(outPath)
            except:
                pass

    def test_eq(self):
        self.assertEquals(ODBCConfig(), ODBCConfig())

        configA = ODBCConfig(data=self.simple_data)
        configB = ODBCConfig()
        configB.sources = {'testdb' : self.ACCESS_CONFIGSTR}
        configB.source_specs = {'testdb' : {'Database' : 'testdb.mdb'}}
        self.assertEquals(configA, configB)

if __name__ == "__main__":
    unittest.main()
