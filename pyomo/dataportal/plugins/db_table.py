#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

try:
    unicode
except:
    basestring = str
    long = int

import os.path
import re
import sys
import shutil
from decimal import Decimal
from six import iteritems

try:
    import pyodbc
    pyodbc_available=True
except ImportError:
    pyodbc_available=False

try:
    import pypyodbc
    pypyodbc_available=True
except Exception:
    pypyodbc_available=False

try:
    import sqlite3
    sqlite3_available=True
except ImportError:
    sqlite3_available=False

try:
    import pymysql
    pymysql_available=True
except ImportError:
    pymysql_available=False

from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory


# format=
# using=
# query=
# user=
# password=
# table=

class db_Table(TableData):

    def __init__(self):
        TableData.__init__(self)
        self.using = None

    def open(self):
        if self.filename is None:
            raise IOError("No data source name was specified")
        if self.filename[0] == '"':
            self.filename = self.filename[1:-1]
        #
        # Initialize self.db
        #
        if not self.using is None:
            self.options.using = self.using
        self.db = None
        if self._data is not None:
            self.db = self._data
        else:
            try:
                self.db = self.connect(self.filename, self.options)
            except Exception:
                raise

    def read(self):
        #
        # Get the table from the database
        #
        if self.db is None:
            return
        cursor = self.db.cursor()
        tmp = []
        if self.options.query is None:
            if self.options.table is None:
                raise IOError("Must specify 'query' or 'table' option!")
            self.options.query = 'SELECT * FROM %s' % self.options.table
        elif self.options.query[0] in ("'", '"'):
            self.options.query = self.options.query[1:-1]

        try:
            cursor.execute(self.options.query)
            rows = cursor.fetchall()
            for col in cursor.description:
                tmp.append(col[0])
            tmp = [tmp]
        except sqlite3.OperationalError:
            import logging
            logging.getLogger('pyomo.core').error(
                """Fatal error reading from an external ODBC data source.

This error was generated outside Pyomo by the Python connector to the
external data source:

    %s

for the query:

    %s

It is possible that you have an error in your external data file,
the ODBC connector for this data source is not correctly installed,
or that there is a bug in the ODBC connector.
""" % (self.filename, self.options.query) )
            raise
        for row in rows:
            #print("DATA %s" % str(list(row))) # XXX
            ttmp=[]
            for data in list(row):
                if isinstance(data,Decimal):
                    ttmp.append(float(data))
                elif data is None:
                    ttmp.append('.')
                elif isinstance(data, str) or isinstance(data, basestring):
                    nulidx = data.find('\x00')
                    if nulidx > -1:
                        data = data[:nulidx]
                    ttmp.append(data)
                else:
                    ttmp.append(data)
            tmp.append(ttmp)
        #print('FINAL %s' % str(tmp)) # XXX
        #
        # Process data from the table
        #
        if type(tmp) in (int,long,float):
            if not self.options.param is None:
                self._info = ["param", self.options.param.local_name, ":=", tmp]
            elif len(self.options.symbol_map) == 1:
                self._info = ["param", self.options.symbol_map[self.options.symbol_map.keys()[0]], ":=", tmp]
            else:
                raise IOError("Data looks like a scalar parameter, but multiple parameter names have been specified: %s" % str(self.options.symbol_map))
        elif len(tmp) == 0:
            raise IOError("Empty range '%s'" % self.options.range)
        else:
            #print("_info %s" % str(self._info))
            #print("SETTING DATA %s %s" % (str(tmp[0]), str(tmp[1:]))) # XXX
            #print("OPTIONS %s" % str(self.options))
            self._set_data(tmp[0], tmp[1:])

    def close(self):
        if self._data is None and not self.db is None:
            del self.db

    def connect(self, connection, options, kwds={}):
        try:
            mod = __import__(options.using)
            args = [connection]
            if not options.user is None:
                args.append(options.user)
            if not options.password is None:
                args.append(options.password)
            return mod.connect(*args, **kwds)
        except ImportError:
            return None

#
# NOTE: The pyodbc interface currently doesn't work.  Notably, nothing
# sets the "table" or "query" options, which causes db_table.read() to
# fail.  This interface has been disabled by overriding pyodbc_available
# in sheet.py
#

@DataManagerFactory.register('pyodbc', "%s database interface" % 'pyodbc')
class pyodbc_db_Table(db_Table):

    _drivers = {
        'mdb': ["Microsoft Access Driver (*.mdb)"],
        'xls': ["Microsoft Excel Driver (*.xls, *.xlsx, *.xlsm, *.xlsb)","Microsoft Excel Driver (*.xls)"],
        'xlsx': ["Microsoft Excel Driver (*.xls, *.xlsx, *.xlsm, *.xlsb)"],
        'xlsm': ["Microsoft Excel Driver (*.xls, *.xlsx, *.xlsm, *.xlsb)"],
        'xlsb': ["Microsoft Excel Driver (*.xls, *.xlsx, *.xlsm, *.xlsb)"],
        'mysql': ["MySQL"]
    }
    _drivers['access'] = _drivers['mdb']
    _drivers['excel'] = _drivers['xls']

    def __init__(self):
        db_Table.__init__(self)
        self.using = 'pyodbc'

    def available(self):
        return pyodbc_available

    def requirements(self):
        return 'pyodbc'

    def connect(self, connection, options):
        if not options.driver is None:
            ctype = options.driver
        elif '.' in connection:
            ctype = connection.split('.')[-1]
        elif 'mysql' in connection.lower():
            ctype = 'mysql'
        else:
            ctype = ''
        extras = {}
        if ctype in ['xls', 'xlsx', 'xlsm', 'xlsb', 'excel'] or '.xls' in connection or '.xlsx' in connection or '.xlsm' in connection or '.xlsb' in connection:
            extras['autocommit'] = True

        connection = self.create_connection_string(ctype, connection, options)
        try:
            conn = db_Table.connect(self, connection, options, extras)
        except TypeError:
            raise
        except Exception:
            e = sys.exc_info()[1]
            code = e.args[0]
            if code == 'IM002' or code == '08001':
                if 'HOME' in os.environ:
                    # Need a DSN! Try to add it to $HOME/.odbc.ini ...
                    #
                    # Note: this only works on *nix platforms.  It appears
                    # that ODBC.INI is stored in the registry on windows
                    #
                    # [JDS, 8 Oct 18]: I am not convinced that writing a
                    # .odbc.ini file is necessary; see the "else" branch
                    # below.
                    #
                    odbcIniPath = os.path.join(os.environ['HOME'], '.odbc.ini')
                    if os.path.exists(odbcIniPath):
                        shutil.copy(odbcIniPath, odbcIniPath + '.orig')
                        config = ODBCConfig(filename=odbcIniPath)
                    else:
                        config = ODBCConfig()

                    dsninfo = self.create_dsn_dict(connection, config)
                    dsnid = re.sub('[^A-Za-z0-9]', '', dsninfo['Database']) # Strip filenames of funny characters
                    dsn = 'PYOMO{0}'.format(dsnid)

                    config.add_source(dsn, dsninfo['Driver'])
                    config.add_source_spec(dsn, dsninfo)
                    config.write(odbcIniPath)

                    connstr = "DRIVER={{{0}}};DSN={1}".format(dsninfo['Driver'], dsn)
                else:
                    # Attempt to re-generate the connection string with a Driver
                    config = ODBCConfig()
                    dsninfo = self.create_dsn_dict(connection, config)
                    connstr = []
                    for k,v in iteritems(dsninfo):
                        if ' ' in v and (v[0] != "{" or v[-1] != "}"):
                            connstr.append("%s={%s}" % (k.upper(),v))
                        else:
                            connstr.append("%s=%s" % (k.upper(),v))
                    connstr = ";".join(connstr)

                conn = db_Table.connect(self, connstr, options, extras) # Will raise its own exception on failure

            # Propagate the exception
            else:
                raise

        return conn

    def create_dsn_dict(self, argstr, existing_config):
        result = {}

        parts = argstr.split(';')
        argdict = {}
        for part in parts:
            if len(part) > 0 and '=' in part:
                key, val = part.split('=',1)
                argdict[key.lower().strip()] = val.strip()

        if 'driver' in argdict:
            result['Driver'] = "{0}".format(argdict['driver']).strip("{}")

        if 'dsn' in argdict:
            if argdict['dsn'] in existing_config.source_specs:
                return existing_config.source_specs[argdict['dsn']]
            else:
                import logging
                logger = logging.getLogger("pyomo.core")
                logger.warning("DSN with name {0} not found. Attempting to continue with options...".format(argdict['dsn']))

        if 'dbq' in argdict:
            # Using a file for db access.
            if 'Driver' not in result:
                result['Driver'] = self._drivers[argdict['dbq'].split('.')[-1].lower()]
            result['Database'] = argdict['dbq']
            result['Server'] = 'localhost'
            result['User'] = ''
            result['Password'] = ''
            result['Port'] = '5432'
            result['Description'] = argdict['dbq']
            for k in argdict.keys():
                if k.capitalize() not in result:
                    result[k.capitalize()] = argdict[k]
        else:
            if 'Driver' not in result:
                raise Exception("No driver specified, and no DBQ to infer from")
            elif result['Driver'].lower() == "mysql":
                result['Driver'] = "MySQL"
                result['Server'] = argdict.get('server', 'localhost')
                result['Database'] = argdict.get('database', '')
                result['Port'] = argdict.get('port', '3306')
                result['Socket'] = argdict.get('socket', '')
                result['Option'] = argdict.get('option', '')
                result['Stmt'] = argdict.get('stmt', '')
                result['User'] = argdict.get('user', '')
                result['Password'] = argdict.get('password', '')
                result['Description'] = argdict.get('description', '')
            else:
                raise Exception("Unknown driver type '{0}' for database connection".format(result['Driver']))

        return result

    def create_connection_string(self, ctype, connection, options):
        driver = self._get_driver(ctype)
        if driver:
            if ' ' in driver and (driver[0] != "{" or driver[-1] != "}"):
                return "DRIVER={%s};Dbq=%s;" % (driver, connection)
            else:
                return "DRIVER=%s;Dbq=%s;" % (driver, connection)
        return connection

    def _get_driver(self, ctype):
        # Given a list of possible drivers for this ctype, look to find
        # a match in the pyodbc.drivvers() list.  If a match is found,
        # return it.  Otherwise (arbitrarily) return the first one.  If
        # the ctype is not known, return None.
        drivers = self._drivers.get(ctype,[])
        for driver in drivers:
            if driver in pyodbc.drivers():
                return driver
        if drivers:
            return drivers[0]
        else:
            return None


class ODBCError(Exception):
    def __init__(self, value):
        self.parameter = value
    def __repr__(self):
        return repr(self.parameter)


class ODBCConfig():
    """
    Encapsulates an ODBC configuration file, usually odbc.ini or
    .odbc.ini, as specified by IBM. ODBC config data can be loaded
    either from a file or a string containing the relevant formatted
    data. Calling load() after initialization will update existing
    information in the config object with the new information.
    """

    def __init__(self, filename=None, data=None):
        """
        Create a new ODBC config instance, loading data from
        the given file and/or data string. Once initialized, the
        new config will contain the data represented in both
        arguments, if any. Data specified as a string argument
        will override that in the file.
        """

        # ugh hardcoded strings. See following URL for info:
        # http://publib.boulder.ibm.com/infocenter/idshelp/v10/index.jsp?topic=/com.ibm.odbc.doc/odbc58.htm
        self.ODBC_DS_KEY = 'ODBC Data Sources'
        self.ODBC_INFO_KEY = 'ODBC'

        self.file = filename

        self.sources = {}
        self.source_specs = {}
        self.odbc_info = {}

        self.load(self.file, data)

    def load(self, filename=None, data=None):
        """
        Load data from the given file and/or data string. If
        both are given, data contained in the string will override
        that in the file. If this config object already contains
        data, the new information loaded will update the old,
        replacing where keys are the same.
        """

        sections = {}
        if filename is not None:
            with open(filename, 'r') as fileHandle:
                fileData = fileHandle.read()
                sections.update(self._get_sections(fileData))
        if data is not None:
            sections.update(self._get_sections(data))

        if self.ODBC_DS_KEY in sections:
            self.sources.update(sections[self.ODBC_DS_KEY])
            del sections[self.ODBC_DS_KEY]

        if self.ODBC_INFO_KEY in sections:
            self.odbc_info.update(sections[self.ODBC_INFO_KEY])
            del sections[self.ODBC_INFO_KEY]

        self.source_specs.update(sections)

    def __str__(self):
        return "<ODBC config: {0} sources, {1} source specs>".format(len(self.sources), len(self.source_specs))

    def __eq__(self, other):
        if isinstance(other, ODBCConfig):
            return self.sources == other.sources and self.source_specs == other.source_specs and self.odbc_info == other.odbc_info
        return False

    def odbc_repr(self):
        """
        Get the full, odbc.ini-style representation of this
        ODBC configuration.
        """

        str = "[{0}]\n".format(self.ODBC_DS_KEY)

        for name in self.sources:
            str += "{0} = {1}\n".format(name, self.sources[name])

        for name in self.source_specs:
            str += "\n[{0}]\n".format(name)
            for key in self.source_specs[name]:
                str += "{0} = {1}\n".format(key, self.source_specs[name][key])

        if len(self.odbc_info) > 0:
            str += "\n[{0}]\n".format(self.ODBC_INFO_KEY)
            for key in self.odbc_info:
                str += "{0} = {1}\n".format(key, self.odbc_info[key])

        return str

    def write(self, filename):
        """
        Write the current ODBC configuration to the given file.
        Depends on the odbc_repr() function for a string
        representation of the stored ODBC config information.
        """
        with open(filename, 'w') as f:
            f.write(self.odbc_repr())

    def add_source(self, name, driver):
        """
        Add an ODBC data source to the configuration. A data
        source consists of a unique source name and a driver
        string, which specifies how the source will be loaded.
        If a source name is not unique, it will replace the
        existing source of the same name. A source is required
        in order to have a source specification.
        """

        if name is None or driver is None or len(name) == 0 or len(driver) == 0:
            raise ODBCError("A source must specify both a name and a driver string")
        if name == self.ODBC_DS_KEY or name == self.ODBC_INFO_KEY:
            raise ODBCError("A source cannot use the reserved name '{0}'".format(name))
        self.sources[str(name)] = str(driver)

    def del_source(self, name):
        """
        Remove an ODBC data source from the configuration. If
        any source specifications are based on this source, they
        will be removed as well.
        """

        if name in self.sources:
            if name in self.source_specs:
                del self.source_specs[name]
            del self.sources[name]

    def add_source_spec(self, name, spec):
        """
        Add an ODBC data source specification to the configuration.
        A source specification consists of a unique name and
        a key-value mapping (i.e. dictionary) of options. In order
        to add a source specification, a data source with a matching
        name must exist in the configuration.
        """

        if name is None or spec is None or len(name) == 0:
            raise ODBCError("A source spec must specify both a name and a spec dictionary")
        if name not in self.sources:
            raise ODBCError("A source spec must have a corresponding source; call .add_source() first")

        self.source_specs[name] = dict(spec)

    def del_source_spec(self, name):
        """
        Remove an ODBC data source specification from the
        configuration.
        """

        if name in self.source_specs:
            del self.source_specs[name]

    def set_odbc_info(self, key, value):
        """
        Set an option for the ODBC handling specified in the
        configuration. An option consists of a key-value pair.
        Specifying an existing key will update the current value.
        """

        if key is None or value is None or len(key) == 0 or len(value) == 0:
            raise ODBCError("An ODBC info pair must specify both a key and a value")

        self.odbc_info[str(key)] = str(value)

    def _get_sections(self, data):
        """
        Parse a string for ODBC sections. The parsing algorithm proceeds
        roughly as follows:

        1. Split the string on newline ('\\n') characters.
        2. Remove lines consisting purely of whitespace.
        3. Iterate over lines, storing all key-value pair lines in a dictionary.
        4. When reaching a new section header (denoted by '[str]'), store the old
           key-value pairs under the old section name. Continue from step 3.
        5. On reaching end of data, store the last section and return a mapping
           from section names to dictionaries of key-value pairs in those sections.
        """

        sections = {}
        sectionKey = None
        sectionContents = {}

        emptyLine = re.compile('^[ \t\r]*$')

        lines = data.split('\n')
        for line in lines:
            if emptyLine.match(line):
                # Whitespace only
                pass
            elif len(line) < 2:
                # Not enough room for even 'k='; can't contain info
                raise ODBCError("Malformed line in ODBC config (no meaningful data): " + line)
            elif line[0] == '[' and line[-1] == ']':
                # Starts a new section; '=' has no special meaning here
                sections[sectionKey] = sectionContents
                sectionKey = line[1:-1]
                sectionContents = {}
            else:
                # Not whitespace or section header; must be key=value. No duplicate '=' permitted.
                if '=' not in line:
                    raise ODBCError("Malformed line in ODBC config (no key-value mapping): " + line)

                key, value = line.split("=",1)
                if '=' in value:
                    raise ODBCError("Malformed line in ODBC config (too many '='): " + line)

                sectionContents[key.strip()] = value.strip()
        sections[sectionKey] = sectionContents

        if None in sections:
            del sections[None]

        return sections


@DataManagerFactory.register('pypyodbc', "%s database interface" % 'pypyodbc')
class pypyodbc_db_Table(pyodbc_db_Table):

    def __init__(self):
        pyodbc_db_Table.__init__(self)
        self.using = 'pypyodbc'

    def available(self):
        return pypyodbc_available

    def requirements(self):
        return 'pypyodbc'

    def connect(self, connection, options):
        assert(options['using'] == 'pypyodbc')

        return pyodbc_db_Table.connect(self, connection, options)


@DataManagerFactory.register('sqlite3', "sqlite3 database interface")
class sqlite3_db_Table(db_Table):

    def __init__(self):
        db_Table.__init__(self)
        self.using = 'sqlite3'

    def available(self):
        return sqlite3_available

    def requirements(self):
        return 'sqlite3'

    def connect(self, connection, options):
        assert(options['using'] == 'sqlite3')

        filename = connection
        if not os.path.exists(filename):
            raise Exception("No such file: " + filename)

        con = sqlite3.connect(filename)
        if options.text_factory:
            con.text_factory = options.text_factory
        return con


@DataManagerFactory.register('pymysql', "pymysql database interface")
class pymysql_db_Table(db_Table):

    def __init__(self):
        db_Table.__init__(self)
        self.using = 'pymysql'

    def available(self):
        return pymysql_available

    def requirements(self):
        return 'pymysql'
