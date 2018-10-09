#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['TableData']

from six.moves import xrange

from pyutilib.misc import Options
from pyomo.dataportal.process_data import _process_data


class TableData(object):
    """
    A class used to read/write data from/to a table in an external
    data source.
    """

    def __init__(self):
        """
        Constructor
        """
        self._info=None
        self._data=None
        self.options = Options()
        self.options.ncolumns = 1

    def available(self):
        """
        Returns:
            Return :const:`True` if the data manager is available.
        """
        return True

    def initialize(self, **kwds):
        """
        Initialize the data manager with keyword arguments.

        The `filename` argument is recognized here, and other arguments
        are passed to the :func:`add_options` method.
        """
        self.filename = kwds.pop('filename')
        self.add_options(**kwds)

    def add_options(self, **kwds):
        """
        Add the keyword options to the :class:`Options` object in this
        object.
        """
        self.options.update(kwds)

    def open(self):                        #pragma:nocover
        """
        Open the data manager.
        """
        pass

    def read(self):                         #pragma:nocover
        """
        Read data from the data manager.
        """
        return False

    def write(self, data):                  #pragma:nocover
        """
        Write data to the data manager.
        """
        return False

    def close(self):                        #pragma:nocover
        """
        Close the data manager.
        """
        pass

    def process(self, model, data, default):
        """
        Process the data that was extracted from this data manager and
        return it.
        """
        if model is None:
            model = self.options.model
        if not self.options.namespace in data:
            data[self.options.namespace] = {}
        return _process_data(
          self._info,
          model,
          data[self.options.namespace],
          default,
          self.filename,
          index=self.options.index,
          set=self.options.set,
          param=self.options.param,
          ncolumns = self.options.ncolumns)

    def clear(self):
        """
        Clear the data that was extracted from this table
        """
        self._info = None

    def _set_data(self, headers, rows):
        from pyomo.core.base.sets import Set
        from pyomo.core.base.param import Param

        header_index = []
        if self.options.select is None:
            for i in xrange(len(headers)):
                header_index.append(i)
        else:
            for i in self.options.select:
                try:
                    header_index.append(headers.index(str(i)))
                except:
                    print("Model declaration '%s' not found in returned query columns" %str(i))
                    raise
        self.options.ncolumns = len(headers)

        if not self.options.param is None:
            if not type(self.options.param) in (list, tuple):
                self.options.param = (self.options.param,)
            _params = []
            for p in self.options.param:
                if isinstance(p, Param):
                    self.options.model = p.model()
                    _params.append(p.local_name)
                else:
                    _params.append(p)
            self.options.param = tuple(_params)

        if isinstance(self.options.set, Set):
            self.options.model = self.options.set.model()
            self.options.set = self.options.set.local_name

        if isinstance(self.options.index, Set):
            self.options.model = self.options.index.model()
            self.options.index = self.options.index.local_name

        elif type(self.options.index) in [tuple, list]:
            tmp = []
            for val in self.options.index:
                if isinstance(val, Set):
                    tmp.append(val.local_name)
                    self.options.model = val.model()
                else:
                    tmp.append(val)
            self.options.index = tuple(tmp)

        if self.options.format is None:
            if not self.options.set is None:
                self.options.format = 'set'
            elif not self.options.param is None:
                self.options.format = 'table'
            if self.options.format is None:
                raise ValueError("Unspecified format and  data option")
        elif self.options.set is None and self.options.param is None:
            msg = "Must specify the set or parameter option for data"
            raise IOError(msg)

        if self.options.format == 'set':
            if not self.options.index is None:
                msg = "Cannot specify index for data with the 'set' format: %s"
                raise IOError(msg % str(self.options.index))

            self._info = ["set",self.options.set,":="]
            for row in rows:
                if self.options.ncolumns > 1:
                    self._info.append(tuple(row))
                else:
                    self._info.extend(row)

        elif self.options.format == 'set_array':
            if not self.options.index is None:
                msg = "Cannot specify index for data with the 'set_array' "   \
                      'format: %s'
                raise IOError(msg % str(self.options.index))

            self._info = ["set",self.options.set, ":"]
            self._info.extend(headers[1:])
            self._info.append(":=")
            for row in rows:
                self._info.extend(row)

        elif self.options.format == 'transposed_array':
            self._info = ["param",self.options.param[0],"(tr)",":"]
            self._info.extend(headers[1:])
            self._info.append(":=")
            for row in rows:
                self._info.extend(row)

        elif self.options.format == 'array':
            self._info = ["param",self.options.param[0],":"]
            self._info.extend(headers[1:])
            self._info.append(":=")
            for row in rows:
                self._info.extend(row)

        elif self.options.format == 'table':
            if self.options.index is not None:
                self._info = ["param",":",self.options.index,":"]
            else:
                self._info = ["param",":"]
            for param in self.options.param:
                self._info.append(param)
            self._info.append(":=")
            for row in rows:
                for i in header_index:
                    self._info.append(row[i])
            self.options.ncolumns = len(header_index)
        else:
            msg = "Unknown parameter format: '%s'"
            raise ValueError(msg % self.options.format)

    def _get_table(self):
        from pyomo.core.expr import value

        tmp = []
        if not self.options.columns is None:
            tmp.append(self.options.columns)
        if not self.options.set is None:
            # Create column names
            if self.options.columns is None:
                cols = []
                for i in xrange(self.options.set.dimen):
                    cols.append(self.options.set.local_name+str(i))
                tmp.append(cols)
            # Get rows
            if not self.options.sort is None:
                for data in sorted(self.options.set):
                    if self.options.set.dimen > 1:
                        tmp.append(list(data))
                    else:
                        tmp.append([data])
            else:
                for data in self.options.set:
                    if self.options.set.dimen > 1:
                        tmp.append(list(data))
                    else:
                        tmp.append([data])
        elif not self.options.param is None:
            if type(self.options.param) in (list,tuple):
                _param = self.options.param
            else:
                _param = [self.options.param]
            tmp = []
            # Collect data
            for index in _param[0]:
                if index is None:
                    row = []
                elif type(index) in (list,tuple):
                    row = list(index)
                else:
                    row = [index]
                for param in _param:
                    row.append(value(param[index]))
                tmp.append(row)
            # Create column names
            if self.options.columns is None:
                cols = []
                for i in xrange(len(tmp[0])-len(_param)):
                    cols.append('I'+str(i))
                for param in _param:
                    cols.append(param)
                tmp = [cols] + tmp
        return tmp
