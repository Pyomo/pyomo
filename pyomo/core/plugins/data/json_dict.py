#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

import os.path
import json
try:
    import yaml
    yaml_available = True
except ImportError:
    yaml_available = False

from pyutilib.misc import Options
from pyomo.misc.plugin import alias, Plugin, implements

from pyomo.core.base.plugin import IDataManager, DataManagerFactory


def detuplize(d):
    if type(d) in (list,set):
        ans = []
        for item in d:
            if type(item) in (list,tuple,set):
                ans.append(list(item))
            else:
                ans.append(item)
        return ans
    ans = {}
    for k,v in d.items():
        if type(k) is tuple:
            _d = ans
            j = 0
            for i in k:
                j += 1
                if j == len(k):
                    _d = _d.setdefault(i, v)
                else:
                    _d = _d.setdefault(i, {})
        else:
            ans[k] = v
    return ans

def tuplize(d, ret=None, ndx=None):
    if type(d) is list:
        ans = []
        for val in d:
            if type(val) is list:
                item = []
                for v in val:
                    try:
                        item.append(int(v))
                    except:
                        try:
                            item.append(float(v))
                        except:
                            item.append(v)
                item = tuple(item)
            else:
                try:
                    item = int(val)
                except:
                    try:
                        item = float(val)
                    except:
                        item = val
            ans.append(item)
        return ans
    elif not type(d) is dict:
        return d
    if ret is None:
        ret = {}
        ndx = []
    for k, v in d.items():
        try:
            k = int(k)
        except:
            try:
                k = float(k)
            except:
                pass
        ndx.append(k)
        if type(v) is dict:
            tuplize(v, ret, ndx)
        elif len(ndx) == 1:
            ret[ndx[0]] = v
        else:
            ret[tuple(ndx)] = v
        ndx.pop()
    return ret


class JSONDictionary(Plugin):

    alias("json", "Import data from a JSON file.")

    implements(IDataManager, service=False)

    def __init__(self):
        self._info = {}
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

    def close(self):
        pass

    def read(self):
        """
        This function loads data from a JSON file and tuplizes the nested
        dictionaries and lists of lists.
        """
        if not os.path.exists(self.filename):
            raise IOError("Cannot find file '%s'" % self.filename)
        INPUT = open(self.filename, 'r')
        jdata = json.load(INPUT)
        self._info = {}
        for k,v in jdata.items():
            self._info[k] = tuplize(v)
        INPUT.close()
        return True

    def write(self, data):
        """
        This function creates a JSON file for the specified data.
        """
        OUTPUT = open(self.filename, 'w')
        jdata = {}
        if self.options.data is None:
            for k,v in data.items():
                jdata[k] = detuplize(v)
        elif type(self.options.data) in (list, tuple):
            for k in self.options.data:
                jdata[k] = detuplize(data[k])
        else:
            k = self.options.data
            jdata[k] = detuplize(data[k])
        json.dump(jdata, OUTPUT)
        OUTPUT.close()
        return True

    def process(self, model, data, default):
        """
        Set the data for the selected components
        """
        if not self.options.namespace in data:
            data[self.options.namespace] = {}
        #
        try:
            if self.options.data is None:
                for key in self._info:
                    self._set_data(data, self.options.namespace, key, self._info[key])
            elif type(self.options.data) in (list, tuple):
                for key in self.options.data:
                    self._set_data(data, self.options.namespace, key, self._info[key])
            else:
                key = self.options.data
                self._set_data(data, self.options.namespace, key, self._info[key])
        except KeyError:
            raise IOError("Data value for '%s' is not available in JSON file '%s'" % (key, self.filename))

    def _set_data(self, data, namespace, name, value):
        if type(value) is dict:
            data[namespace][name] = value
        else:
            data[namespace][name] = {None: value}

    def clear(self):
        self._info = {}



class YamlDictionary(Plugin):

    alias("yaml", "Import data from a YAML file.")

    implements(IDataManager, service=False)

    def __init__(self):
        self._info = {}
        self.options = Options()

    def available(self):
        return yaml_available

    def requirements(self):
        return "pyyaml"

    def initialize(self, **kwds):
        self.filename = kwds.pop('filename')
        self.add_options(**kwds)

    def add_options(self, **kwds):
        self.options.update(kwds)

    def open(self):
        if self.filename is None:
            raise IOError("No filename specified")

    def close(self):
        pass

    def read(self):
        """
        This function loads data from a JSON file and tuplizes the nested
        dictionaries and lists of lists.
        """
        if not os.path.exists(self.filename):
            raise IOError("Cannot find file '%s'" % self.filename)
        INPUT = open(self.filename, 'r')
        jdata = yaml.load(INPUT)
        self._info = {}
        for k,v in jdata.items():
            self._info[k] = tuplize(v)
        INPUT.close()
        return True

    def write(self, data):
        """
        This function creates a JSON file for the specified data.
        """
        OUTPUT = open(self.filename, 'w')
        jdata = {}
        if self.options.data is None:
            for k,v in data.items():
                jdata[k] = detuplize(v)
        elif type(self.options.data) in (list, tuple):
            for k in self.options.data:
                jdata[k] = detuplize(data[k])
        else:
            k = self.options.data
            jdata[k] = detuplize(data[k])
        yaml.dump(jdata, OUTPUT)
        OUTPUT.close()
        return True

    def process(self, model, data, default):
        """
        Set the data for the selected components
        """
        if not self.options.namespace in data:
            data[self.options.namespace] = {}
        #
        try:
            if self.options.data is None:
                for key in self._info:
                    self._set_data(data, self.options.namespace, key, self._info[key])
            elif type(self.options.data) in (list, tuple):
                for key in self.options.data:
                    self._set_data(data, self.options.namespace, key, self._info[key])
            else:
                key = self.options.data
                self._set_data(data, self.options.namespace, key, self._info[key])
        except KeyError:
            raise IOError("Data value for '%s' is not available in JSON file '%s'" % (key, self.filename))

    def _set_data(self, data, namespace, name, value):
        if type(value) is dict:
            data[namespace][name] = value
        else:
            data[namespace][name] = {None: value}

    def clear(self):
        self._info = {}


