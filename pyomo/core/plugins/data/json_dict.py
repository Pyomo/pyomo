#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os.path
import json
try:
    import yaml
    yaml_available = True
except ImportError:
    yaml_available = False

from pyutilib.misc import Options

from pyomo.util.plugin import alias, Plugin, implements
from pyomo.core.base.plugin import IDataManager


def detuplize(d, sort=False):
    #print("detuplize %s" % str(d))
    if type(d) in (list,set):
        ans = []
        for item in d:
            if type(item) in (list,tuple,set):
                ans.append(list(item))
            else:
                ans.append(item)
        if sort:
            return sorted(ans)
        return ans
    elif None in d:
        return d[None]
    else:
        #
        # De-tuplize keys via list of key/value pairs
        #
        ans = []
        for k,v in d.items():
            if type(k) is tuple:
                ans.append( {'index':list(k), 'value':v} )
            else:
                ans.append( {'index':k, 'value':v} )
        if sort:
            return sorted(ans, key=lambda x:x['value'])
        return ans

def tuplize(d):
    #print("tuplize %s" % str(d))
    if type(d) is list and len(d) > 0 and not type(d[0]) is dict:
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
    #
    elif type(d) is list:
        ret = {}
        for val in d:
            if type(val['index']) is list:
                index = tuple(val['index'])
            else:
                index = val['index']
            ret[index] = val['value']
        return ret
    #
    elif type(d) is dict:
        return d
    else:
        return {None:d}


class JSONDictionary(Plugin):

    alias("json", "JSON file interface")

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
        INPUT.close()
        if jdata is None or len(jdata) == 0:
            raise IOError("Empty JSON data file")
        self._info = {}
        for k,v in jdata.items():
            self._info[k] = tuplize(v)

    def write(self, data):
        """
        This function creates a JSON file for the specified data.
        """
        with open(self.filename, 'w') as OUTPUT:
            jdata = {}
            if self.options.data is None:
                for k,v in data.items():
                    jdata[k] = detuplize(v)
            elif type(self.options.data) in (list, tuple):
                for k in self.options.data:
                    jdata[k] = detuplize(data[k], sort=self.options.sort)
            else:
                k = self.options.data
                jdata[k] = detuplize(data[k])
            json.dump(jdata, OUTPUT)

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

    alias("yaml", "YAML file interface")

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
        This function loads data from a YAML file and tuplizes the nested
        dictionaries and lists of lists.
        """
        if not os.path.exists(self.filename):
            raise IOError("Cannot find file '%s'" % self.filename)
        INPUT = open(self.filename, 'r')
        jdata = yaml.load(INPUT)
        INPUT.close()
        if jdata is None:
            raise IOError("Empty YAML file")
        self._info = {}
        for k,v in jdata.items():
            self._info[k] = tuplize(v)

    def write(self, data):
        """
        This function creates a YAML file for the specified data.
        """
        with open(self.filename, 'w') as OUTPUT:
            jdata = {}
            if self.options.data is None:
                for k,v in data.items():
                    jdata[k] = detuplize(v)
            elif type(self.options.data) in (list, tuple):
                for k in self.options.data:
                    jdata[k] = detuplize(data[k], sort=self.options.sort)
            else:
                k = self.options.data
                jdata[k] = detuplize(data[k])
            yaml.dump(jdata, OUTPUT)

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
            raise IOError("Data value for '%s' is not available in YAML file '%s'" % (key, self.filename))

    def _set_data(self, data, namespace, name, value):
        if type(value) is dict:
            data[namespace][name] = value
        else:
            data[namespace][name] = {None: value}

    def clear(self):
        self._info = {}


