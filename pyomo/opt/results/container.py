#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['UndefinedData', 'undefined', 'ignore', 'ScalarData', 'ListContainer', 'MapContainer', 'default_print_options', 'ScalarType']

import copy
import math

import pyutilib.math
from pyutilib.misc import Bunch
from pyutilib.enum import EnumValue, Enum

from six import iterkeys, itervalues, iteritems, advance_iterator, StringIO
from six.moves import xrange
try:
    unicode
except NameError:
    basestring = unicode = str

ScalarType = Enum('int', 'time', 'string', 'float', 'enum', 'undefined')

default_print_options = Bunch(schema=False, ignore_time=False)

strict=False

class UndefinedData(object):

    def __str__(self):
        return "<undefined>"

undefined = UndefinedData()
ignore    = UndefinedData()


class ScalarData(object):

    def __init__(self, value=undefined, description=None, units=None, scalar_description=None, type=ScalarType.undefined, required=False):
        self.value = value
        self.description = description
        self.units = units
        self.scalar_description = scalar_description
        self.scalar_type = type
        self._required=required

    def get_value(self):
        if type(self.value) is EnumValue:
            value = str(self.value)
        elif type(self.value) is UndefinedData:
            value = '<undefined>'
        else:
            value = self.value
        return value

    def _repn_(self, option):
        if not option.schema and not self._required and self.value is undefined:
            return ignore
        if option.ignore_time and str(self.scalar_type) == str(ScalarType.time):
            return ignore
        value = self.get_value()

        if option.schema:
            tmp = {'value':value}
            if not self.description is None:
                tmp['description'] = self.description
            if not self.units is None:
                tmp['units'] = self.units
            if not self.scalar_description is None:
                tmp['description'] = self.scalar_description
            if not self.scalar_type is ScalarType.undefined:
                tmp['type'] = self.scalar_type
            return tmp
        if not (self.description is None and self.units is None):
            tmp = {'value':value}
            if not self.description is None:
                tmp['description'] = self.description
            if not self.units is None:
                tmp['units'] = self.units
            return tmp
        return value

    def pprint(self, ostream, option, prefix="", repn=None):
        if not option.schema and not self._required and self.value is undefined:
            return ignore
        if option.ignore_time and str(self.scalar_type) == str(ScalarType.time):
            return ignore

        value = self.yaml_fix(self.get_value())

        if value is pyutilib.math.infinity:
            value = '.inf'
        elif value is - pyutilib.math.infinity:
            value = '-.inf'

        if not option.schema and self.description is None and self.units is None:
            ostream.write(str(value)+'\n')
        else:
            ostream.write("\n")
            ostream.write(prefix+'Value: '+str(value)+'\n')
            if not option.schema:
                if not self.description is None:
                    ostream.write(prefix+'Description: '+self.yaml_fix(self.description)+'\n')
                if not self.units is None:
                    ostream.write(prefix+'Units: '+str(self.units)+'\n')
            else:
                if not self.scalar_description is None:
                    ostream.write(prefix+'Description: '+self.yaml_fix(self.scalar_description)+'\n')
                if not self.scalar_type is ScalarType.undefined:
                    ostream.write(prefix+'Type: '+self.yaml_fix(self.scalar_type)+'\n')

    def yaml_fix(self, val):
        if not isinstance(val,basestring):
            return val
        return val.replace(':','\\x3a')

    def load(self, repn):
        if type(repn) is dict:
            for key in repn:
                setattr(self, key, repn[key])
        else:
            self.value = repn


#
# This class manages a list of MapContainer objects.
#
class ListContainer(object):

    def __init__(self, cls):
        self._cls=cls
        self._list = []
        self._active=True
        self._required=False

    def __len__(self):
        if '_list' in self.__dict__:
            return len(self.__dict__['_list'])
        return 0

    def __getitem__(self,i):
        return self._list[i]

    def clear(self):
        self._list = []

    def delete(self, i):
        del self._list[i]

    def __call__(self,i=0):
        return self._list[i]

    def __getattr__(self,name):
        try:
            return self.__dict__[name]
        except:
            pass
        if len(self) == 0:
            self.add()
        return getattr(self._list[0], name)

    def __setattr__(self,name,val):
        if name == "__class__":
            self.__class__ = val
            return
        if name[0] == "_":
            self.__dict__[name] = val
            return
        if len(self) == 0:
            self.add()
        setattr(self._list[0], name, val)

    def insert(self, obj):
        self._active=True
        self._list.append( obj )

    def add(self):
        self._active=True
        obj = self._cls()
        self._list.append( obj )
        return obj

    def _repn_(self, option):
        if not option.schema and not self._active and not self._required:
            return ignore
        if option.schema and len(self) == 0:
            self.add()
        tmp = []
        for item in self._list:
            tmp.append( item._repn_(option) )
        return tmp

    def pprint(self, ostream, option, prefix="", repn=None):
        if not option.schema and not self._active and not self._required:
            return ignore
        ostream.write("\n")
        i=0
        for i in xrange(len(self._list)):
            item = self._list[i]
            ostream.write(prefix+'- ')
            item.pprint(ostream, option, from_list=True, prefix=prefix+"  ", repn=repn[i])

    def load(self, repn):
        for data in repn:
            item = self.add()
            item.load(data)

    def __getstate__(self):
        return copy.copy(self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        ostream = StringIO()
        option=default_print_options
        self.pprint(ostream, self._option, repn=self._repn_(self._option))
        return ostream.getvalue()


#
# This class manages use-defined attributes in
# a dictionary.  Attributes are translated into
# a string where '_' is replaced by ' ', and where the
# first letter is capitalized.
#
class MapContainer(dict):

    def __getnewargs_ex__(self):
        # Pass arguments to __new__ when unpickling
        return ((0,0),{})

    def __getnewargs__(self):
        # Pass arguments to __new__ when unpickling
        return (0,0)

    def __new__(cls, *args, **kwargs):
        #
        # If the user provides "too many" arguments, then 
        # pre-initialize the '_order' attribute.  This pre-initializes
        # the class during unpickling.
        #
        _instance = super(MapContainer, cls).__new__(cls, *args, **kwargs)
        if len(args) > 1:
            super(MapContainer, _instance).__setattr__('_order',[])
        return _instance

    def __init__(self, ordered=False):
        dict.__init__(self)
        self._active=True
        self._required=False
        self._ordered=ordered
        self._order=[]
        self._option=default_print_options

    def keys(self):
        return self._order

    def __getattr__(self,name):
        try:
            return self.__dict__[name]
        except:
            pass
        try:
            self._active=True
            return self[self._convert(name)]
        except Exception:
            pass
        raise AttributeError("Unknown attribute `"+str(name)+"' for object with type "+str(type(self)))

    def __setattr__(self,name,val):
        if name == "__class__":
            self.__class__ = val
            return
        if name[0] == "_":
            self.__dict__[name] = val
            return
        self._active=True
        tmp = self._convert(name)
        if tmp not in self:
            if strict:
                raise AttributeError("Unknown attribute `"+str(name)+"' for object with type "+str(type(self)))
            self.declare(tmp)
        self._set_value(tmp,val)

    def __setitem__(self, name, val):
        self._active=True
        tmp = self._convert(name)
        if tmp not in self:
            if strict:
                raise AttributeError("Unknown attribute `"+str(name)+"' for object with type "+str(type(self)))
            self.declare(tmp)
        self._set_value(tmp,val)

    def _set_value(self, name, val):
        if isinstance(val,ListContainer) or isinstance(val,MapContainer):
            dict.__setitem__(self, name, val)
        elif isinstance(val,ScalarData):
            dict.__getitem__(self, name).value = val.value
        else:
            dict.__getitem__(self, name).value = val

    def __getitem__(self, name):
        tmp = self._convert(name)
        if tmp not in self:
            raise AttributeError("Unknown attribute `"+str(name)+"' for object with type "+str(type(self)))
        item = dict.__getitem__(self, tmp)
        if isinstance(item,ListContainer) or isinstance(item,MapContainer):
            return item
        return item.value

    def declare(self, name, **kwds):
        if name in self or type(name) is int:
            return
        tmp = self._convert(name)
        self._order.append(tmp)
        if 'value' in kwds and (isinstance(kwds['value'],MapContainer) or isinstance(kwds['value'],ListContainer)):
            if 'active' in kwds:
                kwds['value']._active = kwds['active']
            if 'required' in kwds and kwds['required'] is True:
                kwds['value']._required = True
            dict.__setitem__(self, tmp, kwds['value'])
        else:
            data = ScalarData(**kwds)
            if 'required' in kwds and kwds['required'] is True:
                data._required = True
            #
            # This logic would setup a '_default' value, which copies the
            # initial value of an attribute.  I don't think we need this,
            # but for now I'm going to leave this logic in the code.
            #
            #if 'value' in kwds:
            #    data._default = kwds['value']
            dict.__setitem__(self, tmp, data)

    def _repn_(self, option):
        if not option.schema and not self._active and not self._required:
            return ignore
        if self._ordered:
            tmp = []
            for key in self._order:
                rep = dict.__getitem__(self, key)._repn_(option)
                if not rep == ignore:
                    tmp.append({key:rep})
        else:
            tmp = {}
            for key in self.keys():
                rep = dict.__getitem__(self, key)._repn_(option)
                if not rep == ignore:
                    tmp[key] = rep
        return tmp

    def _convert(self, name):
        if not isinstance(name,basestring):
            return name
        tmp = name.replace('_',' ')
        return tmp[0].upper() + tmp[1:]

    def __repr__(self):
        return str(self._repn_(self._option))

    def __str__(self):
        ostream = StringIO()
        option=default_print_options
        self.pprint(ostream, self._option, repn=self._repn_(self._option))
        return ostream.getvalue()

    def pprint(self, ostream, option, from_list=False, prefix="", repn=None):
        if from_list:
            _prefix=""
        else:
            _prefix=prefix
            ostream.write('\n')
        for key in self._order:
            if not key in repn:
                continue
            item = dict.__getitem__(self,key)
            ostream.write(_prefix+key+": ")
            _prefix=prefix
            if isinstance(item, ListContainer):
                item.pprint(ostream, option, prefix=_prefix, repn=repn[key])
            else:
                item.pprint(ostream, option, prefix=_prefix+"  ", repn=repn[key])

    def load(self, repn):
        for key in repn:
            tmp = self._convert(key)
            if tmp not in self:
                self.declare(tmp)
            item = dict.__getitem__(self,tmp)
            item._active=True
            item.load(repn[key])

    def __getnewargs__(self):
        return (False, False)

    def __getstate__(self):
        return copy.copy(self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)


if __name__ == '__main__':
    d=MapContainer()
    d.declare('f')
    d.declare('g')
    d.declare('h')
    d.declare('i', value=ListContainer(UndefinedData))
    d.declare('j', value=ListContainer(UndefinedData), active=False)
    print("X")
    d.f = 1
    print("Y")
    print(d.f)
    print(d.keys())
    d.g = None
    print(d.keys())
    try:
        print(d.f, d.g, d.h)
    except:
        pass
    d['h'] = None
    print("")
    print("FINAL")
    print(d.f, d.g, d.h, d.i, d.j)
    print(d.i._active, d.j._active)
    d.j.add()
    print(d.i._active, d.j._active)
