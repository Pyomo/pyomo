#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import enum
from io import StringIO
from math import inf

from pyomo.common.collections import Bunch, Sequence, Mapping


class ScalarType(str, enum.Enum):
    int = 'int'
    time = 'time'
    string = 'string'
    float = 'float'
    enum = 'enum'
    undefined = 'undefined'

    # Overloading __str__ is needed to match the behavior of the old
    # pyutilib.enum class (removed June 2020). There are spots in the
    # code base that expect the string representation for items in the
    # enum to not include the class name. New uses of enum shouldn't
    # need to do this.
    def __str__(self):
        return self.value


default_print_options = Bunch(schema=False, ignore_time=False)
strict = False


class UndefinedData(object):
    singleton = {}

    def __new__(cls, name='undefined'):
        if name not in UndefinedData.singleton:
            UndefinedData.singleton[name] = super().__new__(cls)
            UndefinedData.singleton[name].name = name
        return UndefinedData.singleton[name]

    def __deepcopy__(self, memo):
        # Prevent deepcopy from duplicating this object
        return self

    def __reduce__(self):
        return self.__class__, (self.name,)

    def __str__(self):
        return f"<{self.name}>"


undefined = UndefinedData('undefined')
ignore = UndefinedData('ignore')


class ScalarData(object):
    def __init__(
        self,
        value=undefined,
        description=None,
        units=None,
        scalar_description=None,
        type=ScalarType.undefined,
        required=False,
    ):
        self.value = value
        self.description = description
        self.units = units
        self.scalar_description = scalar_description
        self.scalar_type = type
        self._required = required
        self._active = False

    def __eq__(self, other):
        return self.__dict__ == getattr(other, '__dict__', None)

    def get_value(self):
        if isinstance(self.value, enum.Enum):
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
            tmp = {'value': value}
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
            tmp = {'value': value}
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

        if value == inf:
            value = '.inf'
        elif value == -inf:
            value = '-.inf'

        if not option.schema and self.description is None and self.units is None:
            ostream.write(str(value) + '\n')
        else:
            ostream.write("\n")
            ostream.write(prefix + 'Value: ' + str(value) + '\n')
            if not option.schema:
                if not self.description is None:
                    ostream.write(
                        prefix
                        + 'Description: '
                        + self.yaml_fix(self.description)
                        + '\n'
                    )
                if not self.units is None:
                    ostream.write(prefix + 'Units: ' + str(self.units) + '\n')
            else:
                if not self.scalar_description is None:
                    ostream.write(
                        prefix
                        + 'Description: '
                        + self.yaml_fix(self.scalar_description)
                        + '\n'
                    )
                if not self.scalar_type is ScalarType.undefined:
                    ostream.write(
                        prefix + 'Type: ' + self.yaml_fix(self.scalar_type) + '\n'
                    )

    def yaml_fix(self, val):
        if not isinstance(val, str):
            return val
        return val.replace(':', '\\x3a')

    def load(self, repn):
        if type(repn) is dict:
            for key, val in repn.items():
                setattr(self, key, val)
        else:
            self.value = repn


#
# This class manages a list of MapContainer objects.
#
class ListContainer(object):
    def __init__(self, cls):
        self._cls = cls
        self._list = []
        self._active = True
        self._required = False

    def __len__(self):
        if '_list' in self.__dict__:
            return len(self._list)
        return 0

    def __getitem__(self, i):
        return self._list[i]

    def __eq__(self, other):
        return self.__dict__ == getattr(other, '__dict__', None)

    def clear(self):
        self._list = []

    def delete(self, i):
        del self._list[i]

    def __call__(self, i=0):
        return self._list[i]

    def __getattr__(self, name):
        if name[0] == "_":
            super().__getattr__(name)
        if len(self) == 0:
            self.add()
        return getattr(self._list[0], name)

    def __setattr__(self, name, val):
        if name[0] == "_":
            return super().__setattr__(name, val)
        if len(self) == 0:
            self.add()
        setattr(self._list[0], name, val)

    def insert(self, obj):
        self._active = True
        self._list.append(obj)

    def add(self):
        self._active = True
        obj = self._cls()
        self._list.append(obj)
        return obj

    def _repn_(self, option):
        if not option.schema and not self._active and not self._required:
            return ignore
        if option.schema and len(self) == 0:
            self.add()
        tmp = []
        for item in self._list:
            tmp.append(item._repn_(option))
        return tmp

    def pprint(self, ostream, option, prefix="", repn=None):
        if not option.schema and not self._active and not self._required:
            return ignore
        ostream.write("\n")
        i = 0
        for i in range(len(self._list)):
            item = self._list[i]
            ostream.write(prefix + '- ')
            item.pprint(
                ostream, option, from_list=True, prefix=prefix + "  ", repn=repn[i]
            )

    def load(self, repn):
        for data in repn:
            item = self.add()
            item.load(data)

    def __str__(self):
        ostream = StringIO()
        option = default_print_options
        self.pprint(ostream, option, repn=self._repn_(option))
        return ostream.getvalue()


#
# This class manages use-defined attributes in
# a dictionary.  Attributes are translated into
# a string where '_' is replaced by ' ', and where the
# first letter is capitalized.
#
class MapContainer(dict):

    def __init__(self, ordered=False):
        super().__init__()
        self._active = True
        self._required = False
        self._option = default_print_options

    def __eq__(self, other):
        # We need to check both our __dict__ (local attributes) and the
        # underlying dict data (which doesn't show up in the __dict__).
        # So we will use the base __eq__ in addition to checking
        # __dict__.
        #
        # Note: __eq__ can return True, False, or NotImplemented
        base = super().__eq__(other)
        if base == True:
            return self.__dict__ == getattr(other, '__dict__', None)
        return base

    def __getattr__(self, name):
        try:
            self._active = True
            return self[self._convert(name)]
        except Exception:
            pass
        raise AttributeError(
            "Unknown attribute `"
            + str(name)
            + "' for object with type "
            + str(type(self))
        )

    def __setattr__(self, name, val):
        if name[0] == "_":
            return super().__setattr__(name, val)
        self._active = True
        tmp = self._convert(name)
        if tmp not in self:
            if strict:
                raise AttributeError(
                    "Unknown attribute `"
                    + str(name)
                    + "' for object with type "
                    + str(type(self))
                )
            self.declare(tmp)
        self._set_value(tmp, val)

    def __setitem__(self, name, val):
        self._active = True
        tmp = self._convert(name)
        if tmp not in self:
            if strict:
                raise AttributeError(
                    "Unknown attribute `"
                    + str(name)
                    + "' for object with type "
                    + str(type(self))
                )
            self.declare(tmp)
        self._set_value(tmp, val)

    def _set_value(self, name, val):
        if isinstance(val, (ListContainer, MapContainer)):
            super().__setitem__(name, val)
        elif isinstance(val, ScalarData):
            data = super().__getitem__(name)
            data.value = val.value
            data._active = val._active
            data._required = val._required
            data.scalar_type = val.scalar_type
        else:
            data = super().__getitem__(name)
            data.value = val
            data._active = True

    def __getitem__(self, name):
        tmp = self._convert(name)
        if tmp not in self:
            raise AttributeError(
                "Unknown attribute `"
                + str(name)
                + "' for object with type "
                + str(type(self))
            )
        item = super().__getitem__(tmp)
        if isinstance(item, (ListContainer, MapContainer)):
            return item
        return item.value

    def declare(self, name, **kwds):
        if name in self or type(name) is int:
            return
        data = kwds.get('value', None)
        if isinstance(data, (MapContainer, ListContainer)):
            if 'active' in kwds:
                data._active = kwds['active']
            if 'required' in kwds and kwds['required'] is True:
                data._required = True
            super().__setitem__(self._convert(name), data)
        else:
            data = ScalarData(**kwds)
            if 'required' in kwds and kwds['required'] is True:
                data._required = True
            #
            # This logic would setup a '_default' value, which copies the
            # initial value of an attribute.  I don't think we need this,
            # but for now I'm going to leave this logic in the code.
            #
            # if 'value' in kwds:
            #    data._default = kwds['value']
            super().__setitem__(self._convert(name), data)

    def _repn_(self, option):
        if not option.schema and not self._active and not self._required:
            return ignore
        tmp = {}
        for key, val in self.items():
            rep = val._repn_(option)
            if not rep == ignore:
                tmp[key] = rep
        return tmp

    def _convert(self, name):
        if not isinstance(name, str):
            return name
        tmp = name.replace('_', ' ')
        return tmp[0].upper() + tmp[1:]

    def __repr__(self):
        return str(self._repn_(self._option))

    def __str__(self):
        ostream = StringIO()
        self.pprint(ostream, self._option, repn=self._repn_(self._option))
        return ostream.getvalue()

    def pprint(self, ostream, option, from_list=False, prefix="", repn=None):
        if from_list:
            _prefix = ""
        else:
            _prefix = prefix
            ostream.write('\n')
        for key, item in self.items():
            if not key in repn:
                continue
            ostream.write(_prefix + key + ": ")
            _prefix = prefix
            if isinstance(item, ListContainer):
                item.pprint(ostream, option, prefix=_prefix, repn=repn[key])
            else:
                item.pprint(ostream, option, prefix=_prefix + "  ", repn=repn[key])

    def load(self, repn):
        for key, val in repn.items():
            tmp = self._convert(key)
            if tmp not in self:
                self.declare(tmp)
            item = super().__getitem__(tmp)
            item._active = True
            item.load(val)


# Register these as sequence / mapping types (so things like
# assertStructuredAlmostEqual will process them correctly)
Sequence.register(ListContainer)
Mapping.register(MapContainer)
