#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from collections import namedtuple
from copy import deepcopy
from weakref import ref as _weakref_ref

_autoslot_info = namedtuple(
    '_autoslot_info',
    ['has_dict', 'slots', 'slot_mappers', 'field_mappers']
)

class AutoSlots(type):
    """Metaclass to automatically collect __slots__ forgeneric pickling

    This metaclass will add a __auto_slots__ class attribute to the
    class (and all derived classes).  This attribute is a 4-tuple:

       (has_dict, slots, slot_mappers, field_mappers)

    has_dict: bool
        True if this class has a __dict__ attribute (that would need to
        be pickled in addition to the __slots__)

    slots: tuple
        Tuple co all slots declared for this class (the union of any
        slots declared locally with all slots declared on any base class)

    slot_mappers: dict
        Dict mapping index in all_slots to a function with sugnature
        `mapper(encode: bool, val: Any)` that can be used to encode or
        decode that slot

    field_mappers: dict
        Dict mapping field name in __dict__ to a function with sugnature
        `mapper(encode: bool, val: Any)` that can be used to encode or
        decode that field value.

    """
    _ignore_slots = {'__weakref__', '__dict__'}

    def __init__(cls, name, bases, classdict):
        super().__init__(name, bases, classdict)
        AutoSlots.collect_autoslots(cls)

    @staticmethod
    def collect_autoslots(cls):
        has_dict = '__dict__' in dir(cls.__mro__[0])

        slots = []
        seen = set()
        for c in reversed(cls.__mro__):
            for slot in getattr(c, '__slots__', ()):
                if slot in seen:
                    continue
                if slot in AutoSlots._ignore_slots:
                    continue
                seen.add(slot)
                slots.append(slot)
        slots = tuple(slots)

        slot_mappers = {}
        dict_mappers = {}
        for c in reversed(cls.__mro__):
            for slot, mapper in getattr(c, '__autoslot_mappers__', {}).items():
                if slot in seen:
                    slot_mappers[slots.index(slot)] = mapper
                else:
                    dict_mappers[slot] = mapper

        cls.__auto_slots__ = _autoslot_info(
            has_dict, slots, slot_mappers, dict_mappers)

    @staticmethod
    def weakref_mapper(encode, val):
        if val is None:
            return val
        if encode:
            return val()
        else:
            return _weakref_ref(val)

    @staticmethod
    def weakref_sequence_mapper(encode, val):
        if val is None:
            return val
        if encode:
            return val.__class__(v() for v in val)
        else:
            return val.__class__(_weakref_ref(v) for v in val)

    @staticmethod
    def encode_as_none(encode, val):
        """__autoslot_mappers__ mapper that will replace fields with None

        This mapper will encode the field as None (regardless of the
        current field value).  No mapping occurs when restoring a state.

        """
        if encode:
            return None
        else:
            return val

    class Mixin(object):
        __slots__ = ()

        def __init_subclass__(cls, **kwds):
            """Automatically define __auto_slots__ on derived subclasses

            This accomplishes the same thing as the AutoSlots metaclass
            without incurring the overhead / runtime penalty of using a
            metaclass.

            """
            super().__init_subclass__(**kwds)
            AutoSlots.collect_autoslots(cls)

        def __deepcopy__(self, memo):
            memo[id(self)] = ans = self.__class__.__new__(self.__class__)
            ans.__setstate__(deepcopy(self.__getstate__(), memo))
            return ans

        def __getstate__(self):
            slots = [getattr(self, attr) for attr in self.__auto_slots__.slots]
            for idx, mapper in self.__auto_slots__.slot_mappers.items():
                slots[idx] = mapper(True, slots[idx])
            if self.__auto_slots__.has_dict:
                fields = dict(self.__dict__)
                for name, mapper in self.__auto_slots__.field_mappers.items():
                    if name in fields:
                        fields[name] = mapper(True, fields[name])
                slots.append(fields)
            return slots

        def __setstate__(self, state):
            # Map (decode) the slot values
            for idx, mapper in self.__auto_slots__.slot_mappers.items():
                state[idx] = mapper(False, state[idx])
            #
            # Note: per the Python data model docs, we explicitly set the
            # attribute using object.__setattr__() instead of setting
            # self.__dict__[key] = val.
            #
            # Restore the slots
            setter = object.__setattr__
            for attr, val in zip(self.__auto_slots__.slots, state):
                setter(self, attr, val)
            # If this is not slotized, then pull off the __dict__ fields and
            # map their values (if necessary)
            if self.__auto_slots__.has_dict:
                fields = state[-1]
                for name, mapper in self.__auto_slots__.field_mappers.items():
                    if name in fields:
                        fields[name] = mapper(False, fields[name])
                self.__dict__.clear()
                self.__dict__.update(fields)
    
