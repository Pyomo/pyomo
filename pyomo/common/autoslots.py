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
    ['has_dict', 'slots', 'slot_mappers', 'field_mappers', 'cls']
)

class AutoSlots(type):
    """Metaclass to automatically collect __slots__ forgeneric pickling

    This metaclass will add a __auto_slots__ class attribute to the
    class (and all derived classes.  This attribute is a 5-tuple:

       (has_dict, slots, slot_mappers, field_mappers, cls)

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

    cls: type
        The class that the __auto_slots__ tuple was actually attached to
        (used to detect when self.__auto_slots__ is actually reporting
        the attribute from a base class)

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
            has_dict, slots, slot_mappers, dict_mappers, cls)

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
    def remove_field(encode, val):
        if val is None:
            return val
        if encode:
            return None
        else:
            return val

    class Mixin(object):
        __slots__ = ()

        def __deepcopy__(self, memo):
            memo[id(self)] = ans = self.__class__.__new__(self.__class__)
            ans.__setstate__(deepcopy(self.__getstate__(), memo))
            return ans

        def __getstate__(self):
            if self.__auto_slots__.cls is not self.__class__:
                AutoSlots.collect_autoslots(self.__class__)
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
            if self.__auto_slots__.cls is not self.__class__:
                AutoSlots.collect_autoslots(self.__class__)
            # Map the slot values
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
    
# Trigger the definition of the __auto_slots__ on AutoSlots.Mixin.  All
# other derived classes will be triggered on first entry into
# __getstate__ or __setstate__
AutoSlots.collect_autoslots(AutoSlots.Mixin)

AutoSlotsMixin = AutoSlots.Mixin
