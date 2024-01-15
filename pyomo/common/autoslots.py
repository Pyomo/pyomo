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

import types
from collections import namedtuple
from copy import deepcopy
from weakref import ref as _weakref_ref

_autoslot_info = namedtuple(
    '_autoslot_info', ['has_dict', 'slots', 'slot_mappers', 'field_mappers']
)


def _deepcopy_tuple(obj, memo, _id):
    ans = []
    unchanged = True
    for item in obj:
        new_item = fast_deepcopy(item, memo)
        ans.append(new_item)
        if new_item is not item:
            unchanged = False
    if unchanged:
        # Python does not duplicate "unchanged" tuples (i.e. allows the
        # original objecct to be returned from deepcopy()).  We will
        # preserve that behavior here.
        #
        # It also appears to be faster *not* to cache the fact that this
        # particular tuple was unchanged by the deepcopy (Note: the
        # standard library also does not cache the unchanged tuples in
        # the memo)
        #
        #  memo[_id] = obj
        return obj
    memo[_id] = ans = tuple(ans)
    return ans


def _deepcopy_list(obj, memo, _id):
    # Two steps here because a list can include itself
    memo[_id] = ans = []
    ans.extend(fast_deepcopy(x, memo) for x in obj)
    return ans


def _deepcopy_dict(obj, memo, _id):
    # Two steps here because a dict can include itself
    memo[_id] = ans = {}
    for key, val in obj.items():
        ans[fast_deepcopy(key, memo)] = fast_deepcopy(val, memo)
    return ans


def _deepcopier(obj, memo, _id):
    return deepcopy(obj, memo)


_atomic_types = {
    int,
    float,
    bool,
    complex,
    bytes,
    str,
    type,
    range,
    type(None),
    types.BuiltinFunctionType,
    types.FunctionType,
}

_deepcopy_mapper = {tuple: _deepcopy_tuple, list: _deepcopy_list, dict: _deepcopy_dict}


def fast_deepcopy(obj, memo):
    """A faster implementation of copy.deepcopy()

    Python's default implementation of deepcopy has several features that
    are slower than they need to be.  This is an implementation of
    deepcopy that provides special handling to circumvent some of the
    slowest parts of deepcopy().

    """
    if obj.__class__ in _atomic_types:
        return obj
    _id = id(obj)
    if _id in memo:
        return memo[_id]
    else:
        return _deepcopy_mapper.get(obj.__class__, _deepcopier)(obj, memo, _id)


class AutoSlots(type):
    """Metaclass to automatically collect `__slots__` for generic pickling

    The class `__slots__` are collected in reverse MRO order.

    Any fields that require special handling are handled through
    callbacks specified through the `__autoslot_mappers__` class
    attribute.  `__autoslot_mappers__` should be a `dict` that maps the
    field name (either `__slot__` or regular `__dict__` entry) to a
    function with the signature:

        mapper(encode: bool, val: Any) -> Any

    The value from the object field (or state) is passed to the mapper
    function, and the function returns the corrected value.
    `__getstate__` calls the mapper with `encode=True`, and
    `__setstate__` calls the mapper with `encode=False`.
    `__autoslot_mappers__` class attributes are collected and combined
    in reverse MRO order (so duplicate mappers in more derived classes
    will replace mappers defined in base classes).

    :py:class:`AutoSlots` defines several common mapper functions, including:

      - :py:meth:`AutoSlots.weakref_mapper`
      - :py:meth:`AutoSlots.weakref_sequence_mapper`
      - :py:meth:`AutoSlots.encode_as_none`

    Result
    ~~~~~~

    This metaclass will add a `__auto_slots__` class attribute to the
    class (and all derived classes).  This attribute is an instance of a
    :py:class:`_autoslot_info` named 4-tuple:

       (has_dict, slots, slot_mappers, field_mappers)

    has_dict: bool
        True if this class has a `__dict__` attribute (that would need to
        be pickled in addition to the `__slots__`)

    slots: tuple
        Tuple of all slots declared for this class (the union of any
        slots declared locally with all slots declared on any base class)

    slot_mappers: dict
        Dict mapping index in `slots` to a function with signature
        `mapper(encode: bool, val: Any)` that can be used to encode or
        decode that slot

    field_mappers: dict
        Dict mapping field name in `__dict__` to a function with signature
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

        cls.__auto_slots__ = _autoslot_info(has_dict, slots, slot_mappers, dict_mappers)

    @staticmethod
    def weakref_mapper(encode, val):
        """__autoslot_mappers__ mapper for fields that contain weakrefs

        This mapper expects to be passed a field containing either a
        weakref or None.  It will resolve the weakref to a hard
        reference when generating a state, and then convert the hard
        reference back to a weakref when restoring the state.

        """
        if val is None:
            return val
        if encode:
            return val()
        else:
            return _weakref_ref(val)

    @staticmethod
    def weakref_sequence_mapper(encode, val):
        """__autoslot_mappers__ mapper for fields with sequences of weakrefs

        This mapper expects to be passed a field that is a sequence of
        weakrefs.  It will resolve all weakrefs when generating a state,
        and then convert the hard references back to a weakref when
        restoring the state.

        """
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
        """Mixin class to configure a class hierarchy to use AutoSlots

        Inheriting from this class will set up the automatic generation
        of the `__auto_slots__` class attribute, and define the standard
        implementations for `__deepcopy__`, `__getstate__`, and
        `__setstate__`.

        """

        __slots__ = ()

        def __init_subclass__(cls, **kwds):
            """Automatically define `__auto_slots__` on derived subclasses

            This accomplishes the same thing as the AutoSlots metaclass
            without incurring the overhead / runtime penalty of using a
            metaclass.

            """
            super().__init_subclass__(**kwds)
            AutoSlots.collect_autoslots(cls)

        def __deepcopy__(self, memo):
            """Default implementation of `__deepcopy__` based on `__getstate__`

            This defines a default implementation of `__deepcopy__` that
            leverages :py:meth:`__getstate__` and :py:meth:`__setstate__`
            to duplicate an object.  Having a default `__deepcopy__`
            implementation shortcuts significant logic in
            :py:func:`copy.deepcopy()`, thereby speeding up deepcopy
            operations.

            """
            # Note: this implementation avoids deepcopying the temporary
            # 'state' list, significantly speeding things up.
            memo[id(self)] = ans = self.__class__.__new__(self.__class__)
            ans.__setstate__(
                [fast_deepcopy(field, memo) for field in self.__getstate__()]
            )
            return ans

        def __getstate__(self):
            """Generic implementation of `__getstate__`

            This implementation will collect the slots (in order) and
            then the `__dict__` (if necessary) and place everything into a
            `list`.  This standard format is significantly faster to
            generate and deepcopy (when compared to a `dict`), although
            it can be more fragile (changing the number of slots can
            cause a pickle to no longer be loadable)

            Derived classes should not overload this method to provide
            special handling for fields (e.g., to resolve weak
            references).  Instead, special field handlers should be
            declared via the `__autoslot_mappers__` class attribute (see
            :py:class:`AutoSlots`)

            """
            slots = [getattr(self, attr) for attr in self.__auto_slots__.slots]
            # Map (encode) the slot values
            for idx, mapper in self.__auto_slots__.slot_mappers.items():
                slots[idx] = mapper(True, slots[idx])
            # Copy and add the fields from __dict__ (if present)
            if self.__auto_slots__.has_dict:
                fields = dict(self.__dict__)
                # Map (encode) any field values.  It is not an error if
                # the field if not present.
                for name, mapper in self.__auto_slots__.field_mappers.items():
                    if name in fields:
                        fields[name] = mapper(True, fields[name])
                slots.append(fields)
            return slots

        def __setstate__(self, state):
            """Generic implementation of `__setstate__`

            Restore the state generated by :py:meth:`__getstate__()`

            Derived classes should not overload this method to provide
            special handling for fields (e.g., to restore weak
            references).  Instead, special field handlers should be
            declared via the `__autoslot_mappers__` class attribute (see
            :py:class:`AutoSlots`)

            """
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
            # If this class is not fully slotized, then pull off the
            # __dict__ fields and map their values (if necessary)
            if self.__auto_slots__.has_dict:
                fields = state[-1]
                for name, mapper in self.__auto_slots__.field_mappers.items():
                    if name in fields:
                        fields[name] = mapper(False, fields[name])
                # Note that it appears to be faster to clear()/update()
                # than to simplify assign to __dict__.
                self.__dict__.clear()
                self.__dict__.update(fields)
