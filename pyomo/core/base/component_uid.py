#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['ComponentUID']

import logging
import six
import sys
from copy import deepcopy
from pickle import PickleError
from six import iteritems, string_types
from weakref import ref as weakref_ref

from pyutilib.misc.indent_io import StreamIndenter

import pyomo.common
from pyomo.common import deprecated
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base.misc import tabular_writer, sorted_robust

class ComponentUID(object):
    """
    This class provides a system to generate "component unique
    identifiers".  Any component in a model can be described by a CUID,
    and from a CUID you can find the component.  An important feature of
    CUIDs is that they are relative to a model, so you can use a CUID
    generated on one model to find the equivalent component on another
    model.  This is especially useful when you clone a model and want
    to, for example, copy a variable value from the cloned model back to
    the original model.

    The CUID has a string representation that can specify a specific
    component or a group of related components through the use of index
    wildcards (* for a single element in the index, and ** for all
    indexes)

    This class is also used by test_component.py to validate the structure
    of components.
    """

    __slots__ = ( '_cids', )
    tList = [ int, str ]
    tKeys = '#$'
    tDict = {} # ...initialized below

    def __init__(self, component, cuid_buffer=None, context=None):
        # A CUID can be initialized from either a reference component or
        # the string representation.
        if isinstance(component, string_types):
            if context is not None:
                raise ValueError("Context is not allowed when initializing a "
                                 "ComponentUID object from a string type")
            self._cids = tuple(self.parse_cuid(component))
        else:
            self._cids = tuple(self._generate_cuid(component,
                                                   cuid_buffer=cuid_buffer,
                                                   context=context))

    def __str__(self):
        """
        TODO
        """
        a = ""
        for name, args, types in reversed(self._cids):
            if a:
                a += '.' + name
            else:
                a = name
            if types is None:
                a += '[**]'
                continue
            if len(args) == 0:
                continue
            a += '['+','.join(str(x) or '*' for x in args) + ']'
        return a

    def __repr__(self):
        """
        TODO
        """
        a = ""
        for name, args, types in reversed(self._cids):
            if a:
                a += '.' + name
            else:
                a = name
            if types is None:
                a += ':**'
                continue
            if len(args) == 0:
                continue
            a += ':'+','.join( (types[i] if types[i] not in '.' else '')+str(x)
                               for i,x in enumerate(args) )
        return a

    def __getstate__(self):
        return {x:getattr(self, x) for x in ComponentUID.__slots__}

    def __setstate__(self, state):
        for key, val in iteritems(state):
            setattr(self,key,val)

    # Define all comparison operators using the underlying tuple's
    # comparison operators. We will be lazy and assume that the other is
    # a CUID.

    def __hash__(self):
        """
        TODO
        """
        return self._cids.__hash__()

    def __lt__(self, other):
        """
        TODO
        """
        try:
            return self._cids.__lt__(other._cids)
        except AttributeError:
            return self._cids.__lt__(other)

    def __le__(self, other):
        """
        TODO
        """
        try:
            return self._cids.__le__(other._cids)
        except AttributeError:
            return self._cids.__le__(other)

    def __gt__(self, other):
        """
        TODO
        """
        try:
            return self._cids.__gt__(other._cids)
        except AttributeError:
            return self._cids.__gt__(other)

    def __ge__(self, other):
        """
        TODO
        """
        try:
            return self._cids.__ge__(other._cids)
        except AttributeError:
            return self._cids.__ge__(other)

    def __eq__(self, other):
        """
        TODO
        """
        try:
            return self._cids.__eq__(other._cids)
        except AttributeError:
            return self._cids.__eq__(other)

    def __ne__(self, other):
        """
        TODO
        """
        try:
            return self._cids.__ne__(other._cids)
        except AttributeError:
            return self._cids.__ne__(other)

    def _partial_cuid_from_index(self, idx):
        """
        TODO
        """
        tDict = ComponentUID.tDict
        if idx.__class__ is tuple:
            return ( idx, ''.join(tDict.get(type(x), '?') for x in idx) )
        else:
            return ( (idx,), tDict.get(type(idx), '?') )

    def _generate_cuid(self, component, cuid_buffer=None, context=None):
        """
        TODO
        """
        model = component.model()
        if context is None:
            context = model
        orig_component = component
        tDict = ComponentUID.tDict
        if not hasattr(component, '_component'):
            yield ( component.local_name, '**', None )
            component = component.parent_block()
        while component is not context:
            if component is model:
                raise ValueError("Context '%s' does not apply to component "
                                 "'%s'" % (context.name,
                                           orig_component.name))
            c = component.parent_component()
            if c is component:
                yield ( c.local_name, tuple(), '' )
            elif cuid_buffer is not None:
                if id(self) not in cuid_buffer:
                    for idx, obj in iteritems(c):
                        cuid_buffer[id(obj)] = \
                            self._partial_cuid_from_index(idx)
                yield (c.local_name,) + cuid_buffer[id(component)]
            else:
                for idx, obj in iteritems(c):
                    if obj is component:
                        yield (c.local_name,) + self._partial_cuid_from_index(idx)
                        break
            component = component.parent_block()

    def parse_cuid(self, label):
        """
        TODO
        """
        cList = label.split('.')
        tKeys = ComponentUID.tKeys
        tDict = ComponentUID.tDict
        for c in reversed(cList):
            if c[-1] == ']':
                c_info = c[:-1].split('[',1)
            else:
                c_info = c.split(':',1)
            if len(c_info) == 1:
                yield ( c_info[0], tuple(), '' )
            else:
                idx = c_info[1].split(',')
                _type = ''
                for i, val in enumerate(idx):
                    if val == '*':
                        _type += '*'
                        idx[i] = ''
                    elif val[0] in tKeys:
                        _type += val[0]
                        idx[i] = tDict[val[0]](val[1:])
                    elif val[0] in  "\"'" and val[-1] == val[0]:
                        _type += ComponentUID.tDict[str]
                        idx[i] = val[1:-1]
                    else:
                        _type += '.'
                if len(idx) == 1 and idx[0] == '**':
                    yield ( c_info[0], '**', None )
                else:
                    yield ( c_info[0], tuple(idx), _type )

    def find_component_on(self, block):
        """
        TODO
        """
        return self.find_component(block)

    def find_component(self, block):
        """
        Return the (unique) component in the block.  If the CUID contains
        a wildcard in the last component, then returns that component.  If
        there are wildcards elsewhere (or the last component was a partial
        slice), then returns None.  See list_components below.
        """
        obj = block
        for name, idx, types in reversed(self._cids):
            try:
                if len(idx) and idx != '**' and types.strip('*'):
                    obj = getattr(obj, name)[idx]
                else:
                    obj = getattr(obj, name)
            except KeyError:
                if '.' not in types:
                    return None
                tList = ComponentUID.tList
                def _checkIntArgs(_idx, _t, _i):
                    if _i == -1:
                        try:
                            return getattr(obj, name)[tuple(_idx)]
                        except KeyError:
                            return None
                    _orig = _idx[_i]
                    for _cast in tList:
                        try:
                            _idx[_i] = _cast(_orig)
                            ans = _checkIntArgs(_idx, _t, _t.find('.',_i+1))
                            if ans is not None:
                                return ans
                        except ValueError:
                            pass
                    _idx[_i] = _orig
                    return None
                obj = _checkIntArgs(list(idx), types, types.find('.'))
            except AttributeError:
                return None
        return obj

    def _list_components(self, _obj, cids):
        """
        TODO
        """
        if not cids:
            yield _obj
            return

        name, idx, types = cids[-1]
        try:
            obj = getattr(_obj, name)
        except AttributeError:
            return
        if len(idx) == 0:
            for ans in self._list_components(obj, cids[:-1]):
                yield ans
        elif idx != '**' and '*' not in types and '.' not in types:
            try:
                obj = obj[idx]
            except KeyError:
                return
            for ans in self._list_components(obj, cids[:-1]):
                yield ans
        else:
            all =  idx == '**'
            tList = ComponentUID.tList
            for target_idx, target_obj in iteritems(obj):
                if not all and idx != target_idx:
                    _idx, _types = self._partial_cuid_from_index(target_idx)
                    if len(idx) != len(_idx):
                        continue
                    match = True
                    for j in range(len(idx)):
                        if idx[j] == _idx[j] or types[j] == '*':
                            continue
                        elif types[j] == '.':
                            ok = False
                            for _cast in tList:
                                try:
                                    if _cast(idx[j]) == _idx[j]:
                                        ok = True
                                        break
                                except ValueError:
                                    pass
                            if not ok:
                                match = False
                                break
                        else:
                            match = False
                            break
                    if not match:
                        continue
                for ans in self._list_components(target_obj, cids[:-1]):
                    yield ans

    def list_components(self, block):
        """
        TODO
        """
        for ans in self._list_components(block, self._cids):
            yield ans

    def matches(self, component):
        """
        TODO
        """
        tList = ComponentUID.tList
        for i, (name, idx, types) in enumerate(self._generate_cuid(component)):
            if i == len(self._cids):
                return False
            _n, _idx, _types = self._cids[i]
            if _n != name:
                return False
            if _idx == '**' or idx == _idx:
                continue
            if len(idx) != len(_idx):
                return False
            for j in range(len(idx)):
                if idx[j] == _idx[j] or _types[j] == '*':
                    continue
                elif _types[j] == '.':
                    ok = False
                    for _cast in tList:
                        try:
                            if _cast(_idx[j]) == idx[j]:
                                ok = True
                                break
                        except ValueError:
                            pass
                    if not ok:
                        return False
                else:
                    return False
        # Matched if all self._cids were consumed
        return i+1 == len(self._cids)

# WEH - What does it mean to initialize this dictionary outside
#       of the definition of this class?  Is tList populated
#       with all components???
ComponentUID.tDict.update( (ComponentUID.tKeys[i], v)
                           for i,v in enumerate(ComponentUID.tList) )
ComponentUID.tDict.update( (v, ComponentUID.tKeys[i])
                           for i,v in enumerate(ComponentUID.tList) )
