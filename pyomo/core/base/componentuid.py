#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
#####

import bisect
import codecs
import re
import ply.lex

from six import string_types, iteritems

from pyomo.common.collections import ComponentMap
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice

class ComponentUID(object):
    """
    A Component unique identifier

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
    """

    __slots__ = ( '_cids', )

    _lex = None
    _repr_map = {
        slice: lambda x: '*',
        Ellipsis.__class__: lambda x: '**',
        int: repr,
        float: repr,
        str: repr,
    }
    _repr_v1_map = {
        slice: lambda x: '*',
        Ellipsis.__class__: lambda x: '**',
        int: lambda x: '#'+str(x),
        float: lambda x: '#'+str(x),
        str: lambda x: '$'+str(x),
    }

    def __init__(self, component, cuid_buffer=None, context=None):
        # A CUID can be initialized from either a reference component or
        # the string representation.
        if isinstance(component, string_types):
            if context is not None:
                raise ValueError("Context is not allowed when initializing a "
                                 "ComponentUID object from a string type")
            self._cids = tuple(self.parse_cuid(component))
        else:
            self._cids = tuple(self._generate_cuid(
                component, cuid_buffer=cuid_buffer, context=context))

    def __str__(self):
        """
        TODO
        """
        a = ""
        for name, args in reversed(self._cids):
            a += '.' + self._safe_str(repr(name))
            if len(args) == 0:
                continue
            a += '[' + ','.join(
                self._safe_str(self._repr_map.get(x.__class__, str)(x))
                for x in args) + ']'
        return a[1:]  # Strip off the leading '.'

    def __repr__(self):
        """
        TODO
        """
        a = ""
        for name, args in reversed(self._cids):
            a += '.' + repr(name)
            if len(args) == 0:
                continue
            a += '[' + ','.join(
                self._repr_map.get(x.__class__, str)(x)
                for x in args) + ']'
        return a[1:]  # Strip off the leading '.'

    def get_repr(self, version=2):
        if version == 1:
            a = ""
            for name, args in reversed(self._cids):
                a += '.' + name
                if len(args) == 0:
                    continue
                a += ':' + ','.join(
                    self._repr_v1_map.get(x.__class__, str)(x)
                    for x in args)
            return a[1:]  # Strip off the leading '.'
        elif version == 2:
            return repr(self)
        else:
            raise ValueError("Invalid repr version '%s'; expected 1 or 2"
                             % (version,))


    def __getstate__(self):
        return {x:getattr(self, x) for x in ComponentUID.__slots__}

    def __setstate__(self, state):
        for key, val in iteritems(state):
            setattr(self,key,val)

    def __hash__(self):
        """
        TODO
        """
        try:
            return hash(self._cids)
        except TypeError:
            return hash(tuple(
                (name, tuple(
                    (slice, x.start, x.stop, x.step)
                    if x.__class__ is slice else x
                    for x in idx)) for name, idx in self._cids))

    def __lt__(self, other):
        """
        TODO
        """
        try:
            other_cids = other._cids
        except AttributeError:
            raise TypeError("'<' not supported between instances of "
                            "'ComponentUID' and '%s'" % (type(other).__name__))
        for (self_name, self_idx), (other_name, other_idx) in zip(
                self._cids, other_cids):
            if self_name != other_name:
                return self_name < other_name
            for self_i, other_i in zip(self_idx, other_idx):
                if self_i != other_i:
                    if other_i is Ellipsis:
                        return True
                    if self_i is Ellipsis:
                        return False
                    if other_i.__class__ is slice:
                        return True
                    if self_i.__class__ is slice:
                        return False
                    try:
                        return self_i < other_i
                    except:
                        return str(type(self_i)) < str(type(other_i))
            if len(self_idx) != len(other_idx):
                return len(self_idx) < len(other_idx)
        if len(self._cids) != len(other_cids):
            return len(self._cids) < len(other_cids)
        return False

    def __le__(self, other):
        """
        TODO
        """
        return self < other or self == other

    def __gt__(self, other):
        """
        TODO
        """
        return not (self <= other)

    def __ge__(self, other):
        """
        TODO
        """
        return not (self < other)

    def __eq__(self, other):
        """
        TODO
        """
        try:
            other_cids = other._cids
        except AttributeError:
            return False
        return self._cids == other_cids

    def __ne__(self, other):
        """
        TODO
        """
        return not self.__eq__(other)

    @staticmethod
    def _safe_str(x):
        if len(x) < 2 or x[0] not in '"\'' or x[0] != x[-1]:
            return x
        if any(_ in x for _ in ('\\ ' + literals)):
            return x
        return x[1:-1]

    @staticmethod
    def generate_cuid_string_map(block, ctype=None, descend_into=True,
                                 repr_version=2):
        def _record_indexed_object_cuid_strings_v1(obj, cuid_str):
            for idx, data in iteritems(obj):
                if idx.__class__ is tuple:
                    cuid_strings[data] = cuid_str + ':' + ','.join(
                        tDict.get(type(x), '?') + str(x) for x in idx)
                else:
                    cuid_strings[data] \
                        = cuid_str + ':' + tDict.get(type(idx), '?') + str(idx)

        def _record_indexed_object_cuid_strings_v2(obj, cuid_str):
            for idx, data in iteritems(obj):
                if idx.__class__ is tuple:
                    cuid_strings[data] = cuid_str + '[' + ','.join(
                        repr(x) for x in idx) + ']'
                else:
                    cuid_strings[data] \
                        = cuid_str + '[' + repr(idx) + ']'

        _record_indexed_object_cuid_strings = {
            1: _record_indexed_object_cuid_strings_v1,
            2: _record_indexed_object_cuid_strings_v2,
        }[repr_version]
        tDict = {int: '#', float: '#', str: '$'}
        _record_name = {
            1: str,
            2: repr,
        }[repr_version]

        model = block.model()
        cuid_strings = ComponentMap()
        cuid_strings[block] = ComponentUID(block).get_repr(repr_version)
        for blk in block.block_data_objects(descend_into=descend_into):
            if blk not in cuid_strings:
                blk_comp = blk.parent_component()
                cuid_str = _record_name(blk_comp.local_name)
                blk_pblk = blk_comp.parent_block()
                if blk_pblk is not model:
                    cuid_str = cuid_strings[blk_pblk] + '.' + cuid_str
                cuid_strings[blk_comp] = cuid_str
                if blk_comp.is_indexed():
                    _record_indexed_object_cuid_strings(blk_comp, cuid_str)
            for obj in blk.component_objects(ctype=ctype, descend_into=False):
                cuid_str = _record_name(obj.local_name)
                if blk is not model:
                    cuid_str = cuid_strings[blk] + '.' + cuid_str
                cuid_strings[obj] = cuid_str
                if obj.is_indexed():
                    _record_indexed_object_cuid_strings(obj, cuid_str)
        return cuid_strings

    def _generate_cuid(self, component, cuid_buffer=None, context=None):
        """
        TODO
        """
        model = component.model()
        if context is None:
            context = model
        orig_component = component
        # if component.is_indexed():
        #     yield( component.local_name, (Ellipsis,))
        #     component = component.parent_block()
        while component is not context:
            if component is model:
                raise ValueError("Context '%s' does not apply to component "
                                 "'%s'" % (context.name,
                                           orig_component.name))
            c = component.parent_component()
            if c is component:
                yield ( c.local_name, () )
            elif cuid_buffer is not None:
                if id(component) not in cuid_buffer:
                    for idx, obj in iteritems(c):
                        cuid_buffer[id(obj)] = (
                            c.local_name,
                            idx if idx.__class__ is tuple else (idx,)
                        )
                yield cuid_buffer[id(component)]
            else:
                idx = component.index()
                yield c.local_name, idx if idx.__class__ is tuple else (idx,)
            component = component.parent_block()

    def parse_cuid(self, label):
        """
        TODO
        """
        try:
            return self._parse_cuid_v2(label)
        except (OSError, IOError):
            return self._parse_cuid_v1(label)

    def _parse_cuid_v2(self, label):
        if ComponentUID._lex is None:
            ComponentUID._lex = ply.lex.lex()
            ComponentUID._lex.linepos = []

        cuids = []
        name = None
        idx = []
        in_idx = False
        self._lex.input(label)
        while True:
            tok = self._lex.token()
            if not tok:
                break
            if tok.type == '.':
                assert not in_idx
                cuids.append((name, tuple(idx)))
                name = None
                idx = []
            elif tok.type == '[':
                assert not in_idx
                in_idx = True
            elif tok.type == ']':
                assert in_idx
                in_idx = False
            elif in_idx: # starting a component
                if tok.type == ',':
                    pass
                elif tok.type == 'STAR':
                    idx.append(tok.value)
                else:
                    assert tok.type in {'WORD','STRING','NUMBER'}
                    idx.append(tok.value)
            else:
                assert tok.type in {'WORD','STRING'}
                assert name is None
                name = tok.value
        assert not in_idx
        cuids.append((name, tuple(idx)))
        cuids.reverse()
        return cuids
            
    def _parse_cuid_v1(self, label):
        cList = label.split('.')
        for c in reversed(cList):
            if c[-1] == ']':
                c_info = c[:-1].split('[',1)
            else:
                c_info = c.split(':',1)
            if len(c_info) == 1:
                yield ( c_info[0], tuple() )
            else:
                idx = c_info[1].split(',')
                for i, val in enumerate(idx):
                    if val == '*':
                        idx[i] = slice(None)
                    elif val[0] == '$':
                        idx[i] = str(val[1:])
                    elif val[0] == '#':
                        idx[i] = int(val[1:])
                    elif val[0] in  "\"'" and val[-1] == val[0]:
                        idx[i] = val[1:-1]
                    elif _re_number.match(val):
                        _num = float(val)
                        try:
                            _int = int(_num)
                        except:
                            _int = 0
                        idx[i] = _int if _int == _num else _num
                if len(idx) == 1 and idx[0] == '**':
                    yield ( c_info[0], (Ellipsis,) )
                else:
                    yield ( c_info[0], tuple(idx) )

    def _resolve_cuid(self, block):
        obj = block
        for name, idx in reversed(self._cids):
            #print obj.name, name, idx
            try:
                if len(idx):
                    obj = getattr(obj, name)[idx]
                else:
                    obj = getattr(obj, name)
            except KeyError:
                #print sys.exc_info()
                return None
            except AttributeError:
                #print sys.exc_info()
                return None
            except IndexError:
                #print sys.exc_info()
                return None
        return obj

    def find_component(self, block):
        return self.find_component_on(block)

    def find_component_on(self, block):
        """
        Return the (unique) component in the block.  If the CUID contains
        a wildcard in the last component, then returns that component.  If
        there are wildcards elsewhere (or the last component was a partial
        slice), then returns None.  See list_components below.
        """
        obj = self._resolve_cuid(block)
        if isinstance(obj, IndexedComponent_slice):
            return None
        return obj

    def list_components(self, block):
        """
        TODO
        """
        obj = self._resolve_cuid(block)
        if obj is None:
            # The initial generation of a component failed
            return

        if isinstance(obj, IndexedComponent_slice):
            # Suppress slice iteration exceptions
            obj.key_errors_generate_exceptions = False
            obj.attribute_errors_generate_exceptions = False
            for o in obj:
                yield o
        else:
            yield obj

    def matches(self, component, context=None):
        """
        TODO
        """
        for i, (name, idx) in enumerate(self._generate_cuid(component)):
            if i == len(self._cids):
                return False
            s_name, s_idx = self._cids[i]
            if s_name != name:
                return False
            for j, s_idx_val in enumerate(s_idx):
                if j >= len(idx):
                    return False
                if s_idx_val.__class__ is slice:
                    continue
                if s_idx_val is Ellipsis:
                    if len(idx) < len(s_idx) - 1:
                        return False
                    for _k in range(-1, j-len(s_idx), -1):
                        if s_idx[k].__class__ is slice:
                            continue
                        elif s_idx[k] != idx[k]:
                            return False
                    # Everything after the elipsis matched, so we can
                    # move on to the next level.
                    break
                if s_idx_val != idx[j]:
                    return False
        # Matched if all self._cids were consumed
        return i+1 == len(self._cids)



# Known escape sequences:
#   \U{8}: unicode 8-digit hex codes
#   \u{4}: unicode 4-digit hex codes
#   \x{2}: unicode 2-digit hex codes
#   \nnn: octal codes
#   \N{...}" unicode by name
#   \\, \', \", \a, \b, \f, \n, \r, \t, \v
_re_escape_sequences = re.compile(
    r"\\U[a-fA-F0-9]{8}|\\u[a-fA-F0-9]{4}|\\x[a-fA-F0-9]{2}" +
    r"|\\[0-7]{1,3}|\\N\{[^}]+\}|\\[\\'\"abfnrtv]", re.UNICODE | re.VERBOSE)

def _match_escape(match):
    return codecs.decode(match.group(0), 'unicode-escape')

_re_number = re.compile(
    r'[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?')

# Ignore whitespace (space, tab, and linefeed)
t_ignore = " \t\r"

literals = '[],.'

tokens = [
    "WORD",   # unquoted string
    "STRING", # quoted string
    "NUMBER", # raw number
    "STAR",   # either * or **
]

# Numbers only appear in getitem lists, so they must be followed by a
# delimiter token (one of ' ,]')
@ply.lex.TOKEN(r'(?:'+_re_number.pattern+r'|-?inf|nan)(?=[\s\],])')
def t_NUMBER(t):
    _num = float(t.value)
    try:
        _int = int(_num)
    except:
        _int = 0  # a random int
    t.value = _int if _num == _int else _num
    return t

@ply.lex.TOKEN(r'[a-zA-Z_][a-zA-Z_0-9]*')
def t_WORD(t):
    return t

_re_quoted_str = r'"(?:[^"\\]|\\.)*"'
@ply.lex.TOKEN("|".join([_re_quoted_str, _re_quoted_str.replace('"',"'")]))
def t_STRING(t):
    t.value = _re_escape_sequences.sub(_match_escape, t.value[1:-1])
    return t

@ply.lex.TOKEN(r'\*{1,2}')
def t_STAR(t):
    if len(t.value) == 1:
        t.value = slice(None)
    else:
        t.value = Ellipsis
    return t

def _lex_token_column(t):
    # Returns the column number within a line 
    i = bisect.bisect_left(t.lexer.linepos, t.lexpos)
    if i:
        return t.lexpos - t.lexer.linepos[i-1]
    return t.lexpos

# Error handling rule
def t_error(t):
    raise IOError("ERROR: Token %s Value %s Line %s Column %s"
                  % (t.type, t.value, t.lineno, _lex_token_column(t)))
