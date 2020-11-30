#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import bisect
import codecs
import re
import ply.lex

from six import PY2, string_types, iteritems

from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import pickle
from pyomo.common.deprecation import deprecated
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import Reference

class _PickleEllipsis(object):
    "A work around for the non-picklability of Ellipsis in Python 2"
    pass

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

    @staticmethod
    def _safe_str_tuple(x):
        return '(' + ','.join(ComponentUID._safe_str(_) for _ in x) + ',)'

    @staticmethod
    def _pickle(x):
        return '|'+repr(pickle.dumps(x, protocol=2))

    @staticmethod
    def _safe_str(x):
        if not isinstance(x, string_types):
            return ComponentUID._repr_map.get(
                x.__class__, ComponentUID._pickle)(x)
        else:
            x = repr(x)
            if x[1] == '|':
                return x
            if any(_ in x for _ in ('\\ ' + literals)):
                return x
            if _re_number.match(x[1:-1]):
                return x
            return x[1:-1]

    _lex = None
    _repr_map = {
        slice: lambda x: '*',
        Ellipsis.__class__: lambda x: '**',
        int: repr,
        float: repr,
        str: repr,
        # Note: the function is unbound at this point; extract with __func__
        tuple: _safe_str_tuple.__func__,
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
            try:
                self._cids = tuple(self._parse_cuid_v2(component))
            except (OSError, IOError):
                self._cids = tuple(self._parse_cuid_v1(component))
        else:
            self._cids = tuple(self._generate_cuid(
                component, cuid_buffer=cuid_buffer, context=context))

    def __str__(self):
        "Return a 'nicely formatted' string representation of the CUID"
        a = ""
        for name, args in self._cids:
            a += '.' + self._safe_str(name)
            if args:
                a += '[' + ','.join(self._safe_str(x) for x in args) + ']'
        return a[1:]  # Strip off the leading '.'

    # str() is sufficiently safe / unique to be usable as repr()
    __repr__ = __str__

    def get_repr(self, version=2):
        if version == 1:
            _unknown = lambda x: '?'+str(x)
            a = ""
            for name, args in self._cids:
                a += '.' + name
                if len(args) == 0:
                    continue
                a += ':' + ','.join(
                    self._repr_v1_map.get(x.__class__, _unknown)(x)
                    for x in args)
            return a[1:]  # Strip off the leading '.'
        elif version == 2:
            return repr(self)
        else:
            raise ValueError("Invalid repr version '%s'; expected 1 or 2"
                             % (version,))

    def __getstate__(self):
        ans = {x:getattr(self, x) for x in ComponentUID.__slots__}
        if PY2:
            # Ellipsis is not picklable
            ans['_cids'] = tuple(
                (k, tuple(_PickleEllipsis if i is Ellipsis else i
                          for i in v)) for k,v in ans['_cids'])
        return ans

    def __setstate__(self, state):
        if PY2:
            # Ellipsis is not picklable
            state['_cids'] = tuple(
                (k, tuple(Ellipsis if i is _PickleEllipsis else i
                          for i in v)) for k,v in state['_cids'])
        for key, val in iteritems(state):
            setattr(self,key,val)

    def __hash__(self):
        """Return a deterministic hash for this ComponentUID"""
        try:
            return hash(self._cids)
        except TypeError:
            # Special handling for unhashable data (slices)
            return hash(tuple(
                (name, tuple(
                    (slice, x.start, x.stop, x.step)
                    if x.__class__ is slice else x
                    for x in idx)) for name, idx in self._cids))

    def __lt__(self, other):
        """Return True if this CUID <= the 'other' CUID

        This method defines a lexicographic sorting order for
        ComponentUID objects.  Nominally this is equivalent to sorting
        tuples or strings (elements are compared in order, with the
        first difference determining the ordering; longer tuples / lists
        are sorted after shorter ones).  This includes special handling
        for slice and ellipsis, where slice is sorted after any specific
        index, and ellipsis is sorted after everything else.

        Following Python 3 convention, this will raise a TypeError if
        `other` is not a ComponentUID.

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
                        # If both are slices, fall through to use '<' below
                        if self_i.__class__ is not slice:
                            return True
                    elif self_i.__class__ is slice:
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
        "Return True if this CUID <= the 'other' CUID"
        return self < other or self == other

    def __gt__(self, other):
        "Return True if this CUID > the 'other' CUID"
        return not (self <= other)

    def __ge__(self, other):
        "Return True if this CUID >= the 'other' CUID"
        return not (self < other)

    def __eq__(self, other):
        """Return True if this CUID is exactly equal to `other`

        This will return False (and not raise an exception) if `other`
        is not a ComponentUID.
        """
        try:
            other_cids = other._cids
        except AttributeError:
            return False
        return self._cids == other_cids

    def __ne__(self, other):
        """Return True if this CUID is not exactly equal to `other`

        This will return True (and not raise an exception) if `other`
        is not a ComponentUID.
        """
        return not self.__eq__(other)

    @staticmethod
    def generate_cuid_string_map(block, ctype=None, descend_into=True,
                                 repr_version=2):
        def _record_indexed_object_cuid_strings_v1(obj, cuid_str):
            _unknown = lambda x: '?'+str(x)
            for idx, data in iteritems(obj):
                if idx.__class__ is tuple and len(idx) > 1:
                    cuid_strings[data] = cuid_str + ':' + ','.join(
                        ComponentUID._repr_v1_map.get(x.__class__, _unknown)(x)
                        for x in idx)
                else:
                    cuid_strings[data] \
                        = cuid_str + ':' + ComponentUID._repr_v1_map.get(
                            idx.__class__, _unknown)(idx)

        def _record_indexed_object_cuid_strings_v2(obj, cuid_str):
            for idx, data in iteritems(obj):
                if idx.__class__ is tuple and len(idx) > 1:
                    cuid_strings[data] = cuid_str + '[' + ','.join(
                        ComponentUID._safe_str(x) for x in idx) + ']'
                else:
                    cuid_strings[data] \
                        = cuid_str + '[' + ComponentUID._safe_str(idx) + ']'

        _record_indexed_object_cuid_strings = {
            1: _record_indexed_object_cuid_strings_v1,
            2: _record_indexed_object_cuid_strings_v2,
        }[repr_version]
        _record_name = {
            1: str,
            2: ComponentUID._safe_str,
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
        "Return the list of (name, idx) pairs for the specified component"
        model = component.model()
        if context is None:
            context = model
        orig_component = component
        rcuid = []
        while component is not context:
            if component is model:
                raise ValueError("Context '%s' does not apply to component "
                                 "'%s'" % (context.name,
                                           orig_component.name))
            c = component.parent_component()
            if c is component:
                rcuid.append(( c.local_name, () ))
            elif cuid_buffer is not None:
                if id(component) not in cuid_buffer:
                    c_local_name = c.local_name
                    for idx, obj in iteritems(c):
                        if idx.__class__ is not tuple or len(idx) == 1:
                            idx = (idx,)
                        cuid_buffer[id(obj)] = (c_local_name, idx)
                rcuid.append(cuid_buffer[id(component)])
            else:
                idx = component.index()
                if idx.__class__ is not tuple or len(idx) == 1:
                    idx = (idx,)
                rcuid.append((c.local_name, idx))
            component = component.parent_block()
        rcuid.reverse()
        return rcuid

    def _parse_cuid_v2(self, label):
        """Parse a string (v2 repr format) and yield name, idx pairs

        This attempts to parse a string (nominally returned by
        get_repr()) to generate the sequence of (name, idx) pairs for
        the _cuids data structure.

        """
        if ComponentUID._lex is None:
            ComponentUID._lex = ply.lex.lex()

        name = None
        idx_stack = []
        idx = ()
        self._lex.input(label)
        while True:
            tok = self._lex.token()
            if not tok:
                break
            if tok.type == '.':
                assert not idx_stack
                yield (name, idx)
                name = None
                idx = ()
            elif tok.type == '[':
                idx_stack.append([])
            elif tok.type == ']':
                idx = tuple(idx_stack.pop())
                assert not idx_stack
            elif tok.type == '(':
                assert idx_stack
                idx_stack.append([])
            elif tok.type == ')':
                tmp = tuple(idx_stack.pop())
                idx_stack[-1].append(tmp)
            elif idx_stack: # processing a component index
                if tok.type == ',':
                    pass
                elif tok.type == 'STAR':
                    idx_stack[-1].append(tok.value)
                else:
                    assert tok.type in {'WORD','STRING','NUMBER','PICKLE'}
                    idx_stack[-1].append(tok.value)
            else:
                assert tok.type in {'WORD','STRING'}
                assert name is None
                name = tok.value
        assert not idx_stack
        yield (name, idx)
            
    def _parse_cuid_v1(self, label):
        """Parse a string (v1 repr format) and yield name, idx pairs

        This attempts to parse a string (nominally returned by
        get_repr()) to generate the sequence of (name, idx) pairs for
        the _cuids data structure.

        """
        cList = label.split('.')
        for c in cList:
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
                        idx[i] = _int_or_float(val[1:])
                    elif val[0] in  "\"'" and val[-1] == val[0]:
                        idx[i] = val[1:-1]
                    elif _re_number.match(val):
                        idx[i] = _int_or_float(val)
                if len(idx) == 1 and idx[0] == '**':
                    yield ( c_info[0], (Ellipsis,) )
                else:
                    yield ( c_info[0], tuple(idx) )

    def _resolve_cuid(self, block):
        obj = block
        for name, idx in self._cids:
            try:
                if not idx:
                    obj = getattr(obj, name)
                elif len(idx) == 1:
                    obj = getattr(obj, name)[idx[0]]
                else:
                    obj = getattr(obj, name)[idx]
            except KeyError:
                return None
            except AttributeError:
                return None
            except IndexError:
                return None
        return obj

    @deprecated("ComponentUID.find_component() is deprecated. "
                "Use ComponentUID.find_component_on()", version='TBD')
    def find_component(self, block):
        return self.find_component_on(block)

    def find_component_on(self, block):
        """
        Return the (unique) component in the block.  If the CUID contains
        a wildcard in the last component, then returns that component.  If
        there are wildcards elsewhere (or the last component was a partial
        slice), then returns a reference.  See also list_components below.
        """
        obj = self._resolve_cuid(block)
        if isinstance(obj, IndexedComponent_slice):
            # Suppress slice iteration exceptions
            obj.key_errors_generate_exceptions = False
            obj.attribute_errors_generate_exceptions = False
            obj = Reference(obj)
            try:
                next(iter(obj))
            except StopIteration:
                obj = None
        return obj

    def list_components(self, block):
        "Generator returning all components matching this ComponentUID"
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
        """Return True if this ComponentUID matches specified component

        This is equivalent to:

            `component in ComponentSet(self.list_components())`
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
                        if s_idx[_k].__class__ is slice:
                            continue
                        elif s_idx[_k] != idx[_k]:
                            return False
                    # Everything after the elipsis matched, so we can
                    # move on to the next level.
                    break
                if s_idx_val != idx[j]:
                    return False
        # Matched if all self._cids were consumed
        return i+1 == len(self._cids)


def _int_or_float(n):
    _num = float(n)
    try:
        _int = int(n)
    except:
        _int = 0  # a random int
    return _int if _num == _int else _num

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
    r'(?:[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|-?inf|nan)')

# Ignore whitespace (space, tab, and linefeed)
t_ignore = " \t\r"

literals = '()[],.'

tokens = [
    "WORD",   # unquoted string
    "STRING", # quoted string
    "NUMBER", # raw number
    "STAR",   # either * or **
    "PICKLE", # a pickled index object
]

# Numbers only appear in getitem lists, so they must be followed by a
# delimiter token (one of ' ,]')
@ply.lex.TOKEN(_re_number.pattern+r'(?=[\s\],])')
def t_NUMBER(t):
    t.value = _int_or_float(t.value)
    return t

@ply.lex.TOKEN(r'[a-zA-Z_][a-zA-Z_0-9]*')
def t_WORD(t):
    return t

_quoted_str = r"'(?:[^'\\]|\\.)*'"
_general_str = "|".join([_quoted_str, _quoted_str.replace("'",'"')])
@ply.lex.TOKEN(_general_str)
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

@ply.lex.TOKEN(r'\|b?(?:'+_general_str+")")
def t_PICKLE(t):
    start = 3 if t.value[1] == 'b' else 2
    unescaped = _re_escape_sequences.sub(_match_escape, t.value[start:-1])
    if PY2:
        rawstr = unescaped.encode('latin-1')
    else:
        rawstr = bytes(list(ord(_) for _ in unescaped))
    t.value = pickle.loads(rawstr)
    return t

# Error handling rule
def t_error(t):
    # Note this parser does not allow "\n", so lexpos is the column number
    raise IOError("ERROR: Token '%s' Line %s Column %s"
                  % (t.value, t.lineno, t.lexpos+1))
