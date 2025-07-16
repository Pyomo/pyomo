# -*- coding: utf-8 -*-
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
from collections import namedtuple
from io import StringIO

import pyomo.common.unittest as unittest

from pyomo.common.formatting import tostr, tabular_writer, StreamIndenter


class DerivedList(list):
    pass


class DerivedTuple(tuple):
    pass


class DerivedDict(dict):
    pass


class DerivedStr(str):
    pass


NamedTuple = namedtuple('NamedTuple', ['x', 'y'])


class TestToStr(unittest.TestCase):
    def test_new_type_float(self):
        self.assertEqual(tostr(0.5), '0.5')
        self.assertIs(tostr.handlers[float], tostr.handlers[None])

    def test_new_type_int(self):
        self.assertEqual(tostr(0), '0')
        self.assertIs(tostr.handlers[int], tostr.handlers[None])

    def test_new_type_str(self):
        self.assertEqual(tostr(DerivedStr(1)), '1')
        self.assertIs(tostr.handlers[DerivedStr], tostr.handlers[str])

    def test_new_type_list(self):
        self.assertEqual(tostr(DerivedList([1, 2])), '[1, 2]')
        self.assertIs(tostr.handlers[DerivedList], tostr.handlers[list])

    def test_new_type_dict(self):
        self.assertEqual(tostr(DerivedDict({1: 2})), '{1: 2}')
        self.assertIs(tostr.handlers[DerivedDict], tostr.handlers[dict])

    def test_new_type_tuple(self):
        self.assertEqual(tostr(DerivedTuple([1, 2])), '(1, 2)')
        self.assertIs(tostr.handlers[DerivedTuple], tostr.handlers[tuple])

    def test_new_type_namedtuple(self):
        self.assertEqual(tostr(NamedTuple(1, 2)), 'NamedTuple(x=1, y=2)')
        self.assertIs(tostr.handlers[NamedTuple], tostr.handlers[None])


class TestTabularWriter(unittest.TestCase):
    def test_unicode_table(self):
        # Test that an embedded unicode character does not foul up the
        # table alignment
        os = StringIO()
        data = {1: ("a", 1), (2, 3): ("∧", 2)}
        tabular_writer(os, "", data.items(), ["s", "val"], lambda k, v: v)
        ref = u"""
Key    : s : val
     1 : a :   1
(2, 3) : ∧ :   2
"""
        self.assertEqual(ref.strip(), os.getvalue().strip())

    def test_tuple_list_dict(self):
        os = StringIO()
        data = {(1,): (["a", 1], 1), ('2', 3): ({1: 'a', 2: '2'}, '2')}
        tabular_writer(os, "", data.items(), ["s", "val"], lambda k, v: v)
        ref = u"""
Key      : s                : val
    (1,) :         ['a', 1] :   1
('2', 3) : {1: 'a', 2: '2'} :   2
"""
        self.assertEqual(ref.strip(), os.getvalue().strip())

    def test_no_header(self):
        os = StringIO()
        data = {(2,): (["a", 1], 1), (1, 3): ({1: 'a', 2: '2'}, '2')}
        tabular_writer(os, "", data.items(), [], lambda k, v: v)
        ref = u"""
{1: 'a', 2: '2'} : 2
        ['a', 1] : 1
"""
        self.assertEqual(ref.strip(), os.getvalue().strip())

    def test_no_data(self):
        os = StringIO()
        data = {}
        tabular_writer(os, "", data.items(), ['s', 'val'], lambda k, v: v)
        ref = u"""
Key : s : val
"""
        self.assertEqual(ref.strip(), os.getvalue().strip())

    def test_multiline_generator(self):
        os = StringIO()
        data = {'a': 0, 'b': 1, 'c': 3}

        def _data_gen(i, j):
            for n in range(j):
                yield (n, chr(ord('a') + n) * j)

        tabular_writer(os, "", data.items(), ['i', 'j'], _data_gen)
        ref = u"""
Key : i    : j
  a : None : None
  b :    0 :    a
  c :    0 :  aaa
    :    1 :  bbb
    :    2 :  ccc
"""
        self.assertEqual(ref.strip(), os.getvalue().strip())

    def test_multiline_generator_exception(self):
        os = StringIO()
        data = {'a': 0, 'b': 1, 'c': 3}

        def _data_gen(i, j):
            if i == 'b':
                raise ValueError("invalid")
            for n in range(j):
                yield (n, chr(ord('a') + n) * j)

        tabular_writer(os, "", data.items(), ['i', 'j'], _data_gen)
        ref = u"""
Key : i    : j
  a : None : None
  b : None : None
  c :    0 :  aaa
    :    1 :  bbb
    :    2 :  ccc
"""
        self.assertEqual(ref.strip(), os.getvalue().strip())

    def test_data_exception(self):
        os = StringIO()
        data = {'a': 0, 'b': 1, 'c': 3}

        def _data_gen(i, j):
            if i == 'b':
                raise ValueError("invalid")
            return (j, i * (j + 1))

        tabular_writer(os, "", data.items(), ['i', 'j'], _data_gen)
        ref = u"""
Key : i    : j
  a :    0 :    a
  b : None : None
  c :    3 : cccc
"""
        self.assertEqual(ref.strip(), os.getvalue().strip())

    def test_multiline_alignment(self):
        os = StringIO()
        data = {'a': 1, 'b': 2, 'c': 3}

        def _data_gen(i, j):
            for n in range(j):
                _str = chr(ord('a') + n) * (j + 1)
                if n % 2:
                    _str = list(_str)
                    _str[1] = ' '
                    _str = ''.join(_str)
                yield (n, _str)

        tabular_writer(os, "", data.items(), ['i', 'j'], _data_gen)
        ref = u"""
Key : i : j
  a : 0 : aa
  b : 0 : aaa
    : 1 : b b
  c : 0 : aaaa
    : 1 : b bb
    : 2 : cccc
"""
        self.assertEqual(ref.strip(), os.getvalue().strip())


class TestStreamIndenter(unittest.TestCase):
    def test_noprefix(self):
        OUT1 = StringIO()
        OUT2 = StreamIndenter(OUT1)
        OUT2.write('Hello?\nHello, world!')
        self.assertEqual('    Hello?\n    Hello, world!', OUT2.getvalue())

    def test_prefix(self):
        prefix = 'foo:'
        OUT1 = StringIO()
        OUT2 = StreamIndenter(OUT1, prefix)
        OUT2.write('Hello?\nHello, world!')
        self.assertEqual('foo:Hello?\nfoo:Hello, world!', OUT2.getvalue())

    def test_blank_lines(self):
        OUT1 = StringIO()
        OUT2 = StreamIndenter(OUT1)
        OUT2.write('Hello?\n\nText\n\nHello, world!')
        self.assertEqual('    Hello?\n\n    Text\n\n    Hello, world!', OUT2.getvalue())

    def test_writelines(self):
        OUT1 = StringIO()
        OUT2 = StreamIndenter(OUT1)
        OUT2.writelines(['Hello?\n', '\n', 'Text\n', '\n', 'Hello, world!'])
        self.assertEqual('    Hello?\n\n    Text\n\n    Hello, world!', OUT2.getvalue())
