# -*- coding: utf-8 -*-
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
#
# Unit Tests for pyomo.base.misc
#
from io import StringIO

import pyomo.common.unittest as unittest

from pyomo.common.sorting import sorted_robust, _robust_sort_keyfcn


# The following are custom types used for testing sorted_robust.  They
# are declared at the module scope to ensure consistent generation of
# the class __name__.
class LikeFloat(object):
    def __init__(self, n):
        self.n = n

    def __lt__(self, other):
        return self.n < other

    def __gt__(self, other):
        return self.n > other


class Comparable(object):
    def __init__(self, n):
        self.n = str(n)

    def __lt__(self, other):
        return self.n < other

    def __gt__(self, other):
        return self.n > other


class ToStr(object):
    def __init__(self, n):
        self.n = str(n)

    def __str__(self):
        return self.n


class NoStr(object):
    def __init__(self, n):
        self.n = str(n)

    def __str__(self):
        raise ValueError('')


class TestSortedRobust(unittest.TestCase):
    def test_sorted_robust(self):
        # Note: as types are sorted by name, int < str < tuple
        a = sorted_robust([3, 2, 1])
        self.assertEqual(a, [1, 2, 3])

        # Testthat ints and floats are sorted as "numbers"
        a = sorted_robust([3, 2.1, 1])
        self.assertEqual(a, [1, 2.1, 3])

        a = sorted_robust([3, '2', 1])
        self.assertEqual(a, [1, 3, '2'])

        a = sorted_robust([('str1', 'str1'), (1, 'str2')])
        self.assertEqual(a, [(1, 'str2'), ('str1', 'str1')])

        a = sorted_robust([((1,), 'str2'), ('str1', 'str1')])
        self.assertEqual(a, [('str1', 'str1'), ((1,), 'str2')])

        a = sorted_robust([('str1', 'str1'), ((1,), 'str2')])
        self.assertEqual(a, [('str1', 'str1'), ((1,), 'str2')])

    def test_user_key(self):
        # ensure it doesn't throw an error
        # Test for issue https://github.com/Pyomo/pyomo/issues/2019
        sorted_robust([(("10_1", 2), None), ((10, 2), None)], key=lambda x: x[0])

    def test_unknown_types(self):
        orig = [
            LikeFloat(4),  # 0
            Comparable('hello'),  # 1
            LikeFloat(1),  # 2
            2.0,  # 3
            Comparable('world'),  # 4
            ToStr(1),  # 5
            NoStr('bogus'),  # 6
            ToStr('a'),  # 7
            ToStr('A'),  # 8
            3,  # 9
        ]

        ref = [orig[i] for i in (1, 4, 6, 5, 8, 7, 2, 3, 9, 0)]
        ans = sorted_robust(orig)
        self.assertEqual(len(orig), len(ans))
        for _r, _a in zip(ref, ans):
            self.assertIs(_r, _a)
        self.assertEqual(_robust_sort_keyfcn._typemap[LikeFloat], (1, float.__name__))
        self.assertEqual(
            _robust_sort_keyfcn._typemap[Comparable], (1, Comparable.__name__)
        )
        self.assertEqual(_robust_sort_keyfcn._typemap[ToStr], (2, ToStr.__name__))
        self.assertEqual(_robust_sort_keyfcn._typemap[NoStr], (3, NoStr.__name__))


if __name__ == "__main__":
    unittest.main()
