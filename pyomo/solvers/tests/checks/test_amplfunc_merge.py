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

import pyomo.common.unittest as unittest
from pyomo.solvers.amplfunc_merge import unique_paths, amplfunc_merge


class TestAMPLFUNCStringMerge(unittest.TestCase):
    def test_merge_no_dup(self):
        s1 = "my/place/l1.so\nanother/place/l1.so"
        s2 = "my/place/l2.so"
        sm = unique_paths(s1, s2)
        sm_list = sm.split("\n")
        self.assertEqual(len(sm_list), 3)
        # The order of lines should be maintained with the second string
        # following the first
        self.assertEqual(sm_list[0], "my/place/l1.so")
        self.assertEqual(sm_list[1], "another/place/l1.so")
        self.assertEqual(sm_list[2], "my/place/l2.so")

    def test_merge_empty1(self):
        s1 = ""
        s2 = "my/place/l2.so"
        sm = unique_paths(s1, s2)
        sm_list = sm.split("\n")
        self.assertEqual(len(sm_list), 1)
        self.assertEqual(sm_list[0], "my/place/l2.so")

    def test_merge_empty2(self):
        s1 = "my/place/l2.so"
        s2 = ""
        sm = unique_paths(s1, s2)
        sm_list = sm.split("\n")
        self.assertEqual(len(sm_list), 1)
        self.assertEqual(sm_list[0], "my/place/l2.so")

    def test_merge_empty_both(self):
        s1 = ""
        s2 = ""
        sm = unique_paths(s1, s2)
        sm_list = sm.split("\n")
        self.assertEqual(len(sm_list), 1)
        self.assertEqual(sm_list[0], "")

    def test_merge_bad_type(self):
        self.assertRaises(AttributeError, unique_paths, "", 3)
        self.assertRaises(AttributeError, unique_paths, 3, "")
        self.assertRaises(AttributeError, unique_paths, 3, 3)
        self.assertRaises(AttributeError, unique_paths, None, "")
        self.assertRaises(AttributeError, unique_paths, "", None)
        self.assertRaises(AttributeError, unique_paths, 2.3, "")
        self.assertRaises(AttributeError, unique_paths, "", 2.3)

    def test_merge_duplicate1(self):
        s1 = "my/place/l1.so\nanother/place/l1.so"
        s2 = "my/place/l1.so\nanother/place/l1.so"
        sm = unique_paths(s1, s2)
        sm_list = sm.split("\n")
        self.assertEqual(len(sm_list), 2)
        # The order of lines should be maintained with the second string
        # following the first
        self.assertEqual(sm_list[0], "my/place/l1.so")
        self.assertEqual(sm_list[1], "another/place/l1.so")

    def test_merge_duplicate2(self):
        s1 = "my/place/l1.so\nanother/place/l1.so"
        s2 = "my/place/l1.so"
        sm = unique_paths(s1, s2)
        sm_list = sm.split("\n")
        self.assertEqual(len(sm_list), 2)
        # The order of lines should be maintained with the second string
        # following the first
        self.assertEqual(sm_list[0], "my/place/l1.so")
        self.assertEqual(sm_list[1], "another/place/l1.so")

    def test_merge_extra_linebreaks(self):
        s1 = "\nmy/place/l1.so\nanother/place/l1.so\n"
        s2 = "\nmy/place/l1.so\n\n"
        sm = unique_paths(s1, s2)
        sm_list = sm.split("\n")
        self.assertEqual(len(sm_list), 2)
        # The order of lines should be maintained with the second string
        # following the first
        self.assertEqual(sm_list[0], "my/place/l1.so")
        self.assertEqual(sm_list[1], "another/place/l1.so")


class TestAMPLFUNCMerge(unittest.TestCase):
    def test_merge_no_dup(self):
        env = {
            "AMPLFUNC": "my/place/l1.so\nanother/place/l1.so",
            "PYOMO_AMPLFUNC": "my/place/l2.so",
        }
        sm = amplfunc_merge(env)
        sm_list = sm.split("\n")
        self.assertEqual(len(sm_list), 3)
        self.assertEqual(sm_list[0], "my/place/l1.so")
        self.assertEqual(sm_list[1], "another/place/l1.so")
        self.assertEqual(sm_list[2], "my/place/l2.so")

    def test_merge_empty1(self):
        env = {"AMPLFUNC": "", "PYOMO_AMPLFUNC": "my/place/l2.so"}
        sm = amplfunc_merge(env)
        sm_list = sm.split("\n")
        self.assertEqual(len(sm_list), 1)
        self.assertEqual(sm_list[0], "my/place/l2.so")

    def test_merge_empty2(self):
        env = {"AMPLFUNC": "my/place/l2.so", "PYOMO_AMPLFUNC": ""}
        sm = amplfunc_merge(env)
        sm_list = sm.split("\n")
        self.assertEqual(len(sm_list), 1)
        self.assertEqual(sm_list[0], "my/place/l2.so")

    def test_merge_empty_both(self):
        env = {"AMPLFUNC": "", "PYOMO_AMPLFUNC": ""}
        sm = amplfunc_merge(env)
        sm_list = sm.split("\n")
        self.assertEqual(len(sm_list), 1)
        self.assertEqual(sm_list[0], "")

    def test_merge_duplicate1(self):
        env = {
            "AMPLFUNC": "my/place/l1.so\nanother/place/l1.so",
            "PYOMO_AMPLFUNC": "my/place/l1.so\nanother/place/l1.so",
        }
        sm = amplfunc_merge(env)
        sm_list = sm.split("\n")
        self.assertEqual(len(sm_list), 2)
        self.assertEqual(sm_list[0], "my/place/l1.so")
        self.assertEqual(sm_list[1], "another/place/l1.so")

    def test_merge_no_pyomo(self):
        env = {"AMPLFUNC": "my/place/l1.so\nanother/place/l1.so"}
        sm = amplfunc_merge(env)
        sm_list = sm.split("\n")
        self.assertEqual(len(sm_list), 2)
        self.assertEqual(sm_list[0], "my/place/l1.so")
        self.assertEqual(sm_list[1], "another/place/l1.so")

    def test_merge_no_user(self):
        env = {"PYOMO_AMPLFUNC": "my/place/l1.so\nanother/place/l1.so"}
        sm = amplfunc_merge(env)
        sm_list = sm.split("\n")
        self.assertEqual(len(sm_list), 2)
        self.assertEqual(sm_list[0], "my/place/l1.so")
        self.assertEqual(sm_list[1], "another/place/l1.so")

    def test_merge_nothing(self):
        env = {}
        sm = amplfunc_merge(env)
        sm_list = sm.split("\n")
        self.assertEqual(len(sm_list), 1)
        self.assertEqual(sm_list[0], "")
