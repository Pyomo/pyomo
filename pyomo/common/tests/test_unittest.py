#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import datetime

import pyomo.common.unittest as unittest

class TestPyomoUnittest(unittest.TestCase):
    def test_assertRelativeEqual_comparison(self):
        a = 1
        b = 1
        self.assertRelativeEqual(a, b)
        # default relative tolerance is 1e-7.  This should have a delta
        # of "exactly" 1e-7, but due to roundoff error, it is not.  The
        # choice of 9.999e-8 and 9.999-e7 is specifically so that
        # roundoff error doesn't cause tests to fail
        b -= 9.999e-8
        self.assertRelativeEqual(a, b)
        b -= 9.999e-8
        with self.assertRaisesRegex(self.failureException, '1 !~= 0.9999'):
            self.assertRelativeEqual(a, b)

        b = 1
        self.assertRelativeEqual(a, b, delta=1e-6)
        b -= 9.999e-7
        self.assertRelativeEqual(a, b, delta=1e-6)
        b -= 9.999e-7
        with self.assertRaisesRegex(self.failureException, '1 !~= 0.999'):
            self.assertRelativeEqual(a, b, delta=1e-6)

        b = 1
        self.assertRelativeEqual(a, b, places=6)
        b -= 9.999e-7
        self.assertRelativeEqual(a, b, places=6)
        b -= 9.999e-7
        with self.assertRaisesRegex(self.failureException, '1 !~= 0.999'):
            self.assertRelativeEqual(a, b, places=6)
        
    def test_assertRelativeEqual_errorChecking(self):
        with self.assertRaisesRegex(ValueError, 
                                    "Cannot specify both places and delta"):
            self.assertRelativeEqual(1,1,places=7,delta=1)
        
    def test_assertRelativeEqual_str(self):
        self.assertRelativeEqual("hi", "hi")
        with self.assertRaisesRegex(self.failureException, "'hi' !~= 'hello'"):
            self.assertRelativeEqual("hi", "hello")
        with self.assertRaisesRegex(self.failureException, "'hi' !~= \['h',"):
            self.assertRelativeEqual("hi", ['h','i'])

    def test_assertRelativeEqual_othertype(self):
        a = datetime.datetime(1,1,1)
        b = datetime.datetime(1,1,1)
        self.assertRelativeEqual(a, b)
        b = datetime.datetime(1,1,2)
        with self.assertRaisesRegex(self.failureException,
                                    "datetime.* !~= datetime"):
            self.assertRelativeEqual(a, b)
        
    def test_assertRelativeEqual_list(self):
        a = [1,2]
        b = [1,2,3]
        with self.assertRaisesRegex(self.failureException,
                                    'sequences are different sizes \(2 != 3\)'):
            self.assertRelativeEqual(a, b)
        self.assertRelativeEqual(a, b, allow_second_superset=True)
        a.append(3)
        

        self.assertRelativeEqual(a, b)
        b[1] -= 1.999e-7
        self.assertRelativeEqual(a, b)
        b[1] -= 1.999e-7
        with self.assertRaisesRegex(self.failureException, '2 !~= 1.999'):
            self.assertRelativeEqual(a, b)

    def test_assertRelativeEqual_dict(self):
        a = {1:2, 3:4}
        b = {1:2, 3:4, 5:6}
        with self.assertRaisesRegex(self.failureException,
                                    'mappings are different sizes \(2 != 3\)'):
            self.assertRelativeEqual(a, b)
        self.assertRelativeEqual(a, b, allow_second_superset=True)
        a[5] = 6
        

        self.assertRelativeEqual(a, b)
        b[1] -= 1.999e-7
        self.assertRelativeEqual(a, b)
        b[1] -= 1.999e-7
        with self.assertRaisesRegex(self.failureException, '2 !~= 1.999'):
            self.assertRelativeEqual(a, b)

        del b[1]
        b[6] = 6
        with self.assertRaisesRegex(self.failureException,
                                    'key \(1\) from first not found in second'):
            self.assertRelativeEqual(a, b)

    def test_assertRelativeEqual_nested(self):
        a = {1.1: [1,2,3], 'a': 'hi', 3: {1:2, 3:4}}
        b = {1.1: [1,2,3], 'a': 'hi', 3: {1:2, 3:4}}
        self.assertRelativeEqual(a, b)
        b[1.1][2] -= 1.999e-7
        b[3][1] -= 9.999e-8
        self.assertRelativeEqual(a, b)
        b[1.1][2] -= 1.999e-7
        with self.assertRaisesRegex(self.failureException,
                                    '3 !~= 2.999'):
            self.assertRelativeEqual(a, b)
