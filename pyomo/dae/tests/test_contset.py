#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# Unit Tests for ContinuousSet() Objects
#

import os
from os.path import abspath, dirname

import pyutilib.th as unittest

from pyomo.environ import ConcreteModel, AbstractModel, Set
from pyomo.dae import ContinuousSet
from pyomo.common.log import LoggingIntercept
from six import StringIO

currdir = dirname(abspath(__file__)) + os.sep


class TestContinuousSet(unittest.TestCase):

    # test __init__
    def test_init(self):
        model = ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 1))

        model = ConcreteModel()
        model.t = ContinuousSet(initialize=[1, 2, 3])

        model = ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 5), initialize=[1, 3, 5])

        # Expected ValueError because a ContinuousSet component
        # must contain at least two values upon construction
        with self.assertRaises(ValueError):
            model.t = ContinuousSet()

    # test bad keyword arguments
    def test_bad_kwds(self):
        model = ConcreteModel()
        with self.assertRaises(TypeError):
            model.t = ContinuousSet(bounds=(0, 1), filter=True)

        with self.assertRaises(TypeError):
            model.t = ContinuousSet(bounds=(0, 1), dimen=2)

        with self.assertRaises(TypeError):
            model.t = ContinuousSet(bounds=(0, 1), virtual=True)

        with self.assertRaises(TypeError):
            model.t = ContinuousSet(bounds=(0, 1), validate=True)

    # test valid declarations
    def test_valid_declaration(self):
        model = ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 1))
        self.assertEqual(len(model.t), 2)
        self.assertIn(0, model.t)
        self.assertIn(1, model.t)

        model = ConcreteModel()
        model.t = ContinuousSet(initialize=[1, 2, 3])
        self.assertEqual(len(model.t), 3)
        self.assertEqual(model.t.first(), 1)
        self.assertEqual(model.t.last(), 3)

        model = ConcreteModel()
        model.t = ContinuousSet(bounds=(1, 3), initialize=[1, 2, 3])
        self.assertEqual(len(model.t), 3)
        self.assertEqual(model.t.first(), 1)
        self.assertEqual(model.t.last(), 3)

        model = ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 4), initialize=[1, 2, 3])
        self.assertEqual(len(model.t), 5)
        self.assertEqual(model.t.first(), 0)
        self.assertEqual(model.t.last(), 4)

        model = ConcreteModel()
        with self.assertRaisesRegexp(
                ValueError, "value is not in the domain \[0..4\]"):
            model.t = ContinuousSet(bounds=(0, 4), initialize=[1, 2, 3, 5])
        # self.assertEqual(len(model.t), 5)
        # self.assertEqual(model.t.first(), 0)
        # self.assertEqual(model.t.last(), 5)
        # self.assertNotIn(4, model.t)
        # del model.t

        model = ConcreteModel()
        with self.assertRaisesRegexp(
                ValueError, "value is not in the domain \[2..6\]"):
            model.t = ContinuousSet(bounds=(2, 6), initialize=[1, 2, 3, 5])
        # self.assertEqual(len(model.t), 5)
        # self.assertEqual(model.t.first(), 1)
        # self.assertEqual(model.t.last(), 6)
        # del model.t

        model = ConcreteModel()
        with self.assertRaisesRegexp(
                ValueError, "value is not in the domain \[2..4\]"):
            model.t = ContinuousSet(bounds=(2, 4), initialize=[1, 3, 5])
        # self.assertEqual(len(model.t), 3)
        # self.assertNotIn(2, model.t)
        # self.assertNotIn(4, model.t)

    # test invalid declarations
    def test_invalid_declaration(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1, 2, 3])
        with self.assertRaises(TypeError):
            model.t = ContinuousSet(model.s, bounds=(0, 1))

        model = ConcreteModel()
        with self.assertRaises(ValueError):
            model.t = ContinuousSet(bounds=(0, 0))

        model = ConcreteModel()
        with self.assertRaises(ValueError):
            model.t = ContinuousSet(initialize=[1])

        model = ConcreteModel()
        with self.assertRaises(ValueError):
            model.t = ContinuousSet(bounds=(None, 1))

        model = ConcreteModel()
        with self.assertRaises(ValueError):
            model.t = ContinuousSet(bounds=(0, None))

        model = ConcreteModel()
        with self.assertRaises(ValueError):
            model.t = ContinuousSet(initialize=[(1, 2), (3, 4)])

        model = ConcreteModel()
        with self.assertRaises(ValueError):
            model.t = ContinuousSet(initialize=['foo', 'bar'])

    # test the get_changed method
    def test_get_changed(self):
        model = ConcreteModel()
        model.t = ContinuousSet(initialize=[1, 2, 3])
        self.assertFalse(model.t.get_changed())
        self.assertEqual(model.t._changed, model.t.get_changed())

    # test the set_changed method
    def test_set_changed(self):
        model = ConcreteModel()
        model.t = ContinuousSet(initialize=[1, 2, 3])
        self.assertFalse(model.t._changed)
        model.t.set_changed(True)
        self.assertTrue(model.t._changed)
        model.t.set_changed(False)
        self.assertFalse(model.t._changed)

        with self.assertRaises(ValueError):
            model.t.set_changed(3)

    # test get_upper_element_boundary
    def test_get_upper_element_boundary(self):
        m = ConcreteModel()
        m.t = ContinuousSet(initialize=[1, 2, 3])
        self.assertEqual(m.t.get_upper_element_boundary(1.5), 2)
        self.assertEqual(m.t.get_upper_element_boundary(2.5), 3)
        self.assertEqual(m.t.get_upper_element_boundary(2), 2)

        log_out = StringIO()
        with LoggingIntercept(log_out, 'pyomo.dae'):
            temp = m.t.get_upper_element_boundary(3.5)
        self.assertIn('Returning the upper bound', log_out.getvalue())

    # test get_lower_element_boundary
    def test_get_lower_element_boundary(self):
        m = ConcreteModel()
        m.t = ContinuousSet(initialize=[1, 2, 3])
        self.assertEqual(m.t.get_lower_element_boundary(1.5), 1)
        self.assertEqual(m.t.get_lower_element_boundary(2.5), 2)
        self.assertEqual(m.t.get_lower_element_boundary(2), 2)

        log_out = StringIO()
        with LoggingIntercept(log_out, 'pyomo.dae'):
            temp = m.t.get_lower_element_boundary(0.5)
        self.assertIn('Returning the lower bound', log_out.getvalue())

    def test_duplicate_construct(self):
        m = ConcreteModel()
        m.t = ContinuousSet(initialize=[1,2,3])
        self.assertEqual(m.t, [1,2,3])
        self.assertEqual(m.t._fe, [1,2,3])
        m.t.add(1.5)
        m.t.add(2.5)
        self.assertEqual(m.t, [1,1.5,2,2.5,3])
        self.assertEqual(m.t._fe, [1,2,3])
        m.t.construct()
        self.assertEqual(m.t, [1,1.5,2,2.5,3])
        self.assertEqual(m.t._fe, [1,2,3])

    def test_find_nearest_index(self):
        m = ConcreteModel()
        m.time = ContinuousSet(bounds=(0, 5))

        # Undiscretized
        i = m.time.find_nearest_index(1)
        self.assertEqual(i, 1)
        i = m.time.find_nearest_index(1, tolerance=0.5)
        self.assertEqual(i, None)

        i = m.time.find_nearest_index(-0.01, tolerance=0.1)
        self.assertEqual(i, 1)
        i = m.time.find_nearest_index(-0.01, tolerance=0.001)
        self.assertEqual(i, None)

        i = m.time.find_nearest_index(6, tolerance=2)
        self.assertEqual(i, 2)
        i = m.time.find_nearest_index(6, tolerance=1)
        self.assertEqual(i, 2)

        # This test relies on the behavior for tiebreaks
        i = m.time.find_nearest_index(2.5)
        self.assertEqual(i, 1)

        m.del_component(m.time)
        init_list = []
        for i in range(5):
            i0 = float(i)
            i1 = round((i+0.15)*1e4)/1e4
            i2 = round((i+0.64)*1e4)/1e4
            # Round to get rid of numerical error due to float addition
            init_list.extend([i, i1, i2])
        init_list.append(5.0)
        m.time = ContinuousSet(initialize=init_list)

        # "Discretized"
        i = m.time.find_nearest_index(1.01, tolerance=0.1)
        self.assertEqual(i, 4)
        i = m.time.find_nearest_index(1.01, tolerance=0.001)
        self.assertEqual(i, None)

        i = m.time.find_nearest_index(3.5)
        self.assertEqual(i, 12)
        i = m.time.find_nearest_index(3.5, tolerance=0.1)
        self.assertEqual(i, None)

        i = m.time.find_nearest_index(-1)
        self.assertEqual(i, 1)
        i = m.time.find_nearest_index(-1, tolerance=1)
        self.assertEqual(i, 1)

        i = m.time.find_nearest_index(5.5)
        self.assertEqual(i, 16)
        i = m.time.find_nearest_index(5.5, tolerance=0.49)
        self.assertEqual(i, None)

        i = m.time.find_nearest_index(2.64, tolerance=1e-8)
        self.assertEqual(i, 9)
        i = m.time.find_nearest_index(2.64, tolerance=0)
        self.assertEqual(i, 9)

        i = m.time.find_nearest_index(5, tolerance=0)
        self.assertEqual(i, 16)

        i = m.time.find_nearest_index(0, tolerance=0)
        self.assertEqual(i, 1)

        # This test fails. I get:
        # delta_left == 2.075-2.0 == 0.07500000000000018
        # delta_right == 2.15-2.075 == 0.07499999999999973
        # Index 8 is returned (time[8] == 2.15), even though
        # "tie"-break logic suggests index 7 should be.
        #i = m.time.find_nearest_index(2.075)
        #self.assertEqual(i, 7)


class TestIO(unittest.TestCase):

    def setUp(self):
        #
        # Create Model
        #
        self.model = AbstractModel()
        self.instance = None

    def tearDown(self):
        if os.path.exists("diffset.dat"):
            os.remove("diffset.dat")
        self.model = None
        self.instance = None

    def test_io1(self):
        OUTPUT = open("diffset.dat", "w")
        OUTPUT.write("data;\n")
        OUTPUT.write("set A := 1 3 5 7;\n")
        OUTPUT.write("end;\n")
        OUTPUT.close()
        self.model.A = ContinuousSet()
        self.instance = self.model.create_instance("diffset.dat")
        self.assertEqual(len(self.instance.A), 4)

    def test_io2(self):
        OUTPUT = open("diffset.dat", "w")
        OUTPUT.write("data;\n")
        OUTPUT.write("set A := 1 3 5;\n")
        OUTPUT.write("end;\n")
        OUTPUT.close()
        self.model.A = ContinuousSet(bounds=(0, 4))
        with self.assertRaisesRegexp(
                ValueError, "The value is not in the domain \[0..4\]"):
            self.instance = self.model.create_instance("diffset.dat")
        #self.assertEqual(len(self.instance.A), 4)

    def test_io3(self):
        OUTPUT = open("diffset.dat", "w")
        OUTPUT.write("data;\n")
        OUTPUT.write("set A := 1 3 5;\n")
        OUTPUT.write("end;\n")
        OUTPUT.close()
        self.model.A = ContinuousSet(bounds=(2, 6))
        with self.assertRaisesRegexp(
                ValueError, "The value is not in the domain \[2..6\]"):
            self.instance = self.model.create_instance("diffset.dat")
        #self.assertEqual(len(self.instance.A), 4)

    def test_io4(self):
        OUTPUT = open("diffset.dat", "w")
        OUTPUT.write("data;\n")
        OUTPUT.write("set A := 1 3 5;\n")
        OUTPUT.write("end;\n")
        OUTPUT.close()
        self.model.A = ContinuousSet(bounds=(2, 4))
        with self.assertRaisesRegexp(
                ValueError, "The value is not in the domain \[2..4\]"):
            self.instance = self.model.create_instance("diffset.dat")
        #self.assertEqual(len(self.instance.A), 3)

    def test_io5(self):
        OUTPUT = open("diffset.dat", "w")
        OUTPUT.write("data;\n")
        OUTPUT.write("set A := 1 3 5;\n")
        OUTPUT.write("end;\n")
        OUTPUT.close()
        self.model.A = ContinuousSet(bounds=(0, 6))
        self.instance = self.model.create_instance("diffset.dat")
        self.assertEqual(len(self.instance.A), 5)

    def test_io6(self):
        OUTPUT = open("diffset.dat", "w")
        OUTPUT.write("data;\n")
        OUTPUT.write("set B := 1;\n")
        OUTPUT.write("end;\n")
        OUTPUT.close()

        # Expected ValueError because data set has only one value and no
        # bounds are specified
        self.model.B = ContinuousSet()
        with self.assertRaises(ValueError):
            self.instance = self.model.create_instance("diffset.dat")

    def test_io7(self):
        OUTPUT = open("diffset.dat", "w")
        OUTPUT.write("data;\n")
        OUTPUT.write("set B := 1;\n")
        OUTPUT.write("end;\n")
        OUTPUT.close()
        self.model.B = ContinuousSet(bounds=(0, 1))
        self.instance = self.model.create_instance("diffset.dat")
        self.assertEqual(len(self.instance.B), 2)

if __name__ == "__main__":
    unittest.main()
