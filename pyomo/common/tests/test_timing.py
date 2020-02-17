#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest

from six import StringIO

from pyomo.common.log import LoggingIntercept
from pyomo.common.timing import ConstructionTimer, report_timing
from pyomo.environ import ConcreteModel, RangeSet, Var

class TestTiming(unittest.TestCase):
    def test_raw_construction_timer(self):
        a = ConstructionTimer(None)
        self.assertIn(
            "ConstructionTimer object for NoneType (unknown); ",
            str(a))

    def test_report_timing(self):
        # Create a set to ensure that the global sets have already been
        # constructed (this is an issue until the new set system is
        # merged in and the GlobalSet objects are not automatically
        # created by pyomo.core
        m = ConcreteModel()
        m.x = Var([1,2])

        ref = """
           0 seconds to construct Block ConcreteModel; 1 index total
           0 seconds to construct RangeSet r; 1 index total
           0 seconds to construct Var x; 2 indicies total
""".strip()

        os = StringIO()
        try:
            report_timing(os)
            m = ConcreteModel()
            m.r = RangeSet(2)
            m.x = Var(m.r)
            self.assertEqual(os.getvalue().strip(), ref)
        finally:
            report_timing(False)
        buf = StringIO()
        with LoggingIntercept(buf, 'pyomo'):
            m = ConcreteModel()
            m.r = RangeSet(2)
            m.x = Var(m.r)
            self.assertEqual(os.getvalue().strip(), ref)
            self.assertEqual(buf.getvalue().strip(), "")

