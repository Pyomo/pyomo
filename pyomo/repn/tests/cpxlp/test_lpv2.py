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

from io import StringIO

import pyomo.common.unittest as unittest

from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Block, Constraint, Var, Objective, Suffix

from pyomo.repn.plugins.lp_writer import LPWriter

class TestLPv2(unittest.TestCase):
    def test_warn_export_suffixes(self):
        m = ConcreteModel()
        m.x = Var()
        m.obj = Objective(expr=m.x)
        m.con = Constraint(expr=m.x>=2)
        m.b = Block()
        m.ignored = Suffix(direction=Suffix.IMPORT)
        m.duals = Suffix(direction=Suffix.IMPORT_EXPORT)
        m.b.duals = Suffix(direction=Suffix.IMPORT_EXPORT)
        m.b.scaling = Suffix(direction=Suffix.EXPORT)
        
        # Empty suffixes are ignored
        writer = LPWriter()
        with LoggingIntercept() as LOG:
            writer.write(m, StringIO())
        self.assertEqual(LOG.getvalue(), "")

        # Import are ignored, export and import/export are warned
        m.duals[m.con] = 5
        m.ignored[m.x] = 6
        m.b.scaling[m.x] = 7

        writer = LPWriter()
        with LoggingIntercept() as LOG:
            writer.write(m, StringIO())
        self.assertEqual(LOG.getvalue(), """EXPORT Suffix 'duals' found on 1 block:
    duals
LP writer cannot export suffixes to LP files.  Skipping.
EXPORT Suffix 'scaling' found on 1 block:
    b.scaling
LP writer cannot export suffixes to LP files.  Skipping.
""")

        # Counting works correctly
        m.b.duals[m.x] = 7

        writer = LPWriter()
        with LoggingIntercept() as LOG:
            writer.write(m, StringIO())
        self.assertEqual(LOG.getvalue(), """EXPORT Suffix 'duals' found on 2 blocks:
    duals
    b.duals
LP writer cannot export suffixes to LP files.  Skipping.
EXPORT Suffix 'scaling' found on 1 block:
    b.scaling
LP writer cannot export suffixes to LP files.  Skipping.
""")
