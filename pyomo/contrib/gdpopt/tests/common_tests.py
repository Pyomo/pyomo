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

from math import fabs
from pyomo.environ import value


def check_8PP_solution(self, eight_process, results):
    self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1e-2)
    self.assertTrue(fabs(value(results.problem.upper_bound) - 68) <= 1e-2)

    # Check discrete solution
    # use 2
    self.assertTrue(value(eight_process.use_unit_1or2.disjuncts[1].indicator_var))
    self.assertFalse(value(eight_process.use_unit_1or2.disjuncts[0].indicator_var))
    # use 4
    self.assertTrue(value(eight_process.use_unit_4or5ornot.disjuncts[0].indicator_var))
    self.assertFalse(value(eight_process.use_unit_4or5ornot.disjuncts[1].indicator_var))
    self.assertFalse(value(eight_process.use_unit_4or5ornot.disjuncts[2].indicator_var))
    # use 6
    self.assertTrue(value(eight_process.use_unit_6or7ornot.disjuncts[0].indicator_var))
    self.assertFalse(value(eight_process.use_unit_6or7ornot.disjuncts[1].indicator_var))
    self.assertFalse(value(eight_process.use_unit_6or7ornot.disjuncts[2].indicator_var))
    # use 8
    self.assertTrue(value(eight_process.use_unit_8ornot.disjuncts[0].indicator_var))
    self.assertFalse(value(eight_process.use_unit_8ornot.disjuncts[1].indicator_var))


def check_8PP_logical_solution(self, eight_process, results):
    self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1e-2)
    self.assertTrue(fabs(value(results.problem.upper_bound) - 68) <= 1e-2)

    # Check discrete solution
    # use 2
    self.assertTrue(value(eight_process.use_unit[2].indicator_var))
    self.assertFalse(value(eight_process.no_unit[2].indicator_var))
    # use 4
    self.assertTrue(value(eight_process.use_unit[4].indicator_var))
    self.assertFalse(value(eight_process.no_unit[4].indicator_var))
    # use 6
    self.assertTrue(value(eight_process.use_unit[6].indicator_var))
    self.assertFalse(value(eight_process.no_unit[6].indicator_var))
    # use 8
    self.assertTrue(value(eight_process.use_unit[8].indicator_var))
    self.assertFalse(value(eight_process.no_unit[8].indicator_var))

    self.assertFalse(value(eight_process.use_unit[1].indicator_var))
    self.assertTrue(value(eight_process.no_unit[1].indicator_var))
    self.assertFalse(value(eight_process.use_unit[3].indicator_var))
    self.assertTrue(value(eight_process.no_unit[3].indicator_var))
    self.assertFalse(value(eight_process.use_unit[5].indicator_var))
    self.assertTrue(value(eight_process.no_unit[5].indicator_var))
    self.assertFalse(value(eight_process.use_unit[7].indicator_var))
    self.assertTrue(value(eight_process.no_unit[7].indicator_var))
