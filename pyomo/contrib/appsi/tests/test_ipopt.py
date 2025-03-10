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

from pyomo.common import unittest
from pyomo.contrib.appsi.solvers import ipopt


ipopt_available = ipopt.Ipopt().available()


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestIpoptInterface(unittest.TestCase):
    def test_has_linear_solver(self):
        opt = ipopt.Ipopt()
        self.assertTrue(
            any(
                map(
                    opt.has_linear_solver,
                    [
                        'mumps',
                        'ma27',
                        'ma57',
                        'ma77',
                        'ma86',
                        'ma97',
                        'pardiso',
                        'pardisomkl',
                        'spral',
                        'wsmp',
                    ],
                )
            )
        )
        self.assertFalse(opt.has_linear_solver('bogus_linear_solver'))
