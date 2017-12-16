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
import io
import unittest

import pyomo.environ as pe


class TestFileObject(unittest.TestCase):

    @staticmethod
    def test_file_object():
        model = pe.AbstractModel()

        model.n = pe.Param()
        model.days_horizon = pe.Param()
        model.delta_t = pe.Param()

        input_dat = io.StringIO("""
        param n := 672 ;
        param days_horizon := 7 ;
        param delta_t := 0.25 ;
        """)
        model.create_instance(input_dat)


if __name__ == "__main__":
    unittest.main()
