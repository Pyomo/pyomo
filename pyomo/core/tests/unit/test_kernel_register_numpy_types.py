#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available
from pyomo.common.log import LoggingIntercept

# Boolean
numpy_bool_names = []
if numpy_available:
    numpy_bool_names.append('bool_')
# Integers
numpy_int_names = []
if numpy_available:
    numpy_int_names.append('int_')
    numpy_int_names.append('intc')
    numpy_int_names.append('intp')
    numpy_int_names.append('int8')
    numpy_int_names.append('int16')
    numpy_int_names.append('int32')
    numpy_int_names.append('int64')
    numpy_int_names.append('uint8')
    numpy_int_names.append('uint16')
    numpy_int_names.append('uint32')
    numpy_int_names.append('uint64')
# Reals
numpy_float_names = []
if numpy_available:
    numpy_float_names.append('float_')
    numpy_float_names.append('float16')
    numpy_float_names.append('float32')
    numpy_float_names.append('float64')
    numpy_float_names.append('ndarray')
# Complex
numpy_complex_names = []
if numpy_available:
    numpy_complex_names.append('complex_')
    numpy_complex_names.append('complex64')
    numpy_complex_names.append('complex128')


class TestNumpyRegistration(unittest.TestCase):
    def test_deprecation(self):
        with LoggingIntercept() as LOG:
            import pyomo.core.kernel.register_numpy_types as rnt
        self.assertRegex(
            LOG.getvalue(),
            "DEPRECATED: pyomo.core.kernel.register_numpy_types is deprecated.")
        self.assertEqual(sorted(rnt.numpy_bool_names),
                         sorted(numpy_bool_names))
        self.assertEqual(sorted(rnt.numpy_int_names),
                         sorted(numpy_int_names))
        self.assertEqual(sorted(rnt.numpy_float_names),
                         sorted(numpy_float_names))
        self.assertEqual(sorted(rnt.numpy_complex_names),
                         sorted(numpy_complex_names))
