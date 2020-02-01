#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import uuid

import pyutilib.th as unittest

from pyomo.pysp.scenariotree.util import \
    (_max_int32,
     _max_uint32,
     _convert_range_one_to_max_int32,
     _convert_range_zero_to_max_int32,
     _convert_range_one_to_max_uint32,
     _convert_range_zero_to_max_uint32,
     scenario_tree_id_to_pint32,
     scenario_tree_id_to_nzint32,
     scenario_tree_id_to_puint32,
     scenario_tree_id_to_nzuint32,
     _max_int64,
     _max_uint64,
     _convert_range_one_to_max_int64,
     _convert_range_zero_to_max_int64,
     _convert_range_one_to_max_uint64,
     _convert_range_zero_to_max_uint64,
     scenario_tree_id_to_pint64,
     scenario_tree_id_to_nzint64,
     scenario_tree_id_to_puint64,
     scenario_tree_id_to_nzuint64)

class TestScenarioTreeIDToInteger(unittest.TestCase):

    _name = "Root"

    def test_scenario_tree_id_to_pint32(self):
        self.assertEqual(_convert_range_one_to_max_int32(0),
                         1)
        self.assertEqual(_convert_range_one_to_max_int32(1),
                         1)
        self.assertEqual(_convert_range_one_to_max_int32(2),
                         2)
        self.assertEqual(_convert_range_one_to_max_int32(_max_int32-1),
                         _max_int32-1)
        self.assertEqual(_convert_range_one_to_max_int32(_max_int32),
                         _max_int32)
        self.assertEqual(_convert_range_one_to_max_int32(_max_int32+1),
                         1)
        v = scenario_tree_id_to_pint32(self._name, str(uuid.uuid4()))
        self.assertTrue(1 <= v <= 2**31 -1)

    def test_scenario_tree_id_to_nzint32(self):
        self.assertEqual(_convert_range_zero_to_max_int32(0),
                         0)
        self.assertEqual(_convert_range_zero_to_max_int32(1),
                         1)
        self.assertEqual(_convert_range_zero_to_max_int32(2),
                         2)
        self.assertEqual(_convert_range_zero_to_max_int32(_max_int32-1),
                         _max_int32-1)
        self.assertEqual(_convert_range_zero_to_max_int32(_max_int32),
                         _max_int32)
        self.assertEqual(_convert_range_zero_to_max_int32(_max_int32+1),
                         0)
        v = scenario_tree_id_to_nzint32(self._name, str(uuid.uuid4()))
        self.assertTrue(0 <= v <= 2**31 -1)

    def test_scenario_tree_id_to_puint32(self):
        self.assertEqual(_convert_range_one_to_max_uint32(0),
                         1)
        self.assertEqual(_convert_range_one_to_max_uint32(1),
                         1)
        self.assertEqual(_convert_range_one_to_max_uint32(2),
                         2)
        self.assertEqual(_convert_range_one_to_max_uint32(_max_uint32-1),
                         _max_uint32-1)
        self.assertEqual(_convert_range_one_to_max_uint32(_max_uint32),
                         _max_uint32)
        self.assertEqual(_convert_range_one_to_max_uint32(_max_uint32+1),
                         1)
        v = scenario_tree_id_to_puint32(self._name, str(uuid.uuid4()))
        self.assertTrue(1 <= v <= 2**32 -1)

    def test_scenario_tree_id_to_nzuint32(self):
        self.assertEqual(_convert_range_zero_to_max_uint32(0),
                         0)
        self.assertEqual(_convert_range_zero_to_max_uint32(1),
                         1)
        self.assertEqual(_convert_range_zero_to_max_uint32(2),
                         2)
        self.assertEqual(_convert_range_zero_to_max_uint32(_max_uint32-1),
                         _max_uint32-1)
        self.assertEqual(_convert_range_zero_to_max_uint32(_max_uint32),
                         _max_uint32)
        self.assertEqual(_convert_range_zero_to_max_uint32(_max_uint32+1),
                         0)
        v = scenario_tree_id_to_nzuint32(self._name, str(uuid.uuid4()))
        self.assertTrue(0 <= v <= 2**32 -1)

    def test_scenario_tree_id_to_pint64(self):
        self.assertEqual(_convert_range_one_to_max_int64(0),
                         1)
        self.assertEqual(_convert_range_one_to_max_int64(1),
                         1)
        self.assertEqual(_convert_range_one_to_max_int64(2),
                         2)
        self.assertEqual(_convert_range_one_to_max_int64(_max_int64-1),
                         _max_int64-1)
        self.assertEqual(_convert_range_one_to_max_int64(_max_int64),
                         _max_int64)
        self.assertEqual(_convert_range_one_to_max_int64(_max_int64+1),
                         1)
        v = scenario_tree_id_to_pint64(self._name, str(uuid.uuid4()))
        self.assertTrue(1 <= v <= 2**63 -1)

    def test_scenario_tree_id_to_nzint64(self):
        self.assertEqual(_convert_range_zero_to_max_int64(0),
                         0)
        self.assertEqual(_convert_range_zero_to_max_int64(1),
                         1)
        self.assertEqual(_convert_range_zero_to_max_int64(2),
                         2)
        self.assertEqual(_convert_range_zero_to_max_int64(_max_int64-1),
                         _max_int64-1)
        self.assertEqual(_convert_range_zero_to_max_int64(_max_int64),
                         _max_int64)
        self.assertEqual(_convert_range_zero_to_max_int64(_max_int64+1),
                         0)
        v = scenario_tree_id_to_nzint64(self._name, str(uuid.uuid4()))
        self.assertTrue(0 <= v <= 2**63 -1)

    def test_scenario_tree_id_to_puint64(self):
        self.assertEqual(_convert_range_one_to_max_uint64(0),
                         1)
        self.assertEqual(_convert_range_one_to_max_uint64(1),
                         1)
        self.assertEqual(_convert_range_one_to_max_uint64(2),
                         2)
        self.assertEqual(_convert_range_one_to_max_uint64(_max_uint64-1),
                         _max_uint64-1)
        self.assertEqual(_convert_range_one_to_max_uint64(_max_uint64),
                         _max_uint64)
        self.assertEqual(_convert_range_one_to_max_uint64(_max_uint64+1),
                         1)
        v = scenario_tree_id_to_puint64(self._name, str(uuid.uuid4()))
        self.assertTrue(1 <= v <= 2**64 -1)

    def test_scenario_tree_id_to_nzuint64(self):
        self.assertEqual(_convert_range_zero_to_max_uint64(0),
                         0)
        self.assertEqual(_convert_range_zero_to_max_uint64(1),
                         1)
        self.assertEqual(_convert_range_zero_to_max_uint64(2),
                         2)
        self.assertEqual(_convert_range_zero_to_max_uint64(_max_uint64-1),
                         _max_uint64-1)
        self.assertEqual(_convert_range_zero_to_max_uint64(_max_uint64),
                         _max_uint64)
        self.assertEqual(_convert_range_zero_to_max_uint64(_max_uint64+1),
                         0)
        v = scenario_tree_id_to_nzuint64(self._name, str(uuid.uuid4()))
        self.assertTrue(0 <= v <= 2**64 -1)

if __name__ == "__main__":
    unittest.main()
