#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pickle

import pyutilib.th as unittest
from pyomo.core.expr.numvalue import NumericValue
from pyomo.kernel import pprint
from pyomo.core.tests.unit.kernel.test_dict_container import \
    _TestActiveDictContainerBase
from pyomo.core.tests.unit.kernel.test_tuple_container import \
    _TestActiveTupleContainerBase
from pyomo.core.tests.unit.kernel.test_list_container import \
    _TestActiveListContainerBase
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.objective import (IObjective,
                                         objective,
                                         objective_dict,
                                         objective_tuple,
                                         objective_list,
                                         minimize,
                                         maximize)
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.block import block

class Test_objective(unittest.TestCase):

    def test_pprint(self):
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        v = variable()
        o = objective(expr=v**2)
        pprint(o)
        b = block()
        b.o = o
        pprint(o)
        pprint(b)
        m = block()
        m.b = b
        pprint(o)
        pprint(b)
        pprint(m)

    def test_ctype(self):
        o = objective()
        self.assertIs(o.ctype, IObjective)
        self.assertIs(type(o), objective)
        self.assertIs(type(o)._ctype, IObjective)

    def test_pickle(self):
        o = objective(sense=maximize,
                      expr=1.0)
        self.assertEqual(o.sense, maximize)
        self.assertEqual(o.expr, 1.0)
        self.assertEqual(o.parent, None)
        oup = pickle.loads(
            pickle.dumps(o))
        self.assertEqual(oup.sense, maximize)
        self.assertEqual(oup.expr, 1.0)
        self.assertEqual(oup.parent, None)
        b = block()
        b.o = o
        self.assertIs(o.parent, b)
        bup = pickle.loads(
            pickle.dumps(b))
        oup = bup.o
        self.assertEqual(oup.sense, maximize)
        self.assertEqual(oup.expr, 1.0)
        self.assertIs(oup.parent, bup)

    def test_init(self):
        o = objective()
        self.assertTrue(o.parent is None)
        self.assertEqual(o.ctype, IObjective)
        self.assertEqual(o.expr, None)
        self.assertEqual(o.sense, minimize)
        self.assertEqual(o.is_minimizing(), True)

    def test_sense(self):
        o = objective()
        o.sense = maximize
        self.assertEqual(o.sense, maximize)
        self.assertEqual(o.is_minimizing(), False)
        with self.assertRaises(ValueError):
            o.sense = 100
        self.assertEqual(o.sense, maximize)
        self.assertEqual(o.is_minimizing(), False)

    def test_type(self):
        o = objective()
        self.assertTrue(isinstance(o, ICategorizedObject))
        self.assertTrue(isinstance(o, IObjective))
        self.assertTrue(isinstance(o, NumericValue))

    def test_active(self):
        o = objective()
        self.assertEqual(o.active, True)
        o.deactivate()
        self.assertEqual(o.active, False)
        o.activate()
        self.assertEqual(o.active, True)

        b = block()
        self.assertEqual(b.active, True)
        b.deactivate()
        self.assertEqual(b.active, False)
        b.o = o
        self.assertEqual(o.active, True)
        self.assertEqual(b.active, False)
        o.deactivate()
        self.assertEqual(o.active, False)
        self.assertEqual(b.active, False)
        b.activate()
        self.assertEqual(o.active, False)
        self.assertEqual(b.active, True)
        b.activate(shallow=False)
        self.assertEqual(o.active, True)
        self.assertEqual(b.active, True)
        b.deactivate(shallow=False)
        self.assertEqual(o.active, False)
        self.assertEqual(b.active, False)

class Test_objective_dict(_TestActiveDictContainerBase,
                          unittest.TestCase):
    _container_type = objective_dict
    _ctype_factory = lambda self: objective()

class Test_objective_tuple(_TestActiveTupleContainerBase,
                           unittest.TestCase):
    _container_type = objective_tuple
    _ctype_factory = lambda self: objective()

class Test_objective_list(_TestActiveListContainerBase,
                          unittest.TestCase):
    _container_type = objective_list
    _ctype_factory = lambda self: objective()

if __name__ == "__main__":
    unittest.main()
