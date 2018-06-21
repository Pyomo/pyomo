import pickle

import pyutilib.th as unittest
from pyomo.core.expr.numvalue import NumericValue
import pyomo.kernel
from pyomo.core.tests.unit.test_component_dict import \
    _TestActiveComponentDictBase
from pyomo.core.tests.unit.test_component_tuple import \
    _TestActiveComponentTupleBase
from pyomo.core.tests.unit.test_component_list import \
    _TestActiveComponentListBase
from pyomo.core.kernel.component_interface import (ICategorizedObject,
                                                   IComponent,
                                                   IComponentContainer,
                                                   _ActiveObjectMixin)
from pyomo.core.kernel.component_objective import (IObjective,
                                                   objective,
                                                   objective_dict,
                                                   objective_tuple,
                                                   objective_list,
                                                   minimize,
                                                   maximize)
from pyomo.core.kernel.component_variable import variable
from pyomo.core.kernel.component_block import block
from pyomo.core.kernel.set_types import (RealSet,
                                         IntegerSet)
from pyomo.core.base.objective import Objective

class Test_objective(unittest.TestCase):

    def test_pprint(self):
        import pyomo.kernel
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        v = variable()
        o = objective(expr=v**2)
        pyomo.kernel.pprint(o)
        b = block()
        b.o = o
        pyomo.kernel.pprint(o)
        pyomo.kernel.pprint(b)
        m = block()
        m.b = b
        pyomo.kernel.pprint(o)
        pyomo.kernel.pprint(b)
        pyomo.kernel.pprint(m)

    def test_ctype(self):
        o = objective()
        self.assertIs(o.ctype, Objective)
        self.assertIs(type(o).ctype, Objective)
        self.assertIs(objective.ctype, Objective)

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
        self.assertEqual(o.ctype, Objective)
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
        self.assertTrue(isinstance(o, IComponent))
        self.assertTrue(isinstance(o, _ActiveObjectMixin))
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

class Test_objective_dict(_TestActiveComponentDictBase,
                          unittest.TestCase):
    _container_type = objective_dict
    _ctype_factory = lambda self: objective()

class Test_objective_tuple(_TestActiveComponentTupleBase,
                           unittest.TestCase):
    _container_type = objective_tuple
    _ctype_factory = lambda self: objective()

class Test_objective_list(_TestActiveComponentListBase,
                          unittest.TestCase):
    _container_type = objective_list
    _ctype_factory = lambda self: objective()

if __name__ == "__main__":
    unittest.main()
