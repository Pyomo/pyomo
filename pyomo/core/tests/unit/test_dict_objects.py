#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pyutilib.th as unittest
from pyomo.core.base import (ConcreteModel, Var)
from pyomo.core.beta.dict_objects import (ExpressionDict,
                                          ConstraintDict,
                                          ObjectiveDict)
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.base.expression import _GeneralExpressionData

from six import StringIO

class TestDictObjects(unittest.TestCase):

    def test_init1(self):
        model = ConcreteModel()
        model.c = ConstraintDict()
        model.e = ExpressionDict()
        model.o = ObjectiveDict()
        self.assertEqual(model.c.is_indexed(), True)
        self.assertEqual(model.e.is_indexed(), True)
        self.assertEqual(model.o.is_indexed(), True)
        self.assertEqual(model.c.is_constructed(), True)
        self.assertEqual(model.e.is_constructed(), True)
        self.assertEqual(model.o.is_constructed(), True)

    def test_init2(self):
        model = ConcreteModel()
        model.x = Var()
        index = ['a', 1, None]
        model.c = ConstraintDict((i, model.x >= 1) for i in index)
        model.e = ExpressionDict((i, model.x**2) for i in index)
        model.o = ObjectiveDict((i, model.x) for i in index)
        self.assertEqual(model.c.is_indexed(), True)
        self.assertEqual(model.e.is_indexed(), True)
        self.assertEqual(model.o.is_indexed(), True)
        self.assertEqual(model.c.is_constructed(), True)
        self.assertEqual(model.e.is_constructed(), True)
        self.assertEqual(model.o.is_constructed(), True)

    def test_len1(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = ConstraintDict()
        model.e = ExpressionDict()
        model.o = ObjectiveDict()
        self.assertEqual(len(model.c), 0)
        self.assertEqual(len(model.e), 0)
        self.assertEqual(len(model.o), 0)

    def test_len2(self):
        model = ConcreteModel()
        model.x = Var()
        index = ['a', 1, None]
        model.c = ConstraintDict((i, model.x >= 1) for i in index)
        model.e = ExpressionDict((i, model.x**2) for i in index)
        model.o = ObjectiveDict((i, model.x) for i in index)
        self.assertEqual(len(model.c), 3)
        self.assertEqual(len(model.e), 3)
        self.assertEqual(len(model.o), 3)

    def test_setitem(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = ConstraintDict()
        model.e = ExpressionDict()
        model.o = ObjectiveDict()
        index = ['a', 1, None]
        for i in index:
            self.assertTrue(i not in model.c)
            self.assertTrue(i not in model.e)
            self.assertTrue(i not in model.o)
        for cnt, i in enumerate(index, 1):
            model.c[i] = model.x+5 >= 1
            model.e[i] = 1
            model.o[i] = model.x**2
            self.assertEqual(len(model.c), cnt)
            self.assertEqual(len(model.e), cnt)
            self.assertEqual(len(model.o), cnt)
            self.assertTrue(i in model.c)
            self.assertTrue(i in model.e)
            self.assertTrue(i in model.o)

    # make sure an existing Data object is NOT replaced
    # by a call to setitem but simply updated.
    def test_setitem_exists(self):
        model = ConcreteModel()
        model.x = Var()
        index = ['a', 1, None]
        model.c = ConstraintDict((i, model.x+5 >= 1) for i in index)
        model.e = ExpressionDict((i, 1) for i in index)
        model.o = ObjectiveDict((i, model.x**2) for i in index)
        self.assertEqual(len(model.c), 3)
        self.assertEqual(len(model.e), 3)
        self.assertEqual(len(model.o), 3)
        for i in index:
            self.assertTrue(i in model.c)
            self.assertTrue(i in model.e)
            self.assertTrue(i in model.o)
            cdata = model.c[i]
            edata = model.e[i]
            odata = model.o[i]
            model.c[i] = model.x**2 >= 1
            model.e[i] = model.x
            model.o[i] = 5
            self.assertEqual(len(model.c), 3)
            self.assertEqual(len(model.e), 3)
            self.assertEqual(len(model.o), 3)
            self.assertTrue(i in model.c)
            self.assertTrue(i in model.e)
            self.assertTrue(i in model.o)
            self.assertEqual(id(cdata), id(model.c[i]))
            self.assertEqual(id(edata), id(model.e[i]))
            self.assertEqual(id(odata), id(model.o[i]))

    # make sure an existing Data object IS replaced
    # by a call to setitem and not simply updated.
    # TODO: clean this up by making the _*Data objects
    #       public
    def test_setitem_exists_overwrite(self):
        model = ConcreteModel()
        model.x = Var()
        index = ['a', 1, None]
        model.c = ConstraintDict((i, model.x+5 >= 1) for i in index)
        model.e = ExpressionDict((i, 1) for i in index)
        model.o = ObjectiveDict((i, model.x**2) for i in index)
        self.assertEqual(len(model.c), 3)
        self.assertEqual(len(model.e), 3)
        self.assertEqual(len(model.o), 3)
        for i in index:
            self.assertTrue(i in model.c)
            self.assertTrue(i in model.e)
            self.assertTrue(i in model.o)
            cdata = model.c[i]
            edata = model.e[i]
            odata = model.o[i]
            model.c[i] = _GeneralConstraintData(model.x**2 >= 1)
            model.e[i] = _GeneralExpressionData(model.x)
            model.o[i] = _GeneralObjectiveData(5)
            self.assertEqual(len(model.c), 3)
            self.assertEqual(len(model.e), 3)
            self.assertEqual(len(model.o), 3)
            self.assertTrue(i in model.c)
            self.assertTrue(i in model.e)
            self.assertTrue(i in model.o)
            self.assertNotEqual(id(cdata), id(model.c[i]))
            self.assertNotEqual(id(edata), id(model.e[i]))
            self.assertNotEqual(id(odata), id(model.o[i]))

    def test_delitem(self):
        model = ConcreteModel()
        model.x = Var()
        index = ['a', 1, None]
        model.c = ConstraintDict((i, model.x+5 >= 1) for i in index)
        model.e = ExpressionDict((i, 1) for i in index)
        model.o = ObjectiveDict((i, model.x**2) for i in index)
        self.assertEqual(len(model.c), 3)
        self.assertEqual(len(model.e), 3)
        self.assertEqual(len(model.o), 3)
        for cnt, i in enumerate(index, 1):
            self.assertTrue(i in model.c)
            self.assertTrue(i in model.e)
            self.assertTrue(i in model.o)
            del model.c[i]
            del model.e[i]
            del model.o[i]
            self.assertEqual(len(model.c), 3-cnt)
            self.assertEqual(len(model.e), 3-cnt)
            self.assertEqual(len(model.o), 3-cnt)
            self.assertTrue(i not in model.c)
            self.assertTrue(i not in model.e)
            self.assertTrue(i not in model.o)

    def test_iter(self):
        model = ConcreteModel()
        model.x = Var()
        index = ['a', 1, None]
        model.c = ConstraintDict((i, model.x+5 >= 1) for i in index)
        model.e = ExpressionDict((i, 1) for i in index)
        model.o = ObjectiveDict((i, model.x**2) for i in index)
        self.assertEqual(len(model.c), 3)
        self.assertEqual(len(model.e), 3)
        self.assertEqual(len(model.o), 3)
        for comp in [model.c, model.e, model.o]:
            comp_index = [i for i in comp]
            self.assertEqual(len(comp_index), 3)
            self.assertTrue(comp_index[0] in index)
            self.assertTrue(comp_index[1] in index)
            self.assertTrue(comp_index[2] in index)

    def test_model_clone(self):
        model = ConcreteModel()
        index = ['a', 1, None]
        model.x = Var()
        model.c = ConstraintDict((i, model.x >= 1) for i in index)
        model.e = ExpressionDict((i, model.x**2) for i in index)
        model.o = ObjectiveDict((i, model.x) for i in index)
        inst = model.clone()
        self.assertNotEqual(id(inst.x), id(model.x))
        self.assertNotEqual(id(inst.c), id(model.c))
        self.assertNotEqual(id(inst.e), id(model.e))
        self.assertNotEqual(id(inst.o), id(model.o))
        for i in index:
            self.assertNotEqual(id(inst.c[i]), id(model.c[i]))
            self.assertNotEqual(id(inst.e[i]), id(model.e[i]))
            self.assertNotEqual(id(inst.o[i]), id(model.o[i]))

if __name__ == "__main__":
    unittest.main()


    
