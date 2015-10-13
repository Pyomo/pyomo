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
from pyomo.core.beta.list_objects import (XExpressionList,
                                          XConstraintList,
                                          XObjectiveList)
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.base.expression import _GeneralExpressionData

from six import StringIO

class TestComponentList(unittest.TestCase):

    def test_init1(self):
        model = ConcreteModel()
        model.c = XConstraintList()
        model.e = XExpressionList()
        model.o = XObjectiveList()
        self.assertEqual(model.c.is_indexed(), True)
        self.assertEqual(model.e.is_indexed(), True)
        self.assertEqual(model.o.is_indexed(), True)
        self.assertEqual(model.c.is_constructed(), True)
        self.assertEqual(model.e.is_constructed(), True)
        self.assertEqual(model.o.is_constructed(), True)

    def test_init2(self):
        model = ConcreteModel()
        model.x = Var()
        index = range(5)
        model.c = XConstraintList(_GeneralConstraintData(model.x >= 1) for i in index)
        model.e = XExpressionList(_GeneralExpressionData(model.x**2) for i in index)
        model.o = XObjectiveList(_GeneralObjectiveData(model.x) for i in index)
        self.assertEqual(model.c.is_indexed(), True)
        self.assertEqual(model.e.is_indexed(), True)
        self.assertEqual(model.o.is_indexed(), True)
        self.assertEqual(model.c.is_constructed(), True)
        self.assertEqual(model.e.is_constructed(), True)
        self.assertEqual(model.o.is_constructed(), True)

    def test_len1(self):
        model = ConcreteModel()
        model.c = XConstraintList()
        model.e = XExpressionList()
        model.o = XObjectiveList()
        self.assertEqual(len(model.c), 0)
        self.assertEqual(len(model.e), 0)
        self.assertEqual(len(model.o), 0)

    def test_len2(self):
        model = ConcreteModel()
        model.x = Var()
        index = range(5)
        model.c = XConstraintList(_GeneralConstraintData(model.x >= 1) for i in index)
        model.e = XExpressionList(_GeneralExpressionData(model.x**2) for i in index)
        model.o = XObjectiveList(_GeneralObjectiveData(model.x) for i in index)
        self.assertEqual(len(model.c), len(index))
        self.assertEqual(len(model.e), len(index))
        self.assertEqual(len(model.o), len(index))

    def test_append(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = XConstraintList()
        model.e = XExpressionList()
        model.o = XObjectiveList()
        index = range(5)
        self.assertEqual(len(model.c), 0)
        self.assertEqual(len(model.e), 0)
        self.assertEqual(len(model.o), 0)
        for i in index:
            c_new = _GeneralConstraintData(model.x+5 >= 1)
            model.c.append(c_new)
            self.assertEqual(id(model.c[-1]), id(c_new))
            e_new = _GeneralExpressionData(1)
            model.e.append(e_new)
            self.assertEqual(id(model.e[-1]), id(e_new))
            o_new = _GeneralObjectiveData(model.x**2)
            model.o.append(o_new)
            self.assertEqual(id(model.o[-1]), id(o_new))
            self.assertEqual(len(model.c), i+1)
            self.assertEqual(len(model.e), i+1)
            self.assertEqual(len(model.o), i+1)

    def test_insert(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = XConstraintList()
        model.e = XExpressionList()
        model.o = XObjectiveList()
        index = range(5)
        self.assertEqual(len(model.c), 0)
        self.assertEqual(len(model.e), 0)
        self.assertEqual(len(model.o), 0)
        for i in index:
            c_new = _GeneralConstraintData(model.x+5 >= 1)
            model.c.insert(0, c_new)
            self.assertEqual(id(model.c[0]), id(c_new))
            e_new = _GeneralExpressionData(1)
            model.e.insert(0, e_new)
            self.assertEqual(id(model.e[0]), id(e_new))
            o_new = _GeneralObjectiveData(model.x**2)
            model.o.insert(0, o_new)
            self.assertEqual(id(model.o[0]), id(o_new))
            self.assertEqual(len(model.c), i+1)
            self.assertEqual(len(model.e), i+1)
            self.assertEqual(len(model.o), i+1)

    def test_setitem(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = XConstraintList()
        model.e = XExpressionList()
        model.o = XObjectiveList()
        index = range(5)
        for i in index:
            model.c.append(_GeneralConstraintData(model.x+5 >= 1))
            model.e.append(_GeneralExpressionData(1))
            model.o.append(_GeneralObjectiveData(model.x**2))
        for i in index:
            c_new = _GeneralConstraintData(model.x+5 >= 1)
            self.assertNotEqual(id(c_new), id(model.c[i]))
            model.c[i] = c_new
            e_new = _GeneralExpressionData(1)
            self.assertNotEqual(id(e_new), id(model.e[i]))
            model.e[i] = e_new
            o_new = _GeneralObjectiveData(model.x**2)
            self.assertNotEqual(id(o_new), id(model.o[i]))
            model.o[i] = o_new
            self.assertEqual(len(model.c), len(index))
            self.assertEqual(len(model.e), len(index))
            self.assertEqual(len(model.o), len(index))
            self.assertEqual(id(c_new), id(model.c[i]))
            self.assertEqual(id(e_new), id(model.e[i]))
            self.assertEqual(id(o_new), id(model.o[i]))


    # For now just test that implicit assignment raises an exception
    # (we may reexamine allowing implicit assignment / update in the
    # future
    def test_setitem_implicit_failure(self):
        model = ConcreteModel()
        model.x = Var()
        index = range(5)
        try:
            model.c = XConstraintList(model.x+5 >= 1 for i in index)
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")
        try:
            model.e = XExpressionList(1 for i in index)
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")
        try:
            model.o = XObjectiveList(model.x**2 for i in index)
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")

        model.c = XConstraintList()
        model.e = XExpressionList()
        model.o = XObjectiveList()
        try:
            model.c.append(model.x+5 >= 1)
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")
        try:
            model.e.append(1)
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")
        try:
            model.o.append(model.x**2)
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")

        model.c.append(_GeneralConstraintData(model.x+5 >= 1))
        model.e.append(_GeneralExpressionData(1))
        model.o.append(_GeneralObjectiveData(model.x**2))
        try:
            model.c.insert(0, model.x+5 >= 1)
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")
        try:
            model.e.insert(0, 1)
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")
        try:
            model.o.insert(0, model.x**2)
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")

    # make sure an existing Data object IS replaced
    # by a call to setitem and not simply updated.
    def test_setitem_exists_overwrite(self):
        model = ConcreteModel()
        model.x = Var()
        index = range(5)
        model.c = XConstraintList(_GeneralConstraintData(model.x+5 >= 1) for i in index)
        model.e = XExpressionList(_GeneralExpressionData(1) for i in index)
        model.o = XObjectiveList(_GeneralObjectiveData(model.x**2) for i in index)
        self.assertEqual(len(model.c), len(index))
        self.assertEqual(len(model.e), len(index))
        self.assertEqual(len(model.o), len(index))
        for i in index:
            cdata = model.c[i]
            edata = model.e[i]
            odata = model.o[i]
            self.assertEqual(id(cdata.parent_component()), id(model.c))
            self.assertEqual(id(edata.parent_component()), id(model.e))
            self.assertEqual(id(odata.parent_component()), id(model.o))
            model.c[i] = _GeneralConstraintData(model.x**2 >= 1)
            model.e[i] = _GeneralExpressionData(model.x)
            model.o[i] = _GeneralObjectiveData(5)
            self.assertEqual(len(model.c), len(index))
            self.assertEqual(len(model.e), len(index))
            self.assertEqual(len(model.o), len(index))
            self.assertNotEqual(id(cdata), id(model.c[i]))
            self.assertNotEqual(id(edata), id(model.e[i]))
            self.assertNotEqual(id(odata), id(model.o[i]))
            self.assertEqual(cdata.parent_component(), None)
            self.assertEqual(edata.parent_component(), None)
            self.assertEqual(odata.parent_component(), None)

    def test_delitem(self):
        model = ConcreteModel()
        model.x = Var()
        index = range(5)
        model.c = XConstraintList(_GeneralConstraintData(model.x+5 >= 1) for i in index)
        model.e = XExpressionList(_GeneralExpressionData(1) for i in index)
        model.o = XObjectiveList(_GeneralObjectiveData(model.x**2) for i in index)
        self.assertEqual(len(model.c), len(index))
        self.assertEqual(len(model.e), len(index))
        self.assertEqual(len(model.o), len(index))
        for i in index:
            cdata = model.c[0]
            edata = model.e[0]
            odata = model.o[0]
            self.assertEqual(id(cdata.parent_component()), id(model.c))
            self.assertEqual(id(edata.parent_component()), id(model.e))
            self.assertEqual(id(odata.parent_component()), id(model.o))
            del model.c[0]
            del model.e[0]
            del model.o[0]
            self.assertEqual(len(model.c), len(index)-(i+1))
            self.assertEqual(len(model.e), len(index)-(i+1))
            self.assertEqual(len(model.o), len(index)-(i+1))
            self.assertEqual(cdata.parent_component(), None)
            self.assertEqual(edata.parent_component(), None)
            self.assertEqual(odata.parent_component(), None)

    def test_iter(self):
        model = ConcreteModel()
        model.x = Var()
        index = range(5)
        model.c = XConstraintList(_GeneralConstraintData(model.x+5 >= 1) for i in index)
        model.e = XExpressionList(_GeneralExpressionData(1) for i in index)
        model.o = XObjectiveList(_GeneralObjectiveData(model.x**2) for i in index)
        self.assertEqual(len(model.c), len(index))
        self.assertEqual(len(model.e), len(index))
        self.assertEqual(len(model.o), len(index))
        for comp in [model.c, model.e, model.o]:
            raw_list = comp[:]
            self.assertEqual(type(raw_list), list)
            for c1, c2 in zip(raw_list, comp):
                self.assertEqual(id(c1), id(c2))



    def test_reverse(self):
        model = ConcreteModel()
        index = range(5)
        model.x = Var()
        model.c = XConstraintList(_GeneralConstraintData(model.x >= 1) for i in index)
        model.e = XExpressionList(_GeneralExpressionData(model.x**2) for i in index)
        model.o = XObjectiveList(_GeneralObjectiveData(model.x) for i in index)
        for comp in [model.c, model.e, model.o]:
            raw_list = comp[:]
            self.assertEqual(type(raw_list), list)
            comp.reverse()
            raw_list.reverse()
            for c1, c2 in zip(comp, raw_list):
                self.assertEqual(id(c1), id(c2))

    def test_remove(self):
        model = ConcreteModel()
        index = range(5)
        model.x = Var()
        model.c = XConstraintList(_GeneralConstraintData(model.x >= 1) for i in index)
        model.e = XExpressionList(_GeneralExpressionData(model.x**2) for i in index)
        model.o = XObjectiveList(_GeneralObjectiveData(model.x) for i in index)
        for comp in [model.c, model.e, model.o]:
            for i in index:
                cdata = comp[0]
                self.assertEqual(cdata in comp, True)
                comp.remove(cdata)
                self.assertEqual(cdata in comp, False)

    def test_pop(self):
        model = ConcreteModel()
        index = range(5)
        model.x = Var()
        model.c = XConstraintList(_GeneralConstraintData(model.x >= 1) for i in index)
        model.e = XExpressionList(_GeneralExpressionData(model.x**2) for i in index)
        model.o = XObjectiveList(_GeneralObjectiveData(model.x) for i in index)
        for comp in [model.c, model.e, model.o]:
            for i in index:
                cdata = comp[-1]
                self.assertEqual(cdata in comp, True)
                last = comp.pop()
                self.assertEqual(cdata in comp, False)
                self.assertEqual(id(cdata), id(last))

    def test_index(self):
        model = ConcreteModel()
        index = range(5)
        model.x = Var()
        model.c = XConstraintList(_GeneralConstraintData(model.x >= 1) for i in index)
        model.e = XExpressionList(_GeneralExpressionData(model.x**2) for i in index)
        model.o = XObjectiveList(_GeneralObjectiveData(model.x) for i in index)
        for comp in [model.c, model.e, model.o]:
            for i in index:
                cdata = comp[i]
                self.assertEqual(comp.index(cdata), i)

    def test_extend(self):
        model = ConcreteModel()
        index = range(5)
        model.x = Var()
        model.c = XConstraintList(_GeneralConstraintData(model.x >= 1) for i in index)
        c_more_list = [_GeneralConstraintData(model.x >= 1) for i in index]
        model.e = XExpressionList(_GeneralExpressionData(model.x**2) for i in index)
        e_more_list = [_GeneralExpressionData(model.x**2) for i in index]
        model.o = XObjectiveList(_GeneralObjectiveData(model.x) for i in index)
        o_more_list = [_GeneralObjectiveData(model.x) for i in index]
        for comp, comp_more in [(model.c, c_more_list),
                                (model.e, e_more_list),
                                (model.o, o_more_list)]:

            self.assertEqual(len(comp), len(index))
            self.assertTrue(len(comp_more) > 0)
            for cdata in comp_more:
                self.assertEqual(cdata.parent_component(), None)
            comp.extend(comp_more)
            for cdata in comp_more:
                self.assertEqual(id(cdata.parent_component()),
                                 id(comp))

    def test_count(self):
        model = ConcreteModel()
        index = range(5)
        model.x = Var()
        model.c = XConstraintList(_GeneralConstraintData(model.x >= 1) for i in index)
        model.e = XExpressionList(_GeneralExpressionData(model.x**2) for i in index)
        model.o = XObjectiveList(_GeneralObjectiveData(model.x) for i in index)
        for comp in [model.c, model.e, model.o]:
            for i in index:
                self.assertEqual(comp.count(comp[i]), 1)

    def test_model_clone(self):
        model = ConcreteModel()
        index = range(5)
        model.x = Var()
        model.c = XConstraintList(_GeneralConstraintData(model.x >= 1) for i in index)
        model.e = XExpressionList(_GeneralExpressionData(model.x**2) for i in index)
        model.o = XObjectiveList(_GeneralObjectiveData(model.x) for i in index)
        inst = model.clone()
        self.assertNotEqual(id(inst.x), id(model.x))
        self.assertNotEqual(id(inst.c), id(model.c))
        self.assertNotEqual(id(inst.e), id(model.e))
        self.assertNotEqual(id(inst.o), id(model.o))
        for i in index:
            self.assertNotEqual(id(inst.c[i]), id(model.c[i]))
            self.assertNotEqual(id(inst.e[i]), id(model.e[i]))
            self.assertNotEqual(id(inst.o[i]), id(model.o[i]))

    # Befault, assigning a new component to a dict container makes it
    # active (unless the default active state for the container
    # datatype is False, which is not the case for any currently
    # existing implementations).
    def test_active(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = XConstraintList()
        self.assertEqual(model.c.active, True)
        model.c.deactivate()
        self.assertEqual(model.c.active, False)
        model.c.append(_GeneralConstraintData(model.x >= 1))
        self.assertEqual(model.c.active, True)
        model.c.deactivate()
        self.assertEqual(model.c.active, False)
        model.c.insert(0, _GeneralConstraintData(model.x >= 1))
        self.assertEqual(model.c.active, True)

        model.e = XExpressionList()
        self.assertEqual(model.e.active, True)
        model.e.deactivate()
        self.assertEqual(model.e.active, False)
        model.e.append(_GeneralExpressionData(model.x**2))
        self.assertEqual(model.e.active, True)
        model.e.deactivate()
        self.assertEqual(model.e.active, False)
        model.e.insert(0, _GeneralExpressionData(model.x**2))
        self.assertEqual(model.e.active, True)

        model.o = XObjectiveList()
        self.assertEqual(model.o.active, True)
        model.o.deactivate()
        self.assertEqual(model.o.active, False)
        model.o.append(_GeneralObjectiveData(model.x))
        self.assertEqual(model.o.active, True)
        model.o.deactivate()
        self.assertEqual(model.o.active, False)
        model.o.insert(0, _GeneralObjectiveData(model.x))
        self.assertEqual(model.o.active, True)

    def test_cname(self):
        model = ConcreteModel()
        index = range(5)
        model.x = Var()
        model.c = XConstraintList(_GeneralConstraintData(model.x >= 1) for i in index)
        model.e = XExpressionList(_GeneralExpressionData(model.x**2) for i in index)
        model.o = XObjectiveList(_GeneralObjectiveData(model.x) for i in index)
        for comp in [model.c, model.e, model.o]:
            for i in index:
                compdata = comp[i]
                self.assertEqual(compdata.cname(False),
                                 compdata.cname(True))
                if isinstance(comp, XConstraintList):
                    prefix = "c"
                elif isinstance(comp, XExpressionList):
                    prefix = "e"
                elif isinstance(comp, XObjectiveList):
                    prefix = "o"
                else:
                    assert False
                cname = prefix + "["+str(i)+"]"
                self.assertEqual(compdata.cname(False),
                                 cname)

if __name__ == "__main__":
    unittest.main()

