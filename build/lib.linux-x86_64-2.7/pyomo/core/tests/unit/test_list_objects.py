#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest
from pyomo.core.base import (ConcreteModel, Var, Reals)
from pyomo.core.beta.list_objects import (XVarList,
                                          XConstraintList,
                                          XObjectiveList,
                                          XExpressionList)
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.base.expression import _GeneralExpressionData

class _TestComponentListBase(object):

    _ctype = None
    _cdatatype = None

    def setUp(self):
        self.model = ConcreteModel()
        self.model.x = Var()
        # set by derived class
        self._arg = None

    def tearDown(self):
        self.model = None
        self._arg = None

    def test_init1(self):
        model = self.model
        model.c = self._ctype()
        self.assertEqual(model.c.is_indexed(), True)
        self.assertEqual(model.c.is_constructed(), True)

    def test_init2(self):
        model = self.model
        index = range(5)
        model.c = self._ctype(self._cdatatype(self._arg()) for i in index)
        self.assertEqual(model.c.is_indexed(), True)
        self.assertEqual(model.c.is_constructed(), True)
        with self.assertRaises(TypeError):
            model.d = self._ctype(*tuple(self._cdatatype(self._arg())
                                         for i in index))

    def test_len1(self):
        model = self.model
        model.c = self._ctype()
        self.assertEqual(len(model.c), 0)

    def test_len2(self):
        model = self.model
        index = range(5)
        model.c = self._ctype(self._cdatatype(self._arg()) for i in index)
        self.assertEqual(len(model.c), len(index))

    def test_append(self):
        model = self.model
        model.c = self._ctype()
        index = range(5)
        self.assertEqual(len(model.c), 0)
        for i in index:
            c_new = self._cdatatype(self._arg())
            model.c.append(c_new)
            self.assertEqual(id(model.c[-1]), id(c_new))
            self.assertEqual(len(model.c), i+1)

    def test_insert(self):
        model = self.model
        model.c = self._ctype()
        index = range(5)
        self.assertEqual(len(model.c), 0)
        for i in index:
            c_new = self._cdatatype(self._arg())
            model.c.insert(0, c_new)
            self.assertEqual(id(model.c[0]), id(c_new))
            self.assertEqual(len(model.c), i+1)

    def test_setitem(self):
        model = self.model
        model.c = self._ctype()
        index = range(5)
        for i in index:
            model.c.append(self._cdatatype(self._arg()))
        for i in index:
            c_new = self._cdatatype(self._arg())
            self.assertNotEqual(id(c_new), id(model.c[i]))
            model.c[i] = c_new
            self.assertEqual(len(model.c), len(index))
            self.assertEqual(id(c_new), id(model.c[i]))

    def test_wrong_type_init(self):
        model = self.model
        index = range(5)
        with self.assertRaises(TypeError):
            model.c = self._ctype(self._arg() for i in index)

    def test_wrong_type_append(self):
        model = self.model
        model.c = self._ctype()
        model.c.append(self._cdatatype(self._arg()))
        with self.assertRaises(TypeError):
            model.c.append(self._arg())

    def test_wrong_type_insert(self):
        model = self.model
        model.c = self._ctype()
        model.c.append(self._cdatatype(self._arg()))
        model.c.insert(0, self._cdatatype(self._arg()))
        with self.assertRaises(TypeError):
            model.c.insert(0, self._arg())

    def test_wrong_type_setitem(self):
        model = self.model
        model.c = self._ctype()
        model.c.append(self._cdatatype(self._arg()))
        model.c[0] = self._cdatatype(self._arg())
        with self.assertRaises(TypeError):
            model.c[0] = self._arg()

    def test_has_parent_init(self):
        model = self.model
        model.c = self._ctype()
        model.c.append(self._cdatatype(self._arg()))
        with self.assertRaises(ValueError):
            model.c.append(model.c[0])
        with self.assertRaises(ValueError):
            model.d = self._ctype(model.c)

    def test_has_parent_append(self):
        model = self.model
        model.c = self._ctype()
        model.c.append(self._cdatatype(self._arg()))
        with self.assertRaises(ValueError):
            model.c.append(model.c[0])
        d = []
        d.append(model.c[0])
        model.d = self._ctype()
        with self.assertRaises(ValueError):
            model.d.append(model.c[0])

    def test_has_parent_insert(self):
        model = self.model
        model.c = self._ctype()
        model.c.append(self._cdatatype(self._arg()))
        model.c.insert(0, self._cdatatype(self._arg()))
        with self.assertRaises(ValueError):
            model.c.insert(0, model.c[0])
        d = []
        d.insert(0, model.c[0])
        model.d = self._ctype()
        with self.assertRaises(ValueError):
            model.d.insert(0, model.c[0])

    def test_has_parent_setitem(self):
        model = self.model
        model.c = self._ctype()
        model.c.append(self._cdatatype(self._arg()))
        model.c[0] = self._cdatatype(self._arg())
        model.c[0] = model.c[0]
        model.c.append(self._cdatatype(self._arg()))
        with self.assertRaises(ValueError):
            model.c[0] = model.c[1]

    # make sure an existing Data object IS replaced
    # by a call to setitem and not simply updated.
    def test_setitem_exists_overwrite(self):
        model = self.model
        index = range(5)
        model.c = self._ctype(self._cdatatype(self._arg()) for i in index)
        self.assertEqual(len(model.c), len(index))
        for i in index:
            cdata = model.c[i]
            self.assertEqual(id(cdata.parent_component()), id(model.c))
            model.c[i] = self._cdatatype(self._arg())
            self.assertEqual(len(model.c), len(index))
            self.assertNotEqual(id(cdata), id(model.c[i]))
            self.assertEqual(cdata.parent_component(), None)

    def test_delitem(self):
        model = self.model
        index = range(5)
        model.c = self._ctype(self._cdatatype(self._arg()) for i in index)
        self.assertEqual(len(model.c), len(index))
        for i in index:
            cdata = model.c[0]
            self.assertEqual(id(cdata.parent_component()), id(model.c))
            del model.c[0]
            self.assertEqual(len(model.c), len(index)-(i+1))
            self.assertEqual(cdata.parent_component(), None)

    def test_iter(self):
        model = self.model
        index = range(5)
        model.c = self._ctype(self._cdatatype(self._arg()) for i in index)
        self.assertEqual(len(model.c), len(index))
        raw_list = model.c[:]
        self.assertEqual(type(raw_list), list)
        for c1, c2 in zip(raw_list, model.c):
            self.assertEqual(id(c1), id(c2))

    def test_reverse(self):
        model = self.model
        index = range(5)
        model.c = self._ctype(self._cdatatype(self._arg()) for i in index)
        raw_list = model.c[:]
        self.assertEqual(type(raw_list), list)
        model.c.reverse()
        raw_list.reverse()
        for c1, c2 in zip(model.c, raw_list):
            self.assertEqual(id(c1), id(c2))

    def test_remove(self):
        model = self.model
        model = ConcreteModel()
        index = range(5)
        model.c = self._ctype(self._cdatatype(self._arg()) for i in index)
        for i in index:
            cdata = model.c[0]
            self.assertEqual(cdata in model.c, True)
            model.c.remove(cdata)
            self.assertEqual(cdata in model.c, False)

    def test_pop(self):
        model = self.model
        index = range(5)
        model.c = self._ctype(self._cdatatype(self._arg()) for i in index)
        for i in index:
            cdata = model.c[-1]
            self.assertEqual(cdata in model.c, True)
            last = model.c.pop()
            self.assertEqual(cdata in model.c, False)
            self.assertEqual(id(cdata), id(last))

    def test_index(self):
        model = self.model
        index = range(5)
        model.c = self._ctype(self._cdatatype(self._arg()) for i in index)
        for i in index:
            cdata = model.c[i]
            self.assertEqual(model.c.index(cdata), i)
            self.assertEqual(model.c.index(cdata, start=i), i)
            with self.assertRaises(ValueError):
                model.c.index(cdata, start=i+1)
            with self.assertRaises(ValueError):
                model.c.index(cdata, start=i, stop=i)
            with self.assertRaises(ValueError):
                model.c.index(cdata, stop=i)
            self.assertEqual(model.c.index(cdata, start=i, stop=i+1), i)
            with self.assertRaises(ValueError):
                model.c.index(cdata, start=i+1, stop=i+1)
            self.assertEqual(model.c.index(cdata, start=-len(index)+i), i)
            if i == index[-1]:
                self.assertEqual(model.c.index(cdata, start=-len(index)+i+1), i)
            else:
                with self.assertRaises(ValueError):
                    self.assertEqual(model.c.index(cdata, start=-len(index)+i+1), i)
            if i == index[-1]:
                with self.assertRaises(ValueError):
                    self.assertEqual(model.c.index(cdata, stop=-len(index)+i+1), i)
            else:
                self.assertEqual(model.c.index(cdata, stop=-len(index)+i+1), i)
        tmp = self._cdatatype(self._arg())
        with self.assertRaises(ValueError):
            model.c.index(tmp)
        with self.assertRaises(ValueError):
            model.c.index(tmp, stop=len(model.c)+1)

    def test_extend(self):
        model = self.model
        index = range(5)
        model.c = self._ctype(self._cdatatype(self._arg()) for i in index)
        c_more_list = [self._cdatatype(self._arg()) for i in index]
        self.assertEqual(len(model.c), len(index))
        self.assertTrue(len(c_more_list) > 0)
        for cdata in c_more_list:
            self.assertEqual(cdata.parent_component(), None)
        model.c.extend(c_more_list)
        for cdata in c_more_list:
            self.assertEqual(id(cdata.parent_component()),
                             id(model.c))

    def test_count(self):
        model = self.model
        index = range(5)
        model.c = self._ctype(self._cdatatype(self._arg()) for i in index)
        for i in index:
            self.assertEqual(model.c.count(model.c[i]), 1)

    def test_model_clone(self):
        model = self.model
        index = range(5)
        model.c = self._ctype(self._cdatatype(self._arg()) for i in index)
        inst = model.clone()
        self.assertNotEqual(id(inst.c), id(model.c))
        for i in index:
            self.assertNotEqual(id(inst.c[i]), id(model.c[i]))

    def test_name(self):
        model = self.model
        index = range(5)
        model.c = self._ctype(self._cdatatype(self._arg()) for i in index)
        prefix = "c"
        for i in index:
            cdata = model.c[i]
            self.assertEqual(cdata.local_name,
                             cdata.name)
            cname = prefix + "["+str(i)+"]"
            self.assertEqual(cdata.local_name,
                             cname)

class _TestActiveComponentListBase(_TestComponentListBase):

    def test_activate(self):
        model = self.model
        index = list(range(4))
        model.c = self._ctype(self._cdatatype(self._arg())
                              for i in index)
        self.assertEqual(len(model.c), len(index))
        self.assertEqual(model.c.active, True)
        model.c._active = False
        for i in index:
            model.c[i]._active = False
        self.assertEqual(model.c.active, False)
        for i in index:
            self.assertEqual(model.c[i].active, False)
        model.c.activate()
        self.assertEqual(model.c.active, True)

    def test_activate(self):
        model = self.model
        index = list(range(4))
        model.c = self._ctype(self._cdatatype(self._arg())
                              for i in index)
        self.assertEqual(len(model.c), len(index))
        self.assertEqual(model.c.active, True)
        for i in index:
            self.assertEqual(model.c[i].active, True)
        model.c.deactivate()
        self.assertEqual(model.c.active, False)
        for i in index:
            self.assertEqual(model.c[i].active, False)

    # Befault, assigning a new component to a dict container makes it
    # active (unless the default active state for the container
    # datatype is False, which is not the case for any currently
    # existing implementations).
    def test_active(self):
        model = self.model
        model = ConcreteModel()
        model.c = self._ctype()
        self.assertEqual(model.c.active, True)
        model.c.deactivate()
        self.assertEqual(model.c.active, False)
        model.c.append(self._cdatatype(self._arg()))
        self.assertEqual(model.c.active, True)
        model.c.deactivate()
        self.assertEqual(model.c.active, False)
        model.c.insert(0, self._cdatatype(self._arg()))
        self.assertEqual(model.c.active, True)

class TestVarList(_TestComponentListBase,
                  unittest.TestCase):
    _ctype = XVarList
    _cdatatype = _GeneralVarData
    def setUp(self):
        _TestComponentListBase.setUp(self)
        self._arg = lambda: Reals

class TestExpressionList(_TestComponentListBase,
                         unittest.TestCase):
    _ctype = XExpressionList
    _cdatatype = _GeneralExpressionData
    def setUp(self):
        _TestComponentListBase.setUp(self)
        self._arg = lambda: self.model.x**3

#
# Test components that include activate/deactivate
# functionality.
#

class TestConstraintList(_TestActiveComponentListBase,
                         unittest.TestCase):
    _ctype = XConstraintList
    _cdatatype = _GeneralConstraintData
    def setUp(self):
        _TestComponentListBase.setUp(self)
        self._arg = lambda: self.model.x >= 1

class TestObjectiveList(_TestActiveComponentListBase,
                        unittest.TestCase):
    _ctype = XObjectiveList
    _cdatatype = _GeneralObjectiveData
    def setUp(self):
        _TestComponentListBase.setUp(self)
        self._arg = lambda: self.model.x**2

if __name__ == "__main__":
    unittest.main()
