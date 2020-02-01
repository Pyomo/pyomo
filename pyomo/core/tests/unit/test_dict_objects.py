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
from pyomo.core.beta.dict_objects import (VarDict,
                                          ConstraintDict,
                                          ObjectiveDict,
                                          ExpressionDict)
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.base.expression import _GeneralExpressionData

class _TestComponentDictBase(object):

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
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._ctype((i, self._cdatatype(self._arg()))
                              for i in index)
        self.assertEqual(model.c.is_indexed(), True)
        self.assertEqual(model.c.is_constructed(), True)
        with self.assertRaises(TypeError):
            model.d = \
                self._ctype(*tuple((i, self._cdatatype(self._arg()))
                                   for i in index))

    def test_len1(self):
        model = self.model
        model = ConcreteModel()
        model.c = self._ctype()
        self.assertEqual(len(model.c), 0)

    def test_len2(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._ctype((i, self._cdatatype(self._arg()))
                              for i in index)
        self.assertEqual(len(model.c), len(index))

    def test_setitem(self):
        model = self.model
        model = ConcreteModel()
        model.c = self._ctype()
        index = ['a', 1, None, (1,), (1,2)]
        for i in index:
            self.assertTrue(i not in model.c)
        for cnt, i in enumerate(index, 1):
            model.c[i] = self._cdatatype(self._arg())
            self.assertEqual(len(model.c), cnt)
            self.assertTrue(i in model.c)

    # The immediately following this one was originally written when
    # implicit assignment for update was supported. This should be
    # examined more carefully before supporting it.
    # For now just test that implicit assignment raises an exception
    def test_wrong_type_init(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        with self.assertRaises(TypeError):
            model.c = self._ctype((i, self._arg()) for i in index)

    def test_wrong_type_update(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._ctype()
        with self.assertRaises(TypeError):
            model.c.update((i, self._arg()) for i in index)

    def test_wrong_type_setitem(self):
        model = self.model
        model.c = self._ctype()
        with self.assertRaises(TypeError):
            model.c[1] = self._arg()
        model.c[1] = self._cdatatype(self._arg())
        with self.assertRaises(TypeError):
            model.c[1] = self._arg()

    def test_has_parent_init(self):
        model = self.model
        model.c = self._ctype()
        model.c[1] = self._cdatatype(self._arg())
        with self.assertRaises(ValueError):
            model.d = self._ctype(model.c)
        with self.assertRaises(ValueError):
            model.d = self._ctype(dict(model.c))

    def test_has_parent_update(self):
        model = self.model
        model.c = self._ctype()
        model.c[1] = self._cdatatype(self._arg())
        model.c.update(model.c)
        self.assertEqual(len(model.c), 1)
        model.c.update(dict(model.c))
        self.assertEqual(len(model.c), 1)
        model.d = self._ctype()
        with self.assertRaises(ValueError):
            model.d.update(model.c)
        with self.assertRaises(ValueError):
            model.d.update(dict(model.c))

    def test_has_parent_setitem(self):
        model = self.model
        model.c = self._ctype()
        model.c[1] = self._cdatatype(self._arg())
        model.c[1] = model.c[1]
        with self.assertRaises(ValueError):
            model.c[2] = model.c[1]
        model.d = self._ctype()
        with self.assertRaises(ValueError):
            model.d[None] = model.c[1]

    """
    # make sure an existing Data object is NOT replaced
    # by a call to setitem but simply updated.
    def test_setitem_exists(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._ctype((i, self._arg()) for i in index)
        self.assertEqual(len(model.c), len(index))
        for i in index:
            self.assertTrue(i in model.c)
            cdata = model.c[i]
            model.c[i] = self._arg()
            self.assertEqual(len(model.c), len(index))
            self.assertTrue(i in model.c)
            self.assertEqual(id(cdata), id(model.c[i]))
    """

    # make sure an existing Data object IS replaced
    # by a call to setitem and not simply updated.
    def test_setitem_exists_overwrite(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._ctype((i, self._cdatatype(self._arg()))
                              for i in index)
        self.assertEqual(len(model.c), len(index))
        for i in index:
            self.assertTrue(i in model.c)
            cdata = model.c[i]
            model.c[i] = self._cdatatype(self._arg())
            self.assertEqual(len(model.c), len(index))
            self.assertTrue(i in model.c)
            self.assertNotEqual(id(cdata), id(model.c[i]))
            self.assertEqual(cdata.parent_component(), None)

    def test_delitem(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._ctype((i, self._cdatatype(self._arg()))
                              for i in index)
        self.assertEqual(len(model.c), len(index))
        for cnt, i in enumerate(index, 1):
            self.assertTrue(i in model.c)
            cdata = model.c[i]
            self.assertEqual(id(cdata.parent_component()),
                             id(model.c))
            del model.c[i]
            self.assertEqual(len(model.c), len(index)-cnt)
            self.assertTrue(i not in model.c)
            self.assertEqual(cdata.parent_component(), None)

    def test_iter(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._ctype((i, self._cdatatype(self._arg()))
                              for i in index)
        self.assertEqual(len(model.c), len(index))
        comp_index = [i for i in model.c]
        self.assertEqual(len(comp_index), len(index))
        for idx in comp_index:
            self.assertTrue(idx in index)

    def test_model_clone(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._ctype((i, self._cdatatype(self._arg()))
                              for i in index)
        inst = model.clone()
        self.assertNotEqual(id(inst.c), id(model.c))
        for i in index:
            self.assertNotEqual(id(inst.c[i]), id(model.c[i]))

    def test_keys(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        raw_constraint_dict = {i:self._cdatatype(self._arg()) for i in index}
        model.c = self._ctype(raw_constraint_dict)
        self.assertEqual(sorted(list(raw_constraint_dict.keys()), key=str),
                         sorted(list(model.c.keys()), key=str))

    def test_values(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        raw_constraint_dict = {i:self._cdatatype(self._arg()) for i in index}
        model.c = self._ctype(raw_constraint_dict)
        self.assertEqual(
            sorted(list(id(_v)
                        for _v in raw_constraint_dict.values()),
                   key=str),
            sorted(list(id(_v)
                        for _v in model.c.values()),
                   key=str))

    def test_items(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        raw_constraint_dict = {i:self._cdatatype(self._arg()) for i in index}
        model.c = self._ctype(raw_constraint_dict)
        self.assertEqual(
            sorted(list((_i, id(_v))
                        for _i,_v in raw_constraint_dict.items()),
                   key=str),
            sorted(list((_i, id(_v))
                        for _i,_v in model.c.items()),
                   key=str))

    def test_update(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        raw_constraint_dict = {i:self._cdatatype(self._arg()) for i in index}
        model.c = self._ctype()
        model.c.update(raw_constraint_dict)
        self.assertEqual(sorted(list(raw_constraint_dict.keys()), key=str),
                         sorted(list(model.c.keys()), key=str))

    def test_name(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._ctype((i, self._cdatatype(self._arg()))
                              for i in index)
        index_to_string = {}
        index_to_string['a'] = '[a]'
        index_to_string[1] = '[1]'
        index_to_string[None] = '[None]'
        # I don't like that (1,) looks the same as 1, but oh well
        index_to_string[(1,)] = '[1]'
        index_to_string[(1,2)] = '[1,2]'
        prefix = "c"
        for i in index:
            cdata = model.c[i]
            self.assertEqual(cdata.local_name,
                             cdata.name)
            cname = prefix + index_to_string[i]
            self.assertEqual(cdata.local_name,
                             cname)

    def test_clear(self):
        model = self.model
        model.c = self._ctype()
        model.c[1] = self._cdatatype(self._arg())
        c1 = model.c[1]
        with self.assertRaises(ValueError):
            model.c[None] = c1
        model.d = self._ctype()
        with self.assertRaises(ValueError):
            model.d[1] = c1
        model.c.clear()
        model.d[1] = c1

    def test_eq(self):
        model = self.model
        model.c = self._ctype()
        model.c[1] = self._cdatatype(self._arg())
        model.d = self._ctype()
        model.d[1] = self._cdatatype(self._arg())

        self.assertNotEqual(model.c, [])
        self.assertFalse(model.c == [])

        self.assertTrue(model.c == model.c)
        self.assertEqual(model.c, model.c)
        self.assertTrue(model.c == dict(model.c))
        self.assertEqual(model.c, dict(model.c))
        self.assertTrue(dict(model.c) == model.c)
        self.assertEqual(dict(model.c), model.c)

        self.assertFalse(model.d == model.c)
        self.assertTrue(model.d != model.c)
        self.assertNotEqual(model.d, model.c)

        self.assertFalse(model.c == model.d)
        self.assertTrue(model.c != model.d)
        self.assertNotEqual(model.c, model.d)

class _TestActiveComponentDictBase(_TestComponentDictBase):

    def test_activate(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._ctype((i, self._cdatatype(self._arg()))
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
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._ctype((i, self._cdatatype(self._arg()))
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
        model.c = self._ctype()
        self.assertEqual(model.c.active, True)
        model.c.deactivate()
        self.assertEqual(model.c.active, False)
        model.c[1] = self._cdatatype(self._arg())
        self.assertEqual(model.c.active, True)

class TestVarDict(_TestComponentDictBase,
                  unittest.TestCase):
    _ctype = VarDict
    _cdatatype = _GeneralVarData
    def setUp(self):
        _TestComponentDictBase.setUp(self)
        self._arg = lambda: Reals

class TestExpressionDict(_TestComponentDictBase,
                         unittest.TestCase):
    _ctype = ExpressionDict
    _cdatatype = _GeneralExpressionData
    def setUp(self):
        _TestComponentDictBase.setUp(self)
        self._arg = lambda: self.model.x**3

#
# Test components that include activate/deactivate
# functionality.
#

class TestConstraintDict(_TestActiveComponentDictBase,
                         unittest.TestCase):
    _ctype = ConstraintDict
    _cdatatype = _GeneralConstraintData
    def setUp(self):
        _TestComponentDictBase.setUp(self)
        self._arg = lambda: self.model.x >= 1

class TestObjectiveDict(_TestActiveComponentDictBase,
                        unittest.TestCase):
    _ctype = ObjectiveDict
    _cdatatype = _GeneralObjectiveData
    def setUp(self):
        _TestComponentDictBase.setUp(self)
        self._arg = lambda: self.model.x**2

if __name__ == "__main__":
    unittest.main()
