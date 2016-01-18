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

class TestComponentDict(unittest.TestCase):

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
        index = ['a', 1, None, (1,), (1,2)]
        model.c = ConstraintDict((i, _GeneralConstraintData(model.x >= 1)) for i in index)
        model.e = ExpressionDict((i, _GeneralExpressionData(model.x**2)) for i in index)
        model.o = ObjectiveDict((i, _GeneralObjectiveData(model.x)) for i in index)
        self.assertEqual(model.c.is_indexed(), True)
        self.assertEqual(model.e.is_indexed(), True)
        self.assertEqual(model.o.is_indexed(), True)
        self.assertEqual(model.c.is_constructed(), True)
        self.assertEqual(model.e.is_constructed(), True)
        self.assertEqual(model.o.is_constructed(), True)

    def test_len1(self):
        model = ConcreteModel()
        model.c = ConstraintDict()
        model.e = ExpressionDict()
        model.o = ObjectiveDict()
        self.assertEqual(len(model.c), 0)
        self.assertEqual(len(model.e), 0)
        self.assertEqual(len(model.o), 0)

    def test_len2(self):
        model = ConcreteModel()
        model.x = Var()
        index = ['a', 1, None, (1,), (1,2)]
        model.c = ConstraintDict((i, _GeneralConstraintData(model.x >= 1)) for i in index)
        model.e = ExpressionDict((i, _GeneralExpressionData(model.x**2)) for i in index)
        model.o = ObjectiveDict((i, _GeneralObjectiveData(model.x)) for i in index)
        self.assertEqual(len(model.c), len(index))
        self.assertEqual(len(model.e), len(index))
        self.assertEqual(len(model.o), len(index))

    def test_setitem(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = ConstraintDict()
        model.e = ExpressionDict()
        model.o = ObjectiveDict()
        index = ['a', 1, None, (1,), (1,2)]
        for i in index:
            self.assertTrue(i not in model.c)
            self.assertTrue(i not in model.e)
            self.assertTrue(i not in model.o)
        for cnt, i in enumerate(index, 1):
            model.c[i] = _GeneralConstraintData(model.x+5 >= 1)
            model.e[i] = _GeneralExpressionData(1)
            model.o[i] = _GeneralObjectiveData(model.x**2)
            self.assertEqual(len(model.c), cnt)
            self.assertEqual(len(model.e), cnt)
            self.assertEqual(len(model.o), cnt)
            self.assertTrue(i in model.c)
            self.assertTrue(i in model.e)
            self.assertTrue(i in model.o)

    # The immediately following this one was originally written when
    # implicit assignment for update was supported. This should be
    # examined more carefully before supporting it.
    # For now just test that implicit assignment raises an exception
    def test_setitem_implicit_failure(self):
        model = ConcreteModel()
        model.x = Var()
        index = ['a', 1, None, (1,), (1,2)]
        try:
            model.c = ConstraintDict((i, model.x+5 >= 1) for i in index)
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")
        try:
            model.e = ExpressionDict((i, 1) for i in index)
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")
        try:
            model.o = ObjectiveDict((i, model.x**2) for i in index)
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")

        model.c = ConstraintDict()
        model.e = ExpressionDict()
        model.o = ObjectiveDict()
        try:
            model.c[1] = model.x+5 >= 1
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")
        try:
            model.e[1] = 1
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")
        try:
            model.o[1] = model.x**2
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")

        model.c[1] = _GeneralConstraintData(model.x+5 >= 1)
        model.e[1] = _GeneralExpressionData(1)
        model.o[1] = _GeneralObjectiveData(model.x**2)
        try:
            model.c[1] = model.x+5 >= 1
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")
        try:
            model.e[1] = 1
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")
        try:
            model.o[1] = model.x**2
        except TypeError:
            pass
        else:
            self.fail("Expected TypeError")
    """
    # make sure an existing Data object is NOT replaced
    # by a call to setitem but simply updated.
    def test_setitem_exists(self):
        model = ConcreteModel()
        model.x = Var()
        index = ['a', 1, None, (1,), (1,2)]
        model.c = ConstraintDict((i, model.x+5 >= 1) for i in index)
        model.e = ExpressionDict((i, 1) for i in index)
        model.o = ObjectiveDict((i, model.x**2) for i in index)
        self.assertEqual(len(model.c), len(index))
        self.assertEqual(len(model.e), len(index))
        self.assertEqual(len(model.o), len(index))
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
            self.assertEqual(len(model.c), len(index))
            self.assertEqual(len(model.e), len(index))
            self.assertEqual(len(model.o), len(index))
            self.assertTrue(i in model.c)
            self.assertTrue(i in model.e)
            self.assertTrue(i in model.o)
            self.assertEqual(id(cdata), id(model.c[i]))
            self.assertEqual(id(edata), id(model.e[i]))
            self.assertEqual(id(odata), id(model.o[i]))
    """

    # make sure an existing Data object IS replaced
    # by a call to setitem and not simply updated.
    def test_setitem_exists_overwrite(self):
        model = ConcreteModel()
        model.x = Var()
        index = ['a', 1, None, (1,), (1,2)]
        model.c = ConstraintDict((i, _GeneralConstraintData(model.x+5 >= 1)) for i in index)
        model.e = ExpressionDict((i, _GeneralExpressionData(1)) for i in index)
        model.o = ObjectiveDict((i, _GeneralObjectiveData(model.x**2)) for i in index)
        self.assertEqual(len(model.c), len(index))
        self.assertEqual(len(model.e), len(index))
        self.assertEqual(len(model.o), len(index))
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
            self.assertEqual(len(model.c), len(index))
            self.assertEqual(len(model.e), len(index))
            self.assertEqual(len(model.o), len(index))
            self.assertTrue(i in model.c)
            self.assertTrue(i in model.e)
            self.assertTrue(i in model.o)
            self.assertNotEqual(id(cdata), id(model.c[i]))
            self.assertNotEqual(id(edata), id(model.e[i]))
            self.assertNotEqual(id(odata), id(model.o[i]))
            self.assertEqual(cdata.parent_component(), None)
            self.assertEqual(edata.parent_component(), None)
            self.assertEqual(odata.parent_component(), None)

    def test_delitem(self):
        model = ConcreteModel()
        model.x = Var()
        index = ['a', 1, None, (1,), (1,2)]
        model.c = ConstraintDict((i, _GeneralConstraintData(model.x+5 >= 1)) for i in index)
        model.e = ExpressionDict((i, _GeneralExpressionData(1)) for i in index)
        model.o = ObjectiveDict((i, _GeneralObjectiveData(model.x**2)) for i in index)
        self.assertEqual(len(model.c), len(index))
        self.assertEqual(len(model.e), len(index))
        self.assertEqual(len(model.o), len(index))
        for cnt, i in enumerate(index, 1):
            self.assertTrue(i in model.c)
            self.assertTrue(i in model.e)
            self.assertTrue(i in model.o)
            cdata = model.c[i]
            edata = model.e[i]
            odata = model.o[i]
            self.assertEqual(id(cdata.parent_component()), id(model.c))
            self.assertEqual(id(edata.parent_component()), id(model.e))
            self.assertEqual(id(odata.parent_component()), id(model.o))
            del model.c[i]
            del model.e[i]
            del model.o[i]
            self.assertEqual(len(model.c), len(index)-cnt)
            self.assertEqual(len(model.e), len(index)-cnt)
            self.assertEqual(len(model.o), len(index)-cnt)
            self.assertTrue(i not in model.c)
            self.assertTrue(i not in model.e)
            self.assertTrue(i not in model.o)
            self.assertEqual(cdata.parent_component(), None)
            self.assertEqual(edata.parent_component(), None)
            self.assertEqual(odata.parent_component(), None)

    def test_iter(self):
        model = ConcreteModel()
        model.x = Var()
        index = ['a', 1, None, (1,), (1,2)]
        model.c = ConstraintDict((i, _GeneralConstraintData(model.x+5 >= 1)) for i in index)
        model.e = ExpressionDict((i, _GeneralExpressionData(1)) for i in index)
        model.o = ObjectiveDict((i, _GeneralObjectiveData(model.x**2)) for i in index)
        self.assertEqual(len(model.c), len(index))
        self.assertEqual(len(model.e), len(index))
        self.assertEqual(len(model.o), len(index))
        for comp in [model.c, model.e, model.o]:
            comp_index = [i for i in comp]
            self.assertEqual(len(comp_index), len(index))
            self.assertTrue(comp_index[0] in index)
            self.assertTrue(comp_index[1] in index)
            self.assertTrue(comp_index[2] in index)

    def test_model_clone(self):
        model = ConcreteModel()
        index = ['a', 1, None, (1,), (1,2)]
        model.x = Var()
        model.c = ConstraintDict((i, _GeneralConstraintData(model.x >= 1)) for i in index)
        model.e = ExpressionDict((i, _GeneralExpressionData(model.x**2)) for i in index)
        model.o = ObjectiveDict((i, _GeneralObjectiveData(model.x)) for i in index)
        inst = model.clone()
        self.assertNotEqual(id(inst.x), id(model.x))
        self.assertNotEqual(id(inst.c), id(model.c))
        self.assertNotEqual(id(inst.e), id(model.e))
        self.assertNotEqual(id(inst.o), id(model.o))
        for i in index:
            self.assertNotEqual(id(inst.c[i]), id(model.c[i]))
            self.assertNotEqual(id(inst.e[i]), id(model.e[i]))
            self.assertNotEqual(id(inst.o[i]), id(model.o[i]))

    def test_keys(self):
        model = ConcreteModel()
        index = ['a', 1, None, (1,), (1,2)]
        model.x = Var()
        raw_constraint_dict = dict((i, _GeneralConstraintData(model.x >= 1)) for i in index)
        model.c = ConstraintDict(raw_constraint_dict)
        raw_expression_dict = dict((i, _GeneralExpressionData(model.x**2)) for i in index)
        model.e = ExpressionDict(raw_expression_dict)
        raw_objective_dict = dict((i, _GeneralObjectiveData(model.x)) for i in index)
        model.o = ObjectiveDict(raw_objective_dict)
        self.assertEqual(sorted(list(raw_constraint_dict.keys()), key=str),
                         sorted(list(model.c.keys()), key=str)
                        )
        self.assertEqual(sorted(list(raw_expression_dict.keys()), key=str),
                         sorted(list(model.e.keys()), key=str))
        self.assertEqual(sorted(list(raw_objective_dict.keys()), key=str),
                         sorted(list(model.o.keys()), key=str))

    def test_values(self):
        model = ConcreteModel()
        index = ['a', 1, None, (1,), (1,2)]
        model.x = Var()
        raw_constraint_dict = dict((i, _GeneralConstraintData(model.x >= 1)) for i in index)
        model.c = ConstraintDict(raw_constraint_dict)
        raw_expression_dict = dict((i, _GeneralExpressionData(model.x**2)) for i in index)
        model.e = ExpressionDict(raw_expression_dict)
        raw_objective_dict = dict((i, _GeneralObjectiveData(model.x)) for i in index)
        model.o = ObjectiveDict(raw_objective_dict)
        self.assertEqual(sorted(list(id(_v) for _v in raw_constraint_dict.values()), key=str),
                         sorted(list(id(_v) for _v in model.c.values()), key=str))
        self.assertEqual(sorted(list(id(_v) for _v in raw_expression_dict.values()), key=str),
                         sorted(list(id(_v) for _v in model.e.values()),key=str))
        self.assertEqual(sorted(list(id(_v) for _v in raw_objective_dict.values()), key=str),
                         sorted(list(id(_v) for _v in model.o.values()), key=str))

    def test_items(self):
        model = ConcreteModel()
        index = ['a', 1, None, (1,), (1,2)]
        model.x = Var()
        raw_constraint_dict = dict((i, _GeneralConstraintData(model.x >= 1)) for i in index)
        model.c = ConstraintDict(raw_constraint_dict)
        raw_expression_dict = dict((i, _GeneralExpressionData(model.x**2)) for i in index)
        model.e = ExpressionDict(raw_expression_dict)
        raw_objective_dict = dict((i, _GeneralObjectiveData(model.x)) for i in index)
        model.o = ObjectiveDict(raw_objective_dict)
        self.assertEqual(sorted(list((_i, id(_v)) for _i,_v in raw_constraint_dict.items()), key=str),
                         sorted(list((_i, id(_v)) for _i,_v in model.c.items()), key=str))
        self.assertEqual(sorted(list((_i, id(_v)) for _i,_v in raw_expression_dict.items()), key=str),
                         sorted(list((_i, id(_v)) for _i,_v in model.e.items()), key=str))
        self.assertEqual(sorted(list((_i, id(_v)) for _i,_v in raw_objective_dict.items()), key=str),
                         sorted(list((_i, id(_v)) for _i,_v in model.o.items()), key=str))

    def test_update(self):
        model = ConcreteModel()
        index = ['a', 1, None, (1,), (1,2)]
        model.x = Var()
        raw_constraint_dict = dict((i, _GeneralConstraintData(model.x >= 1)) for i in index)
        model.c = ConstraintDict()
        model.c.update(raw_constraint_dict)
        raw_expression_dict = dict((i, _GeneralExpressionData(model.x**2)) for i in index)
        model.e = ExpressionDict()
        model.e.update(raw_expression_dict)
        raw_objective_dict = dict((i, _GeneralObjectiveData(model.x)) for i in index)
        model.o = ObjectiveDict()
        model.o.update(raw_objective_dict)
        self.assertEqual(sorted(list(raw_constraint_dict.keys()), key=str),
                         sorted(list(model.c.keys()), key=str))
        self.assertEqual(sorted(list(raw_expression_dict.keys()), key=str),
                         sorted(list(model.e.keys()), key=str))
        self.assertEqual(sorted(list(raw_objective_dict.keys()), key=str),
                         sorted(list(model.o.keys()), key=str))

    # Befault, assigning a new component to a dict container makes it
    # active (unless the default active state for the container
    # datatype is False, which is not the case for any currently
    # existing implementations).
    def test_active(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = ConstraintDict()
        self.assertEqual(model.c.active, True)
        model.c.deactivate()
        self.assertEqual(model.c.active, False)
        model.c[1] = _GeneralConstraintData(model.x >= 1)
        self.assertEqual(model.c.active, True)

        model.e = ExpressionDict()
        self.assertEqual(model.e.active, True)
        model.e.deactivate()
        self.assertEqual(model.e.active, False)
        model.e[1] = _GeneralExpressionData(model.x**2)
        self.assertEqual(model.e.active, True)

        model.o = ObjectiveDict()
        self.assertEqual(model.o.active, True)
        model.o.deactivate()
        self.assertEqual(model.o.active, False)
        model.o[1] = _GeneralObjectiveData(model.x)
        self.assertEqual(model.o.active, True)

    def test_cname(self):
        model = ConcreteModel()
        index = ['a', 1, None, (1,), (1,2)]
        model.x = Var()
        model.c = ConstraintDict((i, _GeneralConstraintData(model.x >= 1)) for i in index)
        model.e = ExpressionDict((i, _GeneralExpressionData(model.x**2)) for i in index)
        model.o = ObjectiveDict((i, _GeneralObjectiveData(model.x)) for i in index)
        index_to_string = {}
        index_to_string['a'] = '[a]'
        index_to_string[1] = '[1]'
        index_to_string[None] = '[None]'
        # I don't like that (1,) looks the same as 1, but oh well
        index_to_string[(1,)] = '[1]'
        index_to_string[(1,2)] = '[1,2]'
        for comp in [model.c, model.e, model.o]:
            for i in index:
                compdata = comp[i]
                self.assertEqual(compdata.cname(False),
                                 compdata.cname(True))
                if isinstance(comp, ConstraintDict):
                    prefix = "c"
                elif isinstance(comp, ExpressionDict):
                    prefix = "e"
                elif isinstance(comp, ObjectiveDict):
                    prefix = "o"
                else:
                    assert False
                cname = prefix + index_to_string[i]
                self.assertEqual(compdata.cname(False),
                                 cname)

if __name__ == "__main__":
    unittest.main()

