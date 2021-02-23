#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from six.moves import zip_longest

import pyutilib.th as unittest

import pyomo.environ as pyo
import pyomo.kernel as pmo
from pyomo.util.components import iter_component, rename_components

class TestUtilComponents(unittest.TestCase):

    def test_rename_components(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var([1, 2, 3], bounds=(-10, 10), initialize=5.0)
        model.z = pyo.Var(bounds=(10, 20))
        model.obj = pyo.Objective(expr=model.z + model.x[1])

        def con_rule(m, i):
            return m.x[i] + m.z == i
        model.con = pyo.Constraint([1, 2, 3], rule=con_rule)
        model.zcon = pyo.Constraint(expr=model.z >= model.x[2])
        model.b = pyo.Block()
        model.b.bx = pyo.Var([1,2,3], initialize=42)
        model.b.bz = pyo.Var(initialize=42)

        c_list = list(model.component_objects(ctype=[pyo.Var,pyo.Constraint,pyo.Objective]))
        name_map = rename_components(model=model,
                                     component_list=c_list,
                                     prefix='scaled_')

        self.assertEquals(name_map[model.scaled_obj], 'obj')
        self.assertEquals(name_map[model.scaled_x], 'x')
        self.assertEquals(name_map[model.scaled_con], 'con')
        self.assertEquals(name_map[model.scaled_zcon], 'zcon')
        self.assertEquals(name_map[model.b.scaled_bz], 'b.bz')

        self.assertEquals(model.scaled_obj.name, 'scaled_obj')
        self.assertEquals(model.scaled_x.name, 'scaled_x')
        self.assertEquals(model.scaled_con.name, 'scaled_con')
        self.assertEquals(model.scaled_zcon.name, 'scaled_zcon')
        self.assertEquals(model.b.name, 'b')
        self.assertEquals(model.b.scaled_bz.name, 'b.scaled_bz')

    def assertSameComponents(self, obj, other_obj):
        for i, j in zip_longest(obj, other_obj):
            self.assertEqual(id(i), id(j))

    def test_iter_component_base(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var([1, 2, 3], initialize=0)
        model.z = pyo.Var(initialize=0)

        def con_rule(m, i):
            return m.x[i] + m.z == i

        model.con = pyo.Constraint([1, 2, 3], rule=con_rule)
        model.zcon = pyo.Constraint(expr=model.z >= model.x[2])

        self.assertSameComponents(list(iter_component(model.x)), list(model.x.values()))
        self.assertSameComponents(list(iter_component(model.z)), [model.z[None]])
        self.assertSameComponents(
            list(iter_component(model.con)), list(model.con.values())
        )
        self.assertSameComponents(list(iter_component(model.zcon)), [model.zcon[None]])

    def test_iter_component_kernel(self):
        model = pmo.block()
        model.x = pmo.variable_list(pmo.variable(value=0) for _ in [1, 2, 3])
        model.z = pmo.variable(value=0)

        model.con = pmo.constraint_dict(
            (i, pmo.constraint(expr=model.x[i - 1] + model.z == i)) for i in [1, 2, 3]
        )
        model.zcon = pmo.constraint(expr=model.z >= model.x[2])

        model.param_t = pmo.parameter_tuple(pmo.parameter(value=36) for _ in [1, 2, 3])
        model.param = pmo.parameter(value=42)

        self.assertSameComponents(list(iter_component(model.x)), list(model.x))
        self.assertSameComponents(list(iter_component(model.z)), [model.z])
        self.assertSameComponents(
            list(iter_component(model.con)), list(model.con.values())
        )
        self.assertSameComponents(list(iter_component(model.zcon)), [model.zcon])
        self.assertSameComponents(
            list(iter_component(model.param_t)), list(model.param_t)
        )
        self.assertSameComponents(list(iter_component(model.param)), [model.param])


if __name__ == '__main__':
    # t = TestUtilComponents()
    # t.test_rename_components()
    unittest.main()
