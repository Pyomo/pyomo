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

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, Block
from pyomo.kernel import block, variable, variable_list, constraint, constraint_dict, parameter, parameter_tuple
from pyomo.util.components import iter_component, rename_components

class TestUtilComponents(unittest.TestCase):

    def test_rename_components(self):
        model =  ConcreteModel()
        model.x =  Var([1, 2, 3], bounds=(-10, 10), initialize=5.0)
        model.z =  Var(bounds=(10, 20))
        model.obj =  Objective(expr=model.z + model.x[1])

        def con_rule(m, i):
            return m.x[i] + m.z == i
        model.con =  Constraint([1, 2, 3], rule=con_rule)
        model.zcon =  Constraint(expr=model.z >= model.x[2])
        model.b =  Block()
        model.b.bx =  Var([1,2,3], initialize=42)
        model.b.bz =  Var(initialize=42)

        c_list = list(model.component_objects(ctype=[ Var, Constraint, Objective]))
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
        model =  ConcreteModel()
        model.x =  Var([1, 2, 3], initialize=0)
        model.z =  Var(initialize=0)

        def con_rule(m, i):
            return m.x[i] + m.z == i

        model.con =  Constraint([1, 2, 3], rule=con_rule)
        model.zcon =  Constraint(expr=model.z >= model.x[2])

        self.assertSameComponents(list(iter_component(model.x)), list(model.x.values()))
        self.assertSameComponents(list(iter_component(model.z)), [model.z[None]])
        self.assertSameComponents(
            list(iter_component(model.con)), list(model.con.values())
        )
        self.assertSameComponents(list(iter_component(model.zcon)), [model.zcon[None]])

    def test_iter_component_kernel(self):
        model =  block()
        model.x =  variable_list( variable(value=0) for _ in [1, 2, 3])
        model.z =  variable(value=0)

        model.con =  constraint_dict(
            (i,  constraint(expr=model.x[i - 1] + model.z == i)) for i in [1, 2, 3]
        )
        model.zcon =  constraint(expr=model.z >= model.x[2])

        model.param_t =  parameter_tuple( parameter(value=36) for _ in [1, 2, 3])
        model.param =  parameter(value=42)

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
