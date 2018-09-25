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
import pyomo.environ as pe
from pyomo.util.components import rename_components

class TestUtilComponents(unittest.TestCase):

    def test_rename_components(self):
        model = pe.ConcreteModel()
        model.x = pe.Var([1, 2, 3], bounds=(-10, 10), initialize=5.0)
        model.z = pe.Var(bounds=(10, 20))
        model.obj = pe.Objective(expr=model.z + model.x[1])

        def con_rule(m, i):
            return m.x[i] + m.z == i
        model.con = pe.Constraint([1, 2, 3], rule=con_rule)
        model.zcon = pe.Constraint(expr=model.z >= model.x[2])
        model.b = pe.Block()
        model.b.bx = pe.Var([1,2,3], initialize=42)
        model.b.bz = pe.Var(initialize=42)

        c_list = list(model.component_objects(ctype=[pe.Var,pe.Constraint,pe.Objective]))
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

if __name__ == '__main__':
    # t = TestUtilComponents()
    # t.test_rename_components()
    unittest.main()
