##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# This software is distributed under the 3-clause BSD License.
##############################################################################
"""
Test data model tree for QTreeView.  These tests need PyQt.
"""

import pyutilib.th as unittest
from pyomo.environ import *
from pyomo.contrib.viewer.model_browser import ComponentDataModel
try:
    no_pyqt = False
    from pyomo.contrib.viewer.ui_data import UIData
    import pyomo.contrib.viewer.ui as ui
except:
    no_pyqt = True
    class UIData(object):
        model = None
        def __init__(*args, **kwargs):
            pass


@unittest.skipIf(no_pyqt, "PyQt needed to test tree data model")
class TestDataModel(unittest.TestCase):
    def setUp(self):
        # Borrowed this test model from the trust region tests
        m = ConcreteModel(name="tm")
        m.z = Var(range(3), domain=Reals, initialize=2.)
        m.x = Var(range(2), initialize=2.)
        m.x[1] = 1.0

        m.b1 = Block()
        m.b1.e1 = Expression(expr=m.x[0] + m.x[1])

        def blackbox(a,b):
            return sin(a-b)
        self.bb = ExternalFunction(blackbox)

        m.obj = Objective(
            expr=(m.z[0]-1.0)**2 + (m.z[0]-m.z[1])**2 + (m.z[2]-1.0)**2 \
                + (m.x[0]-1.0)**4 + (m.x[1]-1.0)**6 # + m.bb(m.x[0],m.x[1])
            )
        m.c1 = Constraint(expr=m.x[0] * m.z[0]**2 + self.bb(m.x[0],m.x[1]) == 2*sqrt(2.0))
        m.c2 = Constraint(expr=m.z[2]**4 * m.z[1]**2 + m.z[1] == 8+sqrt(2.0))

        self.m = m.clone()

    def test_create_tree_var(self):
        ui_data = UIData(model=self.m)
        # Defaults to variables and two columns name and value
        data_model = ComponentDataModel(parent=None, ui_data=ui_data)
        # There should be one root item
        assert(len(data_model.rootItems)==1)
        assert(data_model.rootItems[0].data==self.m)
        # The children should be in the model construction order,
        # and the indexes are sorted
        children = data_model.rootItems[0].children
        assert(children[0].data == self.m.z)
        assert(children[1].data == self.m.x)
        assert(children[2].data == self.m.b1)
        # Check the data display role The rows in the tree should be:
        #   0. Model
        #     0. z
        #       0. z[0], 2
        #       1. z[1], 2
        #       2. z[2], 2
        #    1. x
        #       0. x[0], 2
        #       1. x[1], 1
        #    2. b1
        root_index = data_model.index(0,0)
        assert(data_model.data(root_index)=="tm")
        zidx = data_model.index(0,0,parent=root_index)
        assert(data_model.data(zidx)=="z")
        xidx = data_model.index(1,0,parent=root_index)
        assert(data_model.data(xidx)=="x")
        b1idx = data_model.index(2,0,parent=root_index)
        assert(data_model.data(b1idx)=="b1")
        idx = data_model.index(0,0,parent=zidx)
        assert(data_model.data(idx)=="z[0]")
        idx = data_model.index(0,1,parent=zidx)
        assert(abs(data_model.data(idx) - 2.0) < 0.0001)
        idx = data_model.index(1,0,parent=zidx)
        assert(data_model.data(idx)=="z[1]")
        idx = data_model.index(1,1,parent=zidx)
        assert(abs(data_model.data(idx) - 2.0) < 0.0001)

    def test_create_tree_con(self):
        ui_data = UIData(model=self.m)
        # Make a tree with constraints
        data_model = ComponentDataModel(parent=None, ui_data=ui_data,
                                        components=(Constraint,),
                                        columns=["name", "active"])
        # There should be one root item
        assert(len(data_model.rootItems)==1)
        assert(data_model.rootItems[0].data==self.m)
        # The children should be in the model construction order,
        # and the indexes are sorted
        children = data_model.rootItems[0].children
        assert(children[0].data == self.m.b1)
        assert(children[1].data == self.m.c1)
        assert(children[2].data == self.m.c2)
        # Check the data display role The rows in the tree should be:
        #   0. Model
        #     0. c1, True
        #     1. c2, True
        #     2. b1, True
        root_index = data_model.index(0,0)
        assert(data_model.data(root_index)=="tm")
        idx = data_model.index(0,0,parent=root_index)
        assert(data_model.data(idx)=="b1")
        idx = data_model.index(0,1,parent=root_index)
        assert(data_model.data(idx)==True)
        idx = data_model.index(1,0,parent=root_index)
        assert(data_model.data(idx)=="c1")
        idx = data_model.index(2,0,parent=root_index)
        assert(data_model.data(idx)=="c2")

    def test_create_tree_expr(self):
        ui_data = UIData(model=self.m)
        # Make a tree with constraints
        data_model = ComponentDataModel(parent=None, ui_data=ui_data,
                                        components=(Expression,),
                                        columns=["name", "value"])
        # There should be one root item
        assert(len(data_model.rootItems)==1)
        assert(data_model.rootItems[0].data==self.m)
        # The children should be in the model construction order,
        # and the indexes are sorted
        children = data_model.rootItems[0].children
        assert(children[0].data == self.m.b1)
        assert(children[0].children[0].data == self.m.b1.e1)
        ui_data.calculate_expressions()
        # Check the data display role The rows in the tree should be:
        #   0. Model
        #     0. b1,
        #       0. e1, value
        root_index = data_model.index(0,0)
        b1_index = data_model.index(0,0,parent=root_index)
        e1_index0 = data_model.index(0,0,parent=b1_index)
        e1_index1 = data_model.index(0,1,parent=b1_index)
        assert(data_model.data(e1_index0)=="b1.e1")
        assert(abs(data_model.data(e1_index1) - 3.0) < 0.0001)

    def test_update_tree_expr(self):
        ui_data = UIData(model=self.m)
        # Make a tree with constraints
        data_model = ComponentDataModel(parent=None, ui_data=ui_data,
                                        components=(Expression,),
                                        columns=["name", "value"])

        self.m.newe = Expression(expr=self.m.x[0] + self.m.x[1])

        data_model._update_tree()

        # There should be one root item
        assert(len(data_model.rootItems)==1)
        assert(data_model.rootItems[0].data==self.m)
        # The children should be in the model construction order,
        # and the indexes are sorted
        children = data_model.rootItems[0].children
        assert(children[0].data == self.m.b1)
        assert(children[0].children[0].data == self.m.b1.e1)
        ui_data.calculate_expressions()
        # Check the data display role The rows in the tree should be:
        #   0. Model
        #     0. b1,
        #       0. e1, value
        root_index = data_model.index(0,0)
        b1_index = data_model.index(0,0,parent=root_index)
        e1_index0 = data_model.index(0,0,parent=b1_index)
        e1_index1 = data_model.index(0,1,parent=b1_index)
        assert(data_model.data(e1_index0)=="b1.e1")
        assert(abs(data_model.data(e1_index1) - 3.0) < 0.0001)
        # Check that in the update the new expression was added
        found = False
        for i in children:
            if id(i.data) == id(self.m.newe):
                found = True
                break
        assert(found)
