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
import pyomo.kernel as pmo
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.variable import IVariable
from pyomo.core.kernel.constraint import IConstraint


class IJunk(IBlock):
    __slots__ = ()
class junk(pmo.block):
    _ctype = IJunk
class junk_list(pmo.block_list):
    __slots__ = ()
    _ctype = IJunk

_model = pmo.block()
_model.v = pmo.variable()
_model.V = pmo.variable_list()
_model.V.append(pmo.variable())
_model.V.append(pmo.variable_list())
_model.V[1].append(pmo.variable())
_model.c = pmo.constraint()
_model.C = pmo.constraint_list()
_model.C.append(pmo.constraint())
_model.C.append(pmo.constraint_list())
_model.C[1].append(pmo.constraint())
b_clone = _model.clone()
_model.b = b_clone.clone()
_model.B = pmo.block_list()
_model.B.append(b_clone.clone())
_model.B.append(pmo.block_list())
_model.B[1].append(b_clone.clone())
del b_clone
_model.j = junk()
_model.J = junk_list()
_model.J.append(junk())
_model.J.append(junk_list())
_model.J[1].append(junk())
_model.J[1][0].b = pmo.block()
_model.J[1][0].b.v = pmo.variable()
model_clone = _model.clone()
_model.k = pmo.block()
_model.K = pmo.block_list()
_model.K.append(model_clone.clone())
del model_clone

class Test_kernel(unittest.TestCase):

    def test_no_ctype_collisions(self):
        hash_set = set()
        hash_list = list()
        for cls in [pmo.variable,
                    pmo.constraint,
                    pmo.objective,
                    pmo.expression,
                    pmo.parameter,
                    pmo.suffix,
                    pmo.sos,
                    pmo.block]:
            ctype = cls._ctype
            hash_set.add(hash(ctype))
            hash_list.append(hash(ctype))
        self.assertEqual(len(hash_set),
                         len(hash_list))

    def test_component_data_objects_hack(self):
        model = _model.clone()
        self.assertEqual(
            [str(obj) for obj in model.component_data_objects()],
            [str(obj) for obj in model.components()])
        self.assertEqual(
            [str(obj) for obj in model.component_data_objects(ctype=IVariable)],
            [str(obj) for obj in model.components(ctype=IVariable)])
        self.assertEqual(
            [str(obj) for obj in model.component_data_objects(ctype=IConstraint)],
            [str(obj) for obj in model.components(ctype=IConstraint)])
        self.assertEqual(
            [str(obj) for obj in model.component_data_objects(ctype=IBlock)],
            [str(obj) for obj in model.components(ctype=IBlock)])
        self.assertEqual(
            [str(obj) for obj in model.component_data_objects(ctype=IJunk)],
            [str(obj) for obj in model.components(ctype=IJunk)])
        for item in pmo.preorder_traversal(model):
            item.deactivate()
            self.assertEqual(
                [str(obj) for obj in model.component_data_objects(active=True)],
                [str(obj) for obj in model.components(active=True)])
            self.assertEqual(
                [str(obj) for obj in model.component_data_objects(ctype=IVariable, active=True)],
                [str(obj) for obj in model.components(ctype=IVariable, active=True)])
            self.assertEqual(
                [str(obj) for obj in model.component_data_objects(ctype=IConstraint, active=True)],
                [str(obj) for obj in model.components(ctype=IConstraint, active=True)])
            self.assertEqual(
                [str(obj) for obj in model.component_data_objects(ctype=IBlock, active=True)],
                [str(obj) for obj in model.components(ctype=IBlock, active=True)])
            self.assertEqual(
                [str(obj) for obj in model.component_data_objects(ctype=IJunk, active=True)],
                [str(obj) for obj in model.components(ctype=IJunk, active=True)])
            item.activate()

    def test_component_objects_hack(self):
        model = _model.clone()
        objs = {key: [] for key in
                [None, IVariable, IConstraint, IBlock, IJunk]}
        for item in pmo.heterogeneous_containers(model):
            objs[None].extend(item.component_objects(descend_into=False))
            self.assertEqual(
                [str(obj) for obj in item.component_objects(descend_into=False)],
                [str(obj) for obj in item.children()])
            objs[IVariable].extend(item.component_objects(ctype=IVariable,
                                                          descend_into=False))
            self.assertEqual(
                [str(obj) for obj in item.component_objects(ctype=IVariable,
                                                            descend_into=False)],
                [str(obj) for obj in item.children(ctype=IVariable)])
            objs[IConstraint].extend(item.component_objects(ctype=IConstraint,
                                                            descend_into=False))
            self.assertEqual(
                [str(obj) for obj in item.component_objects(ctype=IConstraint,
                                                            descend_into=False)],
                [str(obj) for obj in item.children(ctype=IConstraint)])
            objs[IBlock].extend(item.component_objects(ctype=IBlock,
                                                       descend_into=False))
            self.assertEqual(
                [str(obj) for obj in item.component_objects(ctype=IBlock,
                                                            descend_into=False)],
                [str(obj) for obj in item.children(ctype=IBlock)])
            objs[IJunk].extend(item.component_objects(ctype=IJunk,
                                                      descend_into=False))
            self.assertEqual(
                [str(obj) for obj in item.component_objects(ctype=IJunk,
                                                            descend_into=False)],
                [str(obj) for obj in item.children(ctype=IJunk)])
        all_ = []
        for key in objs:
            if key is None:
                continue
            names = [str(obj) for obj in objs[key]]
            self.assertEqual(
                sorted([str(obj) for obj in model.component_objects(ctype=key)]),
                sorted(names))
            all_.extend(names)
        self.assertEqual(
            sorted([str(obj) for obj in model.component_objects()]),
            sorted(all_))
        self.assertEqual(
            sorted([str(obj) for obj in objs[None]]),
            sorted(all_))
        model.deactivate()
        self.assertEqual(
            sorted([str(obj) for obj in model.component_objects()]),
            sorted(all_))
        self.assertEqual(
            [str(obj) for obj in model.component_objects(descend_into=False,
                                                         active=True)],
            [])
        self.assertEqual(
            [str(obj) for obj in model.component_objects(descend_into=True,
                                                         active=True)],
            [])

    def test_block_data_objects_hack(self):
        model = _model.clone()
        model.deactivate()
        self.assertEqual(
            [str(obj) for obj in model.block_data_objects(active=True)],
            [])
        self.assertEqual(
            [str(obj) for obj in model.block_data_objects()],
            [str(model)]+[str(obj) for obj in model.components(ctype=IBlock)])
        model.activate()
        self.assertEqual(
            [str(obj) for obj in model.block_data_objects(active=True)],
            [str(model)]+[str(obj) for obj in model.components(ctype=IBlock)])
        self.assertEqual(
            [str(obj) for obj in model.block_data_objects()],
            [str(model)]+[str(obj) for obj in model.components(ctype=IBlock)])

if __name__ == "__main__":
    unittest.main()
