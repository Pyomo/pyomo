#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import tempfile
import os
import pickle
import random
import collections
import itertools

import pyutilib.th as unittest
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.symbol_map import SymbolMap
import pyomo.kernel as pmo
from pyomo.common.log import LoggingIntercept
from pyomo.core.tests.unit.kernel.test_dict_container import \
    _TestActiveDictContainerBase
from pyomo.core.tests.unit.kernel.test_tuple_container import \
    _TestActiveTupleContainerBase
from pyomo.core.tests.unit.kernel.test_list_container import \
    _TestActiveListContainerBase
from pyomo.core.kernel.base import \
    (ICategorizedObject,
     ICategorizedObjectContainer)
from pyomo.core.kernel.heterogeneous_container import \
    (heterogeneous_containers,
     IHeterogeneousContainer)
from pyomo.common.collections import ComponentMap
from pyomo.core.kernel.suffix import suffix
from pyomo.core.kernel.constraint import (IConstraint,
                                          constraint,
                                          constraint_dict,
                                          constraint_list)
from pyomo.core.kernel.parameter import (parameter,
                                         parameter_dict,
                                         parameter_list)
from pyomo.core.kernel.expression import (expression,
                                          data_expression,
                                          expression_dict,
                                          expression_list)
from pyomo.core.kernel.objective import (objective,
                                         objective_dict,
                                         objective_list)
from pyomo.core.kernel.variable import (IVariable,
                                        variable,
                                        variable_dict,
                                        variable_list)
from pyomo.core.kernel.block import (IBlock,
                                     block,
                                     block_dict,
                                     block_tuple,
                                     block_list)
from pyomo.core.kernel.sos import sos
from pyomo.opt.results import Solution

from six import StringIO

def _path_to_object_exists(obj, descendent):
    if descendent is obj:
        return True
    else:
        parent = descendent.parent
        if parent is None:
            return False
        else:
            return _path_to_object_exists(obj, parent)

def _active_path_to_object_exists(obj, descendent):
    if descendent is obj:
        return True
    else:
        parent = descendent.parent
        if parent is None:
            return False
        else:
            if getattr(descendent, "active", True):
                return _active_path_to_object_exists(obj, parent)
            else:
                return False

def _collect_expr_components(exp):
    ans = {}
    if isinstance(exp, ICategorizedObject):
        ans[id(exp)] = exp
    if exp.__class__ in native_numeric_types:
        return ans
    if exp.is_expression_type():
        for subexp in exp.args:
            ans.update(_collect_expr_components(subexp))
    return ans

class IJunk(IBlock):
    __slots__ = ()
class junk(pmo.block):
    _ctype = IJunk
class junk_list(pmo.block_list):
    __slots__ = ()
    _ctype = IJunk

class TestHeterogeneousContainer(unittest.TestCase):

    model = pmo.block()
    model.v = pmo.variable()
    model.V = pmo.variable_list()
    model.V.append(pmo.variable())
    model.V.append(pmo.variable_list())
    model.V[1].append(pmo.variable())
    model.c = pmo.constraint()
    model.C = pmo.constraint_list()
    model.C.append(pmo.constraint())
    model.C.append(pmo.constraint_list())
    model.C[1].append(pmo.constraint())
    b_clone = model.clone()
    model.b = b_clone.clone()
    model.B = pmo.block_list()
    model.B.append(b_clone.clone())
    model.B.append(pmo.block_list())
    model.B[1].append(b_clone.clone())
    del b_clone
    model.j = junk()
    model.J = junk_list()
    model.J.append(junk())
    model.J.append(junk_list())
    model.J[1].append(junk())
    model.J[1][0].b = pmo.block()
    model.J[1][0].b.v = pmo.variable()
    model_clone = model.clone()
    model.k = pmo.block()
    model.K = pmo.block_list()
    model.K.append(model_clone.clone())
    del model_clone

    def test_preorder_traversal(self):
        model = self.model.clone()

        order = list(str(obj) for obj in pmo.preorder_traversal(model))
        self.assertEqual(order,
                         ['<block>',
                          'v','V','V[0]','V[1]','V[1][0]',
                          'c','C','C[0]','C[1]','C[1][0]',
                          'b',
                          'b.v','b.V','b.V[0]','b.V[1]','b.V[1][0]',
                          'b.c','b.C','b.C[0]','b.C[1]','b.C[1][0]',
                          'B',
                          'B[0]',
                          'B[0].v','B[0].V','B[0].V[0]','B[0].V[1]','B[0].V[1][0]',
                          'B[0].c','B[0].C','B[0].C[0]','B[0].C[1]','B[0].C[1][0]',
                          'B[1]',
                          'B[1][0]',
                          'B[1][0].v','B[1][0].V','B[1][0].V[0]','B[1][0].V[1]','B[1][0].V[1][0]',
                          'B[1][0].c','B[1][0].C','B[1][0].C[0]','B[1][0].C[1]','B[1][0].C[1][0]',
                          'j',
                          'J',
                          'J[0]',
                          'J[1]',
                          'J[1][0]',
                          'J[1][0].b',
                          'J[1][0].b.v',
                          'k',
                          'K',
                          'K[0]',
                          'K[0].v','K[0].V','K[0].V[0]','K[0].V[1]','K[0].V[1][0]',
                          'K[0].c','K[0].C','K[0].C[0]','K[0].C[1]','K[0].C[1][0]',
                          'K[0].b',
                          'K[0].b.v','K[0].b.V','K[0].b.V[0]','K[0].b.V[1]','K[0].b.V[1][0]',
                          'K[0].b.c','K[0].b.C','K[0].b.C[0]','K[0].b.C[1]','K[0].b.C[1][0]',
                          'K[0].B',
                          'K[0].B[0]',
                          'K[0].B[0].v','K[0].B[0].V','K[0].B[0].V[0]','K[0].B[0].V[1]','K[0].B[0].V[1][0]',
                          'K[0].B[0].c','K[0].B[0].C','K[0].B[0].C[0]','K[0].B[0].C[1]','K[0].B[0].C[1][0]',
                          'K[0].B[1]',
                          'K[0].B[1][0]',
                          'K[0].B[1][0].v','K[0].B[1][0].V','K[0].B[1][0].V[0]','K[0].B[1][0].V[1]','K[0].B[1][0].V[1][0]',
                          'K[0].B[1][0].c','K[0].B[1][0].C','K[0].B[1][0].C[0]','K[0].B[1][0].C[1]','K[0].B[1][0].C[1][0]',
                          'K[0].j',
                          'K[0].J',
                          'K[0].J[0]',
                          'K[0].J[1]',
                          'K[0].J[1][0]',
                          'K[0].J[1][0].b',
                          'K[0].J[1][0].b.v'])

        order = list(str(obj) for obj in pmo.preorder_traversal(
            model,
            descend=lambda x: (x is not model.k) and (x is not model.K)))
        self.assertEqual(order,
                         ['<block>',
                          'v','V','V[0]','V[1]','V[1][0]',
                          'c','C','C[0]','C[1]','C[1][0]',
                          'b',
                          'b.v','b.V','b.V[0]','b.V[1]','b.V[1][0]',
                          'b.c','b.C','b.C[0]','b.C[1]','b.C[1][0]',
                          'B',
                          'B[0]',
                          'B[0].v','B[0].V','B[0].V[0]','B[0].V[1]','B[0].V[1][0]',
                          'B[0].c','B[0].C','B[0].C[0]','B[0].C[1]','B[0].C[1][0]',
                          'B[1]',
                          'B[1][0]',
                          'B[1][0].v','B[1][0].V','B[1][0].V[0]','B[1][0].V[1]','B[1][0].V[1][0]',
                          'B[1][0].c','B[1][0].C','B[1][0].C[0]','B[1][0].C[1]','B[1][0].C[1][0]',
                          'j',
                          'J',
                          'J[0]',
                          'J[1]',
                          'J[1][0]',
                          'J[1][0].b',
                          'J[1][0].b.v',
                          'k',
                          'K'])

        order = list(str(obj) for obj in pmo.preorder_traversal(model,
                                                                ctype=IBlock))
        self.assertEqual(order,
                         ['<block>',
                          'b',
                          'B',
                          'B[0]',
                          'B[1]',
                          'B[1][0]',
                          'j',
                          'J',
                          'J[0]',
                          'J[1]',
                          'J[1][0]',
                          'J[1][0].b',
                          'k',
                          'K',
                          'K[0]',
                          'K[0].b',
                          'K[0].B',
                          'K[0].B[0]',
                          'K[0].B[1]',
                          'K[0].B[1][0]',
                          'K[0].j',
                          'K[0].J',
                          'K[0].J[0]',
                          'K[0].J[1]',
                          'K[0].J[1][0]',
                          'K[0].J[1][0].b'])

        order = list(str(obj) for obj in pmo.preorder_traversal(model,
                                                                ctype=IVariable))
        self.assertEqual(order,
                         ['<block>',
                          'v','V','V[0]','V[1]','V[1][0]',
                          'b',
                          'b.v','b.V','b.V[0]','b.V[1]','b.V[1][0]',
                          'B',
                          'B[0]',
                          'B[0].v','B[0].V','B[0].V[0]','B[0].V[1]','B[0].V[1][0]',
                          'B[1]',
                          'B[1][0]',
                          'B[1][0].v','B[1][0].V','B[1][0].V[0]','B[1][0].V[1]','B[1][0].V[1][0]',
                          'j',
                          'J',
                          'J[0]',
                          'J[1]',
                          'J[1][0]',
                          'J[1][0].b',
                          'J[1][0].b.v',
                          'k',
                          'K',
                          'K[0]',
                          'K[0].v','K[0].V','K[0].V[0]','K[0].V[1]','K[0].V[1][0]',
                          'K[0].b',
                          'K[0].b.v','K[0].b.V','K[0].b.V[0]','K[0].b.V[1]','K[0].b.V[1][0]',
                          'K[0].B',
                          'K[0].B[0]',
                          'K[0].B[0].v','K[0].B[0].V','K[0].B[0].V[0]','K[0].B[0].V[1]','K[0].B[0].V[1][0]',
                          'K[0].B[1]',
                          'K[0].B[1][0]',
                          'K[0].B[1][0].v','K[0].B[1][0].V','K[0].B[1][0].V[0]','K[0].B[1][0].V[1]','K[0].B[1][0].V[1][0]',
                          'K[0].j',
                          'K[0].J',
                          'K[0].J[0]',
                          'K[0].J[1]',
                          'K[0].J[1][0]',
                          'K[0].J[1][0].b',
                          'K[0].J[1][0].b.v'])

    def test_components(self):
        model = self.model.clone()
        checked = []
        def descend_into(x):
            self.assertTrue(x._is_heterogeneous_container)
            checked.append(x.name)
            return True
        order = list(str(obj) for obj in model.components(
            descend_into=descend_into))
        self.assertEqual(checked,
                         ['b',
                          'B[0]',
                          'B[1][0]',
                          'j',
                          'J[0]',
                          'J[1][0]',
                          'J[1][0].b',
                          'k',
                          'K[0]',
                          'K[0].b',
                          'K[0].B[0]',
                          'K[0].B[1][0]',
                          'K[0].j',
                          'K[0].J[0]',
                          'K[0].J[1][0]',
                          'K[0].J[1][0].b'])
        self.assertEqual(order,
                         ['v','V[0]','V[1][0]',
                          'c','C[0]','C[1][0]',
                          'b',
                          'b.v','b.V[0]','b.V[1][0]',
                          'b.c','b.C[0]','b.C[1][0]',
                          'B[0]',
                          'B[0].v','B[0].V[0]','B[0].V[1][0]',
                          'B[0].c','B[0].C[0]','B[0].C[1][0]',
                          'B[1][0]',
                          'B[1][0].v','B[1][0].V[0]','B[1][0].V[1][0]',
                          'B[1][0].c','B[1][0].C[0]','B[1][0].C[1][0]',
                          'j',
                          'J[0]',
                          'J[1][0]',
                          'J[1][0].b',
                          'J[1][0].b.v',
                          'k',
                          'K[0]',
                          'K[0].v','K[0].V[0]','K[0].V[1][0]',
                          'K[0].c','K[0].C[0]','K[0].C[1][0]',
                          'K[0].b',
                          'K[0].b.v','K[0].b.V[0]','K[0].b.V[1][0]',
                          'K[0].b.c','K[0].b.C[0]','K[0].b.C[1][0]',
                          'K[0].B[0]',
                          'K[0].B[0].v','K[0].B[0].V[0]','K[0].B[0].V[1][0]',
                          'K[0].B[0].c','K[0].B[0].C[0]','K[0].B[0].C[1][0]',
                          'K[0].B[1][0]',
                          'K[0].B[1][0].v','K[0].B[1][0].V[0]','K[0].B[1][0].V[1][0]',
                          'K[0].B[1][0].c','K[0].B[1][0].C[0]','K[0].B[1][0].C[1][0]',
                          'K[0].j',
                          'K[0].J[0]',
                          'K[0].J[1][0]',
                          'K[0].J[1][0].b',
                          'K[0].J[1][0].b.v'])
        vlist = [str(obj) for obj in model.components(ctype=IVariable)]
        self.assertEqual(len(vlist), len(set(vlist)))
        clist = [str(obj) for obj in model.components(ctype=IConstraint)]
        self.assertEqual(len(clist), len(set(clist)))
        blist = [str(obj) for obj in model.components(ctype=IBlock)]
        self.assertEqual(len(blist), len(set(blist)))
        jlist = [str(obj) for obj in model.components(ctype=IJunk)]
        self.assertEqual(len(jlist), len(set(jlist)))

        for l1, l2 in itertools.product([vlist, clist, blist, jlist],
                                        repeat=2):
            if l1 is l2:
                continue
            self.assertEqual(set(l1).intersection(set(l2)), set([]))
        self.assertEqual(len(vlist)+len(clist)+len(blist)+len(jlist),
                         len(order))

    def test_getname(self):
        model = self.model.clone()
        self.assertEqual(model.J[1][0].b.v.getname(fully_qualified=True),
                         'J[1][0].b.v')
        self.assertEqual(model.J[1][0].b.v.getname(fully_qualified=True,
                                                   relative_to=model.J[1][0]),
                         'b.v')
        self.assertEqual(model.J[1][0].b.v.getname(fully_qualified=True,
                                                   relative_to=model.J[1]),
                         '[0].b.v')
        self.assertEqual(model.J[1][0].b.v.getname(fully_qualified=True,
                                                   relative_to=model.J),
                         '[1][0].b.v')
        self.assertEqual(model.J[1][0].b.v.getname(fully_qualified=True,
                                                   relative_to=model),
                         'J[1][0].b.v')

    def test_heterogeneous_containers(self):
        order = list(str(obj) for obj in heterogeneous_containers(self.model.V))
        self.assertEqual(order, [])
        order = list(str(obj) for obj in heterogeneous_containers(self.model.v))
        self.assertEqual(order, [])

        order = list(str(obj) for obj in heterogeneous_containers(self.model))
        self.assertEqual(order,
                         ['<block>',
                          'b',
                          'B[0]',
                          'B[1][0]',
                          'k',
                          'K[0]',
                          'K[0].b',
                          'K[0].B[0]',
                          'K[0].B[1][0]',
                          'K[0].j',
                          'K[0].J[0]',
                          'K[0].J[1][0]',
                          'K[0].J[1][0].b',
                          'j',
                          'J[0]',
                          'J[1][0]',
                          'J[1][0].b'])
        def f(x):
            # do not descend below heterogeneous containers
            # stored on self.model
            self.assertTrue(x._is_heterogeneous_container)
            parent = x.parent
            while parent is not None:
                if parent is self.model:
                    return False
                parent = parent.parent
            return True
        order1 = list(str(obj) for obj in heterogeneous_containers(
            self.model,
            descend_into=f))
        order2 = list(str(obj) for obj in heterogeneous_containers(
            self.model,
            descend_into=lambda x: True if (x is self.model) else False))
        self.assertEqual(order1, order2)
        self.assertEqual(order1,
                         ['<block>',
                          'b',
                          'B[0]',
                          'B[1][0]',
                          'k',
                          'K[0]',
                          'j',
                          'J[0]',
                          'J[1][0]'])
        order = list(str(obj) for obj in heterogeneous_containers(
            self.model,
            ctype=IBlock))
        self.assertEqual(order,
                         ['<block>',
                          'b',
                          'B[0]',
                          'B[1][0]',
                          'k',
                          'K[0]',
                          'K[0].b',
                          'K[0].B[0]',
                          'K[0].B[1][0]',
                          'K[0].J[1][0].b',
                          'J[1][0].b'])
        order = list(str(obj) for obj in heterogeneous_containers(
            self.model,
            ctype=IJunk))
        self.assertEqual(order,
                         ['K[0].j',
                          'K[0].J[0]',
                          'K[0].J[1][0]',
                          'j',
                          'J[0]',
                          'J[1][0]'])
        order = list(str(obj) for obj in heterogeneous_containers(
            self.model.K,
            ctype=IJunk))
        self.assertEqual(order,
                         ['K[0].j',
                          'K[0].J[0]',
                          'K[0].J[1][0]'])
        order = list(str(obj) for obj in heterogeneous_containers(
            self.model.K[0],
            ctype=IJunk))
        self.assertEqual(order,
                         ['K[0].j',
                          'K[0].J[0]',
                          'K[0].J[1][0]'])
        order = list(str(obj) for obj in heterogeneous_containers(
            self.model.K[0].j,
            ctype=IJunk))
        self.assertEqual(order,
                         ['K[0].j'])
        order = list(str(obj) for obj in heterogeneous_containers(
            self.model.K[0].j,
            ctype=IBlock))
        self.assertEqual(order,
                         [])

class TestMisc(unittest.TestCase):

    def test_reserved_attributes(self):
        b = block()
        self.assertTrue(len(block._block_reserved_words) > 0)
        for name in block._block_reserved_words:
            with self.assertRaises(ValueError):
                setattr(b, name, 1)
        with self.assertRaises(ValueError):
            b._active = 1
        with self.assertRaises(ValueError):
            b._parent = 1
        with self.assertRaises(ValueError):
            b._storage_key = 1
        with self.assertRaises(AttributeError):
            b.active = 1

    def test_pprint(self):
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        B = block()
        B.s = suffix()
        B.b = block()
        B.v = variable()
        pmo.pprint(B)
        B.c = constraint()
        B.e = expression()
        B.o = objective()
        B.p = parameter()
        B.s = sos([])
        pmo.pprint(B)
        b = block()
        b.B = B
        pmo.pprint(B)
        pmo.pprint(b)
        m = block()
        m.b = b
        pmo.pprint(B)
        pmo.pprint(b)
        pmo.pprint(m)

    def test_ctype(self):
        b = block()
        self.assertIs(b.ctype, IBlock)
        self.assertIs(type(b), block)
        self.assertIs(type(b)._ctype, IBlock)

    def test_write(self):
        b = block()
        b.v = variable()
        b.y = variable(value=1.0, fixed=True)
        b.p = parameter(value=2.0)
        b.e = expression(b.v * b.p + 1)
        b.eu = data_expression(b.p**2 + 10)
        b.el = data_expression(-b.p**2 - 10)
        b.o = objective(b.v + b.y)
        b.c1 = constraint((b.el + 1, b.e + 2, b.eu + 2))
        b.c2 = constraint(lb=b.el, body=b.v)
        b.c3 = constraint(body=b.v, ub=b.eu)

        fid, fname = tempfile.mkstemp(suffix='.lp')
        os.close(fid)
        assert fname.endswith('.lp')
        smap = b.write(fname)
        os.remove(fname)
        self.assertTrue(isinstance(smap, SymbolMap))

        fid, fname = tempfile.mkstemp(suffix='.lp')
        os.close(fid)
        assert fname.endswith('.lp')
        smap = b.write(fname, format='nl')
        os.remove(fname)
        self.assertTrue(isinstance(smap, SymbolMap))

        fid, fname = tempfile.mkstemp(suffix='.sdfsdfsf')
        os.close(fid)
        assert fname.endswith('.sdfsdfsf')
        with self.assertRaises(ValueError):
            b.write(fname)
        os.remove(fname)

        fid, fname = tempfile.mkstemp(suffix='.sdfsdfsf')
        os.close(fid)
        assert fname.endswith('.sdfsdfsf')
        with self.assertRaises(ValueError):
            b.write(fname, format="sdfdsfsdfsf")
        os.remove(fname)

    def test_load_solution(self):
        sm = SymbolMap()
        m = block()
        sm.addSymbol(m, 'm')
        m.v = variable()
        sm.addSymbol(m.v, 'v')
        m.c = constraint()
        sm.addSymbol(m.c, 'c')
        m.o = objective()
        sm.addSymbol(m.o, 'o')
        m.vsuffix = suffix(direction=suffix.IMPORT)
        m.osuffix = suffix(direction=suffix.IMPORT)
        m.csuffix = suffix(direction=suffix.IMPORT)
        m.msuffix = suffix(direction=suffix.IMPORT)

        soln = Solution()
        soln.symbol_map = sm
        soln.variable['v'] = {"Value": 1.0,
                              "vsuffix": 'v'}
        soln.variable['ONE_VAR_CONSTANT'] = None
        soln.constraint['c'] = {"csuffix": 'c'}
        soln.constraint['ONE_VAR_CONSTANT'] = None
        soln.objective['o'] = {"osuffix": 'o'}
        soln.problem["msuffix"] = 'm'

        m.load_solution(soln)

        self.assertEqual(m.v.value, 1.0)
        self.assertEqual(m.csuffix[m.c], 'c')
        self.assertEqual(m.osuffix[m.o], 'o')
        self.assertEqual(m.msuffix[m], 'm')

        soln.variable['vv'] = {"Value": 1.0,
                               "vsuffix": 'v'}
        with self.assertRaises(KeyError):
            m.load_solution(soln)
        del soln.variable['vv']

        soln.constraint['cc'] = {"csuffix": 'c'}
        with self.assertRaises(KeyError):
            m.load_solution(soln)
        del soln.constraint['cc']

        soln.objective['oo'] = {"osuffix": 'o'}
        with self.assertRaises(KeyError):
            m.load_solution(soln)
        del soln.objective['oo']

        m.v.fix()
        with self.assertRaises(ValueError):
            m.load_solution(soln,
                            allow_consistent_values_for_fixed_vars=False)

        m.v.fix(1.1)
        m.load_solution(soln,
                        allow_consistent_values_for_fixed_vars=True,
                        comparison_tolerance_for_fixed_vars=0.5)
        m.v.fix(1.1)
        with self.assertRaises(ValueError):
            m.load_solution(soln,
                            allow_consistent_values_for_fixed_vars=True,
                            comparison_tolerance_for_fixed_vars=0.05)

        del soln.variable['v']

        m.v.free()
        m.v.value = None
        m.load_solution(soln)
        self.assertEqual(m.v.stale, True)
        self.assertEqual(m.v.value, None)

        soln.default_variable_value = 1.0
        m.load_solution(soln)
        self.assertEqual(m.v.stale, False)
        self.assertEqual(m.v.value, 1.0)

        m.v.fix(1.0)
        with self.assertRaises(ValueError):
            m.load_solution(soln,
                            allow_consistent_values_for_fixed_vars=False)

        m.v.fix(1.1)
        m.load_solution(soln,
                        allow_consistent_values_for_fixed_vars=True,
                        comparison_tolerance_for_fixed_vars=0.5)
        m.v.fix(1.1)
        with self.assertRaises(ValueError):
            m.load_solution(soln,
                            allow_consistent_values_for_fixed_vars=True,
                            comparison_tolerance_for_fixed_vars=0.05)

    # a temporary test to make sure solve and load
    # functionality work (will be moved elsewhere in the
    # future)
    def test_solve_load(self):
        b = block()
        b.v = variable()
        b.y = variable(value=1.0, fixed=True)
        b.p = parameter(value=2.0)
        b.e = expression(b.v * b.p + 1)
        b.eu = data_expression(b.p**2 + 10)
        b.el = data_expression(-b.p**2 - 10)
        b.o = objective(b.v + b.y)
        b.c1 = constraint((b.el + 1, b.e + 2, b.eu + 2))
        b.c2 = constraint(lb=b.el, body=b.v)
        b.c3 = constraint(body=b.v, ub=b.eu)
        b.dual = suffix(direction=suffix.IMPORT)

        import pyomo.environ
        from pyomo.opt.base.solvers import UnknownSolver
        from pyomo.opt import SolverFactory
        from pyomo.opt import SolverStatus, TerminationCondition

        opt = SolverFactory("glpk")
        if isinstance(opt, UnknownSolver) or \
           (not opt.available()):
            raise unittest.SkipTest("glpk solver not available")
        status = opt.solve(b)
        self.assertEqual(status.solver.status,
                         SolverStatus.ok)
        self.assertEqual(status.solver.termination_condition,
                         TerminationCondition.optimal)
        self.assertAlmostEqual(b.o(), -7, places=5)
        self.assertAlmostEqual(b.v(), -8, places=5)
        self.assertAlmostEqual(b.y(), 1.0, places=5)

        opt = SolverFactory("glpk")
        if isinstance(opt, UnknownSolver) or \
           (not opt.available()):
            raise unittest.SkipTest("glpk solver not available")
        status = opt.solve(b, symbolic_solver_labels=True)
        self.assertEqual(status.solver.status,
                         SolverStatus.ok)
        self.assertEqual(status.solver.termination_condition,
                         TerminationCondition.optimal)
        self.assertAlmostEqual(b.o(), -7, places=5)
        self.assertAlmostEqual(b.v(), -8, places=5)
        self.assertAlmostEqual(b.y(), 1.0, places=5)

        opt = SolverFactory("ipopt")
        if isinstance(opt, UnknownSolver) or \
           (not opt.available()):
            raise unittest.SkipTest("ipopt solver not available")
        status = opt.solve(b)
        self.assertEqual(status.solver.status,
                         SolverStatus.ok)
        self.assertEqual(status.solver.termination_condition,
                         TerminationCondition.optimal)
        self.assertAlmostEqual(b.o(), -7, places=5)
        self.assertAlmostEqual(b.v(), -8, places=5)
        self.assertAlmostEqual(b.y(), 1.0, places=5)

        opt = SolverFactory("ipopt")
        if isinstance(opt, UnknownSolver):
            raise unittest.SkipTest("ipopt solver not available")
        status = opt.solve(b, symbolic_solver_labels=True)
        self.assertEqual(status.solver.status,
                         SolverStatus.ok)
        self.assertEqual(status.solver.termination_condition,
                         TerminationCondition.optimal)
        self.assertAlmostEqual(b.o(), -7, places=5)
        self.assertAlmostEqual(b.v(), -8, places=5)
        self.assertAlmostEqual(b.y(), 1.0, places=5)

    def test_traversal(self):

        b = block()
        b.v = variable()
        b.c1 = constraint()
        b.c1.deactivate()
        b.c2 = constraint_list()
        b.c2.append(constraint_list())
        b.B = block_list()
        b.B.append(block_list())
        b.B[0].append(block())
        b.B[0][0].c = constraint()
        b.B[0][0].b = block()
        b.B[0].deactivate()
        b._activate_large_storage_mode()

        def descend(obj):
            self.assertTrue(obj._is_container)
            return True
        self.assertEqual(
            [obj.name for obj in pmo.preorder_traversal(b,
                                                        active=None,
                                                        descend=descend)],
            [None,'v','c1','c2','c2[0]','B','B[0]','B[0][0]','B[0][0].c','B[0][0].b'])
        self.assertEqual(
            [obj.name for obj in pmo.preorder_traversal(b,
                                                        descend=descend)],
            [None,'v','c2','c2[0]','B'])
        self.assertEqual(
            [obj.name for obj in pmo.preorder_traversal(b,
                                                        active=True,
                                                        descend=descend)],
            [None,'v','c2','c2[0]','B'])

        self.assertEqual(
            [obj.name for obj in pmo.preorder_traversal(
                b,
                active=None,
                ctype=IConstraint,
                descend=descend)],
            [None,'c1','c2','c2[0]','B','B[0]','B[0][0]','B[0][0].c','B[0][0].b'])
        self.assertEqual(
            [obj.name for obj in pmo.preorder_traversal(
                b,
                ctype=IConstraint,
                descend=descend)],
            [None, 'c2', 'c2[0]', 'B'])
        self.assertEqual(
            [obj.name for obj in pmo.preorder_traversal(
                b,
                ctype=IConstraint,
                active=True,
                descend=descend)],
            [None, 'c2', 'c2[0]', 'B'])

        m = pmo.block()
        m.B = pmo.block_list()
        m.B.append(pmo.block())
        m.B[0].v = pmo.variable()

        self.assertEqual(
            [obj.name for obj in pmo.preorder_traversal(
                m,
                ctype=IVariable)],
            [None, 'B', 'B[0]', 'B[0].v'])
        self.assertEqual(
            [obj.name for obj in pmo.preorder_traversal(
                m.B,
                ctype=IVariable)],
            ['B', 'B[0]', 'B[0].v'])
        self.assertEqual(
            [obj.name for obj in pmo.preorder_traversal(
                m.B[0],
                ctype=IVariable)],
            ['B[0]', 'B[0].v'])
        self.assertEqual(
            [obj.name for obj in pmo.preorder_traversal(
                m.B[0].v,
                ctype=IVariable)],
            ['B[0].v'])

    # test how clone behaves when there are
    # references to components on a different block
    def test_clone1(self):
        b = block()
        b.v = variable()
        b.b = block()
        b.b.e = expression(b.v**2)
        b.b.v = variable()
        b.bdict = block_dict()
        b.bdict[0] = block()
        b.bdict[0].e = expression(b.v**2)
        b.blist = block_list()
        b.blist.append(block())
        b.blist[0].e = expression(b.v**2 + b.b.v**2)

        bc = b.clone()
        self.assertIsNot(b.v, bc.v)
        self.assertIsNot(b.b.e, bc.b.e)
        self.assertIsNot(b.bdict[0].e, bc.bdict[0].e)
        self.assertIsNot(b.blist[0].e, bc.blist[0].e)

        #
        # check that the expressions on cloned sub-blocks
        # reference the original variables for blocks "out-of-scope"
        #
        b_b = b.b.clone()
        self.assertIsNot(b_b.e, b.b.e)
        self.assertTrue(len(_collect_expr_components(b.b.e.expr)) == 1)
        self.assertIs(list(_collect_expr_components(b.b.e.expr).values())[0],
                      b.v)
        self.assertTrue(len(_collect_expr_components(b_b.e.expr)) == 1)
        self.assertIs(list(_collect_expr_components(b_b.e.expr).values())[0],
                      b.v)

        b_bdict0 = b.bdict[0].clone()
        self.assertIsNot(b_bdict0.e, b.bdict[0].e)
        self.assertTrue(len(_collect_expr_components(b.bdict[0].e.expr)) == 1)
        self.assertIs(list(_collect_expr_components(b.bdict[0].e.expr).values())[0],
                      b.v)
        self.assertTrue(len(_collect_expr_components(b_bdict0.e.expr)) == 1)
        self.assertIs(list(_collect_expr_components(b_bdict0.e.expr).values())[0],
                      b.v)

        b_blist0 = b.blist[0].clone()
        self.assertIsNot(b_blist0.e, b.blist[0].e)
        self.assertTrue(len(_collect_expr_components(b.blist[0].e.expr)) == 2)
        self.assertEqual(sorted(list(id(v_) for v_ in _collect_expr_components(b.blist[0].e.expr).values())),
                         sorted(list(id(v_) for v_ in [b.v, b.b.v])))
        self.assertTrue(len(_collect_expr_components(b_blist0.e.expr)) == 2)
        self.assertEqual(sorted(list(id(v_) for v_ in _collect_expr_components(b_blist0.e.expr).values())),
                         sorted(list(id(v_) for v_ in [b.v, b.b.v])))

    # test bulk clone behavior
    def test_clone2(self):
        b = block()
        b.v = variable()
        b.vdict = variable_dict(((i, variable())
                                for i in range(10)))
        b.vlist = variable_list(variable()
                                for i in range(10))
        b.o = objective(b.v + b.vdict[0] + b.vlist[0])
        b.odict = objective_dict(((i, objective(b.v + b.vdict[i]))
                                 for i in b.vdict))
        b.olist = objective_list(objective(b.v + v_)
                                 for i,v_ in enumerate(b.vdict))
        b.c = constraint(b.v >= 1)
        b.cdict = constraint_dict(((i, constraint(b.vdict[i] == i))
                                  for i in b.vdict))
        b.clist = constraint_list(constraint((0, v_, i))
                                  for i, v_ in enumerate(b.vlist))
        b.p = parameter()
        b.pdict = parameter_dict(((i, parameter(i))
                                 for i in b.vdict))
        b.plist = parameter_list(parameter(i)
                                 for i in range(len(b.vlist)))
        b.e = expression(b.v * b.p + 1)
        b.edict = expression_dict(((i, expression(b.vdict[i] * b.pdict[i] + 1))
                                  for i in b.vdict))
        b.elist = expression_list(expression(v_ * b.plist[i] + 1)
                                  for i,v_ in enumerate(b.vlist))

        self.assertIs(b.parent, None)

        #
        # clone the block
        #
        bc = b.clone()
        self.assertIs(bc.parent, None)
        self.assertIsNot(b, bc)
        self.assertTrue(len(list(b.children())) > 0)
        self.assertEqual(len(list(b.children())),
                         len(list(bc.children())))
        for c1, c2 in zip(b.children(), bc.children()):
            self.assertIs(c1.parent, b)
            self.assertIs(c2.parent, bc)
            self.assertIsNot(c1, c2)
            self.assertEqual(c1.name, c2.name)

        self.assertEqual(len(list(b.components())),
                         len(list(bc.components())))
        for c1, c2 in zip(b.components(), bc.components()):
            self.assertIsNot(c1, c2)
            self.assertEqual(c1.name, c2.name)
            if hasattr(c1,'expr'):
                self.assertIsNot(c1.expr, c2.expr)
                self.assertEqual(str(c1.expr), str(c2.expr))
                self.assertEqual(len(_collect_expr_components(c1.expr)),
                                 len(_collect_expr_components(c2.expr)))
                for subc1, subc2 in zip(_collect_expr_components(c1.expr).values(),
                                        _collect_expr_components(c2.expr).values()):
                    self.assertIsNot(subc1, subc2)
                    self.assertEqual(subc1.name, subc2.name)

        bc_init = bc.clone()
        b.bc = bc
        self.assertIs(b.parent, None)
        self.assertIs(bc.parent, b)
        #
        # clone the block with the newly added sub-block
        #

        bcc = b.clone()
        self.assertIsNot(b, bcc)
        self.assertEqual(len(list(b.children())),
                         len(list(bcc.children())))
        for c1, c2 in zip(b.children(), bcc.children()):
            self.assertIs(c1.parent, b)
            self.assertIs(c2.parent, bcc)
            self.assertIsNot(c1, c2)
            self.assertEqual(c1.name, c2.name)

        self.assertEqual(len(list(b.components())),
                         len(list(bcc.components())))
        self.assertTrue(hasattr(bcc, 'bc'))
        for c1, c2 in zip(b.components(), bcc.components()):
            self.assertIsNot(c1, c2)
            self.assertEqual(c1.name, c2.name)
            if hasattr(c1,'expr'):
                self.assertIsNot(c1.expr, c2.expr)
                self.assertEqual(str(c1.expr), str(c2.expr))
                self.assertEqual(len(_collect_expr_components(c1.expr)),
                                 len(_collect_expr_components(c2.expr)))
                for subc1, subc2 in zip(_collect_expr_components(c1.expr).values(),
                                        _collect_expr_components(c2.expr).values()):
                    self.assertIsNot(subc1, subc2)
                    self.assertEqual(subc1.name, subc2.name)

        #
        # clone the sub-block
        #
        sub_bc = b.bc.clone()
        self.assertIs(sub_bc.parent, None)
        self.assertIs(bc_init.parent, None)
        self.assertIs(bc.parent, b)
        self.assertIs(b.parent, None)

        self.assertIsNot(bc_init, sub_bc)
        self.assertIsNot(bc, sub_bc)
        self.assertEqual(len(list(bc_init.children())),
                         len(list(sub_bc.children())))
        for c1, c2 in zip(bc_init.children(), sub_bc.children()):
            self.assertIs(c1.parent, bc_init)
            self.assertIs(c2.parent, sub_bc)
            self.assertIsNot(c1, c2)
            self.assertEqual(c1.name, c2.name)

        self.assertEqual(len(list(bc_init.components())),
                         len(list(sub_bc.components())))
        for c1, c2 in zip(bc_init.components(), sub_bc.components()):
            self.assertIsNot(c1, c2)
            self.assertEqual(c1.name, c2.name)
            if hasattr(c1,'expr'):
                self.assertIsNot(c1.expr, c2.expr)
                self.assertEqual(str(c1.expr), str(c2.expr))
                self.assertEqual(len(_collect_expr_components(c1.expr)),
                                 len(_collect_expr_components(c2.expr)))
                for subc1, subc2 in zip(_collect_expr_components(c1.expr).values(),
                                        _collect_expr_components(c2.expr).values()):
                    self.assertIsNot(subc1, subc2)
                    self.assertEqual(subc1.name, subc2.name)

    def test_activate(self):
        b = block()
        self.assertEqual(b.active, True)
        b.deactivate()
        self.assertEqual(b.active, False)
        c = constraint()
        v = variable()
        self.assertEqual(c.active, True)
        c.deactivate()
        self.assertEqual(c.active, False)
        b.c = c
        b.v = v
        self.assertEqual(c.active, False)
        self.assertEqual(b.active, False)
        del b.c
        c.activate()
        self.assertEqual(c.active, True)
        self.assertEqual(b.active, False)
        b.c = c
        self.assertEqual(c.active, True)
        self.assertEqual(b.active, False)

        bdict = block_dict()
        self.assertEqual(bdict.active, True)
        bdict.deactivate()
        self.assertEqual(bdict.active, False)
        bdict[None] = b
        self.assertEqual(bdict.active, False)
        del bdict[None]
        self.assertEqual(bdict.active, False)
        b.activate()
        self.assertEqual(c.active, True)
        self.assertEqual(b.active, True)
        bdict[None] = b
        self.assertEqual(bdict.active, False)

        bdict.deactivate()
        self.assertEqual(c.active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(bdict.active, False)

        bdict.deactivate(shallow=False)
        self.assertEqual(c.active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(bdict.active, False)

        b.activate()
        self.assertEqual(c.active, False)
        self.assertEqual(b.active, True)
        self.assertEqual(bdict.active, False)

        b.activate(shallow=False)
        self.assertEqual(c.active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(bdict.active, False)

        bdict.deactivate()
        self.assertEqual(c.active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(bdict.active, False)

        bdict.deactivate(shallow=False)
        self.assertEqual(c.active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(bdict.active, False)

        bdict.activate()
        self.assertEqual(c.active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(bdict.active, True)

        bdict.activate(shallow=False)
        self.assertEqual(c.active, True)
        self.assertEqual(b.active, True)
        self.assertEqual(bdict.active, True)

        bdict.deactivate(shallow=False)
        self.assertEqual(c.active, False)
        self.assertEqual(b.active, False)
        self.assertEqual(bdict.active, False)

    # this is a randomized test
    def test_ordering(self):
        b = block()
        attr_types = [variable,
                      constraint,
                      parameter,
                      expression,
                      data_expression,
                      objective,
                      variable,
                      block]
        key_types = [float, int]
        keys = collections.deque()
        objs = collections.deque()
        for i in range(100):
            obj = random.choice(attr_types)()
            key = str(random.choice(key_types)(i))
            setattr(b, key, obj)
            keys.append(key)
            objs.append(obj)
            self.assertEqual(len(list(b.children())), len(objs))

        for i in range(100):
            children = list(b.children())
            for key, obj, child in zip(keys, objs, children):
                assert obj is child
                assert getattr(b, key) is child
                assert key == child.storage_key
            setattr(b, children[0].storage_key, children[0])
            keys.rotate(-1)
            objs.rotate(-1)

class _Test_block_base(object):

    _children = None
    _child_key = None
    _components_no_descend = None
    _components = None
    _blocks_no_descend = None
    _blocks = None
    _block = None

    def test_overwrite_warning(self):
        b = self._block.clone()
        name = "x"
        while hasattr(b,name):
            name += "x"
        out = StringIO()
        with LoggingIntercept(out, 'pyomo.core'):
            setattr(b,name,variable())
            setattr(b,name,getattr(b,name))
        assert out.getvalue() == "", str(out.getvalue())
        with LoggingIntercept(out, 'pyomo.core'):
            setattr(b,name,variable())
        assert out.getvalue() == \
            ("Implicitly replacing attribute %s (type=variable) "
             "on block with new object (type=variable). This "
             "is usually indicative of a modeling error. "
             "To avoid this warning, delete the original "
             "object from the block before assigning a new "
             "object.\n" % (name))
        out = StringIO()
        with LoggingIntercept(out, 'pyomo.core'):
            setattr(b,name,1.0)
        assert out.getvalue() == \
            ("Implicitly replacing attribute %s (type=variable) "
             "on block with new object (type=float). This "
             "is usually indicative of a modeling error. "
             "To avoid this warning, delete the original "
             "object from the block before assigning a new "
             "object.\n" % (name))

    def test_clone(self):
        b = self._block
        bc = b.clone()
        self.assertIsNot(b, bc)
        self.assertEqual(len(list(b.children())),
                         len(list(bc.children())))
        for c1, c2 in zip(b.children(), bc.children()):
            self.assertIs(c1.parent, b)
            self.assertIs(c2.parent, bc)
            self.assertIsNot(c1, c2)
            self.assertEqual(c1.name, c2.name)

        self.assertEqual(len(list(b.components())),
                         len(list(bc.components())))
        for c1, c2 in zip(b.components(), bc.components()):
            self.assertIsNot(c1, c2)
            self.assertEqual(c1.name, c2.name)

    def test_pickle(self):
        b = pickle.loads(
            pickle.dumps(self._block))
        self.assertEqual(len(list(pmo.preorder_traversal(b, active=None))),
                         len(self._names)+1)
        names = pmo.generate_names(b, active=None)
        self.assertEqual(sorted(names.values()),
                         sorted(self._names.values()))

    def test_preorder_traversal(self):
        # this first test makes failures a
        # little easier to debug
        self.assertEqual(
            [str(obj) for obj in pmo.preorder_traversal(self._block,
                                                        active=None)],
            [str(obj) for obj in self._preorder])
        self.assertEqual(
            [id(obj) for obj in pmo.preorder_traversal(self._block,
                                                       active=None)],
            [id(obj) for obj in self._preorder])

        # this first test makes failures a
        # little easier to debug
        self.assertEqual(
            [str(obj) for obj in pmo.preorder_traversal(
                self._block,
                active=None,
                ctype=IVariable)],
            [str(obj) for obj in self._preorder
             if obj.ctype in (IBlock, IVariable)])
        self.assertEqual(
            [id(obj) for obj in pmo.preorder_traversal(
                self._block,
                active=None,
                ctype=IVariable)],
            [id(obj) for obj in self._preorder
             if obj.ctype in (IBlock, IVariable)])

    def test_preorder_traversal_descend_check(self):
        def descend(x):
            self.assertTrue(x._is_container)
            return True
        order = list(pmo.preorder_traversal(self._block,
                                            active=None,
                                            descend=descend))
        # this first test makes failures a
        # little easier to debug
        self.assertEqual(
            [str(obj) for obj in order],
            [str(obj) for obj in self._preorder])
        self.assertEqual(
            [id(obj) for obj in order],
            [id(obj) for obj in self._preorder])

        def descend(x):
            self.assertTrue(x._is_container)
            return True
        order = list(pmo.preorder_traversal(self._block,
                                            active=None,
                                            ctype=IVariable,
                                            descend=descend))
        # this first test makes failures a
        # little easier to debug
        self.assertEqual(
            [str(obj) for obj in order],
            [str(obj) for obj in self._preorder
             if obj.ctype in (IBlock, IVariable)])
        self.assertEqual(
            [id(obj) for obj in order],
            [id(obj) for obj in self._preorder
             if obj.ctype in (IBlock, IVariable)])

        def descend(x):
            if x.parent is self._block:
                return False
            return True
        order = list(pmo.preorder_traversal(self._block,
                                            active=None,
                                            descend=descend))
        # this first test makes failures a
        # little easier to debug
        self.assertEqual(
            [str(obj) for obj in order],
            [str(obj) for obj in self._preorder
             if (obj.parent is None) or \
                (obj.parent is self._block)])
        self.assertEqual(
            [id(obj) for obj in order],
            [id(obj) for obj in self._preorder
             if (obj.parent is None) or \
                (obj.parent is self._block)])

    def test_child(self):
        for child in self._child_key:
            parent = child.parent
            self.assertTrue(parent is not None)
            self.assertTrue(id(child) in set(
                id(_c) for _c in self._children[parent]))
            self.assertIs(parent.child(self._child_key[child]),
                          child)
            with self.assertRaises(KeyError):
                parent.child("_not_a_valid_child_key_")

    def test_children(self):
        for obj in self._children:
            self.assertTrue(isinstance(obj, ICategorizedObjectContainer))
            if isinstance(obj, IBlock):
                for child in obj.children():
                    self.assertTrue(child.parent is obj)
                # this first test makes failures a
                # little easier to debug
                self.assertEqual(
                    sorted(str(child)
                           for child in obj.children()),
                    sorted(str(child)
                           for child in self._children[obj]))
                self.assertEqual(
                    set(id(child) for child in obj.children()),
                    set(id(child) for child in self._children[obj]))
                # this first test makes failures a
                # little easier to debug
                self.assertEqual(
                    sorted(str(child)
                           for child in obj.children(ctype=IBlock)),
                    sorted(str(child)
                           for child in self._children[obj]
                           if child.ctype is IBlock))
                self.assertEqual(
                    set(id(child) for child in obj.children(ctype=IBlock)),
                    set(id(child) for child in self._children[obj]
                        if child.ctype is IBlock))
                # this first test makes failures a
                # little easier to debug
                self.assertEqual(
                    sorted(str(child)
                           for child in obj.children(ctype=IVariable)),
                    sorted(str(child)
                           for child in self._children[obj]
                           if child.ctype is IVariable))
                self.assertEqual(
                    set(id(child) for child in obj.children(ctype=IVariable)),
                    set(id(child) for child in self._children[obj]
                        if child.ctype is IVariable))
            elif isinstance(obj, ICategorizedObjectContainer):
                for child in obj.children():
                    self.assertTrue(child.parent is obj)
                self.assertEqual(
                    set(id(child) for child in obj.children()),
                    set(id(child) for child in self._children[obj]))
            else:
                self.assertEqual(len(self._children[obj]), 0)

    def test_components_no_descend_active_None(self):
        for obj in self._components_no_descend:
            self.assertTrue(isinstance(obj, ICategorizedObjectContainer))
            self.assertTrue(isinstance(obj, IBlock))
            for c in obj.components(descend_into=False):
                self.assertTrue(
                    _path_to_object_exists(obj, c))
            # test ctype=IBlock
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.components(ctype=IBlock,
                                      active=None,
                                      descend_into=False)),
                sorted(str(_b)
                       for _b in
                       self._components_no_descend[obj][IBlock]))
            self.assertEqual(
                set(id(_b) for _b in
                    obj.components(ctype=IBlock,
                                   active=None,
                                   descend_into=False)),
                set(id(_b) for _b in
                    self._components_no_descend[obj][IBlock]))
            # test ctype=IVariable
            self.assertEqual(
                sorted(str(_v)
                       for _v in
                       obj.components(ctype=IVariable,
                                      active=None,
                                      descend_into=False)),
                sorted(str(_v)
                       for _v in
                       self._components_no_descend[obj][IVariable]))
            self.assertEqual(
                set(id(_v) for _v in
                    obj.components(ctype=IVariable,
                                   active=None,
                                   descend_into=False)),
                set(id(_v) for _v in
                    self._components_no_descend[obj][IVariable]))
            # test no ctype
            self.assertEqual(
                sorted(str(_c)
                       for _c in
                       obj.components(active=None,
                                      descend_into=False)),
                sorted(str(_c)
                       for ctype in
                       self._components_no_descend[obj]
                       for _c in
                       self._components_no_descend[obj][ctype]))
            self.assertEqual(
                set(id(_c) for _c in
                    obj.components(active=None,
                                   descend_into=False)),
                set(id(_c) for ctype in
                    self._components_no_descend[obj]
                    for _c in
                    self._components_no_descend[obj][ctype]))

    def test_components_no_descend_active_True(self):
        for obj in self._components_no_descend:
            self.assertTrue(isinstance(obj, ICategorizedObjectContainer))
            self.assertTrue(isinstance(obj, IBlock))
            # test ctype=IBlock
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.components(ctype=IBlock,
                                      active=True,
                                      descend_into=False)),
                sorted(str(_b)
                       for _b in
                       self._components_no_descend[obj][IBlock]
                       if _b.active)
                if getattr(obj, 'active', True) else [])
            self.assertEqual(
                set(id(_b) for _b in
                    obj.components(ctype=IBlock,
                                   active=True,
                                   descend_into=False)),
                set(id(_b) for _b in
                    self._components_no_descend[obj][IBlock]
                    if _b.active)
                if getattr(obj, 'active', True) else set())
            # test ctype=IVariable
            self.assertEqual(
                sorted(str(_v)
                       for _v in
                       obj.components(ctype=IVariable,
                                      active=True,
                                      descend_into=False)),
                sorted(str(_v)
                       for _v in
                       self._components_no_descend[obj][IVariable])
                if getattr(obj, 'active', True) else [])
            self.assertEqual(
                set(id(_v) for _v in
                    obj.components(ctype=IVariable,
                                   active=True,
                                   descend_into=False)),
                set(id(_v) for _v in
                    self._components_no_descend[obj][IVariable])
                if getattr(obj, 'active', True) else set())
            # test no ctype
            self.assertEqual(
                sorted(str(_c)
                       for _c in
                       obj.components(active=True,
                                      descend_into=False)),
                sorted(str(_c)
                       for ctype in
                       self._components_no_descend[obj]
                       for _c in
                       self._components_no_descend[obj][ctype]
                       if getattr(_c, "active", True))
                if getattr(obj, 'active', True) else [])
            self.assertEqual(
                set(id(_c) for _c in
                    obj.components(active=True,
                                   descend_into=False)),
                set(id(_c) for ctype in
                    self._components_no_descend[obj]
                    for _c in
                    self._components_no_descend[obj][ctype]
                    if getattr(_c, "active", True))
                if getattr(obj, 'active', True) else set())

    def test_components_active_None(self):
        for obj in self._components:
            self.assertTrue(isinstance(obj, ICategorizedObjectContainer))
            self.assertTrue(isinstance(obj, IBlock))
            for c in obj.components(descend_into=True):
                self.assertTrue(
                    _path_to_object_exists(obj, c))
            # test ctype=IBlock
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.components(ctype=IBlock,
                                      active=None,
                                      descend_into=True)),
                sorted(str(_b)
                       for _b in
                       self._components[obj][IBlock]))
            self.assertEqual(
                set(id(_b) for _b in
                    obj.components(ctype=IBlock,
                                   active=None,
                                   descend_into=True)),
                set(id(_b) for _b in
                    self._components[obj][IBlock]))
            # test ctype=IVariable
            self.assertEqual(
                sorted(str(_v)
                       for _v in
                       obj.components(ctype=IVariable,
                                      active=None,
                                      descend_into=True)),
                sorted(str(_v)
                       for _v in
                       self._components[obj][IVariable]))
            self.assertEqual(
                set(id(_v) for _v in
                    obj.components(ctype=IVariable,
                                   active=None,
                                   descend_into=True)),
                set(id(_v) for _v in
                    self._components[obj][IVariable]))
            # test no ctype
            self.assertEqual(
                sorted(str(_c)
                       for _c in
                       obj.components(active=None,
                                      descend_into=True)),
                sorted(str(_c)
                       for ctype in
                       self._components[obj]
                       for _c in
                       self._components[obj][ctype]))
            self.assertEqual(
                set(id(_c) for _c in
                    obj.components(active=None,
                                   descend_into=True)),
                set(id(_c) for ctype in
                    self._components[obj]
                    for _c in
                    self._components[obj][ctype]))

    def test_components_active_True(self):
        for obj in self._components:
            self.assertTrue(isinstance(obj, ICategorizedObjectContainer))
            self.assertTrue(isinstance(obj, IBlock))
            # test ctype=IBlock
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.components(ctype=IBlock,
                                      active=True,
                                      descend_into=True)),
                sorted(str(_b)
                       for _b in
                       self._components[obj][IBlock]
                       if _b.active)
                if getattr(obj, 'active', True) else [])
            self.assertEqual(
                set(id(_b) for _b in
                    obj.components(ctype=IBlock,
                                   active=True,
                                   descend_into=True)),
                set(id(_b) for _b in
                    self._components[obj][IBlock]
                    if _b.active)
                if getattr(obj, 'active', True) else set())
            # test ctype=IVariable
            self.assertEqual(
                sorted(str(_v)
                       for _v in
                       obj.components(ctype=IVariable,
                                      active=True,
                                      descend_into=True)),
                sorted(str(_v)
                       for _v in
                       self._components[obj][IVariable]
                       if _active_path_to_object_exists(obj, _v))
                if getattr(obj, 'active', True) else [])
            self.assertEqual(
                set(id(_v) for _v in
                    obj.components(ctype=IVariable,
                                   active=True,
                                   descend_into=True)),
                set(id(_v) for _v in
                    self._components[obj][IVariable]
                    if _active_path_to_object_exists(obj, _v))
                if getattr(obj, 'active', True) else set())
            # test no ctype
            self.assertEqual(
                sorted(str(_c)
                       for _c in
                       obj.components(active=True,
                                      descend_into=True)),
                sorted(str(_c)
                       for ctype in
                       self._components[obj]
                       for _c in
                       self._components[obj][ctype]
                       if _active_path_to_object_exists(obj, _c))
                if getattr(obj, 'active', True) else [])
            self.assertEqual(
                set(id(_c) for _c in
                    obj.components(active=True,
                                   descend_into=True)),
                set(id(_c) for ctype in
                    self._components[obj]
                    for _c in
                    self._components[obj][ctype]
                    if _active_path_to_object_exists(obj, _c))
                if getattr(obj, 'active', True) else set())

class _Test_block(_Test_block_base):

    _do_clone = None

    @classmethod
    def setUpClass(cls):
        assert cls._do_clone is not None
        model = cls._block = block()
        model.v_1 = variable()
        model.vdict_1 = variable_dict()
        model.vdict_1[None] = variable()
        model.vlist_1 = variable_list()
        model.vlist_1.append(variable())
        model.vlist_1.append(variable())
        model.b_1 = block()
        model.b_1.v_2 = variable()
        model.b_1.b_2 = block()
        model.b_1.b_2.b_3 = block()
        model.b_1.b_2.v_3 = variable()
        model.b_1.b_2.vlist_3 = variable_list()
        model.b_1.b_2.vlist_3.append(variable())
        model.b_1.b_2.deactivate(shallow=False)
        model.bdict_1 = block_dict()
        model.blist_1 = block_list()
        model.blist_1.append(block())
        model.blist_1[0].v_2 = variable()
        model.blist_1[0].b_2 = block()

        if cls._do_clone:
            model = cls._block = model.clone()

        #
        # Manually encode the correct output
        # for tests in the base testing class
        #

        cls._preorder = [model,
                         model.v_1,
                         model.vdict_1,
                         model.vdict_1[None],
                         model.vlist_1,
                         model.vlist_1[0],
                         model.vlist_1[1],
                         model.b_1,
                         model.b_1.v_2,
                         model.b_1.b_2,
                         model.b_1.b_2.b_3,
                         model.b_1.b_2.v_3,
                         model.b_1.b_2.vlist_3,
                         model.b_1.b_2.vlist_3[0],
                         model.bdict_1,
                         model.blist_1,
                         model.blist_1[0],
                         model.blist_1[0].v_2,
                         model.blist_1[0].b_2]

        cls._names = ComponentMap()
        cls._names[model.v_1] = "v_1"
        cls._names[model.vdict_1] = "vdict_1"
        cls._names[model.vdict_1[None]] = "vdict_1[None]"
        cls._names[model.vlist_1] = "vlist_1"
        cls._names[model.vlist_1[0]] = "vlist_1[0]"
        cls._names[model.vlist_1[1]] = "vlist_1[1]"
        cls._names[model.b_1] = "b_1"
        cls._names[model.b_1.v_2] = "b_1.v_2"
        cls._names[model.b_1.b_2] = "b_1.b_2"
        cls._names[model.b_1.b_2.b_3] = "b_1.b_2.b_3"
        cls._names[model.b_1.b_2.v_3] = "b_1.b_2.v_3"
        cls._names[model.b_1.b_2.vlist_3] = "b_1.b_2.vlist_3"
        cls._names[model.b_1.b_2.vlist_3[0]] = "b_1.b_2.vlist_3[0]"
        cls._names[model.bdict_1] = "bdict_1"
        cls._names[model.blist_1] = "blist_1"
        cls._names[model.blist_1[0]] = "blist_1[0]"
        cls._names[model.blist_1[0].v_2] = "blist_1[0].v_2"
        cls._names[model.blist_1[0].b_2] = "blist_1[0].b_2"

        cls._children = ComponentMap()
        cls._children[model] = [model.v_1,
                                model.vdict_1,
                                model.vlist_1,
                                model.b_1,
                                model.bdict_1,
                                model.blist_1]
        cls._children[model.vdict_1] = [model.vdict_1[None]]
        cls._children[model.vlist_1] = [model.vlist_1[0],
                                        model.vlist_1[1]]
        cls._children[model.b_1] = [model.b_1.v_2,
                                    model.b_1.b_2]
        cls._children[model.b_1.b_2] = [model.b_1.b_2.v_3,
                                        model.b_1.b_2.vlist_3,
                                        model.b_1.b_2.b_3]
        cls._children[model.b_1.b_2.b_3] = []
        cls._children[model.b_1.b_2.vlist_3] = \
            [model.b_1.b_2.vlist_3[0]]
        cls._children[model.bdict_1] = []
        cls._children[model.blist_1] = [model.blist_1[0]]
        cls._children[model.blist_1[0]] = [model.blist_1[0].v_2,
                                           model.blist_1[0].b_2]

        cls._child_key = ComponentMap()
        cls._child_key[model.v_1] = "v_1"
        cls._child_key[model.vdict_1] = "vdict_1"
        cls._child_key[model.vlist_1] = "vlist_1"
        cls._child_key[model.b_1] = "b_1"
        cls._child_key[model.bdict_1] = "bdict_1"
        cls._child_key[model.blist_1] = "blist_1"
        cls._child_key[model.vdict_1[None]] = None
        cls._child_key[model.vlist_1[0]] = 0
        cls._child_key[model.vlist_1[1]] = 1
        cls._child_key[model.b_1.v_2] = "v_2"
        cls._child_key[model.b_1.b_2] = "b_2"
        cls._child_key[model.b_1.b_2.b_3] = "b_3"
        cls._child_key[model.b_1.b_2.v_3] = "v_3"
        cls._child_key[model.b_1.b_2.vlist_3] = "vlist_3"
        cls._child_key[model.b_1.b_2.vlist_3[0]] = 0
        cls._child_key[model.blist_1[0]] = 0
        cls._child_key[model.blist_1[0].v_2] = "v_2"
        cls._child_key[model.blist_1[0].b_2] = "b_2"

        cls._components_no_descend = ComponentMap()
        cls._components_no_descend[model] = {}
        cls._components_no_descend[model][IVariable] = \
            [model.v_1,
             model.vdict_1[None],
             model.vlist_1[0],
             model.vlist_1[1]]
        cls._components_no_descend[model][IBlock] = \
            [model.b_1,
             model.blist_1[0]]
        cls._components_no_descend[model.b_1] = {}
        cls._components_no_descend[model.b_1][IVariable] = \
            [model.b_1.v_2]
        cls._components_no_descend[model.b_1][IBlock] = \
            [model.b_1.b_2]
        cls._components_no_descend[model.b_1.b_2] = {}
        cls._components_no_descend[model.b_1.b_2][IVariable] = \
            [model.b_1.b_2.v_3,
             model.b_1.b_2.vlist_3[0]]
        cls._components_no_descend[model.b_1.b_2][IBlock] = \
            [model.b_1.b_2.b_3]
        cls._components_no_descend[model.b_1.b_2.b_3] = {}
        cls._components_no_descend[model.b_1.b_2.b_3][IVariable] = []
        cls._components_no_descend[model.b_1.b_2.b_3][IBlock] = []
        cls._components_no_descend[model.blist_1[0]] = {}
        cls._components_no_descend[model.blist_1[0]][IVariable] = \
            [model.blist_1[0].v_2]
        cls._components_no_descend[model.blist_1[0]][IBlock] = \
            [model.blist_1[0].b_2]
        cls._components_no_descend[model.blist_1[0].b_2] = {}
        cls._components_no_descend[model.blist_1[0].b_2][IVariable] = []
        cls._components_no_descend[model.blist_1[0].b_2][IBlock] = []

        cls._components = ComponentMap()
        cls._components[model] = {}
        cls._components[model][IVariable] = \
            [model.v_1,
             model.vdict_1[None],
             model.vlist_1[0],
             model.vlist_1[1],
             model.b_1.v_2,
             model.b_1.b_2.v_3,
             model.b_1.b_2.vlist_3[0],
             model.blist_1[0].v_2]
        cls._components[model][IBlock] = \
            [model.b_1,
             model.blist_1[0],
             model.b_1.b_2,
             model.b_1.b_2.b_3,
             model.blist_1[0].b_2]
        cls._components[model.b_1] = {}
        cls._components[model.b_1][IVariable] = \
            [model.b_1.v_2,
             model.b_1.b_2.v_3,
             model.b_1.b_2.vlist_3[0]]
        cls._components[model.b_1][IBlock] = \
            [model.b_1.b_2,
             model.b_1.b_2.b_3]
        cls._components[model.b_1.b_2] = {}
        cls._components[model.b_1.b_2][IVariable] = \
            [model.b_1.b_2.v_3,
             model.b_1.b_2.vlist_3[0]]
        cls._components[model.b_1.b_2][IBlock] = \
            [model.b_1.b_2.b_3]
        cls._components[model.b_1.b_2.b_3] = {}
        cls._components[model.b_1.b_2.b_3][IVariable] = []
        cls._components[model.b_1.b_2.b_3][IBlock] = []
        cls._components[model.blist_1[0]] = {}
        cls._components[model.blist_1[0]][IVariable] = \
            [model.blist_1[0].v_2]
        cls._components[model.blist_1[0]][IBlock] = \
            [model.blist_1[0].b_2]
        cls._components[model.blist_1[0].b_2] = {}
        cls._components[model.blist_1[0].b_2][IVariable] = []
        cls._components[model.blist_1[0].b_2][IBlock] = []

        cls._blocks_no_descend = ComponentMap()
        for obj in cls._components_no_descend:
            cls._blocks_no_descend[obj] = \
                [obj] + cls._components_no_descend[obj][IBlock]

        cls._blocks = ComponentMap()
        for obj in cls._components:
            cls._blocks[obj] = \
                [obj] + cls._components[obj][IBlock]

    def test_init(self):
        b = block()
        self.assertTrue(b.parent is None)
        self.assertEqual(b.ctype, IBlock)

    def test_type(self):
        b = block()
        self.assertTrue(isinstance(b, ICategorizedObject))
        self.assertTrue(isinstance(b, ICategorizedObjectContainer))
        self.assertTrue(isinstance(b, IHeterogeneousContainer))
        self.assertTrue(isinstance(b, IBlock))

    def test_overwrite(self):
        b = block()
        v = b.v = variable()
        self.assertIs(v.parent, b)
        b.v = variable()
        self.assertTrue(v.parent is None)

        # the same component can overwrite itself
        b = block()
        v = b.v = variable()
        self.assertIs(v.parent, b)
        b.v = v
        self.assertTrue(v.parent is b)

        b = block()
        c = b.c = constraint()
        self.assertIs(c.parent, b)
        b.c = constraint()
        self.assertTrue(c.parent is None)

        # the same component can overwrite itself
        b = block()
        c = b.c = constraint()
        self.assertIs(c.parent, b)
        b.c = c
        self.assertTrue(c.parent is b)

        b = block()
        v = b.v = variable()
        self.assertIs(v.parent, b)
        b.v = constraint()
        self.assertTrue(v.parent is None)

        b = block()
        c = b.c = variable()
        self.assertIs(c.parent, b)
        b.c = variable()
        self.assertTrue(c.parent is None)

    def test_already_has_parent(self):
        b1 = block()
        v = b1.v = variable()
        b2 = block()
        with self.assertRaises(ValueError):
            b2.v = v
        self.assertTrue(v.parent is b1)
        del b1.v
        b2.v = v
        self.assertTrue(v.parent is b2)

    def test_delattr(self):
        b = block()
        with self.assertRaises(AttributeError):
            del b.not_an_attribute
        c = b.b = block()
        self.assertIs(c.parent, b)
        del b.b
        self.assertIs(c.parent, None)
        b.b = c
        self.assertIs(c.parent, b)
        b.x = 2
        self.assertTrue(hasattr(b, 'x'))
        self.assertEqual(b.x, 2)
        del b.x
        self.assertTrue(not hasattr(b, 'x'))

    def test_collect_ctypes_small_block_storage(self):
        b = block()
        self.assertEqual(b.collect_ctypes(active=None),
                         set())
        self.assertEqual(b.collect_ctypes(),
                         set())
        self.assertEqual(b.collect_ctypes(active=True),
                         set())
        b.x = variable()
        self.assertEqual(b.collect_ctypes(active=None),
                         set([IVariable]))
        self.assertEqual(b.collect_ctypes(),
                         set([IVariable]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([IVariable]))
        b.y = constraint()
        self.assertEqual(b.collect_ctypes(active=None),
                         set([IVariable, IConstraint]))
        self.assertEqual(b.collect_ctypes(),
                         set([IVariable, IConstraint]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([IVariable, IConstraint]))
        b.y.deactivate()
        self.assertEqual(b.collect_ctypes(active=None),
                         set([IVariable, IConstraint]))
        self.assertEqual(b.collect_ctypes(),
                         set([IVariable]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([IVariable]))
        B = block()
        B.b = b
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=None),
                         set([IBlock]))
        self.assertEqual(B.collect_ctypes(descend_into=False),
                         set([IBlock]))
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=True),
                         set([IBlock]))
        self.assertEqual(B.collect_ctypes(active=None),
                         set([IBlock, IVariable, IConstraint]))
        self.assertEqual(B.collect_ctypes(),
                         set([IBlock, IVariable]))
        self.assertEqual(B.collect_ctypes(active=True),
                         set([IBlock, IVariable]))
        b.deactivate()
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=None),
                         set([IBlock]))
        self.assertEqual(B.collect_ctypes(descend_into=False),
                         set([]))
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=True),
                         set([]))
        self.assertEqual(B.collect_ctypes(active=None),
                         set([IBlock, IVariable, IConstraint]))
        self.assertEqual(B.collect_ctypes(),
                         set([]))
        self.assertEqual(B.collect_ctypes(active=True),
                         set([]))
        B.x = variable()
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=None),
                         set([IBlock, IVariable]))
        self.assertEqual(B.collect_ctypes(descend_into=False),
                         set([IVariable]))
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=True),
                         set([IVariable]))
        self.assertEqual(B.collect_ctypes(active=None),
                         set([IBlock, IVariable, IConstraint]))
        self.assertEqual(B.collect_ctypes(),
                         set([IVariable]))
        self.assertEqual(B.collect_ctypes(active=True),
                         set([IVariable]))
        del b.y
        self.assertEqual(b.collect_ctypes(active=None),
                         set([IVariable]))
        self.assertEqual(b.collect_ctypes(),
                         set([]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([]))
        b.activate()
        self.assertEqual(b.collect_ctypes(active=None),
                         set([IVariable]))
        self.assertEqual(b.collect_ctypes(),
                         set([IVariable]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([IVariable]))

        del b.x
        self.assertEqual(b.collect_ctypes(), set())

    def test_collect_ctypes_large_block_storage(self):
        b = block()
        b._activate_large_storage_mode()
        self.assertEqual(b.collect_ctypes(active=None),
                         set())
        self.assertEqual(b.collect_ctypes(),
                         set())
        self.assertEqual(b.collect_ctypes(active=True),
                         set())
        b.x = variable()
        self.assertEqual(b.collect_ctypes(active=None),
                         set([IVariable]))
        self.assertEqual(b.collect_ctypes(),
                         set([IVariable]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([IVariable]))
        b.y = constraint()
        self.assertEqual(b.collect_ctypes(active=None),
                         set([IVariable, IConstraint]))
        self.assertEqual(b.collect_ctypes(),
                         set([IVariable, IConstraint]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([IVariable, IConstraint]))
        b.y.deactivate()
        self.assertEqual(b.collect_ctypes(active=None),
                         set([IVariable, IConstraint]))
        self.assertEqual(b.collect_ctypes(),
                         set([IVariable]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([IVariable]))
        B = block()
        b._activate_large_storage_mode()
        B.b = b
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=None),
                         set([IBlock]))
        self.assertEqual(B.collect_ctypes(descend_into=False),
                         set([IBlock]))
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=True),
                         set([IBlock]))
        self.assertEqual(B.collect_ctypes(active=None),
                         set([IBlock, IVariable, IConstraint]))
        self.assertEqual(B.collect_ctypes(),
                         set([IBlock, IVariable]))
        self.assertEqual(B.collect_ctypes(active=True),
                         set([IBlock, IVariable]))
        b.deactivate()
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=None),
                         set([IBlock]))
        self.assertEqual(B.collect_ctypes(descend_into=False),
                         set([]))
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=True),
                         set([]))
        self.assertEqual(B.collect_ctypes(active=None),
                         set([IBlock, IVariable, IConstraint]))
        self.assertEqual(B.collect_ctypes(),
                         set([]))
        self.assertEqual(B.collect_ctypes(active=True),
                         set([]))
        B.x = variable()
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=None),
                         set([IBlock, IVariable]))
        self.assertEqual(B.collect_ctypes(descend_into=False),
                         set([IVariable]))
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=True),
                         set([IVariable]))
        self.assertEqual(B.collect_ctypes(active=None),
                         set([IBlock, IVariable, IConstraint]))
        self.assertEqual(B.collect_ctypes(),
                         set([IVariable]))
        self.assertEqual(B.collect_ctypes(active=True),
                         set([IVariable]))
        del b.y
        self.assertEqual(b.collect_ctypes(active=None),
                         set([IVariable]))
        self.assertEqual(b.collect_ctypes(),
                         set([]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([]))
        b.activate()
        self.assertEqual(b.collect_ctypes(active=None),
                         set([IVariable]))
        self.assertEqual(b.collect_ctypes(),
                         set([IVariable]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([IVariable]))

        del b.x
        self.assertEqual(b.collect_ctypes(), set())

class Test_block_noclone(_Test_block, unittest.TestCase):
    _do_clone = False

class Test_block_clone(_Test_block, unittest.TestCase):
    _do_clone = True

class _MyBlockBaseBase(block):
    __slots__ = ()
    def __init__(self):
        super(_MyBlockBaseBase, self).__init__()

class _MyBlockBase(_MyBlockBaseBase):
    __slots__ = ("b",)
    def __init__(self):
        super(_MyBlockBase, self).__init__()
        self.b = block()
        self.b.v = variable()

class _MyBlock(_MyBlockBase):
    # testing when a __dict__ might appear (no __slots__)
    def __init__(self):
        super(_MyBlock, self).__init__()
        self.b.blist = block_list()
        self.b.blist.append(block())
        self.v = variable()
        self.n = 2.0

class _Test_small_block(_Test_block_base):

    _do_clone = None

    @classmethod
    def setUpClass(cls):
        assert cls._do_clone is not None
        cls._myblock_type = _MyBlock
        model = cls._block = _MyBlock()

        if cls._do_clone:
            model = cls._block = model.clone()

        #
        # Manually encode the correct output
        # for tests in the base testing class
        #

        cls._preorder = [model,
                         model.b,
                         model.b.v,
                         model.b.blist,
                         model.b.blist[0],
                         model.v]

        cls._names = ComponentMap()
        cls._names[model.b] = "b"
        cls._names[model.b.v] = "b.v"
        cls._names[model.b.blist] = "b.blist"
        cls._names[model.b.blist[0]] = "b.blist[0]"
        cls._names[model.v] = "v"

        cls._children = ComponentMap()
        cls._children[model] = [model.b,
                                model.v]
        cls._children[model.b] = [model.b.v,
                                  model.b.blist]
        cls._children[model.b.blist] = [model.b.blist[0]]
        cls._children[model.b.blist[0]] = []

        cls._child_key = ComponentMap()
        cls._child_key[model.b] = "b"
        cls._child_key[model.b.v] = "v"
        cls._child_key[model.b.blist] = "blist"
        cls._child_key[model.b.blist[0]] = 0
        cls._child_key[model.v] = "v"

        cls._components_no_descend = ComponentMap()
        cls._components_no_descend[model] = {}
        cls._components_no_descend[model][IVariable] = [model.v]
        cls._components_no_descend[model][IBlock] = [model.b]
        cls._components_no_descend[model.b] = {}
        cls._components_no_descend[model.b][IVariable] = [model.b.v]
        cls._components_no_descend[model.b][IBlock] = [model.b.blist[0]]
        cls._components_no_descend[model.b.blist[0]] = {}
        cls._components_no_descend[model.b.blist[0]][IBlock] = []
        cls._components_no_descend[model.b.blist[0]][IVariable] = []

        cls._components = ComponentMap()
        cls._components[model] = {}
        cls._components[model][IVariable] = [model.v, model.b.v]
        cls._components[model][IBlock] = [model.b, model.b.blist[0]]
        cls._components[model.b] = {}
        cls._components[model.b][IVariable] = [model.b.v]
        cls._components[model.b][IBlock] = [model.b.blist[0]]
        cls._components[model.b.blist[0]] = {}
        cls._components[model.b.blist[0]][IBlock] = []
        cls._components[model.b.blist[0]][IVariable] = []

        cls._blocks_no_descend = ComponentMap()
        for obj in cls._components_no_descend:
            cls._blocks_no_descend[obj] = \
                [obj] + cls._components_no_descend[obj][IBlock]

        cls._blocks = ComponentMap()
        for obj in cls._components:
            cls._blocks[obj] = \
                [obj] + cls._components[obj][IBlock]

    # override this test method on the base class
    def test_collect_ctypes(self):
        self.assertEqual(self._block.collect_ctypes(active=None),
                         set([IBlock, IVariable]))
        self.assertEqual(self._block.collect_ctypes(),
                         set([IBlock, IVariable]))
        self.assertEqual(self._block.collect_ctypes(active=True),
                         set([IBlock, IVariable]))
        self.assertEqual(self._block.collect_ctypes(descend_into=False),
                         set([IBlock, IVariable]))
        self.assertEqual(self._block.collect_ctypes(active=True,
                                                    descend_into=False),
                         set([IBlock, IVariable]))

        self._block.b.deactivate()
        try:
            self.assertEqual(self._block.collect_ctypes(active=None),
                             set([IBlock, IVariable]))
            self.assertEqual(self._block.collect_ctypes(),
                             set([IVariable]))
            self.assertEqual(self._block.collect_ctypes(active=True),
                             set([IVariable]))
            self.assertEqual(self._block.collect_ctypes(active=None,
                                                        descend_into=False),
                             set([IBlock, IVariable]))
            self.assertEqual(self._block.collect_ctypes(descend_into=False),
                             set([IVariable]))
            self.assertEqual(self._block.collect_ctypes(active=True,
                                                        descend_into=False),
                             set([IVariable]))
        finally:
            # use a finally block in case there is a failure above
            self._block.b.activate()

    def test_customblock_delattr(self):
        b = _MyBlock()
        with self.assertRaises(AttributeError):
            del b.not_an_attribute
        c = b.b
        self.assertIs(c.parent, b)
        del b.b
        self.assertIs(c.parent, None)
        b.b = c
        self.assertIs(c.parent, b)
        b.x = 2
        self.assertTrue(hasattr(b, 'x'))
        self.assertEqual(b.x, 2)
        del b.x
        self.assertTrue(not hasattr(b, 'x'))

    def test_customblock_setattr(self):
        b = _MyBlockBase()
        self.assertIs(b.b.parent, b)
        self.assertIs(b.b.v.parent, b.b)
        with self.assertRaises(ValueError):
            b.b = b.b.v
        self.assertIs(b.b.parent, b)
        self.assertIs(b.b.v.parent, b.b)
        c = b.b
        self.assertIs(c.parent, b)
        # test the edge case in setattr
        b.b = c
        self.assertIs(c.parent, b)
        assert not hasattr(b,"g")
        with self.assertRaises(ValueError):
            b.g = b.b
        self.assertIs(b.b.parent, b)
        b.g = 1
        with self.assertRaises(ValueError):
            b.g = b.b
        self.assertEqual(b.g, 1)
        self.assertIs(b.b.parent, b)
        # test an overwrite
        b.b = block()
        self.assertIs(c.parent, None)
        self.assertIs(b.b.parent, b)

    def test_customblock__with_dict_setattr(self):
        # This one was given a __dict__
        b = _MyBlock()
        self.assertIs(b.v.parent, b)
        self.assertIs(b.b.parent, b)
        with self.assertRaises(ValueError):
            b.v = b.b
        self.assertIs(b.v.parent, b)
        self.assertIs(b.b.parent, b)
        b.not_an_attribute = 2
        v = b.v
        self.assertIs(v.parent, b)
        # test the edge case in setattr
        b.v = v
        self.assertIs(v.parent, b)
        # test an overwrite
        b.v = variable()
        self.assertIs(v.parent, None)
        self.assertIs(b.v.parent, b)

    def test_inactive_behavior(self):
        b = _MyBlock()
        b.deactivate()
        self.assertNotEqual(len(list(pmo.preorder_traversal(b,
                                                            active=None))), 0)
        self.assertEqual(len(list(pmo.preorder_traversal(b))), 0)
        self.assertEqual(len(list(pmo.preorder_traversal(b,
                                                         active=True))), 0)

        def descend(x):
            return True
        self.assertNotEqual(
            len(list(pmo.preorder_traversal(b,
                                            active=None,
                                            descend=descend))),
            0)
        self.assertEqual(
            len(list(pmo.preorder_traversal(b,
                                            descend=descend))),
            0)
        self.assertEqual(
            len(list(pmo.preorder_traversal(b,
                                            active=True,
                                            descend=descend))),
            0)
        def descend(x):
            descend.seen.append(x)
            return x.active
        descend.seen = []
        self.assertEqual(
            len(list(pmo.preorder_traversal(b,
                                            active=None,
                                            descend=descend))),
            1)
        self.assertEqual(len(descend.seen), 1)
        self.assertIs(descend.seen[0], b)

        self.assertNotEqual(len(list(b.components(active=None))), 0)
        self.assertEqual(len(list(b.components())), 0)
        self.assertEqual(len(list(b.components(active=True))), 0)

        self.assertNotEqual(len(list(pmo.generate_names(b, active=None))), 0)
        self.assertEqual(len(list(pmo.generate_names(b))), 0)
        self.assertEqual(len(list(pmo.generate_names(b, active=True))), 0)

class Test_small_block_noclone(_Test_small_block, unittest.TestCase):
    _do_clone = False

class Test_small_block_clone(_Test_small_block, unittest.TestCase):
    _do_clone = True

class Test_block_dict(_TestActiveDictContainerBase,
                      unittest.TestCase):
    _container_type = block_dict
    _ctype_factory = lambda self: block()

class Test_block_tuple(_TestActiveTupleContainerBase,
                       unittest.TestCase):
    _container_type = block_tuple
    _ctype_factory = lambda self: block()

class Test_block_list(_TestActiveListContainerBase,
                      unittest.TestCase):
    _container_type = block_list
    _ctype_factory = lambda self: block()

if __name__ == "__main__":
    unittest.main()
