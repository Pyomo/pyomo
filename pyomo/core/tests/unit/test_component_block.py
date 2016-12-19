import tempfile
import os
import pickle

import pyutilib.th as unittest
from pyomo.core.base.component_interface import (ICategorizedObject,
                                                 IActiveObject,
                                                 IComponent,
                                                 IComponentContainer,
                                                 _IActiveComponentContainer)
from pyomo.core.tests.unit.test_component_dict import \
    _TestActiveComponentDictBase
from pyomo.core.tests.unit.test_component_list import \
    _TestActiveComponentListBase
from pyomo.core.base.component_map import ComponentMap
from pyomo.core.base.component_suffix import suffix
from pyomo.core.base.component_constraint import (constraint,
                                                  constraint_dict,
                                                  constraint_list)
from pyomo.core.base.component_parameter import (parameter,
                                                 parameter_dict,
                                                 parameter_list)
from pyomo.core.base.component_expression import (expression,
                                                  data_expression,
                                                  expression_dict,
                                                  expression_list)
from pyomo.core.base.component_objective import (objective,
                                                 objective_dict,
                                                 objective_list)
from pyomo.core.base.component_variable import (IVariable,
                                                variable,
                                                variable_dict,
                                                variable_list)
from pyomo.core.base.component_block import (IBlockStorage,
                                             block,
                                             block_dict,
                                             block_list,
                                             StaticBlock)
from pyomo.core.base.block import Block
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
import pyomo.core.base.expr

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
    if isinstance(exp, IComponent):
        ans[id(exp)] = exp
    if exp.is_expression():
        if exp.__class__ is pyomo.core.base.expr._ProductExpression:
            for subexp in exp._numerator:
                ans.update(_collect_expr_components(subexp))
            for subexp in exp._denominator:
                ans.update(_collect_expr_components(subexp))
        else:
            for subexp in exp._args:
                ans.update(_collect_expr_components(subexp))
    return ans

class TestMisc(unittest.TestCase):

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
        b.c1 = constraint(b.el + 1 <= b.e + 2 <= b.eu + 2)
        b.c2 = constraint(lb=b.el, body=b.v)
        b.c3 = constraint(body=b.v, ub=b.eu)
        b.dual = suffix(direction=suffix.IMPORT)

        import pyomo.environ
        #import pyomo.opt.base
        from pyomo.opt.base.solvers import UnknownSolver
        from pyomo.opt import SolverFactory
        from pyomo.opt import SolverStatus, TerminationCondition

        opt = SolverFactory("glpk")
        if isinstance(opt, UnknownSolver):
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
        if isinstance(opt, UnknownSolver):
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
        if isinstance(opt, UnknownSolver):
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
        b.c2 = constraint()

        self.assertEqual(
            [obj.name for obj in b.preorder_traversal()],
            [None, 'v', 'c1', 'c2'])
        self.assertEqual(
            [obj.name for obj in b.preorder_traversal(active=True)],
            [None, 'v', 'c2'])
        self.assertEqual(
            [obj.name for obj in b.preorder_traversal(ctype=Constraint)],
            [None, 'c1', 'c2'])
        self.assertEqual(
            [obj.name for obj in b.preorder_traversal(ctype=Constraint,
                                                      active=True)],
            [None, 'c2'])

        self.assertEqual(
            [obj.name for obj in b.postorder_traversal()],
            ['v', 'c1', 'c2', None])
        self.assertEqual(
            [obj.name for obj in b.postorder_traversal(active=True)],
            ['v', 'c2', None])
        self.assertEqual(
            [obj.name for obj in b.postorder_traversal(ctype=Constraint)],
            ['c1', 'c2', None])
        self.assertEqual(
            [obj.name for obj in b.postorder_traversal(ctype=Constraint,
                                                       active=True)],
            ['c2', None])

    # test how clone behaves when there are
    # references to components on a different block
    def test_clone1(self):
        b = block()
        b.v = variable()
        b.b = block()
        b.b.e = expression(b.v**2)
        b.b.v = variable()
        b.bdict = block_dict(ordered=True)
        b.bdict[0] = block()
        b.bdict[0].e = expression(b.v**2)
        b.blist = block_list()
        b.blist.append(block())
        b.blist[0].e = expression(b.v**2 + b.b.v**2)

        bc = b.clone()
        self.assertIsNot(b.v, bc.v)
        self.assertIs(b.v.root_block, b)
        self.assertIs(bc.v.root_block, bc)
        self.assertIsNot(b.b.e, bc.b.e)
        self.assertIs(b.b.e.root_block, b)
        self.assertIs(bc.b.e.root_block, bc)
        self.assertIsNot(b.bdict[0].e, bc.bdict[0].e)
        self.assertIs(b.bdict[0].e.root_block, b)
        self.assertIs(bc.bdict[0].e.root_block, bc)
        self.assertIsNot(b.blist[0].e, bc.blist[0].e)
        self.assertIs(b.blist[0].e.root_block, b)
        self.assertIs(bc.blist[0].e.root_block, bc)

        #
        # check that the expressions on cloned sub-blocks
        # reference the original variables for blocks "out-of-scope"
        #
        b_b = b.b.clone()
        self.assertIsNot(b_b.e, b.b.e)
        self.assertIs(b.b.e.root_block, b)
        self.assertTrue(len(_collect_expr_components(b.b.e.expr)) == 1)
        self.assertIs(list(_collect_expr_components(b.b.e.expr).values())[0],
                      b.v)
        self.assertIs(b_b.e.root_block, b_b)
        self.assertTrue(len(_collect_expr_components(b_b.e.expr)) == 1)
        self.assertIs(list(_collect_expr_components(b_b.e.expr).values())[0],
                      b.v)

        b_bdict0 = b.bdict[0].clone()
        self.assertIsNot(b_bdict0.e, b.bdict[0].e)
        self.assertIs(b.bdict[0].e.root_block, b)
        self.assertTrue(len(_collect_expr_components(b.bdict[0].e.expr)) == 1)
        self.assertIs(list(_collect_expr_components(b.bdict[0].e.expr).values())[0],
                      b.v)
        self.assertIs(b_bdict0.e.root_block, b_bdict0)
        self.assertTrue(len(_collect_expr_components(b_bdict0.e.expr)) == 1)
        self.assertIs(list(_collect_expr_components(b_bdict0.e.expr).values())[0],
                      b.v)

        b_blist0 = b.blist[0].clone()
        self.assertIsNot(b_blist0.e, b.blist[0].e)
        self.assertIs(b.blist[0].e.root_block, b)
        self.assertTrue(len(_collect_expr_components(b.blist[0].e.expr)) == 2)
        self.assertEqual(sorted(list(id(v_) for v_ in _collect_expr_components(b.blist[0].e.expr).values())),
                         sorted(list(id(v_) for v_ in [b.v, b.b.v])))
        self.assertIs(b_blist0.e.root_block, b_blist0)
        self.assertTrue(len(_collect_expr_components(b_blist0.e.expr)) == 2)
        self.assertEqual(sorted(list(id(v_) for v_ in _collect_expr_components(b_blist0.e.expr).values())),
                         sorted(list(id(v_) for v_ in [b.v, b.b.v])))

    # test bulk clone behavior
    def test_clone2(self):
        b = block()
        b.v = variable()
        b.vdict = variable_dict(((i, variable())
                                for i in range(10)),
                                ordered=True)
        b.vlist = variable_list(variable()
                                for i in range(10))
        b.o = objective(b.v + b.vdict[0] + b.vlist[0])
        b.odict = objective_dict(((i, objective(b.v + b.vdict[i]))
                                 for i in b.vdict),
                                 ordered=True)
        b.olist = objective_list(objective(b.v + v_)
                                 for i,v_ in enumerate(b.vdict))
        b.c = constraint(b.v >= 1)
        b.cdict = constraint_dict(((i, constraint(b.vdict[i] == i))
                                  for i in b.vdict),
                                  ordered=True)
        b.clist = constraint_list(constraint(0 <= v_ <= i)
                                  for i, v_ in enumerate(b.vlist))
        b.p = parameter()
        b.pdict = parameter_dict(((i, parameter(i))
                                 for i in b.vdict),
                                 ordered=True)
        b.plist = parameter_list(parameter(i)
                                 for i in range(len(b.vlist)))
        b.e = expression(b.v * b.p + 1)
        b.edict = expression_dict(((i, expression(b.vdict[i] * b.pdict[i] + 1))
                                  for i in b.vdict),
                                  ordered=True)
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
            self.assertIs(c1.root_block, b)
            self.assertIs(c2.parent, bc)
            self.assertIs(c2.root_block, bc)
            self.assertIsNot(c1, c2)
            self.assertEqual(c1.name, c2.name)

        self.assertEqual(len(list(b.components())),
                         len(list(bc.components())))
        for c1, c2 in zip(b.components(), bc.components()):
            self.assertIs(c1.root_block, b)
            self.assertIs(c2.root_block, bc)
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
                    self.assertIs(subc1.root_block, b)
                    self.assertIs(subc2.root_block, bc)

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
            self.assertIs(c1.root_block, b)
            self.assertIs(c2.parent, bcc)
            self.assertIs(c2.root_block, bcc)
            self.assertIsNot(c1, c2)
            self.assertEqual(c1.name, c2.name)

        self.assertEqual(len(list(b.components())),
                         len(list(bcc.components())))
        self.assertTrue(hasattr(bcc, 'bc'))
        for c1, c2 in zip(b.components(), bcc.components()):
            self.assertIs(c1.root_block, b)
            self.assertIs(c2.root_block, bcc)
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
                    self.assertIs(subc1.root_block, b)
                    self.assertIs(subc2.root_block, bcc)

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
            self.assertIs(c1.root_block, bc_init)
            self.assertIs(c2.parent, sub_bc)
            self.assertIs(c2.root_block, sub_bc)
            self.assertIsNot(c1, c2)
            self.assertEqual(c1.name, c2.name)

        self.assertEqual(len(list(bc_init.components())),
                         len(list(sub_bc.components())))
        for c1, c2 in zip(bc_init.components(), sub_bc.components()):
            self.assertIs(c1.root_block, bc_init)
            self.assertIs(c2.root_block, sub_bc)
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
                    self.assertIs(subc1.root_block, bc_init)
                    self.assertIs(subc2.root_block, sub_bc)

class _Test_block_base(object):

    _children = None
    _child_key = None
    _components_no_descend = None
    _components = None
    _blocks_no_descend = None
    _blocks = None
    _block = None

    def test_clone(self):
        b = self._block
        bc = b.clone()
        self.assertIsNot(b, bc)
        self.assertEqual(len(list(b.children())),
                         len(list(bc.children())))
        for c1, c2 in zip(b.children(), bc.children()):
            self.assertIs(c1.parent, b)
            self.assertIs(c1.root_block, b)
            self.assertIs(c2.parent, bc)
            self.assertIs(c2.root_block, bc)
            self.assertIsNot(c1, c2)
            self.assertEqual(c1.name, c2.name)

        self.assertEqual(len(list(b.components())),
                         len(list(bc.components())))
        for c1, c2 in zip(b.components(), bc.components()):
            self.assertIs(c1.root_block, b)
            self.assertIs(c2.root_block, bc)
            self.assertIsNot(c1, c2)
            self.assertEqual(c1.name, c2.name)

    def test_pickle(self):
        b = pickle.loads(
            pickle.dumps(self._block))
        self.assertEqual(len(list(b.preorder_traversal())),
                         len(self._names)+1)
        self.assertEqual(len(list(b.postorder_traversal())),
                         len(self._names)+1)
        names = b.generate_names()
        self.assertEqual(sorted(names.values()),
                         sorted(self._names.values()))

    def test_preorder_traversal(self):
        # this first test makes failures a
        # little easier to debug
        self.assertEqual(
            [str(obj) for obj in self._block.preorder_traversal()],
            [str(obj) for obj in self._preorder])
        self.assertEqual(
            [id(obj) for obj in self._block.preorder_traversal()],
            [id(obj) for obj in self._preorder])
        self.assertEqual(
            [(k,id(obj)) for k,obj in self._block.preorder_traversal(return_key=True)],
            [((obj.parent.child_key(obj) if obj.parent is not None else None),id(obj)) for obj in self._preorder])

        # this first test makes failures a
        # little easier to debug
        self.assertEqual(
            [str(obj) for obj in self._block.preorder_traversal(ctype=Var)],
            [str(obj) for obj in self._preorder if obj.ctype in (Block, Var)])
        self.assertEqual(
            [id(obj) for obj in self._block.preorder_traversal(ctype=Var)],
            [id(obj) for obj in self._preorder if obj.ctype in (Block, Var)])

        # this first test makes failures a
        # little easier to debug
        self.assertEqual(
            [str(obj) for obj in self._block.preorder_traversal(ctype=Var,
                                                                include_parent_blocks=False)],
            [str(obj) for obj in self._preorder if obj.ctype is Var])
        self.assertEqual(
            [id(obj) for obj in self._block.preorder_traversal(ctype=Var,
                                                               include_parent_blocks=False)],
            [id(obj) for obj in self._preorder if obj.ctype is Var])

    def test_postorder_traversal(self):
        # this first test makes failures a
        # little easier to debug
        self.assertEqual(
            [str(obj) for obj in self._block.postorder_traversal()],
            [str(obj) for obj in self._postorder])
        self.assertEqual(
            [id(obj) for obj in self._block.postorder_traversal()],
            [id(obj) for obj in self._postorder])
        self.assertEqual(
            [(k,id(obj)) for k,obj in self._block.postorder_traversal(return_key=True)],
            [((obj.parent.child_key(obj) if obj.parent is not None else None),id(obj)) for obj in self._postorder])

        # this first test makes failures a
        # little easier to debug
        self.assertEqual(
            [str(obj) for obj in self._block.postorder_traversal(ctype=Var)],
            [str(obj) for obj in self._postorder if obj.ctype in (Block, Var)])
        self.assertEqual(
            [id(obj) for obj in self._block.postorder_traversal(ctype=Var)],
            [id(obj) for obj in self._postorder if obj.ctype in (Block, Var)])

        # this first test makes failures a
        # little easier to debug
        self.assertEqual(
            [str(obj) for obj in self._block.postorder_traversal(ctype=Var,
                                                                include_parent_blocks=False)],
            [str(obj) for obj in self._postorder if obj.ctype is Var])
        self.assertEqual(
            [id(obj) for obj in self._block.postorder_traversal(ctype=Var,
                                                               include_parent_blocks=False)],
            [id(obj) for obj in self._postorder if obj.ctype is Var])

    def test_child_key(self):
        for child in self._child_key:
            parent = child.parent
            self.assertTrue(parent is not None)
            self.assertTrue(id(child) in set(
                id(_c) for _c in self._children[parent]))
            self.assertEqual(self._child_key[child],
                             parent.child_key(child))

    def test_child_key_no_entry(self):
        v = variable()
        with self.assertRaises(ValueError):
            self._block.child_key(v)

    def test_children(self):
        for obj in self._children:
            self.assertTrue(isinstance(obj, IComponentContainer))
            if isinstance(obj, IBlockStorage):
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
                self.assertEqual(
                    set((key,id(child)) for key,child in obj.children(return_key=True)),
                    set((self._child_key[child],id(child)) for child in obj.children()))
                # this first test makes failures a
                # little easier to debug
                self.assertEqual(
                    sorted(str(child)
                           for child in obj.children(ctype=Block)),
                    sorted(str(child)
                           for child in self._children[obj]
                           if child.ctype is Block))
                self.assertEqual(
                    set(id(child) for child in obj.children(ctype=Block)),
                    set(id(child) for child in self._children[obj]
                        if child.ctype is Block))
                self.assertEqual(
                    set((key,id(child)) for key,child in obj.children(ctype=Block,
                                                                      return_key=True)),
                    set((self._child_key[child],id(child)) for child in obj.children(ctype=Block)
                         if child.ctype is Block))
                # this first test makes failures a
                # little easier to debug
                self.assertEqual(
                    sorted(str(child)
                           for child in obj.children(ctype=Var)),
                    sorted(str(child)
                           for child in self._children[obj]
                           if child.ctype is Var))
                self.assertEqual(
                    set(id(child) for child in obj.children(ctype=Var)),
                    set(id(child) for child in self._children[obj]
                        if child.ctype is Var))
                self.assertEqual(
                    set((key,id(child)) for key,child in obj.children(ctype=Var,
                                                                      return_key=True)),
                    set((self._child_key[child],id(child)) for child in obj.children(ctype=Var)
                         if child.ctype is Var))
            elif isinstance(obj, IComponentContainer):
                for child in obj.children():
                    self.assertTrue(child.parent is obj)
                self.assertEqual(
                    set(id(child) for child in obj.children()),
                    set(id(child) for child in self._children[obj]))
                self.assertEqual(
                    set((key,id(child)) for key,child in obj.children(return_key=True)),
                    set((self._child_key[child],id(child)) for child in obj.children()))
            else:
                self.assertEqual(len(self._children[obj]), 0)

    def test_components_no_descend_active_None(self):
        for obj in self._components_no_descend:
            self.assertTrue(isinstance(obj, IComponentContainer))
            self.assertTrue(isinstance(obj, IBlockStorage))
            for c in obj.components(descend_into=False):
                self.assertTrue(
                    _path_to_object_exists(obj, c))
            # test ctype=Block
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.components(ctype=Block,
                                      active=None,
                                      descend_into=False)),
                sorted(str(_b)
                       for _b in
                       self._components_no_descend[obj][Block]))
            self.assertEqual(
                set(id(_b) for _b in
                    obj.components(ctype=Block,
                                   active=None,
                                   descend_into=False)),
                set(id(_b) for _b in
                    self._components_no_descend[obj][Block]))
            # test ctype=Var
            self.assertEqual(
                sorted(str(_v)
                       for _v in
                       obj.components(ctype=Var,
                                      active=None,
                                      descend_into=False)),
                sorted(str(_v)
                       for _v in
                       self._components_no_descend[obj][Var]))
            self.assertEqual(
                set(id(_v) for _v in
                    obj.components(ctype=Var,
                                   active=None,
                                   descend_into=False)),
                set(id(_v) for _v in
                    self._components_no_descend[obj][Var]))
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
            self.assertTrue(isinstance(obj, IComponentContainer))
            self.assertTrue(isinstance(obj, IBlockStorage))
            # test ctype=Block
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.components(ctype=Block,
                                      active=True,
                                      descend_into=False)),
                sorted(str(_b)
                       for _b in
                       self._components_no_descend[obj][Block]
                       if _b.active)
                if getattr(obj, 'active', True) else [])
            self.assertEqual(
                set(id(_b) for _b in
                    obj.components(ctype=Block,
                                   active=True,
                                   descend_into=False)),
                set(id(_b) for _b in
                    self._components_no_descend[obj][Block]
                    if _b.active)
                if getattr(obj, 'active', True) else set())
            # test ctype=Var
            self.assertEqual(
                sorted(str(_v)
                       for _v in
                       obj.components(ctype=Var,
                                      active=True,
                                      descend_into=False)),
                sorted(str(_v)
                       for _v in
                       self._components_no_descend[obj][Var])
                if getattr(obj, 'active', True) else [])
            self.assertEqual(
                set(id(_v) for _v in
                    obj.components(ctype=Var,
                                   active=True,
                                   descend_into=False)),
                set(id(_v) for _v in
                    self._components_no_descend[obj][Var])
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
            self.assertTrue(isinstance(obj, IComponentContainer))
            self.assertTrue(isinstance(obj, IBlockStorage))
            for c in obj.components(descend_into=True):
                self.assertTrue(
                    _path_to_object_exists(obj, c))
            # test ctype=Block
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.components(ctype=Block,
                                      active=None,
                                      descend_into=True)),
                sorted(str(_b)
                       for _b in
                       self._components[obj][Block]))
            self.assertEqual(
                set(id(_b) for _b in
                    obj.components(ctype=Block,
                                   active=None,
                                   descend_into=True)),
                set(id(_b) for _b in
                    self._components[obj][Block]))
            # test ctype=Var
            self.assertEqual(
                sorted(str(_v)
                       for _v in
                       obj.components(ctype=Var,
                                      active=None,
                                      descend_into=True)),
                sorted(str(_v)
                       for _v in
                       self._components[obj][Var]))
            self.assertEqual(
                set(id(_v) for _v in
                    obj.components(ctype=Var,
                                   active=None,
                                   descend_into=True)),
                set(id(_v) for _v in
                    self._components[obj][Var]))
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
            self.assertTrue(isinstance(obj, IComponentContainer))
            self.assertTrue(isinstance(obj, IBlockStorage))
            # test ctype=Block
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.components(ctype=Block,
                                      active=True,
                                      descend_into=True)),
                sorted(str(_b)
                       for _b in
                       self._components[obj][Block]
                       if _b.active)
                if getattr(obj, 'active', True) else [])
            self.assertEqual(
                set(id(_b) for _b in
                    obj.components(ctype=Block,
                                   active=True,
                                   descend_into=True)),
                set(id(_b) for _b in
                    self._components[obj][Block]
                    if _b.active)
                if getattr(obj, 'active', True) else set())
            # test ctype=Var
            self.assertEqual(
                sorted(str(_v)
                       for _v in
                       obj.components(ctype=Var,
                                      active=True,
                                      descend_into=True)),
                sorted(str(_v)
                       for _v in
                       self._components[obj][Var]
                       if _active_path_to_object_exists(obj, _v))
                if getattr(obj, 'active', True) else [])
            self.assertEqual(
                set(id(_v) for _v in
                    obj.components(ctype=Var,
                                   active=True,
                                   descend_into=True)),
                set(id(_v) for _v in
                    self._components[obj][Var]
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

    def test_blocks_no_descend_active_None(self):
        for obj in self._blocks_no_descend:
            self.assertTrue(isinstance(obj, IComponentContainer))
            self.assertTrue(isinstance(obj, IBlockStorage))
            for c in obj.blocks(descend_into=True):
                self.assertTrue(
                    _path_to_object_exists(obj, c))
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.blocks(active=None,
                                  descend_into=False)),
                sorted(str(_b)
                       for _b in
                       self._blocks_no_descend[obj]))
            self.assertEqual(
                set(id(_b) for _b in
                    obj.blocks(active=None,
                               descend_into=False)),
                set(id(_b) for _b in
                    self._blocks_no_descend[obj]))

    def test_blocks_no_descend_active_True(self):
        for obj in self._blocks_no_descend:
            self.assertTrue(isinstance(obj, IComponentContainer))
            self.assertTrue(isinstance(obj, IBlockStorage))
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.blocks(active=True,
                                  descend_into=False)),
                sorted(str(_b)
                       for _b in
                       self._blocks_no_descend[obj]
                       if _b.active)
                if getattr(obj, 'active', True) else [])
            self.assertEqual(
                set(id(_b) for _b in
                    obj.blocks(active=True,
                               descend_into=False)),
                set(id(_b) for _b in
                    self._blocks_no_descend[obj]
                    if _b.active)
                if getattr(obj, 'active', True) else set())

    def test_blocks_active_None(self):
        for obj in self._blocks:
            self.assertTrue(isinstance(obj, IComponentContainer))
            self.assertTrue(isinstance(obj, IBlockStorage))
            for c in obj.blocks(descend_into=True):
                self.assertTrue(
                    _path_to_object_exists(obj, c))
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.blocks(active=None,
                                  descend_into=True)),
                sorted(str(_b)
                       for _b in
                       self._blocks[obj]))
            self.assertEqual(
                set(id(_b) for _b in
                    obj.blocks(active=None,
                               descend_into=True)),
                set(id(_b) for _b in
                    self._blocks[obj]))

    def test_blocks_active_True(self):
        for obj in self._blocks:
            self.assertTrue(isinstance(obj, IComponentContainer))
            self.assertTrue(isinstance(obj, IBlockStorage))
            self.assertEqual(
                sorted(str(_b)
                       for _b in
                       obj.blocks(active=True,
                                  descend_into=True)),
                sorted(str(_b)
                       for _b in
                       self._blocks[obj]
                       if _b.active)
                if getattr(obj, 'active', True) else [])
            self.assertEqual(
                set(id(_b) for _b in
                    obj.blocks(active=True,
                               descend_into=True)),
                set(id(_b) for _b in
                    self._blocks[obj]
                    if _b.active)
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
        model.b_1.b_2.deactivate()
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

        cls._postorder = [model.v_1,
                          model.vdict_1[None],
                          model.vdict_1,
                          model.vlist_1[0],
                          model.vlist_1[1],
                          model.vlist_1,
                          model.b_1.v_2,
                          model.b_1.b_2.b_3,
                          model.b_1.b_2.v_3,
                          model.b_1.b_2.vlist_3[0],
                          model.b_1.b_2.vlist_3,
                          model.b_1.b_2,
                          model.b_1,
                          model.bdict_1,
                          model.blist_1[0].v_2,
                          model.blist_1[0].b_2,
                          model.blist_1[0],
                          model.blist_1,
                          model]
        assert len(cls._preorder) == len(cls._postorder)

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
        cls._components_no_descend[model][Var] = \
            [model.v_1,
             model.vdict_1[None],
             model.vlist_1[0],
             model.vlist_1[1]]
        cls._components_no_descend[model][Block] = \
            [model.b_1,
             model.blist_1[0]]
        cls._components_no_descend[model.b_1] = {}
        cls._components_no_descend[model.b_1][Var] = \
            [model.b_1.v_2]
        cls._components_no_descend[model.b_1][Block] = \
            [model.b_1.b_2]
        cls._components_no_descend[model.b_1.b_2] = {}
        cls._components_no_descend[model.b_1.b_2][Var] = \
            [model.b_1.b_2.v_3,
             model.b_1.b_2.vlist_3[0]]
        cls._components_no_descend[model.b_1.b_2][Block] = \
            [model.b_1.b_2.b_3]
        cls._components_no_descend[model.b_1.b_2.b_3] = {}
        cls._components_no_descend[model.b_1.b_2.b_3][Var] = []
        cls._components_no_descend[model.b_1.b_2.b_3][Block] = []
        cls._components_no_descend[model.blist_1[0]] = {}
        cls._components_no_descend[model.blist_1[0]][Var] = \
            [model.blist_1[0].v_2]
        cls._components_no_descend[model.blist_1[0]][Block] = \
            [model.blist_1[0].b_2]
        cls._components_no_descend[model.blist_1[0].b_2] = {}
        cls._components_no_descend[model.blist_1[0].b_2][Var] = []
        cls._components_no_descend[model.blist_1[0].b_2][Block] = []

        cls._components = ComponentMap()
        cls._components[model] = {}
        cls._components[model][Var] = \
            [model.v_1,
             model.vdict_1[None],
             model.vlist_1[0],
             model.vlist_1[1],
             model.b_1.v_2,
             model.b_1.b_2.v_3,
             model.b_1.b_2.vlist_3[0],
             model.blist_1[0].v_2]
        cls._components[model][Block] = \
            [model.b_1,
             model.blist_1[0],
             model.b_1.b_2,
             model.b_1.b_2.b_3,
             model.blist_1[0].b_2]
        cls._components[model.b_1] = {}
        cls._components[model.b_1][Var] = \
            [model.b_1.v_2,
             model.b_1.b_2.v_3,
             model.b_1.b_2.vlist_3[0]]
        cls._components[model.b_1][Block] = \
            [model.b_1.b_2,
             model.b_1.b_2.b_3]
        cls._components[model.b_1.b_2] = {}
        cls._components[model.b_1.b_2][Var] = \
            [model.b_1.b_2.v_3,
             model.b_1.b_2.vlist_3[0]]
        cls._components[model.b_1.b_2][Block] = \
            [model.b_1.b_2.b_3]
        cls._components[model.b_1.b_2.b_3] = {}
        cls._components[model.b_1.b_2.b_3][Var] = []
        cls._components[model.b_1.b_2.b_3][Block] = []
        cls._components[model.blist_1[0]] = {}
        cls._components[model.blist_1[0]][Var] = \
            [model.blist_1[0].v_2]
        cls._components[model.blist_1[0]][Block] = \
            [model.blist_1[0].b_2]
        cls._components[model.blist_1[0].b_2] = {}
        cls._components[model.blist_1[0].b_2][Var] = []
        cls._components[model.blist_1[0].b_2][Block] = []

        cls._blocks_no_descend = ComponentMap()
        for obj in cls._components_no_descend:
            cls._blocks_no_descend[obj] = \
                [obj] + cls._components_no_descend[obj][Block]

        cls._blocks = ComponentMap()
        for obj in cls._components:
            cls._blocks[obj] = \
                [obj] + cls._components[obj][Block]

    def test_init(self):
        b = block()
        self.assertTrue(b.parent is None)
        self.assertEqual(b.ctype, Block)

    def test_type(self):
        b = block()
        self.assertTrue(isinstance(b, ICategorizedObject))
        self.assertTrue(isinstance(b, IActiveObject))
        self.assertTrue(isinstance(b, IComponent))
        self.assertTrue(isinstance(b, IComponentContainer))
        self.assertTrue(isinstance(b, _IActiveComponentContainer))
        self.assertTrue(isinstance(b, IBlockStorage))

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

    def test_collect_ctypes(self):
        b = block()
        self.assertEqual(b.collect_ctypes(),
                         set())
        self.assertEqual(b.collect_ctypes(active=True),
                         set())
        b.x = variable()
        self.assertEqual(b.collect_ctypes(),
                         set([Var]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([Var]))
        b.y = constraint()
        self.assertEqual(b.collect_ctypes(),
                         set([Var, Constraint]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([Var, Constraint]))
        b.y.deactivate()
        self.assertEqual(b.collect_ctypes(),
                         set([Var, Constraint]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([Var]))
        B = block()
        B.b = b
        self.assertEqual(B.collect_ctypes(descend_into=False),
                         set([Block]))
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=True),
                         set([Block]))
        self.assertEqual(B.collect_ctypes(),
                         set([Block, Var, Constraint]))
        self.assertEqual(B.collect_ctypes(active=True),
                         set([Block, Var]))
        b.deactivate()
        self.assertEqual(B.collect_ctypes(descend_into=False),
                         set([Block]))
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=True),
                         set([]))
        self.assertEqual(B.collect_ctypes(),
                         set([Block, Var, Constraint]))
        self.assertEqual(B.collect_ctypes(active=True),
                         set([]))
        B.x = variable()
        self.assertEqual(B.collect_ctypes(descend_into=False),
                         set([Block, Var]))
        self.assertEqual(B.collect_ctypes(descend_into=False,
                                          active=True),
                         set([Var]))
        self.assertEqual(B.collect_ctypes(),
                         set([Block, Var, Constraint]))
        self.assertEqual(B.collect_ctypes(active=True),
                         set([Var]))
        del b.y

        # a block DOES check its own .active flag apparently
        self.assertEqual(b.collect_ctypes(),
                         set([Var]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([]))
        b.activate()
        self.assertEqual(b.collect_ctypes(),
                         set([Var]))
        self.assertEqual(b.collect_ctypes(active=True),
                         set([Var]))

        del b.x
        self.assertEqual(b.collect_ctypes(), set())

class Test_block_noclone(_Test_block, unittest.TestCase):
    _do_clone = False

class Test_block_clone(_Test_block, unittest.TestCase):
    _do_clone = True

class _MyBlockBaseBase(StaticBlock):
    __slots__ = ()
    def __init__(self):
        super(_MyBlockBaseBase, self).__init__()

class _MyBlockBase(_MyBlockBaseBase):
    __slots__ = ("b")
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

class _Test_StaticBlock(_Test_block_base):

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

        cls._postorder = [model.b.v,
                          model.b.blist[0],
                          model.b.blist,
                          model.b,
                          model.v,
                          model]
        assert len(cls._preorder) == len(cls._postorder)

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
        cls._components_no_descend[model][Var] = [model.v]
        cls._components_no_descend[model][Block] = [model.b]
        cls._components_no_descend[model.b] = {}
        cls._components_no_descend[model.b][Var] = [model.b.v]
        cls._components_no_descend[model.b][Block] = [model.b.blist[0]]
        cls._components_no_descend[model.b.blist[0]] = {}
        cls._components_no_descend[model.b.blist[0]][Block] = []
        cls._components_no_descend[model.b.blist[0]][Var] = []

        cls._components = ComponentMap()
        cls._components[model] = {}
        cls._components[model][Var] = [model.v, model.b.v]
        cls._components[model][Block] = [model.b, model.b.blist[0]]
        cls._components[model.b] = {}
        cls._components[model.b][Var] = [model.b.v]
        cls._components[model.b][Block] = [model.b.blist[0]]
        cls._components[model.b.blist[0]] = {}
        cls._components[model.b.blist[0]][Block] = []
        cls._components[model.b.blist[0]][Var] = []

        cls._blocks_no_descend = ComponentMap()
        for obj in cls._components_no_descend:
            cls._blocks_no_descend[obj] = \
                [obj] + cls._components_no_descend[obj][Block]

        cls._blocks = ComponentMap()
        for obj in cls._components:
            cls._blocks[obj] = \
                [obj] + cls._components[obj][Block]

    # override this test method on the base class
    def test_collect_ctypes(self):
        self.assertEqual(self._block.collect_ctypes(),
                         set([Block, Var]))
        self.assertEqual(self._block.collect_ctypes(active=True),
                         set([Block, Var]))
        self.assertEqual(self._block.collect_ctypes(descend_into=False),
                         set([Block, Var]))
        self.assertEqual(self._block.collect_ctypes(active=True,
                                                    descend_into=False),
                         set([Block, Var]))

        self._block.b.deactivate()
        self.assertEqual(self._block.collect_ctypes(),
                         set([Block, Var]))
        self.assertEqual(self._block.collect_ctypes(active=True),
                         set([Var]))
        self.assertEqual(self._block.collect_ctypes(descend_into=False),
                         set([Block, Var]))
        self.assertEqual(self._block.collect_ctypes(active=True,
                                                    descend_into=False),
                         set([Var]))
        self._block.b.activate()

    def test_staticblock_delattr(self):
        b = _MyBlock()
        with self.assertRaises(AttributeError):
            del b.not_an_attribute
        c = b.b
        self.assertIs(c.parent, b)
        del b.b
        self.assertIs(c.parent, None)
        b.b = c
        self.assertIs(c.parent, b)

    def test_staticblock_setattr(self):
        b = _MyBlockBase()
        self.assertIs(b.b.parent, b)
        self.assertIs(b.b.v.parent, b.b)
        with self.assertRaises(ValueError):
            b.b = b.b.v
        self.assertIs(b.b.parent, b)
        self.assertIs(b.b.v.parent, b.b)
# added a __dict__ in recent commit
#        with self.assertRaises(AttributeError):
#            b.not_an_attribute = 2
        c = b.b
        self.assertIs(c.parent, b)
        # test the edge case in setattr
        b.b = c
        self.assertIs(c.parent, b)
        # test an overwrite
        b.b = block()
        self.assertIs(c.parent, None)
        self.assertIs(b.b.parent, b)

    def test_staticblock__with_dict_setattr(self):
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
        self.assertNotEqual(len(list(b.preorder_traversal())), 0)
        self.assertEqual(len(list(b.preorder_traversal(active=True))), 0)
        self.assertNotEqual(len(list(b.postorder_traversal())), 0)
        self.assertEqual(len(list(b.postorder_traversal(active=True))), 0)
        self.assertNotEqual(len(list(b.components())), 0)
        self.assertEqual(len(list(b.components(active=True))), 0)
        self.assertNotEqual(len(list(b.blocks())), 0)
        self.assertEqual(len(list(b.blocks(active=True))), 0)
        self.assertNotEqual(len(list(b.generate_names())), 0)
        self.assertEqual(len(list(b.generate_names(active=True))), 0)

class Test_StaticBlock_noclone(_Test_StaticBlock, unittest.TestCase):
    _do_clone = False

class Test_StaticBlock_clone(_Test_StaticBlock, unittest.TestCase):
    _do_clone = True

class Test_block_dict(_TestActiveComponentDictBase,
                      unittest.TestCase):
    _container_type = block_dict
    _ctype_factory = lambda self: block()

class Test_block_list(_TestActiveComponentListBase,
                      unittest.TestCase):
    _container_type = block_list
    _ctype_factory = lambda self: block()

if __name__ == "__main__":
    unittest.main()
