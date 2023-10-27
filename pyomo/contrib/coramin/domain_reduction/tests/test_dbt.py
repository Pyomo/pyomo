from coramin.domain_reduction.dbt import TreeBlock, TreeBlockError, convert_pyomo_model_to_bipartite_graph, \
    _VarNode, _ConNode, _RelNode, split_metis, num_cons_in_graph, collect_vars_to_tighten_by_block, decompose_model, \
    perform_dbt, OBBTMethod, FilterMethod
import unittest
import pyomo.environ as pe
import coramin
from networkx import is_bipartite
from pyomo.common.collections import ComponentSet
from networkx import Graph
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr import differentiate
from egret.thirdparty.get_pglib_opf import get_pglib_opf
from egret.data.model_data import ModelData
from egret.models.acopf import create_psv_acopf_model
import os
from coramin.utils.pyomo_utils import get_objective
import filecmp
from pyomo.contrib import appsi
import pytest


class TestTreeBlock(unittest.TestCase):
    def test_tree_block(self):
        b = TreeBlock(concrete=True)
        with self.assertRaises(TreeBlockError):
            b.is_leaf()
        with self.assertRaises(TreeBlockError):
            b.x = pe.Var()
        with self.assertRaises(TreeBlockError):
            children = b.children
        with self.assertRaises(TreeBlockError):
            linking_constraints = b.linking_constraints
        with self.assertRaises(TreeBlockError):
            num_stages = b.num_stages()
        with self.assertRaises(TreeBlockError):
            stage_blocks = list(b.stage_blocks(0))
        with self.assertRaises(TreeBlockError):
            stage = b.get_block_stage(b)
        b.setup(children_keys=list())
        self.assertTrue(b.is_leaf())
        b.x = pe.Var()  # make sure we can add components just like a regular block
        b.x.setlb(-1)
        with self.assertRaises(TreeBlockError):
            linking_constraints = b.linking_constraints
        with self.assertRaises(TreeBlockError):
            children = b.children
        self.assertEqual(b.num_stages(), 1)
        stage0_blocks = list(b.stage_blocks(0))
        self.assertEqual(len(stage0_blocks), 1)
        self.assertIs(stage0_blocks[0], b)
        stage1_blocks = list(b.stage_blocks(1))
        self.assertEqual(len(stage1_blocks), 0)
        self.assertEqual(b.get_block_stage(b), 0)

        b = TreeBlock(concrete=True)
        b.setup(children_keys=[1, 2])
        b.children[1].setup(children_keys=list())
        b.children[2].setup(children_keys=['a', 'b'])
        b.children[2].children['a'].setup(children_keys=list())
        b.children[2].children['b'].setup(children_keys=list())
        self.assertFalse(b.is_leaf())
        self.assertTrue(b.children[1].is_leaf())
        self.assertFalse(b.children[2].is_leaf())
        self.assertTrue(b.children[2].children['a'].is_leaf())
        self.assertTrue(b.children[2].children['b'].is_leaf())

        with self.assertRaises(TreeBlockError):
            b.x = pe.Var()
        b.children[1].x = pe.Var()
        with self.assertRaises(TreeBlockError):
            b.children[2].x = pe.Var()
        b.children[2].children['a'].x = pe.Var()
        b.children[2].children['b'].x = pe.Var()
        self.assertEqual(len(list(b.component_data_objects(pe.Var, descend_into=True, sort=True))), 3)

        self.assertEqual(b.num_stages(), 3)
        with self.assertRaises(TreeBlockError):
            b.children[1].num_stages()
        with self.assertRaises(TreeBlockError):
            b.children[2].num_stages()
        with self.assertRaises(TreeBlockError):
            b.children[2].children['a'].num_stages()
        with self.assertRaises(TreeBlockError):
            b.children[2].children['b'].num_stages()

        stage0_blocks = list(b.stage_blocks(0))
        stage1_blocks = list(b.stage_blocks(1))
        stage2_blocks = list(b.stage_blocks(2))
        stage3_blocks = list(b.stage_blocks(3))
        self.assertEqual(len(stage0_blocks), 1)
        self.assertEqual(len(stage1_blocks), 2)
        self.assertEqual(len(stage2_blocks), 2)
        self.assertEqual(len(stage3_blocks), 0)
        self.assertIs(stage0_blocks[0], b)
        self.assertIs(stage1_blocks[0], b.children[1])
        self.assertIs(stage1_blocks[1], b.children[2])
        self.assertIs(stage2_blocks[0], b.children[2].children['a'])
        self.assertIs(stage2_blocks[1], b.children[2].children['b'])
        with self.assertRaises(TreeBlockError):
            list(b.children[2].stage_blocks(0))
        b.children[1].deactivate()
        stage1_blocks = list(b.stage_blocks(1, active=True))
        self.assertEqual(len(stage1_blocks), 1)
        self.assertIs(stage1_blocks[0], b.children[2])
        stage1_blocks = list(b.stage_blocks(1))
        self.assertEqual(len(stage1_blocks), 2)
        self.assertIs(stage1_blocks[0], b.children[1])
        self.assertIs(stage1_blocks[1], b.children[2])

        self.assertEqual(b.get_block_stage(b), 0)
        self.assertEqual(b.get_block_stage(b.children[1]), 1)
        self.assertEqual(b.get_block_stage(b.children[2]), 1)
        self.assertEqual(b.get_block_stage(b.children[2].children['a']), 2)
        self.assertEqual(b.get_block_stage(b.children[2].children['b']), 2)
        b.children[1].foo = pe.Block()
        self.assertIs(b.get_block_stage(b.children[1].foo), None)
        with self.assertRaises(TreeBlockError):
            b.children[2].get_block_stage(b.children[2].children['a'])

        self.assertEqual(len(list(b.linking_constraints.values())), 0)


class TestGraphConversion(unittest.TestCase):
    def test_convert_pyomo_model_to_bipartite_graph(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.c1 = pe.Constraint(expr=m.z >= m.x + m.y)
        m.c2 = coramin.relaxations.PWXSquaredRelaxation()
        m.c2.build(x=m.x, aux_var=m.z)
        m.c3 = pe.Constraint(expr=m.z >= m.x - m.y)

        graph = convert_pyomo_model_to_bipartite_graph(m)
        self.assertTrue(is_bipartite(graph))
        self.assertEqual(graph.number_of_nodes(), 6)
        self.assertEqual(graph.number_of_edges(), 8)
        graph_node_comps = ComponentSet([i.comp for i in graph.nodes()])
        self.assertEqual(len(graph_node_comps), 6)
        self.assertIn(m.x, graph_node_comps)
        self.assertIn(m.y, graph_node_comps)
        self.assertIn(m.z, graph_node_comps)
        self.assertIn(m.c1, graph_node_comps)
        self.assertIn(m.c2, graph_node_comps)
        self.assertIn(m.c3, graph_node_comps)
        graph_edge_comps = {(id(i.comp), id(j.comp)) for i, j in graph.edges()}
        self.assertTrue(((id(m.x), id(m.c1)) in graph_edge_comps) or ((id(m.c1), id(m.x)) in graph_edge_comps))
        self.assertTrue(((id(m.y), id(m.c1)) in graph_edge_comps) or ((id(m.c1), id(m.y)) in graph_edge_comps))
        self.assertTrue(((id(m.z), id(m.c1)) in graph_edge_comps) or ((id(m.c1), id(m.z)) in graph_edge_comps))
        self.assertTrue(((id(m.x), id(m.c2)) in graph_edge_comps) or ((id(m.c2), id(m.x)) in graph_edge_comps))
        self.assertFalse(((id(m.y), id(m.c2)) in graph_edge_comps) or ((id(m.c2), id(m.y)) in graph_edge_comps))
        self.assertTrue(((id(m.z), id(m.c2)) in graph_edge_comps) or ((id(m.c2), id(m.z)) in graph_edge_comps))
        self.assertTrue(((id(m.x), id(m.c3)) in graph_edge_comps) or ((id(m.c3), id(m.x)) in graph_edge_comps))
        self.assertTrue(((id(m.y), id(m.c3)) in graph_edge_comps) or ((id(m.c3), id(m.y)) in graph_edge_comps))
        self.assertTrue(((id(m.z), id(m.c3)) in graph_edge_comps) or ((id(m.c3), id(m.z)) in graph_edge_comps))
        self.assertEqual(num_cons_in_graph(graph=graph, include_rels=True), 3)
        self.assertEqual(num_cons_in_graph(graph=graph, include_rels=False), 2)


class TestSplit(unittest.TestCase):
    def setUp(self):
        m = pe.ConcreteModel()
        self.m = m
        m.v1 = pe.Var()
        m.v2 = pe.Var(bounds=(-1, 1))
        m.v3 = pe.Var(bounds=(-1, 1))
        m.v6 = pe.Var()
        m.v4 = pe.Var(bounds=(-1, 1))
        m.v5 = pe.Var(bounds=(-1, 1))

        m.c1 = pe.Constraint(expr=m.v1 - m.v2 - m.v3 == 0)
        m.c2 = pe.Constraint(expr=m.v6 - m.v4 - m.v5 == 0)
        m.r1 = coramin.relaxations.PWMcCormickRelaxation()
        m.r1.set_input(x1=m.v4, x2=m.v5, aux_var=m.v6)
        m.r2 = coramin.relaxations.PWMcCormickRelaxation()
        m.r2.set_input(x1=m.v3, x2=m.v4, aux_var=m.v2)

    def test_split_metis(self):
        m = self.m

        g = Graph()
        v1 = _VarNode(m.v1)
        v2 = _VarNode(m.v2)
        v3 = _VarNode(m.v3)
        v4 = _VarNode(m.v4)
        v5 = _VarNode(m.v5)
        v6 = _VarNode(m.v6)
        c1 = _ConNode(m.c1)
        c2 = _ConNode(m.c2)
        r1 = _RelNode(m.r1)
        r2 = _RelNode(m.r2)

        g.add_edge(v2, r2)
        g.add_edge(v3, r2)
        g.add_edge(v4, r2)
        g.add_edge(v1, c1)
        g.add_edge(v2, c1)
        g.add_edge(v3, c1)
        g.add_edge(v4, r1)
        g.add_edge(v5, r1)
        g.add_edge(v6, r1)
        g.add_edge(v4, c2)
        g.add_edge(v5, c2)
        g.add_edge(v6, c2)

        tree, partitioning_ratio = split_metis(graph=g, model=m)
        self.assertAlmostEqual(partitioning_ratio, 3*12/(14*1+6*2+6*2))

        children = list(tree.children)
        self.assertEqual(len(children), 2)
        graph_a = children[0]
        graph_b = children[1]
        if v1 in graph_b.nodes():
            graph_a, graph_b = graph_b, graph_a

        graph_a_nodes = set(graph_a.nodes())
        graph_b_nodes = set(graph_b.nodes())
        self.assertIn(v1, graph_a_nodes)
        self.assertIn(v2, graph_a_nodes)
        self.assertIn(v3, graph_a_nodes)
        self.assertIn(v4, graph_b_nodes)
        self.assertIn(v5, graph_b_nodes)
        self.assertIn(v6, graph_b_nodes)
        self.assertIn(r2, graph_a_nodes)
        self.assertIn(c1, graph_a_nodes)
        self.assertIn(r1, graph_b_nodes)
        self.assertIn(c2, graph_b_nodes)
        self.assertEqual(len(graph_a_nodes), 6)
        self.assertEqual(len(graph_b_nodes), 5)
        v4_hat = list(graph_a_nodes - {v1, v2, v3, c1, r2})[0]

        graph_a_edges = set(graph_a.edges())
        graph_b_edges = set(graph_b.edges())
        self.assertTrue((v2, r2) in graph_a_edges or (r2, v2) in graph_a_edges)
        self.assertTrue((v3, r2) in graph_a_edges or (r2, v3) in graph_a_edges)
        self.assertTrue((v4_hat, r2) in graph_a_edges or (r2, v4_hat) in graph_a_edges)
        self.assertTrue((v1, c1) in graph_a_edges or (c1, v1) in graph_a_edges)
        self.assertTrue((v2, c1) in graph_a_edges or (c1, v2) in graph_a_edges)
        self.assertTrue((v3, c1) in graph_a_edges or (c1, v3) in graph_a_edges)
        self.assertTrue((v4, r1) in graph_b_edges or (r1, v4) in graph_b_edges)
        self.assertTrue((v5, r1) in graph_b_edges or (r1, v5) in graph_b_edges)
        self.assertTrue((v6, r1) in graph_b_edges or (r1, v6) in graph_b_edges)
        self.assertTrue((v4, c2) in graph_b_edges or (c2, v4) in graph_b_edges)
        self.assertTrue((v5, c2) in graph_b_edges or (c2, v5) in graph_b_edges)
        self.assertTrue((v6, c2) in graph_b_edges or (c2, v6) in graph_b_edges)
        self.assertEqual(len(graph_a_edges), 6)
        self.assertEqual(len(graph_b_edges), 6)

        edges_between_children = list(tree.edges_between_children)
        self.assertEqual(len(edges_between_children), 1)
        edge = edges_between_children[0]
        self.assertTrue((v4 is edge.node1 and v4_hat is edge.node2) or (v4 is edge.node2 and v4_hat is edge.node1))

        new_model = TreeBlock(concrete=True)
        component_map = tree.build_pyomo_model(block=new_model)
        new_vars = list(coramin.relaxations.nonrelaxation_component_data_objects(new_model,
                                                                                 ctype=pe.Var,
                                                                                 descend_into=True,
                                                                                 sort=True))
        new_cons = list(coramin.relaxations.nonrelaxation_component_data_objects(new_model,
                                                                                 ctype=pe.Constraint,
                                                                                 active=True,
                                                                                 descend_into=True,
                                                                                 sort=True))
        new_rels = list(coramin.relaxations.relaxation_data_objects(new_model,
                                                                    descend_into=True,
                                                                    active=True,
                                                                    sort=True))
        self.assertEqual(len(new_vars), 7)
        self.assertEqual(len(new_cons), 3)
        self.assertEqual(len(new_rels), 2)
        self.assertEqual(len(new_model.children), 2)
        self.assertEqual(len(new_model.linking_constraints), 1)
        self.assertEqual(new_model.num_stages(), 2)

        stage0_vars = list(new_model.component_data_objects(pe.Var, descend_into=False, sort=True))
        stage0_cons = list(new_model.component_data_objects(pe.Constraint, descend_into=False, sort=True, active=True))
        stage0_rels = list(coramin.relaxations.relaxation_data_objects(new_model,
                                                                       descend_into=False,
                                                                       active=True,
                                                                       sort=True))
        self.assertEqual(len(stage0_vars), 0)
        self.assertEqual(len(stage0_cons), 1)
        self.assertEqual(len(stage0_rels), 0)

        block_a = new_model.children[0]
        block_b = new_model.children[1]
        block_a_vars = ComponentSet(coramin.relaxations.nonrelaxation_component_data_objects(block_a,
                                                                                             ctype=pe.Var,
                                                                                             descend_into=True,
                                                                                             sort=True))
        block_b_vars = ComponentSet(coramin.relaxations.nonrelaxation_component_data_objects(block_b,
                                                                                             ctype=pe.Var,
                                                                                             descend_into=True,
                                                                                             sort=True))
        block_a_cons = ComponentSet(coramin.relaxations.nonrelaxation_component_data_objects(block_a,
                                                                                             ctype=pe.Constraint,
                                                                                             descend_into=True,
                                                                                             active=True,
                                                                                             sort=True))
        block_b_cons = ComponentSet(coramin.relaxations.nonrelaxation_component_data_objects(block_b,
                                                                                             ctype=pe.Constraint,
                                                                                             descend_into=True,
                                                                                             active=True,
                                                                                             sort=True))
        block_a_rels = ComponentSet(coramin.relaxations.relaxation_data_objects(block_a,
                                                                                descend_into=True,
                                                                                active=True,
                                                                                sort=True))
        block_b_rels = ComponentSet(coramin.relaxations.relaxation_data_objects(block_b,
                                                                                descend_into=True,
                                                                                active=True,
                                                                                sort=True))
        if component_map[m.v1] not in block_a_vars:
            block_a, block_b = block_b, block_a
            block_a_vars, block_b_vars = block_b_vars, block_a_vars
            block_a_cons, block_b_cons = block_b_cons, block_a_cons
            block_a_rels, block_b_rels = block_b_rels, block_a_rels

        self.assertEqual(len(block_a_vars), 4)
        self.assertEqual(len(block_a_cons), 1)
        self.assertEqual(len(block_a_rels), 1)
        self.assertEqual(len(block_b_vars), 3)
        self.assertEqual(len(block_b_cons), 1)
        self.assertEqual(len(block_b_rels), 1)

        v1 = component_map[m.v1]
        v2 = component_map[m.v2]
        v3 = component_map[m.v3]
        v4_a = block_a.vars['v4']
        v4_b = block_b.vars['v4']
        v5 = component_map[m.v5]
        v6 = component_map[m.v6]

        self.assertIs(v1, block_a.vars['v1'])
        self.assertIs(v2, block_a.vars['v2'])
        self.assertIs(v3, block_a.vars['v3'])
        self.assertIs(v5, block_b.vars['v5'])
        self.assertIs(v6, block_b.vars['v6'])

        self.assertEqual(v2.lb, -1)
        self.assertEqual(v2.ub, 1)
        self.assertEqual(v3.lb, -1)
        self.assertEqual(v3.ub, 1)
        self.assertEqual(v4_a.lb, -1)
        self.assertEqual(v4_a.ub, 1)
        self.assertEqual(v4_b.lb, -1)
        self.assertEqual(v4_b.ub, 1)
        self.assertEqual(v5.lb, -1)
        self.assertEqual(v5.ub, 1)
        self.assertEqual(v1.lb, None)
        self.assertEqual(v1.ub, None)
        self.assertEqual(v6.lb, None)
        self.assertEqual(v6.ub, None)

        linking_con = new_model.linking_constraints[1]
        linking_con_vars = ComponentSet(identify_variables(linking_con.body))
        self.assertEqual(len(linking_con_vars), 2)
        self.assertIn(v4_a, linking_con_vars)
        self.assertIn(v4_b, linking_con_vars)
        derivs = differentiate(expr=linking_con.body, mode=differentiate.Modes.reverse_symbolic)
        self.assertTrue((derivs[v4_a] == 1 and derivs[v4_b] == -1) or
                        (derivs[v4_a] == -1 and derivs[v4_b] == 1))
        self.assertEqual(linking_con.lower, 0)
        self.assertEqual(linking_con.upper, 0)

        c1 = block_a.cons['c1']
        c2 = block_b.cons['c2']
        r1 = block_b.rels.r1
        r2 = block_a.rels.r2
        c1_vars = ComponentSet(identify_variables(c1.body))
        c2_vars = ComponentSet(identify_variables(c2.body))
        self.assertEqual(len(c1_vars), 3)
        self.assertEqual(len(c2_vars), 3)
        self.assertIn(v1, c1_vars)
        self.assertIn(v2, c1_vars)
        self.assertIn(v3, c1_vars)
        self.assertIn(v4_b, c2_vars)
        self.assertIn(v5, c2_vars)
        self.assertIn(v6, c2_vars)
        self.assertIs(r1.get_aux_var(), v6)
        self.assertIs(r2.get_aux_var(), v2)
        r1_rhs_vars = ComponentSet(r1.get_rhs_vars())
        r2_rhs_vars = ComponentSet(r2.get_rhs_vars())
        self.assertIn(v3, r2_rhs_vars)
        self.assertIn(v4_a, r2_rhs_vars)
        self.assertIn(v4_b, r1_rhs_vars)
        self.assertIn(v5, r1_rhs_vars)
        self.assertTrue(isinstance(r1, coramin.relaxations.PWMcCormickRelaxationData))
        self.assertTrue(isinstance(r2, coramin.relaxations.PWMcCormickRelaxationData))
        c1_derivs = differentiate(c1.body, mode=differentiate.Modes.reverse_symbolic)
        c2_derivs = differentiate(c2.body, mode=differentiate.Modes.reverse_symbolic)
        self.assertEqual(c1_derivs[v1], 1)
        self.assertEqual(c1_derivs[v2], -1)
        self.assertEqual(c1_derivs[v3], -1)
        self.assertEqual(c2_derivs[v4_b], -1)
        self.assertEqual(c2_derivs[v5], -1)
        self.assertEqual(c2_derivs[v6], 1)
        self.assertEqual(c1.lower, 0)
        self.assertEqual(c1.upper, 0)
        self.assertEqual(c2.lower, 0)
        self.assertEqual(c2.upper, 0)


class TestNumCons(unittest.TestCase):
    def test_num_cons(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var()
        m.z = pe.Var()
        m.r = coramin.relaxations.PWUnivariateRelaxation()
        m.r.build(x=m.x,
                  aux_var=m.y,
                  shape=coramin.utils.FunctionShape.CONVEX,
                  f_x_expr=pe.exp(m.x))
        m.c = pe.Constraint(expr=m.z == 2*m.x)
        g = convert_pyomo_model_to_bipartite_graph(m)
        self.assertEqual(num_cons_in_graph(g, include_rels=False), 1)
        self.assertEqual(num_cons_in_graph(g), 2)


class TestDecompose(unittest.TestCase):
    def helper(self, case, min_partition_ratio, expected_termination):
        """
        we rely on other tests to make sure the relaxation is constructed
        correctly. This test just checks the decomposition.
        """

        test_dir = os.path.dirname(os.path.abspath(__file__))
        pglib_dir = os.path.join(test_dir, 'pglib-opf-master')
        if not os.path.isdir(pglib_dir):
            get_pglib_opf(download_dir=test_dir)
        md = ModelData.read(filename=os.path.join(pglib_dir, case))
        m, scaled_md = create_psv_acopf_model(md)
        opt = pe.SolverFactory('ipopt')
        res = opt.solve(m, tee=False)

        relaxed_m = coramin.relaxations.relax(m,
                                              in_place=False,
                                              use_fbbt=False,
                                              fbbt_options={'deactivate_satisfied_constraints': True,
                                                            'max_iter': 2},
                                              use_alpha_bb=False)
        (decomposed_m,
         component_map,
         termination_reason) = decompose_model(model=relaxed_m,
                                               max_leaf_nnz=1000,
                                               min_partition_ratio=1.4,
                                               limit_num_stages=True)
        self.assertEqual(termination_reason, expected_termination)
        if expected_termination == coramin.domain_reduction.dbt.DecompositionStatus.normal:
            self.assertGreaterEqual(decomposed_m.num_stages(), 2)

        for r in coramin.relaxations.relaxation_data_objects(block=relaxed_m, descend_into=True,
                                                             active=True, sort=True):
            r.rebuild(build_nonlinear_constraint=True)
        for r in coramin.relaxations.relaxation_data_objects(block=decomposed_m, descend_into=True,
                                                             active=True, sort=True):
            r.rebuild(build_nonlinear_constraint=True)
        relaxed_res = opt.solve(relaxed_m, tee=False)
        decomposed_res = opt.solve(decomposed_m, tee=False)

        self.assertEqual(res.solver.termination_condition, pe.TerminationCondition.optimal)
        self.assertEqual(relaxed_res.solver.termination_condition, pe.TerminationCondition.optimal)
        self.assertEqual(decomposed_res.solver.termination_condition, pe.TerminationCondition.optimal)
        obj = get_objective(m)
        relaxed_obj = get_objective(relaxed_m)
        decomposed_obj = get_objective(decomposed_m)
        val = pe.value(obj.expr)
        relaxed_val = pe.value(relaxed_obj.expr)
        decomposed_val = pe.value(decomposed_obj.expr)
        relaxed_rel_diff = abs(val - relaxed_val) / val
        decomposed_rel_diff = abs(val - decomposed_val) / val
        self.assertAlmostEqual(relaxed_rel_diff, 0, 5)
        self.assertAlmostEqual(decomposed_rel_diff, 0, 5)

        relaxed_vars = list(coramin.relaxations.nonrelaxation_component_data_objects(relaxed_m,
                                                                                     pe.Var,
                                                                                     sort=True,
                                                                                     descend_into=True))
        relaxed_vars = [v for v in relaxed_vars if not v.fixed]
        relaxed_cons = list(coramin.relaxations.nonrelaxation_component_data_objects(relaxed_m,
                                                                                     pe.Constraint,
                                                                                     active=True,
                                                                                     sort=True,
                                                                                     descend_into=True))
        relaxed_rels = list(coramin.relaxations.relaxation_data_objects(relaxed_m,
                                                                        descend_into=True,
                                                                        active=True,
                                                                        sort=True))
        decomposed_vars = list(coramin.relaxations.nonrelaxation_component_data_objects(decomposed_m,
                                                                                        pe.Var,
                                                                                        sort=True,
                                                                                        descend_into=True))
        decomposed_cons = list(coramin.relaxations.nonrelaxation_component_data_objects(decomposed_m,
                                                                                        pe.Constraint,
                                                                                        active=True,
                                                                                        sort=True,
                                                                                        descend_into=True))
        decomposed_rels = list(coramin.relaxations.relaxation_data_objects(decomposed_m,
                                                                           descend_into=True,
                                                                           active=True,
                                                                           sort=True))
        linking_cons = list()
        for stage in range(decomposed_m.num_stages()):
            for block in decomposed_m.stage_blocks(stage):
                if not block.is_leaf():
                    linking_cons.extend(block.linking_constraints.values())
        relaxed_vars_mapped = list()
        for i in relaxed_vars:
            relaxed_vars_mapped.append(component_map[i])
        relaxed_vars_mapped = ComponentSet(relaxed_vars_mapped)
        var_diff = ComponentSet(decomposed_vars) - relaxed_vars_mapped
        extra_vars = ComponentSet()
        for c in linking_cons:
            for v in identify_variables(c.body, include_fixed=True):
                extra_vars.add(v)
        for v in coramin.relaxations.nonrelaxation_component_data_objects(decomposed_m, pe.Var, descend_into=True):
            if 'dbt_partition_vars' in str(v) or 'obj_var' in str(v):
                extra_vars.add(v)
        extra_vars = extra_vars - relaxed_vars_mapped
        partition_cons = ComponentSet()
        obj_cons = ComponentSet()
        for c in coramin.relaxations.nonrelaxation_component_data_objects(decomposed_m, pe.Constraint, active=True, descend_into=True):
            if 'dbt_partition_cons' in str(c):
                partition_cons.add(c)
            elif 'obj_con' in str(c):
                obj_cons.add(c)
        for v in var_diff:
            self.assertIn(v, extra_vars)
        var_diff = relaxed_vars_mapped - ComponentSet(decomposed_vars)
        self.assertEqual(len(var_diff), 0)
        self.assertEqual(len(relaxed_vars) + len(extra_vars), len(decomposed_vars))

        rcs = list()
        for i in relaxed_cons + linking_cons + list(partition_cons) + list(obj_cons):
            rcs.append(str(i))
        dcs = [str(i) for i in decomposed_cons]

        def _reformat(s: str) -> str:
            s = s.split('.cons')
            if len(s) > 1:
                s = s[1]
                s = s.lstrip('[')
                s = s.rstrip(']')
            elif s[0].startswith('cons'):
                s = s[0]
                s = s.lstrip('cons')
                s = s.lstrip('[')
                s = s.rstrip(']')
            else:
                s = s[0]
            s = s.replace('"', '')
            s = s.replace("'", "")
            return s

        rcs = set([_reformat(i) for i in rcs])
        dcs = set([_reformat(i) for i in dcs])

        self.assertEqual(rcs, dcs)

        # self.assertEqual(len(relaxed_cons) + len(linking_cons) + len(partition_cons) - len(partition_cons)/3 + len(obj_cons), len(decomposed_cons))
        self.assertEqual(len(relaxed_rels), len(decomposed_rels))

    def test_decompose1(self):
        self.helper('pglib_opf_case5_pjm.m', min_partition_ratio=1.5, expected_termination=coramin.domain_reduction.dbt.DecompositionStatus.problem_too_small)

    def test_decompose2(self):
        self.helper('pglib_opf_case30_ieee.m', min_partition_ratio=1.5, expected_termination=coramin.domain_reduction.dbt.DecompositionStatus.normal)

    def test_decompose3(self):
        self.helper('pglib_opf_case118_ieee.m', min_partition_ratio=1.5, expected_termination=coramin.domain_reduction.dbt.DecompositionStatus.normal)

    def test_decompose4(self):
        self.helper('pglib_opf_case14_ieee.m', min_partition_ratio=1.4, expected_termination=coramin.domain_reduction.dbt.DecompositionStatus.normal)


class TestVarsToTightenByBlock(unittest.TestCase):
    def test_vars_to_tighten_by_block(self):
        m = TreeBlock(concrete=True)
        m.setup(children_keys=[1, 2])
        b1 = m.children[1]
        b2 = m.children[2]
        b1.setup(children_keys=list())
        b2.setup(children_keys=list())

        b1.x = pe.Var(bounds=(-1, 1))
        b1.y = pe.Var()
        b1.z = pe.Var()
        b1.aux = pe.Var()

        b2.x = pe.Var(bounds=(-1, 1))
        b2.y = pe.Var()
        b2.z = pe.Var()
        b2.aux = pe.Var()

        b1.c = pe.Constraint(expr=b1.x + b1.y + b1.z == 0)
        b2.c = pe.Constraint(expr=b2.x + b2.y + b2.z == 0)

        b1.r = coramin.relaxations.PWUnivariateRelaxation()
        b1.r.set_input(x=b1.x,
                       aux_var=b1.aux,
                       shape=coramin.utils.FunctionShape.CONVEX,
                       f_x_expr=pe.exp(b1.x))
        b1.r.rebuild()

        b2.r = coramin.relaxations.PWXSquaredRelaxation()
        b2.r.set_input(x=b2.x,
                       aux_var=b2.aux,
                       relaxation_side=coramin.utils.RelaxationSide.UNDER)
        b2.r.rebuild()

        m.linking_constraints.add(b1.z == b2.z)

        vars_to_tighten_by_block = collect_vars_to_tighten_by_block(m, method='full_space')
        self.assertEqual(len(vars_to_tighten_by_block), 3)
        vars_to_tighten = vars_to_tighten_by_block[m]
        self.assertEqual(len(vars_to_tighten), 0)
        vars_to_tighten = vars_to_tighten_by_block[b1]
        self.assertEqual(len(vars_to_tighten), 1)
        self.assertIn(b1.x, vars_to_tighten)
        vars_to_tighten = vars_to_tighten_by_block[b2]
        self.assertEqual(len(vars_to_tighten), 1)
        self.assertIn(b2.x, vars_to_tighten)

        vars_to_tighten_by_block = collect_vars_to_tighten_by_block(m, method='leaves')
        self.assertEqual(len(vars_to_tighten_by_block), 3)
        vars_to_tighten = vars_to_tighten_by_block[m]
        self.assertEqual(len(vars_to_tighten), 0)
        vars_to_tighten = vars_to_tighten_by_block[b1]
        self.assertEqual(len(vars_to_tighten), 1)
        self.assertIn(b1.x, vars_to_tighten)
        vars_to_tighten = vars_to_tighten_by_block[b2]
        self.assertEqual(len(vars_to_tighten), 1)
        self.assertIn(b2.x, vars_to_tighten)

        vars_to_tighten_by_block = collect_vars_to_tighten_by_block(m, method='dbt')
        self.assertEqual(len(vars_to_tighten_by_block), 3)
        vars_to_tighten = vars_to_tighten_by_block[m]
        self.assertEqual(len(vars_to_tighten), 1)
        self.assertTrue(b1.z in vars_to_tighten or b2.z in vars_to_tighten)
        vars_to_tighten = vars_to_tighten_by_block[b1]
        self.assertEqual(len(vars_to_tighten), 1)
        self.assertIn(b1.x, vars_to_tighten)
        vars_to_tighten = vars_to_tighten_by_block[b2]
        self.assertEqual(len(vars_to_tighten), 1)
        self.assertIn(b2.x, vars_to_tighten)


class TestDBT(unittest.TestCase):
    def get_model(self):
        m = TreeBlock(concrete=True)
        m.setup(children_keys=[0, 1])
        b0 = m.children[0]
        b1 = m.children[1]
        b0.setup(children_keys=list())
        b1.setup(children_keys=list())

        b0.x = pe.Var(bounds=(-1, 1))
        b0.y = pe.Var(bounds=(-5, 5))
        b0.p = pe.Param(initialize=1.0, mutable=True)
        b0.c = coramin.relaxations.PWUnivariateRelaxation()
        b0.c.build(x=b0.x, aux_var=b0.y, shape=coramin.utils.FunctionShape.CONVEX, f_x_expr=b0.p*b0.x)

        b1.x = pe.Var(bounds=(-5, 5))
        b1.y = pe.Var(bounds=(-5, 5))
        b1.p = pe.Param(initialize=1.0, mutable=True)
        b1.c = coramin.relaxations.PWUnivariateRelaxation()
        b1.c.build(x=b1.x, aux_var=b1.y, shape=coramin.utils.FunctionShape.CONVEX, f_x_expr=b1.p*b1.x)

        m.linking_constraints.add(b0.y == b1.y)

        return m

    def test_full_space(self):
        m = self.get_model()
        b0 = m.children[0]
        b1 = m.children[1]
        opt = appsi.solvers.Gurobi()
        perform_dbt(relaxation=m, solver=opt, obbt_method=OBBTMethod.FULL_SPACE, filter_method=FilterMethod.NONE)
        self.assertAlmostEqual(b0.x.lb, -1)
        self.assertAlmostEqual(b0.x.ub, 1)
        self.assertAlmostEqual(b0.y.lb, -5)
        self.assertAlmostEqual(b0.y.ub, 5)
        self.assertAlmostEqual(b1.x.lb, -1)
        self.assertAlmostEqual(b1.x.ub, 1)
        self.assertAlmostEqual(b1.y.lb, -5)
        self.assertAlmostEqual(b1.y.ub, 5)

    def test_leaves(self):
        m = self.get_model()
        b0 = m.children[0]
        b1 = m.children[1]
        opt = appsi.solvers.Gurobi()
        perform_dbt(relaxation=m, solver=opt, obbt_method=OBBTMethod.LEAVES, filter_method=FilterMethod.NONE)
        self.assertAlmostEqual(b0.x.lb, -1)
        self.assertAlmostEqual(b0.x.ub, 1)
        self.assertAlmostEqual(b0.y.lb, -5)
        self.assertAlmostEqual(b0.y.ub, 5)
        self.assertAlmostEqual(b1.x.lb, -5)
        self.assertAlmostEqual(b1.x.ub, 5)
        self.assertAlmostEqual(b1.y.lb, -5)
        self.assertAlmostEqual(b1.y.ub, 5)

    def test_dbt(self):
        m = self.get_model()
        b0 = m.children[0]
        b1 = m.children[1]
        opt = appsi.solvers.Gurobi()
        perform_dbt(relaxation=m, solver=opt, obbt_method=OBBTMethod.DECOMPOSED, filter_method=FilterMethod.NONE)
        self.assertAlmostEqual(b0.x.lb, -1)
        self.assertAlmostEqual(b0.x.ub, 1)
        self.assertAlmostEqual(b0.y.lb, -1)
        self.assertAlmostEqual(b0.y.ub, 1)
        self.assertAlmostEqual(b1.x.lb, -1)
        self.assertAlmostEqual(b1.x.ub, 1)
        self.assertAlmostEqual(b1.y.lb, -1)
        self.assertAlmostEqual(b1.y.ub, 1)

    def test_dbt_with_filter(self):
        m = self.get_model()
        b0 = m.children[0]
        b1 = m.children[1]
        opt = appsi.solvers.Gurobi()
        perform_dbt(relaxation=m, solver=opt, obbt_method=OBBTMethod.DECOMPOSED, filter_method=FilterMethod.AGGRESSIVE)
        self.assertAlmostEqual(b0.x.lb, -1)
        self.assertAlmostEqual(b0.x.ub, 1)
        self.assertAlmostEqual(b0.y.lb, -1)
        self.assertAlmostEqual(b0.y.ub, 1)
        self.assertAlmostEqual(b1.x.lb, -1)
        self.assertAlmostEqual(b1.x.ub, 1)
        self.assertAlmostEqual(b1.y.lb, -1)
        self.assertAlmostEqual(b1.y.ub, 1)


class TestDBTWithECP(unittest.TestCase):
    def create_model(self):
        m = coramin.domain_reduction.TreeBlock(concrete=True)
        m.setup(children_keys=[1, 2])
        m.children[1].setup(children_keys=[1, 2])
        m.children[2].setup(children_keys=[1, 2])
        m.children[1].children[1].setup(children_keys=list())
        m.children[1].children[2].setup(children_keys=list())
        m.children[2].children[1].setup(children_keys=list())
        m.children[2].children[2].setup(children_keys=list())

        b1 = m.children[1].children[1]
        b2 = m.children[1].children[2]
        b3 = m.children[2].children[1]
        b4 = m.children[2].children[2]

        b1.x1 = pe.Var(bounds=(0.5, 5))
        b1.x2 = pe.Var(bounds=(0.5, 5))
        b1.x3 = pe.Var(bounds=(0.5, 5))

        b2.x4 = pe.Var(bounds=(0.5, 5))
        b2.x5 = pe.Var(bounds=(0.5, 5))
        b2.x6 = pe.Var(bounds=(0.5, 5))

        b3.x7 = pe.Var(bounds=(0.5, 5))
        b3.x8 = pe.Var(bounds=(0.5, 5))
        b3.x9 = pe.Var(bounds=(0.5, 5))

        b4.x10 = pe.Var(bounds=(0.5, 5))
        b4.x11 = pe.Var(bounds=(0.5, 5))
        b4.x12 = pe.Var(bounds=(0.5, 5))

        b1.c1 = pe.Constraint(expr=b1.x1 == b1.x2 ** 2 - b1.x3 ** 2)
        b1.c2 = pe.Constraint(expr=b1.x2 == pe.log(b1.x3) + b1.x3)

        b2.c1 = pe.Constraint(expr=b2.x4 == b2.x5 * b2.x6)
        b2.c2 = pe.Constraint(expr=b2.x5 == b2.x6 ** 2)

        b3.c1 = pe.Constraint(expr=b3.x7 == pe.log(b3.x8) - pe.log(b3.x9))
        b3.c2 = pe.Constraint(expr=b3.x8 + b3.x9 == 4)

        b4.c1 = pe.Constraint(expr=b4.x10 == b4.x11 * b4.x12 - b4.x12)
        b4.c2 = pe.Constraint(expr=b4.x11 + b4.x12 == 4)

        m.children[1].linking_constraints.add(b1.x3 == b2.x6)
        m.children[2].linking_constraints.add(b3.x9 == b4.x10)
        m.linking_constraints.add(b1.x3 == b3.x9)

        m.obj = pe.Objective(
            expr=b1.x1 + b1.x2 + b1.x3 + b2.x4 + b2.x5 + b2.x6 + b3.x7 + b3.x8 + b3.x9 + b4.x10 + b4.x11 + b4.x12)

        return m

    @pytest.mark.parallel
    @pytest.mark.two_proc
    @pytest.mark.three_proc
    def test_bounds_tightening(self):
        from mpi4py import MPI

        comm: MPI.Comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        m = self.create_model()
        coramin.relaxations.relax(m, descend_into=True, in_place=True)
        opt = coramin.algorithms.ECPBounder(subproblem_solver=appsi.solvers.Gurobi())
        opt.config.keep_cuts = False
        opt.config.feasibility_tol = 1e-5
        coramin.domain_reduction.perform_dbt(m, opt, filter_method=coramin.domain_reduction.FilterMethod.NONE,
                                             parallel=True)
        m.write(f'rank{rank}.lp')
        comm.Barrier()
        if rank == 0:
            self.assertTrue(filecmp.cmp('rank1.lp', f'rank{rank}.lp'), f'rank {rank}')
        else:
            self.assertTrue(filecmp.cmp('rank0.lp', f'rank{rank}.lp'), f'rank {rank}')

        # the next bit of code is needed to ensure the above test actually tests what we think it is testing
        m = self.create_model()
        coramin.relaxations.relax(m, descend_into=True, in_place=True)
        opt = coramin.algorithms.ECPBounder(subproblem_solver=appsi.solvers.Gurobi())
        opt.config.keep_cuts = False
        opt.config.feasibility_tol = 1e-5
        coramin.domain_reduction.perform_dbt(m, opt, filter_method=coramin.domain_reduction.FilterMethod.NONE,
                                             parallel=True, update_relaxations_between_stages=False)
        m.write(f'rank{rank}.lp')
        comm.Barrier()
        if rank == 0:
            self.assertFalse(filecmp.cmp('rank1.lp', f'rank{rank}.lp'))
        else:
            self.assertFalse(filecmp.cmp('rank0.lp', f'rank{rank}.lp'))
