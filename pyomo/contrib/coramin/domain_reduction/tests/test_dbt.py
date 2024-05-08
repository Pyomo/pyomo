from pyomo.contrib.coramin.domain_reduction.dbt import (
    TreeBlock,
    TreeBlockError,
    convert_pyomo_model_to_bipartite_graph,
    _VarNode,
    _ConNode,
    _RelNode,
    split_metis,
    num_cons_in_graph,
    collect_vars_to_tighten_by_block,
    decompose_model,
    perform_dbt,
    OBBTMethod,
    FilterMethod,
    DecompositionStatus,
    compute_partition_ratio,
)
from pyomo.common import unittest
import pyomo.environ as pe
from pyomo.contrib import coramin
from networkx import is_bipartite
from pyomo.common.collections import ComponentSet
from networkx import Graph
import filecmp
from pyomo.contrib import appsi
import pytest
from pyomo.contrib.coramin.utils.compare_models import is_relaxation, is_equivalent
from pyomo.contrib.coramin.utils.pyomo_utils import active_cons, active_vars

try:
    import metis

    metis_available = True
except:
    metis_available = False


if not metis_available:
    raise unittest.SkipTest('metis is not available')


class TestDecomposition(unittest.TestCase):
    def test_decomp1(self):
        m1 = pe.ConcreteModel()
        m1.x = x = pe.Var([1, 2, 3, 4, 5, 6], bounds=(-10, 10))
        m1.c = c = pe.ConstraintList()

        c.add(x[1] == x[2] + x[3])
        c.add(x[4] == x[5] + x[6])
        c.add(x[2] <= 2 * x[3] + 1)
        c.add(x[5] >= 2 * x[6] + 1)

        m2, reason = decompose_model(m1)
        self.assertEqual(reason, DecompositionStatus.normal)
        self.assertTrue(is_equivalent(m1, m2, appsi.solvers.Highs()))
        self.assertEqual(len(m2.children), 2)
        self.assertEqual(len(list(active_cons(m2.children[0]))), 2)
        self.assertEqual(len(list(active_cons(m2.children[1]))), 2)
        self.assertEqual(len(list(active_vars(m2.children[0]))), 3)
        self.assertEqual(len(list(active_vars(m2.children[1]))), 3)
        self.assertEqual(m2.get_block_stage(m2), 0)
        self.assertEqual(m2.get_block_stage(m2.children[0]), 1)
        self.assertEqual(m2.get_block_stage(m2.children[1]), 1)
        self.assertEqual(list(m2.stage_blocks(0)), [m2])
        self.assertEqual(list(m2.stage_blocks(1)), [m2.children[0], m2.children[1]])

    def test_decomp2(self):
        m1 = pe.ConcreteModel()
        m1.x = x = pe.Var([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], bounds=(-10, 10))
        m1.c = c = pe.ConstraintList()

        c.add(x[1] == x[2] + x[3])
        c.add(x[4] == x[5] + x[6])
        c.add(x[2] <= 2 * x[3] + 1)
        c.add(x[5] >= 2 * x[6] + 1)
        c.add(x[1] == x[4])

        c.add(x[7] == x[8] + x[9])
        c.add(x[10] == x[11] + x[12])
        c.add(x[8] <= 2 * x[9] + 1)
        c.add(x[11] >= 2 * x[12] + 1)
        c.add(x[7] == x[10])

        m2, reason = decompose_model(m1, limit_num_stages=False)
        self.assertEqual(reason, DecompositionStatus.normal)
        self.assertTrue(is_equivalent(m1, m2, appsi.solvers.Highs()))
        self.assertEqual(len(m2.children), 2)
        self.assertEqual(len(list(active_cons(m2.children[0]))), 5)
        self.assertEqual(len(list(active_cons(m2.children[1]))), 5)
        self.assertEqual(len(list(active_vars(m2.children[0]))), 6)
        self.assertEqual(len(list(active_vars(m2.children[1]))), 6)
        self.assertEqual(m2.get_block_stage(m2), 0)
        self.assertEqual(m2.get_block_stage(m2.children[0]), 1)
        self.assertEqual(m2.get_block_stage(m2.children[1]), 1)
        self.assertEqual(list(m2.stage_blocks(0)), [m2])
        self.assertEqual(list(m2.stage_blocks(1)), [m2.children[0], m2.children[1]])
        self.assertEqual(
            list(m2.stage_blocks(2)),
            [
                m2.children[0].children[0],
                m2.children[0].children[1],
                m2.children[1].children[0],
                m2.children[1].children[1],
            ],
        )

        for b in [m2.children[0], m2.children[1]]:
            self.assertEqual(len(b.children), 2)
            self.assertIn(len(list(active_cons(b.children[0]))), {2, 3})
            self.assertIn(len(list(active_cons(b.children[1]))), {2, 3})
            self.assertIn(len(list(active_vars(b.children[0]))), {3, 4})
            self.assertIn(len(list(active_vars(b.children[1]))), {3, 4})
            self.assertEqual(m2.get_block_stage(b), 1)
            self.assertEqual(m2.get_block_stage(b.children[0]), 2)
            self.assertEqual(m2.get_block_stage(b.children[1]), 2)
            self.assertEqual(len(b.coupling_vars), 1)

    def test_decomp3(self):
        m1 = pe.ConcreteModel()
        m1.x = x = pe.Var([1, 2, 3, 4, 5, 6], bounds=(-10, 10))
        m1.c = c = pe.ConstraintList()
        m1.rels = pe.Block()

        c.add(x[1] == x[2] + x[3])
        c.add(x[4] == x[5] + x[6])
        m1.rels.rel1 = coramin.relaxations.PWMcCormickRelaxation()
        m1.rels.rel1.build(x[2], x[3], aux_var=x[1])
        m1.rels.rel2 = coramin.relaxations.PWMcCormickRelaxation()
        m1.rels.rel2.build(x[5], x[6], aux_var=x[4])

        m2, reason = decompose_model(m1)
        self.assertEqual(reason, DecompositionStatus.normal)
        self.assertTrue(is_equivalent(m1, m2, appsi.solvers.Highs()))
        self.assertEqual(len(m2.children), 2)
        self.assertEqual(
            len(
                list(
                    coramin.relaxations.iterators.nonrelaxation_component_data_objects(
                        m2.children[0], pe.Constraint, descend_into=True, active=True
                    )
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                list(
                    coramin.relaxations.iterators.nonrelaxation_component_data_objects(
                        m2.children[1], pe.Constraint, descend_into=True, active=True
                    )
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                list(
                    coramin.relaxations.iterators.relaxation_data_objects(
                        m2.children[0], descend_into=True, active=True
                    )
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                list(
                    coramin.relaxations.iterators.relaxation_data_objects(
                        m2.children[1], descend_into=True, active=True
                    )
                )
            ),
            1,
        )
        self.assertEqual(len(list(active_vars(m2.children[0]))), 3)
        self.assertEqual(len(list(active_vars(m2.children[1]))), 3)
        self.assertEqual(m2.get_block_stage(m2), 0)
        self.assertEqual(m2.get_block_stage(m2.children[0]), 1)
        self.assertEqual(m2.get_block_stage(m2.children[1]), 1)
        self.assertEqual(list(m2.stage_blocks(0)), [m2])
        self.assertEqual(list(m2.stage_blocks(1)), [m2.children[0], m2.children[1]])
        pr = compute_partition_ratio(m1, m2)
        self.assertAlmostEqual(pr, 2)

    def test_objective(self):
        m1 = pe.ConcreteModel()
        m1.x = x = pe.Var([1, 2, 3, 4, 5, 6], bounds=(-10, 10))
        m1.c = c = pe.ConstraintList()

        c.add(x[1] == x[2] + x[3])
        c.add(x[4] == x[5] + x[6])
        c.add(x[2] <= 2 * x[3] + 1)
        c.add(x[5] >= 2 * x[6] + 1)
        m1.obj = pe.Objective(expr=sum(x.values()))

        m2, reason = decompose_model(m1)
        self.assertEqual(reason, DecompositionStatus.normal)
        opt = appsi.solvers.Highs()
        res1 = opt.solve(m1)
        res2 = opt.solve(m2)
        self.assertAlmostEqual(
            res1.best_feasible_objective, res2.best_feasible_objective
        )
        self.assertEqual(len(m2.children), 2)
        self.assertIn(len(list(active_cons(m2.children[0]))), {3, 4})
        self.assertIn(len(list(active_cons(m2.children[1]))), {3, 4})
        self.assertIn(len(list(active_vars(m2.children[0]))), {4, 5})
        self.assertIn(len(list(active_vars(m2.children[1]))), {4, 5})
        self.assertEqual(m2.get_block_stage(m2), 0)
        self.assertEqual(m2.get_block_stage(m2.children[0]), 1)
        self.assertEqual(m2.get_block_stage(m2.children[1]), 1)
        self.assertEqual(list(m2.stage_blocks(0)), [m2])
        self.assertEqual(list(m2.stage_blocks(1)), [m2.children[0], m2.children[1]])

    def test_refine_partition(self):
        m1 = pe.ConcreteModel()
        m1.x = x = pe.Var([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], bounds=(-10, 10))
        m1.c = c = pe.ConstraintList()

        c.add(x[1] == x[2] + x[3])
        c.add(x[4] == x[5] + x[6])
        c.add(x[2] <= 2 * x[3] + 1)
        c.add(x[5] >= 2 * x[6] + 1)
        c.add(x[1] == x[4])

        c.add(x[7] == x[8] + x[9])
        c.add(x[10] == x[11] + x[12])
        c.add(x[8] <= 2 * x[9] + 1)
        c.add(x[11] >= 2 * x[12] + 1)
        c.add(x[7] == x[10])

        c.add(sum(x.values()) == 1)
        m2, reason = decompose_model(m1)
        self.assertEqual(reason, DecompositionStatus.normal)
        opt = appsi.solvers.Highs()
        opt.config.stream_solver = True
        self.assertTrue(is_relaxation(m1, m2, appsi.solvers.Highs(), bigM=1000))
        self.assertTrue(is_relaxation(m2, m1, opt, bigM=1000))
        self.assertTrue(is_equivalent(m1, m2, appsi.solvers.Highs(), bigM=1000))
        self.assertEqual(len(m2.children), 2)
        self.assertEqual(len(m2.coupling_vars), 1)
        self.assertIn(len(list(active_cons(m2.children[0]))), {6, 7})
        self.assertIn(len(list(active_cons(m2.children[1]))), {6, 7})
        self.assertEqual(len(m2.coupling_vars), 1)
        self.assertIn(len(list(active_vars(m2.children[0]))), {7, 8})
        self.assertIn(len(list(active_vars(m2.children[1]))), {7, 8})


class TestTreeBlock(unittest.TestCase):
    def test_tree_block(self):
        b = TreeBlock(concrete=True)
        with self.assertRaises(TreeBlockError):
            b.is_leaf()
        with self.assertRaises(TreeBlockError):
            b.children
        with self.assertRaises(TreeBlockError):
            b.num_stages()
        with self.assertRaises(TreeBlockError):
            list(b.stage_blocks(0))
        with self.assertRaises(TreeBlockError):
            b.get_block_stage(b)
        b.setup(children_keys=list(), coupling_vars=list())
        self.assertTrue(b.is_leaf())
        b.x = pe.Var()  # make sure we can add components just like a regular block
        b.x.setlb(-1)
        with self.assertRaises(TreeBlockError):
            b.children
        self.assertEqual(b.num_stages(), 1)
        stage0_blocks = list(b.stage_blocks(0))
        self.assertEqual(len(stage0_blocks), 1)
        self.assertIs(stage0_blocks[0], b)
        stage1_blocks = list(b.stage_blocks(1))
        self.assertEqual(len(stage1_blocks), 0)
        self.assertEqual(b.get_block_stage(b), 0)

        b = TreeBlock(concrete=True)
        b.setup(children_keys=[1, 2], coupling_vars=list())
        b.children[1].setup(children_keys=list(), coupling_vars=list())
        b.children[2].setup(children_keys=['a', 'b'], coupling_vars=list())
        b.children[2].children['a'].setup(children_keys=list(), coupling_vars=list())
        b.children[2].children['b'].setup(children_keys=list(), coupling_vars=list())
        self.assertFalse(b.is_leaf())
        self.assertTrue(b.children[1].is_leaf())
        self.assertFalse(b.children[2].is_leaf())
        self.assertTrue(b.children[2].children['a'].is_leaf())
        self.assertTrue(b.children[2].children['b'].is_leaf())

        b.children[1].x = pe.Var()
        b.children[2].children['a'].x = pe.Var()
        b.children[2].children['b'].x = pe.Var()
        self.assertEqual(
            len(list(b.component_data_objects(pe.Var, descend_into=True, sort=True))), 3
        )

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

        self.assertEqual(len(b.coupling_vars), 0)


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
        self.assertTrue(
            ((id(m.x), id(m.c1)) in graph_edge_comps)
            or ((id(m.c1), id(m.x)) in graph_edge_comps)
        )
        self.assertTrue(
            ((id(m.y), id(m.c1)) in graph_edge_comps)
            or ((id(m.c1), id(m.y)) in graph_edge_comps)
        )
        self.assertTrue(
            ((id(m.z), id(m.c1)) in graph_edge_comps)
            or ((id(m.c1), id(m.z)) in graph_edge_comps)
        )
        self.assertTrue(
            ((id(m.x), id(m.c2)) in graph_edge_comps)
            or ((id(m.c2), id(m.x)) in graph_edge_comps)
        )
        self.assertFalse(
            ((id(m.y), id(m.c2)) in graph_edge_comps)
            or ((id(m.c2), id(m.y)) in graph_edge_comps)
        )
        self.assertTrue(
            ((id(m.z), id(m.c2)) in graph_edge_comps)
            or ((id(m.c2), id(m.z)) in graph_edge_comps)
        )
        self.assertTrue(
            ((id(m.x), id(m.c3)) in graph_edge_comps)
            or ((id(m.c3), id(m.x)) in graph_edge_comps)
        )
        self.assertTrue(
            ((id(m.y), id(m.c3)) in graph_edge_comps)
            or ((id(m.c3), id(m.y)) in graph_edge_comps)
        )
        self.assertTrue(
            ((id(m.z), id(m.c3)) in graph_edge_comps)
            or ((id(m.c3), id(m.z)) in graph_edge_comps)
        )
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
        self.assertAlmostEqual(partitioning_ratio, 3 * 12 / (12 * 1 + 6 * 2 + 6 * 2))

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

        coupling_vars = list(tree.coupling_vars)
        self.assertEqual(len(coupling_vars), 1)
        cv = coupling_vars[0]
        self.assertEqual(v4, cv)

        new_model = TreeBlock(concrete=True)
        tree.build_pyomo_model(block=new_model)
        new_vars = list(active_vars(new_model))
        new_cons = list(
            coramin.relaxations.nonrelaxation_component_data_objects(
                new_model,
                ctype=pe.Constraint,
                active=True,
                descend_into=True,
                sort=True,
            )
        )
        new_rels = list(
            coramin.relaxations.relaxation_data_objects(
                new_model, descend_into=True, active=True, sort=True
            )
        )
        self.assertEqual(len(new_vars), 6)
        self.assertEqual(len(new_cons), 2)
        self.assertEqual(len(new_rels), 2)
        self.assertEqual(len(new_model.children), 2)
        self.assertEqual(len(new_model.coupling_vars), 1)
        self.assertEqual(new_model.num_stages(), 2)

        stage0_vars = list(
            new_model.component_data_objects(pe.Var, descend_into=False, sort=True)
        )
        stage0_cons = list(
            new_model.component_data_objects(
                pe.Constraint, descend_into=False, sort=True, active=True
            )
        )
        stage0_rels = list(
            coramin.relaxations.relaxation_data_objects(
                new_model, descend_into=False, active=True, sort=True
            )
        )
        self.assertEqual(len(stage0_vars), 0)
        self.assertEqual(len(stage0_cons), 0)
        self.assertEqual(len(stage0_rels), 0)

        block_a = new_model.children[0]
        block_b = new_model.children[1]
        block_a_vars = ComponentSet(active_vars(block_a))
        block_b_vars = ComponentSet(active_vars(block_b))
        block_a_cons = ComponentSet(
            coramin.relaxations.nonrelaxation_component_data_objects(
                block_a, ctype=pe.Constraint, descend_into=True, active=True, sort=True
            )
        )
        block_b_cons = ComponentSet(
            coramin.relaxations.nonrelaxation_component_data_objects(
                block_b, ctype=pe.Constraint, descend_into=True, active=True, sort=True
            )
        )
        block_a_rels = ComponentSet(
            coramin.relaxations.relaxation_data_objects(
                block_a, descend_into=True, active=True, sort=True
            )
        )
        block_b_rels = ComponentSet(
            coramin.relaxations.relaxation_data_objects(
                block_b, descend_into=True, active=True, sort=True
            )
        )

        self.assertIn(len(block_a_vars), {3, 4})
        self.assertEqual(len(block_a_cons), 1)
        self.assertEqual(len(block_a_rels), 1)
        self.assertIn(len(block_b_vars), {3, 4})
        self.assertEqual(len(block_b_cons), 1)
        self.assertEqual(len(block_b_rels), 1)

        self.assertEqual(new_model.coupling_vars, [m.v4])


class TestNumCons(unittest.TestCase):
    def test_num_cons(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var()
        m.z = pe.Var()
        m.r = coramin.relaxations.PWUnivariateRelaxation()
        m.r.build(
            x=m.x,
            aux_var=m.y,
            shape=coramin.utils.FunctionShape.CONVEX,
            f_x_expr=pe.exp(m.x),
        )
        m.c = pe.Constraint(expr=m.z == 2 * m.x)
        g = convert_pyomo_model_to_bipartite_graph(m)
        self.assertEqual(num_cons_in_graph(g, include_rels=False), 1)
        self.assertEqual(num_cons_in_graph(g), 2)


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
        b2.aux = pe.Var()

        b1.c = pe.Constraint(expr=b1.x + b1.y + b1.z == 0)
        b2.c = pe.Constraint(expr=b2.x + b2.y + b1.z == 0)

        b1.r = coramin.relaxations.PWUnivariateRelaxation()
        b1.r.set_input(
            x=b1.x,
            aux_var=b1.aux,
            shape=coramin.utils.FunctionShape.CONVEX,
            f_x_expr=pe.exp(b1.x),
        )
        b1.r.rebuild()

        b2.r = coramin.relaxations.PWXSquaredRelaxation()
        b2.r.set_input(
            x=b2.x, aux_var=b2.aux, relaxation_side=coramin.utils.RelaxationSide.UNDER
        )
        b2.r.rebuild()

        m.coupling_vars.append(b1.z)

        vars_to_tighten_by_block = collect_vars_to_tighten_by_block(
            m, method='full_space'
        )
        self.assertEqual(len(vars_to_tighten_by_block), 1)
        vars_to_tighten = vars_to_tighten_by_block[m]
        self.assertEqual(len(vars_to_tighten), 1)
        self.assertIn(b1.x, vars_to_tighten)

        vars_to_tighten_by_block = collect_vars_to_tighten_by_block(m, method='leaves')
        self.assertIn(len(vars_to_tighten_by_block), {2, 3})
        vars_to_tighten = vars_to_tighten_by_block[m]
        self.assertEqual(len(vars_to_tighten), 0)
        vars_to_tighten = vars_to_tighten_by_block[b1]
        self.assertEqual(len(vars_to_tighten), 1)
        self.assertIn(b1.x, vars_to_tighten)
        vars_to_tighten = vars_to_tighten_by_block[b2]
        self.assertEqual(len(vars_to_tighten), 0)

        vars_to_tighten_by_block = collect_vars_to_tighten_by_block(m, method='dbt')
        self.assertEqual(len(vars_to_tighten_by_block), 3)
        vars_to_tighten = vars_to_tighten_by_block[m]
        self.assertEqual(len(vars_to_tighten), 1)
        self.assertTrue(b1.z in vars_to_tighten)
        vars_to_tighten = vars_to_tighten_by_block[b1]
        self.assertEqual(len(vars_to_tighten), 1)
        self.assertIn(b1.x, vars_to_tighten)
        vars_to_tighten = vars_to_tighten_by_block[b2]
        self.assertEqual(len(vars_to_tighten), 0)


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
        b0.c.build(
            x=b0.x,
            aux_var=b0.y,
            shape=coramin.utils.FunctionShape.CONVEX,
            f_x_expr=b0.p * b0.x,
        )

        b1.x = pe.Var(bounds=(-5, 5))
        b1.p = pe.Param(initialize=1.0, mutable=True)
        b1.c = coramin.relaxations.PWUnivariateRelaxation()
        b1.c.build(
            x=b1.x,
            aux_var=b0.y,
            shape=coramin.utils.FunctionShape.CONVEX,
            f_x_expr=b1.p * b1.x,
        )

        m.coupling_vars.append(b0.y)

        return m

    def test_full_space(self):
        m = self.get_model()
        b0 = m.children[0]
        b1 = m.children[1]
        opt = appsi.solvers.Highs()
        perform_dbt(
            relaxation=m,
            solver=opt,
            obbt_method=OBBTMethod.FULL_SPACE,
            filter_method=FilterMethod.NONE,
        )
        self.assertAlmostEqual(b0.x.lb, -1)
        self.assertAlmostEqual(b0.x.ub, 1)
        self.assertAlmostEqual(b0.y.lb, -5)
        self.assertAlmostEqual(b0.y.ub, 5)
        self.assertAlmostEqual(b1.x.lb, -1)
        self.assertAlmostEqual(b1.x.ub, 1)

    def test_leaves(self):
        m = self.get_model()
        b0 = m.children[0]
        b1 = m.children[1]
        opt = appsi.solvers.Gurobi()
        perform_dbt(
            relaxation=m,
            solver=opt,
            obbt_method=OBBTMethod.LEAVES,
            filter_method=FilterMethod.NONE,
        )
        self.assertAlmostEqual(b0.x.lb, -1)
        self.assertAlmostEqual(b0.x.ub, 1)
        self.assertAlmostEqual(b0.y.lb, -5)
        self.assertAlmostEqual(b0.y.ub, 5)
        self.assertAlmostEqual(b1.x.lb, -5)
        self.assertAlmostEqual(b1.x.ub, 5)

    def test_dbt(self):
        m = self.get_model()
        b0 = m.children[0]
        b1 = m.children[1]
        opt = appsi.solvers.Highs()
        perform_dbt(
            relaxation=m,
            solver=opt,
            obbt_method=OBBTMethod.DECOMPOSED,
            filter_method=FilterMethod.NONE,
        )
        self.assertAlmostEqual(b0.x.lb, -1)
        self.assertAlmostEqual(b0.x.ub, 1)
        self.assertAlmostEqual(b0.y.lb, -1)
        self.assertAlmostEqual(b0.y.ub, 1)
        self.assertAlmostEqual(b1.x.lb, -1)
        self.assertAlmostEqual(b1.x.ub, 1)

    def test_dbt_with_filter(self):
        m = self.get_model()
        b0 = m.children[0]
        b1 = m.children[1]
        opt = appsi.solvers.Highs()
        perform_dbt(
            relaxation=m,
            solver=opt,
            obbt_method=OBBTMethod.DECOMPOSED,
            filter_method=FilterMethod.AGGRESSIVE,
        )
        self.assertAlmostEqual(b0.x.lb, -1)
        self.assertAlmostEqual(b0.x.ub, 1)
        self.assertAlmostEqual(b0.y.lb, -1)
        self.assertAlmostEqual(b0.y.ub, 1)
        self.assertAlmostEqual(b1.x.lb, -1)
        self.assertAlmostEqual(b1.x.ub, 1)


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

        b1.c1 = pe.Constraint(expr=b1.x1 == b1.x2**2 - b1.x3**2)
        b1.c2 = pe.Constraint(expr=b1.x2 == pe.log(b1.x3) + b1.x3)

        b2.c1 = pe.Constraint(expr=b2.x4 == b2.x5 * b2.x6)
        b2.c2 = pe.Constraint(expr=b2.x5 == b2.x6**2)

        b3.c1 = pe.Constraint(expr=b3.x7 == pe.log(b3.x8) - pe.log(b3.x9))
        b3.c2 = pe.Constraint(expr=b3.x8 + b3.x9 == 4)

        b4.c1 = pe.Constraint(expr=b4.x10 == b4.x11 * b4.x12 - b4.x12)
        b4.c2 = pe.Constraint(expr=b4.x11 + b4.x12 == 4)

        m.children[1].linking_constraints.add(b1.x3 == b2.x6)
        m.children[2].linking_constraints.add(b3.x9 == b4.x10)
        m.linking_constraints.add(b1.x3 == b3.x9)

        m.obj = pe.Objective(
            expr=b1.x1
            + b1.x2
            + b1.x3
            + b2.x4
            + b2.x5
            + b2.x6
            + b3.x7
            + b3.x8
            + b3.x9
            + b4.x10
            + b4.x11
            + b4.x12
        )

        return m

    @pytest.mark.mpi
    def test_bounds_tightening(self):
        from mpi4py import MPI

        comm: MPI.Comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        m = self.create_model()
        coramin.relaxations.relax(m, descend_into=True, in_place=True)
        opt = coramin.algorithms.ECPBounder(subproblem_solver=appsi.solvers.Gurobi())
        opt.config.keep_cuts = False
        opt.config.feasibility_tol = 1e-5
        coramin.domain_reduction.perform_dbt(
            m,
            opt,
            filter_method=coramin.domain_reduction.FilterMethod.NONE,
            parallel=True,
        )
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
        coramin.domain_reduction.perform_dbt(
            m,
            opt,
            filter_method=coramin.domain_reduction.FilterMethod.NONE,
            parallel=True,
            update_relaxations_between_stages=False,
        )
        m.write(f'rank{rank}.lp')
        comm.Barrier()
        if rank == 0:
            self.assertFalse(filecmp.cmp('rank1.lp', f'rank{rank}.lp'))
        else:
            self.assertFalse(filecmp.cmp('rank0.lp', f'rank{rank}.lp'))
