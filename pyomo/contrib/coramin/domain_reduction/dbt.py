import networkx as nx
from typing import Sequence, MutableSet, Optional
from pyomo.contrib.fbbt.fbbt import fbbt, compute_bounds_on_expr
import time
import enum
import warnings
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from .obbt import perform_obbt as normal_obbt
from .filters import aggressive_filter
import pyomo.environ as pe
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet
from pyomo.common.collections.orderedset import OrderedSet
from coramin.relaxations.iterators import relaxation_data_objects, nonrelaxation_component_data_objects
from pyomo.core.expr.visitor import replace_expressions
import logging
import networkx
try:
    import metis
    metis_available = True
except ImportError:
    metis_available = False    
import numpy as np
import math
from pyomo.core.base.block import declare_custom_block, _BlockData
from coramin.utils.pyomo_utils import get_objective
from pyomo.core.base.var import _GeneralVarData
from coramin.relaxations.copy_relaxation import copy_relaxation_with_local_data
from coramin.relaxations.relaxations_base import BaseRelaxationData
from coramin.utils import RelaxationSide
from collections import defaultdict
from pyomo.core.expr import numeric_expr
from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
from pyomo.common.modeling import unique_component_name
from coramin.relaxations.split_expr import flatten_expr


logger = logging.getLogger(__name__)


class DecompositionError(Exception):
    pass


class TreeBlockError(DecompositionError):
    pass


@declare_custom_block(name='TreeBlock')
class TreeBlockData(_BlockData):
    def __init__(self, component):
        _BlockData.__init__(self, component)
        self._children_index = None
        self._children = None
        self._linking_constraints = None
        self._already_setup = False
        self._is_leaf = None
        self._is_root = True
        self._allow_changes = False

    def setup(self, children_keys):
        assert not self._already_setup
        self._already_setup = True
        if len(children_keys) == 0:
            self._is_leaf = True
        else:
            self._is_leaf = False
            del self._children_index
            del self._children
            del self._linking_constraints
            self._allow_changes = True
            self._children_index = pe.Set(initialize=children_keys)
            self._children = TreeBlock(self._children_index)
            self._linking_constraints = pe.ConstraintList()
            self._allow_changes = False
            for key in children_keys:
                child = self.children[key]
                child._is_root = False

    def _assert_setup(self):
        if not self._already_setup:
            raise TreeBlockError('The TreeBlock has not been setup yet. Please call the setup method.')

    def is_leaf(self):
        self._assert_setup()
        return self._is_leaf

    def add_component(self, name, val):
        self._assert_setup()
        if self.is_leaf() or self._allow_changes or not isinstance(val, _GeneralVarData):
            _BlockData.add_component(self, name, val)
        else:
            raise TreeBlockError('Pyomo variables cannot be added to a TreeBlock unless it is a leaf.')

    @property
    def children(self):
        self._assert_setup()
        if self.is_leaf():
            raise TreeBlockError('Leaf TreeBlocks do not have children. Please check the is_leaf method')
        return self._children

    @property
    def linking_constraints(self):
        self._assert_setup()
        if self.is_leaf():
            raise TreeBlockError('leaf TreeBlocks do not have linking_constraints. Please check the is_leaf method.')
        return self._linking_constraints

    def _num_stages(self):
        self._assert_setup()
        num_stages = 1
        if not self.is_leaf():
            num_stages += max([child._num_stages() for child in self.children.values()])
        return num_stages

    def num_stages(self):
        if not self._is_root:
            raise TreeBlockError('The num_stages method can only be called from the root TreeBlock')
        return self._num_stages()

    @staticmethod
    def _stage_blocks(children, count, stage):
        if count == stage:
            for child in children.values():
                yield child
        else:
            for child in children.values():
                if not child.is_leaf():
                    for b in TreeBlockData._stage_blocks(child.children, count+1, stage):
                        yield b

    def stage_blocks(self, stage, active=None):
        self._assert_setup()
        if not self._is_root:
            raise TreeBlockError('The num_stages method can only be called from the root TreeBlock')
        if stage == 0:
            if (active and self.active) or (not active):
                yield self
        elif not self.is_leaf():
            for b in self._stage_blocks(self.children, 1, stage):
                if (active and b.active) or (not active):
                    yield b

    def get_block_stage(self, block):
        self._assert_setup()
        if not self._is_root:
            raise TreeBlockError('The get_block_stage method can only be called from the root TreeBlock.')
        for stage_ndx in range(self.num_stages()):
            stage_blocks = OrderedSet(self.stage_blocks(stage_ndx))
            if block in stage_blocks:
                return stage_ndx
        return None


class _Node(object):
    def __init__(self, comp):
        self.comp = comp

    def is_var(self):
        return False

    def is_con(self):
        return False

    def is_rel(self):
        return False

    def __repr__(self):
        return str(self.comp)

    def __str__(self):
        return str(self.comp)

    def __eq__(self, other):
        if isinstance(other, _Node):
            return self.comp is other.comp
        return False

    def __hash__(self):
        return hash(id(self.comp))


class _VarNode(_Node):
    def is_var(self):
        return True


class _ConNode(_Node):
    def is_con(self):
        return True


class _RelNode(_Node):
    def is_rel(self):
        return True


class _Edge(object):
    def __init__(self, node1: _VarNode, node2: _Node):
        assert node1.is_var()
        self.node1 = node1
        self.node2 = node2

    def __str__(self):
        s = 'Edge from {0} to {1}'.format(str(self.node1), str(self.node2))
        return s


class _Tree(object):
    def __init__(self, children=None, edges_between_children=None):
        """
        Parameters
        ----------
        children: list or collections.abc.Iterable of _Tree or networkx.Graph
        edges_between_children: list or collections.abc.Iterable of _Edge
        """
        self.children = OrderedSet()
        self.edges_between_children = OrderedSet()
        if children is not None:
            self.children.update(children)
        if edges_between_children is not None:
            self.edges_between_children.update(edges_between_children)

    def build_pyomo_model(self, block):
        """
        Parameters
        ----------
        block: TreeBlockData
            empty TreeBlock

        Returns
        -------
        component_map: pe.ComponentMap
        """
        block.setup(children_keys=list(range(len(self.children))))
        component_map = pe.ComponentMap()
        replacement_map_by_child = dict()

        for i, child in enumerate(self.children):
            if isinstance(child, _Tree):
                tmp_component_map = child.build_pyomo_model(block=block.children[i])
            elif isinstance(child, networkx.Graph):
                block.children[i].setup(children_keys=list())
                tmp_component_map = build_pyomo_model_from_graph(graph=child, block=block.children[i])
            else:
                raise ValueError('Unexpected child type: {0}'.format(str(type(child))))
            replacement_map_by_child[child] = tmp_component_map
            component_map.update(tmp_component_map)

        logger.debug('creating linking cons linking the children of {0}'.format(str(block)))
        for edge in self.edges_between_children:
            logger.debug('adding linking constraint for edge {0}'.format(str(edge)))
            if edge.node1.comp is not edge.node2.comp:
                raise DecompositionError('Edge {0} node1.comp is not node2.comp'.format(edge))
            if edge.node1.comp not in component_map:
                logger.warning('Edge {0} node {1} is not in the component map'.format(str(edge), str(edge.node1)))
            all_children = list(self.children)
            assert len(all_children) == 2
            child0 = all_children[0]
            child1 = all_children[1]
            v1 = replacement_map_by_child[child0][edge.node1.comp]
            v2 = replacement_map_by_child[child1][edge.node2.comp]
            assert v1 is not v2
            block.linking_constraints.add(v1 == v2)

        return component_map

    def log(self, prefix=''):
        logger.debug(prefix + '# Edges: {0}'.format(len(self.edges_between_children)))
        for _child in self.children:
            if isinstance(_child, _Tree):
                _child.log(prefix=prefix + '  ')
            else:
                logger.debug(prefix + '  Leaf: # NNZ: {0}'.format(_child.number_of_edges()))


def _is_dominated(ndx, num_cuts, balance, num_cuts_array, balance_array):
    cut_diff = ((num_cuts - num_cuts_array) >= 0)
    balance_diff = ((abs(balance - 0.5) - abs(balance_array - 0.5)) >= 0)
    cut_diff[ndx] = False
    balance_diff[ndx] = False
    return np.any(cut_diff & balance_diff)


def _networkx_to_adjacency_list(graph: networkx.Graph):
    adj_list = list()
    node_to_ndx_map = dict()
    for ndx, node in enumerate(graph.nodes):
        node_to_ndx_map[node] = ndx

    for ndx, node in enumerate(graph.nodes):
        adj_list.append(list())
        for other_node in graph.adj[node].keys():
            other_ndx = node_to_ndx_map[other_node]
            adj_list[ndx].append(other_ndx)

    return adj_list


def choose_metis_partition(graph, max_size_diff_trials, seed_trials):
    """
    Parameters
    ----------
    graph: networkx.Graph
    max_size_diff_trials: list of float
    seed_trials: list of int

    Returns
    -------
    max_size_diff_selected: float
    seed_selected: float
    """
    if not metis_available:
        raise ImportError('Cannot perform graph partitioning without metis. Please install metis (including the python bindings).')
    cut_list = list()
    for _max_size_diff in max_size_diff_trials:
        for _seed in seed_trials:
            if _seed is None:
                edgecuts, parts = metis.part_graph(_networkx_to_adjacency_list(graph), nparts=2, ubvec=[1 + _max_size_diff])
            else:
                edgecuts, parts = metis.part_graph(_networkx_to_adjacency_list(graph), nparts=2, ubvec=[1 + _max_size_diff], seed=_seed)
            cut_list.append((edgecuts, sum(parts)/graph.number_of_nodes(), _max_size_diff, _seed))
    cut_list.sort(key=lambda i: i[0])

    ############################
    # get the "pareto front" obtained with metis
    ############################
    num_cuts_array = np.array([i[0] for i in cut_list])
    balance_array = np.array([i[1] for i in cut_list])

    pareto_list = list()
    for ndx, partition in enumerate(cut_list):
        num_cuts = partition[0]
        balance = partition[1]
        if not _is_dominated(ndx, num_cuts, balance, num_cuts_array, balance_array):
            pareto_list.append(partition)
    if len(pareto_list) == 0:
        pareto_list.append(cut_list[0])

    selection = 0
    chosen_partition = pareto_list[selection]
    max_size_diff_selected = chosen_partition[2]
    seed_selected = chosen_partition[3]
    return max_size_diff_selected, seed_selected


def evaluate_partition(original_graph, tree):
    """
    Parameters
    ----------
    original_graph: networkx.Graph
    tree: _Tree
    """
    original_graph_nnz = original_graph.number_of_edges()
    original_graph_n_vars_to_tighten = len(collect_vars_to_tighten_from_graph(graph=original_graph))
    original_obbt_nnz = original_graph_nnz * original_graph_n_vars_to_tighten

    tree_obbt_nnz = 0
    tree_nnz = 0
    assert len(tree.children) == 2
    for child in tree.children:
        assert isinstance(child, networkx.Graph)
        child_nnz = child.number_of_edges()
        tree_nnz += child_nnz
        child_n_vars_to_tighten = len(collect_vars_to_tighten_from_graph(graph=child))
        tree_obbt_nnz += child_nnz * child_n_vars_to_tighten
    tree_nnz += 2 * len(tree.edges_between_children)
    tree_obbt_nnz += tree_nnz * len(tree.edges_between_children)
    partitioning_ratio = original_obbt_nnz / tree_obbt_nnz
    return partitioning_ratio


def _refine_partition(graph: nx.Graph, model: _BlockData,
                      removed_edges: Sequence[_Edge],
                      graph_a_nodes: MutableSet[_Node],
                      graph_b_nodes: MutableSet[_Node]):
    con_count = defaultdict(int)
    for edge in removed_edges:
        n1, n2 = edge.node1, edge.node2
        if n1.is_con():
            con_count[n1.comp] += 1
        if n2.is_con():
            con_count[n2.comp] += 1

    for c, count in con_count.items():
        if count < 3:
            continue

        new_body = flatten_expr(c.body)

        if type(new_body) is not numeric_expr.SumExpression:
            logger.info(f'Constraint {str(c)} is contributing to {count} removed '
                        f'edges, but we cannot split the constraint because the '
                        f'body is not a SumExpression.')
            continue

        graph_a_args = list()
        graph_b_args = list()
        correct_structure = True
        for arg in new_body.args:
            graph_a_arg_vars = ComponentSet()
            graph_b_arg_vars = ComponentSet()
            for v in identify_variables(arg, include_fixed=False):
                v_node = _VarNode(v)
                assert v_node in graph_a_nodes or v_node in graph_b_nodes
                if v_node in graph_a_nodes:
                    graph_a_arg_vars.add(v)
                else:
                    graph_b_arg_vars.add(v)
            if len(graph_a_arg_vars) > 0 and len(graph_b_arg_vars) > 0:
                correct_structure = False
                break
            if len(graph_a_arg_vars) > 0:
                graph_a_args.append(arg)
            elif len(graph_b_arg_vars) > 0:
                graph_b_args.append(arg)
            else:
                graph_a_args.append(arg)

        if not correct_structure:
            logger.info(f'Constriant {str(c)} is contributing to {count} removed '
                        f'edges, but we cannot split the constraint because some of '
                        f'the terms in the SumExpression contain variables from both '
                        f'partitions.')
            continue

        # update the model
        if not hasattr(model, 'dbt_partition_vars'):
            model.dbt_partition_vars = pe.VarList()
            model.dbt_partition_cons = pe.ConstraintList()

        graph_a_var = model.dbt_partition_vars.add()
        graph_b_var = model.dbt_partition_vars.add()

        if c.lower is not None and c.upper is not None:
            new_c1 = model.dbt_partition_cons.add(graph_a_var == sum(graph_a_args))
            new_c2 = model.dbt_partition_cons.add(graph_b_var == sum(graph_b_args))
            if c.equality:
                new_c3 = model.dbt_partition_cons.add(graph_a_var + graph_b_var == c.lower)
            else:
                new_c3 = model.dbt_partition_cons.add((c.lower, graph_a_var + graph_b_var, c.upper))
        elif c.lower is None:
            assert c.upper is not None
            new_c1 = model.dbt_partition_cons.add(graph_a_var >= sum(graph_a_args))
            new_c2 = model.dbt_partition_cons.add(graph_b_var >= sum(graph_b_args))
            new_c3 = model.dbt_partition_cons.add(graph_a_var + graph_b_var <= c.upper)
        else:
            assert c.upper is None
            new_c1 = model.dbt_partition_cons.add(graph_a_var <= sum(graph_a_args))
            new_c2 = model.dbt_partition_cons.add(graph_b_var <= sum(graph_b_args))
            new_c3 = model.dbt_partition_cons.add(graph_a_var + graph_b_var >= c.lower)
        c.deactivate()

        # update the graph
        graph.remove_node(_ConNode(c))
        graph.add_node(_VarNode(graph_a_var))
        graph.add_node(_VarNode(graph_b_var))
        for new_con in [new_c1, new_c2, new_c3]:
            graph.add_node(_ConNode(new_con))
            for v in identify_variables(new_con.body, include_fixed=False):
                graph.add_edge(_VarNode(v), _ConNode(new_con))

        # update removed_edges
        new_removed_edges = list()
        for e in removed_edges:
            if e.node2.comp is not c:
                new_removed_edges.append(e)

        new_removed_edges.append(_Edge(_VarNode(graph_a_var), _ConNode(new_c3)))
        removed_edges = new_removed_edges

        # update graph_a_nodes and graph_b_nodes
        graph_a_nodes.discard(_ConNode(c))
        graph_b_nodes.discard(_ConNode(c))
        graph_a_nodes.add(_VarNode(graph_a_var))
        graph_b_nodes.add(_VarNode(graph_b_var))
        graph_a_nodes.add(_ConNode(new_c1))
        graph_b_nodes.add(_ConNode(new_c2))
        graph_b_nodes.add(_ConNode(new_c3))

    return removed_edges


def split_metis(graph, model):
    """
    Parameters
    ----------
    graph: networkx.Graph
    model: _BlockData

    Returns
    -------
    tree: _Tree
    """
    if not metis_available:
        raise ImportError('Cannot perform graph partitioning without metis. Please install metis (including the python bindings).')
    max_size_diff, seed = choose_metis_partition(graph, max_size_diff_trials=[0.15], seed_trials=list(range(10)))
    if seed is None:
        edgecuts, parts = metis.part_graph(_networkx_to_adjacency_list(graph), nparts=2, ubvec=[1 + max_size_diff])
    else:
        edgecuts, parts = metis.part_graph(_networkx_to_adjacency_list(graph), nparts=2, ubvec=[1 + max_size_diff], seed=seed)

    graph_a_nodes = OrderedSet()
    graph_b_nodes = OrderedSet()
    for ndx, n in enumerate(graph.nodes()):
        if parts[ndx] == 0:
            graph_a_nodes.add(n)
        else:
            assert parts[ndx] == 1
            graph_b_nodes.add(n)

    removed_edges = list()
    for n1, n2 in graph.edges():
        if not n1.is_var():
            assert n2.is_var()
            n1, n2 = n2, n1
        else:
            assert not n2.is_var()
        if n1 in graph_a_nodes and n2 in graph_a_nodes:
            continue
        elif n1 in graph_b_nodes and n2 in graph_b_nodes:
            continue
        else:
            removed_edges.append(_Edge(n1, n2))

    removed_edges = _refine_partition(graph=graph, model=model,
                                      removed_edges=removed_edges,
                                      graph_a_nodes=graph_a_nodes,
                                      graph_b_nodes=graph_b_nodes)

    graph_a_edges = list()
    graph_b_edges = list()
    for n1, n2 in graph.edges():
        if not n1.is_var():
            assert n2.is_var()
            n1, n2 = n2, n1
        else:
            assert not n2.is_var()
        if n1 in graph_a_nodes and n2 in graph_a_nodes:
            graph_a_edges.append((n1, n2))
        elif n1 in graph_b_nodes and n2 in graph_b_nodes:
            graph_b_edges.append((n1, n2))
        else:
            continue

    linking_edges = list()
    new_var_nodes_dict = dict()
    for e in removed_edges:
        n1, n2 = e.node1, e.node2
        assert n1.is_var()
        assert not n2.is_var()
        if n1 in new_var_nodes_dict:
            new_var_node = new_var_nodes_dict[n1]
        else:
            new_var_node = _VarNode(n1.comp)
            new_var_nodes_dict[n1] = new_var_node
            linking_edge = _Edge(n1, new_var_node)
            linking_edges.append(linking_edge)
        if n1 in graph_a_nodes:
            assert n2 in graph_b_nodes
            graph_b_edges.append((new_var_node, n2))
        else:
            assert n1 in graph_b_nodes
            assert n2 in graph_a_nodes
            graph_a_edges.append((new_var_node, n2))

    graph_a = networkx.Graph()
    graph_b = networkx.Graph()

    graph_a.add_nodes_from(graph_a_nodes)
    graph_b.add_nodes_from(graph_b_nodes)
    graph_a.add_edges_from(graph_a_edges)
    graph_b.add_edges_from(graph_b_edges)

    if ((graph_a.number_of_nodes() >= 0.99 * graph.number_of_nodes()) or
            (graph_b.number_of_nodes() >= 0.99 * graph.number_of_nodes())):
        raise DecompositionError('Failed to partition graph')

    tree = _Tree(children=[graph_a, graph_b], edges_between_children=linking_edges)

    partitioning_ratio = evaluate_partition(original_graph=graph, tree=tree)

    return tree, partitioning_ratio


def convert_pyomo_model_to_bipartite_graph(m: _BlockData):
    """
    Parameters
    ----------
    m: _BlockData

    Returns
    -------
    graph: networkx.Graph
    """
    graph = networkx.Graph()
    var_map = pe.ComponentMap()

    for v in nonrelaxation_component_data_objects(m, pe.Var, sort=True, descend_into=True):
        if v.fixed:
            continue
        var_map[v] = _VarNode(v)
        graph.add_node(var_map[v])

    for b in relaxation_data_objects(m, descend_into=True, active=True, sort=True):
        node2 = _RelNode(b)
        for v in (list(b.get_rhs_vars()) + [b.get_aux_var()]):
            node1 = var_map[v]
            graph.add_edge(node1, node2)

    for c in nonrelaxation_component_data_objects(m, pe.Constraint, active=True, sort=True, descend_into=True):
        node2 = _ConNode(c)
        for v in identify_variables(c.body, include_fixed=False):
            node1 = var_map[v]
            graph.add_edge(node1, node2)

    return graph


def build_pyomo_model_from_graph(graph, block):
    """
    Parameters
    ----------
    graph: networkx.Graph
    block: pe.Block

    Returns
    -------
    component_map: pe.ComponentMap
    """
    vars = list()
    cons = list()
    rels = list()
    var_names = list()
    con_names = list()
    rel_names = list()
    for node in graph.nodes():
        if node.is_var():
            vars.append(node)
            var_names.append(node.comp.getname(fully_qualified=True).replace('.', '_'))
        elif node.is_con():
            cons.append(node)
            con_names.append(node.comp.getname(fully_qualified=True).replace('.', '_'))
        else:
            assert node.is_rel()
            rels.append(node)
            rel_names.append(node.comp.getname(fully_qualified=True).replace('.', '_'))

    assert len(vars) == len(set(vars))
    assert len(cons) == len(set(cons))
    assert len(rels) == len(set(rels))

    block.var_names = pe.Set(initialize=var_names)
    block.con_names = pe.Set(initialize=con_names)
    block.vars = pe.Var(block.var_names)
    block.cons = pe.Constraint(block.con_names)
    block.rels = pe.Block()

    component_map = pe.ComponentMap()
    for v_name, v in zip(var_names, vars):
        new_v = block.vars[v_name]
        component_map[v.comp] = new_v
        new_v.setlb(v.comp.lb)
        new_v.setub(v.comp.ub)
        new_v.domain = v.comp.domain
        if v.comp.is_fixed():
            new_v.fix(v.comp.value)
        new_v.set_value(v.comp.value, skip_validation=True)

    var_map = {id(k): v for k, v in component_map.items()}

    for c_name, c in zip(con_names, cons):
        if c.comp.equality:
            block.cons[c_name] = (replace_expressions(c.comp.body, substitution_map=var_map,
                                                      remove_named_expressions=True) == c.comp.lower)
        else:
            block.cons[c_name] = (pe.inequality(lower=c.comp.lower,
                                                body=replace_expressions(c.comp.body, substitution_map=var_map,
                                                                         remove_named_expressions=True),
                                                upper=c.comp.upper))
        component_map[c.comp] = block.cons[c_name]

    for r_name, r in zip(rel_names, rels):
        new_rel = copy_relaxation_with_local_data(r.comp, var_map)
        setattr(block.rels, r_name, new_rel)
        new_rel.rebuild()
        component_map[r.comp] = new_rel

    return component_map


def num_cons_in_graph(graph, include_rels=True):
    res = 0

    if include_rels:
        for n in graph.nodes():
            if n.is_con() or n.is_rel():
                res += 1
    else:
        for n in graph.nodes():
            if n.is_con():
                res += 1

    return res


class DecompositionStatus(enum.Enum):
    normal = 0  # the model was successfullay decomposed at least once and no exception was raised
    error = 1  # an exception was raised
    bad_ratio = 2  # the model could not be decomposed at all because the min_parition_ratio was not satisfied
    problem_too_small = 3  # the model could not be decomposed at all because the number of jacobian nonzeros in the original problem was less than max_leaf_nnz


def compute_partition_ratio(original_model: _BlockData, decomposed_model: TreeBlockData):
    graph = convert_pyomo_model_to_bipartite_graph(original_model)
    pr_numerator = graph.number_of_edges() * len(collect_vars_to_tighten(original_model))

    pr_denominator = 0
    vars_to_tighten_by_block = collect_vars_to_tighten_by_block(decomposed_model, 'dbt')
    for block, vars_to_tighten in vars_to_tighten_by_block.items():
        pr_denominator += len(vars_to_tighten) * convert_pyomo_model_to_bipartite_graph(block).number_of_edges()
    pr = pr_numerator / pr_denominator
    return pr


def _reformulate_objective(model):
    current_obj = get_objective(model)
    if current_obj is None:
        raise ValueError('No active objective found!')
    if not current_obj.expr.is_variable_type():
        obj_var_name = unique_component_name(model, 'obj_var')
        obj_var = pe.Var(bounds=compute_bounds_on_expr(current_obj.expr))
        model.add_component(obj_var_name, obj_var)
        model.del_component(current_obj)
        new_objective = pe.Objective(expr=model.obj_var)
        new_obj_name = unique_component_name(model, 'objective')
        model.add_component(new_obj_name, new_objective)
        if current_obj.sense == pe.minimize:
            obj_con = pe.Constraint(expr=current_obj.expr <= obj_var)
        else:
            obj_con = pe.Constraint(expr=current_obj.expr >= obj_var)
            new_objective.sense = pe.maximize
        obj_con_name = unique_component_name(model, 'obj_con')
        model.add_component(obj_con_name, obj_con)


def _eliminate_mutable_params(model):
    sub_map = dict()
    for p in nonrelaxation_component_data_objects(model, pe.Param, descend_into=True):
        sub_map[id(p)] = p.value

    for c in nonrelaxation_component_data_objects(model, pe.Constraint, active=True, descend_into=True):
        if c.lower is None:
            new_lower = None
        else:
            new_lower = replace_expressions(c.lower, sub_map,
                                            descend_into_named_expressions=True,
                                            remove_named_expressions=True)
        new_body = replace_expressions(c.body, sub_map,
                                       descend_into_named_expressions=True,
                                       remove_named_expressions=True)
        if c.upper is None:
            new_upper = None
        else:
            new_upper = replace_expressions(c.upper, sub_map,
                                            descend_into_named_expressions=True,
                                            remove_named_expressions=True)
        c.set_value((new_lower, new_body, new_upper))


def _decompose_model(model: _BlockData, max_leaf_nnz: Optional[int] = None,
                     min_partition_ratio: float = 1.25, limit_num_stages: bool = True):
    """
    Parameters
    ----------
    model: _BlockData
        The model to decompose
    max_leaf_nnz: int
        maximum number nonzeros in the constraint jacobian of the leaves
    min_partition_ratio: float
        If the partition ration is less than min_partition_ratio, the partition is not
        accepted and partitioning stops. This value should be between 1 and 2.
    limit_num_stages: bool
        If True, partitioning will stop before the number of stages produced exceeds
        round(math.log10(number of nonzeros in the constraint jacobian of model))

    Returns
    -------
    new_model: TreeBlockData
        The decomposed model
    component_map: pe.ComponentMap
        A ComponentMap mapping varialbes and constraints in model to those in new_model
    termination_reason: DecompositionStatus
        An enum member from DecompositionStatus
    """

    # by reformulating the objective, we can make better use of the incumbent when
    # doing OBBT
    _reformulate_objective(model)
    # we don't want the original param objects to be in the new model
    _eliminate_mutable_params(model)

    graph = convert_pyomo_model_to_bipartite_graph(model)
    logger.debug('converted pyomo model to bipartite graph')
    original_nnz = graph.number_of_edges()
    if limit_num_stages:
        max_stages = round(math.log10(original_nnz))
    else:
        max_stages = math.inf
    logger.debug('NNZ in original graph: {0}'.format(original_nnz))
    logger.debug('maximum number of stages: {0}'.format(max_stages))
    if max_leaf_nnz is None:
        max_leaf_nnz = 0.1 * original_nnz

    if original_nnz <= max_leaf_nnz or num_cons_in_graph(graph) <= 1:
        if original_nnz <= max_leaf_nnz:
            logger.debug('too few NNZ in original graph; not decomposing')
        else:
            logger.debug('Cannot decompose graph with less than 2 constraints.')
        new_model = TreeBlock(concrete=True)
        new_model.setup(children_keys=list())
        component_map = build_pyomo_model_from_graph(graph=graph, block=new_model)
        termination_reason = DecompositionStatus.problem_too_small
        logger.debug('done building pyomo model from graph')
    else:
        root_tree, partitioning_ratio = split_metis(graph=graph, model=model)
        logger.debug('partitioned original tree; partitioning ratio: {ratio}'.format(
            ratio=partitioning_ratio))
        if partitioning_ratio < min_partition_ratio:
            logger.debug('obtained bad partitioning ratio; abandoning partition')
            new_model = TreeBlock(concrete=True)
            new_model.setup(children_keys=list())
            component_map = build_pyomo_model_from_graph(graph=graph, block=new_model)
            termination_reason = DecompositionStatus.bad_ratio
            logger.debug('done building pyomo model from graph')
        else:
            parent = root_tree

            termination_reason = DecompositionStatus.normal
            needs_split = list()
            for child in parent.children:
                logger.debug(
                    'number of NNZ in child: {0}'.format(child.number_of_edges()))
                if child.number_of_edges() > max_leaf_nnz and num_cons_in_graph(
                        child) > 1:
                    needs_split.append((child, parent, 1))

            while len(needs_split) > 0:
                logger.debug('needs_split: {0}'.format(str(needs_split)))
                _graph, _parent, _stage = needs_split.pop()
                try:
                    if _stage + 1 >= max_stages:
                        logger.debug(f'stage {_stage}: not partitiong graph with '
                                     f'{_graph.number_of_edges()} NNZ due to the max '
                                     f'stages rule;')
                        continue
                    logger.debug(f'stage {_stage}: partitioning graph with '
                                 f'{_graph.number_of_edges()} NNZ')
                    sub_tree, partitioning_ratio = split_metis(graph=_graph,
                                                               model=model)
                    logger.debug(
                        'partitioning ratio: {ratio}'.format(ratio=partitioning_ratio))
                    if partitioning_ratio > min_partition_ratio:
                        logger.debug('partitioned {0}'.format(str(_graph)))
                        _parent.children.discard(_graph)
                        _parent.children.add(sub_tree)

                        for child in sub_tree.children:
                            logger.debug('number of NNZ in child: {0}'.format(
                                child.number_of_edges()))
                            if (child.number_of_edges() > max_leaf_nnz
                                    and num_cons_in_graph(child) > 1):
                                needs_split.append((child, sub_tree, _stage + 1))
                    else:
                        logger.debug(
                            'obtained bad partitioning ratio; abandoning partition')
                except DecompositionError:
                    termination_reason = DecompositionStatus.error
                    logger.error('failed to partition graph with {0} NNZ'.format(
                        _graph.number_of_edges()))

            logger.debug('Tree Info:')
            root_tree.log()

            new_model = TreeBlock(concrete=True)
            component_map = root_tree.build_pyomo_model(block=new_model)
            logger.debug('done building pyomo model from tree')

    obj = get_objective(model)
    if obj is not None:
        var_map = {id(k): v for k, v in component_map.items()}
        new_model.objective = pe.Objective(
            expr=replace_expressions(obj.expr, substitution_map=var_map,
                                     remove_named_expressions=True),
            sense=obj.sense)
        logger.debug('done adding objective to new model')
    else:
        logger.debug('No objective was found to add to the new model')

    return new_model, component_map, termination_reason


def decompose_model(model: _BlockData, max_leaf_nnz: Optional[int] = None,
                    min_partition_ratio: float = 1.25, limit_num_stages: bool = True):
    """
    Parameters
    ----------
    model: _BlockData
        The model to decompose
    max_leaf_nnz: int
        maximum number nonzeros in the constraint jacobian of the leaves
    min_partition_ratio: float
        If the partition ration is less than min_partition_ratio, the partition is not
        accepted and partitioning stops. This value should be between 1 and 2.
    limit_num_stages: bool
        If True, partitioning will stop before the number of stages produced exceeds
        round(math.log10(number of nonzeros in the constraint jacobian of model))

    Returns
    -------
    new_model: TreeBlockData
        The decomposed model
    component_map: pe.ComponentMap
        A ComponentMap mapping varialbes and constraints in model to those in new_model
    termination_reason: DecompositionStatus
        An enum member from DecompositionStatus
    """
    # we have to clone the model because we modify it in _refine_partition
    all_comps = list(ComponentSet(
        nonrelaxation_component_data_objects(model, pe.Var, descend_into=True)))
    all_comps.extend(ComponentSet(
        nonrelaxation_component_data_objects(model, pe.Constraint, active=True,
                                             descend_into=True)))
    all_comps.extend(relaxation_data_objects(model, descend_into=True, active=True))
    all_comps.extend(ComponentSet(
        nonrelaxation_component_data_objects(model, pe.Objective, active=True,
                                             descend_into=True)))
    tmp_name = unique_component_name(model, 'all_comps')
    setattr(model, tmp_name, all_comps)
    new_model = model.clone()
    old_to_new_comps_map = pe.ComponentMap(zip(getattr(model, tmp_name),
                                               getattr(new_model, tmp_name)))
    delattr(model, tmp_name)
    delattr(new_model, tmp_name)
    model = new_model

    tmp = _decompose_model(model, max_leaf_nnz=max_leaf_nnz,
                           min_partition_ratio=min_partition_ratio,
                           limit_num_stages=limit_num_stages)
    tree_model, component_map, termination_reason = tmp

    for orig_comp, clone_comp in list(old_to_new_comps_map.items()):
        if clone_comp in component_map:
            old_to_new_comps_map[orig_comp] = component_map[clone_comp]

    return tree_model, old_to_new_comps_map, termination_reason


def collect_vars_to_tighten_from_graph(graph):
    vars_to_tighten = ComponentSet()

    for n in graph.nodes():
        if n.is_rel():
            rel: BaseRelaxationData = n.comp
            if rel.is_rhs_convex() and rel.relaxation_side == RelaxationSide.UNDER and not rel.use_linear_relaxation:
                continue
            if rel.is_rhs_concave() and rel.relaxation_side == RelaxationSide.OVER and not rel.use_linear_relaxation:
                continue
            vars_to_tighten.update(rel.get_rhs_vars())
        elif n.is_var():
            v = n.comp
            if v.is_binary() or v.is_integer():
                vars_to_tighten.add(v)

    return vars_to_tighten


def collect_vars_to_tighten(block):
    graph = convert_pyomo_model_to_bipartite_graph(block)
    vars_to_tighten = collect_vars_to_tighten_from_graph(graph=graph)
    return vars_to_tighten


def collect_vars_to_tighten_by_block(m, method):
    """
    Parameters
    ----------
    m: TreeBlockData
    method: str
        'full_space', 'dbt', or 'leaves'

    Returns
    -------
    vars_to_tighten_by_block: dict
        maps Block to ComponentSet of Var
    """
    assert method in {'full_space', 'dbt', 'leaves'}

    vars_to_tighten_by_block = dict()

    assert isinstance(m, TreeBlockData)

    all_vars_to_account_for = collect_vars_to_tighten(m)

    for stage in range(m.num_stages()):
        for block in m.stage_blocks(stage, active=True):
            if block.is_leaf():
                vars_to_tighten_by_block[block] = collect_vars_to_tighten(block=block)
            elif method == 'leaves':
                vars_to_tighten_by_block[block] = ComponentSet()
            elif method == 'full_space':
                vars_to_tighten_by_block[block] = ComponentSet()
            else:
                vars_to_tighten_by_block[block] = ComponentSet()
                for c in block.linking_constraints.values():
                    if c.active:
                        vars_in_con = list(identify_variables(c.body))
                        vars_to_tighten_by_block[block].add(vars_in_con[0])

    for block, vars_to_tighten in vars_to_tighten_by_block.items():
        for v in vars_to_tighten:
            all_vars_to_account_for.discard(v)

    if len(all_vars_to_account_for) != 0:
        raise RuntimeError('There are variables that need tightened that are unaccounted for!')

    return vars_to_tighten_by_block


class OBBTMethod(enum.Enum):
    FULL_SPACE = 1
    DECOMPOSED = 2
    LEAVES = 3


class FilterMethod(enum.Enum):
    NONE = 1
    AGGRESSIVE = 2


class DBTInfo(object):
    """
    Attributes
    ----------
    num_coupling_vars_to_tighten: int
        The total number of coupling variables that need tightened. Note that this includes
        coupling variables that get filtered. If you subtract num_coupling_vars_attempted 
        and num_coupling_vars_filtered from num_coupling_vars_to_tighten, you should get 
        the number of coupling variables that were not tightened due to a time limit.
    num_coupling_vars_attempted: int
        The number of coupling variables for which tightening was attempted.
    num_coupling_vars_successful: int
        The number of coupling variables for which tightening was attempted and the solver 
        terminated optimally.
    num_coupling_vars_filtered: int
        The number of coupling vars that did not need to be tightened (identified by filtering).
    num_vars_to_tighten: int
        The total number of nonlinear and discrete variables that need tightened. Note that 
        this includes variables that get filtered. If you subtract num_vars_attempted and 
        num_vars_filtered from num_vars_to_tighten, you should get the number of nonlinear 
        and discrete variables that were not tightened due to a time limit.
    num_vars_attempted: int
        The number of variables for which tightening was attempted.
    num_vars_successful: int
        The number of variables for which tightening was attempted and the solver 
        terminated optimally.
    num_vars_filtered: int
        The number of vars that did not need to be tightened (identified by filtering).
    """
    def __init__(self):
        self.num_coupling_vars_to_tighten = None
        self.num_coupling_vars_attempted = None
        self.num_coupling_vars_successful = None
        self.num_coupling_vars_filtered = None
        self.num_vars_to_tighten = None
        self.num_vars_attempted = None
        self.num_vars_successful = None
        self.num_vars_filtered = None

    def __str__(self):
        s = f'num_coupling_vars_to_tighten: {self.num_coupling_vars_to_tighten}\n'
        s += f'num_coupling_vars_attempted: {self.num_coupling_vars_attempted}\n'
        s += f'num_coupling_vars_successful: {self.num_coupling_vars_successful}\n'
        s += f'num_coupling_vars_filtered: {self.num_coupling_vars_filtered}\n'
        s += f'num_vars_to_tighten: {self.num_vars_to_tighten}\n'
        s += f'num_vars_attempted: {self.num_vars_attempted}\n'
        s += f'num_vars_successful: {self.num_vars_successful}\n'
        s += f'num_vars_filtered: {self.num_vars_filtered}\n'
        return s


def _update_var_bounds(varlist, new_lower_bounds, new_upper_bounds, feasibility_tol, safety_tol, max_acceptable_bound):
    for ndx, v in enumerate(varlist):
        new_lb = new_lower_bounds[ndx]
        new_ub = new_upper_bounds[ndx]
        orig_lb = v.lb
        orig_ub = v.ub

        if new_lb is None:
            new_lb = -math.inf
        if new_ub is None:
            new_ub = math.inf
        if orig_lb is None:
            orig_lb = -math.inf
        if orig_ub is None:
            orig_ub = math.inf

        rel_lb_safety = safety_tol * abs(new_lb)
        rel_ub_safety = safety_tol * abs(new_ub)
        new_lb -= max(safety_tol, rel_lb_safety)
        new_ub += max(safety_tol, rel_ub_safety)

        if new_lb < -max_acceptable_bound:
            new_lb = -math.inf
        if new_ub > max_acceptable_bound:
            new_ub = math.inf

        if new_lb > new_ub:
            msg = 'variable ub is less than lb; var: {0}; lb: {1}; ub: {2}'.format(str(v), new_lb, new_ub)
            if new_lb > new_ub + feasibility_tol:
                raise ValueError(msg)
            else:
                logger.warning(msg + '; decreasing lb and increasing ub by {0}'.format(feasibility_tol))
                warnings.warn(msg)
                new_lb -= feasibility_tol
                new_ub += feasibility_tol

        if new_lb < orig_lb:
            new_lb = orig_lb
        if new_ub > orig_ub:
            new_ub = orig_ub

        if new_lb > -math.inf:
            v.setlb(new_lb)
        if new_ub < math.inf:
            v.setub(new_ub)


def perform_dbt(relaxation, solver, obbt_method=OBBTMethod.DECOMPOSED,
                filter_method=FilterMethod.AGGRESSIVE, time_limit=math.inf,
                objective_bound=None, with_progress_bar=False, parallel=False,
                vars_to_tighten_by_block=None, feasibility_tol=0,
                safety_tol=0, max_acceptable_bound=math.inf, update_relaxations_between_stages=True):
    """This function performs optimization-based bounds tightening (OBBT) with a decomposition scheme.

    Parameters
    ----------
    relaxation: dbt.decomp.decompose.TreeBlockData
        The relaxation to use for OBBT.
    solver: pyomo solver object
        The solver to use for the OBBT problems.
    obbt_method: OBBTMethod
        An enum member from OBBTMethod. The default is OBBTMethod.DECOMPOSED. If obbt_method
        is OBBTMethod.DECOMPOSED, then only the coupling variables in the linking constraints
        will be tightened with non-leaf blocks in relaxation. The nonlinear and discrete
        variables will only be tightened with the leaf blocks in relaxation. See the
        documentation on TreeBlockData for more details on leaf blocks. If the method is
        OBBTMethod.FULL_SPACE, then all of the nonlinear and discrete variables will
        be tightened with the root block from relaxation. If the method is
        OBBTMethod.LEAVES, then the nonlinear and discrete variables will be tightened
        with the leaf blocks from relaxation (none of the coupling variables will be
        tightened).
    filter_method: FilterMethod
        An enum member from FilterMethod. The default is FilterMethod.AGGRESSIVE. If
        filter_method is FilterMethod.AGGRESSIVE, then aggressive filtering will be
        performed at every stage of OBBT using the
        coramin.domain_reduction.filters.aggressive_filter function which is based on

            Gleixner, Ambros M., et al. "Three enhancements for
            optimization-based bound tightening." Journal of Global
            Optimization 67.4 (2017): 731-757.

        If filter_method is FilterMethod.NONE, then no filtering will be performed.
    time_limit: float
        If the time spent in this function exceeds time_limit, OBBT will be terminated
        early.
    objective_bound: float
        A lower or upper bound on the objective. If this is not None, then a constraint will be added to the
        bounds tightening problems constraining the objective to be less than/greater than objective_bound.
    with_progress_bar: bool
    parallel: bool
        If True, then OBBT will automatically be performed in parallel if mpirun or mpiexec was used;
        If False, then OBBT will not run in parallel even if mpirun or mpiexec was used;
    vars_to_tighten_by_block: dict
        Dictionary mapping TreeBlockData to ComponentSet. This dictionary indicates which variables
        should be tightened with which parts of the TreeBlockData. If None is passed (default=None),
        then, the function collect_vars_to_tighten_by_block is used to get the dict.
    feasibility_tol: float
        If the lower bound for a computed variable is larger than the computed upper bound by more than
        feasibility_tol, then an error is raised. If the computed lower bound is larger than the computed
        upper bound, but by less than feasibility_tol, then the computed lower bound is decreased by
        feasibility tol (but will not be set lower than the original lower bound) and the computed upper
        bound is increased by feasibility_tol (but will not be set higher than the original upper bound).
    safety_tol: float
        Computed lower bounds will be decreased by max(safety_tol, safety_tol*abs(new_lb) and
        computed upper bounds will be increased by max(safety_tol, safety_tol*abs(new_ub) where
        new_lb and new_ub are the bounds computed from OBBT/DBT. The purpose of this is to
        account for numerical error in the solution of the OBBT problems and to avoid cutting
        off valid portions of the feasible region.
    max_acceptable_bound: float
        If the upper bound computed for a variable is larger than max_acceptable_bound, then the 
        computed bound will be rejected. If the lower bound computed for a variable is less than 
        -max_acceptable_bound, then the computed bound will be rejected.
    update_relaxations_between_stages: bool
        This is meant for unit testing only and should not be modified

    Returns
    -------
    dbt_info: DBTInfo

    """
    t0 = time.time()
    
    if not isinstance(relaxation, TreeBlockData):
        raise ValueError('relaxation must be an instance of dbt.decomp.TreeBlockData.')
    if obbt_method not in OBBTMethod:
        raise ValueError('obbt_method must be a member of OBBTMethod.')
    if filter_method not in FilterMethod:
        raise ValueError('filter_method must a member of FilterMethod.')
    if isinstance(solver, PersistentSolver):
        using_persistent_solver = True
    else:
        using_persistent_solver = False

    dbt_info = DBTInfo()
    dbt_info.num_coupling_vars_to_tighten = 0
    dbt_info.num_coupling_vars_attempted = 0
    dbt_info.num_coupling_vars_successful = 0
    dbt_info.num_coupling_vars_filtered = 0
    dbt_info.num_vars_to_tighten = 0
    dbt_info.num_vars_attempted = 0
    dbt_info.num_vars_successful = 0
    dbt_info.num_vars_filtered = 0

    assert obbt_method in OBBTMethod
    if vars_to_tighten_by_block is None:
        if obbt_method == OBBTMethod.DECOMPOSED:
            _method = 'dbt'
        elif obbt_method == OBBTMethod.FULL_SPACE:
            _method = 'full_space'
        else:
            _method = 'leaves'
        vars_to_tighten_by_block = collect_vars_to_tighten_by_block(relaxation, _method)

    var_to_relaxation_map = pe.ComponentMap()
    for r in relaxation_data_objects(relaxation, descend_into=True, active=True):
        for v in r.get_rhs_vars():
            if v not in var_to_relaxation_map:
                var_to_relaxation_map[v] = list()
            var_to_relaxation_map[v].append(r)

    num_stages = relaxation.num_stages()

    for stage in range(num_stages):
        stage_blocks = list(relaxation.stage_blocks(stage))
        for block in stage_blocks:
            vars_to_tighten = vars_to_tighten_by_block[block]
            if obbt_method == OBBTMethod.FULL_SPACE or block.is_leaf():
                dbt_info.num_vars_to_tighten += 2 * len(vars_to_tighten)
            else:
                dbt_info.num_coupling_vars_to_tighten += 2 * len(vars_to_tighten)

    if obbt_method == OBBTMethod.FULL_SPACE:
        all_vars_to_tighten = ComponentSet()
        for block, block_vars_to_tighten in vars_to_tighten_by_block.items():
            all_vars_to_tighten.update(block_vars_to_tighten)
        if filter_method == FilterMethod.AGGRESSIVE:
            logger.debug('starting full space filter')
            res = aggressive_filter(candidate_variables=all_vars_to_tighten, relaxation=relaxation,
                                    solver=solver, tolerance=1e-4, objective_bound=objective_bound)
            full_space_lb_vars, full_space_ub_vars = res
            logger.debug('finished full space filter')
        else:
            full_space_lb_vars = all_vars_to_tighten
            full_space_ub_vars = all_vars_to_tighten
    else:
        full_space_lb_vars = None
        full_space_ub_vars = None
    
    for stage in range(num_stages):
        logger.info(f'Performing DBT on stage {stage+1} of {num_stages}')
        if time.time() - t0 >= time_limit:
            break
        
        stage_blocks = list(relaxation.stage_blocks(stage))
        logger.debug('DBT stage {0} of {1} with {1} blocks'.format(stage, num_stages, len(stage_blocks)))

        for block_ndx, block in enumerate(stage_blocks):
            logger.info(f'performing DBT on block {block_ndx+1} of {len(stage_blocks)} in stage {stage+1}')
            if time.time() - t0 >= time_limit:
                break

            if obbt_method in {OBBTMethod.LEAVES, OBBTMethod.FULL_SPACE} and (not block.is_leaf()):
                continue
            if obbt_method == OBBTMethod.FULL_SPACE:
                block_to_tighten_with = relaxation
                _ub = objective_bound
            else:
                block_to_tighten_with = block
                if stage == 0:
                    _ub = objective_bound
                else:
                    _ub = None
                    
            vars_to_tighten = vars_to_tighten_by_block[block]

            if filter_method == FilterMethod.AGGRESSIVE:
                logger.debug('starting filter')
                if obbt_method == OBBTMethod.FULL_SPACE:
                    lb_vars = ComponentSet([v for v in vars_to_tighten if v in full_space_lb_vars])
                    ub_vars = ComponentSet([v for v in vars_to_tighten if v in full_space_ub_vars])
                else:
                    res = aggressive_filter(candidate_variables=vars_to_tighten, relaxation=block_to_tighten_with,
                                            solver=solver, tolerance=1e-4, objective_bound=_ub)
                    lb_vars, ub_vars = res
                if block.is_leaf():
                    dbt_info.num_vars_filtered += 2*len(vars_to_tighten) - len(lb_vars) - len(ub_vars)
                else:
                    dbt_info.num_coupling_vars_filtered += 2*len(vars_to_tighten) - len(lb_vars) - len(ub_vars)
                logger.debug('done filtering')
            else:
                lb_vars = list(vars_to_tighten)
                ub_vars = list(vars_to_tighten)

            logger.debug(f'performing OBBT (LB) on variables {str([str(i) for i in lb_vars])}')
            res = normal_obbt(block_to_tighten_with, solver=solver, varlist=lb_vars,
                              objective_bound=_ub, with_progress_bar=with_progress_bar,
                              direction='lbs', time_limit=(time_limit - (time.time() - t0)),
                              update_bounds=False, parallel=parallel, collect_obbt_info=True,
                              progress_bar_string=f'DBT LBs Stage {stage+1} of {num_stages} Block {block_ndx+1} of {len(stage_blocks)}')
            lower, unused_upper, obbt_info = res
            if block.is_leaf():
                dbt_info.num_vars_attempted += obbt_info.num_problems_attempted
                dbt_info.num_vars_successful += obbt_info.num_successful_problems
            else:
                dbt_info.num_coupling_vars_attempted += obbt_info.num_problems_attempted
                dbt_info.num_coupling_vars_successful += obbt_info.num_successful_problems

            logger.debug('done tightening lbs')

            logger.debug(f'performing OBBT (UB) on variables {str([str(i) for i in ub_vars])}')
            res = normal_obbt(block_to_tighten_with, solver=solver, varlist=ub_vars,
                              objective_bound=_ub, with_progress_bar=with_progress_bar,
                              direction='ubs', time_limit=(time_limit - (time.time() - t0)),
                              update_bounds=False, parallel=parallel, collect_obbt_info=True,
                              progress_bar_string=f'DBT UBs Stage {stage+1} of {num_stages} Block {block_ndx+1} of {len(stage_blocks)}')

            unused_lower, upper, obbt_info = res
            if block.is_leaf():
                dbt_info.num_vars_attempted += obbt_info.num_problems_attempted
                dbt_info.num_vars_successful += obbt_info.num_successful_problems
            else:
                dbt_info.num_coupling_vars_attempted += obbt_info.num_problems_attempted
                dbt_info.num_coupling_vars_successful += obbt_info.num_successful_problems

            _update_var_bounds(varlist=lb_vars, new_lower_bounds=lower,
                               new_upper_bounds=unused_upper, feasibility_tol=feasibility_tol,
                               safety_tol=safety_tol, max_acceptable_bound=max_acceptable_bound)

            _update_var_bounds(varlist=ub_vars, new_lower_bounds=unused_lower,
                               new_upper_bounds=upper, feasibility_tol=feasibility_tol,
                               safety_tol=safety_tol, max_acceptable_bound=max_acceptable_bound)

            if update_relaxations_between_stages:
                # this is needed to ensure consistency for parallel computing; this accounts
                # for side effects from the OBBT problems; in particular, if the solver ever
                # rebuilds relaxations, then the processes could become out of sync without
                # this code
                all_tightened_vars = ComponentSet(lb_vars)
                all_tightened_vars.update(ub_vars)
                for v in all_tightened_vars:
                    if v in var_to_relaxation_map:
                        for r in var_to_relaxation_map[v]:
                            r.rebuild()

            logger.debug('done tightening ubs')

            if not block.is_leaf():
                for c in block.linking_constraints.values():
                    fbbt(c)

    return dbt_info


def push_integers(block):
    """
    Parameters
    ----------
    block: pyomo.core.base.block._BlockData
        The block for which integer variables should be relaxed.

    Returns
    -------
    relaxed_binary_vars: ComponentSet of pyomo.core.base.var._GeneralVarData
    relaxed_integer_vars: ComponentSet or pyomo.core.base.var._GeneralVarData
    """
    relaxed_binary_vars = ComponentSet()
    relaxed_integer_vars = ComponentSet()
    for v in block.component_data_objects(pe.Var, descend_into=True, sort=True):
        if v.fixed:
            continue
        if v.is_binary():
            relaxed_binary_vars.add(v)
            orig_lb = v.lb
            orig_ub = v.ub
            v.domain = pe.Reals
            v.setlb(orig_lb)
            v.setub(orig_ub)
        elif v.is_integer():
            relaxed_integer_vars.add(v)
            v.domain = pe.Reals

    return relaxed_binary_vars, relaxed_integer_vars


def pop_integers(relaxed_binary_vars, relaxed_integer_vars):
    for v in relaxed_binary_vars:
        v.domain = pe.Binary
    for v in relaxed_integer_vars:
        v.domain = pe.Integers


def perform_dbt_with_integers_relaxed(relaxation, solver, obbt_method=OBBTMethod.DECOMPOSED,
                                      filter_method=FilterMethod.AGGRESSIVE, time_limit=math.inf,
                                      objective_bound=None, with_progress_bar=False, parallel=False,
                                      vars_to_tighten_by_block=None, feasibility_tol=0,
                                      integer_tol=1e-2, safety_tol=0, max_acceptable_bound=math.inf):
    """
    This function performs optimization-based bounds tightening (OBBT) with a decomposition scheme.
    However, all OBBT problems are solved with the binary and integer variables relaxed.

    Parameters
    ----------
    relaxation: dbt.decomp.decompose.TreeBlockData
        The relaxation to use for OBBT.
    solver: pyomo solver object
        The solver to use for the OBBT problems.
    obbt_method: OBBTMethod
        An enum member from OBBTMethod. The default is OBBTMethod.DECOMPOSED. If obbt_method
        is OBBTMethod.DECOMPOSED, then only the coupling variables in the linking constraints
        will be tightened with non-leaf blocks in relaxation. The nonlinear and discrete
        variables will only be tightened with the leaf blocks in relaxation. See the
        documentation on TreeBlockData for more details on leaf blocks. If the method is
        OBBTMethod.FULL_SPACE, then all of the nonlinear and discrete variables will
        be tightened with the root block from relaxation. If the method is
        OBBTMethod.LEAVES, then the nonlinear and discrete variables will be tightened
        with the leaf blocks from relaxation (none of the coupling variables will be
        tightened).
    filter_method: FilterMethod
        An enum member from FilterMethod. The default is FilterMethod.AGGRESSIVE. If
        filter_method is FilterMethod.AGGRESSIVE, then aggressive filtering will be
        performed at every stage of OBBT using the
        coramin.domain_reduction.filters.aggressive_filter function which is based on

            Gleixner, Ambros M., et al. "Three enhancements for
            optimization-based bound tightening." Journal of Global
            Optimization 67.4 (2017): 731-757.

        If filter_method is FilterMethod.NONE, then no filtering will be performed.
    time_limit: float
        If the time spent in this function exceeds time_limit, OBBT will be terminated
        early.
    objective_bound: float
        A lower or upper bound on the objective. If this is not None, then a constraint will be added to the
        bounds tightening problems constraining the objective to be less than/greater than objective_bound.
    with_progress_bar: bool
    parallel: bool
        If True, then OBBT will automatically be performed in parallel if mpirun or mpiexec was used;
        If False, then OBBT will not run in parallel even if mpirun or mpiexec was used;
    vars_to_tighten_by_block: dict
        Dictionary mapping TreeBlockData to ComponentSet. This dictionary indicates which variables
        should be tightened with which parts of the TreeBlockData. If None is passed (default=None),
        then, the function collect_vars_to_tighten_by_block is used to get the dict.
    feasibility_tol: float
        If the lower bound computed for a variable is larger than the computed upper bound by more than
        feasibility_tol, then an error is raised. If the computed lower bound is larger than the computed
        upper bound, but by less than feasibility_tol, then the computed lower bound is decreased by
        feasibility tol (but will not be set lower than the original lower bound) and the computed upper
        bound is increased by feasibility_tol (but will not be set higher than the original upper bound).
    integer_tol: float
        If the lower bound computed for an integer variable is greater than the largest integer less than
        the computed lower bound by more than integer_tol, then the lower bound is increased to the smallest
        integer greater than the computed lower bound. Similar logic holds for the upper bound.
    safety_tol: float
        Computed lower bounds will be decreased by max(safety_tol, safety_tol*abs(new_lb) and
        computed upper bounds will be increased by max(safety_tol, safety_tol*abs(new_ub) where
        new_lb and new_ub are the bounds computed from OBBT/DBT. The purpose of this is to
        account for numerical error in the solution of the OBBT problems and to avoid cutting
        off valid portions of the feasible region.
    max_acceptable_bound: float
        If the upper bound computed for a variable is larger than max_acceptable_bound, then the 
        computed bound will be rejected. If the lower bound computed for a variable is less than 
        -max_acceptable_bound, then the computed bound will be rejected.

    Returns
    -------
    dbt_info: DBTInfo
    """
    assert obbt_method in OBBTMethod
    if vars_to_tighten_by_block is None:
        if obbt_method == OBBTMethod.DECOMPOSED:
            _method = 'dbt'
        elif obbt_method == OBBTMethod.FULL_SPACE:
            _method = 'full_space'
        else:
            _method = 'leaves'
        vars_to_tighten_by_block = collect_vars_to_tighten_by_block(relaxation, _method)

    relaxed_binary_vars, relaxed_integer_vars = push_integers(relaxation)

    dbt_info = perform_dbt(relaxation=relaxation,
                           solver=solver,
                           obbt_method=obbt_method,
                           filter_method=filter_method,
                           time_limit=time_limit,
                           objective_bound=objective_bound,
                           with_progress_bar=with_progress_bar,
                           parallel=parallel,
                           vars_to_tighten_by_block=vars_to_tighten_by_block,
                           feasibility_tol=feasibility_tol,
                           safety_tol=safety_tol,
                           max_acceptable_bound=max_acceptable_bound)

    pop_integers(relaxed_binary_vars, relaxed_integer_vars)

    for v in (list(relaxed_binary_vars) + list(relaxed_integer_vars)):
        lb = v.lb
        ub = v.ub
        if lb is None:
            lb = -math.inf
        if ub is None:
            ub = math.inf
        if lb > -math.inf:
            lb = max(math.floor(lb), math.ceil(lb - integer_tol))
        if ub < math.inf:
            ub = min(math.ceil(ub), math.floor(ub + integer_tol))
        if lb > -math.inf:
            v.setlb(lb)
        if ub < math.inf:
            v.setub(ub)

    return dbt_info
