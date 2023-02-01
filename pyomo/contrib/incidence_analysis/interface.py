#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import enum
import textwrap
from pyomo.core.base.block import Block
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.base.reference import Reference
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.current import EqualityExpression
from pyomo.util.subsystems import create_subsystem_block
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.dependencies import scipy_available, attempt_import
from pyomo.common.dependencies import networkx as nx
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.connected import (
    get_independent_submatrices,
)
from pyomo.contrib.incidence_analysis.triangularize import (
    get_scc_of_projection,
    block_triangularize,
    get_diagonal_blocks,
    get_blocks_from_maps,
)
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import (
    dulmage_mendelsohn,
    RowPartition,
    ColPartition,
)

if scipy_available:
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
    import scipy as sp

plotly, plotly_available = attempt_import("plotly")
if plotly_available:
    go = plotly.graph_objects


def _check_unindexed(complist):
    for comp in complist:
        if comp.is_indexed():
            raise ValueError(
                "Variables and constraints must be unindexed "
                "ComponentData objects. Got %s, which is indexed." % comp.name
            )


def get_incidence_graph(variables, constraints, include_fixed=True):
    return get_bipartite_incidence_graph(
        variables, constraints, include_fixed=include_fixed
    )


def get_bipartite_incidence_graph(variables, constraints, include_fixed=True):
    """Return the bipartite incidence graph of Pyomo variables and constraints.

    Each node in the returned graph is an integer. The convention is that,
    for a graph with N variables and M constraints, nodes 0 through M-1
    correspond to constraints and nodes M through M+N-1 correspond to variables.
    Nodes correspond to variables and constraints in the provided orders.
    For consistency with NetworkX's "convention", constraint nodes are tagged
    with `bipartite=0` while variable nodes are tagged with `bipartite=1`,
    although these attributes are not used.

    Arguments:
    ----------
    variables: List of Pyomo VarData objects
        Variables that will appear in incidence graph
    constraints: List of Pyomo ConstraintData objects
        Constraints that will appear in incidence graph
    include_fixed: Bool
        Flag for whether fixed variable should be included in the incidence

    Returns:
    --------
    NetworkX Graph

    """
    _check_unindexed(variables + constraints)
    N = len(variables)
    M = len(constraints)
    graph = nx.Graph()
    graph.add_nodes_from(range(M), bipartite=0)
    graph.add_nodes_from(range(M, M + N), bipartite=1)
    var_node_map = ComponentMap((v, M + i) for i, v in enumerate(variables))
    for i, con in enumerate(constraints):
        for var in identify_variables(con.expr, include_fixed=include_fixed):
            if var in var_node_map:
                graph.add_edge(i, var_node_map[var])
    return graph


def extract_bipartite_subgraph(graph, nodes0, nodes1):
    """Return the bipartite subgraph of a graph.

    Two lists of nodes to project onto must be provided. These will correspond
    to the "bipartite sets" in the subgraph. If the two sets provided have
    M and N nodes, the subgraph will have nodes 0 through M+N, with the first
    M corresponding to the first set provided and the last M corresponding
    to the second set.

    Parameters
    ----------
    graph: NetworkX Graph
        The graph from which a subgraph is extracted
    nodes0: list
        A list of nodes in the original graph that will form the first
        bipartite set of the projected graph (and have ``bipartite=0``
    nodes1: list
        A list of nodes in the original graph that will form the second
        bipartite set of the projected graph (and have ``bipartite=1``

    Returns
    -------
    subgraph: NetworkX Graph
        Graph containing integer nodes corresponding to positions in the
        provided lists, with edges where corresponding nodes are adjacent
        in the original graph.

    """
    subgraph = nx.Graph()
    sub_M = len(nodes0)
    sub_N = len(nodes1)
    subgraph.add_nodes_from(range(sub_M), bipartite=0)
    subgraph.add_nodes_from(range(sub_M, sub_M + sub_N), bipartite=1)

    old_new_map = {}
    for i, node in enumerate(nodes0 + nodes1):
        if node in old_new_map:
            raise RuntimeError("Node %s provided more than once.")
        old_new_map[node] = i

    for (node1, node2) in graph.edges():
        if node1 in old_new_map and node2 in old_new_map:
            new_node_1 = old_new_map[node1]
            new_node_2 = old_new_map[node2]
            if (
                subgraph.nodes[new_node_1]["bipartite"]
                == subgraph.nodes[new_node_2]["bipartite"]
            ):
                raise RuntimeError(
                    "Subgraph is not bipartite. Found an edge between nodes"
                    " %s and %s (in the original graph)." % (node1, node2)
                )
            subgraph.add_edge(new_node_1, new_node_2)
    return subgraph


def _generate_variables_in_constraints(constraints, include_fixed=False):
    known_vars = ComponentSet()
    for con in constraints:
        for var in identify_variables(con.expr, include_fixed=include_fixed):
            if var not in known_vars:
                known_vars.add(var)
                yield var


def get_structural_incidence_matrix(variables, constraints, include_fixed=True):
    """
    This function gets the incidence matrix of Pyomo constraints and variables.

    Arguments
    ---------
    variables: List of Pyomo VarData objects
    constraints: List of Pyomo ConstraintData objects
    include_fixed: Bool
        Flag for whether fixed variables should be included in the matrix
        nonzeros

    Returns
    -------
    A scipy.sparse coo matrix. Rows are indices into the user-provided list of
    constraints, columns are indices into the user-provided list of variables.
    Entries are 1.0.

    """
    _check_unindexed(variables + constraints)
    N, M = len(variables), len(constraints)
    var_idx_map = ComponentMap((v, i) for i, v in enumerate(variables))
    rows = []
    cols = []
    for i, con in enumerate(constraints):
        cols.extend(
            var_idx_map[v]
            for v in identify_variables(con.expr, include_fixed=include_fixed)
            if v in var_idx_map
        )
        rows.extend([i] * (len(cols) - len(rows)))
    assert len(rows) == len(cols)
    data = [1.0] * len(rows)
    matrix = sp.sparse.coo_matrix((data, (rows, cols)), shape=(M, N))
    return matrix


def get_numeric_incidence_matrix(variables, constraints):
    """
    This function gets the numeric incidence matrix (Jacobian) of Pyomo
    constraints with respect to variables.
    """
    # NOTE: There are several ways to get a numeric incidence matrix
    # from a Pyomo model. Here we get the numeric incidence matrix by
    # creating a temporary block and using the PyNumero ASL interface.
    comps = list(variables) + list(constraints)
    _check_unindexed(comps)
    block = create_subsystem_block(constraints, variables)
    block._obj = Objective(expr=0)
    nlp = PyomoNLP(block)
    return nlp.extract_submatrix_jacobian(variables, constraints)


class IncidenceGraphInterface(object):
    """
    The purpose of this class is to allow the user to easily
    analyze graphs of variables and contraints in a Pyomo
    model without constructing multiple PyomoNLPs.
    """

    def __init__(
        self,
        model=None,
        active=True,
        include_fixed=False,
        include_inequality=True,
    ):
        """ """
        # If the user gives us a model or an NLP, we assume they want us
        # to cache the incidence graph for fast analysis later on.
        # WARNING: This cache will become invalid if the user alters their
        # model.
        if model is None:
            self.incidence_graph = None
        elif isinstance(model, PyomoNLP):
            if not active:
                raise ValueError(
                    "Cannot get the Jacobian of inactive constraints from the "
                    "nl interface (PyomoNLP).\nPlease set the `active` flag "
                    "to True."
                )
            if include_fixed:
                raise ValueError(
                    "Cannot get the Jacobian with respect to fixed variables "
                    "from the nl interface (PyomoNLP).\nPlease set the "
                    "`include_fixed` flag to False."
                )
            nlp = model
            self.variables = nlp.get_pyomo_variables()
            self.constraints = [
                con for con in nlp.get_pyomo_constraints()
                if include_inequality or isinstance(con.expr, EqualityExpression)
            ]
            self.var_index_map = ComponentMap(
                (var, idx) for idx, var in enumerate(self.variables)
            )
            self.con_index_map = ComponentMap(
                (con, idx) for idx, con in enumerate(self.constraints)
            )
            if include_inequality:
                incidence_matrix = nlp.evaluate_jacobian()
            else:
                incidence_matrix = nlp.evaluate_jacobian_eq()
            nxb = nx.algorithms.bipartite
            self.incidence_graph = nxb.from_biadjacency_matrix(incidence_matrix)
        elif isinstance(model, Block):
            self.constraints = [
                con for con in model.component_data_objects(
                    Constraint, active=active
                )
                if include_inequality or isinstance(con.expr, EqualityExpression)
            ]
            self.variables = list(
                _generate_variables_in_constraints(
                    self.constraints, include_fixed=include_fixed
                )
            )
            self.var_index_map = ComponentMap(
                (var, i) for i, var in enumerate(self.variables)
            )
            self.con_index_map = ComponentMap(
                (con, i) for i, con in enumerate(self.constraints)
            )
            self.incidence_graph = get_bipartite_incidence_graph(
                self.variables,
                self.constraints,
                # TODO: include_fixed=include_fixed?
            )
        else:
            raise TypeError(
                "Unsupported type for incidence matrix. Expected "
                "%s or %s but got %s." % (PyomoNLP, Block, type(model))
            )

        self.row_block_map = None
        self.col_block_map = None

    def _validate_input(self, variables, constraints):
        if variables is None:
            if self.incidence_graph is None:
                raise ValueError("Neither variables nor a model have been provided.")
            else:
                variables = self.variables
        if constraints is None:
            if self.incidence_graph is None:
                raise ValueError("Neither constraints nor a model have been provided.")
            else:
                constraints = self.constraints

        _check_unindexed(variables + constraints)
        return variables, constraints

    def _extract_submatrix(self, variables, constraints):
        # Assumes variables and constraints are valid
        if self.incidence_graph is None:
            return get_structural_incidence_matrix(
                variables,
                constraints,
                include_fixed=False,
            )
        else:
            N = len(variables)
            M = len(constraints)
            old_new_var_indices = dict(
                (self.var_index_map[v], i) for i, v in enumerate(variables)
            )
            old_new_con_indices = dict(
                (self.con_index_map[c], i) for i, c in enumerate(constraints)
            )
            # FIXME: This will fail if I don't have an incidence matrix
            # cached.
            coo = self.incidence_matrix
            new_row = []
            new_col = []
            new_data = []
            for r, c, e in zip(coo.row, coo.col, coo.data):
                if r in old_new_con_indices and c in old_new_var_indices:
                    new_row.append(old_new_con_indices[r])
                    new_col.append(old_new_var_indices[c])
                    new_data.append(e)
            return sp.sparse.coo_matrix(
                (new_data, (new_row, new_col)),
                shape=(M, N),
            )

    def _extract_subgraph(self, variables, constraints):
        if self.incidence_graph is None:
            return get_bipartite_incidence_graph(
                # Does include_fixed matter here if I'm providing the variables?
                variables, constraints, include_fixed=False
            )
        else:
            constraint_nodes = [self.con_index_map[con] for con in constraints]

            # Note that this is the number of constraints in the original graph,
            # not the subgraph.
            M = len(self.constraints)
            variable_nodes = [M + self.var_index_map[var] for var in variables]
            subgraph = extract_bipartite_subgraph(
                self.incidence_graph, constraint_nodes, variable_nodes
            )
            return subgraph

    @property
    def incidence_matrix(self):
        if self.incidence_graph is None:
            return None
        else:
            M = len(self.constraints)
            N = len(self.variables)
            row = []
            col = []
            data = []
            # Here we assume that the incidence graph is bipartite with nodes
            # 0 through M-1 forming one of the bipartite sets.
            for i in range(M):
                assert self.incidence_graph.nodes[i]["bipartite"] == 0
                for j in self.incidence_graph[i]:
                    assert self.incidence_graph.nodes[j]["bipartite"] == 1
                    row.append(i)
                    col.append(j-M)
                    data.append(1.0)
            return sp.sparse.coo_matrix(
                (data, (row, col)),
                shape=(M, N),
            )

    def get_adjacent_to(self, component):
        """Return a list of components adjacent to the provided component
        in the cached bipartite incidence graph of variables and constraints

        Parameters
        ----------
        component: ComponentData
            The variable or constraint data object whose adjacent components
            are returned

        Returns
        -------
        list of ComponentData
            List of constraint or variable data objects adjacent to the
            provided component

        """
        if self.incidence_graph is None:
            raise RuntimeError(
                "Cannot get components adjacent to %s if an incidence graph"
                " is not cached." % component
            )
        _check_unindexed([component])
        M = len(self.constraints)
        N = len(self.variables)
        if component in self.var_index_map:
            vnode = M + self.var_index_map[component]
            adj = self.incidence_graph[vnode]
            adj_comps = [self.constraints[i] for i in adj]
        elif component in self.con_index_map:
            cnode = self.con_index_map[component]
            adj = self.incidence_graph[cnode]
            adj_comps = [self.variables[j-M] for j in adj]
        else:
            raise RuntimeError(
                "Cannot find component %s in the cached incidence graph."
                % component
            )
        return adj_comps

    def maximum_matching(self, variables=None, constraints=None):
        """
        Returns a maximal matching between the constraints and variables,
        in terms of a map from constraints to variables.
        """
        variables, constraints = self._validate_input(variables, constraints)
        graph = self._extract_subgraph(variables, constraints)
        con_nodes = list(range(len(constraints)))
        matching = maximum_matching(graph, top_nodes=con_nodes)
        # Matching maps constraint nodes to variable nodes. Here we need to
        # know the convention according to which the graph was constructed.
        M = len(constraints)
        return ComponentMap(
            (constraints[i], variables[j-M]) for i, j in matching.items()
        )

    def get_connected_components(self, variables=None, constraints=None):
        """
        Return lists of lists of variables and constraints that appear in
        different connected components of the bipartite graph of variables
        and constraints.
        """
        variables, constraints = self._validate_input(variables, constraints)
        graph = self._extract_subgraph(variables, constraints)
        nxc = nx.algorithms.components
        M = len(constraints)
        N = len(variables)
        connected_components = list(nxc.connected_components(graph))

        con_blocks = [
            sorted([i for i in comp if i < M]) for comp in connected_components
        ]
        con_blocks = [[constraints[i] for i in block] for block in con_blocks]
        var_blocks = [
            sorted([j for j in comp if j >= M]) for comp in connected_components
        ]
        var_blocks = [[variables[i-M] for i in block] for block in var_blocks]

        return var_blocks, con_blocks

    def block_triangularize(self, variables=None, constraints=None):
        """
        Returns two ComponentMaps. A map from variables to their blocks
        in a block triangularization of the incidence matrix, and a
        map from constraints to their blocks in a block triangularization
        of the incidence matrix.
        """
        variables, constraints = self._validate_input(variables, constraints)
        graph = self._extract_subgraph(variables, constraints)

        M = len(constraints)
        con_nodes = list(range(M))
        sccs = get_scc_of_projection(graph, con_nodes)
        row_idx_map = {r: idx for idx, scc in enumerate(sccs) for r, _ in scc}
        col_idx_map = {c-M: idx for idx, scc in enumerate(sccs) for _, c in scc}
        # Cache maps in case we want to get diagonal blocks quickly in the
        # future.
        row_block_map = row_idx_map
        col_block_map = {j-M: idx for j, idx in col_idx_map.items()}
        self.row_block_map = row_block_map
        self.col_block_map = col_block_map
        con_block_map = ComponentMap(
            (constraints[i], idx) for i, idx in row_block_map.items()
        )
        var_block_map = ComponentMap(
            (variables[j], idx) for j, idx in col_block_map.items()
        )
        # Switch the order of the maps here to match the method call.
        # Hopefully this does not get too confusing...
        return var_block_map, con_block_map

    # TODO: Update this method with a new name and deprecate old name.
    def get_diagonal_blocks(self, variables=None, constraints=None):
        """
        Returns the diagonal blocks in a block triangularization of the
        incidence matrix of the provided constraints with respect to the
        provided variables.

        Returns
        -------
        tuple of lists
        The first list contains lists that partition the variables,
        the second lists contains lists that partition the constraints.

        """
        variables, constraints = self._validate_input(variables, constraints)
        matrix = self._extract_submatrix(variables, constraints)

        # TODO: Again, this functionality does not really make sense for
        # general bipartite graphs...
        if self.row_block_map is None or self.col_block_map is None:
            block_rows, block_cols = get_diagonal_blocks(matrix)
        else:
            block_rows, block_cols = get_blocks_from_maps(
                self.row_block_map, self.col_block_map
            )
        block_cons = [[constraints[i] for i in block] for block in block_rows]
        block_vars = [[variables[i] for i in block] for block in block_cols]
        return block_vars, block_cons

    def dulmage_mendelsohn(self, variables=None, constraints=None):
        """
        Returns the Dulmage-Mendelsohn partition of the incidence graph
        of the provided variables and constraints.

        Returns:
        --------
        ColPartition namedtuple and RowPartition namedtuple.
        The ColPartition is returned first to match the order of variables
        and constraints in the method arguments.
        These partition variables (columns) and constraints (rows)
        into overconstrained, underconstrained, unmatched, and square.

        """
        variables, constraints = self._validate_input(variables, constraints)
        graph = self._extract_subgraph(variables, constraints)
        M = len(constraints)
        top_nodes = list(range(M))
        row_partition, col_partition = dulmage_mendelsohn(
            graph, top_nodes=top_nodes
        )
        con_partition = RowPartition(
            *[[constraints[i] for i in subset] for subset in row_partition]
        )
        var_partition = ColPartition(
            *[[variables[i-M] for i in subset] for subset in col_partition]
        )
        # Switch the order of the maps here to match the method call.
        # Hopefully this does not get too confusing...
        return var_partition, con_partition

    def remove_nodes(self, nodes, constraints=None):
        """
        Removes the specified variables and constraints (columns and
        rows) from the cached incidence matrix. This is a "projection"
        of the variable and constraint vectors, rather than something
        like a vertex elimination.
        For the puropse of this method, there is no need to distinguish
        between variables and constraints. However, we provide the
        "constraints" argument so a call signature similar to other methods
        in this class is still valid.

        Arguments:
        ----------
        nodes: List
            VarData or ConData objects whose columns or rows will be
            removed from the incidence matrix.
        constraints: List
            VarData or ConData objects whose columns or rows will be
            removed from the incidence matrix.

        """
        if constraints is None:
            constraints = []
        if self.incidence_graph is None:
            raise RuntimeError(
                "Attempting to remove variables and constraints from cached "
                "incidence matrix,\nbut no incidence matrix has been cached."
            )
        to_exclude = ComponentSet(nodes)
        to_exclude.update(constraints)
        vars_to_include = [v for v in self.variables if v not in to_exclude]
        cons_to_include = [c for c in self.constraints if c not in to_exclude]
        incidence_graph = self._extract_subgraph(vars_to_include, cons_to_include)
        # update attributes
        self.variables = vars_to_include
        self.constraints = cons_to_include
        self.incidence_graph = incidence_graph
        self.var_index_map = ComponentMap(
            (var, i) for i, var in enumerate(self.variables)
        )
        self.con_index_map = ComponentMap(
            (con, i) for i, con in enumerate(self.constraints)
        )
        self.row_block_map = None
        self.col_block_map = None

    def plot(self, variables=None, constraints=None, title=None, show=True):
        """Plot the bipartite incidence graph of variables and constraints
        """
        variables, constraints = self._validate_input(variables, constraints)
        graph = self._extract_subgraph(variables, constraints)
        M = len(constraints)

        left_nodes = list(range(M))
        pos_dict = nx.drawing.bipartite_layout(graph, nodes=left_nodes)

        edge_x = []
        edge_y = []
        for start_node, end_node in graph.edges():
            x0, y0 = pos_dict[start_node]
            x1, y1 = pos_dict[end_node]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
        )

        node_x = []
        node_y = []
        node_text = []
        node_color = []
        for node in graph.nodes():
            x, y = pos_dict[node]
            node_x.append(x)
            node_y.append(y)
            if node < M:
                # According to convention, we are a constraint node
                c = constraints[node]
                node_color.append('red')
                body_text = '<br>'.join(
                    textwrap.wrap(
                        str(c.body), width=120, subsequent_indent="    "
                    )
                )
                node_text.append(
                    f'{str(c)}<br>lb: {str(c.lower)}<br>body: {body_text}<br>'
                    f'ub: {str(c.upper)}<br>active: {str(c.active)}'
                )
            else:
                # According to convention, we are a variable node
                v = variables[node-M]
                node_color.append('blue')
                node_text.append(
                    f'{str(v)}<br>lb: {str(v.lb)}<br>ub: {str(v.ub)}<br>'
                    f'value: {str(v.value)}<br>domain: {str(v.domain)}<br>'
                    f'fixed: {str(v.is_fixed())}'
                )
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(color=node_color, size=10),
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        if title is not None:
            fig.update_layout(title=dict(text=title))
        if show:
            fig.show()
