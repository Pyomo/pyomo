#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""Utility functions and a utility class for interfacing Pyomo components with
useful graph algorithms.

"""

import enum
import textwrap
from pyomo.core.base.block import BlockData
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.expr import EqualityExpression
from pyomo.util.subsystems import create_subsystem_block
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.dependencies import (
    attempt_import,
    networkx as nx,
    scipy as sp,
    plotly,
)
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.contrib.incidence_analysis.config import get_config_from_kwds
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.connected import get_independent_submatrices
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
from pyomo.contrib.incidence_analysis.incidence import get_incident_variables
from pyomo.contrib.pynumero.asl import AmplInterface

pyomo_nlp, pyomo_nlp_available = attempt_import(
    "pyomo.contrib.pynumero.interfaces.pyomo_nlp"
)
asl_available = pyomo_nlp_available & AmplInterface.available()


def _check_unindexed(complist):
    for comp in complist:
        if comp.is_indexed():
            raise RuntimeError(
                "Variables and constraints must be unindexed "
                "ComponentData objects. Got %s, which is indexed." % comp.name
            )


def get_incidence_graph(variables, constraints, **kwds):
    config = get_config_from_kwds(**kwds)
    return get_bipartite_incidence_graph(variables, constraints, **config)


def get_bipartite_incidence_graph(variables, constraints, **kwds):
    """Return the bipartite incidence graph of Pyomo variables and constraints.

    Each node in the returned graph is an integer. The convention is that,
    for a graph with N variables and M constraints, nodes 0 through M-1
    correspond to constraints and nodes M through M+N-1 correspond to variables.
    Nodes correspond to variables and constraints in the provided orders.
    For consistency with NetworkX's "convention", constraint nodes are tagged
    with ``bipartite=0`` while variable nodes are tagged with ``bipartite=1``,
    although these attributes are not used.

    Parameters
    ---------
    variables: List of Pyomo VarData objects
        Variables that will appear in incidence graph
    constraints: List of Pyomo ConstraintData objects
        Constraints that will appear in incidence graph
    include_fixed: Bool
        Flag for whether fixed variable should be included in the incidence

    Returns
    -------
    ``networkx.Graph``

    """
    # Note that this ConfigDict contains the visitor that we will re-use
    # when constructing constraints.
    config = get_config_from_kwds(**kwds)
    _check_unindexed(variables + constraints)
    N = len(variables)
    M = len(constraints)
    graph = nx.Graph()
    graph.add_nodes_from(range(M), bipartite=0)
    graph.add_nodes_from(range(M, M + N), bipartite=1)
    var_node_map = ComponentMap((v, M + i) for i, v in enumerate(variables))
    for i, con in enumerate(constraints):
        for var in get_incident_variables(con.body, **config):
            if var in var_node_map:
                graph.add_edge(i, var_node_map[var])
    return graph


def extract_bipartite_subgraph(graph, nodes0, nodes1):
    """Return the bipartite subgraph of a graph.

    Two lists of nodes to project onto must be provided. These will correspond
    to the "bipartite sets" in the subgraph. If the two sets provided have
    M and N nodes, the subgraph will have nodes 0 through M+N-1, with the first
    M corresponding to the first set provided and the last N corresponding
    to the second set.

    Parameters
    ----------
    graph: NetworkX Graph
        The graph from which a subgraph is extracted
    nodes0: list
        A list of nodes in the original graph that will form the first
        bipartite set of the projected graph (and have ``bipartite=0``)
    nodes1: list
        A list of nodes in the original graph that will form the second
        bipartite set of the projected graph (and have ``bipartite=1``)

    Returns
    -------
    subgraph: ``networkx.Graph``
        Graph containing integer nodes corresponding to positions in the
        provided lists, with edges where corresponding nodes are adjacent
        in the original graph.

    """
    subgraph = graph.subgraph(nodes0 + nodes1)
    # TODO: Any error checking that nodes are valid bipartition?
    for node in nodes0:
        bipartite = graph.nodes[node]["bipartite"]
        if bipartite != 0:
            raise RuntimeError(
                "Invalid bipartite sets. Node {node} in set 0 has"
                " bipartite={bipartite}"
            )
    for node in nodes1:
        bipartite = graph.nodes[node]["bipartite"]
        if bipartite != 1:
            raise RuntimeError(
                "Invalid bipartite sets. Node {node} in set 1 has"
                " bipartite={bipartite}"
            )
    old_new_map = {}
    for i, node in enumerate(nodes0 + nodes1):
        if node in old_new_map:
            raise RuntimeError("Node %s provided more than once.")
        old_new_map[node] = i
    relabeled_subgraph = nx.relabel_nodes(subgraph, old_new_map)
    return relabeled_subgraph


def _generate_variables_in_constraints(constraints, **kwds):
    # Note: We construct a visitor here
    config = get_config_from_kwds(**kwds)
    known_vars = ComponentSet()
    for con in constraints:
        for var in get_incident_variables(con.body, **config):
            if var not in known_vars:
                known_vars.add(var)
                yield var


def get_structural_incidence_matrix(variables, constraints, **kwds):
    """Return the incidence matrix of Pyomo constraints and variables

    Parameters
    ---------
    variables: List of Pyomo VarData objects
    constraints: List of Pyomo ConstraintData objects
    include_fixed: Bool
        Flag for whether fixed variables should be included in the matrix
        nonzeros

    Returns
    -------
    ``scipy.sparse.coo_matrix``
        COO matrix. Rows are indices into the user-provided list of constraints,
        columns are indices into the user-provided list of variables.
        Entries are 1.0.

    """
    config = get_config_from_kwds(**kwds)
    _check_unindexed(variables + constraints)
    N, M = len(variables), len(constraints)
    var_idx_map = ComponentMap((v, i) for i, v in enumerate(variables))
    rows = []
    cols = []
    for i, con in enumerate(constraints):
        cols.extend(
            var_idx_map[v]
            for v in get_incident_variables(con.body, **config)
            if v in var_idx_map
        )
        rows.extend([i] * (len(cols) - len(rows)))
    assert len(rows) == len(cols)
    data = [1.0] * len(rows)
    matrix = sp.sparse.coo_matrix((data, (rows, cols)), shape=(M, N))
    return matrix


def get_numeric_incidence_matrix(variables, constraints):
    """Return the "numeric incidence matrix" (Jacobian) of Pyomo variables
    and constraints.

    Each matrix value is the derivative of a constraint body with respect
    to a variable. Rows correspond to constraints and columns correspond to
    variables. Entries are included even if the value of the derivative is
    zero.
    Only active constraints and unfixed variables that participate in these
    constraints are included.

    Parameters
    ---------
    variables: List of Pyomo VarData objects
    constraints: List of Pyomo ConstraintData objects

    Returns
    -------
    ``scipy.sparse.coo_matrix``
        COO matrix. Rows are indices into the user-provided list of constraints,
        columns are indices into the user-provided list of variables.

    """
    # NOTE: There are several ways to get a numeric incidence matrix
    # from a Pyomo model. Here we get the numeric incidence matrix by
    # creating a temporary block and using the PyNumero ASL interface.
    comps = list(variables) + list(constraints)
    _check_unindexed(comps)
    block = create_subsystem_block(constraints, variables)
    block._obj = Objective(expr=0)
    nlp = pyomo_nlp.PyomoNLP(block)
    return nlp.extract_submatrix_jacobian(variables, constraints)


class IncidenceGraphInterface(object):
    """An interface for applying graph algorithms to Pyomo variables and
    constraints

    Parameters
    ----------
    model: Pyomo BlockData or PyNumero PyomoNLP, default ``None``
        An object from which an incidence graph will be constructed.
    active: Bool, default ``True``
        Whether only active constraints should be included in the incidence
        graph. Cannot be set to ``False`` if the ``model`` is provided as
        a PyomoNLP.
    include_fixed: Bool, default ``False``
        Whether to include fixed variables in the incidence graph. Cannot
        be set to ``False`` if ``model`` is a PyomoNLP.
    include_inequality: Bool, default ``True``
        Whether to include inequality constraints (those whose expressions
        are not instances of ``EqualityExpression``) in the incidence graph.
        If a PyomoNLP is provided, setting to ``False`` uses the
        ``evaluate_jacobian_eq`` method instead of ``evaluate_jacobian``
        rather than checking constraint expression types.

    """

    def __init__(self, model=None, active=True, include_inequality=True, **kwds):
        """Construct an IncidenceGraphInterface object"""
        # If the user gives us a model or an NLP, we assume they want us
        # to cache the incidence graph for fast analysis later on.
        # WARNING: This cache will become invalid if the user alters their
        # model.
        self._config = get_config_from_kwds(**kwds)
        if model is None:
            self._incidence_graph = None
            self._variables = None
            self._constraints = None
        elif isinstance(model, BlockData):
            self._constraints = [
                con
                for con in model.component_data_objects(Constraint, active=active)
                if include_inequality or isinstance(con.expr, EqualityExpression)
            ]
            self._variables = list(
                _generate_variables_in_constraints(self._constraints, **self._config)
            )
            self._var_index_map = ComponentMap(
                (var, i) for i, var in enumerate(self._variables)
            )
            self._con_index_map = ComponentMap(
                (con, i) for i, con in enumerate(self._constraints)
            )
            self._incidence_graph = get_bipartite_incidence_graph(
                self._variables, self._constraints, **self._config
            )
        elif pyomo_nlp_available and isinstance(model, pyomo_nlp.PyomoNLP):
            if not active:
                raise ValueError(
                    "Cannot get the Jacobian of inactive constraints from the "
                    "nl interface (PyomoNLP).\nPlease set the `active` flag "
                    "to True."
                )
            if kwds:
                raise ValueError(
                    "Incidence graph generation options, e.g. include_fixed, method,"
                    " and linear_only, are not supported when generating a graph"
                    " from a PyomoNLP."
                )
            nlp = model
            self._variables = nlp.get_pyomo_variables()
            self._constraints = [
                con
                for con in nlp.get_pyomo_constraints()
                if include_inequality or isinstance(con.expr, EqualityExpression)
            ]
            self._var_index_map = ComponentMap(
                (var, idx) for idx, var in enumerate(self._variables)
            )
            self._con_index_map = ComponentMap(
                (con, idx) for idx, con in enumerate(self._constraints)
            )
            if include_inequality:
                incidence_matrix = nlp.evaluate_jacobian()
            else:
                incidence_matrix = nlp.evaluate_jacobian_eq()
            nxb = nx.algorithms.bipartite
            self._incidence_graph = nxb.from_biadjacency_matrix(incidence_matrix)
        elif isinstance(model, tuple):
            # model is a tuple of (nx.Graph, list[pyo.Var], list[pyo.Constraint])
            # We could potentially accept a tuple (variables, constraints).
            # TODO: Disallow kwargs if this type of "model" is provided?
            nx_graph, variables, constraints = model
            self._variables = list(variables)
            self._constraints = list(constraints)
            self._var_index_map = ComponentMap(
                (var, i) for i, var in enumerate(self._variables)
            )
            self._con_index_map = ComponentMap(
                (con, i) for i, con in enumerate(self._constraints)
            )
            # For now, don't check any properties of this graph. We could check
            # for a bipartition that matches the variable and constraint lists.
            self._incidence_graph = nx_graph
        else:
            raise TypeError(
                "Unsupported type for incidence graph. Expected PyomoNLP"
                " or BlockData but got %s." % type(model)
            )

    @property
    def variables(self):
        """The variables participating in the incidence graph"""
        if self._incidence_graph is None:
            raise RuntimeError("Cannot get variables when nothing is cached")
        return self._variables

    @property
    def constraints(self):
        """The constraints participating in the incidence graph"""
        if self._incidence_graph is None:
            raise RuntimeError("Cannot get constraints when nothing is cached")
        return self._constraints

    @property
    def n_edges(self):
        """The number of edges in the incidence graph, or the number of
        structural nonzeros in the incidence matrix
        """
        # The number of structural nonzeros in the incidence matrix
        if self._incidence_graph is None:
            raise RuntimeError(
                "Cannot get number of edges (nonzeros) when nothing is cached"
            )
        return len(self._incidence_graph.edges)

    @property
    @deprecated(
        msg="``var_index_map`` is deprecated. Please use ``get_matrix_coord`` instead.",
        version="6.5.0",
    )
    def var_index_map(self):
        return self._var_index_map

    @property
    @deprecated(
        msg="``con_index_map`` is deprecated. Please use ``get_matrix_coord`` instead.",
        version="6.5.0",
    )
    def con_index_map(self):
        return self._con_index_map

    @property
    @deprecated(
        msg="The ``row_block_map`` attribute is deprecated and will be removed.",
        version="6.5.0",
    )
    def row_block_map(self):
        return None

    @property
    @deprecated(
        msg="The ``col_block_map`` attribute is deprecated and will be removed.",
        version="6.5.0",
    )
    def col_block_map(self):
        return None

    def get_matrix_coord(self, component):
        """Return the row or column coordinate of the component in the incidence
        *matrix* of variables and constraints

        Variables will return a column coordinate and constraints will return
        a row coordinate.

        Parameters
        ----------
        component: ``ComponentData``
            Component whose coordinate to locate

        Returns
        -------
        ``int``
            Column or row coordinate of the provided variable or constraint

        """
        if self._incidence_graph is None:
            raise RuntimeError(
                "Cannot get the coordinate of %s if an incidence graph"
                " is not cached." % component.name
            )
        _check_unindexed([component])
        if component in self._var_index_map and component in self._con_index_map:
            raise RuntimeError(
                "%s is in both variable and constraint maps."
                " This should not happen." % component.name
            )
        elif component in self._var_index_map:
            return self._var_index_map[component]
        elif component in self._con_index_map:
            return self._con_index_map[component]
        else:
            raise RuntimeError(
                "%s is not included in the incidence graph" % component.name
            )

    def _validate_input(self, variables, constraints):
        if variables is None:
            if self._incidence_graph is None:
                raise ValueError("Neither variables nor a model have been provided.")
            else:
                variables = self.variables
        elif self._incidence_graph is not None:
            # If variables were provided and an incidence graph is cached,
            # make sure the provided variables exist in the graph.
            for var in variables:
                if var not in self._var_index_map:
                    raise KeyError(
                        f"Variable {var} does not exist in the cached"
                        " incidence graph."
                    )
        if constraints is None:
            if self._incidence_graph is None:
                raise ValueError("Neither constraints nor a model have been provided.")
            else:
                constraints = self.constraints
        elif self._incidence_graph is not None:
            # If constraints were provided and an incidence graph is cached,
            # make sure the provided constraints exist in the graph.
            for con in constraints:
                if con not in self._con_index_map:
                    raise KeyError(
                        f"Constraint {con} does not exist in the cached"
                        " incidence graph."
                    )

        _check_unindexed(variables + constraints)
        return variables, constraints

    def _extract_subgraph(self, variables, constraints):
        if self._incidence_graph is None:
            # Note that we pass along self._config here, so any kwds used
            # in construction will apply to these incidence graphs.
            return get_bipartite_incidence_graph(variables, constraints, **self._config)
        else:
            constraint_nodes = [self._con_index_map[con] for con in constraints]

            # Note that this is the number of constraints in the original graph,
            # not the subgraph.
            M = len(self.constraints)
            variable_nodes = [M + self._var_index_map[var] for var in variables]
            subgraph = extract_bipartite_subgraph(
                self._incidence_graph, constraint_nodes, variable_nodes
            )
            return subgraph

    def subgraph(self, variables, constraints):
        """Extract a subgraph defined by the provided variables and constraints

        Underlying data structures are copied, and constraints are not reinspected
        for incidence variables (the edges from this incidence graph are used).

        Returns
        -------
        ``IncidenceGraphInterface``
            A new incidence graph containing only the specified variables and
            constraints, and the edges between pairs thereof.

        """
        nx_subgraph = self._extract_subgraph(variables, constraints)
        subgraph = IncidenceGraphInterface(
            (nx_subgraph, variables, constraints), **self._config
        )
        return subgraph

    @property
    def incidence_matrix(self):
        """The structural incidence matrix of variables and constraints.

        Variables correspond to columns and constraints correspond to rows.
        All matrix entries have value 1.0.

        """
        if self._incidence_graph is None:
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
                assert self._incidence_graph.nodes[i]["bipartite"] == 0
                for j in self._incidence_graph[i]:
                    assert self._incidence_graph.nodes[j]["bipartite"] == 1
                    row.append(i)
                    col.append(j - M)
                    data.append(1.0)
            return sp.sparse.coo_matrix((data, (row, col)), shape=(M, N))

    def get_adjacent_to(self, component):
        """Return a list of components adjacent to the provided component
        in the cached bipartite incidence graph of variables and constraints

        Parameters
        ----------
        component: ``ComponentData``
            The variable or constraint data object whose adjacent components
            are returned

        Returns
        -------
        list of ComponentData
            List of constraint or variable data objects adjacent to the
            provided component

        Example
        -------

        .. doctest::
           :skipif: not networkx_available

           >>> import pyomo.environ as pyo
           >>> from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
           >>> m = pyo.ConcreteModel()
           >>> m.x = pyo.Var([1, 2])
           >>> m.eq1 = pyo.Constraint(expr=m.x[1]**2 == 7)
           >>> m.eq2 = pyo.Constraint(expr=m.x[1]*m.x[2] == 3)
           >>> m.eq3 = pyo.Constraint(expr=m.x[1] + 2*m.x[2] == 5)
           >>> igraph = IncidenceGraphInterface(m)
           >>> adj_to_x2 = igraph.get_adjacent_to(m.x[2])
           >>> print([c.name for c in adj_to_x2])
           ['eq2', 'eq3']

        """
        if self._incidence_graph is None:
            raise RuntimeError(
                "Cannot get components adjacent to %s if an incidence graph"
                " is not cached." % component
            )
        _check_unindexed([component])
        M = len(self.constraints)
        N = len(self.variables)
        if component in self._var_index_map:
            vnode = M + self._var_index_map[component]
            adj = self._incidence_graph[vnode]
            adj_comps = [self.constraints[i] for i in adj]
        elif component in self._con_index_map:
            cnode = self._con_index_map[component]
            adj = self._incidence_graph[cnode]
            adj_comps = [self.variables[j - M] for j in adj]
        else:
            raise RuntimeError(
                "Cannot find component %s in the cached incidence graph." % component
            )
        return adj_comps

    def maximum_matching(self, variables=None, constraints=None):
        """Return a maximum cardinality matching of variables and constraints.

        The matching maps constraints to their matched variables.

        Returns
        -------
        ``ComponentMap``
            A map from constraints to their matched variables.

        """
        variables, constraints = self._validate_input(variables, constraints)
        graph = self._extract_subgraph(variables, constraints)
        con_nodes = list(range(len(constraints)))
        matching = maximum_matching(graph, top_nodes=con_nodes)
        # Matching maps constraint nodes to variable nodes. Here we need to
        # know the convention according to which the graph was constructed.
        M = len(constraints)
        return ComponentMap(
            (constraints[i], variables[j - M]) for i, j in matching.items()
        )

    def get_connected_components(self, variables=None, constraints=None):
        """Partition variables and constraints into weakly connected components
        of the incidence graph

        These correspond to diagonal blocks in a block diagonalization of the
        incidence matrix.

        Returns
        -------
        var_blocks: list of lists of variables
            Partition of variables into connected components
        con_blocks: list of lists of constraints
            Partition of constraints into corresponding connected components

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
        var_blocks = [[variables[i - M] for i in block] for block in var_blocks]

        return var_blocks, con_blocks

    # NOTE: That this replaces the <=6.4.4 block_triangularize function
    def map_nodes_to_block_triangular_indices(self, variables=None, constraints=None):
        """Map variables and constraints to indices of their diagonal blocks in
        a block lower triangular permutation

        Returns
        -------
        var_block_map: ``ComponentMap``
            Map from variables to their diagonal blocks in a block
            triangularization
        con_block_map: ``ComponentMap``
            Map from constraints to their diagonal blocks in a block
            triangularization

        """
        variables, constraints = self._validate_input(variables, constraints)
        graph = self._extract_subgraph(variables, constraints)

        M = len(constraints)
        con_nodes = list(range(M))
        sccs = get_scc_of_projection(graph, con_nodes)
        row_idx_map = {r: idx for idx, scc in enumerate(sccs) for r, _ in scc}
        col_idx_map = {c - M: idx for idx, scc in enumerate(sccs) for _, c in scc}
        con_block_map = ComponentMap(
            (constraints[i], idx) for i, idx in row_idx_map.items()
        )
        var_block_map = ComponentMap(
            (variables[j], idx) for j, idx in col_idx_map.items()
        )
        # Switch the order of the maps here to match the method call.
        # Hopefully this does not get too confusing...
        return var_block_map, con_block_map

    def block_triangularize(self, variables=None, constraints=None):
        """Compute an ordered partition of the provided variables and
        constraints such that their incidence matrix is block lower triangular

        Subsets in the partition correspond to the strongly connected components
        of the bipartite incidence graph, projected with respect to a perfect
        matching.

        Returns
        -------
        var_partition: list of lists
            Partition of variables. The inner lists hold unindexed variables.
        con_partition: list of lists
            Partition of constraints. The inner lists hold unindexed constraints.

        Example
        -------

        .. doctest::
           :skipif: not networkx_available

           >>> import pyomo.environ as pyo
           >>> from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
           >>> m = pyo.ConcreteModel()
           >>> m.x = pyo.Var([1, 2])
           >>> m.eq1 = pyo.Constraint(expr=m.x[1]**2 == 7)
           >>> m.eq2 = pyo.Constraint(expr=m.x[1]*m.x[2] == 3)
           >>> igraph = IncidenceGraphInterface(m)
           >>> vblocks, cblocks = igraph.block_triangularize()
           >>> print([[v.name for v in vb] for vb in vblocks])
           [['x[1]'], ['x[2]']]
           >>> print([[c.name for c in cb] for cb in cblocks])
           [['eq1'], ['eq2']]

        .. note::

           **Breaking change in Pyomo 6.5.0**

           The pre-6.5.0 ``block_triangularize`` method returned maps from
           each variable or constraint to the index of its block in a block
           lower triangularization as the original intent of this function
           was to identify when variables do or don't share a diagonal block
           in this partition. Since then, the dominant use case of
           ``block_triangularize`` has been to partition variables and
           constraints into these blocks and inspect or solve each block
           individually. A natural return type for this functionality is the
           ordered partition of variables and constraints, as lists of lists.
           This functionality was previously available via the
           ``get_diagonal_blocks`` method, which was confusing as it did not
           capture that the partition was the diagonal of a block
           *triangularization* (as opposed to diagonalization). The pre-6.5.0
           functionality of ``block_triangularize`` is still available via the
           ``map_nodes_to_block_triangular_indices`` method.

        """
        variables, constraints = self._validate_input(variables, constraints)
        graph = self._extract_subgraph(variables, constraints)

        M = len(constraints)
        con_nodes = list(range(M))
        sccs = get_scc_of_projection(graph, con_nodes)
        var_partition = [[variables[j - M] for _, j in scc] for scc in sccs]
        con_partition = [[constraints[i] for i, _ in scc] for scc in sccs]
        return var_partition, con_partition

    @deprecated(
        msg=(
            "``IncidenceGraphInterface.get_diagonal_blocks`` is deprecated."
            " Please use ``IncidenceGraphInterface.block_triangularize``"
            " instead."
        ),
        version="6.5.0",
    )
    def get_diagonal_blocks(self, variables=None, constraints=None):
        variables, constraints = self._validate_input(variables, constraints)
        graph = self._extract_subgraph(variables, constraints)
        M = len(constraints)
        con_nodes = list(range(M))
        sccs = get_scc_of_projection(graph, con_nodes)
        block_cons = [[constraints[i] for i, _ in scc] for scc in sccs]
        block_vars = [[variables[j - M] for _, j in scc] for scc in sccs]
        return block_vars, block_cons

    def dulmage_mendelsohn(self, variables=None, constraints=None):
        """Partition variables and constraints according to the Dulmage-
        Mendelsohn characterization of the incidence graph

        Variables are partitioned into the following subsets:

        - **unmatched** - Variables not matched in a particular maximum
          cardinality matching
        - **underconstrained** - Variables that *could possibly be* unmatched
          in a maximum cardinality matching
        - **square** - Variables in the well-constrained subsystem
        - **overconstrained** - Variables matched with constraints that can
          possibly be unmatched

        Constraints are partitioned into the following subsets:

        - **underconstrained** - Constraints matched with variables that can
          possibly be unmatched
        - **square** - Constraints in the well-constrained subsystem
        - **overconstrained** - Constraints that *can possibly be* unmatched
          with a maximum cardinality matching
        - **unmatched** - Constraints that were not matched in a particular
          maximum cardinality matching

        While the Dulmage-Mendelsohn decomposition does not specify an order
        within any of these subsets, the order returned by this function
        preserves the maximum matching that is used to compute the decomposition.
        That is, zipping "corresponding" variable and constraint subsets yields
        pairs in this maximum matching. For example:

        .. doctest::
           :hide:
           :skipif: not (networkx_available and scipy_available)

           >>> # Hidden code block creating a dummy model so the following doctest runs
           >>> import pyomo.environ as pyo
           >>> from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
           >>> model = pyo.ConcreteModel()
           >>> model.x = pyo.Var([1,2,3])
           >>> model.eq = pyo.Constraint(expr=sum(m.x[:]) == 1)

        .. doctest::
           :skipif: not (networkx_available and scipy_available)

           >>> igraph = IncidenceGraphInterface(model)
           >>> var_dmpartition, con_dmpartition = igraph.dulmage_mendelsohn()
           >>> vdmp = var_dmpartition
           >>> cdmp = con_dmpartition
           >>> matching = list(zip(
           ...     vdmp.underconstrained + vdmp.square + vdmp.overconstrained,
           ...     cdmp.underconstrained + cdmp.square + cdmp.overconstrained,
           ... ))
           >>> # matching is a valid maximum matching of variables and constraints!

        Returns
        -------
        var_partition: ``ColPartition`` named tuple
            Partitions variables into square, underconstrained, overconstrained,
            and unmatched.
        con_partition: ``RowPartition`` named tuple
            Partitions constraints into square, underconstrained,
            overconstrained, and unmatched.

        Example
        -------

        .. doctest::
           :skipif: not networkx_available

           >>> import pyomo.environ as pyo
           >>> from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
           >>> m = pyo.ConcreteModel()
           >>> m.x = pyo.Var([1, 2])
           >>> m.eq1 = pyo.Constraint(expr=m.x[1]**2 == 7)
           >>> m.eq2 = pyo.Constraint(expr=m.x[1]*m.x[2] == 3)
           >>> m.eq3 = pyo.Constraint(expr=m.x[1] + 2*m.x[2] == 5)
           >>> igraph = IncidenceGraphInterface(m)
           >>> var_dmp, con_dmp = igraph.dulmage_mendelsohn()
           >>> print([v.name for v in var_dmp.overconstrained])
           ['x[1]', 'x[2]']
           >>> print([c.name for c in con_dmp.overconstrained])
           ['eq1', 'eq2']
           >>> print([c.name for c in con_dmp.unmatched])
           ['eq3']

        """
        variables, constraints = self._validate_input(variables, constraints)
        graph = self._extract_subgraph(variables, constraints)
        M = len(constraints)
        top_nodes = list(range(M))
        row_partition, col_partition = dulmage_mendelsohn(graph, top_nodes=top_nodes)
        con_partition = RowPartition(
            *[[constraints[i] for i in subset] for subset in row_partition]
        )
        var_partition = ColPartition(
            *[[variables[i - M] for i in subset] for subset in col_partition]
        )
        # Switch the order of the maps here to match the method call.
        # Hopefully this does not get too confusing...
        return var_partition, con_partition

    def remove_nodes(self, variables=None, constraints=None):
        """Removes the specified variables and constraints (columns and
        rows) from the cached incidence matrix.

        This is a "projection" of the variable and constraint vectors, rather
        than something like a vertex elimination. For the puropse of this
        method, there is no need to distinguish between variables and
        constraints. However, we provide the "constraints" argument so a call
        signature similar to other methods in this class is still valid.

        Parameters
        ----------
        variables: list
            VarData objects whose nodes will be removed from the incidence graph
        constraints: list
            ConData objects whose nodes will be removed from the incidence graph

        .. note::

           **Deprecation in Pyomo v6.7.2**

           The pre-6.7.2 implementation of ``remove_nodes`` allowed variables and
           constraints to remove to be specified in a single list. This made
           error checking difficult, and indeed, if invalid components were
           provided, we carried on silently instead of throwing an error or
           warning. As part of a fix to raise an error if an invalid component
           (one that is not part of the incidence graph) is provided, we now require
           variables and constraints to be specified separately.

        """
        if constraints is None:
            constraints = []
        if variables is None:
            variables = []
        if self._incidence_graph is None:
            raise RuntimeError(
                "Attempting to remove variables and constraints from cached "
                "incidence matrix,\nbut no incidence matrix has been cached."
            )

        vars_to_validate = []
        cons_to_validate = []
        depr_msg = (
            "In IncidenceGraphInterface.remove_nodes, passing variables and"
            " constraints in the same list is deprecated. Please separate your"
            " variables and constraints and pass them in the order variables,"
            " constraints."
        )
        if any(var in self._con_index_map for var in variables) or any(
            con in self._var_index_map for con in constraints
        ):
            deprecation_warning(depr_msg, version="6.7.2")
        # If we received variables/constraints in the same list, sort them.
        # Any unrecognized objects will be caught by _validate_input.
        for var in variables:
            if var in self._con_index_map:
                cons_to_validate.append(var)
            else:
                vars_to_validate.append(var)
        for con in constraints:
            if con in self._var_index_map:
                vars_to_validate.append(con)
            else:
                cons_to_validate.append(con)

        variables, constraints = self._validate_input(
            vars_to_validate, cons_to_validate
        )
        v_exclude = ComponentSet(variables)
        c_exclude = ComponentSet(constraints)
        vars_to_include = [v for v in self.variables if v not in v_exclude]
        cons_to_include = [c for c in self.constraints if c not in c_exclude]
        incidence_graph = self._extract_subgraph(vars_to_include, cons_to_include)
        # update attributes
        self._variables = vars_to_include
        self._constraints = cons_to_include
        self._incidence_graph = incidence_graph
        self._var_index_map = ComponentMap(
            (var, i) for i, var in enumerate(vars_to_include)
        )
        self._con_index_map = ComponentMap(
            (con, i) for i, con in enumerate(cons_to_include)
        )

    def plot(self, variables=None, constraints=None, title=None, show=True):
        """Plot the bipartite incidence graph of variables and constraints"""
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
        edge_trace = plotly.graph_objects.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
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
                node_color.append("red")
                body_text = "<br>".join(
                    textwrap.wrap(str(c.body), width=120, subsequent_indent="    ")
                )
                node_text.append(
                    f"{str(c)}<br>lb: {str(c.lower)}<br>body: {body_text}<br>"
                    f"ub: {str(c.upper)}<br>active: {str(c.active)}"
                )
            else:
                # According to convention, we are a variable node
                v = variables[node - M]
                node_color.append("blue")
                node_text.append(
                    f"{str(v)}<br>lb: {str(v.lb)}<br>ub: {str(v.ub)}<br>"
                    f"value: {str(v.value)}<br>domain: {str(v.domain)}<br>"
                    f"fixed: {str(v.is_fixed())}"
                )
        node_trace = plotly.graph_objects.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=node_text,
            marker=dict(color=node_color, size=10),
        )
        fig = plotly.graph_objects.Figure(data=[edge_trace, node_trace])
        if title is not None:
            fig.update_layout(title=dict(text=title))
        if show:
            fig.show()

    def add_edge(self, variable, constraint):
        """Adds an edge between variable and constraint in the incidence graph

        Parameters
        ----------
        variable: VarData
            A variable in the graph
        constraint: ConstraintData
            A constraint in the graph
        """
        if self._incidence_graph is None:
            raise RuntimeError(
                "Attempting to add edge in an incidence graph from cached "
                "incidence graph,\nbut no incidence graph has been cached."
            )

        if variable not in self._var_index_map:
            raise RuntimeError("%s is not a variable in the incidence graph" % variable)

        if constraint not in self._con_index_map:
            raise RuntimeError(
                "%s is not a constraint in the incidence graph" % constraint
            )

        var_id = self._var_index_map[variable] + len(self._con_index_map)
        con_id = self._con_index_map[constraint]

        self._incidence_graph.add_edge(var_id, con_id)
