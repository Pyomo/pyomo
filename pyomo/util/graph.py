import networkx as nx
import pyomo.environ as pe
import plotly.graph_objects as go
from pyomo.core.base.block import _BlockData
from pyomo.core.base.var import _GeneralVarData, ScalarVar
from pyomo.core.base.constraint import _GeneralConstraintData, ScalarConstraint
from pyomo.core.base.objective import _GeneralObjectiveData, ScalarObjective
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import OrderedSet, ComponentSet
from typing import Optional
import textwrap


class _CompNode(object):
    def __init__(self, comp):
        self.comp = comp

    def __eq__(self, other):
        if type(other) is _CompNode:
            return self.comp is other.comp
        return False

    def __hash__(self):
        return hash(id(self.comp))


def graph_from_pyomo(m: _BlockData,
                     include_objective: bool = True,
                     active: bool = True) -> nx.Graph:
    graph = nx.Graph()

    for v in ComponentSet(m.component_data_objects(pe.Var, descend_into=True)):
        graph.add_node(_CompNode(v))

    for con in OrderedSet(m.component_data_objects(pe.Constraint, descend_into=True, active=active)):
        graph.add_node(con)
        for v in identify_variables(con.body, include_fixed=True):
            graph.add_edge(con, _CompNode(v))

    if include_objective:
        for obj in ComponentSet(m.component_data_objects(pe.Objective, descend_into=True, active=active)):
            graph.add_node(_CompNode(obj))
            for v in identify_variables(obj.expr, include_fixed=True):
                graph.add_edge(_CompNode(obj), _CompNode(v))

    return graph


def plot_pyomo_model(m: _BlockData,
                     include_objective: bool = True,
                     active: bool = True,
                     plot_title: Optional[str] = None,
                     bipartite_plot: bool = False,
                     show_plot: bool = True):
    graph = graph_from_pyomo(m, include_objective=include_objective, active=active)
    if bipartite_plot:
        left_nodes = [c for c in OrderedSet(m.component_data_objects(pe.Constraint, descend_into=True, active=active))]
        left_nodes.extend(_CompNode(obj) for obj in ComponentSet(m.component_data_objects(pe.Objective, descend_into=True, active=active)))
        pos_dict = nx.drawing.bipartite_layout(graph, nodes=left_nodes)
    else:
        pos_dict = nx.drawing.spring_layout(graph, seed=0)

    edge_x = list()
    edge_y = list()
    for start_node, end_node in graph.edges():
        x0, y0 = pos_dict[start_node]
        x1, y1 = pos_dict[end_node]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x = list()
    node_y = list()
    node_text = list()
    node_color = list()
    for node in graph.nodes():
        x, y = pos_dict[node]
        node_x.append(x)
        node_y.append(y)
        if type(node) == _CompNode and type(node.comp) in {_GeneralVarData, ScalarVar}:
            v: _GeneralVarData = node.comp
            node_color.append('blue')
            node_text.append(f'{str(v)}<br>lb: {str(v.lb)}<br>ub: {str(v.ub)}<br>'
                             f'value: {str(v.value)}<br>domain: {str(v.domain)}<br>'
                             f'fixed: {str(v.is_fixed())}')
        elif type(node) in {ScalarConstraint, _GeneralConstraintData}:
            c: _GeneralConstraintData = node
            node_color.append('red')
            body_text = '<br>'.join(textwrap.wrap(str(c.body), width=120, subsequent_indent="    "))
            node_text.append(f'{str(c)}<br>lb: {str(c.lower)}<br>body: {body_text}<br>'
                             f'ub: {str(c.upper)}<br>active: {str(c.active)}')
        elif type(node) == _CompNode and type(node.comp) in {ScalarObjective, _GeneralObjectiveData}:
            c: _GeneralObjectiveData = node.comp
            node_color.append('green')
            expr_text = '<br>'.join(textwrap.wrap(str(c.expr), width=120, subsequent_indent="    "))
            node_text.append(f'{str(c)}<br>expr: {expr_text}<br>'
                             f'active: {str(c.active)}')
        else:
            raise ValueError(f'Unexpected node type: {type(node)}')
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text, marker=dict(color=node_color, size=10))

    fig = go.Figure(data=[edge_trace, node_trace])
    if plot_title is not None:
        fig.update_layout(title=dict(text=plot_title))
    if show_plot:  # this option is mostly for unit tests
        fig.show()
