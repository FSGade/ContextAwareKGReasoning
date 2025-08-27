"""
Graph conversion utilities for transforming NetworkX graphs to PyTorch Geometric formats.

This module provides functionality to convert NetworkX graph objects into PyTorch
Geometric's HeteroData format, maintaining node and edge attributes, types, and
structure. It supports heterogeneous graphs with different node and edge types.

Functions
---------
from_hetero_networkx(G, node_type_attribute, edge_type_attribute=None, ...)
    Convert a NetworkX graph to a PyTorch Geometric HeteroData object.

Private Functions
----------------
_get_edge_attributes(G, edge_indexes, edge_attrs=None)
    Collect attributes of graph edges in a dictionary.

_get_node_attributes(G, nodes, expected_node_attrs=None)
    Collect attributes of graph nodes in a dictionary.

Examples
--------
>>> import networkx as nx
>>> from graph_convert import from_hetero_networkx

# Create a heterogeneous NetworkX graph
>>> G = nx.Graph()
>>> G.add_node(1, type='user', feature=torch.tensor([1.0]))
>>> G.add_node(2, type='item', feature=torch.tensor([0.5]))
>>> G.add_edge(1, 2, type='rates', weight=5.0)

# Convert to PyG HeteroData
>>> data = from_hetero_networkx(G,
...                            node_type_attribute='type',
...                            edge_type_attribute='type')

Notes
-----
- Requires both PyTorch Geometric and NetworkX installations
- Node types are required and must be specified for all nodes
- Edge types are optional (defaults to "to" if not specified)
- Supports automatic feature collation through group_node_attrs and group_edge_attrs
- Handles both directed and undirected input graphs (converts to directed internally)

Dependencies
-----------
- torch
- torch_geometric
- networkx
- collections.defaultdict
- typing

See Also
--------
torch_geometric.data.HeteroData : https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.HeteroData
networkx : https://networkx.org/

Author
------
Frederik S. Gade <fzsg@novonordisk.com>
Simon Popelier (from_hetero_networkx())

Version
-------
1.0.0
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any
from typing import Iterable
from typing import Literal

import networkx as nx
import numpy as np
from knowledge_graph.core.graph import KnowledgeGraph
    
def edges_iter(G, with_rev=False):
    for edge in G.edges(data=True):
        yield edge
        if with_rev:
            rev_edge_data = dict(edge[2])
            rev_edge_data['type'] = 'rev_' + rev_edge_data['type']
            yield edge[1], edge[0], rev_edge_data

# Stolen from https://github.com/pyg-team/pytorch_geometric/pull/7744/files
def to_hetero_torch_geometric(
    G: Any,
    node_type_attribute: str = "type",
    edge_type_attribute: str | None = "type",
    graph_attrs: Iterable[str] | None = None,
    nodes: list | None = None,
    group_node_attrs: list[str] | Literal["all"] | None = None,
    group_node_attrs_exclude: list[str] | None = None,
    group_edge_attrs: list[str] | Literal["all"] | None = None,
    group_edge_attrs_exclude: list[str] | None = None,
    add_reverse: bool = False
) -> Any:
    """Converts a KnowledgeGraph HeteroData instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        node_type_attribute (str): The attribute containing the type of a
            node. For the resulting structure to be valid, this attribute
            must be set for every node in the graph. Values contained in
            this attribute will be casted as :obj:`string` if possible. If
            not, the function will raise an error.
        edge_type_attribute (str, optional): The attribute containing the
            type of an edge. If set to :obj:`None`, the value :obj:`"to"`
            will be used in the final structure. Otherwise, this attribute
            must be set for every edge in the graph. (default: :obj:`None`)
        graph_attrs (iterable of str, optional): The graph attributes to be
            copied. (default: :obj:`None`)
        nodes (list, optional): The list of nodes whose attributes are to
            be collected. If set to :obj:`None`, all nodes of the graph
            will be included. (default: :obj:`None`)
        group_node_attrs (List[str] or "all", optional): The node attributes to
            be concatenated and added to :obj:`data.x`. They must be present
            for all nodes of each type. (default: :obj:`None`)
        group_edge_attrs (List[str] or "all", optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`. They must be
            present for all edge of each type. (default: :obj:`None`)
    Example:
        >>> data = from_hetero_networkx(G, node_type_attribute="type",
        ...                    edge_type_attribute="type")
        <torch_geometric.data.HeteroData()>
    :rtype: :class:`torch_geometric.data.HeteroData`
    """
    import torch
    from torch_geometric.data import HeteroData

    G = _prep_kg_for_heterodata(G)
    # TODO: Refactor such that there is no need to run _prep_kg_for_heterodata

    G = G.to_directed() if not nx.is_directed(G) else G

    if nodes is not None:
        G = nx.subgraph(G, nodes)

    hetero_data_dict = {}

    node_to_group_id = {}
    node_to_group = {}
    group_to_nodes = defaultdict(list)
    group_to_edges = defaultdict(list)

    for node, node_data in G.nodes(data=True):
        if node_type_attribute not in node_data:
            raise KeyError(
                f"Given node_type_attribute: {node_type_attribute} \
                missing from node {node}."
            )
        node_type = str(node_data[node_type_attribute])
        group_to_nodes[node_type].append(node)
        node_to_group_id[node] = len(group_to_nodes[node_type]) - 1
        node_to_group[node] = node_type
        
    G_edges = list(edges_iter(G, with_rev=add_reverse))
    
    from pprint import pprint
    pprint(G_edges[:5])

    for i, (node_a, node_b, edge_data) in enumerate(G_edges):
        if edge_type_attribute is not None:
            if edge_type_attribute not in edge_data:
                raise KeyError(
                    f"Given edge_type_attribute: {edge_type_attribute} \
                    missing from edge {(node_a, node_b)}."
                )
            # node_type_a, edge_type, node_type_b = edge_data[
            #    edge_type_attribute]

            node_type_a = node_to_group[node_a]
            node_type_b = node_to_group[node_b]
            edge_type = edge_data[edge_type_attribute]

            if (
                node_to_group[node_a] != node_type_a
                or node_to_group[node_b] != node_type_b
            ):
                raise ValueError(
                    f"Edge {node_a}-{node_b} of type\
                        {edge_data[edge_type_attribute]} joins nodes of types\
                        {node_to_group[node_a]} and {node_to_group[node_b]}."
                )
        else:
            edge_type = "to"
        group_to_edges[
            (node_to_group[node_a], edge_type, node_to_group[node_b])
        ].append(i)

    for group, group_nodes in group_to_nodes.items():
        hetero_data_dict[str(group)] = {
            k: v
            for k, v in _get_node_attributes(G, nodes=group_nodes).items()
            if k != node_type_attribute
        }

    for edge_group, group_edges in group_to_edges.items():
        group_name = "__".join(edge_group)
        hetero_data_dict[group_name] = {
            k: v
            for k, v in _get_edge_attributes(
                G_edges, edge_indexes=group_edges
            ).items()
            if k != edge_type_attribute
        }
        global_edge_index = [G_edges[edge][:2] for edge in group_edges]
        group_edge_index = [
            (node_to_group_id[node_a], node_to_group_id[node_b])
            for node_a, node_b in global_edge_index
        ]
        hetero_data_dict[group_name]["edge_index"] = (
            torch.tensor(group_edge_index, dtype=torch.long)
            .t()
            .contiguous()
            .view(2, -1)
        )
    graph_items = G.graph
    if graph_attrs is not None:
        graph_items = {
            k: v for k, v in graph_items.items() if k in graph_attrs
        }
    for key, value in graph_items.items():
        hetero_data_dict[str(key)] = value

    for group, group_dict in hetero_data_dict.items():
        if isinstance(group_dict, dict):
            xs = []
            is_edge_group = group in [
                "__".join(k) for k in group_to_edges.keys()
            ]
            if is_edge_group:
                group_attrs = group_edge_attrs
                group_attrs_exclude = group_edge_attrs_exclude
            else:
                group_attrs = group_node_attrs
                group_attrs_exclude = group_node_attrs_exclude
            for key, value in group_dict.items():
                if isinstance(value, (tuple, list)) and any(isinstance(
                    _value, torch.Tensor
                ) for _value in value):
                    value_0 = next((x for x in value if isinstance(x, torch.Tensor)), None)
                    value_zero_imputed = [
                        (
                            v
                            if isinstance(v, torch.Tensor)
                            else torch.zeros_like(value_0)
                        )
                        for v in value
                    ]
                    hetero_data_dict[group][key] = torch.stack(
                        value_zero_imputed, dim=0
                    )
                    print("Stack tensors:", group, key)
                else:
                    tensor_conversion_successful = False
                    try:
                        hetero_data_dict[group][key] = torch.tensor(value)
                        print("To tensor:", group, key)
                        tensor_conversion_successful = True
                    except (ValueError, TypeError, RuntimeError) as e:
                        print("Torch error:", group, key)
                        print(e)
                    if not tensor_conversion_successful:
                        try:
                            #hetero_data_dict[group][key] = np.asarray(value)
                            #print("To numpy array:", group, key)
                            hetero_data_dict[group][key] = None
                        except (ValueError, TypeError, RuntimeError) as e:
                            #print("Numpy error:", group, key)
                            print(e)
                if (
                    group_attrs is not None
                    and (
                        key in group_attrs
                        or (
                            group_attrs == "all"
                            and key not in ("name", "type")
                        )
                    )
                    and (
                        group_attrs_exclude is None
                        or key not in group_attrs_exclude
                    )
                ):

                    _x = hetero_data_dict[group][key]
                    print("Type of", key, "in group", group, "is", type(_x))
                    # print(group, key, len(_x) if not isinstance(_x, torch.Tensor) else _x.shape, _x)
                    if isinstance(_x, torch.Tensor) and key != "edge_index":
                        _x = _x.unsqueeze(1) if _x.dim() == 1 else _x
                        print("Unsqueeze and append", group, key)
                        xs.append(_x)  # .view(-1, 1))
            if group_attrs is not None:
                if len(group_attrs) != len(xs) and group_attrs != "all":
                    # print(group_attrs, xs)
                    raise KeyError(
                        f"Missing required attribute in group: {group}"
                    )
                if is_edge_group:
                    hetero_data_dict[group]["edge_attr"] = torch.cat(xs, dim=1)
                else:
                    for _x in xs:
                        print(
                            _x.shape
                            if isinstance(_x, torch.Tensor)
                            else len(_x)
                        )
                    hetero_data_dict[group]["x"] = torch.cat(xs, dim=1)
        else:
            value = group_dict
            if isinstance(value, (tuple, list)) and isinstance(
                value[0], torch.Tensor
            ):
                hetero_data_dict[group] = torch.stack(value, dim=0)
            else:
                try:
                    hetero_data_dict[group] = torch.tensor(value)
                except (ValueError, TypeError):
                    pass
        
        hetero_data_groups = list(hetero_data_dict.keys())
        for hetero_data_group in hetero_data_groups:
            hetero_data_keys = list(hetero_data_dict[hetero_data_group].keys())
            for key in hetero_data_keys:
                if hetero_data_dict[hetero_data_group][key] is None:
                    del hetero_data_dict[hetero_data_group][key]

    return HeteroData(**hetero_data_dict)

def _prep_kg_for_heterodata(kg: KnowledgeGraph):
    """
    Prepare a knowledge graph for conversion to PyG HeteroData format.
    This function modifies the input graph in-place to make it compatible with
    the from_hetero_networkx conversion function. It performs the following operations:
    1. Converts tuple node identifiers (id, type) into separate node attributes
    2. Relabels nodes to use only their ID component
    3. Removes any 'group' attributes that might interfere with conversion
    4. Sanitizes node and edge types by removing non-alphanumeric characters
       and replacing dashes/spaces with underscores
    Parameters
    ----------
    G : KnowledgeGraph
        Input knowledge graph where nodes are tuples of (id, type).
        The graph is modified in-place.
    Notes
    -----
    - Assumes nodes are stored as tuples of (id, type)
    - Modifies the graph in-place
    - Removes 'group' attributes if present
    """
    import re

    def sanitize_type(type_str):
        # Replace dashes and spaces with underscores
        intermediate = re.sub(r'[-\s]', '_', str(type_str))
        # Remove any character that isn't alphanumeric or underscore
        sanitized = re.sub(r'[^\w]', '', intermediate)
        return sanitized

    _kg = deepcopy(kg)
    
    # Create dictionary of node attributes from tuple identifiers with sanitized types
    kg_type_dict = {
        entity: {"x": i, "type": sanitize_type(entity.type)}
        for (i, entity) in enumerate(_kg.nodes)
    }
    
    # Set node attributes
    nx.set_node_attributes(_kg, kg_type_dict)
    
    # Sanitize edge types
    for edge in _kg.edges(data=True):
        if 'type' in edge[2]:
            edge[2]['type'] = sanitize_type(edge[2]['type'])
    
    # Remove any 'group' attributes
    for _, d in _kg.nodes(data=True):
        if "group" in d:
            del d["group"]
            
    return _kg

# def _prep_kg_for_heterodata(kg: KnowledgeGraph):
#     """
#     Prepare a knowledge graph for conversion to PyG HeteroData format.

#     This function modifies the input graph in-place to make it compatible with
#     the from_hetero_networkx conversion function. It performs the following operations:
#     1. Converts tuple node identifiers (id, type) into separate node attributes
#     2. Relabels nodes to use only their ID component
#     3. Removes any 'group' attributes that might interfere with conversion

#     Parameters
#     ----------
#     G : KnowledgeGraph
#         Input knowledge graph where nodes are tuples of (id, type).
#         The graph is modified in-place.

#     Notes
#     -----
#     - Assumes nodes are stored as tuples of (id, type)
#     - Modifies the graph in-place
#     - Removes 'group' attributes if present
#     """
#     _kg = deepcopy(kg)

#     # Create dictionary of node attributes from tuple identifiers
#     kg_type_dict = {
#         entity: {"x": i, "type": entity.type}
#         for (i, entity) in enumerate(_kg.nodes)
#     }
#     # Set node attributes
#     nx.set_node_attributes(_kg, kg_type_dict)
#     # Relabel nodes to use only ID component
#     # nx.relabel_nodes(_kg, lambda x: x[0], copy=True)
#     # Remove any 'group' attributes
#     for _, d in _kg.nodes(data=True):
#         if "group" in d:
#             del d["group"]

#     return _kg


def _get_edge_attributes(
    edge_to_data: list, edge_indexes: list, edge_attrs: Iterable | None = None
) -> dict:
    """Collects the attributes of a list of graph edges in a dictionary.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        edge_indexes (list, optional): The list of edge indexes whose
            attributes are to be collected. If set to :obj:`None`, all
            edges of the graph will be included. (default: :obj:`None`)
        edge_attrs (iterable, optional): The list of expected attributes to
            be found in every edge. If set to :obj:`None`, the first
            edge encountered will set the values for the rest of the
            process. (default: :obj:`None`)
    Raises:
        ValueError: If some of the edges do not share the same list
        of attributes as the rest, an error will be raised.
    """
    data = defaultdict(list)

    edge_attrs = set()

    for edge_index in edge_indexes:
        _, _, feat_dict = edge_to_data[edge_index]
        edge_attrs.update(feat_dict.keys())

    for edge_index in edge_indexes:
        _, _, feat_dict = edge_to_data[edge_index]
        for feat in set(edge_attrs) - set(feat_dict.keys()):
            feat_dict[feat] = None
            # raise ValueError("Not all edges contain the same attributes.")
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    return data


def _get_node_attributes(
    kg: KnowledgeGraph,
    nodes: list,
    expected_node_attrs: Iterable | None = None,
) -> dict:
    """Collects the attributes of a list of graph nodes in a dictionary.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        nodes (list, optional): The list of nodes whose attributes are to
            be collected. If set to :obj:`None`, all nodes of the graph
            will be included. (default: :obj:`None`)
        expected_node_attrs (iterable, optional): The list of expected
            attributes to be found in every node. If set to :obj:`None`,
            the first node encountered will set the values for the rest
            of the process. (default: :obj:`None`)
    Raises:
        ValueError: If some of the nodes do not share the same
        list of attributes as the rest, an error will be raised.
    """
    data = defaultdict(list)

    node_to_data = kg.nodes(data=True)

    node_attrs = set()

    for node in nodes:
        feat_dict = node_to_data[node]
        node_attrs.update(feat_dict.keys())

    for node in nodes:
        feat_dict = node_to_data[node]
        for feat in set(node_attrs) - set(feat_dict.keys()):
            feat_dict[feat] = None
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    return data
