from typing import Dict, Tuple, Optional
import warnings
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_scatter import scatter_add

from knowledge_graph.core.graph import KnowledgeGraph


def build_relation_graph(graph):

    # expect the graph is already with inverse edges

    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(
        dim=0
    )  # (num_edges, 2)
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T,
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]],
        (num_rels, num_nodes),
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, torch.ones(Eh.shape[0], device=device), (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(
        dim=0
    )  # (num_edges, 2)

    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T,
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]],
        (num_rels, num_nodes),
    )
    Et = torch.sparse_coo_tensor(
        Et.T, torch.ones(Et.shape[0], device=device), (num_nodes, num_rels)
    )

    warnings.filterwarnings(
        "ignore", ".*Sparse CSR tensor support is in beta state.*"
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    hh_edges = torch.cat(
        [
            Ahh.indices().T,
            torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(
                0
            ),
        ],
        dim=1,
    )  # head to head
    tt_edges = torch.cat(
        [
            Att.indices().T,
            torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(
                1
            ),
        ],
        dim=1,
    )  # tail to tail
    ht_edges = torch.cat(
        [
            Aht.indices().T,
            torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(
                2
            ),
        ],
        dim=1,
    )  # head to tail
    th_edges = torch.cat(
        [
            Ath.indices().T,
            torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(
                3
            ),
        ],
        dim=1,
    )  # tail to head

    rel_graph = Data(
        edge_index=torch.cat(
            [
                hh_edges[:, [0, 1]].T,
                tt_edges[:, [0, 1]].T,
                ht_edges[:, [0, 1]].T,
                th_edges[:, [0, 1]].T,
            ],
            dim=1,
        ),
        edge_type=torch.cat(
            [hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]],
            dim=0,
        ),
        num_nodes=num_rels,
        num_relations=4,
    )

    graph.relation_graph = rel_graph
    return graph


class CustomULTRADataset(InMemoryDataset):
    """Custom dataset class for ULTRA compatibility."""

    def __init__(
        self,
        train_data: Data,
        valid_data: Data,
        test_data: Data,
        full_data: Data,
        entity_to_idx: Dict[str, int],
        relation_to_idx: Dict[str, int],
        root: Optional[str] = None,
        transform=None,
        pre_transform=None,
    ):
        super().__init__(root, transform, pre_transform)
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        self._data = full_data
        self.data, self.slices = self.collate(
            [train_data, valid_data, test_data]
        )

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx == 0:
                return self.train_data
            elif idx == 1:
                return self.valid_data
            elif idx == 2:
                return self.test_data
            else:
                raise IndexError(f"Index {idx} out of range")
        else:
            return super().__getitem__(idx)

    def __len__(self):
        return 3


class ULTRAAdapter:
    """
    Adapter for converting KnowledgeGraph to ULTRA's format.
    ULTRA expects:
    - Edge indices and types in PyG Data format
    - Separate train/valid/test splits
    - Relation graphs for each split
    """

    @staticmethod
    def to_dataset(
        kg: KnowledgeGraph,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        root_dir: Optional[str] = None,
        seed: int = 42,
    ) -> CustomULTRADataset:
        """
        Convert KnowledgeGraph to ULTRA's dataset format.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to convert
        split_ratios : tuple[float, float, float]
            Ratios for train/valid/test splits
        root_dir : str, optional
            Directory to save the dataset

        Returns
        -------
        CustomULTRADataset
            Dataset in ULTRA's format
        """
        # Create entity and relation mappings
        entity_to_idx: Dict[str, int] = {}
        relation_to_idx: Dict[str, int] = {}

        # Map entities and relations to indices
        for node in kg.nodes():
            if node.name not in entity_to_idx:
                entity_to_idx[node.name] = len(entity_to_idx)

        for _, _, data in kg.edges(data=True):
            rel_type = data.get("type")
            if rel_type and rel_type not in relation_to_idx:
                relation_to_idx[rel_type] = len(relation_to_idx)

        # Convert edges to tensor format
        edges = list(kg.edges(data=True))
        edge_index = torch.tensor(
            [
                [entity_to_idx[h.name], entity_to_idx[t.name]]
                for h, t, _ in edges
            ],
            dtype=torch.long,
        ).t()

        edge_type = torch.tensor(
            [relation_to_idx[data["type"]] for _, _, data in edges],
            dtype=torch.long,
        )

        # Split edges into train/valid/test
        num_edges = len(edges)
        train_size = int(num_edges * split_ratios[0])
        valid_size = int(num_edges * split_ratios[1])

        # Create random indices for splitting
        torch.manual_seed(seed)
        indices = torch.randperm(num_edges)
        train_idx = indices[:train_size]
        valid_idx = indices[train_size : train_size + valid_size]
        test_idx = indices[train_size + valid_size :]

        # Create splits
        train_edge_index = edge_index[:, train_idx]
        train_edge_type = edge_type[train_idx]
        valid_edge_index = edge_index[:, valid_idx]
        valid_edge_type = edge_type[valid_idx]
        test_edge_index = edge_index[:, test_idx]
        test_edge_type = edge_type[test_idx]

        # Create bidirectional edges for the graph structure
        num_relations = len(relation_to_idx)
        train_graph_index = torch.cat(
            [train_edge_index, train_edge_index.flip(0)], dim=1
        )
        train_graph_type = torch.cat(
            [train_edge_type, train_edge_type + num_relations]
        )

        # Create Data objects for each split
        train_data = Data(
            edge_index=train_graph_index,
            edge_type=train_graph_type,
            target_edge_index=train_edge_index,
            target_edge_type=train_edge_type,
            num_nodes=len(entity_to_idx),
            num_relations=num_relations * 2,  # *2 for inverse relations
        )

        valid_data = Data(
            edge_index=train_graph_index,  # Use training graph for validation
            edge_type=train_graph_type,
            target_edge_index=valid_edge_index,
            target_edge_type=valid_edge_type,
            num_nodes=len(entity_to_idx),
            num_relations=num_relations * 2,
        )

        test_data = Data(
            edge_index=train_graph_index,  # Use training graph for testing
            edge_type=train_graph_type,
            target_edge_index=test_edge_index,
            target_edge_type=test_edge_type,
            num_nodes=len(entity_to_idx),
            num_relations=num_relations * 2,
        )

        # Create full data object
        full_data = Data(
            target_edge_index=edge_index,
            target_edge_type=edge_type,
            num_nodes=len(entity_to_idx),
            num_relations=num_relations * 2,
        )

        # Build relation graphs (required by ULTRA)
        train_data = build_relation_graph(train_data)
        valid_data = build_relation_graph(valid_data)
        test_data = build_relation_graph(test_data)

        # Create and return dataset
        dataset = CustomULTRADataset(
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            full_data=full_data,
            entity_to_idx=entity_to_idx,
            relation_to_idx=relation_to_idx,
            root=root_dir,
        )

        return dataset
