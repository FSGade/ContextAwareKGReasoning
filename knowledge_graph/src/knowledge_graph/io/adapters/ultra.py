"""ULTRA conversion module."""

import warnings
from collections.abc import Iterator
from typing import Any, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_scatter import scatter_add
from tqdm.autonotebook import tqdm

from knowledge_graph.core.graph import KnowledgeGraph


def build_relation_graph(graph):
    """_summary_.

    Args:
        graph (_type_): _description_

    Returns:
        _type_: _description_

    """
    # expect the graph is already with inverse edges

    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T,
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]],
        (num_rels, num_nodes),
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, torch.ones(Eh.shape[0], device=device), (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)

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

    warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    hh_edges = torch.cat(
        [
            Ahh.indices().T,
            torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0),
        ],
        dim=1,
    )  # head to head
    tt_edges = torch.cat(
        [
            Att.indices().T,
            torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1),
        ],
        dim=1,
    )  # tail to tail
    ht_edges = torch.cat(
        [
            Aht.indices().T,
            torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2),
        ],
        dim=1,
    )  # head to tail
    th_edges = torch.cat(
        [
            Ath.indices().T,
            torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3),
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
        entity_to_idx: dict[str, int],
        idx_to_type: dict[int, str],
        relation_to_idx: dict[str, int],
        root: str | None = None,
        transform=None,
        pre_transform=None,
    ):
        """_summary_.

        Args:
            train_data (Data): _description_
            valid_data (Data): _description_
            test_data (Data): _description_
            full_data (Data): _description_
            entity_to_idx (Dict[str, int]): _description_
            relation_to_idx (Dict[str, int]): _description_
            root (Optional[str], optional): _description_. Defaults to None.
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.

        """
        super().__init__(root, transform, pre_transform)
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.entity_to_idx = entity_to_idx
        self.idx_to_type = idx_to_type
        self.relation_to_idx = relation_to_idx
        self._data = full_data
        self.data, self.slices = self.collate([train_data, valid_data, test_data])

    def __getitem__(self, idx):
        """_summary_.

        Args:
            idx (_type_): _description_

        Raises:
            IndexError: _description_

        Returns:
            _type_: _description_

        """
        if isinstance(idx, int):
            if idx == 0:
                return self.train_data
            if idx == 1:
                return self.valid_data
            if idx == 2:
                return self.test_data
            raise IndexError(f"Index {idx} out of range")
        return super().__getitem__(idx)

    def __len__(self):
        """_summary_.

        Returns:
            int: _description_

        """
        return 3


class ULTRAAdapter:
    """Adapter for converting KnowledgeGraph to ULTRA's format.
    ULTRA expects:
    - Edge indices and types in PyG Data format
    - Separate train/valid/test splits
    - Relation graphs for each split
    """

    @staticmethod
    def _progress_wrapper(
        iterable: Iterator[Any],
        show_progress: bool,
        desc: str,
        total: int | None = None,
    ):
        """Helper method to wrap iterables with optional progress bars."""
        if show_progress:
            if total is not None:
                return tqdm(iterable, desc=desc, total=total)
            return tqdm(iterable, desc=desc)
        return iterable

    @staticmethod
    def _create_progress_bar(show_progress: bool, total: int, desc: str):
        """Helper method to create progress bars."""
        if show_progress:
            return tqdm(total=total, desc=desc)
        return None

    @staticmethod
    def _update_progress(pbar, desc: str | None = None):
        """Helper method to update progress bars."""
        if pbar is not None:
            if desc:
                pbar.set_description(desc)
            pbar.update(1)

    @staticmethod
    def _close_progress(pbar):
        """Helper method to close progress bars."""
        if pbar is not None:
            pbar.close()

    @staticmethod
    def to_dataset(
        kg: "KnowledgeGraph",
        root_dir: str | None = None,
        show_progress: bool = True,
        subgraph_of: Optional["KnowledgeGraph"] = None,
        split_attr: str = "split",
        default_split_if_missing: str | None = "excluded",
        split_type: str = "transductive",  # "transductive", "inductive", or "fully-inductive"
    ) -> CustomULTRADataset:
        """Convert KnowledgeGraph to ULTRA's dataset format using attribute-based splits.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to convert.
        root_dir : str, optional
            Directory to save the dataset.
        show_progress : bool, default True
            Whether to show progress bars.
        subgraph_of : KnowledgeGraph, optional
            If provided, use this graph to define entity/relation mappings (schema).
        split_attr : str, default "split"
            Name of the edge attribute from which to read the split label.
        default_split_if_missing : Optional[str], default "excluded"
            If an edge has no split attribute (or it's invalid), use this split label.
            Set to None to raise a ValueError on missing/invalid split.
        split_type : str, default "transductive"
            - "transductive": all splits use the train fact graph
            - "inductive": train/valid use train fact graph; test uses test fact graph
            - "fully-inductive": each split uses its own fact graph (train/valid/test)

        """
        split_type = split_type.lower()
        if split_type not in {"transductive", "inductive", "fully-inductive"}:
            raise ValueError(
                f"Invalid split_type '{split_type}'. "
                "Expected one of {'transductive', 'inductive', 'fully-inductive'}."
            )

        # Create entity and relation mappings
        entity_to_idx: dict[str, int] = {}
        relation_to_idx: dict[str, int] = {}
        idx_to_type: dict[int, str] = {}

        if show_progress:
            print("Processing schema...")

        schema_kg = kg if subgraph_of is None else subgraph_of

        # Map entities and relations to indices (based on schema_kg)
        node_idx_counter = 0
        for node in schema_kg.nodes():
            if node.name not in entity_to_idx:
                entity_to_idx[node.name] = node_idx_counter
                idx_to_type[node_idx_counter] = node.type
                node_idx_counter += 1

        relation_to_idx = {
            r: i for i, r in enumerate(sorted(schema_kg.schema.get_edge_types()))
        }

        num_edges = len(kg.edges())
        schema_num_edges = sum(kg.schema.get_edge_type_usage().values())
        if schema_num_edges != num_edges:
            print(
                f"[WARNING]: The number of edges in the graph ({num_edges}) is not the same as registered in the schema ({schema_num_edges})"
            )

        # Convert edges to tensor format and collect split indices
        if show_progress:
            print("Converting edges to tensor format...")

        pbar = ULTRAAdapter._create_progress_bar(
            show_progress, num_edges, "Converting edge indices"
        )

        edge_list = []
        edge_type_list = []

        attr_train_idx: list[int] = []
        attr_valid_idx: list[int] = []
        attr_test_idx: list[int] = []

        idx_counter = 0
        for h, t, data in kg.edges(data=True):
            # map entities using schema_kg (may include all nodes)
            try:
                h_idx = entity_to_idx[h.name]
                t_idx = entity_to_idx[t.name]
            except KeyError as e:
                # Entity not present in schema mapping
                raise KeyError(
                    f"Entity '{e.args[0]}' not found in mapping. "
                    "Ensure 'subgraph_of' includes all entities you want to index, or omit 'subgraph_of'."
                )
            r_idx = relation_to_idx[data["type"]]

            edge_list.append([h_idx, t_idx])
            edge_type_list.append(r_idx)

            raw = data.get(split_attr, None)
            if raw is None:
                if default_split_if_missing is None:
                    raise ValueError(
                        f"Edge ({h.name}, {t.name}, type={data.get('type')}) is missing split attribute '{split_attr}'"
                    )
                split_label = default_split_if_missing
            else:
                split_label = str(raw).lower()

            # normalize "val" to "valid"
            if split_label == "val":
                split_label = "valid"

            if split_label not in {"train", "valid", "test", "excluded"}:
                if default_split_if_missing is None:
                    raise ValueError(
                        f"Edge ({h.name}, {t.name}, type={data.get('type')}) has invalid split '{raw}'"
                    )
                split_label = default_split_if_missing

            if split_label == "train":
                attr_train_idx.append(idx_counter)
            elif split_label == "valid":
                attr_valid_idx.append(idx_counter)
            elif split_label == "test":
                attr_test_idx.append(idx_counter)
            # edges labeled "excluded" are ignored

            idx_counter += 1
            ULTRAAdapter._update_progress(pbar)

        ULTRAAdapter._close_progress(pbar)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t()  # shape [2, E]
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)  # shape [E]

        # Split edges into train/valid/test (attribute-based only)
        if show_progress:
            print("Collecting train/valid/test targets from split attributes...")

        train_idx = torch.tensor(attr_train_idx, dtype=torch.long)
        valid_idx = torch.tensor(attr_valid_idx, dtype=torch.long)
        test_idx = torch.tensor(attr_test_idx, dtype=torch.long)

        # Target edges per split
        train_edge_index = edge_index[:, train_idx]
        train_edge_type = edge_type[train_idx]

        valid_edge_index = edge_index[:, valid_idx]
        valid_edge_type = edge_type[valid_idx]

        test_edge_index = edge_index[:, test_idx]
        test_edge_type = edge_type[test_idx]

        # Create bidirectional "fact graphs" depending on split_type
        if show_progress:
            print(f"Creating fact graphs for split_type='{split_type}'...")

        num_relations = len(relation_to_idx)

        def make_fact_graph(idx_tensor: torch.Tensor, type_tensor: torch.Tensor):
            # Bidirectional edges + inverse relation types
            if idx_tensor.numel() == 0:
                # Return empty graph with 0 edges
                return torch.empty((2, 0), dtype=torch.long), torch.empty(
                    (0,), dtype=torch.long
                )
            bi_idx = torch.cat([idx_tensor, idx_tensor.flip(0)], dim=1)
            bi_type = torch.cat([type_tensor, type_tensor + num_relations])
            return bi_idx, bi_type

        # Train fact graph
        train_fact_index, train_fact_type = make_fact_graph(
            train_edge_index, train_edge_type
        )

        # Valid/test fact graphs depending on split_type
        if split_type == "transductive":
            valid_fact_index, valid_fact_type = train_fact_index, train_fact_type
            test_fact_index, test_fact_type = train_fact_index, train_fact_type
        elif split_type == "inductive":
            valid_fact_index, valid_fact_type = train_fact_index, train_fact_type
            test_fact_index, test_fact_type = make_fact_graph(
                test_edge_index, test_edge_type
            )
        else:  # fully-inductive
            valid_fact_index, valid_fact_type = make_fact_graph(
                valid_edge_index, valid_edge_type
            )
            test_fact_index, test_fact_type = make_fact_graph(
                test_edge_index, test_edge_type
            )

        # Create Data objects for each split
        if show_progress:
            print("Creating data objects...")

        pbar = ULTRAAdapter._create_progress_bar(
            show_progress, 4, "Creating data objects"
        )

        ULTRAAdapter._update_progress(pbar, "Creating training data")
        train_data = Data(
            edge_index=train_fact_index,
            edge_type=train_fact_type,
            target_edge_index=train_edge_index,
            target_edge_type=train_edge_type,
            num_nodes=len(entity_to_idx),
            num_relations=num_relations * 2,  # inverse relations included
        )

        ULTRAAdapter._update_progress(pbar, "Creating validation data")
        valid_data = Data(
            edge_index=valid_fact_index,
            edge_type=valid_fact_type,
            target_edge_index=valid_edge_index,
            target_edge_type=valid_edge_type,
            num_nodes=len(entity_to_idx),
            num_relations=num_relations * 2,
        )

        ULTRAAdapter._update_progress(pbar, "Creating test data")
        test_data = Data(
            edge_index=test_fact_index,
            edge_type=test_fact_type,
            target_edge_index=test_edge_index,
            target_edge_type=test_edge_type,
            num_nodes=len(entity_to_idx),
            num_relations=num_relations * 2,
        )

        ULTRAAdapter._update_progress(pbar, "Creating full data")
        full_data = Data(
            target_edge_index=edge_index,
            target_edge_type=edge_type,
            num_nodes=len(entity_to_idx),
            num_relations=num_relations * 2,
        )

        ULTRAAdapter._close_progress(pbar)

        # Build relation graphs (required by ULTRA)
        if show_progress:
            print("Building relation graphs...")

        pbar = ULTRAAdapter._create_progress_bar(
            show_progress, 3, "Building relation graphs"
        )
        ULTRAAdapter._update_progress(pbar, "Building training relation graph")
        train_data = build_relation_graph(train_data)
        ULTRAAdapter._update_progress(pbar, "Building validation relation graph")
        valid_data = build_relation_graph(valid_data)
        ULTRAAdapter._update_progress(pbar, "Building test relation graph")
        test_data = build_relation_graph(test_data)
        ULTRAAdapter._close_progress(pbar)

        # Create and return dataset
        if show_progress:
            print("Creating final dataset...")

        dataset = CustomULTRADataset(
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            full_data=full_data,
            entity_to_idx=entity_to_idx,
            idx_to_type=idx_to_type,
            relation_to_idx=relation_to_idx,
            root=root_dir,
        )

        if show_progress:
            print("Dataset creation completed!")

        return dataset
