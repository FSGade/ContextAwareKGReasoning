"""StATIK conversion module."""

import array
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, FeatureExtractionPipeline

from knowledge_graph.core.graph import KnowledgeGraph


# https://github.com/Elanmarkowitz/StATIK/blob/main/data/left_contiguous_csr.py
class LeftContiguousCSR:
    """_summary_."""

    def __init__(self, indptr: np.ndarray, degrees: np.ndarray, data: np.ndarray):
        """_summary_.

        Args:
            indptr (np.ndarray): _description_
            degrees (np.ndarray): _description_
            data (np.ndarray): _description_

        """
        self.indptr = indptr
        self.degrees = degrees
        self.data = data

    def __getitem__(self, i):
        """_summary_.

        Args:
            i (_type_): _description_

        Returns:
            _type_: _description_

        """
        start_ind = self.indptr[i]
        end_ind = start_ind + self.degrees[i]
        return self.data[start_ind:end_ind]

    def save(self, filepath):
        """_summary_.

        Args:
            filepath (_type_): _description_

        """
        np.savez(filepath, indptr=self.indptr, degrees=self.degrees, data=self.data)

    @staticmethod
    def load(filepath):
        """_summary_.

        Args:
            filepath (_type_): _description_

        Returns:
            _type_: _description_

        """
        npzfile = np.load(filepath)
        return LeftContiguousCSR(npzfile["indptr"], npzfile["degrees"], npzfile["data"])

    @staticmethod
    def join(first, second):
        """_summary_.

        Args:
            first (_type_): _description_
            second (_type_): _description_

        Returns:
            _type_: _description_

        """
        indptr = np.concatenate([first.indptr, second.indptr + len(first.data)])
        degrees = np.concatenate([first.degrees, second.degrees])
        data = np.concatenate([first.data, second.data])
        return LeftContiguousCSR(indptr, degrees, data)


# class ProcessWikidata5M(ProcessWordNet):
#     DATASET_INFO = {
#     'dataset': 'Wikidata5M',
#     'url': 'https://surfdrive.surf.nl/files/index.php/s/TEE96zweMxsoGmR/download',
#     'train': 'ind-train.tsv',
#     'test': 'ind-test.tsv',
#     'dev': 'ind-dev.tsv',
#     'ent_desc': 'entity2textlong.txt',
#     'ent_desc2': 'entity2text.txt',
#     'rel_desc': 'relation2text.txt'
# }

#     def __init__(self, root_data_dir=None, dataset_info=None):
#         dataset_info = self.DATASET_INFO if dataset_info is None else dataset_info
#         super(ProcessWikidata5M, self).__init__(root_data_dir=root_data_dir, dataset_info=dataset_info)

# def load_original_data(root_data_dir: str, dataset_name: str) -> ProcessWikidata5M:
#     return ProcessWikidata5M(root_data_dir=root_data_dir)

NUM_WORDS = int(os.environ["n_words"]) if "n_words" in os.environ else 24


class KGProcessedDataset:
    """KG processed dataset."""

    def __init__(self, root_data_dir: str, dataset_name: str):
        """Hollowed out KGProcessedDataset, so init does nothing.

        Args:
            root_data_dir (str): _description_
            dataset_name (str): _description_

        """
        print("Hollowed out KGProcessedDataset, so init does nothing.")
        self.num_relations: int | None = None
        self.num_entities: int | None = None
        self.relation_feat = None
        self.entity_feat = None
        self.train_hrt = None
        self.valid_hrt = None
        self.test_hrt = None
        self.entity_descs = None
        self.relation_descs = None
        self.entity2id = None
        self.relation2id = None
        self.entity_text = None
        self.relation_text = None
        # load_dir = os.path.join(root_data_dir, dataset_name, "processed")
        # print('Loading processed loaded.')
        # loaded = load_original_data(root_data_dir, dataset_name)
        # self.num_entities = loaded.num_entities
        # self.num_relations = loaded.num_relations
        # self.entity_feat = loaded.entity_feat
        # self.relation_feat = loaded.relation_feat
        # self.entity_text = loaded.entity_text
        # self.relation_text = loaded.relation_text
        # self.train_hrt = loaded.train_hrt
        # self.valid_hrt = loaded.valid_hrt
        # self.test_hrt = loaded.test_hrt
        # self.train_edge_lccsr: LeftContiguousCSR = LeftContiguousCSR.load(os.path.join(load_dir, 'train_edge_lccsr.npz'))
        # self.train_relation_lccsr: LeftContiguousCSR = LeftContiguousCSR.load(os.path.join(load_dir, 'train_rel_lccsr.npz'))
        # self.valid_edge_lccsr: LeftContiguousCSR = LeftContiguousCSR.load(os.path.join(load_dir, 'valid_edge_lccsr.npz'))
        # self.valid_relation_lccsr: LeftContiguousCSR = LeftContiguousCSR.load(os.path.join(load_dir, 'valid_rel_lccsr.npz'))
        # self.test_edge_lccsr: LeftContiguousCSR = LeftContiguousCSR.load(os.path.join(load_dir, 'test_edge_lccsr.npz'))
        # self.test_relation_lccsr: LeftContiguousCSR = LeftContiguousCSR.load(os.path.join(load_dir, 'test_rel_lccsr.npz'))
        # self.train_degrees = np.load(os.path.join(load_dir, 'train_degrees.npy'))
        # self.train_indegrees = np.load(os.path.join(load_dir, 'train_indegrees.npy'))
        # self.train_outdegrees = np.load(os.path.join(load_dir, 'train_outdegrees.npy'))
        # self.valid_degrees = np.load(os.path.join(load_dir, 'valid_degrees.npy'))
        # self.valid_indegrees = np.load(os.path.join(load_dir, 'valid_indegrees.npy'))
        # self.valid_outdegrees = np.load(os.path.join(load_dir, 'valid_outdegrees.npy'))
        # self.test_degrees = np.load(os.path.join(load_dir, 'test_degrees.npy'))
        # self.test_indegrees = np.load(os.path.join(load_dir, 'test_indegrees.npy'))
        # self.test_outdegrees = np.load(os.path.join(load_dir, 'valid_outdegrees.npy'))
        self.feature_dim = None
        # self.train_h_filter = self._load_file_if_present(os.path.join(load_dir, 'train_h_filter.npy'))
        # self.train_t_filter = self._load_file_if_present(os.path.join(load_dir, 'train_t_filter.npy'))
        # self.valid_h_filter = self._load_file_if_present(os.path.join(load_dir, 'valid_h_filter.npy'))
        # self.valid_t_filter = self._load_file_if_present(os.path.join(load_dir, 'valid_t_filter.npy'))
        # self.test_h_filter = self._load_file_if_present(os.path.join(load_dir, 'test_h_filter.npy'))
        # self.test_t_filter = self._load_file_if_present(os.path.join(load_dir, 'test_t_filter.npy'))

        self.train_entities = (
            None  # np.load(os.path.join(load_dir, 'train_entities.npy'))
        )
        self.valid_entities = (
            None  # np.load(os.path.join(load_dir, 'valid_entities.npy'))
        )
        self.test_entities = (
            None  # np.load(os.path.join(load_dir, 'test_entities.npy'))
        )
        self.train_targets = (
            None  # np.load(os.path.join(load_dir, 'train_targets.npy'))
        )
        self.valid_targets = (
            None  # np.load(os.path.join(load_dir, 'valid_targets.npy'))
        )
        self.test_targets = None  # np.load(os.path.join(load_dir, 'test_targets.npy'))

    @staticmethod
    def _load_file_if_present(filepath):
        """_summary_.

        Args:
            filepath (_type_): _description_

        Returns:
            _type_: _description_

        """
        if os.path.isfile(filepath):
            return np.load(filepath)
        return None

    @staticmethod
    def get_first_n_words(desc, n=NUM_WORDS):
        """_summary_.

        Args:
            desc (_type_): _description_
            n (_type_, optional): _description_. Defaults to NUM_WORDS.

        Returns:
            _type_: _description_

        """
        words = desc.split(" ")
        return " ".join(words[:n])

    @torch.no_grad()
    def get_entity_features(
        self, embedding_model_name="bert-base-cased", embedding_batch_size=128
    ):
        """_summary_.

        Args:
            embedding_model_name (str, optional): _description_. Defaults to "bert-base-cased".
            embedding_batch_size (int, optional): _description_. Defaults to 128.

        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Creating features using language model.")
        # if self.dataset_info['dataset'] == 'FB15k-237':
        tokenizer = BertTokenizer.from_pretrained(embedding_model_name)
        model = BertModel.from_pretrained(embedding_model_name)
        dp_model = torch.nn.DataParallel(model)
        pipeline = FeatureExtractionPipeline(model, tokenizer, device=device)

        hidden_size = model.config.hidden_size

        self.entity_feat = np.zeros((self.num_entities, hidden_size), dtype=np.float64)
        assert len(self.entity_feat) == len(self.entity_text)
        for i in tqdm(range(0, self.num_entities, embedding_batch_size)):
            batch_slice = slice(i, i + embedding_batch_size)
            batch = [
                self.get_first_n_words(desc)
                for desc in self.entity_text[batch_slice].tolist()
            ]
            inputs = tokenizer(batch, padding=True, return_tensors="pt")
            self.entity_feat[batch_slice] = dp_model(**inputs)[0][:, 0, :].cpu().numpy()

        self.relation_feat = np.array(
            [
                np.array(pipeline(self.get_first_n_words(e)))[0, 0, :].flatten()
                for e in self.relation_text
            ]
        )


class StATIKAdapter:
    """Adapter for converting KnowledgeGraph to StATIK's format."""

    @staticmethod
    def create_processed_dataset(
        kg: KnowledgeGraph,
        save_dir: str | None = None,
        transfer_setting: bool = False,
        embedding_model_name="bert-base-cased",
    ) -> KGProcessedDataset:
        """Convert KnowledgeGraph to StATIK's KGProcessedDataset format.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to convert
        save_dir : str, optional
            Directory to save processed files
        transfer_setting : bool, default=False
            Whether to use transfer learning setting

        Returns
        -------
        KGProcessedDataset
            A StATIK-compatible processed dataset

        """
        # Create mappings for entities and relations
        entity_to_idx: dict[NamedTuple, int] = {}
        relation_to_idx: dict[str, int] = {}

        # Map entities and relations to indices
        for node in kg.nodes():
            if node not in entity_to_idx:
                entity_to_idx[node] = len(entity_to_idx)

        for _, _, data in kg.edges(data=True):
            rel_type = data.get("type")
            if rel_type and rel_type not in relation_to_idx:
                relation_to_idx[rel_type] = len(relation_to_idx)

        # Create HRT arrays
        edges = list(kg.edges(data=True))
        train_hrt = np.array(
            [
                [
                    entity_to_idx[h],
                    relation_to_idx[data["type"]],
                    entity_to_idx[t],
                ]
                for h, t, data in edges
            ],
            dtype=np.int64,
        )

        # Split into train/valid/test (for demonstration, using all as train)
        num_edges = len(train_hrt)  # TODO: Change
        valid_split = int(num_edges * 0.1)
        test_split = int(num_edges * 0.2)

        valid_hrt = train_hrt[-test_split:-valid_split]
        test_hrt = train_hrt[-valid_split:]
        train_hrt = train_hrt[:-test_split]

        # Create LCCSR matrices and degrees for each split
        splits = {
            "train": train_hrt,
            "valid": (
                valid_hrt
                if transfer_setting
                else np.concatenate([train_hrt, valid_hrt])
            ),
            "test": (
                test_hrt
                if transfer_setting
                else np.concatenate([train_hrt, valid_hrt, test_hrt])
            ),
        }

        processed_data = {}
        num_entities = len(entity_to_idx)
        num_relations = len(relation_to_idx)

        for stage, hrt_group in splits.items():
            # Create LCCSR matrices
            rel_lccsr, edge_lccsr, degrees, indegrees, outdegrees = (
                StATIKAdapter._create_lccsr(
                    num_entities=num_entities,
                    num_edges=len(hrt_group),
                    hrt=hrt_group,
                    num_relations=num_relations,
                )
            )

            if save_dir:
                save_path = Path(save_dir)  # / stage
                save_path.mkdir(parents=True, exist_ok=True)

                rel_lccsr.save(str(save_path / f"{stage}_rel_lccsr.npz"))
                edge_lccsr.save(str(save_path / f"{stage}_edge_lccsr.npz"))
                np.save(str(save_path / f"{stage}_degrees.npy"), degrees)
                np.save(str(save_path / f"{stage}_indegrees.npy"), indegrees)
                np.save(str(save_path / f"{stage}_outdegrees.npy"), outdegrees)

            processed_data[stage] = {
                "edge_lccsr": edge_lccsr,
                "relation_lccsr": rel_lccsr,
                "degrees": degrees,
                "indegrees": indegrees,
                "outdegrees": outdegrees,
                "hrt": hrt_group,
            }

        # Create entity features (placeholder - you might want to customize this)
        entity_feat = np.zeros(
            (num_entities, 768), dtype=np.float32
        )  # Using 768 as example dimension
        relation_feat = np.zeros((num_relations, 768), dtype=np.float32)

        # Create text representations
        entity_text = np.array(
            [
                str(node)
                for node in sorted(entity_to_idx.keys(), key=lambda x: entity_to_idx[x])
            ]
        )
        relation_text = np.array(
            sorted(relation_to_idx.keys(), key=lambda x: relation_to_idx[x])
        )

        # Create entity targets
        train_entities = np.unique(train_hrt[:, [0, 2]])
        valid_entities = np.unique(valid_hrt[:, [0, 2]])
        test_entities = np.unique(test_hrt[:, [0, 2]])

        train_targets = np.zeros(num_entities, dtype=bool)
        train_targets[train_entities] = True

        valid_targets = np.zeros(num_entities, dtype=bool)
        valid_targets[valid_entities] = True
        if not transfer_setting:
            valid_targets[train_entities] = True

        test_targets = np.zeros(num_entities, dtype=bool)
        test_targets[test_entities] = True
        if not transfer_setting:
            test_targets[train_entities] = True
            test_targets[valid_entities] = True

        if save_dir:
            save_path = Path(save_dir)
            np.save(str(save_path / "train_targets.npy"), train_targets)
            np.save(str(save_path / "valid_targets.npy"), valid_targets)
            np.save(str(save_path / "test_targets.npy"), test_targets)
            np.save(str(save_path / "train_entities.npy"), train_entities)
            np.save(str(save_path / "valid_entities.npy"), valid_entities)
            np.save(str(save_path / "test_entities.npy"), test_entities)

        # Create filters
        filters = {}
        for stage, data in processed_data.items():
            edge_lccsr = data["edge_lccsr"]
            rel_lccsr = data["relation_lccsr"]
            triples = data["hrt"]
            targets = locals()[f"{stage}_targets"]

            h_filter = StATIKAdapter._create_filtered_candidates(
                triples[:, [2, 1]],
                edge_lccsr,
                rel_lccsr,
                num_entities,
                num_relations,
                head_pred=True,
            )
            t_filter = StATIKAdapter._create_filtered_candidates(
                triples[:, [0, 1]],
                edge_lccsr,
                rel_lccsr,
                num_entities,
                num_relations,
                head_pred=False,
            )

            h_filter = h_filter[:, targets]
            t_filter = t_filter[:, targets]

            if save_dir:
                np.save(str(save_path / f"{stage}_h_filter.npy"), h_filter)
                np.save(str(save_path / f"{stage}_t_filter.npy"), t_filter)

            filters[stage] = (h_filter, t_filter)

        # Create and return the processed dataset
        dataset = KGProcessedDataset.__new__(KGProcessedDataset)

        # Set basic attributes
        dataset.num_entities = num_entities
        dataset.num_relations = num_relations
        dataset.entity_text = entity_text
        dataset.relation_text = relation_text

        # Default embedding is bert-base-cased
        if embedding_model_name is not None:
            dataset.get_entity_features(embedding_model_name=embedding_model_name)
        else:
            dataset.entity_feat = entity_feat
            dataset.relation_feat = relation_feat

        dataset.feature_dim = entity_feat.shape[1]

        # Set split-specific data
        for stage in ["train", "valid", "test"]:
            data = processed_data[stage]
            setattr(dataset, f"{stage}_hrt", data["hrt"])
            setattr(dataset, f"{stage}_edge_lccsr", data["edge_lccsr"])
            setattr(dataset, f"{stage}_relation_lccsr", data["relation_lccsr"])
            setattr(dataset, f"{stage}_degrees", data["degrees"])
            setattr(dataset, f"{stage}_indegrees", data["indegrees"])
            setattr(dataset, f"{stage}_outdegrees", data["outdegrees"])

            h_filter, t_filter = filters[stage]
            setattr(dataset, f"{stage}_h_filter", h_filter)
            setattr(dataset, f"{stage}_t_filter", t_filter)

        # Set target information
        dataset.train_targets = train_targets
        dataset.valid_targets = valid_targets
        dataset.test_targets = test_targets
        dataset.train_entities = train_entities
        dataset.valid_entities = valid_entities
        dataset.test_entities = test_entities

        np.save(str(save_path / "dataset.npy"), dataset)

        return dataset

    @staticmethod
    def _create_lccsr(
        num_entities: int, num_edges: int, hrt: np.ndarray, num_relations: int
    ) -> tuple[
        LeftContiguousCSR,
        LeftContiguousCSR,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Create LCCSR matrices and degree information."""
        # Create dictionaries for edges and relations
        edge_dict: dict[Any, Any] = defaultdict(lambda: array.array("i"))
        relation_dict: dict[Any, Any] = defaultdict(lambda: array.array("i"))

        # Initialize degree arrays
        degrees = np.zeros(num_entities, dtype=np.int32)
        indegrees = np.zeros(num_entities, dtype=np.int32)
        outdegrees = np.zeros(num_entities, dtype=np.int32)

        # Build dictionaries
        for _hrt in hrt:
            h, r, t = map(int, _hrt)
            r_inv = r + num_relations  # Inverse relation ID

            # Forward edge
            edge_dict[h].append(t)
            relation_dict[h].append(r)
            degrees[h] += 1
            outdegrees[h] += 1

            # Inverse edge
            edge_dict[t].append(h)
            relation_dict[t].append(r_inv)
            degrees[t] += 1
            indegrees[t] += 1

        # Create CSR arrays
        edge_csr_data = np.zeros(2 * num_edges, dtype=np.int32)
        edge_csr_indptr = np.zeros(num_entities + 1, dtype=np.int32)
        rel_csr_data = np.zeros(2 * num_edges, dtype=np.int16)
        rel_csr_indptr = np.zeros(num_entities + 1, dtype=np.int32)

        # Fill CSR arrays
        num_prev = 0
        for i in range(num_entities):
            deg = degrees[i]
            edge_csr_indptr[i] = num_prev
            rel_csr_indptr[i] = num_prev

            if deg > 0:
                edge_csr_data[num_prev : num_prev + deg] = np.array(
                    edge_dict[i], dtype=np.int32
                )
                rel_csr_data[num_prev : num_prev + deg] = np.array(
                    relation_dict[i], dtype=np.int16
                )

            num_prev += deg

        edge_csr_indptr[-1] = num_prev
        rel_csr_indptr[-1] = num_prev

        # Create LCCSR objects
        rel_lccsr = LeftContiguousCSR(rel_csr_indptr, degrees, rel_csr_data)
        edge_lccsr = LeftContiguousCSR(edge_csr_indptr, degrees, edge_csr_data)

        return rel_lccsr, edge_lccsr, degrees, indegrees, outdegrees

    @staticmethod
    def _create_filtered_candidates(
        queries: np.ndarray,
        edge_lccsr: LeftContiguousCSR,
        relation_lccsr: LeftContiguousCSR,
        num_entities: int,
        num_relations: int,
        head_pred: bool = True,
    ) -> np.ndarray:
        """Create filtered candidates for evaluation."""
        candidate_filter = np.ones((queries.shape[0], num_entities), dtype=bool)

        for i in range(queries.shape[0]):
            s, r = queries[i]
            if head_pred:
                r = r + num_relations
            to_mask = edge_lccsr[s][relation_lccsr[s] == r]
            candidate_filter[i, to_mask] = False

        return candidate_filter
