import os
import random
from typing import Optional


class DuetGraphAdapter:
    @staticmethod
    def _make_mappings_from_graph(
        kg, sort_entities: bool = True, sort_relations: bool = True
    ):
        # entities
        entities = [n.name for n in kg.nodes()]
        if sort_entities:
            entities = sorted(set(entities))
        else:
            entities = list(dict.fromkeys(entities))  # stable order, unique
        entity2id = {e: i for i, e in enumerate(entities)}
        # relations (including inverse required by loaders)
        rels = (
            sorted(kg.schema.get_edge_types())
            if sort_relations
            else list(kg.schema.get_edge_types())
        )
        relations = []
        for r in rels:
            relations.append(r)
            relations.append("-" + r)
        relation2id = {r: i for i, r in enumerate(relations)}
        return entity2id, relation2id

    @staticmethod
    def _encode_edge_triplet(h_name, r_name, t_name, entity2id, relation2id):
        h = entity2id[h_name]
        t = entity2id[t_name]
        r = relation2id[r_name]
        rev_r_name = "-" + r_name
        rev_r = relation2id[rev_r_name]
        return [(h, r, t), (t, rev_r, h)]

    @staticmethod
    def _write_entities_relations(base_dir, entity2id, relation2id):
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, "entities.txt"), "w", encoding="utf-8") as f:
            f.writelines(
                f"{e}\t{i}\n" for e, i in sorted(entity2id.items(), key=lambda x: x[1])
            )
        with open(os.path.join(base_dir, "relations.txt"), "w", encoding="utf-8") as f:
            f.writelines(
                f"{r}\t{i}\n"
                for r, i in sorted(relation2id.items(), key=lambda x: x[1])
            )

    @staticmethod
    def _write_triplets_names(base_dir, filename, triplets):
        # triplets are tuples of (head_name, relation_name, tail_name)
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, filename), "w", encoding="utf-8") as f:
            f.writelines(f"{h}\t{r}\t{t}\n" for h, r, t in triplets)

    @staticmethod
    def export_transductive(
        kg: "KnowledgeGraph",
        out_dir: str,
        split_mode: str = "random",
        split_attr: str = "split",
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        default_split_if_missing: str | None = "train",
        sort_entities: bool = True,
        sort_relations: bool = True,
        seed: int | None = None,
        show_progress: bool = True,
    ):
        if seed is not None:
            random.seed(seed)

        if show_progress:
            print("Building mappings...")
        entity2id, relation2id = DuetGraphAdapter._make_mappings_from_graph(
            kg, sort_entities=sort_entities, sort_relations=sort_relations
        )
        # Write entities/relations now
        DuetGraphAdapter._write_entities_relations(out_dir, entity2id, relation2id)

        # Collect labeled edges
        labeled_edges = []  # (h_name, r_name, t_name, split)
        for h, t, data in kg.edges(data=True):
            r = data["type"]
            if split_mode.lower() == "attribute":
                lab = data.get(split_attr, default_split_if_missing)
                if lab is None:
                    raise ValueError(
                        f"Edge ({h.name},{t.name},{r}) missing split '{split_attr}'"
                    )
                lab = str(lab).lower()
                if lab == "val":
                    lab = "valid"
                if lab not in {"train", "valid", "test"}:
                    lab = default_split_if_missing
                labeled_edges.append((h.name, r, t.name, lab))
            else:
                labeled_edges.append((h.name, r, t.name, None))

        # Random split if needed
        if split_mode.lower() == "random":
            idxs = list(range(len(labeled_edges)))
            random.shuffle(idxs)
            n = len(idxs)
            n_tr = int(n * split_ratios[0])
            n_va = int(n * split_ratios[1])
            n_te = int(n * split_ratios[2])
            excess = n_tr + n_va + n_te - n
            if excess > 0:
                n_te -= excess
            train_idx = set(idxs[:n_tr])
            valid_idx = set(idxs[n_tr : n_tr + n_va])
            test_idx = set(idxs[n_tr + n_va : n_tr + n_va + n_te])
            named_train, named_valid, named_test = [], [], []
            for i, (h, r, t, _) in enumerate(labeled_edges):
                if i in train_idx:
                    named_train.append((h, r, t))
                elif i in valid_idx:
                    named_valid.append((h, r, t))
                elif i in test_idx:
                    named_test.append((h, r, t))
        else:
            named_train = [(h, r, t) for (h, r, t, s) in labeled_edges if s == "train"]
            named_valid = [(h, r, t) for (h, r, t, s) in labeled_edges if s == "valid"]
            named_test = [(h, r, t) for (h, r, t, s) in labeled_edges if s == "test"]

        # Write name triples (no inverse here; loader will add inverse)
        DuetGraphAdapter._write_triplets_names(out_dir, "train.txt", named_train)
        DuetGraphAdapter._write_triplets_names(out_dir, "valid.txt", named_valid)
        DuetGraphAdapter._write_triplets_names(out_dir, "test.txt", named_test)
        if show_progress:
            print("Transductive export complete.")

    @staticmethod
    def export_inductive(
        base_kg: Optional["KnowledgeGraph"] = None,
        ind_kg: Optional["KnowledgeGraph"] = None,
        out_dir: str = "./data",
        ind_out_dir: str | None = None,
        # Option A: provide two graphs (base_kg and ind_kg). If provided, we ignore attribute inference.
        # Option B: provide one graph and infer inductive sets via attributes:
        infer_from_single_kg: bool = False,
        single_kg: Optional["KnowledgeGraph"] = None,
        inductive_entity_attr: str = "inductive",  # node bool attr marking inductive entity set
        trans_split_attr: str = "split",  # on edges in base set: "train"/"valid"/"test"
        ind_split_attr: str = "ind_split",  # on edges in inductive set: "train"/"valid"/"test"
        default_trans_split: str | None = "train",
        default_ind_split: str | None = "train",
        sort_entities: bool = True,
        sort_relations: bool = True,
        show_progress: bool = True,
    ):
        """Writes files for InductiveKnowledgeGraph:
          - out_dir: entities.txt (train entities), relations.txt (shared), train.txt, valid.txt, test.txt
          - ind_out_dir: entities.txt (inductive test entities), train.txt, valid.txt, test.txt
        Behavior mirrors your InductiveKnowledgeGraph loader:
          - Base path relations.txt must contain both r and -r.
          - Base path entities.txt lists training entity space.
          - ind_out_dir entities.txt lists inductive test entity space (disjoint from base).
          - Base train/valid/test edges are encoded into base path (used for train/valid; test unused by your class but we write for completeness).
          - Inductive train/valid/test edges encoded into ind_out_dir; test edges are used as test_triplets by your class, and ind train edges form test_graph.
        """
        if ind_out_dir is None:
            ind_out_dir = out_dir + "_ind"

        # Build from two graphs explicitly
        if not infer_from_single_kg:
            if base_kg is None or ind_kg is None:
                raise ValueError(
                    "Provide both base_kg and ind_kg when infer_from_single_kg=False"
                )

            if show_progress:
                print("Building mappings from base graph (relations shared)...")
            # Relations come from base_kg schema; must include inverse names with '-'
            base_entity2id, relation2id = DuetGraphAdapter._make_mappings_from_graph(
                base_kg, sort_entities=sort_entities, sort_relations=sort_relations
            )
            # For inductive entity space, use entities from ind_kg
            ind_entities = [n.name for n in ind_kg.nodes()]
            ind_entities = (
                sorted(set(ind_entities))
                if sort_entities
                else list(dict.fromkeys(ind_entities))
            )
            ind_entity2id = {e: i for i, e in enumerate(ind_entities)}

            # Write entities/relations for base path
            DuetGraphAdapter._write_entities_relations(
                out_dir, base_entity2id, relation2id
            )

            # Write entities for inductive path (no relations.txt needed; loader reads relations from base path)
            os.makedirs(ind_out_dir, exist_ok=True)
            with open(
                os.path.join(ind_out_dir, "entities.txt"), "w", encoding="utf-8"
            ) as f:
                f.writelines(
                    f"{e}\t{i}\n"
                    for e, i in sorted(ind_entity2id.items(), key=lambda x: x[1])
                )

            # Collect and encode edges from base_kg by attribute split if present, otherwise put all in train
            def collect_named_by_attr(kg, split_attr, default_split):
                splits = {"train": [], "valid": [], "test": []}
                for h, t, data in kg.edges(data=True):
                    r = data["type"]
                    lab = data.get(split_attr, default_split)
                    if lab is None:
                        raise ValueError(
                            f"Edge ({h.name},{t.name},{r}) missing split '{split_attr}'"
                        )
                    lab = str(lab).lower()
                    if lab == "val":
                        lab = "valid"
                    if lab not in {"train", "valid", "test"}:
                        lab = default_split
                    splits[lab].append((h.name, r, t.name))
                return splits

            base_splits = collect_named_by_attr(
                base_kg, trans_split_attr, default_trans_split
            )
            ind_splits = collect_named_by_attr(
                ind_kg, ind_split_attr, default_ind_split
            )

            # Write base path triplets
            DuetGraphAdapter._write_triplets(out_dir, "train.txt", base_splits["train"])
            DuetGraphAdapter._write_triplets(out_dir, "valid.txt", base_splits["valid"])
            DuetGraphAdapter._write_triplets(out_dir, "test.txt", base_splits["test"])

            # Write inductive path triplets
            DuetGraphAdapter._write_triplets(
                ind_out_dir, "train.txt", ind_splits["train"]
            )
            DuetGraphAdapter._write_triplets(
                ind_out_dir, "valid.txt", ind_splits["valid"]
            )
            DuetGraphAdapter._write_triplets(
                ind_out_dir, "test.txt", ind_splits["test"]
            )

            if show_progress:
                print(
                    f"Inductive export complete. Base at {out_dir}, inductive at {ind_out_dir}"
                )
            return

        # Build from a single graph, inferring inductive vs transductive spaces
        if single_kg is None:
            raise ValueError("infer_from_single_kg=True requires single_kg")

        if show_progress:
            print("Inferring entity partitions from node attribute...")
        # Partition entities
        trans_entities, ind_entities = [], []
        for n in single_kg.nodes():
            is_ind = (
                bool(getattr(n, "attributes", {}).get(inductive_entity_attr, False))
                if hasattr(n, "attributes")
                else bool(getattr(n, inductive_entity_attr, False))
            )
            if is_ind:
                ind_entities.append(n.name)
            else:
                trans_entities.append(n.name)
        # Ensure disjoint sets
        overlap = set(trans_entities) & set(ind_entities)
        if overlap:
            raise ValueError(
                f"Entities cannot be both transductive and inductive: {overlap}"
            )

        # Build mappings
        trans_entities = (
            sorted(set(trans_entities))
            if sort_entities
            else list(dict.fromkeys(trans_entities))
        )
        ind_entities = (
            sorted(set(ind_entities))
            if sort_entities
            else list(dict.fromkeys(ind_entities))
        )
        base_entity2id = {e: i for i, e in enumerate(trans_entities)}
        ind_entity2id = {e: i for i, e in enumerate(ind_entities)}
        # Relations from schema with inverse
        rels = (
            sorted(single_kg.schema.get_edge_types())
            if sort_relations
            else list(single_kg.schema.get_edge_types())
        )
        relations = []
        for r in rels:
            relations.append(r)
            relations.append("-" + r)
        relation2id = {r: i for i, r in enumerate(relations)}

        # Write entities/relations
        DuetGraphAdapter._write_entities_relations(out_dir, base_entity2id, relation2id)
        os.makedirs(ind_out_dir, exist_ok=True)
        with open(
            os.path.join(ind_out_dir, "entities.txt"), "w", encoding="utf-8"
        ) as f:
            f.writelines(
                f"{e}\t{i}\n"
                for e, i in sorted(ind_entity2id.items(), key=lambda x: x[1])
            )

        # Split edges into base or inductive domain based on head/tail entity membership
        if show_progress:
            print("Splitting edges between transductive and inductive domains...")
        base_edges = []
        ind_edges = []
        for h, t, data in single_kg.edges(data=True):
            r_name = data["type"]
            h_name, t_name = h.name, t.name
            in_base = (h_name in base_entity2id) and (t_name in base_entity2id)
            in_ind = (h_name in ind_entity2id) and (t_name in ind_entity2id)
            if in_base and in_ind:
                raise ValueError(
                    f"Edge ({h_name},{t_name}) appears in both partitions."
                )
            if in_base:
                lab_raw = data.get(trans_split_attr, default_trans_split)
                lab = "valid" if str(lab_raw).lower() == "val" else str(lab_raw).lower()
                if lab not in {"train", "valid", "test"}:
                    lab = default_trans_split
                if lab is None:
                    raise ValueError(f"Base edge missing split {trans_split_attr}")
                enc = DuetGraphAdapter._encode_edge_triplet(
                    h_name, r_name, t_name, base_entity2id, relation2id
                )
                base_edges.append((lab, enc))
            elif in_ind:
                lab_raw = data.get(ind_split_attr, default_ind_split)
                lab = "valid" if str(lab_raw).lower() == "val" else str(lab_raw).lower()
                if lab not in {"train", "valid", "test"}:
                    lab = default_ind_split
                if lab is None:
                    raise ValueError(f"Ind edge missing split {ind_split_attr}")
                enc = DuetGraphAdapter._encode_edge_triplet(
                    h_name, r_name, t_name, ind_entity2id, relation2id
                )
                ind_edges.append((lab, enc))
            else:
                # Edge between partitions; skip or raise
                raise ValueError(
                    f"Edge ({h_name},{t_name}) connects transductive and inductive entities; not allowed in this scheme."
                )

        # Aggregate and write
        base_splits = {"train": [], "valid": [], "test": []}
        for lab, enc in base_edges:
            base_splits[lab].extend(enc)
        ind_splits = {"train": [], "valid": [], "test": []}
        for lab, enc in ind_edges:
            ind_splits[lab].extend(enc)

        DuetGraphAdapter._write_triplets(out_dir, "train.txt", base_splits["train"])
        DuetGraphAdapter._write_triplets(out_dir, "valid.txt", base_splits["valid"])
        DuetGraphAdapter._write_triplets(out_dir, "test.txt", base_splits["test"])

        DuetGraphAdapter._write_triplets(ind_out_dir, "train.txt", ind_splits["train"])
        DuetGraphAdapter._write_triplets(ind_out_dir, "valid.txt", ind_splits["valid"])
        DuetGraphAdapter._write_triplets(ind_out_dir, "test.txt", ind_splits["test"])

        if show_progress:
            print(
                f"Inductive export complete. Base at {out_dir}, inductive at {ind_out_dir}"
            )
