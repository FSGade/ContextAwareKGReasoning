"""Load iKraph as a KnowledgeGraph object."""

import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

from knowledge_graph import KnowledgeGraph, print_kg_stats


def load_nodes(filepath: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load node information from NER_ID_dict_cap_final.json.

    Returns:
        Dictionary mapping biokdeid to node information
    """
    print("Loading node information...")
    with open(filepath, encoding="utf-8") as f:
        nodes = json.load(f)

    # Create mapping from biokdeid to node info
    node_map = {}
    for node in tqdm(nodes, desc="Processing nodes", total=len(nodes)):
        node_map[node["biokdeid"]] = node

    return node_map


def load_relation_schema(filepath: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load relation schema from RelTypeInt.json.

    Returns:
        Dictionary mapping relation type to schema information
    """
    print("Loading relation schema...")
    with open(filepath, encoding="utf-8") as f:
        relations = json.load(f)

    # Create mapping from intRep to relation info
    rel_map = {}
    for rel in relations:
        rel_map[rel["intRep"]] = rel

    return rel_map


def parse_relation_id(rel_id: str) -> Dict[str, str]:
    """Parse a relation ID string into its components.

    Args:
        rel_id: String in format Node1ID.Node2ID.RelationID.CorrelationID.Direction.Source

    Returns:
        Dictionary of parsed components
    """
    parts = rel_id.split(".")
    return {
        "node1_id": parts[0],
        "node2_id": parts[1],
        "relation_id": parts[2],
        "correlation_id": parts[3],
        "direction": parts[4],
        "source": parts[5],
    }


def load_ikraph(base_path: str | Path) -> KnowledgeGraph:
    """Load iKraph knowledge graph from JSON files.

    Args:
        base_path: Path to directory containing the JSON files

    Returns:
        KnowledgeGraph object containing the iKraph data
    """
    # Initialize the KnowledgeGraph
    graph = KnowledgeGraph()

    # Load nodes
    node_map = load_nodes(Path(base_path) / "NER_ID_dict_cap_final.json")

    # Load relation schema
    rel_schema = load_relation_schema(Path(base_path) / "RelTypeInt.json")

    # Add nodes to graph
    print("Adding nodes to graph...")
    for biokdeid, node_info in tqdm(
        node_map.items(), desc="Adding nodes", total=len(node_map)
    ):
        # Create node tuple (name, type)
        node_tuple = (node_info["official name"], node_info["type"])

        # Prepare node attributes
        node_attrs = {
            "id": biokdeid,
            "kind": node_info["type"],
            "name": node_info["official name"],
            "common_name": node_info["common name"],
            "subtype": node_info["subtype"],
            "external_id": node_info["id"]#,
            #"species": node_info["species"]
        }

        graph.add_node(node_tuple, **node_attrs)

    # Load and add PubMed relationships
    print("Loading PubMed relationships...")
    with open(Path(base_path) / "PubMedList.json") as f:
        pubmed_rels = json.load(f)

    for pubmed_rel in tqdm(
        pubmed_rels, desc="Adding PubMed edges", total=len(pubmed_rels)
    ):
        rel_id = pubmed_rel["id"]
        evidence_list = pubmed_rel["list"]

        rel_info = parse_relation_id(rel_id)

        # Get nodes
        if (
            rel_info["node1_id"] not in node_map
            or rel_info["node2_id"] not in node_map
        ):
            continue

        source_info = node_map[rel_info["node1_id"]]
        target_info = node_map[rel_info["node2_id"]]

        source = (source_info["official name"], source_info["type"])
        target = (target_info["official name"], target_info["type"])

        direction = rel_info["direction"]

        source_species = source_info["species"]
        target_species = target_info["species"]

        if (
            source_species[0] != target_species[0]
            and source_species[0] == "NA"
        ):
            species = target_species
        else:
            species = source_species

        species_id = species[0]
        species_name = species[1]

        if direction == "21":
            source, target = target, source
            direction = "1"
        elif direction == "12":
            direction = "1"

        correlation_type = int(rel_info["correlation_id"]) - 1

        # Add edge for each piece of evidence
        for evidence in evidence_list:
            score, doc_id, probability, novelty = evidence

            edge_attrs = {
                "type": rel_schema[rel_info["relation_id"]]["relType"],
                "kind": rel_schema[rel_info["relation_id"]]["relType"],
                "source": "PubMed",
                "correlation_type": correlation_type,
                "direction": direction,
                "score": score,
                "document_id": doc_id,
                "probability": probability,
                "novelty": bool(novelty),
                "relation_id": rel_id,
                "species_id": species_id,
                "species": species_name,
            }

            graph.add_edge(source, target, **edge_attrs)

    # Load and add database relationships
    print("Loading database relationships...")
    with open(Path(base_path) / "DBRelations.json") as f:
        db_rels = json.load(f)

    print("Adding database relationships...")
    for rel in tqdm(db_rels, desc="Adding DB edges", total=len(db_rels)):
        # Get nodes
        if (
            rel["node_one_id"] not in node_map
            or rel["node_two_id"] not in node_map
        ):
            continue

        source_info = node_map[rel["node_one_id"]]
        target_info = node_map[rel["node_two_id"]]

        source = (source_info["official name"], source_info["type"])
        target = (target_info["official name"], target_info["type"])

        direction = rel_info["direction"]

        source_species = source_info["species"]
        target_species = target_info["species"]

        if (
            source_species[0] != target_species[0]
            and source_species[0] == "NA"
        ):
            species = target_species
        else:
            species = source_species

        species_id = species[0]
        species_name = species[1]

        if direction == "21":
            source, target = target, source
            direction = "1"
        elif direction == "12":
            direction = "1"

        correlation_type = int(rel_info["correlation_id"]) - 1

        edge_attrs = {
            "type": rel_schema[rel["relationship_type"]]["relType"],
            "kind": rel_schema[rel["relationship_type"]]["relType"],
            "source": rel["source"],
            "correlation_type": correlation_type,
            "direction": direction,
            "probability": rel["prob"],
            "score": rel["score"],
            "relation_id": rel["relID"],
            "species_id": species_id,
            "species": species_name,
        }

        graph.add_edge(source, target, **edge_attrs)

    return graph


# Example usage
if __name__ == "__main__":
    base_path = Path("/home/projects2/ContextAwareKGReasoning/data")
    raw_path = Path("raw/iKraph_full/")

    print("Starting iKraph loading process...")
    kg = load_ikraph(base_path / raw_path)

    print("Exporting..")
    kg.export_graph(base_path / "graphs/ikraph.pkl", file_format="pickle")

    print("Schema:")
    print(kg.schema)

    print("Summary:")
    print_kg_stats(kg)
