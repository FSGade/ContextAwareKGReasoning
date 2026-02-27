"""Filter iKraph to save PubMed and human edges."""

from pathlib import Path
from contextlib import redirect_stdout
from knowledge_graph import KnowledgeGraph, print_kg_stats
from knowledge_graph.utils.filtering import filter_graph

# Configuration
base_path = Path("/home/projects2/ContextAwareKGReasoning/data")
input_graph = base_path / "graphs/ikraph.pkl"
output_filtered = base_path / "graphs/subsets"
output_info = base_path / "graphs/info"

# If output directory doesn't exist, create it
output_info.mkdir(parents=True, exist_ok=True)
output_filtered.mkdir(parents=True, exist_ok=True)

# Load raw graph
print(f"Loading graph from {input_graph}...")
kg = KnowledgeGraph.import_graph(input_graph)

# Check n edges and nodes
print(f"Loaded: {len(kg.nodes):,} nodes and {len(kg.edges):,} edges")

# Filter to PubMed edges
pubmed = filter_graph(
    kg,
    filter_criterion=lambda d: (d.get("source") == "PubMed")
)

print(f"\nSaving PubMed-only graph to {output_filtered / 'ikraph_pubmed.pkl'}...")
pubmed.export_graph(output_filtered / "ikraph_pubmed.pkl")
print("Done!")

# Check n edges and nodes
print(f"PubMed graph: {len(pubmed.nodes):,} nodes and {len(pubmed.edges):,} edges")

# Filter to human edges
pubmed_human = filter_graph(
    pubmed,
    filter_criterion=lambda d: (d.get("species_id") == "NCBITaxID:9606")
)

print(f"\nSaving PubMed + Human graph to {output_filtered / 'ikraph_pubmed_human.pkl'}...")
pubmed_human.export_graph(output_filtered / "ikraph_pubmed_human.pkl")
print("Done!")


# Check n edges and nodes
print(f"PubMed + Human graph: {len(pubmed_human.nodes):,} nodes and {len(pubmed_human.edges):,} edges")


# Prints schemas to files
print(f"\nSaving graph schema to {output_info / 'ikraph_pubmed_schema.txt'}...")
with open(output_info / "ikraph_pubmed_schema.txt", "w") as f:
    f.write(str(pubmed.schema))
print("Done!")

print(f"\nSaving human graph schema to {output_info / 'ikraph_pubmed_human_schema.txt'}...")
with open(output_info / "ikraph_pubmed_human_schema.txt", "w") as f:
    f.write(str(pubmed_human.schema))
print("Done!")

# Print summary statistics to files
print(f"\nSaving PubMed stats to {output_info / 'ikraph_pubmed_stats.txt'}...")
with open(output_info / "ikraph_pubmed_stats.txt", "w") as f:
    with redirect_stdout(f):
        print_kg_stats(pubmed)
print("Done!")

print(f"\nSaving human stats to {output_info / 'ikraph_pubmed_human_stats.txt'}...")
with open(output_info / "ikraph_pubmed_human_stats.txt", "w") as f:
    with redirect_stdout(f):
        print_kg_stats(pubmed_human)
print("Done!")