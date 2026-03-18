""" Find relevant nodes by keywords and export their IDs for subsetting. """

import sys
import csv
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_graph import KnowledgeGraph


def search_nodes(kg, node_type, keywords, exact_match=False):
    """Search for nodes by keywords in specific node type."""
    matches = []
    
    for node in kg.nodes():
        if node.type != node_type:
            continue
        
        name_lower = node.name.lower()
        
        for keyword in keywords:
            kw_lower = keyword.lower()
            
            if exact_match:
                pattern = r'(?:^|[\s\-_/,;()])' + re.escape(kw_lower) + r'(?:$|[\s\-_/,;()])'
                found = re.search(pattern, name_lower)
            else:
                found = kw_lower in name_lower
            
            if found:
                node_id = getattr(node, 'id', kg.nodes[node].get('id', ''))
                external_id = getattr(node, 'external_id', kg.nodes[node].get('external_id', ''))
                matches.append({
                    'name': node.name,
                    'type': node.type,
                    'node_id': node_id,
                    'external_id': external_id,
                    'degree': kg.degree(node),
                    'keyword': keyword
                })
                break
    
    return matches


def main():
    base_path = Path("/home/projects2/ContextAwareKGReasoning/data")
    input_graph = base_path / "graphs/subsets/ikraph_pubmed_human.pkl"
    output_dir = base_path / "graphs/exploration/pubmed_human"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load graph
    print("Loading graph...")
    kg = KnowledgeGraph.import_graph(str(input_graph))
    print(f"Loaded: {kg.number_of_nodes():,} nodes\n")
    
    # ========================================================================
    # SEARCH CONFIGURATION
    # Format: category -> (node_types, keywords, exact_match)
    # ========================================================================
    
    search_config = {
        'cytokines': (
            ['Gene'],
            ['IL1A', 'IL1B', 'IL6', 'IL12A', 'IL12B', 'IL17A', 'IL18', 'IL23A',
             'TNF', 'TNFA', 'IFNG', 'IL4', 'IL10', 'IL13', 'IL33', 'IL5',
             'TGFB1', 'TGFB', 'IL19'],
            True
        ),
        
        'chemokines': (
            ['Gene'],
            ['CCL2', 'CCL5', 'CCR2', 'CCR5', 'CXCL8', 'CXCL10', 'MCP1', 'MCP-1'],
            True
        ),
        
        'adipokines': (
            ['Gene'],
            ['LEP', 'ADIPOQ', 'RETN', 'NAMPT', 'omentin', 'apelin', 'vaspin',
             'FGF21', 'RBP4', 'SAA'],
            True
        ),
        
        'key_regulators': (
            ['Gene'],
            ['NLRP3', 'NFKB1', 'NFKBIA', 'RELA', 'NOS2', 'ARG1', 'ARG2',
             'SOCS1', 'SOCS3', 'HIF1A', 'CRP', 'PTGS2', 'PPARG'],
            True
        ),
        
        'inflammatory_pathways': (
            ['Pathway', 'Biological Process'],
            ['NF-kappa B', 'JAK-STAT', 'MAPK', 'inflammasome',
             'inflammatory response', 'macrophage activation', 'macrophage polarization'],
            False
        ),
        
        'metabolic_pathways': (
            ['Pathway', 'Biological Process'],
            ['adipogenesis', 'lipolysis', 'thermogenesis', 'fatty acid metabolism',
             'insulin signaling', 'glucose metabolism'],
            False
        ),
        
        'adipose_anatomy': (
            ['Anatomy'],
            ['adipose tissue', 'white adipose tissue', 'brown adipose tissue',
             'visceral adipose', 'subcutaneous adipose'],
            False
        ),
        
        'diseases': (
            ['Disease'],
            ['inflammation', 'obesity', 'insulin resistance', 'type 2 diabetes'],
            False
        ),
        
        'adipocytes': (
            ['CellLine', 'Cell Type', 'Anatomy'],
            ['adipocyte', 'white adipocyte', 'brown adipocyte', 'beige adipocyte',
             'preadipocyte'],
            False
        ),
        
        'immune_cells': (
            ['CellLine', 'Cell Type', 'Anatomy'],
            ['M1 macrophage', 'M2 macrophage', 'macrophage', 'Th1', 'Th2',
             'CD8 T cell', 'regulatory T cell', 'Treg', 'NK cell', 'neutrophil',
             'eosinophil', 'dendritic cell'],
            False
        ),
    }
    
    # Search
    results_by_category = defaultdict(list)
    
    for category, (node_types, keywords, exact) in search_config.items():
        print(f"Searching {category}...")
        
        for node_type in node_types:
            matches = search_nodes(kg, node_type, keywords, exact)
            results_by_category[category].extend(matches)
        
        print(f"  Found {len(results_by_category[category])} matches")
    
    # Save main CSV
    all_results = []
    for category, matches in results_by_category.items():
        for m in matches:
            all_results.append({
                'category': category,
                'node_type': m['type'],
                'node_name': m['name'],
                'node_id': m['node_id'],
                'external_id': m['external_id'],
                'degree': m['degree'],
                'keyword': m['keyword']
            })
    
    output_csv = output_dir / "exploration_results.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['category', 'node_type', 'node_name', 
                                                'node_id', 'external_id', 'degree', 'keyword'])
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: (x['category'], -x['degree'])):
            writer.writerow(r)
    
    print(f"\nSaved main results: {output_csv}")
    
    # Save per-category CSVs
    for category, matches in results_by_category.items():
        category_csv = output_dir / f"{category}.csv"
        with open(category_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['node_type', 'node_name',
                                                    'node_id', 'external_id', 'degree', 'keyword'])
            writer.writeheader()
            for m in sorted(matches, key=lambda x: -x['degree']):
                writer.writerow({
                    'node_type': m['type'],
                    'node_name': m['name'],
                    'node_id': m['node_id'],
                    'external_id': m['external_id'],
                    'degree': m['degree'],
                    'keyword': m['keyword']
                })
        print(f"  {category}.csv")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total matches: {len(all_results)}")
    print(f"Categories: {len(results_by_category)}")
    print()
    for category in sorted(results_by_category.keys()):
        matches = results_by_category[category]
        node_types = set(m['type'] for m in matches)
        print(f"  {category:.<30} {len(matches):>5} matches ({', '.join(node_types)})")

if __name__ == "__main__":
    main()