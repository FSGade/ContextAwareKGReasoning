#!/usr/bin/env python3
"""
Normalize tissue, detailed_tissue, mechanisms, and pathways fields in the
knowledge graph.  Runs in a SINGLE PASS over all edges.

Updates BOTH:
1. Direct edge attributes: data['tissue'], data['detailed_tissue'],
   data['mechanisms'], data['pathways']
2. Context dictionary: data['context']['Tissue'], data['context']['Detailed_Tissue'],
   data['context']['Mechanisms'], data['context']['Pathways']

TISSUE FIELD STRATEGY:
- Split edges with multiple tissues into separate edges (one per tissue)
- Use broad categories (e.g., all adipose variants → "Adipose tissue")
- Each edge has exactly ONE tissue value

DETAILED_TISSUE FIELD STRATEGY:
- Keep specificity (e.g., "White adipose tissue", "Visceral adipose tissue")
- Only normalize format: remove abbreviations, fix case, parse lists
- Can have multiple values (stored as sorted list)

MECHANISMS / PATHWAYS FIELD STRATEGY:
- Split semicolon-delimited compound entries into individual terms
- Sentence-case each term (preserve ~150 known biomedical acronyms)
- Apply canonical mapping for pathways (merge NF-κB variants etc.)
- Sort terms alphabetically, deduplicate
- Rejoin with "; "

CONTEXT FIELD FORMAT NORMALIZATION:
- Convert all list-like strings (commas, semicolons, brackets) to
  consistent format
- Single values remain as strings; multiple values as sorted lists
- Applies to ALL context fields not handled by specific logic above

Usage:
    python normalize_graph.py <input_graph.pkl> [output_graph.pkl] [report.json]
"""

import sys
import re
import ast
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Union, Any, Tuple, Dict, Optional
from datetime import datetime
import copy

# Add project directory to path to import KnowledgeGraph
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_graph import KnowledgeGraph


# ============================================================================
# EMPTY / PLACEHOLDER SENTINELS
# ============================================================================
# Values that mean "no data" across the graph.  Used by tokenize_semicolon_field
# and the diagnostic counters.  We strip trailing periods before checking, so
# "not specified." is also caught.
EMPTY_SENTINELS = frozenset({
    '', 'none', 'not specified', 'nan', 'unknown', 'n/a', 'na',
    'not available', 'not applicable', 'null', 'unspecified',
})


# ============================================================================
# TISSUE FIELD: BROAD CATEGORY MAPPINGS
# ============================================================================

# Map all variants to broad tissue categories
TISSUE_BROAD_CATEGORIES = {
    # All adipose tissue variants → "Adipose tissue"
    'adipose tissue': 'adipose tissue',
    'adipose': 'adipose tissue',
    'fat tissue': 'adipose tissue',
    'fat': 'adipose tissue',
    'fatty tissue': 'adipose tissue',
    'white adipose tissue': 'adipose tissue',
    'brown adipose tissue': 'adipose tissue',
    'visceral adipose tissue': 'adipose tissue',
    'subcutaneous adipose tissue': 'adipose tissue',
    'visceral fat': 'adipose tissue',
    'subcutaneous fat': 'adipose tissue',
    'perivascular adipose tissue': 'adipose tissue',
    'epicardial adipose tissue': 'adipose tissue',
    'visceral white adipose tissue': 'adipose tissue',
    'subcutaneous white adipose tissue': 'adipose tissue',
    'breast adipose tissue': 'adipose tissue',
    'mammary gland white adipose tissue': 'adipose tissue',
    'perigonadal fat': 'adipose tissue',
    'epididymal fat': 'adipose tissue',
    'mesenteric fat': 'adipose tissue',
    'omental fat': 'adipose tissue',
    'epididymal adipose tissue': 'adipose tissue',
    'omental adipose tissue': 'adipose tissue',
    'mesenteric white adipose tissue': 'adipose tissue',
    'breast white adipose tissue': 'adipose tissue',
    'perirenal adipose tissue': 'adipose tissue',
    'retroperitoneal adipose tissue': 'adipose tissue',
    'interscapular brown adipose tissue': 'adipose tissue',
    'beige adipose tissue': 'adipose tissue',
    'brite adipose tissue': 'adipose tissue',

    # Liver variants → "Liver"
    'liver': 'liver', 'liver tissue': 'liver',
    'hepatic tissue': 'liver', 'hepatic': 'liver',

    # Brain variants → "Brain"
    'brain': 'brain', 'brain tissue': 'brain',
    'cerebral cortex': 'brain', 'hippocampus': 'brain',
    'hypothalamus': 'brain', 'cerebellum': 'brain',
    'frontal cortex': 'brain', 'prefrontal cortex': 'brain',
    'amygdala': 'brain', 'striatum': 'brain',
    'substantia nigra': 'brain', 'brainstem': 'brain',

    # Muscle variants
    'muscle': 'muscle', 'muscle tissue': 'muscle',
    'skeletal muscle': 'skeletal muscle',
    'skeletal muscle tissue': 'skeletal muscle',
    'gastrocnemius': 'skeletal muscle', 'quadriceps': 'skeletal muscle',
    'soleus': 'skeletal muscle', 'tibialis anterior': 'skeletal muscle',
    'cardiac muscle': 'cardiac muscle', 'cardiac muscle tissue': 'cardiac muscle',
    'smooth muscle': 'smooth muscle', 'smooth muscle tissue': 'smooth muscle',
    'vascular smooth muscle': 'smooth muscle',

    # Kidney variants → "Kidney"
    'kidney': 'kidney', 'kidney tissue': 'kidney',
    'renal tissue': 'kidney', 'renal': 'kidney',
    'renal cortex': 'kidney', 'renal medulla': 'kidney',

    # Heart variants → "Heart"
    'heart': 'heart', 'heart tissue': 'heart',
    'cardiac tissue': 'heart', 'myocardium': 'heart',
    'myocardial tissue': 'heart',
    'left ventricle': 'heart', 'right ventricle': 'heart',
    'atrium': 'heart', 'left atrium': 'heart', 'right atrium': 'heart',

    # Lung variants → "Lung"
    'lung': 'lung', 'lung tissue': 'lung',
    'pulmonary tissue': 'lung', 'pulmonary': 'lung',

    # Blood variants → "Blood"
    'blood': 'blood', 'plasma': 'blood', 'serum': 'blood',
    'peripheral blood': 'blood', 'blood plasma': 'blood',
    'blood serum': 'blood', 'whole blood': 'blood',

    # Pancreas variants → "Pancreas"
    'pancreas': 'pancreas', 'pancreatic tissue': 'pancreas',
    'pancreatic islets': 'pancreas', 'pancreatic islet': 'pancreas',
    'islets': 'pancreas', 'islet': 'pancreas',
    'islets of langerhans': 'pancreas',
    'beta cells': 'pancreas', 'pancreatic beta cells': 'pancreas',

    # Bone variants
    'bone': 'bone', 'bone tissue': 'bone',
    'bone marrow': 'bone marrow',

    # Intestine variants → "Intestine"
    'intestine': 'intestine', 'intestinal tissue': 'intestine',
    'small intestine': 'intestine', 'large intestine': 'intestine',
    'colon': 'intestine', 'colonic tissue': 'intestine',
    'ileum': 'intestine', 'jejunum': 'intestine',
    'duodenum': 'intestine', 'gut': 'intestine',

    # Spleen → "Spleen"
    'spleen': 'spleen', 'splenic tissue': 'spleen',

    # Nervous tissue
    'nervous tissue': 'nervous tissue', 'nerve tissue': 'nervous tissue',
    'neural tissue': 'nervous tissue', 'neuronal tissue': 'nervous tissue',

    # Vascular → "Vasculature"
    'vascular tissue': 'vasculature', 'vasculature': 'vasculature',
    'blood vessels': 'vasculature', 'blood vessel': 'vasculature',
    'endothelium': 'vasculature', 'endothelial tissue': 'vasculature',
    'vascular endothelium': 'vasculature',
    'aorta': 'vasculature', 'artery': 'vasculature', 'vein': 'vasculature',

    # Tumor → "Tumor"
    'tumor': 'tumor', 'tumor tissue': 'tumor',
    'tumour': 'tumor', 'tumour tissue': 'tumor',
    'cancer tissue': 'tumor', 'neoplasm': 'tumor',
    'neoplastic tissue': 'tumor',

    # Skin → "Skin"
    'skin': 'skin', 'skin tissue': 'skin',
    'dermis': 'skin', 'epidermis': 'skin', 'cutaneous': 'skin',

    # Lymph node
    'lymph node': 'lymph node', 'lymph nodes': 'lymph node',
    'lymphoid tissue': 'lymph node',

    # Other organs
    'thymus': 'thymus', 'thymic tissue': 'thymus',
    'prostate': 'prostate', 'prostate tissue': 'prostate',
    'prostatic tissue': 'prostate',
    'testis': 'testis', 'testes': 'testis', 'testicular tissue': 'testis',
    'ovary': 'ovary', 'ovarian tissue': 'ovary',
    'uterus': 'uterus', 'uterine tissue': 'uterus', 'endometrium': 'uterus',
    'breast': 'breast', 'breast tissue': 'breast',
    'mammary gland': 'breast', 'mammary tissue': 'breast',
    'stomach': 'stomach', 'gastric tissue': 'stomach',
    'thyroid': 'thyroid', 'thyroid gland': 'thyroid', 'thyroid tissue': 'thyroid',
    'adrenal gland': 'adrenal gland', 'adrenal': 'adrenal gland',
    'adrenal cortex': 'adrenal gland', 'adrenal medulla': 'adrenal gland',
    'placenta': 'placenta', 'placental tissue': 'placenta',
    'eye': 'eye', 'retina': 'eye', 'retinal tissue': 'eye',

    # Not specified
    'not specified': 'not specified', 'not specified.': 'not specified',
    'na': 'not specified', 'n/a': 'not specified',
    'unknown': 'not specified', '': 'not specified',
}


# ============================================================================
# DETAILED TISSUE: NORMALIZATION MAPPINGS
# ============================================================================

DETAILED_TISSUE_CANONICAL = {
    # Abbreviations
    'wat': 'White Adipose Tissue', 'bat': 'Brown Adipose Tissue',
    'vat': 'Visceral Adipose Tissue', 'sat': 'Subcutaneous Adipose Tissue',
    'ewat': 'Epididymal White Adipose Tissue',
    'iwat': 'Inguinal White Adipose Tissue',
    'gwat': 'Gonadal White Adipose Tissue',
    'mwat': 'Mesenteric White Adipose Tissue',
    'pvat': 'Perivascular Adipose Tissue', 'eat': 'Epicardial Adipose Tissue',

    # "Fat" synonyms
    'visceral fat': 'Visceral Adipose Tissue',
    'subcutaneous fat': 'Subcutaneous Adipose Tissue',
    'epididymal fat': 'Epididymal Adipose Tissue',
    'perigonadal fat': 'Perigonadal Adipose Tissue',
    'gonadal fat': 'Gonadal Adipose Tissue',
    'mesenteric fat': 'Mesenteric Adipose Tissue',
    'omental fat': 'Omental Adipose Tissue',
    'perirenal fat': 'Perirenal Adipose Tissue',
    'retroperitoneal fat': 'Retroperitoneal Adipose Tissue',
    'inguinal fat': 'Inguinal Adipose Tissue',
    'brown fat': 'Brown Adipose Tissue', 'white fat': 'White Adipose Tissue',
    'beige fat': 'Beige Adipose Tissue',
    'brite fat': 'Beige Adipose Tissue',
    'interscapular fat': 'Interscapular Brown Adipose Tissue',
    'epicardial fat': 'Epicardial Adipose Tissue',
    'pericardial fat': 'Pericardial Adipose Tissue',
    'perivascular fat': 'Perivascular Adipose Tissue',
    'abdominal fat': 'Abdominal Adipose Tissue',
    'belly fat': 'Abdominal Adipose Tissue',
    'breast fat': 'Breast Adipose Tissue',
    'mammary fat': 'Mammary Adipose Tissue',
    'mammary gland fat': 'Mammary Adipose Tissue',
    'fat tissue': 'Adipose Tissue', 'fatty tissue': 'Adipose Tissue',
    'adipose': 'Adipose Tissue', 'fat': 'Adipose Tissue',

    # Visceral/Subcutaneous variants
    'visceral adipose': 'Visceral Adipose Tissue',
    'visceral white adipose tissue': 'Visceral White Adipose Tissue',
    'visceral wat': 'Visceral White Adipose Tissue',
    'subcutaneous adipose': 'Subcutaneous Adipose Tissue',
    'subcutaneous white adipose tissue': 'Subcutaneous White Adipose Tissue',
    'subcutaneous wat': 'Subcutaneous White Adipose Tissue',
    'white adipose': 'White Adipose Tissue',
    'brown adipose': 'Brown Adipose Tissue',
    'interscapular brown adipose tissue': 'Interscapular Brown Adipose Tissue',
    'interscapular bat': 'Interscapular Brown Adipose Tissue',
    'beige adipose': 'Beige Adipose Tissue',
    'beige adipose tissue': 'Beige Adipose Tissue',
    'brite adipose': 'Beige Adipose Tissue',
    'brite adipose tissue': 'Beige Adipose Tissue',

    # Depot-specific variants
    'epididymal adipose': 'Epididymal Adipose Tissue',
    'epididymal white adipose tissue': 'Epididymal White Adipose Tissue',
    'inguinal adipose': 'Inguinal Adipose Tissue',
    'inguinal white adipose tissue': 'Inguinal White Adipose Tissue',
    'gonadal adipose': 'Gonadal Adipose Tissue',
    'gonadal white adipose tissue': 'Gonadal White Adipose Tissue',
    'perigonadal adipose': 'Perigonadal Adipose Tissue',
    'perigonadal adipose tissue': 'Perigonadal Adipose Tissue',
    'perigonadal white adipose tissue': 'Perigonadal White Adipose Tissue',
    'mesenteric adipose': 'Mesenteric Adipose Tissue',
    'mesenteric white adipose tissue': 'Mesenteric White Adipose Tissue',
    'omental adipose': 'Omental Adipose Tissue',
    'mammary adipose': 'Mammary Adipose Tissue',
    'mammary adipose tissue': 'Mammary Adipose Tissue',
    'mammary gland adipose': 'Mammary Adipose Tissue',
    'mammary gland adipose tissue': 'Mammary Adipose Tissue',
    'mammary gland white adipose tissue': 'Mammary White Adipose Tissue',
    'breast adipose': 'Breast Adipose Tissue',
    'breast white adipose tissue': 'Breast White Adipose Tissue',
    'abdominal subcutaneous adipose tissue': 'Subcutaneous Abdominal Adipose Tissue',
    'subcutaneous abdominal adipose tissue': 'Subcutaneous Abdominal Adipose Tissue',
    'abdominal subcutaneous fat': 'Subcutaneous Abdominal Adipose Tissue',
    'subcutaneous abdominal fat': 'Subcutaneous Abdominal Adipose Tissue',
    'abdominal subcutaneous': 'Subcutaneous Abdominal Adipose Tissue',
    'subcutaneous abdominal': 'Subcutaneous Abdominal Adipose Tissue',

    # Cell types
    'adipocytes': 'Adipocytes', 'adipocyte': 'Adipocytes',
    'white adipocytes': 'White Adipocytes', 'white adipocyte': 'White Adipocytes',
    'brown adipocytes': 'Brown Adipocytes', 'brown adipocyte': 'Brown Adipocytes',
    'beige adipocytes': 'Beige Adipocytes', 'beige adipocyte': 'Beige Adipocytes',
    'preadipocytes': 'Preadipocytes', 'preadipocyte': 'Preadipocytes',
    'pre-adipocytes': 'Preadipocytes', 'pre-adipocyte': 'Preadipocytes',
    'hepatocytes': 'Hepatocytes', 'hepatocyte': 'Hepatocytes',
    'macrophages': 'Macrophages', 'macrophage': 'Macrophages',
    'adipose tissue macrophages': 'Adipose Tissue Macrophages',
    'adipose tissue macrophage': 'Adipose Tissue Macrophages',
    'adipose macrophages': 'Adipose Tissue Macrophages',
    'adipose macrophage': 'Adipose Tissue Macrophages',

    # Blood
    'wbc': 'White Blood Cells', 'rbc': 'Red Blood Cells',
    'pbmc': 'Peripheral Blood Mononuclear Cells',
    'peripheral blood mononuclear cell': 'Peripheral Blood Mononuclear Cells',
    'peripheral blood mononuclear cells': 'Peripheral Blood Mononuclear Cells',
    'whole blood': 'Whole Blood', 'blood plasma': 'Plasma',
    'blood serum': 'Serum', 'peripheral blood': 'Peripheral Blood',

    # Breast/Mammary (non-adipose)
    'breast tissue': 'Breast', 'breast': 'Breast',
    'mammary gland': 'Mammary Gland', 'mammary tissue': 'Mammary Gland',

    # Nervous system
    'cns': 'Central Nervous System', 'pns': 'Peripheral Nervous System',
    'brain tissue': 'Brain', 'cerebral cortex': 'Cerebral Cortex',
    'prefrontal cortex': 'Prefrontal Cortex', 'frontal cortex': 'Frontal Cortex',

    # Gastrointestinal
    'gi tract': 'Gastrointestinal Tract', 'gi': 'Gastrointestinal Tract',
    'gastrointestinal': 'Gastrointestinal Tract',
    'small intestine': 'Small Intestine', 'large intestine': 'Large Intestine',
    'colon': 'Colon', 'colonic tissue': 'Colon',

    # Other organs
    'liver tissue': 'Liver', 'hepatic tissue': 'Liver', 'hepatic': 'Liver',
    'skeletal muscle tissue': 'Skeletal Muscle',
    'cardiac muscle tissue': 'Cardiac Muscle',
    'smooth muscle tissue': 'Smooth Muscle', 'heart muscle': 'Cardiac Muscle',
    'vascular smooth muscle': 'Vascular Smooth Muscle',
    'kidney tissue': 'Kidney', 'renal tissue': 'Kidney', 'renal': 'Kidney',
    'heart tissue': 'Heart', 'cardiac tissue': 'Heart',
    'myocardial tissue': 'Myocardium',
    'lung tissue': 'Lung', 'pulmonary tissue': 'Lung', 'pulmonary': 'Lung',
    'pancreatic tissue': 'Pancreas',
    'pancreatic islet': 'Pancreatic Islets', 'pancreatic islets': 'Pancreatic Islets',
    'islets of langerhans': 'Pancreatic Islets',
    'islet': 'Pancreatic Islets', 'islets': 'Pancreatic Islets',
    'beta cell': 'Pancreatic Beta Cells', 'beta cells': 'Pancreatic Beta Cells',
    'pancreatic beta cell': 'Pancreatic Beta Cells',
    'pancreatic beta cells': 'Pancreatic Beta Cells',
    'skin tissue': 'Skin', 'cutaneous': 'Skin', 'cutaneous tissue': 'Skin',
    'tumor tissue': 'Tumor', 'tumour': 'Tumor', 'tumour tissue': 'Tumor',
    'cancer tissue': 'Tumor', 'neoplasm': 'Tumor',
    'neoplastic tissue': 'Tumor',
    'vascular tissue': 'Vasculature',
    'blood vessel': 'Blood Vessels', 'blood vessels': 'Blood Vessels',
    'endothelial tissue': 'Endothelium',
    'vascular endothelium': 'Endothelium',
    'bone tissue': 'Bone', 'splenic tissue': 'Spleen',
    'thymic tissue': 'Thymus',
    'lymphoid tissue': 'Lymphoid Tissue', 'lymph nodes': 'Lymph Node',
}


# ============================================================================
# MECHANISMS / PATHWAYS: BIOMEDICAL ACRONYMS TO PRESERVE
# ============================================================================

PRESERVE_ACRONYMS = {
    # Signaling pathways / kinases
    'NF-κB', 'NF-kB', 'NFκB', 'NFkB', 'MAPK', 'ERK', 'JNK', 'PI3K', 'AKT',
    'mTOR', 'AMPK', 'JAK', 'STAT', 'JAK-STAT', 'JAK/STAT',
    'TGF-β', 'TGFβ', 'TGF-beta', 'SMAD', 'Wnt', 'Notch',
    'EGFR', 'VEGF', 'VEGFR', 'PDGF', 'PDGFR', 'FGFR', 'FGF',
    'HIF-1α', 'HIF1α', 'HIF-1a', 'HIF1A',
    'PKC', 'PKA', 'PKB', 'PLC', 'PLD',
    'Ras', 'Raf', 'MEK', 'SOS', 'GRB2',
    'IKK', 'IκB', 'IKKβ',
    'cAMP', 'cGMP', 'ATP', 'ADP', 'GTP', 'GDP',
    'PPAR', 'PPARα', 'PPARγ', 'PPARδ', 'PPAR-α', 'PPAR-γ', 'PPAR-δ',
    'LXR', 'RXR', 'FXR', 'PXR', 'CAR',
    'SIRT1', 'SIRT2', 'SIRT3', 'SIRT6',
    'CREB', 'AP-1', 'SP1',
    'SHP-1', 'SHP-2', 'SHP1', 'SHP2',
    'ROCK', 'RhoA', 'Rac1', 'Cdc42',
    # Inflammasome / immune
    'NLRP3', 'NLRP1', 'NLRC4', 'AIM2', 'ASC',
    'TLR', 'TLR2', 'TLR4', 'TLR7', 'TLR9',
    'MyD88', 'TRIF', 'TRAF6', 'IRAK',
    'CD4', 'CD8', 'CD36', 'CD68', 'CD163',
    'MHC', 'TCR', 'BCR',
    'Th1', 'Th2', 'Th17', 'Treg',
    # Cytokines / chemokines / adipokines
    'TNF', 'TNF-α', 'TNFα', 'IL-1', 'IL-1β', 'IL1β',
    'IL-2', 'IL-4', 'IL-6', 'IL-8', 'IL-10', 'IL-12', 'IL-13',
    'IL-17', 'IL-18', 'IL-23', 'IL-33',
    'IFN-γ', 'IFNγ', 'IFN-α', 'IFN-β',
    'CCL2', 'CCL5', 'CXCL8', 'CXCL10', 'MCP-1', 'MCP1',
    'MIP-1α', 'MIP-1β', 'RANTES',
    'CRP', 'SAA', 'PAI-1', 'PAI1',
    # Metabolic enzymes / molecules
    'LPL', 'HSL', 'ATGL', 'FASN', 'ACC', 'SCD1',
    'GLUT4', 'GLUT1', 'GLUT2',
    'IRS-1', 'IRS-2', 'IRS1', 'IRS2',
    'PEPCK', 'G6P', 'GLP-1', 'GLP1', 'GIP',
    'HDL', 'LDL', 'VLDL', 'TAG', 'FFA', 'NEFA',
    'CoA', 'CPT1', 'CPT2', 'UCP1', 'UCP2', 'UCP3',
    'PGC-1α', 'PGC1α', 'PGC-1a', 'PGC1A',
    'SREBP', 'SREBP-1', 'SREBP-1c', 'SREBP-2',
    'ChREBP', 'FOXO1', 'FOXO3',
    'HMG-CoA', 'PCSK9',
    'DPP-4', 'DPP4', 'SGLT2', 'SGLT1',
    'ROS', 'RNS', 'NO', 'NOS', 'iNOS', 'eNOS', 'nNOS',
    'SOD', 'SOD1', 'SOD2', 'GPx', 'CAT',
    'NAD+', 'NADH', 'NADPH', 'FAD',
    # Receptors
    'GR', 'AR', 'ER', 'ERα', 'ERβ',
    'RAGE', 'AGE', 'AGEs',
    'ANGPTL3', 'ANGPTL4', 'ANGPTL8',
    'ACE', 'ACE2', 'AT1R', 'AT2R', 'AGT',
    # Growth factors / other
    'IGF-1', 'IGF1', 'IGF-2', 'EGF', 'NGF', 'BDNF',
    'BMP', 'BMP2', 'BMP4', 'BMP7',
    'MMP', 'MMP2', 'MMP9', 'TIMP',
    'PCNA', 'Ki-67',
    'TP53', 'p53', 'RB', 'RB1', 'BRCA1', 'BRCA2',
    'miRNA', 'mRNA', 'siRNA', 'lncRNA', 'DNA', 'RNA',
    'SNP', 'GWAS', 'QTL', 'eQTL',
    # Cell cycle / apoptosis
    'CDK', 'CDK2', 'CDK4', 'CDK6',
    'Bcl-2', 'Bcl2', 'Bax', 'Bak', 'Bad',
    'Caspase-1', 'Caspase-3', 'Caspase-8', 'Caspase-9',
    # Renin-angiotensin system
    'RAS', 'RAAS',
    # Other
    'ECM', 'EMT', 'ER', 'UPR',
    'NAFLD', 'NASH', 'T2D', 'T2DM', 'CVD', 'CHD', 'CKD',
    'BMI', 'HbA1c', 'HOMA-IR', 'GI',
    'CNS', 'PNS', 'BBB', 'WAT', 'BAT', 'SAT', 'VAT',
}

# Case-insensitive lookup: exact forms take priority over no-hyphen variants
_ACRONYM_LOOKUP = {}
for _acr in PRESERVE_ACRONYMS:
    _no_hyphen = _acr.replace('-', '')
    if _no_hyphen.lower() != _acr.lower():
        _ACRONYM_LOOKUP[_no_hyphen.lower()] = _acr
for _acr in PRESERVE_ACRONYMS:
    _ACRONYM_LOOKUP[_acr.lower()] = _acr


# ============================================================================
# CANONICAL PATHWAY MAPPINGS
# ============================================================================

PATHWAY_CANONICAL = {
    # NF-κB family
    'nf-κb signaling pathway': 'NF-κB signaling pathway',
    'nf-κb signaling': 'NF-κB signaling pathway',
    'nf-κb pathway': 'NF-κB signaling pathway',
    'nf-κb': 'NF-κB signaling pathway',
    'nf-kb signaling pathway': 'NF-κB signaling pathway',
    'nf-kb signaling': 'NF-κB signaling pathway',
    'nf-kb pathway': 'NF-κB signaling pathway',
    'nf-kb': 'NF-κB signaling pathway',
    'nfκb signaling pathway': 'NF-κB signaling pathway',
    'nfκb signaling': 'NF-κB signaling pathway',
    'nfκb pathway': 'NF-κB signaling pathway',
    'nfκb': 'NF-κB signaling pathway',
    'nfkb signaling pathway': 'NF-κB signaling pathway',
    'nfkb signaling': 'NF-κB signaling pathway',
    'nfkb pathway': 'NF-κB signaling pathway',
    'nfkb': 'NF-κB signaling pathway',
    'nf-κb-mediated inflammation': 'NF-κB-mediated inflammation',
    'nfκb-mediated inflammation': 'NF-κB-mediated inflammation',
    'nf-kb-mediated inflammation': 'NF-κB-mediated inflammation',
    # Inflammatory signaling
    'inflammatory signaling': 'Inflammatory signaling',
    'inflammatory signaling pathways': 'Inflammatory signaling',
    'inflammatory signaling pathway': 'Inflammatory signaling',
    'inflammation signaling': 'Inflammatory signaling',
    'inflammatory pathway': 'Inflammatory signaling',
    'inflammatory pathways': 'Inflammatory signaling',
    # Renin-angiotensin system
    'renin-angiotensin system (ras)': 'Renin-angiotensin system',
    'renin-angiotensin system': 'Renin-angiotensin system',
    'renin\u2013angiotensin system': 'Renin-angiotensin system',
    'renin-angiotensin-aldosterone system': 'Renin-angiotensin-aldosterone system',
    'raas': 'Renin-angiotensin-aldosterone system',
    'ras pathway': 'Renin-angiotensin system',
    # PPAR family
    'ppar signaling pathway': 'PPAR signaling pathway',
    'ppar signaling': 'PPAR signaling pathway',
    'ppar pathway': 'PPAR signaling pathway',
    'pparγ signaling pathway': 'PPARγ signaling pathway',
    'pparγ signaling': 'PPARγ signaling pathway',
    'pparγ pathway': 'PPARγ signaling pathway',
    'ppargamma signaling pathway': 'PPARγ signaling pathway',
    'ppargamma signaling': 'PPARγ signaling pathway',
    'ppar-γ signaling pathway': 'PPARγ signaling pathway',
    'ppar-γ signaling': 'PPARγ signaling pathway',
    'ppar gamma signaling pathway': 'PPARγ signaling pathway',
    'ppar gamma signaling': 'PPARγ signaling pathway',
    'pparα signaling pathway': 'PPARα signaling pathway',
    'pparα signaling': 'PPARα signaling pathway',
    'ppar-α signaling pathway': 'PPARα signaling pathway',
    'ppar-α signaling': 'PPARα signaling pathway',
    'ppar alpha signaling pathway': 'PPARα signaling pathway',
    'ppar alpha signaling': 'PPARα signaling pathway',
    # AMPK
    'ampk signaling pathway': 'AMPK signaling pathway',
    'ampk signaling': 'AMPK signaling pathway',
    'ampk pathway': 'AMPK signaling pathway',
    # Insulin signaling
    'insulin signaling pathway': 'Insulin signaling pathway',
    'insulin signaling': 'Insulin signaling pathway',
    'insulin pathway': 'Insulin signaling pathway',
    'insulin receptor signaling': 'Insulin receptor signaling',
    'insulin receptor signaling pathway': 'Insulin receptor signaling',
    # MAPK
    'mapk signaling pathway': 'MAPK signaling pathway',
    'mapk signaling': 'MAPK signaling pathway',
    'mapk pathway': 'MAPK signaling pathway',
    'mapk/erk signaling': 'MAPK/ERK signaling pathway',
    'mapk/erk signaling pathway': 'MAPK/ERK signaling pathway',
    'mapk/erk pathway': 'MAPK/ERK signaling pathway',
    # JNK
    'jnk signaling pathway': 'JNK signaling pathway',
    'jnk signaling': 'JNK signaling pathway',
    'jnk pathway': 'JNK signaling pathway',
    # JAK-STAT
    'jak-stat signaling pathway': 'JAK-STAT signaling pathway',
    'jak-stat signaling': 'JAK-STAT signaling pathway',
    'jak-stat pathway': 'JAK-STAT signaling pathway',
    'jak/stat signaling pathway': 'JAK-STAT signaling pathway',
    'jak/stat signaling': 'JAK-STAT signaling pathway',
    'jak/stat pathway': 'JAK-STAT signaling pathway',
    # PI3K/AKT
    'pi3k/akt signaling pathway': 'PI3K/AKT signaling pathway',
    'pi3k/akt signaling': 'PI3K/AKT signaling pathway',
    'pi3k/akt pathway': 'PI3K/AKT signaling pathway',
    'pi3k-akt signaling pathway': 'PI3K/AKT signaling pathway',
    'pi3k-akt signaling': 'PI3K/AKT signaling pathway',
    'pi3k-akt pathway': 'PI3K/AKT signaling pathway',
    'akt signaling pathway': 'PI3K/AKT signaling pathway',
    'akt signaling': 'PI3K/AKT signaling pathway',
    # mTOR
    'mtor signaling pathway': 'mTOR signaling pathway',
    'mtor signaling': 'mTOR signaling pathway',
    'mtor pathway': 'mTOR signaling pathway',
    # Wnt
    'wnt signaling pathway': 'Wnt signaling pathway',
    'wnt signaling': 'Wnt signaling pathway',
    'wnt pathway': 'Wnt signaling pathway',
    'wnt/β-catenin signaling': 'Wnt/β-catenin signaling pathway',
    'wnt/β-catenin signaling pathway': 'Wnt/β-catenin signaling pathway',
    'wnt/beta-catenin signaling': 'Wnt/β-catenin signaling pathway',
    'wnt/beta-catenin signaling pathway': 'Wnt/β-catenin signaling pathway',
    # TGF-β
    'tgf-β signaling pathway': 'TGF-β signaling pathway',
    'tgf-β signaling': 'TGF-β signaling pathway',
    'tgf-β pathway': 'TGF-β signaling pathway',
    'tgf-beta signaling pathway': 'TGF-β signaling pathway',
    'tgf-beta signaling': 'TGF-β signaling pathway',
    'tgfβ signaling pathway': 'TGF-β signaling pathway',
    'tgfβ signaling': 'TGF-β signaling pathway',
    # Apoptosis
    'apoptosis pathway': 'Apoptosis pathway',
    'apoptosis signaling pathway': 'Apoptosis pathway',
    'apoptosis signaling': 'Apoptosis pathway',
    'apoptotic pathway': 'Apoptosis pathway',
    'apoptotic signaling pathway': 'Apoptosis pathway',
    # NLRP3 inflammasome
    'nlrp3 inflammasome pathway': 'NLRP3 inflammasome pathway',
    'nlrp3 inflammasome signaling': 'NLRP3 inflammasome pathway',
    'nlrp3 inflammasome signaling pathway': 'NLRP3 inflammasome pathway',
    'nlrp3 inflammasome': 'NLRP3 inflammasome pathway',
    'nlrp3 inflammasome activation': 'NLRP3 inflammasome activation',
    # EGFR
    'egfr signaling pathway': 'EGFR signaling pathway',
    'egfr signaling': 'EGFR signaling pathway',
    'egfr pathway': 'EGFR signaling pathway',
    # Notch / Hedgehog
    'notch signaling pathway': 'Notch signaling pathway',
    'notch signaling': 'Notch signaling pathway',
    'notch pathway': 'Notch signaling pathway',
    'hedgehog signaling pathway': 'Hedgehog signaling pathway',
    'hedgehog signaling': 'Hedgehog signaling pathway',
    'hedgehog pathway': 'Hedgehog signaling pathway',
    # Cell cycle
    'cell cycle regulation': 'Cell cycle regulation',
    'cell cycle pathway': 'Cell cycle regulation',
    'cell cycle signaling': 'Cell cycle regulation',
}

MECHANISM_CANONICAL = {
    'inflammation': 'Inflammation',
    'chronic inflammation': 'Chronic inflammation',
    'acute inflammation': 'Acute inflammation',
    'systemic inflammation': 'Systemic inflammation',
    'low-grade inflammation': 'Low-grade inflammation',
    'systemic low-grade inflammation': 'Systemic low-grade inflammation',
    'oxidative stress': 'Oxidative stress',
    'endoplasmic reticulum stress': 'Endoplasmic reticulum stress',
    'er stress': 'Endoplasmic reticulum stress',
    'insulin resistance': 'Insulin resistance',
    'adipocyte dysfunction': 'Adipocyte dysfunction',
    'endothelial dysfunction': 'Endothelial dysfunction',
    'mitochondrial dysfunction': 'Mitochondrial dysfunction',
    'adipogenesis': 'Adipogenesis', 'lipogenesis': 'Lipogenesis',
    'lipolysis': 'Lipolysis', 'thermogenesis': 'Thermogenesis',
    'angiogenesis': 'Angiogenesis', 'apoptosis': 'Apoptosis',
    'autophagy': 'Autophagy', 'fibrosis': 'Fibrosis',
    'thrombosis': 'Thrombosis', 'necrosis': 'Necrosis',
    'ferroptosis': 'Ferroptosis', 'pyroptosis': 'Pyroptosis',
}

# ============================================================================
# BRITISH → AMERICAN SPELLING NORMALIZATION
# ============================================================================

BRITISH_TO_AMERICAN = [
    ('isation', 'ization'),
    ('ising', 'izing'),
    ('signalling', 'signaling'),
    ('modelling', 'modeling'),
    ('remodelling', 'remodeling'),
    ('labelling', 'labeling'),
    ('channelling', 'channeling'),
    ('cancelling', 'canceling'),
    ('travelling', 'traveling'),
    ('counselling', 'counseling'),
    ('tumour', 'tumor'),
    ('behaviour', 'behavior'),
    ('colour', 'color'),
    ('favour', 'favor'),
    ('honour', 'honor'),
    ('labour', 'labor'),
    ('neighbourhood', 'neighborhood'),
    ('oestrogen', 'estrogen'),
    ('oedema', 'edema'),
    ('anaemia', 'anemia'),
    ('anaesthesia', 'anesthesia'),
    ('haemorrhage', 'hemorrhage'),
    ('haemoglobin', 'hemoglobin'),
    ('haematocrit', 'hematocrit'),
    ('haematopoietic', 'hematopoietic'),
    ('haemostasis', 'hemostasis'),
    ('haemolysis', 'hemolysis'),
    ('leukaemia', 'leukemia'),
    ('diarrhoea', 'diarrhea'),
    ('foetal', 'fetal'),
    ('foetus', 'fetus'),
    ('coeliac', 'celiac'),
    ('orthopaedic', 'orthopedic'),
    ('paediatric', 'pediatric'),
    ('gynaecological', 'gynecological'),
    ('defence', 'defense'),
    ('offence', 'offense'),
    ('analogue', 'analog'),
    ('catalogue', 'catalog'),
    ('homologue', 'homolog'),
    ('analyse', 'analyze'),
    ('catalyse', 'catalyze'),
    ('paralyse', 'paralyze'),
    ('hydrolyse', 'hydrolyze'),
]


# ============================================================================
# HELPER FUNCTIONS (shared)
# ============================================================================

def remove_parenthetical_abbreviations(text: str) -> str:
    """Remove abbreviations in parentheses from tissue names."""
    pattern = r'\s*\([A-Za-z][A-Za-z0-9/\-]*\)\s*$'
    return re.sub(pattern, '', text).strip()


def normalize_for_category_lookup(text: str) -> str:
    """Normalize text for looking up in category mappings."""
    text = text.strip()
    text = remove_parenthetical_abbreviations(text)
    return text.lower()


def parse_list_value(value: Any) -> Tuple[List[str], bool]:
    """
    Parse a value that could be in various formats into a list of strings.
    Returns: (list of individual values, was_originally_list)
    """
    if value is None:
        return [], False

    if isinstance(value, list):
        result = [str(v).strip() for v in value if v is not None and str(v).strip()]
        return result, True

    value_str = str(value).strip()
    if not value_str:
        return [], False

    # String representation of a list
    if value_str.startswith('[') and value_str.endswith(']'):
        try:
            parsed = ast.literal_eval(value_str)
            if isinstance(parsed, list):
                result = [str(v).strip() for v in parsed if v is not None and str(v).strip()]
                return result, True
        except (ValueError, SyntaxError):
            inner = value_str[1:-1]
            items = [t.strip().strip("'\"") for t in inner.split(',')]
            result = [t for t in items if t]
            return result, True

    # Semicolon delimiter
    if ';' in value_str:
        items = [t.strip() for t in value_str.split(';')]
        result = [t for t in items if t]
        return result, len(result) > 1

    # Comma delimiter (careful — commas can be part of names)
    if ',' in value_str and ':' not in value_str:
        items = [t.strip() for t in value_str.split(',')]
        result = [t for t in items if t]
        if len(result) > 1:
            return result, True

    return [value_str], False


def normalize_to_consistent_format(values: List[str], sort: bool = True) -> Union[str, List[str]]:
    """
    Convert a list of values to a consistent format.
    Single value → string; multiple → sorted list.
    Deduplicates (case-insensitive).
    """
    if not values:
        return 'Not specified'

    seen = set()
    unique = []
    for v in values:
        v_lower = v.lower()
        if v_lower not in seen:
            seen.add(v_lower)
            unique.append(v)

    if sort and len(unique) > 1:
        unique.sort()

    return unique[0] if len(unique) == 1 else unique

def normalize_spelling(text: str) -> str:
    """Normalize British spellings to American English."""
    if not text:
        return text

    result = text
    text_lower = text.lower()

    for british, american in BRITISH_TO_AMERICAN:
        if british in text_lower:
            idx = text_lower.find(british)
            while idx != -1:
                result = result[:idx] + american + result[idx + len(british):]
                text_lower = result.lower()
                idx = text_lower.find(british, idx + len(american))

    return result


# ============================================================================
# TISSUE NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_tissue_to_category(tissue: str) -> str:
    """Normalize tissue to broad category for the TISSUE field."""
    if not tissue or not tissue.strip():
        return 'Not specified'

    lookup_key = normalize_for_category_lookup(tissue)
    if lookup_key in ['not specified', 'not specified.', 'na', 'n/a', 'unknown', '']:
        return 'Not specified'

    if lookup_key in TISSUE_BROAD_CATEGORIES:
        category = TISSUE_BROAD_CATEGORIES[lookup_key]
    else:
        category = lookup_key

    return category.title()


def normalize_detailed_tissue(tissue: str) -> str:
    """Normalize detailed_tissue field — keep specificity but clean format."""
    if not tissue or not tissue.strip():
        return 'Not specified'

    tissue = tissue.strip()
    tissue_lower = tissue.lower()
    if tissue_lower in ['not specified', 'not specified.', 'na', 'n/a', 'unknown', '']:
        return 'Not specified'

    if tissue_lower in DETAILED_TISSUE_CANONICAL:
        return DETAILED_TISSUE_CANONICAL[tissue_lower]

    tissue_cleaned = remove_parenthetical_abbreviations(tissue)
    tissue_cleaned_lower = tissue_cleaned.lower()
    if tissue_cleaned_lower in DETAILED_TISSUE_CANONICAL:
        return DETAILED_TISSUE_CANONICAL[tissue_cleaned_lower]

    # Title case (preserving certain lowercase words)
    lowercase_words = {'of', 'and', 'or', 'the', 'in', 'to', 'for', 'from', 'with', 'a', 'an'}
    words = tissue_cleaned.lower().split()
    normalized_words = []
    for i, word in enumerate(words):
        if i == 0:
            normalized_words.append(word.capitalize())
        elif word in lowercase_words:
            normalized_words.append(word)
        else:
            normalized_words.append(word.capitalize())

    title_cased = ' '.join(normalized_words)
    title_cased_lower = title_cased.lower()
    if title_cased_lower in DETAILED_TISSUE_CANONICAL:
        return DETAILED_TISSUE_CANONICAL[title_cased_lower]

    return title_cased


# ============================================================================
# MECHANISMS / PATHWAYS NORMALIZATION FUNCTIONS
# ============================================================================

def sentence_case_preserve_acronyms(text: str) -> str:
    """
    Sentence case a term while preserving known biomedical acronyms.
    First letter capitalized, acronyms keep canonical form, rest lowercased.
    """
    if not text or not text.strip():
        return text

    text = text.strip()
    words = text.split()
    result_words = []

    for i, word in enumerate(words):
        # Strip surrounding punctuation
        leading_punct = ''
        core_start = 0
        for ch in word:
            if ch in '([':
                leading_punct += ch
                core_start += 1
            else:
                break

        trailing_punct = ''
        core_end = len(word)
        for ch in reversed(word[core_start:]):
            if ch in '.,;:)]':
                trailing_punct = ch + trailing_punct
                core_end -= 1
            else:
                break

        word_clean = word[core_start:core_end]
        word_clean_lower = word_clean.lower()

        # Exact acronym match
        if word_clean_lower in _ACRONYM_LOOKUP:
            result_words.append(leading_punct + _ACRONYM_LOOKUP[word_clean_lower] + trailing_punct)
            continue

        # Hyphenated words: check prefixes for acronyms like "NF-κB-mediated"
        if '-' in word_clean:
            if word_clean_lower in _ACRONYM_LOOKUP:
                result_words.append(leading_punct + _ACRONYM_LOOKUP[word_clean_lower] + trailing_punct)
                continue

            parts = word_clean.split('-')
            found_prefix = False
            for end_idx in range(len(parts), 0, -1):
                prefix = '-'.join(parts[:end_idx])
                if prefix.lower() in _ACRONYM_LOOKUP:
                    rebuilt = _ACRONYM_LOOKUP[prefix.lower()]
                    suffix_parts = parts[end_idx:]
                    if suffix_parts:
                        rebuilt = rebuilt + '-' + '-'.join(p.lower() for p in suffix_parts)
                    result_words.append(leading_punct + rebuilt + trailing_punct)
                    found_prefix = True
                    break

            if found_prefix:
                continue

            lowered = word_clean.lower()
            if i == 0:
                lowered = lowered[0].upper() + lowered[1:] if len(lowered) > 1 else lowered.upper()
            result_words.append(leading_punct + lowered + trailing_punct)
            continue

        # Slash-separated: "MAPK/ERK"
        if '/' in word_clean:
            slash_parts = word_clean.split('/')
            rebuilt = []
            for sp in slash_parts:
                if sp.lower() in _ACRONYM_LOOKUP:
                    rebuilt.append(_ACRONYM_LOOKUP[sp.lower()])
                else:
                    rebuilt.append(sp.lower())
            result_words.append(leading_punct + '/'.join(rebuilt) + trailing_punct)
            continue

        # Regular word
        lowered = word.lower()
        if i == 0:
            lowered = lowered[0].upper() + lowered[1:] if len(lowered) > 1 else lowered.upper()
        result_words.append(lowered)

    return ' '.join(result_words)


def tokenize_semicolon_field(raw_value: Any) -> List[str]:
    """Split a semicolon-delimited field into individual terms."""
    if raw_value is None:
        return []
    if isinstance(raw_value, float):
        if math.isnan(raw_value):
            return []
        return [str(raw_value)]
    if isinstance(raw_value, list):
        terms = []
        for item in raw_value:
            terms.extend(tokenize_semicolon_field(item))
        return terms

    raw_str = str(raw_value).strip()
    # Strip trailing periods before checking (catches "not specified." etc.)
    raw_check = raw_str.rstrip('.').lower()
    if raw_check in EMPTY_SENTINELS:
        return []

    parts = re.split(r'\s*;\s*', raw_str)
    terms = []
    for part in parts:
        part = part.strip()
        if len(part) < 2:
            continue
        # Also filter individual terms that are sentinels
        if part.rstrip('.').lower() in EMPTY_SENTINELS:
            continue
        part = re.sub(r'\s+', ' ', part)
        terms.append(part)
    return terms


def normalize_mechanism_pathway_value(raw_value: Any, field_type: str) -> Optional[str]:
    """
    Normalize a mechanisms or pathways field value.
    Tokenize → sentence case + canonical mapping → deduplicate → sort → rejoin.
    """
    canonical_map = PATHWAY_CANONICAL if field_type == 'pathways' else MECHANISM_CANONICAL

    terms = tokenize_semicolon_field(raw_value)
    if not terms:
        return None

    # Normalize each term
    normalized = []
    for t in terms:
        t = normalize_spelling(t)
        n = sentence_case_preserve_acronyms(t)
        lookup = n.lower().strip()
        if lookup in canonical_map:
            n = canonical_map[lookup]
        normalized.append(n)

    # Deduplicate (case-insensitive)
    seen = set()
    unique = []
    for t in normalized:
        t_lower = t.lower()
        if t_lower not in seen:
            seen.add(t_lower)
            unique.append(t)

    unique.sort(key=lambda x: x.lower())
    return '; '.join(unique)


# ============================================================================
# GENERIC CONTEXT FIELD NORMALIZATION
# ============================================================================

def normalize_context_field_value(value: Any, field_name: str = '') -> Any:
    """
    Normalize any context field value to a consistent format.
    Special handling for Tissue, Detailed_Tissue, Mechanisms, Pathways.
    """
    if value is None:
        return None

    if not isinstance(value, (str, list)):
        return value

    items, was_list = parse_list_value(value)
    if not items:
        return None

    field_lower = field_name.lower()

    # Tissue field → broad categories
    if field_lower == 'tissue':
        normalized_items = [normalize_tissue_to_category(item) for item in items]
        return normalize_to_consistent_format(normalized_items, sort=True)

    # Detailed_Tissue → keep specificity
    if field_lower in ['detailed_tissue', 'detailed tissue']:
        normalized_items = [normalize_detailed_tissue(item) for item in items]
        return normalize_to_consistent_format(normalized_items, sort=True)

    # Mechanisms → sentence case + canonical + sort + rejoin
    if field_lower == 'mechanisms':
        return normalize_mechanism_pathway_value(value, 'mechanisms')

    # Pathways → sentence case + canonical + sort + rejoin
    if field_lower == 'pathways':
        return normalize_mechanism_pathway_value(value, 'pathways')

    # All other fields: just normalize format (no content changes)
    cleaned = [str(item).strip() for item in items if item is not None and str(item).strip()]
    return normalize_to_consistent_format(cleaned, sort=True)


def normalize_all_context_fields(context: dict) -> dict:
    """Normalize all fields in a context dictionary."""
    if not context or not isinstance(context, dict):
        return context

    normalized = {}
    for key, value in context.items():
        if key.startswith('Original_'):
            normalized[key] = value
            continue
        normalized[key] = normalize_context_field_value(value, field_name=key)
    return normalized


# ============================================================================
# MAIN NORMALIZATION PIPELINE
# ============================================================================

def normalize_graph(input_graph_path: str, output_graph_path: str,
                    report_path: Optional[str] = None):
    """
    Normalize tissue, detailed_tissue, mechanisms, and pathways in one pass.
    """
    print(f"Loading graph from: {input_graph_path}")
    kg = KnowledgeGraph.import_graph(input_graph_path)
    print(f"Loaded: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")

    # ------------------------------------------------------------------
    # BEFORE statistics
    # ------------------------------------------------------------------
    print("\nCollecting pre-normalization statistics...")
    stats = {'original_edges': kg.number_of_edges()}

    tissue_before = Counter()
    detailed_tissue_before = Counter()
    mech_before = Counter()
    mech_terms_before = Counter()
    path_before = Counter()
    path_terms_before = Counter()
    context_fields_found = Counter()

    for u, v, key, data in kg.edges(keys=True, data=True):
        ctx = data.get('context', {})
        if isinstance(ctx, dict):
            for fn in ctx:
                context_fields_found[fn] += 1
            if 'Tissue' in ctx:
                tissue_before[str(ctx['Tissue'])] += 1
            if 'Detailed_Tissue' in ctx:
                detailed_tissue_before[str(ctx['Detailed_Tissue'])] += 1

            # Mechanisms
            raw_mech = ctx.get('Mechanisms', ctx.get('mechanisms'))
            if raw_mech and str(raw_mech).rstrip('.').lower() not in EMPTY_SENTINELS:
                mech_before[str(raw_mech)] += 1
                for t in tokenize_semicolon_field(raw_mech):
                    mech_terms_before[t.lower()] += 1

            # Pathways
            raw_path = ctx.get('Pathways', ctx.get('pathways'))
            if raw_path and str(raw_path).rstrip('.').lower() not in EMPTY_SENTINELS:
                path_before[str(raw_path)] += 1
                for t in tokenize_semicolon_field(raw_path):
                    path_terms_before[t.lower()] += 1

    print(f"  Context fields found: {len(context_fields_found)}")
    print(f"  Unique mechanism values: {len(mech_before):,} "
          f"({len(mech_terms_before):,} unique individual terms)")
    print(f"  Unique pathway values: {len(path_before):,} "
          f"({len(path_terms_before):,} unique individual terms)")

    # ------------------------------------------------------------------
    # Process edges
    # ------------------------------------------------------------------
    print("\nProcessing edges...")

    edges_to_process = []
    for u, v, key, data in kg.edges(keys=True, data=True):
        edges_to_process.append((u, v, key, copy.deepcopy(data)))

    kg.clear_edges()

    tissue_after = Counter()
    detailed_tissue_after = Counter()
    mech_after = Counter()
    mech_terms_after = Counter()
    path_after = Counter()
    path_terms_after = Counter()
    mech_merge_log = defaultdict(int)
    path_merge_log = defaultdict(int)

    edges_split = 0
    edges_modified_mechpath = 0

    for u, v, key, data in edges_to_process:
        # --- Tissue handling (can split edges) ---
        ctx = data.get('context', {})
        if not isinstance(ctx, dict):
            ctx = {}

        original_tissue = ctx.get('Tissue', 'Not specified')
        original_detailed = ctx.get('Detailed_Tissue', 'Not specified')

        tissue_items, _ = parse_list_value(original_tissue)
        if not tissue_items:
            tissue_items = ['Not specified']

        tissue_categories = []
        for tissue in tissue_items:
            category = normalize_tissue_to_category(tissue)
            tissue_categories.append(category)

        # Deduplicate tissue categories
        seen = set()
        unique_categories = []
        for cat in tissue_categories:
            cat_lower = cat.lower()
            if cat_lower not in seen:
                seen.add(cat_lower)
                unique_categories.append(cat)

        # Normalize detailed_tissue
        detailed_items, _ = parse_list_value(original_detailed)
        if detailed_items:
            norm_detailed = [normalize_detailed_tissue(dt) for dt in detailed_items]
            normalized_detailed_field = normalize_to_consistent_format(norm_detailed, sort=True)
        else:
            normalized_detailed_field = 'Not specified'

        # Backfill detailed_tissue from tissue if needed
        if normalized_detailed_field == 'Not specified':
            if original_tissue:
                t_items, _ = parse_list_value(original_tissue)
                tissues_extra = []
                for ot in (t_items or []):
                    nd = normalize_detailed_tissue(ot)
                    bc = normalize_tissue_to_category(ot)
                    if nd.lower() != bc.lower() and nd.lower() != 'not specified':
                        if nd not in tissues_extra:
                            tissues_extra.append(nd)
                if tissues_extra:
                    normalized_detailed_field = normalize_to_consistent_format(tissues_extra, sort=True)

        # --- Mechanisms / Pathways normalization ---
        mechpath_modified = False
        for field, field_cap in [('mechanisms', 'Mechanisms'), ('pathways', 'Pathways')]:
            raw = data.get(field)
            raw_from_ctx = ctx.get(field_cap, ctx.get(field))
            raw_value = raw or raw_from_ctx

            if raw_value is None:
                continue
            raw_str = str(raw_value).strip()
            if raw_str.rstrip('.').lower() in EMPTY_SENTINELS:
                continue

            normalized = normalize_mechanism_pathway_value(raw_value, field)
            if normalized is None:
                continue

            if raw_str != normalized:
                mechpath_modified = True
                # Track merge for reporting (truncate for readability)
                if field == 'mechanisms':
                    mech_merge_log[(raw_str[:100], normalized[:100])] += 1
                else:
                    path_merge_log[(raw_str[:100], normalized[:100])] += 1

            data[field] = normalized

        if mechpath_modified:
            edges_modified_mechpath += 1

        # --- Track splitting ---
        if len(unique_categories) > 1:
            edges_split += 1

        # --- Create edges (one per tissue category) ---
        for tissue_category in unique_categories:
            new_data = copy.deepcopy(data)
            new_data['tissue'] = tissue_category
            new_data['detailed_tissue'] = normalized_detailed_field

            # Preserve originals
            if 'context' not in new_data or not isinstance(new_data['context'], dict):
                new_data['context'] = {}
            if 'Tissue' in data.get('context', {}):
                new_data['context']['Original_Tissue'] = data['context']['Tissue']
            if 'Detailed_Tissue' in data.get('context', {}):
                new_data['context']['Original_Detailed_Tissue'] = data['context']['Detailed_Tissue']

            # Normalize ALL context fields
            new_data['context'] = normalize_all_context_fields(new_data['context'])
            new_data['context']['Tissue'] = tissue_category
            new_data['context']['Detailed_Tissue'] = normalized_detailed_field

            # Ensure mechanisms/pathways are also in context
            if 'mechanisms' in new_data and new_data['mechanisms']:
                new_data['context']['Mechanisms'] = new_data['mechanisms']
            if 'pathways' in new_data and new_data['pathways']:
                new_data['context']['Pathways'] = new_data['pathways']

            kg.add_edge(u, v, **new_data)

            # Collect AFTER stats
            tissue_after[tissue_category] += 1
            detailed_tissue_after[str(normalized_detailed_field)] += 1

            mech_val = new_data.get('mechanisms')
            if mech_val and str(mech_val).rstrip('.').lower() not in EMPTY_SENTINELS:
                mech_after[str(mech_val)] += 1
                for t in tokenize_semicolon_field(mech_val):
                    mech_terms_after[t.lower()] += 1

            path_val = new_data.get('pathways')
            if path_val and str(path_val).rstrip('.').lower() not in EMPTY_SENTINELS:
                path_after[str(path_val)] += 1
                for t in tokenize_semicolon_field(path_val):
                    path_terms_after[t.lower()] += 1

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("NORMALIZATION RESULTS")
    print("=" * 80)
    print(f"Original edges:          {stats['original_edges']:,}")
    print(f"Final edges:             {kg.number_of_edges():,}")
    print(f"Edges split (tissue):    {edges_split:,}")
    print(f"Edges modified (mech/path): {edges_modified_mechpath:,}")

    # Tissue stats
    print(f"\n--- TISSUE ---")
    print(f"  Unique values: {len(tissue_before)} -> {len(tissue_after)}")
    for t, c in tissue_after.most_common(15):
        print(f"    {c:>8,}  {t}")

    print(f"\n--- DETAILED_TISSUE ---")
    print(f"  Unique values: {len(detailed_tissue_before)} -> {len(detailed_tissue_after)}")
    for t, c in detailed_tissue_after.most_common(15):
        t_str = str(t)[:70]
        print(f"    {c:>8,}  {t_str}")

    # Mechanisms stats
    print(f"\n--- MECHANISMS ---")
    print(f"  Unique full values: {len(mech_before):,} -> {len(mech_after):,} "
          f"(Δ={len(mech_after) - len(mech_before):,})")
    print(f"  Unique individual terms: {len(mech_terms_before):,} -> {len(mech_terms_after):,} "
          f"(Δ={len(mech_terms_after) - len(mech_terms_before):,})")
    sing_before = sum(1 for c in mech_before.values() if c == 1)
    sing_after = sum(1 for c in mech_after.values() if c == 1)
    print(f"  Singleton values: {sing_before:,} -> {sing_after:,}")
    print(f"  Top 10 full values (after):")
    for v, c in mech_after.most_common(10):
        print(f"    {c:>6,}  {v[:80]}")
    print(f"  Top 10 terms (after):")
    for t, c in mech_terms_after.most_common(10):
        print(f"    {c:>6,}  {t}")

    # Pathways stats
    print(f"\n--- PATHWAYS ---")
    print(f"  Unique full values: {len(path_before):,} -> {len(path_after):,} "
          f"(Δ={len(path_after) - len(path_before):,})")
    print(f"  Unique individual terms: {len(path_terms_before):,} -> {len(path_terms_after):,} "
          f"(Δ={len(path_terms_after) - len(path_terms_before):,})")
    sing_before = sum(1 for c in path_before.values() if c == 1)
    sing_after = sum(1 for c in path_after.values() if c == 1)
    print(f"  Singleton values: {sing_before:,} -> {sing_after:,}")
    print(f"  Top 10 full values (after):")
    for v, c in path_after.most_common(10):
        print(f"    {c:>6,}  {v[:80]}")
    print(f"  Top 10 terms (after):")
    for t, c in path_terms_after.most_common(10):
        print(f"    {c:>6,}  {t}")

    # Example merges
    print(f"\n--- EXAMPLE NORMALIZATION CHANGES ---")
    for label, log in [('MECHANISMS', mech_merge_log), ('PATHWAYS', path_merge_log)]:
        sorted_changes = sorted(log.items(), key=lambda x: -x[1])
        total_changes = sum(log.values())
        if not sorted_changes:
            print(f"  {label}: No changes")
            continue
        print(f"\n  {label}: {len(sorted_changes):,} unique changes ({total_changes:,} edge modifications)")
        for (before, after), count in sorted_changes[:10]:
            print(f"    [{count:>5,}x] {before}")
            print(f"           -> {after}")

    # Save
    print(f"\nSaving normalized graph to: {output_graph_path}")
    kg.export_graph(output_graph_path)

    # Diagnostic report
    if report_path:
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_graph': input_graph_path,
            'output_graph': output_graph_path,
            'original_edges': stats['original_edges'],
            'final_edges': kg.number_of_edges(),
            'edges_split_tissue': edges_split,
            'edges_modified_mechpath': edges_modified_mechpath,
            'tissue': {
                'unique_before': len(tissue_before),
                'unique_after': len(tissue_after),
            },
            'detailed_tissue': {
                'unique_before': len(detailed_tissue_before),
                'unique_after': len(detailed_tissue_after),
            },
            'mechanisms': {
                'unique_values_before': len(mech_before),
                'unique_values_after': len(mech_after),
                'unique_terms_before': len(mech_terms_before),
                'unique_terms_after': len(mech_terms_after),
                'top_10_values': [(v, c) for v, c in mech_after.most_common(10)],
                'top_10_terms': [(t, c) for t, c in mech_terms_after.most_common(10)],
            },
            'pathways': {
                'unique_values_before': len(path_before),
                'unique_values_after': len(path_after),
                'unique_terms_before': len(path_terms_before),
                'unique_terms_after': len(path_terms_after),
                'top_10_values': [(v, c) for v, c in path_after.most_common(10)],
                'top_10_terms': [(t, c) for t, c in path_terms_after.most_common(10)],
            },
            'n_canonical_pathway_mappings': len(PATHWAY_CANONICAL),
            'n_canonical_mechanism_mappings': len(MECHANISM_CANONICAL),
            'n_preserved_acronyms': len(PRESERVE_ACRONYMS),
        }
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Saved report to: {report_path}")

    print("\n✓ Normalization complete!")
    return kg


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python normalize_graph.py <input_graph.pkl> [output_graph.pkl] [report.json]")
        print()
        print("Normalizes in a single pass:")
        print("  - Tissue (broad categories, edge splitting)")
        print("  - Detailed_Tissue (specificity preserved, format cleaned)")
        print("  - Mechanisms (sentence case, acronyms, sort, dedup)")
        print("  - Pathways (sentence case, canonical mapping, sort, dedup)")
        print("  - All other context fields (format consistency)")
        return

    input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        if input_path.endswith('.pkl'):
            output_path = input_path.replace('.pkl', '_normalized.pkl')
        else:
            output_path = input_path + '_normalized.pkl'

    report_path = sys.argv[3] if len(sys.argv) > 3 else output_path.replace('.pkl', '_report.json')
    normalize_graph(input_path, output_path, report_path)


if __name__ == "__main__":
    main()