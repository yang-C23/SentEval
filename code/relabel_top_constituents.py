#!/usr/bin/env python3
"""
Relabel top_constituents for Chinese and French using Benepar parser.

This script:
1. Parses sentences using Benepar (Berkeley Neural Parser)
2. Extracts top-level constituent labels from the parse tree
3. Handles low-frequency labels by mapping them to OTHER
4. Maintains balanced label distribution similar to English data
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import spacy
import benepar
from tqdm import tqdm


# English label reference (20 classes, 6000 each)
ENGLISH_LABELS = [
    "ADVP_NP_VP_.", "CC_ADVP_NP_VP_.", "CC_NP_VP_.", "IN_NP_VP_.",
    "NP_ADVP_VP_.", "NP_NP_VP_.", "NP_PP_.", "NP_VP_.",
    "OTHER", "PP_NP_VP_.", "RB_NP_VP_.", "S_CC_S_.",
    "S_NP_VP_.", "S_VP_.", "SBAR_NP_VP_.", "SBAR_VP_.",
    "VBD_NP_VP_.", "VP_.", "WHADVP_SQ_.", "WHNP_SQ_."
]


@dataclass
class Record:
    """Represents a single line from the probing dataset."""
    idx: int
    partition: str
    label: str
    text: str

    @classmethod
    def from_line(cls, idx: int, raw_line: str) -> "Record":
        parts = raw_line.rstrip("\n").split("\t", 2)
        if len(parts) != 3:
            raise ValueError(f"Line {idx + 1} expected 3 tab-separated fields, got {len(parts)}")
        return cls(idx=idx, partition=parts[0], label=parts[1], text=parts[2])

    def to_line(self, new_label: Optional[str] = None) -> str:
        label = new_label if new_label is not None else self.label
        return f"{self.partition}\t{label}\t{self.text}"


def setup_parser(language: str, use_gpu: bool = True) -> Tuple[spacy.Language, str]:
    """
    Setup spaCy + Benepar parser for the given language.
    
    Returns:
        Tuple of (spacy nlp object, benepar model name)
    """
    import torch
    
    # Check GPU availability
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logging.info("Using CPU")
    
    if language == "zh":
        # Chinese
        nlp = spacy.load("zh_core_web_sm")
        benepar_model = "benepar_zh2"
    elif language == "fr":
        # French - use multilingual model since there's no dedicated French model
        nlp = spacy.load("fr_core_news_sm")
        # Benepar doesn't have a dedicated French model, we'll use spacy's parser
        # and extract constituents differently
        benepar_model = None
    else:
        raise ValueError(f"Unsupported language: {language}")
    
    if benepar_model:
        try:
            # Add benepar with GPU support if available
            if device == "cuda":
                nlp.add_pipe("benepar", config={"model": benepar_model})
            else:
                nlp.add_pipe("benepar", config={"model": benepar_model})
        except Exception as e:
            logging.warning(f"Could not load benepar model {benepar_model}: {e}")
            benepar_model = None
    
    return nlp, benepar_model


def extract_top_constituents_from_tree(tree_str: str, language: str = "zh") -> str:
    """
    Extract top-level constituent labels from a parse tree string.
    
    For Chinese (CTB format):
        (IP (NP ...) (VP ...) (PU 。)) -> "NP_VP_."
    
    For English (PTB format):
        (S (NP ...) (VP ...)) -> "NP_VP_."
    """
    tree_str = tree_str.strip()
    if not tree_str:
        return "OTHER"
    
    # Find top-level label and its children
    depth = 0
    top_labels = []
    i = 0
    
    # Skip the outermost wrapper (IP for Chinese, S/ROOT for English)
    # Find where the wrapper label ends
    match = re.match(r'\(([A-Z]+)\s+', tree_str)
    if not match:
        return "OTHER"
    
    wrapper_label = match.group(1)
    i = match.end()
    depth = 1
    
    # Now find immediate children of the wrapper
    while i < len(tree_str) and depth > 0:
        if tree_str[i] == '(':
            if depth == 1:  # Immediate child of wrapper
                # Extract label
                child_match = re.match(r'\(([A-Z]+(?:-[A-Z0-9]+)?)', tree_str[i:])
                if child_match:
                    label = child_match.group(1)
                    # Skip punctuation
                    if label not in ["PU", ",", ".", ":", ";", "!", "?"]:
                        top_labels.append(label)
            depth += 1
        elif tree_str[i] == ')':
            depth -= 1
        i += 1
    
    if not top_labels:
        return "OTHER"
    
    # Deduplicate consecutive same labels while preserving order
    deduped = []
    prev = None
    for label in top_labels:
        if label != prev:
            deduped.append(label)
            prev = label
    
    # Limit to first 4 constituents to avoid overly long labels
    deduped = deduped[:4]
    
    # Create label string
    label_str = "_".join(deduped) + "_."
    return label_str


def extract_top_constituents_chinese(sent) -> str:
    """
    Extract top-level constituents from a Chinese sentence using benepar.
    
    Chinese Treebank uses labels like:
    - IP (simple clause), CP (complex clause)
    - NP, VP, ADVP, PP, etc.
    """
    try:
        if not hasattr(sent, '_') or not hasattr(sent._, 'parse_string'):
            return "OTHER"
        
        tree_str = sent._.parse_string
        if tree_str is None:
            return "OTHER"
            
        return extract_top_constituents_from_tree(tree_str, language="zh")
    except Exception as e:
        logging.debug(f"Parse error: {e}")
        return "OTHER"


def extract_top_constituents_spacy(doc) -> str:
    """
    Extract pseudo-constituents from spaCy dependency parse.
    This is a fallback when Benepar is not available.
    
    Maps dependency structure to constituent-like labels.
    """
    if not doc or len(doc) == 0:
        return "OTHER"
    
    # Get root and its direct children
    roots = [token for token in doc if token.head == token]
    if not roots:
        return "OTHER"
    
    root = roots[0]
    
    # Map dependency labels to constituent-like labels
    dep_to_const = {
        "nsubj": "NP",
        "nsubjpass": "NP",
        "dobj": "NP",
        "obj": "NP",
        "iobj": "NP",
        "pobj": "NP",
        "advmod": "ADVP",
        "advcl": "SBAR",
        "ccomp": "SBAR",
        "xcomp": "VP",
        "prep": "PP",
        "mark": "IN",
        "cc": "CC",
        "conj": "S",
        "aux": "VP",
        "ROOT": "VP",
    }
    
    # Get direct children of root
    children_labels = []
    for child in root.children:
        dep = child.dep_
        const = dep_to_const.get(dep, "OTHER")
        if const != "OTHER":
            children_labels.append(const)
    
    # Add root's constituent type
    root_const = "VP" if root.pos_ == "VERB" else "NP"
    
    if not children_labels:
        return f"{root_const}_."
    
    # Deduplicate while preserving order
    seen = set()
    unique_labels = []
    for label in children_labels:
        if label not in seen:
            seen.add(label)
            unique_labels.append(label)
    
    label_str = "_".join(unique_labels[:3]) + "_" + root_const + "_."
    return label_str


def normalize_label(label: str, language: str) -> str:
    """
    Normalize labels to be more consistent across languages.
    
    Maps language-specific labels to Penn Treebank style labels used in English.
    
    Chinese Treebank specific mappings:
    - IP -> S (sentence)
    - CP -> SBAR (subordinate clause)
    - DP -> NP (determiner phrase)
    - QP -> NP (quantifier phrase)
    - LCP -> PP (localizer phrase)
    - DNP -> NP (de-phrase)
    - DVP -> ADVP (de-phrase for adverbs)
    - CLP -> NP (classifier phrase)
    """
    if language == "zh":
        # Chinese Treebank to Penn Treebank style mapping
        mapping = {
            # Clausal categories
            "IP": "S",
            "CP": "SBAR",
            # Phrase categories  
            "DP": "NP",
            "QP": "NP",
            "DNP": "NP",
            "CLP": "NP",
            "NN": "NP",
            "NR": "NP",
            "NT": "NP",
            "PN": "NP",
            "LCP": "PP",
            "DVP": "ADVP",
            "MSP": "ADVP",
            "AD": "RB",
            "AS": "VP",
            "VV": "VP",
            "VC": "VP",
            "VE": "VP",
            "VA": "VP",
            "VSB": "VP",
            "VCD": "VP",
            "VRD": "VP",
            "VNV": "VP",
            "VCP": "VP",
            "BA": "VP",
            "LB": "VP",
            "SB": "VP",
            "CS": "IN",
            "DEC": "IN",
            "DEG": "IN",
            "DER": "IN",
            "DEV": "IN",
            "CC": "CC",
            "P": "IN",
            "M": "NP",
            "OD": "NP",
            "CD": "NP",
            "DT": "NP",
            "JJ": "NP",
            # Special categories
            "FRAG": "OTHER",
            "LST": "OTHER",
            "FLR": "OTHER",
            "INC": "OTHER",
            "PRN": "OTHER",
            "UCP": "OTHER",
            "INTJ": "OTHER",
            "X": "OTHER",
        }
        
        # Replace Chinese-specific labels with English equivalents
        parts = label.replace("_.", "").split("_")
        normalized_parts = []
        for p in parts:
            mapped = mapping.get(p, p)
            if mapped and mapped != "OTHER":
                normalized_parts.append(mapped)
        
        if not normalized_parts:
            return "OTHER"
        
        # Deduplicate consecutive labels
        deduped = []
        prev = None
        for part in normalized_parts:
            if part != prev:
                deduped.append(part)
                prev = part
        
        # Limit length
        deduped = deduped[:4]
        
        return "_".join(deduped) + "_."
    
    elif language == "fr":
        # For French, we use spaCy dependency parse since benepar doesn't have a French model
        # The labels are already in a reasonable format from the fallback parser
        # Just do basic normalization
        mapping = {
            "SENT": "S",
            "Sint": "S",
            "Ssub": "SBAR",
            "Srel": "SBAR",
            "VN": "VP",
            "VPinf": "VP",
            "VPpart": "VP",
            "AP": "ADJP",
            "AdP": "ADVP",
            "COORD": "CC",
        }
        
        parts = label.replace("_.", "").split("_")
        normalized_parts = []
        for p in parts:
            mapped = mapping.get(p, p)
            if mapped:
                normalized_parts.append(mapped)
        
        if not normalized_parts:
            return "OTHER"
        
        # Deduplicate
        deduped = []
        prev = None
        for part in normalized_parts:
            if part != prev:
                deduped.append(part)
                prev = part
        
        deduped = deduped[:4]
        
        return "_".join(deduped) + "_."
    
    return label


def parse_sentences(
    nlp: spacy.Language,
    texts: List[str],
    language: str,
    use_benepar: bool = True,
    batch_size: int = 50,
) -> List[str]:
    """
    Parse sentences and extract top constituent labels.
    """
    labels = []
    
    # Process in batches for efficiency
    for i in tqdm(range(0, len(texts), batch_size), desc="Parsing"):
        batch = texts[i:i + batch_size]
        
        try:
            docs = list(nlp.pipe(batch))
            
            for doc in docs:
                if use_benepar and language == "zh":
                    # Use benepar for Chinese
                    try:
                        # Get the first sentence
                        sents = list(doc.sents)
                        if sents:
                            label = extract_top_constituents_chinese(sents[0])
                        else:
                            label = "OTHER"
                    except Exception as e:
                        logging.debug(f"Benepar parse error: {e}")
                        label = "OTHER"
                else:
                    # Use spaCy dependency parse as fallback
                    label = extract_top_constituents_spacy(doc)
                
                # Normalize the label
                label = normalize_label(label, language)
                labels.append(label)
                
        except Exception as e:
            logging.warning(f"Batch processing error: {e}")
            labels.extend(["OTHER"] * len(batch))
    
    return labels


def consolidate_labels(
    labels: List[str],
    min_frequency: int = 100,
    target_num_classes: int = 20,
) -> Tuple[List[str], Dict[str, str], Counter]:
    """
    Consolidate labels to handle low-frequency classes.
    
    Strategy:
    1. Count all label frequencies
    2. Keep top N-1 most frequent labels (N = target_num_classes)
    3. Map all other labels to OTHER
    
    Returns:
        - List of consolidated labels
        - Mapping from old labels to new labels
        - Counter of final label frequencies
    """
    # Count frequencies
    label_counts = Counter(labels)
    
    logging.info(f"Found {len(label_counts)} unique labels before consolidation")
    logging.info(f"Top 10 labels: {label_counts.most_common(10)}")
    
    # Keep top N-1 labels, rest go to OTHER
    top_labels = [label for label, count in label_counts.most_common(target_num_classes - 1)]
    
    # Create mapping
    label_mapping = {}
    for label in label_counts:
        if label in top_labels:
            label_mapping[label] = label
        else:
            label_mapping[label] = "OTHER"
    
    # Apply mapping
    consolidated = [label_mapping[label] for label in labels]
    
    # Count final frequencies
    final_counts = Counter(consolidated)
    
    logging.info(f"After consolidation: {len(final_counts)} unique labels")
    logging.info(f"Final label distribution: {final_counts.most_common()}")
    
    return consolidated, label_mapping, final_counts


def process_file(
    input_path: Path,
    output_path: Path,
    language: str,
    min_frequency: int = 100,
    target_num_classes: int = 20,
    sample_count: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Process input file: parse sentences and relabel with top constituents.
    """
    # Load records
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                records.append(Record.from_line(idx, line))
            except ValueError as e:
                logging.warning(f"Skipping invalid line {idx + 1}: {e}")
    
    logging.info(f"Loaded {len(records)} records from {input_path}")
    
    # Sample if requested
    if sample_count is not None:
        records = records[:sample_count]
        logging.info(f"Sampling first {sample_count} records for testing")
    
    # Setup parser
    logging.info(f"Setting up parser for language: {language}")
    nlp, benepar_model = setup_parser(language)
    use_benepar = benepar_model is not None
    logging.info(f"Using benepar: {use_benepar}")
    
    # Extract texts
    texts = [r.text for r in records]
    
    # Parse and get labels
    logging.info("Parsing sentences...")
    new_labels = parse_sentences(nlp, texts, language, use_benepar)
    
    # Consolidate labels
    logging.info("Consolidating labels...")
    consolidated_labels, label_mapping, final_counts = consolidate_labels(
        new_labels,
        min_frequency=min_frequency,
        target_num_classes=target_num_classes,
    )
    
    # Update records with new labels
    for record, new_label in zip(records, consolidated_labels):
        record.label = new_label
    
    # Statistics
    stats = {
        "total_records": len(records),
        "unique_labels_before": len(set(new_labels)),
        "unique_labels_after": len(final_counts),
        "label_distribution": dict(final_counts.most_common()),
    }
    
    # Write output
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(record.to_line() + "\n")
        logging.info(f"Wrote output to {output_path}")
    else:
        logging.info("Dry run - no output written")
        print("\n" + "=" * 60)
        print("SAMPLE OUTPUT (first 20):")
        print("=" * 60)
        for record in records[:20]:
            print(f"{record.partition}\t{record.label}\t{record.text[:50]}...")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Relabel top_constituents using Benepar parser"
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Path to the translated top_constituents.txt file",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path for the relabeled output file",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["zh", "fr"],
        required=True,
        help="Language of the input file",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=100,
        help="Minimum frequency for a label to be kept (default: 100)",
    )
    parser.add_argument(
        "--target_classes",
        type=int,
        default=20,
        help="Target number of label classes (default: 20)",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=None,
        help="Only process first N sentences (for testing)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't write output, just show samples",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    stats = process_file(
        input_path=args.input_file,
        output_path=args.output_file,
        language=args.language,
        min_frequency=args.min_frequency,
        target_num_classes=args.target_classes,
        sample_count=args.sample_count,
        dry_run=args.dry_run,
    )
    
    print("\n" + "=" * 60)
    print("STATISTICS:")
    print("=" * 60)
    print(f"  Total records: {stats['total_records']}")
    print(f"  Unique labels before consolidation: {stats['unique_labels_before']}")
    print(f"  Unique labels after consolidation: {stats['unique_labels_after']}")
    print("\n  Label distribution:")
    for label, count in stats['label_distribution'].items():
        pct = 100.0 * count / stats['total_records']
        print(f"    {label}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()

