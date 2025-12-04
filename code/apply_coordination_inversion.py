#!/usr/bin/env python3
"""
Apply coordination inversion to translated Chinese and French sentences.

This script:
1. Reads translated probing data (coordination_inversion.txt)
2. For sentences labeled "O", randomly selects 50% to invert
3. Inverts the clause order while preserving the conjunction
4. Updates the label from "O" to "I" for inverted sentences
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Language-specific conjunction patterns
# For Chinese: conjunction appears after Chinese comma (，)
# For French: conjunction appears after comma (, )

CHINESE_CONJUNCTIONS = [
    "并且",
    "而且", 
    "但是",
    "然而",
    "或者",
    "或",
    "所以",
    "因此",
    "既不",
    "也不",
    "不过",
]

FRENCH_CONJUNCTIONS = [
    "et",
    "mais", 
    "ou",
    "donc",
    "car",
    "ni",
    "pourtant",
    "or",
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

    def to_line(self, new_text: Optional[str] = None, new_label: Optional[str] = None) -> str:
        text = new_text if new_text is not None else self.text
        label = new_label if new_label is not None else self.label
        return f"{self.partition}\t{label}\t{text}"


def find_chinese_conjunction(text: str) -> Optional[Tuple[str, int]]:
    """
    Find the Chinese conjunction in text and return (conjunction, comma_position).
    Look for pattern: [clause A]，[conjunction][clause B]
    
    Searches through all Chinese commas to find one followed by a conjunction.
    """
    comma = "，"
    
    # Find all comma positions
    pos = 0
    while True:
        comma_pos = text.find(comma, pos)
        if comma_pos == -1:
            break
        
        # Check what comes after this comma
        after_comma = text[comma_pos + 1:].lstrip()
        
        for conj in CHINESE_CONJUNCTIONS:
            if after_comma.startswith(conj):
                return (conj, comma_pos)
        
        pos = comma_pos + 1
    
    return None


def find_french_conjunction(text: str) -> Optional[Tuple[str, int]]:
    """
    Find the first French conjunction in text and return (conjunction, position).
    Look for pattern: [clause A], [conjunction] [clause B]
    """
    # French uses regular comma followed by space
    # Pattern: ", et ", ", mais ", etc.
    for conj in FRENCH_CONJUNCTIONS:
        # Case-insensitive search for ", conj "
        pattern = rf",\s+({re.escape(conj)})\s+"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return (match.group(1), match.start())
    
    return None


def invert_chinese_sentence(text: str) -> Optional[str]:
    """
    Invert a Chinese sentence with coordination structure.
    
    Original: [分句A]，[连词][分句B]
    Inverted: [分句B]，[连词][分句A]
    """
    result = find_chinese_conjunction(text)
    if result is None:
        return None
    
    conj, comma_pos = result
    comma = "，"
    
    # Extract clause A (before comma)
    clause_a = text[:comma_pos].strip()
    
    # Extract conjunction and clause B (after comma)
    after_comma = text[comma_pos + 1:].lstrip()
    
    # Remove conjunction from the beginning of clause B
    clause_b = after_comma[len(conj):].lstrip()
    
    # Handle trailing punctuation
    # Chinese sentences often end with 。but sometimes have other endings
    trailing_punct = ""
    if clause_b and clause_b[-1] in "。！？!?.":
        trailing_punct = clause_b[-1]
        clause_b = clause_b[:-1].rstrip()
    
    # Construct inverted sentence
    # New structure: [clause B]，[conjunction][clause A][punctuation]
    inverted = f"{clause_b}{comma}{conj}{clause_a}{trailing_punct}"
    
    return inverted


def invert_french_sentence(text: str) -> Optional[str]:
    """
    Invert a French sentence with coordination structure.
    
    Original: [Clause A], [conjunction] [Clause B]
    Inverted: [Clause B], [conjunction] [Clause A]
    """
    result = find_french_conjunction(text)
    if result is None:
        return None
    
    conj, comma_start = result
    
    # Extract clause A (before the comma)
    clause_a = text[:comma_start].strip()
    
    # Find where the conjunction ends
    # Pattern: ", conj " - we need to find the conjunction in context
    pattern = rf",\s+({re.escape(conj)})\s+"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    
    # Extract clause B (after the conjunction)
    clause_b = text[match.end():].strip()
    
    # Handle trailing punctuation
    trailing_punct = ""
    if clause_b and clause_b[-1] in ".!?":
        trailing_punct = clause_b[-1]
        clause_b = clause_b[:-1].rstrip()
    
    # Preserve the original case of the conjunction
    original_conj = match.group(1)
    
    # Construct inverted sentence
    # New structure: [Clause B], [conjunction] [Clause A][punctuation]
    # Capitalize the first letter of clause B (it's now at sentence start)
    if clause_b:
        clause_b = clause_b[0].upper() + clause_b[1:] if len(clause_b) > 1 else clause_b.upper()
    
    # For clause A: only lowercase the first character if it appears to be 
    # capitalized just because it was at sentence start (not a proper noun)
    # Heuristic: if it's a common French pronoun or article, lowercase it
    COMMON_FRENCH_STARTS = {
        # Pronouns
        "Je", "Tu", "Il", "Elle", "On", "Nous", "Vous", "Ils", "Elles",
        # Articles
        "Le", "La", "Les", "Un", "Une", "Des", "De", "Du",
        # Demonstratives
        "Ce", "Cette", "Ces", "Cet", "Cela", "Ceci",
        # Possessives
        "Mon", "Ma", "Mes", "Ton", "Ta", "Tes", "Son", "Sa", "Ses",
        "Notre", "Nos", "Votre", "Vos", "Leur", "Leurs",
        # Contractions and elisions
        "C'", "J'", "L'", "D'", "N'", "S'", "M'", "T'", "Qu'",
        # Common adverbs and prepositions
        "Alors", "Ainsi", "Après", "Avant", "Avec", "Bien", "Dans", "Donc",
        "En", "Et", "Mais", "Pour", "Quand", "Que", "Qui", "Si", "Sur", "Tout",
        # Other common starts
        "Rien", "Personne", "Chaque", "Aucun", "Aucune", "Tout", "Toute",
    }
    
    if clause_a:
        first_word = clause_a.split()[0] if clause_a.split() else ""
        # Check if first word is a common start that should be lowercased
        if first_word in COMMON_FRENCH_STARTS:
            clause_a = clause_a[0].lower() + clause_a[1:] if len(clause_a) > 1 else clause_a.lower()
        # Otherwise keep original capitalization (likely a proper noun)
    
    inverted = f"{clause_b}, {original_conj.lower()} {clause_a}{trailing_punct}"
    
    return inverted


def process_file(
    input_path: Path,
    output_path: Path,
    language: str,
    inversion_ratio: float = 0.5,
    seed: int = 42,
    sample_count: Optional[int] = None,
    dry_run: bool = False,
    remove_english: bool = False,
) -> Dict[str, int]:
    """
    Process the input file and apply inversion to randomly selected sentences.
    
    Args:
        remove_english: If True, remove original English "I" labeled sentences,
                       keeping only translated language sentences.
    
    Returns statistics about the processing.
    """
    random.seed(seed)
    
    # Read all records
    records: List[Record] = []
    with input_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                records.append(Record.from_line(idx, line))
            except ValueError as e:
                logging.warning(f"Skipping invalid line {idx + 1}: {e}")
    
    logging.info(f"Loaded {len(records)} records from {input_path}")
    
    # Select inversion function based on language
    if language == "zh":
        invert_func = invert_chinese_sentence
    elif language == "fr":
        invert_func = invert_french_sentence
    else:
        raise ValueError(f"Unsupported language: {language}")
    
    # Mark original English "I" records BEFORE any processing
    # These are the ones we want to remove later
    original_english_i_indices: set = set()
    if remove_english:
        for i, r in enumerate(records):
            if r.label == "I":
                # Original "I" records are English (from the source dataset)
                original_english_i_indices.add(i)
        logging.info(f"Marked {len(original_english_i_indices)} original English 'I' records for removal")
    
    # Find all "O" labeled records (these are translated sentences)
    o_records = [(i, r) for i, r in enumerate(records) if r.label == "O"]
    logging.info(f"Found {len(o_records)} 'O' labeled records")
    
    # If sample_count is specified, limit to first N
    if sample_count is not None:
        o_records = o_records[:sample_count]
        logging.info(f"Limited to first {sample_count} 'O' records for testing")
    
    # Randomly select which records to invert
    num_to_invert = int(len(o_records) * inversion_ratio)
    indices_to_invert = set(random.sample(range(len(o_records)), num_to_invert))
    
    logging.info(f"Will attempt to invert {num_to_invert} sentences ({inversion_ratio*100:.0f}%)")
    
    # Statistics
    stats = {
        "total_records": len(records),
        "o_records": len(o_records),
        "attempted_inversions": num_to_invert,
        "successful_inversions": 0,
        "failed_inversions": 0,
    }
    
    # Apply inversions
    inverted_indices = set()
    for i, (record_idx, record) in enumerate(o_records):
        if i in indices_to_invert:
            inverted = invert_func(record.text)
            if inverted is not None:
                records[record_idx].text = inverted
                records[record_idx].label = "I"
                inverted_indices.add(record_idx)
                stats["successful_inversions"] += 1
            else:
                stats["failed_inversions"] += 1
                logging.debug(f"Could not invert line {record_idx + 1}: {record.text[:50]}...")
    
    logging.info(f"Successfully inverted {stats['successful_inversions']} sentences")
    logging.info(f"Failed to invert {stats['failed_inversions']} sentences (no conjunction found)")
    
    # Filter out original English "I" records if requested
    if remove_english:
        original_count = len(records)
        # Keep only records that are NOT in the original English "I" set
        records = [r for i, r in enumerate(records) if i not in original_english_i_indices]
        removed_count = original_count - len(records)
        stats["english_i_removed"] = removed_count
        stats["final_record_count"] = len(records)
        logging.info(f"Removed {removed_count} English 'I' records, keeping {len(records)} translated records")
    
    # Write output
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(record.to_line() + "\n")
        logging.info(f"Wrote output to {output_path}")
    else:
        logging.info("Dry run - no output written")
        # Print some examples
        print("\n" + "="*60)
        print("SAMPLE INVERSIONS (first 10):")
        print("="*60)
        count = 0
        for idx in sorted(inverted_indices):
            if count >= 10:
                break
            if idx < len(records):
                print(f"\nLine {idx + 1}:")
                print(f"  Inverted: {records[idx].text}")
                count += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Apply coordination inversion to translated sentences"
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Path to the translated coordination_inversion.txt file",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path for the output file with inversions applied",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["zh", "fr"],
        required=True,
        help="Language of the input file (zh for Chinese, fr for French)",
    )
    parser.add_argument(
        "--inversion_ratio",
        type=float,
        default=0.5,
        help="Ratio of 'O' sentences to invert (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=None,
        help="Only process first N 'O' sentences (for testing)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't write output, just show samples",
    )
    parser.add_argument(
        "--remove_english",
        action="store_true",
        help="Remove original English 'I' labeled sentences, keeping only translated language",
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
        inversion_ratio=args.inversion_ratio,
        seed=args.seed,
        sample_count=args.sample_count,
        dry_run=args.dry_run,
        remove_english=args.remove_english,
    )
    
    print("\n" + "="*60)
    print("STATISTICS:")
    print("="*60)
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

