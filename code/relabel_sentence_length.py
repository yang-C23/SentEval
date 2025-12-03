#!/usr/bin/env python3
"""
Re-label translated sentences based on word count in the target language.
"""

import argparse
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Re-label translated probing dataset based on word count")
    parser.add_argument("--input_file", type=str, required=True, help="Input translated file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file with new labels")
    parser.add_argument("--language", type=str, required=True, choices=["zh", "fr"], 
                        help="Language of the input file (zh for Chinese, fr for French)")
    return parser.parse_args()

def count_words_chinese(sentence):
    """Count words in Chinese sentence using jieba segmentation."""
    try:
        import jieba
        import string
        words = jieba.lcut(sentence)
        # Filter out punctuation (both Chinese and English), whitespace, and ellipsis
        chinese_punct = '，。！？；：、""''（）【】《》…'
        english_punct = string.punctuation  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        all_punct = chinese_punct + english_punct
        
        words = [w.strip() for w in words if w.strip() and not all(c in all_punct or c.isspace() for c in w)]
        return len(words)
    except ImportError:
        print("Warning: jieba not installed. Falling back to character count.")
        # Fallback: use character count for Chinese
        import string
        chinese_punct = '，。！？；：、""''（）【】《》…'
        english_punct = string.punctuation
        all_punct = chinese_punct + english_punct + ' \t\n'
        chars = [c for c in sentence if c not in all_punct]
        return len(chars)

def count_words_french(sentence):
    """Count words in French sentence using space-based tokenization."""
    import string
    # Simple space-based word count
    words = sentence.split()
    # Filter out empty strings and pure punctuation tokens
    # Keep words that contain at least one alphanumeric character
    words = [w for w in words if w.strip() and any(c.isalnum() for c in w)]
    return len(words)

def assign_label_french(word_count):
    """
    Assign label for French based on quantile thresholds.
    Thresholds calculated to match English label distribution (16.67% per label).
    
    - Label 0: <= 7 words (19.77%)
    - Label 1: 8-11 words (17.08%)
    - Label 2: 12-15 words (16.47%)
    - Label 3: 16-19 words (15.55%)
    - Label 4: 20-24 words (17.28%)
    - Label 5: >= 25 words (13.84%)
    """
    if word_count <= 7:
        return 0
    elif word_count <= 11:
        return 1
    elif word_count <= 15:
        return 2
    elif word_count <= 19:
        return 3
    elif word_count <= 24:
        return 4
    else:
        return 5

def assign_label_chinese(word_count):
    """
    Assign label for Chinese based on quantile thresholds.
    Thresholds calculated to match English label distribution (16.67% per label).
    
    - Label 0: <= 6 words (20.43%)
    - Label 1: 7-9 words (18.62%)
    - Label 2: 10-11 words (12.76%)
    - Label 3: 12-14 words (18.18%)
    - Label 4: 15-17 words (14.40%)
    - Label 5: >= 18 words (15.62%)
    """
    if word_count <= 6:
        return 0
    elif word_count <= 9:
        return 1
    elif word_count <= 11:
        return 2
    elif word_count <= 14:
        return 3
    elif word_count <= 17:
        return 4
    else:
        return 5

def main():
    args = parse_args()
    
    # Determine word counting function and label assignment function
    if args.language == "zh":
        print("Using jieba segmentation for Chinese word counting")
        print("Using Chinese-specific thresholds to match English label distribution")
        count_words = count_words_chinese
        assign_label = assign_label_chinese
    elif args.language == "fr":
        print("Using space-based tokenization for French word counting")
        print("Using French-specific thresholds to match English label distribution")
        count_words = count_words_french
        assign_label = assign_label_french
    else:
        raise ValueError(f"Unsupported language: {args.language}")
    
    # Load and process data
    print(f"Loading data from {args.input_file}")
    relabeled_data = []
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    word_count_stats = []
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 3:
                print(f"Warning: Line {line_num} has {len(parts)} fields, expected 3. Skipping.")
                continue
            
            partition, old_label, sentence = parts
            
            # Count words in translated sentence
            word_count = count_words(sentence)
            word_count_stats.append(word_count)
            
            # Assign new label based on word count (using language-specific function)
            new_label = assign_label(word_count)
            label_counts[new_label] += 1
            
            relabeled_data.append((partition, new_label, sentence))
    
    print(f"\nProcessed {len(relabeled_data)} sentences")
    
    # Print statistics
    print(f"\nWord count statistics:")
    print(f"  Min: {min(word_count_stats)}")
    print(f"  Max: {max(word_count_stats)}")
    print(f"  Mean: {sum(word_count_stats)/len(word_count_stats):.2f}")
    
    print(f"\nLabel distribution:")
    for label in sorted(label_counts.keys()):
        print(f"  Label {label}: {label_counts[label]} sentences ({label_counts[label]/len(relabeled_data)*100:.2f}%)")
    
    # Save relabeled data
    print(f"\nSaving relabeled data to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for partition, label, sentence in relabeled_data:
            f.write(f"{partition}\t{label}\t{sentence}\n")
    
    print(f"Successfully saved {len(relabeled_data)} relabeled sentences")

if __name__ == "__main__":
    main()

