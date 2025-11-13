#!/usr/bin/env python3
"""
Translate SentEval probing dataset using NLLB-200-3.3B model.
Supports parallel GPU processing by splitting the dataset into chunks.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Translate SentEval probing dataset")
    parser.add_argument("--input_file", type=str, required=True, help="Input file path")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    parser.add_argument("--target_lang", type=str, required=True, choices=["zh", "fr"], 
                        help="Target language (zh for Chinese, fr for French)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index for processing (-1 for all)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for translation")
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-3.3B", 
                        help="NLLB model name")
    return parser.parse_args()

def load_data(input_file, start_idx=0, end_idx=-1):
    """Load data from input file and return lines within the specified range."""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if end_idx == -1:
        end_idx = len(lines)
    
    return lines[start_idx:end_idx]

def parse_line(line):
    """Parse a line into partition, label, and sentence."""
    parts = line.strip().split('\t')
    if len(parts) != 3:
        raise ValueError(f"Expected 3 fields, got {len(parts)}: {line}")
    return parts[0], parts[1], parts[2]

def translate_batch(sentences, model, tokenizer, src_lang, tgt_lang, device):
    """Translate a batch of sentences."""
    # Set source language for tokenizer
    tokenizer.src_lang = src_lang
    
    # Tokenize input - increase max_length to handle longer sentences
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get target language token ID
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    # Generate translations with increased output length
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=256,  # Changed from max_length to max_new_tokens
            num_beams=4,
            early_stopping=True
        )
    
    # Decode translations
    translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translations

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Language codes for NLLB
    src_lang = "eng_Latn"  # English
    if args.target_lang == "zh":
        tgt_lang = "zho_Hans"  # Simplified Chinese
    elif args.target_lang == "fr":
        tgt_lang = "fra_Latn"  # French
    else:
        raise ValueError(f"Unsupported target language: {args.target_lang}")
    
    print(f"Loading model: {args.model_name}")
    print(f"Translating from {src_lang} to {tgt_lang}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.src_lang = src_lang  # Set source language
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = model.to(device)
    model.eval()
    
    # Verify language codes are valid
    print(f"Verifying language codes...")
    src_lang_id = tokenizer.convert_tokens_to_ids(src_lang)
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    print(f"Source language ID: {src_lang_id} ({src_lang})")
    print(f"Target language ID: {tgt_lang_id} ({tgt_lang})")
    
    print(f"Model loaded successfully on {device}")
    
    # Load data
    print(f"Loading data from {args.input_file}")
    lines = load_data(args.input_file, args.start_idx, args.end_idx)
    print(f"Processing {len(lines)} lines (from {args.start_idx} to {args.start_idx + len(lines)})")
    
    # Parse data
    data = []
    for line in lines:
        partition, label, sentence = parse_line(line)
        data.append((partition, label, sentence))
    
    # Translate in batches
    translated_data = []
    batch_size = args.batch_size
    
    print(f"Starting translation with batch size {batch_size}")
    start_time = time.time()
    
    for i in tqdm(range(0, len(data), batch_size), desc="Translating"):
        batch = data[i:i+batch_size]
        partitions = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        sentences = [item[2] for item in batch]
        
        try:
            translations = translate_batch(sentences, model, tokenizer, src_lang, tgt_lang, device)
            
            for partition, label, translation in zip(partitions, labels, translations):
                translated_data.append((partition, label, translation))
        except Exception as e:
            print(f"\nError translating batch at index {i}: {e}")
            # Fallback to sentence-by-sentence translation for this batch
            for partition, label, sentence in batch:
                try:
                    translation = translate_batch([sentence], model, tokenizer, src_lang, tgt_lang, device)[0]
                    translated_data.append((partition, label, translation))
                except Exception as e2:
                    print(f"Error translating sentence: {sentence[:50]}... Error: {e2}")
                    # Keep original sentence as fallback
                    translated_data.append((partition, label, sentence))
    
    elapsed_time = time.time() - start_time
    print(f"\nTranslation completed in {elapsed_time:.2f} seconds")
    print(f"Average speed: {len(data)/elapsed_time:.2f} sentences/second")
    
    # Save translated data
    print(f"Saving translations to {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for partition, label, translation in translated_data:
            f.write(f"{partition}\t{label}\t{translation}\n")
    
    print(f"Successfully saved {len(translated_data)} translations")

if __name__ == "__main__":
    main()

