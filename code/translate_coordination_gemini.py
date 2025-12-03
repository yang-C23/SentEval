#!/usr/bin/env python3
"""
Parallel translation of SentEval probing task (coordination inversion) using Gemini API.

Only sentences labeled with "O" (non-perturbed) are translated and the output format
matches the original metadata (partition, label, sentence). Failed sentences are logged
and left in English so the line count/order remain identical.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from google import genai  # type: ignore
from tqdm import tqdm


# Thread-local storage avoids sharing a single client instance across threads.
_thread_local: threading.local = threading.local()


LANGUAGE_CONFIG: Dict[str, Dict[str, str]] = {
    "zh": {
        "task": (
            "Translate each English sentence into Simplified Chinese while preserving the original "
            "coordination structure, clause order, and conjunctions. Keep the sentence in the form "
            "\"[分句A]，[连词][分句B]\" so the conjunction remains the first word after the comma, "
            "and keep the sentence length close to the original."
        ),
        "rules": (
            "- Detect coordinating conjunctions (and, but, or, so, for, nor, yet).\n"
            "- Translate them into explicit Chinese connectors such as 并且/而且, 但是/然而, 或者/或, "
            "所以/因此, 既不...也不 for 'nor', and 然而/不过 for 'yet'.\n"
            "- Do not delete, paraphrase, or move the conjunction; it must appear immediately after the comma.\n"
            "- If the English sentence has no coordinating conjunction, keep the clause structure faithful without inventing one.\n"
            "- Always use the Chinese comma character (，) between clauses."
        ),
    },
    "fr": {
        "task": (
            "Translate each English sentence into French while preserving the original coordination "
            "structure, clause order, and conjunctions. Keep the sentence in the form "
            "\"[Proposition A], [Conjonction] Proposition B\" so the conjunction remains the first word "
            "after the comma, and keep the sentence length close to the original."
        ),
        "rules": (
            "- Detect coordinating conjunctions (and, but, or, so, for, nor, yet).\n"
            "- Translate them into explicit French connectors such as et, mais, ou, donc, car, ni, pourtant.\n"
            "- Do not delete, paraphrase, or relocate the conjunction; it must appear immediately after the comma.\n"
            "- If the English sentence lacks a coordinating conjunction, mirror the clause structure faithfully without inventing one.\n"
            "- Always use the standard comma (,) between clauses and respect French spacing conventions."
        ),
    },
}


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

    def to_line(self, new_text: Optional[str] = None) -> str:
        text = new_text if new_text is not None else self.text
        return f"{self.partition}\t{self.label}\t{text}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate SentEval coordination_inversion sentences with Gemini"
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Path to the original coordination_inversion.txt file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for translated output (e.g., SentEval/data_zh/probing).",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="coordination_inversion.txt",
        help="Output filename (defaults to coordination_inversion.txt).",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("SentEval/code/logs"),
        help="Directory to store translation logs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model identifier.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of sentences per batch request.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of parallel API requests.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retry attempts for a failed batch.",
    )
    parser.add_argument(
        "--retry_backoff",
        type=float,
        default=5.0,
        help="Initial backoff (seconds) for retries; grows exponentially.",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=None,
        help="Translate only the first N 'O' sentences (useful for quick tests).",
    )
    parser.add_argument(
        "--sleep_after_error",
        type=float,
        default=2.0,
        help="Sleep (seconds) before retrying after non-rate-limit errors.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run the pipeline without writing output (for testing).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose console logging.",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        choices=sorted(LANGUAGE_CONFIG.keys()),
        default="zh",
        help="Target language for translation (default: zh).",
    )
    return parser.parse_args()


def setup_logging(log_dir: Path, verbose: bool) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"coordination_inversion_{time.strftime('%Y%m%d_%H%M%S')}.log"

    handlers = [logging.FileHandler(log_path, encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.info("Logging to %s", log_path)
    return log_path


def get_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in the environment."
        )
    return key


def get_client(api_key: str) -> genai.Client:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = genai.Client(api_key=api_key)
        _thread_local.client = client
    return client


def load_records(path: Path) -> List[Record]:
    with path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    records: List[Record] = []
    for idx, line in enumerate(lines):
        try:
            records.append(Record.from_line(idx, line))
        except ValueError as exc:
            logging.error("Skipping invalid line %d: %s", idx + 1, exc)
    logging.info("Loaded %d valid lines from %s", len(records), path)
    return records


def chunked(iterable: List[Record], size: int) -> Iterable[List[Record]]:
    for start in range(0, len(iterable), size):
        yield iterable[start : start + size]


def sanitize_translation(text: str) -> str:
    """Remove spurious enumerations the model might prepend to translations."""
    patterns = [
        r"^\s*\d+[\.\)．、]\s*",
        r"^\s*[一二三四五六七八九十百千万]+\s*[、.．)]\s*",
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)
    return cleaned.strip()


def build_prompt(batch: List[Record], target_lang: str) -> str:
    numbered = "\n".join(f"{idx + 1}. {item.text}" for idx, item in enumerate(batch))
    config = LANGUAGE_CONFIG[target_lang]
    prompt = (
        "Role: Professional computational linguist and translator.\n\n"
        f"Task: {config['task']}\n\n"
        "Coordination rules:\n"
        f"{config['rules']}\n\n"
        "Output Requirements:\n"
        "1. Return ONLY the translations.\n"
        "2. Each translation must be on its own line.\n"
        "3. Maintain the input order exactly.\n"
        "4. Do not include numbering, explanations, or additional text.\n\n"
        "Sentences:\n"
        f"{numbered}"
    )
    return prompt


def translate_batch(
    batch: List[Record],
    api_key: str,
    model: str,
    target_lang: str,
) -> List[str]:
    client = get_client(api_key)
    prompt = build_prompt(batch, target_lang)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    output_text: Optional[str] = getattr(response, "text", None)
    if not output_text and hasattr(response, "candidates"):
        # Fallback: concatenate candidate parts if .text is unavailable.
        candidate = response.candidates[0] if response.candidates else None
        if candidate and candidate.content and candidate.content.parts:
            output_text = "".join(
                getattr(part, "text", "") for part in candidate.content.parts
            )

    if not output_text:
        raise RuntimeError("Empty response from Gemini API.")

    lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    if len(lines) != len(batch):
        raise ValueError(
            f"Expected {len(batch)} translations but received {len(lines)}. "
            "Check the prompt or consider reducing batch size."
        )
    return [sanitize_translation(line) for line in lines]


def translate_with_retries(
    batch: List[Record],
    api_key: str,
    model: str,
    max_retries: int,
    backoff: float,
    sleep_after_error: float,
    target_lang: str,
) -> Optional[List[str]]:
    delay = backoff
    for attempt in range(1, max_retries + 1):
        try:
            return translate_batch(batch, api_key, model, target_lang)
        except Exception as exc:
            logging.warning(
                "Batch starting at line %d failed on attempt %d/%d: %s",
                batch[0].idx + 1,
                attempt,
                max_retries,
                exc,
            )
            if attempt == max_retries:
                logging.error(
                    "Giving up on batch starting at line %d after %d attempts.",
                    batch[0].idx + 1,
                    max_retries,
                )
                return None
            time.sleep(delay)
            delay *= 2
            if sleep_after_error > 0:
                time.sleep(sleep_after_error)
    return None


def process_batches(
    batches: List[List[Record]],
    api_key: str,
    model: str,
    max_retries: int,
    retry_backoff: float,
    sleep_after_error: float,
    max_workers: int,
    target_lang: str,
) -> Tuple[Dict[int, str], int]:
    translations: Dict[int, str] = {}
    fallback_failures = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map: Dict[Future, List[Record]] = {}
        for batch in batches:
            future = executor.submit(
                translate_with_retries,
                batch,
                api_key,
                model,
                max_retries,
                retry_backoff,
                sleep_after_error,
                target_lang,
            )
            future_map[future] = batch

        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Batches"):
            batch = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                logging.exception(
                    "Unexpected error for batch starting at line %d: %s",
                    batch[0].idx + 1,
                    exc,
                )
                result = None

            if result is None:
                # Fallback: try sentence by sentence to salvage translations.
                for record in batch:
                    single_result = translate_with_retries(
                        [record],
                        api_key,
                        model,
                        max_retries,
                        retry_backoff,
                        sleep_after_error,
                        target_lang,
                    )
                    if single_result:
                        translations[record.idx] = single_result[0]
                    else:
                        fallback_failures += 1
                        translations[record.idx] = record.text
                        logging.error(
                            "Translation failed for line %d. Keeping original sentence.",
                            record.idx + 1,
                        )
                continue

            for record, translated in zip(batch, result):
                translations[record.idx] = translated

    return translations, fallback_failures


def write_output(
    records: List[Record],
    translations: Dict[int, str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_file:
        for record in records:
            translated_text = translations.get(record.idx, record.text)
            out_file.write(record.to_line(translated_text) + "\n")
    logging.info("Wrote translated data to %s", output_path)


def main() -> None:
    args = parse_args()
    log_path = setup_logging(args.log_dir, args.verbose)
    logging.info("Starting Gemini translation pipeline")

    api_key = get_api_key()
    logging.info("Using Gemini model: %s", args.model)
    logging.info("Target language: %s", args.target_lang)

    records = load_records(args.input_file)
    if not records:
        logging.error("No valid records found. Exiting.")
        return

    target_records = [record for record in records if record.label == "O"]
    logging.info("Found %d sentences labeled 'O' for translation", len(target_records))

    if args.sample_count is not None:
        original_count = len(target_records)
        target_records = target_records[: args.sample_count]
        logging.info(
            "Sample mode: limiting to first %d of %d 'O' sentences",
            len(target_records),
            original_count,
        )

    if not target_records:
        logging.warning("No target sentences to translate. Exiting.")
        return

    batches = list(chunked(target_records, args.batch_size))
    logging.info(
        "Prepared %d batches (batch size=%d, workers=%d)",
        len(batches),
        args.batch_size,
        args.max_workers,
    )

    start_time = time.time()
    translations, fallback_failures = process_batches(
        batches,
        api_key=api_key,
        model=args.model,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        sleep_after_error=args.sleep_after_error,
        max_workers=args.max_workers,
        target_lang=args.target_lang,
    )
    elapsed = time.time() - start_time

    translated_count = sum(
        1 for record in target_records if translations.get(record.idx) != record.text
    )
    logging.info(
        "Translation finished: %d/%d sentences translated (%.2f%%). "
        "%d sentences kept in English after fallback. Duration: %.2fs.",
        translated_count,
        len(target_records),
        100 * translated_count / len(target_records),
        fallback_failures,
        elapsed,
    )

    if args.dry_run:
        logging.info("Dry run enabled; skipping file write. Log file: %s", log_path)
        return

    output_path = args.output_dir / args.output_name
    write_output(records, translations, output_path)
    logging.info("Completed. Log file available at %s", log_path)


if __name__ == "__main__":
    main()

