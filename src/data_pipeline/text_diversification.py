"""
Text Diversification Pipeline for TTS-based ASR Augmentation.

Generates diverse text in Estonian and Slovenian using a 3-category framework:
  1. Paraphrase: Lexical & syntactic rewrites of CV17 sentences (Hard-Synth, arXiv:2411.13159)
  2. Domain Expansion: Novel sentences across underrepresented domains (arXiv:2509.15373)
  3. Morphological Diversity: Sentences targeting rare case/number forms (arXiv:2410.12656)

4-phase pipeline (each phase independently resumable):
  Phase 1 — GENERATE:    Async LLM generation → raw JSONL files
  Phase 2 — VALIDATE:    Async LLM-as-judge on raw sentences → validated JSONL
  Phase 3 — REGENERATE:  Async corrective regeneration of failures → re-validate → append
  Phase 4 — FINALIZE:    Deduplication + test-set leakage removal → final output

Scientific justification:
- Text diversity >> speaker diversity for TTS augmentation (arXiv:2410.16726)
- LLMs lack morphological generalization for agglutinative languages (arXiv:2410.12656),
  requiring explicit case/number prompting for Estonian (14 cases) and Slovenian (6 cases + dual)
- Domain expansion improves OOD generalization (Hard-Synth, arXiv:2411.13159)
- LLM-as-judge validation implements rejection sampling for corpus quality control

Uses OpenAI Responses API with Pydantic structured outputs for reliable parsing.
Async pipeline with controlled concurrency for speed.
"""

import argparse
import asyncio
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP request logs from openai/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANGUAGES = {
    "et": "Estonian",
    "sl": "Slovenian",
}

DOMAINS = [
    "medicine and healthcare",
    "law and legal proceedings",
    "sports and athletics",
    "technology and computing",
    "cooking and food",
    "travel and tourism",
    "weather and climate",
    "education and school",
    "finance and banking",
    "daily conversation and small talk",
]

CATEGORIES = ["paraphrase", "domain", "morphological"]
PHASES = ["generate", "validate", "regenerate", "finalize", "all"]

MAX_REGENERATION_ATTEMPTS = 2
MAX_CONCURRENT_REQUESTS = 20

MODEL_GENERATE = "gpt-5-mini"
MODEL_VALIDATE = "gpt-5-mini"

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


# ---------------------------------------------------------------------------
# Pydantic output schemas (structured outputs)
# ---------------------------------------------------------------------------

class ParaphraseOutput(BaseModel):
    """Output schema for Category 1: Paraphrase generation.

    Based on Hard-Synth (arXiv:2411.13159) methodology — LLM rewrites
    original text with same meaning but different wording/structure,
    producing both paraphrasing (word substitution) and restructuring
    (clause reordering) variants.
    """
    sentences: list[str] = Field(
        description="Exactly 3 paraphrased sentences in the target language",
        min_length=3,
        max_length=3,
    )


class DomainOutput(BaseModel):
    """Output schema for Category 2: Domain expansion.

    Based on findings from Ibaraki & Chiang (arXiv:2509.15373) —
    LLM-generated text introduces vocabulary and structures absent from
    the training corpus. Domain-specific generation ensures coverage of
    specialized terminology underrepresented in CommonVoice (read Wikipedia text).
    """
    sentences: list[str] = Field(
        description="Exactly 5 domain-specific sentences in the target language",
        min_length=5,
        max_length=5,
    )


class MorphologicalSentence(BaseModel):
    """A single morphologically-targeted sentence with its case label."""
    case_label: str = Field(description="Grammatical case and number label, e.g. 'illatiiv' or 'dual nominative'")
    sentence: str = Field(description="Natural sentence (5-15 words) using the specified case form")


class MorphologicalOutput(BaseModel):
    """Output schema for Category 3: Morphological diversity.

    Motivated by findings that LLMs lack human-like morphological
    compositional generalization for agglutinative languages
    (arXiv:2410.12656). Estonian has 14 noun cases; Slovenian has
    6 cases + dual number. Explicit case prompting forces coverage
    of underrepresented morphological patterns that the model would
    otherwise avoid.
    """
    sentences: list[MorphologicalSentence] = Field(
        description="Exactly 8 sentences, each targeting a different grammatical case/number form",
        min_length=8,
        max_length=8,
    )


class ValidationVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class ValidationOutput(BaseModel):
    """Output schema for LLM-as-judge sentence validation.

    Implements rejection sampling with LLM feedback — a two-stage
    generation pipeline where a validator LLM assesses grammatical
    correctness and naturalness of each candidate sentence.
    """
    verdict: ValidationVerdict
    reason: str = Field(
        description="Empty string if PASS. Brief failure reason if FAIL, e.g. 'incorrect case ending'",
    )


class RegenerationOutput(BaseModel):
    """Output schema for corrective regeneration after validation failure."""
    corrected_sentence: str = Field(
        description="A corrected sentence in the target language that fixes the identified issue",
    )


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS_DIR = PROMPTS_DIR / "system"


def load_prompt(name: str) -> str:
    """Load a user prompt template from the prompts/ directory."""
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def load_system_prompt(name: str, **kwargs) -> str:
    """Load a system prompt template from prompts/system/ and format it."""
    path = SYSTEM_PROMPTS_DIR / f"{name}.txt"
    template = path.read_text(encoding="utf-8")
    return template.format(**kwargs)


# ---------------------------------------------------------------------------
# JSONL I/O helpers
# ---------------------------------------------------------------------------

def _append_jsonl(records: list[dict], path: str):
    """Append records to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_jsonl(records: list[dict], path: str):
    """Write records to a JSONL file (overwrite)."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> list[dict]:
    """Read all records from a JSONL file."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _count_jsonl(path: str) -> int:
    """Count lines in a JSONL file without loading it all into memory."""
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_seed_sentences(path: str) -> list[str]:
    """Load seed sentences from a text file (one per line)."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_test_texts(path: str) -> set[str]:
    """Load test/dev texts for leakage checking (one per line, normalized)."""
    if not path or not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


# ---------------------------------------------------------------------------
# PHASE 1: GENERATE — Async generation, save raw sentences
# ---------------------------------------------------------------------------

async def _generate_paraphrases(
    client: AsyncOpenAI,
    sentence: str,
    language_code: str,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    """Category 1: Generate 3 paraphrases of a seed sentence."""
    language = LANGUAGES[language_code]
    system = load_system_prompt("paraphrase", language=language)
    prompt = load_prompt("paraphrase").format(language=language, sentence=sentence)
    async with semaphore:
        response = await client.responses.parse(
            model=MODEL_GENERATE,
            instructions=system,
            input=prompt,
            text_format=ParaphraseOutput,
            reasoning={"effort": "low"},
        )
    return response.output_parsed.sentences


async def _generate_domain_sentences(
    client: AsyncOpenAI,
    domain: str,
    language_code: str,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    """Category 2: Generate 5 domain-specific sentences."""
    language = LANGUAGES[language_code]
    system = load_system_prompt("domain", language=language)
    prompt = load_prompt("domain").format(language=language, domain=domain)
    async with semaphore:
        response = await client.responses.parse(
            model=MODEL_GENERATE,
            instructions=system,
            input=prompt,
            text_format=DomainOutput,
            reasoning={"effort": "low"},
        )
    return response.output_parsed.sentences


async def _generate_morphological_sentences(
    client: AsyncOpenAI,
    seed: str,
    language_code: str,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Category 3: Generate 8 morphologically diverse sentences. Returns dicts with case_label."""
    language = LANGUAGES[language_code]
    system = load_system_prompt(f"morphological_{language_code}")
    prompt = load_prompt(f"morphological_{language_code}").format(seed=seed)
    async with semaphore:
        response = await client.responses.parse(
            model=MODEL_GENERATE,
            instructions=system,
            input=prompt,
            text_format=MorphologicalOutput,
            reasoning={"effort": "low"},
        )
    return [{"sentence": item.sentence, "case_label": item.case_label} for item in response.output_parsed.sentences]


async def _run_generation_batch(
    client: AsyncOpenAI,
    generate_coro,
    language_code: str,
    category: str,
    output_path: str,
    extra_metadata: Optional[dict] = None,
):
    """Run one generation call and append raw sentences to the output JSONL."""
    try:
        sentences = await generate_coro
    except Exception as e:
        logger.warning(f"Generation failed ({category}): {e}")
        return 0

    records = []
    for item in sentences:
        # Morphological returns dicts with case_label, others return plain strings
        if isinstance(item, dict):
            record = {
                "sentence": item["sentence"],
                "language": language_code,
                "category": category,
                "case_label": item.get("case_label", ""),
            }
        else:
            record = {
                "sentence": item,
                "language": language_code,
                "category": category,
            }
        if extra_metadata:
            record.update(extra_metadata)
        records.append(record)

    _append_jsonl(records, output_path)
    return len(records)


async def phase_generate_paraphrase(
    client: AsyncOpenAI,
    seeds: list[str],
    language_code: str,
    semaphore: asyncio.Semaphore,
    output_path: str,
    target_count: int = 2000,
):
    """Generate paraphrase sentences (Category 1)."""
    existing = _count_jsonl(output_path)
    start_idx = existing // 3  # 3 sentences per seed

    batches_needed = (target_count + 2) // 3
    needed_seeds = seeds[start_idx:batches_needed]

    if not needed_seeds:
        logger.info(f"Paraphrase ({language_code}): already have {existing} raw sentences, skipping")
        return existing

    logger.info(f"Paraphrase ({language_code}): resuming from {existing} sentences, {len(needed_seeds)} seeds remaining")
    pbar = tqdm(total=len(needed_seeds), desc=f"Generate paraphrase ({language_code})")
    generated = existing

    chunk_size = MAX_CONCURRENT_REQUESTS
    for i in range(0, len(needed_seeds), chunk_size):
        chunk = needed_seeds[i : i + chunk_size]
        tasks = [
            _run_generation_batch(
                client=client,
                generate_coro=_generate_paraphrases(client, seed, language_code, semaphore),
                language_code=language_code,
                category="paraphrase",
                output_path=output_path,
                extra_metadata={"seed": seed},
            )
            for seed in chunk
        ]
        counts = await asyncio.gather(*tasks)
        generated += sum(counts)
        pbar.update(len(chunk))

    pbar.close()
    logger.info(f"Paraphrase ({language_code}): {generated} total raw sentences")
    return generated


async def phase_generate_domain(
    client: AsyncOpenAI,
    language_code: str,
    semaphore: asyncio.Semaphore,
    output_path: str,
    target_count: int = 2000,
):
    """Generate domain expansion sentences (Category 2)."""
    existing = _count_jsonl(output_path)

    sentences_per_domain = target_count // len(DOMAINS)
    batches_per_domain = (sentences_per_domain + 4) // 5  # 5 sentences per batch

    work_items = []
    for domain in DOMAINS:
        for _ in range(batches_per_domain):
            work_items.append(domain)

    start_idx = existing // 5
    work_items = work_items[start_idx:]

    if not work_items:
        logger.info(f"Domain ({language_code}): already have {existing} raw sentences, skipping")
        return existing

    logger.info(f"Domain ({language_code}): resuming from {existing} sentences, {len(work_items)} batches remaining")
    pbar = tqdm(total=len(work_items), desc=f"Generate domain ({language_code})")
    generated = existing

    chunk_size = MAX_CONCURRENT_REQUESTS
    for i in range(0, len(work_items), chunk_size):
        chunk = work_items[i : i + chunk_size]
        tasks = [
            _run_generation_batch(
                client=client,
                generate_coro=_generate_domain_sentences(client, domain, language_code, semaphore),
                language_code=language_code,
                category="domain",
                output_path=output_path,
                extra_metadata={"domain": domain},
            )
            for domain in chunk
        ]
        counts = await asyncio.gather(*tasks)
        generated += sum(counts)
        pbar.update(len(chunk))

    pbar.close()
    logger.info(f"Domain ({language_code}): {generated} total raw sentences")
    return generated


async def phase_generate_morphological(
    client: AsyncOpenAI,
    seeds: list[str],
    language_code: str,
    semaphore: asyncio.Semaphore,
    output_path: str,
    target_count: int = 2000,
):
    """Generate morphologically diverse sentences (Category 3)."""
    existing = _count_jsonl(output_path)
    start_idx = existing // 8  # 8 sentences per batch

    batches_needed = (target_count + 7) // 8
    work_seeds = [seeds[i % len(seeds)] for i in range(start_idx, batches_needed)]

    if not work_seeds:
        logger.info(f"Morphological ({language_code}): already have {existing} raw sentences, skipping")
        return existing

    logger.info(f"Morphological ({language_code}): resuming from {existing} sentences, {len(work_seeds)} batches remaining")
    pbar = tqdm(total=len(work_seeds), desc=f"Generate morphological ({language_code})")
    generated = existing

    chunk_size = MAX_CONCURRENT_REQUESTS
    for i in range(0, len(work_seeds), chunk_size):
        chunk = work_seeds[i : i + chunk_size]
        tasks = [
            _run_generation_batch(
                client=client,
                generate_coro=_generate_morphological_sentences(client, seed, language_code, semaphore),
                language_code=language_code,
                category="morphological",
                output_path=output_path,
                extra_metadata={"seed": seed},
            )
            for seed in chunk
        ]
        counts = await asyncio.gather(*tasks)
        generated += sum(counts)
        pbar.update(len(chunk))

    pbar.close()
    logger.info(f"Morphological ({language_code}): {generated} total raw sentences")
    return generated


async def run_phase_generate(
    language_code: str,
    seeds: list[str],
    output_dir: str,
    target_per_category: int,
    skip_categories: list[str],
    max_concurrent: int,
):
    """
    Phase 1: Generate all raw sentences across categories.

    Outputs per category:
        {output_dir}/raw_{lang}_paraphrase.jsonl
        {output_dir}/raw_{lang}_domain.jsonl
        {output_dir}/raw_{lang}_morphological.jsonl
    """
    client = AsyncOpenAI(timeout=60.0, max_retries=3)
    semaphore = asyncio.Semaphore(max_concurrent)
    os.makedirs(output_dir, exist_ok=True)

    stats = {}

    if "paraphrase" not in skip_categories:
        logger.info("=== Phase 1 / Category 1: Paraphrase (Hard-Synth methodology) ===")
        path = os.path.join(output_dir, f"raw_{language_code}_paraphrase.jsonl")
        count = await phase_generate_paraphrase(
            client, seeds, language_code, semaphore, path, target_per_category,
        )
        stats["paraphrase_raw"] = count

    if "domain" not in skip_categories:
        logger.info("=== Phase 1 / Category 2: Domain Expansion ===")
        path = os.path.join(output_dir, f"raw_{language_code}_domain.jsonl")
        count = await phase_generate_domain(
            client, language_code, semaphore, path, target_per_category,
        )
        stats["domain_raw"] = count

    if "morphological" not in skip_categories:
        logger.info("=== Phase 1 / Category 3: Morphological Diversity ===")
        path = os.path.join(output_dir, f"raw_{language_code}_morphological.jsonl")
        count = await phase_generate_morphological(
            client, seeds, language_code, semaphore, path, target_per_category,
        )
        stats["morphological_raw"] = count

    total = sum(stats.values())
    logger.info(f"Phase 1 COMPLETE: {total} raw sentences generated ({stats})")
    return stats


# ---------------------------------------------------------------------------
# PHASE 2: VALIDATE — Async LLM-as-judge on all raw sentences
# ---------------------------------------------------------------------------

async def _validate_sentence(
    client: AsyncOpenAI,
    sentence: str,
    language_code: str,
    semaphore: asyncio.Semaphore,
) -> tuple[bool, str]:
    """Validate a single sentence using LLM-as-judge."""
    language = LANGUAGES[language_code]
    system = load_system_prompt("validator", language=language)
    prompt = load_prompt("validator").format(language=language, sentence=sentence)
    async with semaphore:
        response = await client.responses.parse(
            model=MODEL_VALIDATE,
            instructions=system,
            input=prompt,
            text_format=ValidationOutput,
            reasoning={"effort": "low"},
        )
    result = response.output_parsed
    return result.verdict == ValidationVerdict.PASS, result.reason


async def _validate_one_record(
    client: AsyncOpenAI,
    record: dict,
    language_code: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Validate one record and add verdict fields."""
    try:
        passed, reason = await _validate_sentence(
            client, record["sentence"], language_code, semaphore,
        )
        record["passed"] = passed
        record["fail_reason"] = reason if not passed else ""
    except Exception as e:
        logger.warning(f"Validation error: {e} — marking as failed")
        record["passed"] = False
        record["fail_reason"] = f"validation_error: {e}"
    return record


async def run_phase_validate(
    language_code: str,
    output_dir: str,
    skip_categories: list[str],
    max_concurrent: int,
):
    """
    Phase 2: Validate all raw sentences with LLM-as-judge.

    Reads:  {output_dir}/raw_{lang}_{category}.jsonl
    Writes: {output_dir}/validated_{lang}.jsonl  (all sentences with pass/fail verdict)
    """
    client = AsyncOpenAI(timeout=60.0, max_retries=3)
    semaphore = asyncio.Semaphore(max_concurrent)

    validated_path = os.path.join(output_dir, f"validated_{language_code}.jsonl")

    # Load already-validated sentence texts to skip them on resume
    already_validated = set()
    existing_records = _read_jsonl(validated_path)
    for rec in existing_records:
        already_validated.add(rec["sentence"])
    logger.info(f"Already validated: {len(already_validated)} sentences")

    # Collect all raw sentences across categories
    all_raw = []
    for category in CATEGORIES:
        if category in skip_categories:
            continue
        raw_path = os.path.join(output_dir, f"raw_{language_code}_{category}.jsonl")
        records = _read_jsonl(raw_path)
        logger.info(f"Loaded {len(records)} raw sentences from {category}")
        all_raw.extend(records)

    # Filter out already validated
    to_validate = [r for r in all_raw if r["sentence"] not in already_validated]
    logger.info(f"To validate: {len(to_validate)} sentences ({len(all_raw) - len(to_validate)} already done)")

    if not to_validate:
        logger.info("Phase 2: nothing to validate")
        return _summarize_validation(existing_records)

    pbar = tqdm(total=len(to_validate), desc=f"Validate ({language_code})")

    async def _validate_and_track(record):
        result = await _validate_one_record(client, record, language_code, semaphore)
        pbar.update(1)
        return result

    # Process all at once, semaphore controls concurrency, tqdm updates per sentence
    batch_size = 200  # save to disk every 200 sentences
    for i in range(0, len(to_validate), batch_size):
        batch = to_validate[i : i + batch_size]
        tasks = [_validate_and_track(record) for record in batch]
        results = await asyncio.gather(*tasks)
        _append_jsonl(results, validated_path)

    pbar.close()

    all_validated = _read_jsonl(validated_path)
    stats = _summarize_validation(all_validated)
    logger.info(f"Phase 2 COMPLETE: {stats}")
    return stats


def _summarize_validation(records: list[dict]) -> dict:
    """Summarize validation results."""
    total = len(records)
    passed = sum(1 for r in records if r.get("passed"))
    failed = total - passed
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / max(total, 1), 3),
    }


# ---------------------------------------------------------------------------
# PHASE 3: REGENERATE — Fix failed sentences, re-validate, append passes
# ---------------------------------------------------------------------------

async def _regenerate_sentence(
    client: AsyncOpenAI,
    original: str,
    fail_reason: str,
    language_code: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Regenerate a failed sentence with corrective feedback."""
    language = LANGUAGES[language_code]
    system = load_system_prompt("regeneration", language=language)
    prompt = load_prompt("regeneration").format(
        language=language, original=original, fail_reason=fail_reason,
    )
    async with semaphore:
        response = await client.responses.parse(
            model=MODEL_GENERATE,
            instructions=system,
            input=prompt,
            text_format=RegenerationOutput,
            reasoning={"effort": "low"},
        )
    return response.output_parsed.corrected_sentence


async def _regenerate_and_validate_one(
    client: AsyncOpenAI,
    record: dict,
    language_code: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Regenerate a failed sentence up to MAX_REGENERATION_ATTEMPTS times.
    Each attempt: regenerate → validate. If it passes, return the corrected record.
    Regeneration is sequential per sentence (each attempt depends on previous failure).
    """
    current_sentence = record["sentence"]
    current_reason = record.get("fail_reason", "unknown error")
    all_reasons = [current_reason]

    for attempt in range(1, MAX_REGENERATION_ATTEMPTS + 1):
        try:
            new_sentence = await _regenerate_sentence(
                client, current_sentence, current_reason, language_code, semaphore,
            )
            passed, reason = await _validate_sentence(
                client, new_sentence, language_code, semaphore,
            )

            if passed:
                result = dict(record)
                result["sentence"] = new_sentence
                result["passed"] = True
                result["fail_reason"] = ""
                result["regeneration_attempts"] = attempt
                result["original_sentence"] = record["sentence"]
                return result

            current_sentence = new_sentence
            current_reason = reason
            all_reasons.append(reason)

        except Exception as e:
            logger.warning(f"Regeneration attempt {attempt} error: {e}")
            all_reasons.append(f"error: {e}")

    # All attempts exhausted
    result = dict(record)
    result["regeneration_attempts"] = MAX_REGENERATION_ATTEMPTS
    result["all_fail_reasons"] = all_reasons
    result["passed"] = False
    return result


async def run_phase_regenerate(
    language_code: str,
    output_dir: str,
    max_concurrent: int,
):
    """
    Phase 3: Regenerate failed sentences with corrective feedback.

    Reads:  {output_dir}/validated_{lang}.jsonl
    Writes: {output_dir}/regenerated_{lang}.jsonl  (results of regeneration attempts)

    After this phase, the final pool of sentences comes from:
    - validated_{lang}.jsonl records where passed=True
    - regenerated_{lang}.jsonl records where passed=True
    """
    client = AsyncOpenAI(timeout=60.0, max_retries=3)
    semaphore = asyncio.Semaphore(max_concurrent)

    validated_path = os.path.join(output_dir, f"validated_{language_code}.jsonl")
    regenerated_path = os.path.join(output_dir, f"regenerated_{language_code}.jsonl")

    all_validated = _read_jsonl(validated_path)
    failures = [r for r in all_validated if not r.get("passed")]
    logger.info(f"Failed sentences to regenerate: {len(failures)}")

    if not failures:
        logger.info("Phase 3: no failures to regenerate")
        return {"attempted": 0, "recovered": 0}

    # Check what we already regenerated (for resume)
    already_regenerated = set()
    existing_regen = _read_jsonl(regenerated_path)
    for rec in existing_regen:
        already_regenerated.add(rec.get("original_sentence", rec["sentence"]))
    logger.info(f"Already regenerated: {len(already_regenerated)} sentences")

    to_regenerate = [
        f for f in failures
        if f["sentence"] not in already_regenerated
    ]
    logger.info(f"To regenerate: {len(to_regenerate)} sentences")

    if not to_regenerate:
        recovered = sum(1 for r in existing_regen if r.get("passed"))
        return {"attempted": len(existing_regen), "recovered": recovered}

    pbar = tqdm(total=len(to_regenerate), desc=f"Regenerate ({language_code})")

    # Process in smaller chunks — regeneration is heavier (2-3 API calls per sentence)
    chunk_size = max(1, MAX_CONCURRENT_REQUESTS // 3)
    for i in range(0, len(to_regenerate), chunk_size):
        chunk = to_regenerate[i : i + chunk_size]
        tasks = [
            _regenerate_and_validate_one(client, record, language_code, semaphore)
            for record in chunk
        ]
        results = await asyncio.gather(*tasks)
        _append_jsonl(results, regenerated_path)
        pbar.update(len(chunk))

    pbar.close()

    all_regen = _read_jsonl(regenerated_path)
    recovered = sum(1 for r in all_regen if r.get("passed"))
    still_failed = len(all_regen) - recovered
    logger.info(f"Phase 3 COMPLETE: {recovered} recovered, {still_failed} still failed out of {len(all_regen)} attempts")

    return {"attempted": len(all_regen), "recovered": recovered, "still_failed": still_failed}


# ---------------------------------------------------------------------------
# PHASE 4: FINALIZE — Dedup + leakage removal + final outputs
# ---------------------------------------------------------------------------

def jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def deduplicate(
    records: list[dict],
    existing_texts: set[str],
    threshold: float = 0.85,
) -> list[dict]:
    """
    Remove exact duplicates, fuzzy duplicates (Jaccard > threshold),
    and test set leakage.
    """
    seen_exact = set()
    accepted = []

    for record in records:
        text = record["sentence"]
        normalized = text.lower().strip()

        if normalized in seen_exact:
            continue

        if normalized in existing_texts:
            logger.debug(f"Test leakage removed: {text[:60]}...")
            continue

        is_fuzzy_dup = False
        for accepted_record in accepted:
            if jaccard_similarity(normalized, accepted_record["sentence"].lower()) > threshold:
                is_fuzzy_dup = True
                break

        if is_fuzzy_dup:
            continue

        seen_exact.add(normalized)
        accepted.append(record)

    return accepted


def run_phase_finalize(
    language_code: str,
    output_dir: str,
    test_texts_path: str,
) -> dict:
    """
    Phase 4: Collect all passed sentences, deduplicate, remove test leakage, save final output.

    Reads:
        {output_dir}/validated_{lang}.jsonl   (passed sentences from phase 2)
        {output_dir}/regenerated_{lang}.jsonl  (recovered sentences from phase 3)

    Writes:
        {output_dir}/synthetic_text_{lang}.jsonl  — validated + deduped, full metadata (HuggingFace release)
        {output_dir}/sentences_{lang}.txt          — plain text, one per line (TTS input)
        {output_dir}/generation_stats_{lang}.json  — full pipeline statistics
    """
    test_texts = load_test_texts(test_texts_path)
    logger.info(f"Test texts for leakage check: {len(test_texts)}")

    # Collect passed from validation
    validated_path = os.path.join(output_dir, f"validated_{language_code}.jsonl")
    all_validated = _read_jsonl(validated_path)
    passed_from_validation = [r for r in all_validated if r.get("passed")]
    failed_from_validation = [r for r in all_validated if not r.get("passed")]

    # Collect recovered from regeneration
    regenerated_path = os.path.join(output_dir, f"regenerated_{language_code}.jsonl")
    all_regenerated = _read_jsonl(regenerated_path)
    recovered = [r for r in all_regenerated if r.get("passed")]
    still_failed = [r for r in all_regenerated if not r.get("passed")]

    logger.info(
        f"Passed from validation: {len(passed_from_validation)}, "
        f"Recovered from regeneration: {len(recovered)}, "
        f"Still failed: {len(still_failed)}"
    )

    # Merge all passed sentences
    all_passed = passed_from_validation + recovered
    logger.info(f"Total passed before dedup: {len(all_passed)}")

    # Deduplicate + leakage removal
    deduped = deduplicate(all_passed, test_texts)
    logger.info(f"After deduplication + leakage removal: {len(deduped)}")

    # --- Save final outputs ---

    # Full JSONL with metadata (for HuggingFace release)
    jsonl_path = os.path.join(output_dir, f"synthetic_text_{language_code}.jsonl")
    _write_jsonl(deduped, jsonl_path)

    # Plain text (for TTS synthesis input)
    txt_path = os.path.join(output_dir, f"sentences_{language_code}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for record in deduped:
            f.write(record["sentence"] + "\n")

    # Per-category breakdown
    category_stats = {}
    for cat in CATEGORIES:
        cat_records = [r for r in deduped if r.get("category") == cat]
        category_stats[cat] = len(cat_records)

    # Full statistics
    stats = {
        "language": language_code,
        "total_raw": len(all_validated),
        "passed_validation": len(passed_from_validation),
        "failed_validation": len(failed_from_validation),
        "regeneration_attempted": len(all_regenerated),
        "regeneration_recovered": len(recovered),
        "regeneration_still_failed": len(still_failed),
        "total_passed": len(all_passed),
        "dedup_removed": len(all_passed) - len(deduped),
        "final_count": len(deduped),
        "pass_rate": round(len(passed_from_validation) / max(len(all_validated), 1), 3),
        "recovery_rate": round(len(recovered) / max(len(all_regenerated), 1), 3) if all_regenerated else 0.0,
        "per_category": category_stats,
    }

    stats_path = os.path.join(output_dir, f"generation_stats_{language_code}.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Final output: {jsonl_path} ({len(deduped)} sentences)")
    logger.info(f"Plain text for TTS: {txt_path}")
    logger.info(f"Statistics: {stats_path}")
    logger.info(f"Phase 4 COMPLETE: {stats}")

    return stats


# ---------------------------------------------------------------------------
# Orchestrator — run one or all phases
# ---------------------------------------------------------------------------

async def run_pipeline(
    language_code: str,
    seeds_path: str,
    test_texts_path: str,
    output_dir: str,
    phase: str = "all",
    target_per_category: int = 2000,
    skip_categories: Optional[list[str]] = None,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS,
):
    """
    Run one or all phases of the text diversification pipeline.

    Phases:
        generate    — Phase 1: Async LLM generation → raw JSONL files
        validate    — Phase 2: Async LLM-as-judge → validated JSONL
        regenerate  — Phase 3: Corrective regeneration of failures
        finalize    — Phase 4: Dedup + leakage removal → final output
        all         — Run all 4 phases sequentially
    """
    skip_categories = skip_categories or []
    seeds = load_seed_sentences(seeds_path) if seeds_path else []

    logger.info(f"Language: {LANGUAGES[language_code]} ({language_code})")
    logger.info(f"Phase: {phase}")
    logger.info(f"Seed sentences: {len(seeds)}")
    logger.info(f"Target per category: {target_per_category}")
    logger.info(f"Max concurrent requests: {max_concurrent}")
    if skip_categories:
        logger.info(f"Skipping categories: {skip_categories}")

    os.makedirs(output_dir, exist_ok=True)

    if phase in ("generate", "all"):
        logger.info("=" * 60)
        logger.info("PHASE 1: GENERATE")
        logger.info("=" * 60)
        await run_phase_generate(
            language_code, seeds, output_dir, target_per_category,
            skip_categories, max_concurrent,
        )

    if phase in ("validate", "all"):
        logger.info("=" * 60)
        logger.info("PHASE 2: VALIDATE")
        logger.info("=" * 60)
        await run_phase_validate(
            language_code, output_dir, skip_categories, max_concurrent,
        )

    if phase in ("regenerate", "all"):
        logger.info("=" * 60)
        logger.info("PHASE 3: REGENERATE")
        logger.info("=" * 60)
        await run_phase_regenerate(
            language_code, output_dir, max_concurrent,
        )

    if phase in ("finalize", "all"):
        logger.info("=" * 60)
        logger.info("PHASE 4: FINALIZE")
        logger.info("=" * 60)
        run_phase_finalize(
            language_code, output_dir, test_texts_path,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse synthetic text for TTS-based ASR augmentation",
    )
    parser.add_argument(
        "--language", type=str, required=True, choices=["et", "sl"],
        help="Language code: et (Estonian) or sl (Slovenian)",
    )
    parser.add_argument(
        "--phase", type=str, default="all", choices=PHASES,
        help="Pipeline phase to run (default: all). Phases: generate, validate, regenerate, finalize",
    )
    parser.add_argument(
        "--seeds", type=str, default="",
        help="Path to seed sentences file (one per line, extracted from CV17 train). Required for generate phase.",
    )
    parser.add_argument(
        "--test-texts", type=str, default="",
        help="Path to combined test/dev texts for leakage prevention (one per line)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/synthetic_text",
        help="Output directory (default: data/synthetic_text)",
    )
    parser.add_argument(
        "--target-per-category", type=int, default=2000,
        help="Target sentences per category (default: 2000, total ~6000/lang)",
    )
    parser.add_argument(
        "--skip-categories", type=str, nargs="*", default=[],
        choices=CATEGORIES,
        help="Categories to skip for ablation (e.g. --skip-categories morphological)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=MAX_CONCURRENT_REQUESTS,
        help=f"Max concurrent API requests (default: {MAX_CONCURRENT_REQUESTS})",
    )

    args = parser.parse_args()

    # Validate: generate phase requires seeds
    if args.phase in ("generate", "all") and not args.seeds:
        parser.error("--seeds is required for the generate phase")

    asyncio.run(
        run_pipeline(
            language_code=args.language,
            seeds_path=args.seeds,
            test_texts_path=args.test_texts,
            output_dir=args.output_dir,
            phase=args.phase,
            target_per_category=args.target_per_category,
            skip_categories=args.skip_categories,
            max_concurrent=args.max_concurrent,
        )
    )


if __name__ == "__main__":
    main()
