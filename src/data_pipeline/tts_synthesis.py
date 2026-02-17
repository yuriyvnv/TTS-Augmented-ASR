"""
TTS Synthesis Pipeline for ASR Augmentation.

Synthesizes all unique sentences from the text diversification pipeline
using OpenAI gpt-4o-mini-tts with deterministic voice cycling across 11 voices.

Two sources of sentences:
  1. Raw sentences (Phase 1) — used for the unfiltered training manifest
  2. Regenerated replacement sentences (Phase 3, passed only) — used for filtered manifests

Audio is saved as WAV files at OpenAI's native sample rate (24kHz).
Resampling to 16kHz is handled downstream by the training frameworks:
  - HuggingFace: dataset.cast_column("audio", Audio(sampling_rate=16000))
  - NeMo: sample_rate config in training YAML

A JSONL manifest maps each sentence to its audio path with full metadata
(language, category, voice, duration, source).

Resumable: skips sentences whose audio already exists on disk.

Usage:
    uv run python -m src.data_pipeline.tts_synthesis --language et
    uv run python -m src.data_pipeline.tts_synthesis --language sl
    uv run python -m src.data_pipeline.tts_synthesis --language both
"""

import argparse
import asyncio
import hashlib
import json
import logging
import wave
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANGUAGES = {"et": "Estonian", "sl": "Slovenian"}

# All 11 built-in voices for gpt-4o-mini-tts
VOICES = [
    "alloy", "ash", "ballad", "coral", "echo", "fable",
    "nova", "onyx", "sage", "shimmer", "verse",
]

CATEGORIES = ["paraphrase", "domain", "morphological"]

TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_CONCURRENCY = 30  # Tier 3: safe at ~500 RPM

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SYNTHETIC_TEXT_DIR = DATA_DIR / "synthetic_text"

# TTS instructions per language
TTS_INSTRUCTIONS = {
    "et": "Speak naturally in Estonian with clear native pronunciation and natural prosody.",
    "sl": "Speak naturally in Slovenian with clear native pronunciation and natural prosody.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sentence_hash(sentence: str) -> str:
    """Deterministic short hash for a sentence, used for filenames."""
    return hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:16]


def voice_for_sentence(sentence: str) -> str:
    """Deterministic voice assignment via hash modulo."""
    h = int(hashlib.sha256(sentence.encode("utf-8")).hexdigest(), 16)
    return VOICES[h % len(VOICES)]


def collect_sentences(language_code: str) -> dict[str, dict]:
    """Collect all unique sentences that need TTS synthesis.

    Returns a dict mapping sentence text -> metadata dict.

    Two sources:
      1. raw_{lang}_{category}.jsonl — all Phase 1 generated sentences
      2. regenerated_{lang}.jsonl — only passed regenerations (replacement sentences)
    """
    sentences = {}

    # Source 1: All raw sentences
    for category in CATEGORIES:
        raw_path = SYNTHETIC_TEXT_DIR / f"raw_{language_code}_{category}.jsonl"
        if not raw_path.exists():
            logger.warning(f"Raw file not found: {raw_path}")
            continue

        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sent = record["sentence"]
                if sent not in sentences:
                    sentences[sent] = {
                        "language": language_code,
                        "category": record.get("category", category),
                        "source": "raw",
                        "domain": record.get("domain", ""),
                    }

    # Source 2: Regenerated sentences that passed validation
    regen_path = SYNTHETIC_TEXT_DIR / f"regenerated_{language_code}.jsonl"
    if regen_path.exists():
        with open(regen_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not record.get("passed", False):
                    continue
                sent = record["sentence"]
                if sent not in sentences:
                    sentences[sent] = {
                        "language": language_code,
                        "category": record.get("category", ""),
                        "source": "regenerated",
                        "domain": record.get("domain", ""),
                    }
    else:
        logger.warning(f"Regenerated file not found: {regen_path}")

    return sentences


def get_audio_path(output_dir: Path, sentence: str) -> Path:
    """Get the WAV file path for a sentence."""
    return output_dir / f"{sentence_hash(sentence)}.wav"


def wav_duration_seconds(wav_path: Path) -> float:
    """Get duration of a WAV file in seconds.

    OpenAI's streaming WAV sets nframes=0x7FFFFFFF (max int32) as a placeholder,
    so we compute duration from file size and audio parameters instead.
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
        # Compute from actual data size (file size minus 44-byte WAV header)
        data_bytes = wav_path.stat().st_size - 44
        return max(0.0, data_bytes / (sr * ch * sw))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# TTS Synthesis
# ---------------------------------------------------------------------------


async def synthesize_one(
    client: AsyncOpenAI,
    sentence: str,
    voice: str,
    language_code: str,
    output_path: Path,
) -> float:
    """Synthesize a single sentence and save as WAV (native sample rate).

    Returns duration in seconds, or 0.0 on failure.
    """
    try:
        response = await client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=sentence,
            instructions=TTS_INSTRUCTIONS[language_code],
            response_format="wav",
        )

        # Async client: must use aread() to get the full response body
        wav_bytes = await response.aread()

        # Save raw WAV from OpenAI (24kHz) — resampling handled by training frameworks
        output_path.write_bytes(wav_bytes)

        return wav_duration_seconds(output_path)

    except Exception as e:
        logger.error(f"TTS failed for '{sentence[:50]}...': {e}")
        return 0.0


async def run_tts_synthesis(language_code: str, concurrency: int = DEFAULT_CONCURRENCY):
    """Main TTS synthesis pipeline for one language."""
    language_name = LANGUAGES[language_code]
    output_dir = DATA_DIR / "synthetic_audio" / language_code
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = DATA_DIR / "synthetic_audio" / f"tts_manifest_{language_code}.jsonl"

    # Collect all unique sentences
    logger.info(f"Collecting sentences for {language_name}...")
    all_sentences = collect_sentences(language_code)
    logger.info(f"Total unique sentences: {len(all_sentences)}")

    # Load already-synthesized entries from manifest (for resume)
    already_done = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sent = record.get("sentence", "")
                audio_path = Path(record.get("audio_path", ""))
                if sent and audio_path.exists():
                    already_done.add(sent)
        logger.info(f"Already synthesized: {len(already_done)} sentences (resuming)")

    # Filter to pending sentences
    pending = {s: meta for s, meta in all_sentences.items() if s not in already_done}
    logger.info(f"Pending synthesis: {len(pending)} sentences")

    if not pending:
        logger.info("Nothing to synthesize. Done.")
        return

    # Initialize client
    client = AsyncOpenAI(timeout=120.0, max_retries=3)
    semaphore = asyncio.Semaphore(concurrency)
    logger.info(f"Concurrency: {concurrency} parallel requests")

    # Open manifest for appending
    manifest_file = open(manifest_path, "a", encoding="utf-8")

    # Track failed sentences for debugging
    failed_path = DATA_DIR / "synthetic_audio" / f"tts_failed_{language_code}.jsonl"
    failed_file = open(failed_path, "a", encoding="utf-8")

    pbar = tqdm(total=len(pending), desc=f"TTS {language_code}", unit="sent")

    total_duration = 0.0
    success_count = 0
    fail_count = 0

    async def process_one(sentence: str, metadata: dict):
        nonlocal total_duration, success_count, fail_count

        voice = voice_for_sentence(sentence)
        audio_path = get_audio_path(output_dir, sentence)

        async with semaphore:
            if audio_path.exists():
                duration = wav_duration_seconds(audio_path)
            else:
                duration = await synthesize_one(
                    client, sentence, voice, language_code, audio_path,
                )

        if duration > 0:
            record = {
                "sentence": sentence,
                "audio_path": str(audio_path),
                "language": language_code,
                "category": metadata.get("category", ""),
                "source": metadata.get("source", ""),
                "domain": metadata.get("domain", ""),
                "voice": voice,
                "duration_seconds": round(duration, 3),
            }
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest_file.flush()
            total_duration += duration
            success_count += 1
        else:
            failed_file.write(json.dumps({"sentence": sentence, **metadata}, ensure_ascii=False) + "\n")
            failed_file.flush()
            fail_count += 1

        pbar.update(1)

    tasks = [process_one(sentence, metadata) for sentence, metadata in pending.items()]
    await asyncio.gather(*tasks)

    pbar.close()
    manifest_file.close()
    failed_file.close()

    # Summary
    hours = total_duration / 3600
    logger.info(
        f"\n{'='*60}\n"
        f"TTS Synthesis Complete — {language_name}\n"
        f"{'='*60}\n"
        f"  Synthesized:  {success_count} sentences\n"
        f"  Failed:       {fail_count} sentences\n"
        f"  Total audio:  {total_duration:.1f}s ({hours:.2f} hours)\n"
        f"  Manifest:     {manifest_path}\n"
        f"  Audio dir:    {output_dir}\n"
        f"{'='*60}"
    )

    # Save summary stats
    stats_path = DATA_DIR / "synthetic_audio" / f"tts_stats_{language_code}.json"
    stats = {
        "language": language_code,
        "total_unique_sentences": len(all_sentences),
        "already_done_before_run": len(already_done),
        "synthesized_this_run": success_count,
        "failed_this_run": fail_count,
        "total_duration_seconds": round(total_duration, 1),
        "total_duration_hours": round(hours, 2),
        "voices_used": VOICES,
        "model": TTS_MODEL,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved to {stats_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize TTS audio for all unique sentences"
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["et", "sl", "both"],
        help="Language to synthesize",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max parallel API requests (default: {DEFAULT_CONCURRENCY})",
    )
    args = parser.parse_args()

    langs = ["et", "sl"] if args.language == "both" else [args.language]

    async def run_all():
        for lang in langs:
            await run_tts_synthesis(lang, concurrency=args.concurrency)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
