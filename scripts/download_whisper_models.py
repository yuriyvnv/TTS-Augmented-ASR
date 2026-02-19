from huggingface_hub import snapshot_download

REPO = "yuriyvnv/experiments_parakeet"

MODELS = [
    "whisper-large-v3-cv_only_sl-seed42",
    "whisper-large-v3-cv_synth_all_et-seed42",
    "whisper-large-v3-cv_synth_all_sl-seed42",
    "whisper-large-v3-cv_synth_no_morph_et-seed42",
    "whisper-large-v3-cv_synth_no_morph_sl-seed42",
    "whisper-large-v3-cv_synth_unfiltered_et-seed42",
    "whisper-large-v3-cv_synth_unfiltered_sl-seed42",
]

for model in MODELS:
    print(f"\n=== Downloading {model} (best model only, no checkpoints) ===")
    snapshot_download(
        repo_id=REPO,
        allow_patterns=f"{model}/*",
        ignore_patterns=f"{model}/checkpoint-*",
        local_dir="./results/whisper_models",
    )
    print(f"  Done: ./results/whisper_models/{model}")

print("\n=== All downloads complete ===")
