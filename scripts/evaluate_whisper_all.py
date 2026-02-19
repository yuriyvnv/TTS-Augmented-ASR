import subprocess
import sys

MODELS = [
    # (language, model_path, label)
    ("et", None, "zero-shot"),
    ("sl", None, "zero-shot"),
    ("sl", "./results/whisper_models/whisper-large-v3-cv_only_sl-seed42", "cv_only_sl"),
    ("et", "./results/whisper_models/whisper-large-v3-cv_synth_all_et-seed42", "cv_synth_all_et"),
    ("sl", "./results/whisper_models/whisper-large-v3-cv_synth_all_sl-seed42", "cv_synth_all_sl"),
    ("et", "./results/whisper_models/whisper-large-v3-cv_synth_no_morph_et-seed42", "cv_synth_no_morph_et"),
    ("sl", "./results/whisper_models/whisper-large-v3-cv_synth_no_morph_sl-seed42", "cv_synth_no_morph_sl"),
    ("et", "./results/whisper_models/whisper-large-v3-cv_synth_unfiltered_et-seed42", "cv_synth_unfiltered_et"),
    ("sl", "./results/whisper_models/whisper-large-v3-cv_synth_unfiltered_sl-seed42", "cv_synth_unfiltered_sl"),
]

TEST_SETS = ["cv17_validation", "cv17_test", "fleurs_test"]

total = len(MODELS)
for i, (lang, model_path, label) in enumerate(MODELS, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/{total}] Evaluating: {label} ({lang})")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-m", "src.evaluation.evaluate",
        "--model", "whisper-large-v3",
        "--language", lang,
        "--test-sets", *TEST_SETS,
    ]
    if model_path:
        cmd.extend(["--model-path", model_path])

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: {label} ({lang}) failed with exit code {result.returncode}")
    else:
        print(f"=== {label} ({lang}) done ===")

print(f"\n{'='*60}")
print(f"All {total} evaluations complete!")
print(f"{'='*60}")
