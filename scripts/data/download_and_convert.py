from huggingface_hub import snapshot_download
import os
import torch
import nemo.collections.asr as nemo_asr

print("=== Downloading ===")
snapshot_download(
    repo_id="yuriyvnv/experiments_parakeet",
    allow_patterns="parakeet-tdt-cv_synth_unfiltered_sl_s42/**",
    local_dir="./results/download_tmp",
)

download_dir = "./results/download_tmp/parakeet-tdt-cv_synth_unfiltered_sl_s42"

print("=== Downloaded files ===")
for root, dirs, files in os.walk(download_dir):
    for f in files:
        print(os.path.join(root, f))

best_ckpt = None
best_wer = 999
for root, dirs, files in os.walk(download_dir):
    for f in files:
        if f.endswith(".ckpt") and "val_wer" in f and "last" not in f:
            wer = float(f.split("val_wer=")[1].replace(".ckpt", ""))
            if wer < best_wer:
                best_wer = wer
                best_ckpt = os.path.join(root, f)

if best_ckpt is None:
    print("ERROR: No checkpoint found!")
    exit(1)

print(f"\nBest checkpoint: {best_ckpt} (val_wer={best_wer})")

print("=== Loading base Parakeet model ===")
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")

print("=== Loading checkpoint weights ===")
ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"])

out_path = "./results/parakeet-tdt-cv_synth_unfiltered_sl-seed42.nemo"
print(f"=== Saving to {out_path} ===")
model.save_to(out_path)
print("Done!")
