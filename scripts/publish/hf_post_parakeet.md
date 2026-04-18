Parakeet-TDT Goes Multilingual: 4 New ASR Models

Four fine-tuned versions of NVIDIA's Parakeet-TDT-0.6B-v3 for Dutch, Portuguese, Estonian, and Slovenian — among the first community fine-tunes of this architecture for non-English languages.

📊 Results on Common Voice 17 test sets:

🇸🇮 Slovenian: 50.49% → 11.56% WER (-77%)
🇵🇹 Portuguese: 15.86% → 10.71% WER (-32%)
🇪🇪 Estonian: 27.15% → 21.03% WER (-23%)
🇳🇱 Dutch: 5.99% → 5.33% WER (-11%)

All models output cased text with punctuation.

import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained(
    "yuriyvnv/parakeet-tdt-0.6b-dutch"
)
output = model.transcribe(["audio.wav"])
print(output[0].text)

🔗 Models:
🇳🇱 yuriyvnv/parakeet-tdt-0.6b-dutch
🇵🇹 yuriyvnv/parakeet-tdt-0.6b-portuguese
🇪🇪 yuriyvnv/parakeet-tdt-0.6b-estonian
🇸🇮 yuriyvnv/parakeet-tdt-0.6b-slovenian

🏗️ Training: Common Voice 17 + synthetic speech (OpenAI TTS), filtered with WAVe (yuriyvnv/WAVe-1B-Multimodal-PT) for quality. AdamW + cosine annealing, bf16-mixed precision, early stopping on val WER. Timestamps and long-form audio supported.

@hf-audio @NVIDIADev

#asr #speech #parakeet #nvidia #nemo #multilingual #fine-tuning #commonvoice
