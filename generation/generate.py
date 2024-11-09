import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchaudio
from audiocraft_fork.audiocraft.models import MusicGen
from audiocraft_fork.audiocraft.data.audio import audio_write
# from audiocraft.models import MusicGen
# from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=15)  # duration in seconds

path = '../data/synthetic-high/s5-t5/high_tone_1062_phase_1.456_0.wav'

melody, sr = torchaudio.load(path)
# generates using the melody from the given audio
wav, tokens = model.generate_continuation(melody[None], sr, return_tokens=True, progress=True)
print(f"Final generated tokens shape: {tokens.shape}")

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(path.split('/')[-1], one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
