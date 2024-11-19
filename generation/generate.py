import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchaudio
from audiocraft_fork.audiocraft.models import MusicGen
from audiocraft_fork.audiocraft.data.audio import audio_write
# from audiocraft.models import MusicGen
# from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=45)  # duration in seconds

path = '../data/rhythms/raw/tala_pattern/0.wav'

melody, sr = torchaudio.load(path)
# generates using the melody from the given audio
wav, tokens = model.generate_continuation(melody[None], sr, return_tokens=True, progress=True)
print(f"Final generated tokens shape: {tokens.shape}")


audio_write(path.split('/')[-1][:-4], wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
