import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=30)  # duration in seconds

melody, sr = torchaudio.load('../data/synthetic-high/raw/high_tone_1062_phase_1.456.wav')
# generates using the melody from the given audio
wav, tokens = model.generate_continuation(melody[None], sr, return_tokens=True)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
