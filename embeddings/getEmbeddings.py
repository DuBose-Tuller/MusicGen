import torch
import torchaudio
from audiocraft.models import MusicGen

# Load the model
model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.cfg['device'] = 'mps'

# Load and preprocess your audio file
def load_and_preprocess_audio(file_path, target_length=30 * 24000):  # 30 seconds at 24kHz
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample to 24kHz if necessary
    if sample_rate != 24000:
        waveform = torchaudio.transforms.Resample(sample_rate, 24000)(waveform)
    
    # Pad or trim to target length
    if waveform.shape[1] < target_length:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
    else:
        waveform = waveform[:, :target_length]
    
    return waveform


# Load and preprocess audio (using the function from the previous response)
audio_path = "../Datasets/ACPAS-dataset-synthetic-subset/subset_S/Albeniz/1_CPM_Espana_Opus_165_Prelude/S_1275_Kontakt_Giant_soft.wav"
waveform = load_and_preprocess_audio(audio_path).unsqueeze(0)


# Get the audio codes
with torch.no_grad():
    codes, scale = model.compression_model.encode(waveform.to(model.device))

print(codes.shape)

# Get the LM embeddings
with torch.no_grad():
    x = model.lm.transformer(codes)
    for layer in model.lm.transformer.layers:
        x = layer(x)
    lm_embedding = x  # This is the embedding from the last transformer layer

print(lm_embedding.shape)
