import numpy as np
import torchaudio
import torch
import os

def generate_tone(freq, duration, sample_rate, phase_shift=0):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * freq * t + phase_shift)
    return tone

def save_waveform(waveform, filename, sample_rate):
    waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    print(filename)
    torchaudio.save(filename, waveform_tensor, sample_rate)

def main():
    # Parameters
    sample_rate = 32000  # 32 kHz for MusicGen
    duration = 15  # seconds
    num_samples = 16  # per group
    num_phase_shifts = 16  # number of phase-shifted versions per frequency

    # Frequency ranges
    low_freq_range = (130.81, 164.81)  # C3 to E3
    high_freq_range = (1046.5, 1318.51)  # C5 to E5

    # Create output directories
    os.makedirs("synthetic-low/raw", exist_ok=True)
    os.makedirs("synthetic-high/raw", exist_ok=True)

    # Generate low frequency group
    for i in range(num_samples):
        freq = np.random.uniform(*low_freq_range)
        for j in range(num_phase_shifts):
            phase_shift = np.random.uniform(0, 2 * np.pi)
            waveform = generate_tone(freq, duration, sample_rate, phase_shift)
            filename = f"synthetic-low/raw/low_tone_{int(freq):03d}_phase_{phase_shift:0.3f}.wav"
            save_waveform(waveform, filename, sample_rate)

    # Generate high frequency group
    for i in range(num_samples):
        freq = np.random.uniform(*high_freq_range)
        for j in range(num_phase_shifts):
            phase_shift = np.random.uniform(0, 2 * np.pi)
            waveform = generate_tone(freq, duration, sample_rate, phase_shift)
            filename = f"synthetic-high/raw/high_tone_{int(freq):03d}_phase_{phase_shift:0.3f}.wav"
            save_waveform(waveform, filename, sample_rate)

if __name__ == "__main__":
    main()