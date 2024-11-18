import torch
import torchaudio
import numpy as np
from typing import List, Tuple, Optional
import math

class RhythmGenerator:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.base_frequency = 440  # A4 note
        
    def generate_tone(self, duration: float, amplitude: float = 0.5) -> torch.Tensor:
        """Generate a single tone with a simple envelope."""
        num_samples = int(duration * self.sample_rate)
        t = torch.arange(num_samples, dtype=torch.float32) / self.sample_rate
        
        # Generate sine wave at A4
        signal = amplitude * torch.sin(2 * math.pi * self.base_frequency * t)
        
        # Apply envelope (simple ADSR)
        attack = int(0.005 * self.sample_rate)  # 5ms attack
        decay = int(0.05 * self.sample_rate)    # 50ms decay
        release = int(0.05 * self.sample_rate)  # 50ms release
        
        envelope = torch.ones(num_samples)
        # Attack
        envelope[:attack] = torch.linspace(0, 1, attack)
        # Decay
        envelope[attack:attack+decay] = torch.linspace(1, 0.7, decay)
        # Release
        envelope[-release:] = torch.linspace(0.7, 0, release)
        
        return signal * envelope
    
    def generate_western_rhythm(self, pattern: List[int], tempo: int = 120) -> torch.Tensor:
        """Generate a Western rhythm with regular beat divisions.
        
        Args:
            pattern: List of 1s and 0s representing beats and rests
            tempo: Beats per minute
        """
        beat_duration = 60 / tempo  # duration of one beat in seconds
        tone_duration = 0.2  # duration of each tone in seconds
        
        # Initialize empty audio
        total_duration = beat_duration * len(pattern)
        total_samples = int(total_duration * self.sample_rate)
        audio = torch.zeros(total_samples)
        
        # Add tones at beat positions
        for i, beat in enumerate(pattern):
            if beat:
                start_sample = int(i * beat_duration * self.sample_rate)
                tone = self.generate_tone(tone_duration)
                end_sample = min(start_sample + len(tone), total_samples)
                audio[start_sample:end_sample] += tone[:end_sample-start_sample]
        
        return audio
    
    def generate_odd_ratio_rhythm(self, ratio: Tuple[int, int], cycles: int = 4, tempo: int = 120) -> torch.Tensor:
        """Generate a rhythm with odd time ratios between beats.
        
        Args:
            ratio: Tuple of (long, short) beat durations
            cycles: Number of cycles to repeat
            tempo: Base tempo in BPM
        """
        base_duration = 60 / tempo  # duration of one beat in seconds
        long_dur = base_duration * ratio[0] / ratio[1]
        short_dur = base_duration
        
        # Calculate total duration and initialize audio
        cycle_duration = long_dur + short_dur
        total_duration = cycle_duration * cycles
        total_samples = int(total_duration * self.sample_rate)
        audio = torch.zeros(total_samples)
        
        # Generate each cycle
        tone_duration = 0.2
        tone = self.generate_tone(tone_duration)
        
        for cycle in range(cycles):
            # Add long beat
            start_sample = int((cycle * cycle_duration) * self.sample_rate)
            end_sample = min(start_sample + len(tone), total_samples)
            audio[start_sample:end_sample] += tone[:end_sample-start_sample]
            
            # Add short beat
            start_sample = int((cycle * cycle_duration + long_dur) * self.sample_rate)
            end_sample = min(start_sample + len(tone), total_samples)
            audio[start_sample:end_sample] += tone[:end_sample-start_sample]
        
        return audio
    
    def generate_tala_pattern(self, pattern: List[int], cycle_length: int, tempo: int = 120) -> torch.Tensor:
        """Generate a cyclic tala-like rhythm pattern.
        
        Args:
            pattern: List of 1s and 0s representing beats and rests
            cycle_length: Number of beats in one cycle
            tempo: Base tempo in BPM
        """
        assert len(pattern) % cycle_length == 0, "Pattern length must be divisible by cycle length"
        
        beat_duration = 60 / tempo
        tone_duration = 0.2
        
        # Initialize audio
        total_duration = beat_duration * len(pattern)
        total_samples = int(total_duration * self.sample_rate)
        audio = torch.zeros(total_samples)
        
        # Generate pattern
        tone = self.generate_tone(tone_duration)
        
        for i, beat in enumerate(pattern):
            if beat:
                start_sample = int(i * beat_duration * self.sample_rate)
                end_sample = min(start_sample + len(tone), total_samples)
                audio[start_sample:end_sample] += tone[:end_sample-start_sample]
        
        return audio

def generate_dataset(output_dir: str, num_samples: int = 100):
    """Generate a dataset of various rhythmic patterns."""
    generator = RhythmGenerator()
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Western patterns (common time signatures)
    western_patterns = [
        [1, 0, 1, 0],  # Simple duple
        [1, 0, 0, 1, 0, 0],  # Simple triple
        [1, 0, 1, 0, 1, 0, 1, 0],  # Common time
        [1, 0, 1, 0, 1],  # Simple quintuple
    ]
    
    # 2. Odd ratio patterns
    odd_ratios = [
        (3, 2),  # 3:2 ratio
        (5, 3),  # 5:3 ratio
        (7, 4),  # 7:4 ratio
        (4, 3),  # 4:3 ratio
    ]
    
    # 3. Tala-like patterns
    tala_patterns = [
        # Jhaptaal (10 beats)
        ([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 10),
        # Rupak (7 beats)
        ([1, 0, 1, 0, 1, 0, 1], 7),
        # Ada Chautaal (14 beats)
        ([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 14),
    ]
    
    # Generate samples with variations
    for i in range(num_samples):
        # Western patterns
        pattern = western_patterns[i % len(western_patterns)]
        tempo = np.random.randint(80, 160)
        audio = generator.generate_western_rhythm(pattern, tempo)
        torchaudio.save(
            f"{output_dir}/western_pattern_{i}.wav",
            audio.unsqueeze(0),
            generator.sample_rate
        )
        
        # Odd ratio patterns
        ratio = odd_ratios[i % len(odd_ratios)]
        cycles = np.random.randint(3, 7)
        audio = generator.generate_odd_ratio_rhythm(ratio, cycles, tempo)
        torchaudio.save(
            f"{output_dir}/odd_ratio_{i}.wav",
            audio.unsqueeze(0),
            generator.sample_rate
        )
        
        # Tala patterns
        pattern, cycle_len = tala_patterns[i % len(tala_patterns)]
        audio = generator.generate_tala_pattern(pattern, cycle_len, tempo)
        torchaudio.save(
            f"{output_dir}/tala_pattern_{i}.wav",
            audio.unsqueeze(0),
            generator.sample_rate
        )

if __name__ == "__main__":
    # Example usage
    generator = RhythmGenerator()
    
    # Generate a simple Western rhythm
    pattern = [1, 0, 1, 0, 1, 0, 1, 0]  # Common time
    audio = generator.generate_western_rhythm(pattern, tempo=120)
    torchaudio.save("western_rhythm.wav", audio.unsqueeze(0), generator.sample_rate)
    
    # Generate an odd ratio rhythm (3:2)
    audio = generator.generate_odd_ratio_rhythm((3, 2), cycles=4, tempo=120)
    torchaudio.save("odd_ratio_rhythm.wav", audio.unsqueeze(0), generator.sample_rate)
    
    # Generate a tala pattern (Jhaptaal - 10 beats)
    pattern = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    audio = generator.generate_tala_pattern(pattern, 10, tempo=120)
    torchaudio.save("tala_rhythm.wav", audio.unsqueeze(0), generator.sample_rate)
    
    # Generate a full dataset
    generate_dataset("rhythm_dataset", num_samples=100)
