import torch
import torchaudio
import numpy as np
from typing import List, Tuple, Optional
import math

class RhythmGenerator:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.base_frequency = 440  # A4 note
        
    def generate_tone(self, duration: float, amplitude: float = 0.5, accent_level: int = 1) -> torch.Tensor:
        """Generate a single tone with a simple envelope.
        
        Args:
            duration: Duration of the tone in seconds
            amplitude: Base amplitude of the tone
            accent_level: Level of accent (3=sam, 2=tali, 1=regular, 0=khali)
        """
        num_samples = int(duration * self.sample_rate)
        t = torch.arange(num_samples, dtype=torch.float32) / self.sample_rate
        
        # Different characteristics for each accent level
        if accent_level == 3:  # Sam - strongest accent
            frequency = self.base_frequency * 0.75  # Lower pitch
            amplitude *= 2.0  # Loudest
            attack = int(0.015 * self.sample_rate)  # 15ms attack
            decay = int(0.15 * self.sample_rate)    # 150ms decay
            release = int(0.15 * self.sample_rate)  # 150ms release
        elif accent_level == 2:  # Tali - strong subdivision
            frequency = self.base_frequency * 0.8   # Slightly lower pitch
            amplitude *= 1.5  # Louder
            attack = int(0.01 * self.sample_rate)   # 10ms attack
            decay = int(0.1 * self.sample_rate)     # 100ms decay
            release = int(0.1 * self.sample_rate)   # 100ms release
        elif accent_level == 1:  # Regular beat
            frequency = self.base_frequency
            attack = int(0.005 * self.sample_rate)  # 5ms attack
            decay = int(0.05 * self.sample_rate)    # 50ms decay
            release = int(0.05 * self.sample_rate)  # 50ms release
        else:  # Khali (weak beat)
            frequency = self.base_frequency * 1.2   # Slightly higher pitch
            amplitude *= 0.7  # Quieter
            attack = int(0.003 * self.sample_rate)  # 3ms attack
            decay = int(0.03 * self.sample_rate)    # 30ms decay
            release = int(0.03 * self.sample_rate)  # 30ms release
        
        # Generate sine wave
        signal = amplitude * torch.sin(2 * math.pi * frequency * t)
        
        # Apply envelope
        envelope = torch.ones(num_samples)
        # Attack
        envelope[:attack] = torch.linspace(0, 1, attack)
        # Decay
        envelope[attack:attack+decay] = torch.linspace(1, 0.7, decay)
        # Release
        envelope[-release:] = torch.linspace(0.7, 0, release)
        
        return signal * envelope
    
    def generate_western_rhythm(self, pattern: List[int], repeats: int = 4, tempo: int = 120) -> torch.Tensor:
        """Generate a Western rhythm with regular beat divisions.
        
        Args:
            pattern: List where 2=strong beat, 1=weak beat, 0=rest
            repeats: Number of times to repeat the pattern
            tempo: Beats per minute
        """
        beat_duration = 60 / tempo  # duration of one beat in seconds
        tone_duration = 0.3  # duration of each tone in seconds
        
        # Repeat the pattern
        extended_pattern = pattern * repeats
        
        # Initialize empty audio
        total_duration = beat_duration * len(extended_pattern)
        total_samples = int(total_duration * self.sample_rate)
        audio = torch.zeros(total_samples)
        
        # Add tones at beat positions
        for i, beat in enumerate(extended_pattern):
            if beat > 0:
                start_sample = int(i * beat_duration * self.sample_rate)
                tone = self.generate_tone(tone_duration, accent=(beat == 2))
                end_sample = min(start_sample + len(tone), total_samples)
                audio[start_sample:end_sample] += tone[:end_sample-start_sample]
        
        return audio
    
    def generate_odd_ratio_rhythm(self, ratio: Tuple[int, int], cycles: int = 8, tempo: int = 120) -> torch.Tensor:
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
        tone_duration = 0.3
        
        for cycle in range(cycles):
            # Add long beat (accented)
            start_sample = int((cycle * cycle_duration) * self.sample_rate)
            tone = self.generate_tone(tone_duration, accent=True)
            end_sample = min(start_sample + len(tone), total_samples)
            audio[start_sample:end_sample] += tone[:end_sample-start_sample]
            
            # Add short beat (unaccented)
            start_sample = int((cycle * cycle_duration + long_dur) * self.sample_rate)
            tone = self.generate_tone(tone_duration, accent=False)
            end_sample = min(start_sample + len(tone), total_samples)
            audio[start_sample:end_sample] += tone[:end_sample-start_sample]
        
        return audio
    
    def generate_tala_pattern(self, 
                            pattern: List[int], 
                            cycle_length: int, 
                            repeats: int = 4, 
                            tempo: int = 120) -> torch.Tensor:
        """Generate a cyclic tala-like rhythm pattern.
        
        Args:
            pattern: List where:
                3=sam (first beat, strongest)
                2=tali (subdivision marker, strong)
                1=regular beat
                0=khali (empty beat, weak but marked)
                -1=rest (complete silence)
            cycle_length: Number of beats in one cycle
            repeats: Number of times to repeat the full cycle
            tempo: Base tempo in BPM
        """
        assert len(pattern) % cycle_length == 0, "Pattern length must be divisible by cycle length"
        
        beat_duration = 60 / tempo
        tone_duration = 0.3
        
        # Repeat the pattern
        extended_pattern = pattern * repeats
        
        # Initialize audio
        total_duration = beat_duration * len(extended_pattern)
        total_samples = int(total_duration * self.sample_rate)
        audio = torch.zeros(total_samples)
        
        # Generate pattern
        for i, beat in enumerate(extended_pattern):
            if beat > 0:
                start_sample = int(i * beat_duration * self.sample_rate)
                # Sam (first beat of cycle) is accented
                is_sam = beat == 2
                tone = self.generate_tone(tone_duration, accent=is_sam)
                end_sample = min(start_sample + len(tone), total_samples)
                audio[start_sample:end_sample] += tone[:end_sample-start_sample]
        
        return audio

def generate_dataset(output_dir: str, num_samples: int = 100):
    """Generate a dataset of various rhythmic patterns."""
    generator = RhythmGenerator()
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Western patterns (2=strong beat, 1=weak beat, 0=rest)
    western_patterns = [
        [2, 1, 2, 1],  # Simple duple (2/4)
        [2, 1, 1, 2, 1, 1],  # Simple triple (3/4)
        [2, 1, 1, 1],  # Simple quadruple (4/4) with emphasis on 1
        [2, 1, 2, 1, 2],  # Simple quintuple (5/4)
    ]
    
    # 2. Odd ratio patterns
    odd_ratios = [
        (3, 2),  # 3:2 ratio
        (5, 3),  # 5:3 ratio
        (7, 4),  # 7:4 ratio
        (4, 3),  # 4:3 ratio
    ]
    
    # 3. Tala patterns
    # 3=sam (strongest), 2=tali (subdivision), 1=regular beat, 0=khali (weak), -1=rest
    tala_patterns = [
        # Teentaal (16 beats): 4+4+4+4
        ([3, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 0, 1, 1, 1], 16),  # Dha dhin dhin dha | Dha dhin dhin dha | Dha dhin dhin dha | Dha dhin dhin dha
        
        # Jhaptaal (10 beats): 2+3+2+3
        ([3, 1, 2, 1, 1, 2, 1, 2, 1, 1], 10),  # Dhi na | Dhi dhin na | Thi na | Dhi dhin na
        
        # Rupak (7 beats): 3+2+2
        ([3, 1, 1, 2, 1, 2, 1], 7),  # Tin tin na | Dhin na | Dhin na
    ]
    
    # Generate samples with variations
    for i in range(num_samples):
        # Western patterns
        pattern = western_patterns[i % len(western_patterns)]
        tempo = np.random.randint(80, 140)
        audio = generator.generate_western_rhythm(pattern, repeats=4, tempo=tempo)
        torchaudio.save(
            f"{output_dir}/western_pattern_{i}.wav",
            audio.unsqueeze(0),
            generator.sample_rate
        )
        
        # Odd ratio patterns
        ratio = odd_ratios[i % len(odd_ratios)]
        cycles = np.random.randint(6, 10)
        audio = generator.generate_odd_ratio_rhythm(ratio, cycles, tempo)
        torchaudio.save(
            f"{output_dir}/odd_ratio_{i}.wav",
            audio.unsqueeze(0),
            generator.sample_rate
        )
        
        # Tala patterns
        pattern, cycle_len = tala_patterns[i % len(tala_patterns)]
        audio = generator.generate_tala_pattern(pattern, cycle_len, repeats=4, tempo=tempo)
        torchaudio.save(
            f"{output_dir}/tala_pattern_{i}.wav",
            audio.unsqueeze(0),
            generator.sample_rate
        )

if __name__ == "__main__":
    # Example usage
    generator = RhythmGenerator()
    
    # Generate a Western rhythm (4/4 time with accents)
    pattern = [2, 1, 1, 1]  # Strong beat on 1, weak beats on 2,3,4
    audio = generator.generate_western_rhythm(pattern, repeats=4, tempo=120)
    torchaudio.save("western_rhythm.wav", audio.unsqueeze(0), generator.sample_rate)
    
    # Generate an odd ratio rhythm (3:2)
    audio = generator.generate_odd_ratio_rhythm((3, 2), cycles=8, tempo=120)
    torchaudio.save("odd_ratio_rhythm.wav", audio.unsqueeze(0), generator.sample_rate)
    
    # Generate a tala pattern (Teentaal - 16 beats)
    pattern = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Sam on first beat
    audio = generator.generate_tala_pattern(pattern, 16, repeats=4, tempo=120)
    torchaudio.save("tala_rhythm.wav", audio.unsqueeze(0), generator.sample_rate)
    
    # Generate a full dataset
    generate_dataset("rhythm_dataset", num_samples=100)