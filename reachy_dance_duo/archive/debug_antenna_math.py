
import numpy as np
import logging

# Mock dependencies
class MockConfig:
    def __init__(self):
        self.antenna_sensitivity = 3.0
        self.antenna_amplitude = 3.15
        self.antenna_rest_position = 0.1
        self.antenna_drop_max = 3.15 # Legacy support if needed

class MockAnalysis:
    def __init__(self):
        # Create dummy envelopes
        self.energy_envelope = np.array([0.052]) # 0.052 * 5.0 pre-gain ~= 0.26
        self.onset_envelope = np.array([0.0]) # Sustained (no onset)

class MockChoreographer:
    def __init__(self):
        self.config = MockConfig()
        self.analysis = MockAnalysis()
        
    def _get_continuous_antennas(self, current_time: float):
        # Copy-paste logic from current choreographer.py to verify ISOLATED MATH
        
        idx = 0 
        
        # 1. RAW RMS (Volume) - User requested NO normalization
        # HOWEVER: Raw RMS is typically 0.05-0.2. It needs a linear gain boost to be usable.
        # We apply a fixed 5x Pre-Gain. This is NOT normalization (clamping), just amplification.
        rms_val = self.analysis.energy_envelope[idx]
        norm_rms = rms_val * 5.0
        
        # 2. RAW ONSET (Attack) - User requested NO normalization
        onset_val = 0.0
        if idx < len(self.analysis.onset_envelope):
            onset_val = self.analysis.onset_envelope[idx]
            
        # Onset strength can be > 1.0, but we just let it ride.
        norm_onset = onset_val
        
        # 3. MIX SIGNALS [BOOST LOGIC]
        # User wants RAW POWER. Previous logic penalized sustain by 60%.
        # New Logic: Base = RMS. Onset = Multiplier Boost.
        # Combined = RMS * (1.0 + Onset)
        # Sustain (Onset=0): 100% RMS.
        # Hit (Onset=1): 200% RMS.
        
        combined = norm_rms * (1.0 + norm_onset)
        
        # Apply Sensitivity/Gain from Config (User Slider)
        # This is now the ONLY scaling factor. Default 1.0, Max 3.0.
        print(f"Combined Before Sensitivity: {combined}")
        combined = combined * self.config.antenna_sensitivity
        
        # Clip only at the very end to stay in 0.0-1.0 range for the mapper
        normalized_energy = np.clip(combined, 0.0, 1.0) 

        # Map to antenna splay
        # Rest: -0.1 (Left), 0.1 (Right)
        # Max Amplitude: controlled by slider (default ~1.5)
        rest = self.config.antenna_rest_position
        max_travel = self.config.antenna_amplitude
        
        splay = max_travel * normalized_energy
        
        left = rest - splay
        right = -rest + splay
        
        print(f"RMS Raw: {rms_val}")
        print(f"Pre-Gained RMS: {norm_rms}")
        print(f"Norm Onset: {norm_onset}")
        print(f"After Sensitivity ({self.config.antenna_sensitivity}x): {combined}")
        print(f"Normalized Energy (Clipped): {normalized_energy}")
        print(f"Max Travel (Amp): {max_travel}")
        print(f"Splay Result: {splay} radians")
        print(f"Splay Result (Degrees): {splay * 180 / 3.14159}")
        
        return np.array([left, right])

if __name__ == "__main__":
    print("--- Simulating Antenna Logic ---")
    choreo = MockChoreographer()
    choreo._get_continuous_antennas(0.0)
