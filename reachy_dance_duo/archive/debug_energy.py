import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from reachy_dance_suite.behaviors.choreographer import SongAnalyzer

def analyze_energy(file_path):
    print(f"Analyzing: {file_path}")
    
    analyzer = SongAnalyzer()
    analysis = analyzer.analyze(file_path)
    
    print(f"\nDuration: {analysis.duration:.2f}s")
    print(f"Energy Envelope Size: {analysis.energy_envelope.size}")
    
    if analysis.energy_envelope.size == 0:
        print("Error: Empty energy envelope!")
        return

    # Calculate stats exactly as Choreographer does
    rms_min = np.percentile(analysis.energy_envelope, 5)
    rms_max = np.percentile(analysis.energy_envelope, 95)
    
    if rms_max <= rms_min:
        rms_max = rms_min + 0.1
        
    print(f"\nStats:")
    print(f"  Min RMS (5th %): {rms_min:.5f}")
    print(f"  Max RMS (95th %): {rms_max:.5f}")
    print(f"  Mean RMS: {np.mean(analysis.energy_envelope):.5f}")
    print(f"  Max Absolute RMS: {np.max(analysis.energy_envelope):.5f}")

    # Onset Stats
    onset_env = analysis.onset_envelope
    onset_min = np.min(onset_env)
    onset_max = np.max(onset_env)
    if onset_max <= onset_min: onset_max = onset_min + 0.1
    
    print(f"\nOnset Stats:")
    print(f"  Min Onset: {onset_min:.5f}")
    print(f"  Max Onset: {onset_max:.5f}")

    # Simulate frames
    print("\nSimulating frames (Mixed Signal):")
    print(f"{'Time (s)':<8} | {'RMS':<8} | {'N.RMS':<6} | {'Onset':<8} | {'N.Onset':<7} | {'FINAL':<8}")
    print("-" * 75)
    
    # Sample every 5 seconds
    times = np.arange(0, analysis.duration, 2.0)
    for t in times:
        idx = int(t * analysis.envelope_sr / analysis.hop_length)
        if idx < len(analysis.energy_envelope):
            # RMS
            rms = analysis.energy_envelope[idx]
            d_rms = rms_max - rms_min
            if d_rms <= 1e-6: d_rms = 1.0
            n_rms = np.clip((rms - rms_min) / d_rms, 0.0, 1.0)
            
            # Onset
            onset = 0.0
            if idx < len(onset_env):
                onset = onset_env[idx]
            d_onset = onset_max - onset_min
            if d_onset <= 1e-6: d_onset = 1.0
            n_onset = np.clip((onset - onset_min) / d_onset, 0.0, 1.0)
            
            # Mix: Product (Modulation)
            # Signal = RMS (Volume) * Onset (Trigger)
            final = n_rms * n_onset
            final = np.clip(final, 0.0, 1.0)
            
            print(f"{t:<8.3f} | {n_rms:<8.2f} | {onset:<8.4f} | {n_onset:<8.2f} | {final:<8.4f}")
            
    # Sample every 0.05s (50ms) for 5 seconds to see the beat structure
    times = np.arange(10.0, 15.0, 0.05)
    for t in times:
        idx = int(t * analysis.envelope_sr / analysis.hop_length)
        if idx < len(analysis.energy_envelope):
            # Normalization logic repeated for the new loop
            rms = analysis.energy_envelope[idx]
            d_rms = rms_max - rms_min
            if d_rms <= 1e-6: d_rms = 1.0
            n_rms = np.clip((rms - rms_min) / d_rms, 0.0, 1.0)
            
            onset = 0.0
            if idx < len(onset_env):
                onset = onset_env[idx]
            d_onset = onset_max - onset_min
            if d_onset <= 1e-6: d_onset = 1.0
            n_onset = np.clip((onset - onset_min) / d_onset, 0.0, 1.0)

            final = n_rms * n_onset
            final = np.clip(final, 0.0, 1.0)
            
            print(f"{t:<8.3f} | {n_rms:<8.2f} | {onset:<8.4f} | {n_onset:<8.2f} | {final:<8.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file = sys.argv[1]
    else:
        file = "downloads/AOK [zDd-PnvTM20].wav"
    analyze_energy(file)
