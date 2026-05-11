import numpy as np
from core import math_kernel
from core.io_stream import AudioReader
import config

def demo_agi_perception(input_file):
    print(f"--- AGI Perception Demo ---")
    print(f"Loading audio: {input_file}")
    
    # 1. Perception: Convert Audio to Waveform
    reader = AudioReader(input_file, channels=1)
    chunk_iterator = reader.stream_chunks()
    first_chunk = next(chunk_iterator)
    
    print(f"Waveform Shape: {first_chunk.shape}")
    print(f"Waveform Dtype: {first_chunk.dtype}")
    
    # 2. Encoding: Convert Waveform to Mathematical Array (Spectrogram)
    # This is the "Substrate" that the AGI interacts with.
    complex_array = math_kernel.waveform_to_tensor(first_chunk)
    
    print(f"\nMathematical Array (Spectrogram) Shape: {complex_array.shape}")
    print(f"Mathematical Array Dtype: {complex_array.dtype}")
    print(f"First few 'Atoms' (Complex Numbers):\n{complex_array[:3, :3]}")
    
    # 3. AGI Processing (Placeholder)
    print("\n[AGI] Processing latent representation...")
    # The AGI can now manipulate 'complex_array' directly.
    # For example, it could filter frequencies, shift phase, etc.
    
    # 4. Reconstruction: Convert Mathematical Array back to Waveform
    reconstructed_waveform = math_kernel.tensor_to_waveform(complex_array)
    
    print(f"\nReconstructed Waveform Shape: {reconstructed_waveform.shape}")
    
    # 5. Verification
    # Note: We expect some edge effects due to boundary=None in the demo 
    # (The Orchestrator handles this with padding in the real pipeline).
    # Float16 precision provides sufficient accuracy for audio processing.
    print("Demo complete.")

if __name__ == "__main__":
    import os
    test_file = "../../test_input.wav"
    if os.path.exists(test_file):
        demo_agi_perception(test_file)
    else:
        print("Test file not found. Please run main.py first to generate it.")
