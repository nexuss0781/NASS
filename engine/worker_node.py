import numpy as np
import multiprocessing
import warnings
import sys
import os

# Import substrate modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core import math_kernel
from engine.shared_memory import SharedTensor

"""
AGI AUDIO SUBSTRATE - WORKER NODE (PROCESSOR CORE)
--------------------------------------------------
Independent, GIL-free processing unit.
Attaches to RAM -> Transforms -> (AGI Hook) -> Reconstructs -> Trims Overlaps.
"""

def process_chunk(
    worker_id: int, 
    input_shm_name: str, 
    output_shm_name: str, 
    input_shape: tuple, 
    output_shape: tuple
):
    """
    The isolated execution environment for a single CPU core.
    
    Args:
        worker_id: The ID of the core (for logging/debugging).
        input_shm_name: OS-level pointer to the raw audio chunk in RAM.
        output_shm_name: OS-level pointer to the processed output RAM block.
        input_shape: The exact dimensions of the padded input waveform.
        output_shape: The exact dimensions of the trimmed output waveform.
    """
    try:
        # 1. ATTACH TO ZERO-COPY MEMORY (O(1) latency)
        # is_creator=False because the Orchestrator already allocated the RAM.
        input_tensor = SharedTensor(
            name=input_shm_name, 
            shape=input_shape, 
            dtype=config.DTYPE_AUDIO, 
            is_creator=False
        )
        
        output_tensor = SharedTensor(
            name=output_shm_name, 
            shape=output_shape, 
            dtype=config.DTYPE_AUDIO, 
            is_creator=False
        )

        # 2. READ RAW VOLTAGE (Extracting the local array from RAM)
        raw_waveform = input_tensor.data.copy()
        # We copy into local CPU cache here because we are about to mutate it heavily 
        # through matrix operations, and we don't want to lock the shared bus.

        # 3. TRANSFORM TO MATHEMATICAL ARRAY (Time -> Frequency Domain)
        complex_spectrogram = math_kernel.waveform_to_tensor(raw_waveform)

        # =====================================================================
        # [ AGI INTERVENTION HOOK ]
        # This is where your neural networks will live. 
        # complex_spectrogram is a 2D Float16/Complex64 Matrix.
        # You can mutate Phase, Magnitude, or perform Latent Space morphing here.
        # For now, it passes through untouched to guarantee lossless proof.
        # =====================================================================

        # 4. RECONSTRUCT TO WAVEFORM (Frequency -> Time Domain)
        # Suppress NOLA warning as boundary=None is intentional for our chunked processing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            reconstructed_waveform = math_kernel.tensor_to_waveform(complex_spectrogram)

        # 5. OVERLAP TRIMMING (The "Seam" Fixer)
        # The input chunk had config.PADDING_SAMPLES on the left and right.
        # The FFT window distorts the edges of the audio (boundary effects).
        # We slice off the contaminated edges, leaving ONLY the mathematically perfect center.
        
        valid_start = config.PADDING_SAMPLES
        valid_end = valid_start + output_shape[0] # Exact chunk size
        
        pristine_audio = reconstructed_waveform[valid_start:valid_end]

        # 6. WRITE TO OUTPUT BRIDGE
        # This instantly overwrites the RAM block the Main Process is watching.
        output_tensor.data[:] = pristine_audio[:]

        # 7. DETACH FROM MEMORY
        input_tensor.close()
        output_tensor.close()

    except Exception as e:
        # In multiprocessing, standard exceptions get swallowed. We must explicitly catch and print.
        print(f"[WORKER {worker_id} FATAL ERROR] {str(e)}")
        raise e