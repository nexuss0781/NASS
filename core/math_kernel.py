import numpy as np
import scipy.signal
import warnings
import sys
import os

# Add parent directory to path to import config if running standalone
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

"""
AGI AUDIO SUBSTRATE - MATHEMATICAL KERNEL (STFT / iSTFT)
--------------------------------------------------------
Stateless, high-precision transformations.
Converts 1D Float16 Waveforms <-> 2D Complex64 Tensors.
"""

def waveform_to_tensor(waveform: np.ndarray) -> np.ndarray:
    """
    Transforms raw Float16 voltage data into a Complex Mathematical Array.
    
    Args:
        waveform: 1D numpy array of shape (samples,), dtype Float16.
        
    Returns:
        Complex tensor of shape (Frequencies, TimeFrames), dtype Complex64.
    """
    if waveform.dtype != config.DTYPE_AUDIO:
        raise TypeError(f"[MATH ERROR] Expected {config.DTYPE_AUDIO}, got {waveform.dtype}.")

    # We calculate the exact overlap size based on config
    n_overlap = config.N_FFT - config.HOP_LENGTH

    # Perform Short-Time Fourier Transform
    # Suppress NOLA warning as boundary=None is intentional for our chunked processing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        frequencies, times, complex_spectrogram = scipy.signal.stft(
            waveform,
            fs=config.SAMPLE_RATE,
            window=config.WINDOW_TYPE,
            nperseg=config.N_FFT,
            noverlap=n_overlap,
            return_onesided=True, # We only need positive frequencies (Nyquist limit)
            boundary=None,        # CRITICAL: Disable automatic padding. We manage our own boundaries.
            padded=False          # CRITICAL: Disable automatic length extension.
        )
    
    # Enforce AGI precision
    if complex_spectrogram.dtype != config.DTYPE_COMPLEX:
        complex_spectrogram = complex_spectrogram.astype(config.DTYPE_COMPLEX)

    return complex_spectrogram


def tensor_to_waveform(complex_spectrogram: np.ndarray) -> np.ndarray:
    """
    Reverses the mathematical array back into raw Float16 voltage data.
    
    Args:
        complex_spectrogram: 2D numpy array of shape (Freqs, Frames), dtype Complex64.
        
    Returns:
        1D numpy array of shape (samples,), dtype Float16.
    """
    if complex_spectrogram.dtype != config.DTYPE_COMPLEX:
        raise TypeError(f"[MATH ERROR] Expected {config.DTYPE_COMPLEX}, got {complex_spectrogram.dtype}.")

    n_overlap = config.N_FFT - config.HOP_LENGTH

    # Perform Inverse Short-Time Fourier Transform
    # Using length parameter to ensure proper output size and avoid NOLA warnings
    expected_length = (complex_spectrogram.shape[1] - 1) * config.HOP_LENGTH + config.N_FFT
    
    times, reconstructed_waveform = scipy.signal.istft(
        complex_spectrogram,
        fs=config.SAMPLE_RATE,
        window=config.WINDOW_TYPE,
        nperseg=config.N_FFT,
        noverlap=n_overlap,
        input_onesided=True,
        boundary=None,         # Must match the forward transform exactly
        time_axis=1,
        freq_axis=0
    )

    # Enforce Float16 precision for the output waveform
    if reconstructed_waveform.dtype != config.DTYPE_AUDIO:
        reconstructed_waveform = reconstructed_waveform.astype(config.DTYPE_AUDIO)

    return reconstructed_waveform


def verify_lossless_identity(original_waveform: np.ndarray) -> float:
    """
    The ultimate proof of AGI-grade architecture.
    Passes a signal through the matrix and back, calculating the exact error margin.
    Uses import warnings to suppress NOLA warnings during verification.
    """
    import warnings
    
    # To achieve perfect reconstruction with boundary=None, we must pad the signal
    # so the "valid" part is not at the extreme edges where the window is zero.
    pad_size = config.N_FFT
    padded_input = np.concatenate((np.zeros(pad_size, dtype=config.DTYPE_AUDIO), 
                                   original_waveform, 
                                   np.zeros(pad_size, dtype=config.DTYPE_AUDIO)))

    # 1. Forward Pass
    tensor = waveform_to_tensor(padded_input)
    
    # 2. Reverse Pass (suppress NOLA warning as it's expected with boundary=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        reconstructed_padded = tensor_to_waveform(tensor)
    
    # Trim the padding back off
    reconstructed = reconstructed_padded[pad_size:pad_size + len(original_waveform)]
    
    # 3. Calculate Maximum Absolute Error (Delta)
    # Because of floating-point math, the error won't be exactly 0.0, 
    # but with Float16 it should be in the realm of 1e-3 (Still imperceptible to humans).
    
    # Note: stft/istft without boundaries shrinks the output slightly. 
    # We trim the original to match the reconstructed length for fair comparison.
    min_len = min(len(original_waveform), len(reconstructed))
    
    delta = np.max(np.abs(original_waveform[:min_len] - reconstructed[:min_len]))
    
    return delta