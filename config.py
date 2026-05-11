import numpy as np
import os
import logging

"""
AGI AUDIO SUBSTRATE - CONFIGURATION LAYER
-----------------------------------------
1. Physics: 48kHz, Float16 Precision, COLA Constraints.
2. Hardware: CPU Core Mapping, Chunk Sizes.
3. Formats: 360kbps (Default), Lossless (ALAC) M4A, 128kbps (Low).
"""

# ==========================================
# 1. PRECISION & DATA TYPES
# ==========================================
DTYPE_AUDIO = np.float16
DTYPE_COMPLEX = np.complex64
FFMPEG_FORMAT_STR = 'f32le' 
BYTES_PER_SAMPLE = 2

# ==========================================
# 2. THE PHYSICS OF SOUND (STFT)
# ==========================================
SAMPLE_RATE = 48000
N_FFT = 4096
HOP_LENGTH = N_FFT // 4
WIN_LENGTH = N_FFT
WINDOW_TYPE = 'hann'

# ==========================================
# 3. PARALLEL PROCESSING & CHUNKING
# ==========================================
CHUNK_DURATION_SEC = 30
CHUNK_SIZE_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)
PADDING_SAMPLES = N_FFT * 4

# ==========================================
# 4. OUTPUT MODES & COMPRESSION
# ==========================================
# Default quality outputs configuration
ENABLE_M4A_LOSSLESS = False
ENABLE_MP3_128K = False
ENABLE_MP3_360K = True

DEFAULT_QUALITY = '360k'
LOSSLESS_ON = False # Only when turned on

QUALITY_MAP = {
    '360k': {
        'ext': '.mp3',
        'codec': 'libmp3lame',
        'bitrate': '360k'
    },
    '128k': {
        'ext': '.mp3',
        'codec': 'libmp3lame',
        'bitrate': '128k'
    },
    'lossless': {
        'ext': '.m4a',
        'codec': 'alac',
        'bitrate': None
    }
}

FFMPEG_BIN = 'ffmpeg'

# ==========================================
# 5. HARDWARE OPTIMIZATION
# ==========================================
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())

# ==========================================
# 6. VALIDATION
# ==========================================
logger = logging.getLogger(__name__)

def validate_physics():
    if N_FFT % HOP_LENGTH != 0:
        raise ValueError("CRITICAL: N_FFT is not divisible by HOP_LENGTH.")
    logger.debug("[SYSTEM] Physics Loaded. Precision: %s.", DTYPE_AUDIO.__name__)

validate_physics()
