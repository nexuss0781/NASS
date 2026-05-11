import subprocess
import numpy as np
import sys
import os

# Import the laws of physics defined in config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

"""
AGI AUDIO SUBSTRATE - UNIVERSAL I/O STREAM
------------------------------------------
Input:  ANY Audio Format (Decoded to Float16 PCM)
Output: 360kbps (Default), Lossless (ALAC) M4A, 128kbps (Low)
"""

class AudioReader:
    def __init__(self, input_path: str, channels: int = 1):
        """
        Initializes the FFmpeg process to decode ANY audio file into raw Float16 PCM.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"[I/O ERROR] File not found: {input_path}")

        self.input_path = input_path
        self.channels = channels
        
        # Calculate exact byte size for one chunk of audio
        self.chunk_bytes = config.CHUNK_SIZE_SAMPLES * self.channels * config.BYTES_PER_SAMPLE

        # Subprocess Spawning: Direct binary pipe
        # We force -f f32le so FFmpeg handles the conversion from 
        # whatever the source was (mp3/wav) to our AGI standard (Float16 via Float32).
        # Note: FFmpeg outputs f32le, we convert to float16 in numpy
        command =[
            config.FFMPEG_BIN,
            '-hide_banner', '-loglevel', 'error',
            '-i', self.input_path,
            '-f', config.FFMPEG_FORMAT_STR,  # Force raw f32le output
            '-ac', str(self.channels),       # Channel count
            '-ar', str(config.SAMPLE_RATE),  # Enforce substrate sample rate
            'pipe:1'                         # Output to stdout
        ]

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # Prevent pipe deadlock
            bufsize=self.chunk_bytes * 2
        )

    def stream_chunks(self):
        """
        Generator that yields exact Float16 NumPy arrays representing audio chunks.
        Converts from Float32 to Float16 for memory efficiency.
        """
        while True:
            # Read exact number of bytes required for the mathematical chunk (as float32)
            raw_bytes = self.process.stdout.read(self.chunk_bytes * 2)  # f32 is 4 bytes, f16 is 2
            
            if not raw_bytes:
                break # End of stream
            
            # STATE FREEZE: Convert raw bytes to Float32 then to Float16
            audio_chunk = np.frombuffer(raw_bytes, dtype=np.float32).astype(config.DTYPE_AUDIO)
            
            # Padding for the final chunk if it's too short
            if len(audio_chunk) < config.CHUNK_SIZE_SAMPLES * self.channels:
                padding = np.zeros(
                    (config.CHUNK_SIZE_SAMPLES * self.channels) - len(audio_chunk), 
                    dtype=config.DTYPE_AUDIO
                )
                audio_chunk = np.concatenate((audio_chunk, padding))
                
            yield audio_chunk

        # Clean up process
        self.process.stdout.close()
        self.process.wait()


class AudioWriter:
    def __init__(self, output_path: str, channels: int = 1, quality: str = '360k'):
        """
        Initializes the FFmpeg process to encode raw Float16 PCM into the target format.
        """
        # Ensure the output path matches the configured extension
        filename, _ = os.path.splitext(output_path)
        
        quality_info = config.QUALITY_MAP.get(quality, config.QUALITY_MAP['360k'])
        self.output_path = filename + quality_info['ext']
        self.channels = channels

        # Reverse Subprocess: Accept raw Float16 (as Float32) via stdin, encode to target codec
        command =[
            config.FFMPEG_BIN,
            '-y',                            # Overwrite output file
            '-hide_banner', '-loglevel', 'error',
            '-f', config.FFMPEG_FORMAT_STR,  # Input format is f32le (from RAM)
            '-ac', str(self.channels),       
            '-ar', str(config.SAMPLE_RATE),  
            '-i', 'pipe:0',                  # Read from stdin
        ]
        
        # Add codec and bitrate settings
        if quality_info['codec'] == 'alac':
            command.extend(['-c:a', 'alac'])
        else:
            command.extend(['-c:a', quality_info['codec'], '-b:a', quality_info['bitrate']])
            
        command.append(self.output_path)

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # Prevent pipe deadlock
            bufsize=10**7 # Large buffer (10MB) to prevent pipeline stalling
        )

    def write_chunk(self, audio_array: np.ndarray):
        """
        Accepts a mathematically processed Float16 NumPy array and pushes it to the encoder.
        Converts to Float32 for FFmpeg compatibility.
        """
        if audio_array.dtype != config.DTYPE_AUDIO:
            raise TypeError(f"[I/O ERROR] Expected {config.DTYPE_AUDIO}, got {audio_array.dtype}.")
        
        # Convert Float16 to Float32 for FFmpeg
        audio_f32 = audio_array.astype(np.float32)
        
        # Push pure bytes to FFmpeg stdin
        try:
            self.process.stdin.write(audio_f32.tobytes())
        except BrokenPipeError:
            _, stderr = self.process.communicate()
            raise RuntimeError(f"[I/O ERROR] FFmpeg encoding failed: {stderr.decode()}")

    def close(self):
        """
        Finalizes the ALAC/AAC container.
        """
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait()
        print(f"[SYSTEM] Audio successfully frozen to container: {self.output_path}")
