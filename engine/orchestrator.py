import numpy as np
import concurrent.futures
import queue
import sys
import os
import time

# Import substrate modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.io_stream import AudioReader, AudioWriter
from engine.shared_memory import MemoryPool
from engine.worker_node import process_chunk
from core.visualizer import generate_visuals, save_mathematical_representation
from core.math_kernel import waveform_to_tensor

"""
AGI AUDIO SUBSTRATE - THE ORCHESTRATOR
--------------------------------------
Manages Lookahead Padding, Zero-Copy Memory allocation, and 
Strict Sequential Reassembly of parallel-processed chunks.
"""

class Orchestrator:
    def __init__(self, channels: int = 1):
        self.channels = channels
        
        # Determine optimal core count. Leave 1 core free for OS/IO overhead to prevent stutter.
        self.num_workers = max(1, os.cpu_count() - 1)
        print(f"[ORCHESTRATOR] Booting Hive Mind. CPU Cores mapped to AGI Substrate: {self.num_workers}")

        # The exact mathematical shape sent to the worker (Padding + Chunk + Padding)
        self.padded_input_shape = ((config.PADDING_SAMPLES * 2 + config.CHUNK_SIZE_SAMPLES) * self.channels,)
        
        # The exact mathematical shape returning from the worker (Just the Chunk)
        self.trimmed_output_shape = (config.CHUNK_SIZE_SAMPLES * self.channels,)

        # Allocate 2x the memory blocks as workers to allow asynchronous I/O read/write while CPU processes
        self.pool_size = self.num_workers * 2
        
        # Pre-allocate the OS-level RAM blocks
        self.input_memory = MemoryPool(self.pool_size, self.padded_input_shape, config.DTYPE_AUDIO, prefix="agi_input")
        self.output_memory = MemoryPool(self.pool_size, self.trimmed_output_shape, config.DTYPE_AUDIO, prefix="agi_output")

        # Queues to track which memory blocks are free to use in O(1) time
        self.free_blocks = queue.Queue()
        for i in range(self.pool_size):
            self.free_blocks.put(i)

    def process_file(self, input_path: str, output_path: str, quality_list=['360k']):
        """
        The main pipeline execution. Reads, parallelizes, transforms, and encodes.
        """
        reader = AudioReader(input_path, channels=self.channels)
        
        # Initialize multiple writers for different quality outputs
        writers = []
        for quality in quality_list:
            writers.append(AudioWriter(output_path + f"_{quality}", channels=self.channels, quality=quality))
        
        chunk_iterator = reader.stream_chunks()

        # State Variables for the "Lookahead Padding"
        prev_tail = np.zeros(config.PADDING_SAMPLES * self.channels, dtype=config.DTYPE_AUDIO)
        current_chunk = next(chunk_iterator, None)
        
        # We must track futures to guarantee chronological reassembly
        futures_queue = queue.Queue()

        print("[ORCHESTRATOR] Commencing Parallel Processing Matrix...")

        # Boot the Process Pool (bypassing the GIL)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            
            chunk_index = 0
            
            while current_chunk is not None:
                # 1. Lookahead: Peek at the next chunk to grab its head for right-padding
                next_chunk = next(chunk_iterator, None)
                
                if next_chunk is not None:
                    right_pad = next_chunk[:config.PADDING_SAMPLES * self.channels]
                else:
                    # EOF reached: Pad with absolute digital silence
                    right_pad = np.zeros(config.PADDING_SAMPLES * self.channels, dtype=config.DTYPE_AUDIO)

                # 2. Assemble the Padded Matrix (Left Pad + Chunk + Right Pad)
                worker_input = np.concatenate((prev_tail, current_chunk, right_pad))

                # Update state for the next cycle
                prev_tail = current_chunk[-(config.PADDING_SAMPLES * self.channels):]
                current_chunk = next_chunk

                # 3. Acquire free RAM block (Wait if all are currently processing)
                block_id = self.free_blocks.get()
                in_block = self.input_memory.get_block(block_id)
                out_block = self.output_memory.get_block(block_id)

                # 4. Write data to Zero-Copy Input RAM
                in_block.data[:] = worker_input[:]

                # 5. Dispatch Worker to specific CPU Core
                future = executor.submit(
                    process_chunk,
                    chunk_index,
                    in_block.name,
                    out_block.name,
                    self.padded_input_shape,
                    self.trimmed_output_shape
                )
                
                # Store the future alongside its memory block ID to reap it later
                futures_queue.put((future, block_id))
                chunk_index += 1

                # 6. Reap completed futures IN STRICT CHRONOLOGICAL ORDER
                while not futures_queue.empty() and futures_queue.queue[0][0].done():
                    oldest_future, oldest_block_id = futures_queue.get()
                    oldest_future.result() 

                    # Read from Output RAM and push to all FFmpeg writers
                    finished_block = self.output_memory.get_block(oldest_block_id)
                    for writer in writers:
                        writer.write_chunk(finished_block.data)

                    # Wipe memory to prevent ghosting, and return to pool
                    finished_block.wipe()
                    self.free_blocks.put(oldest_block_id)

            # 7. Drain the remaining queue
            while not futures_queue.empty():
                oldest_future, oldest_block_id = futures_queue.get()
                oldest_future.result()
                
                finished_block = self.output_memory.get_block(oldest_block_id)
                for writer in writers:
                    writer.write_chunk(finished_block.data)
                
                finished_block.wipe()
                self.free_blocks.put(oldest_block_id)

        # 8. Finalize all writers
        for writer in writers:
            writer.close()
            
        # 9. Cleanup
        self.input_memory.destroy_all()
        self.output_memory.destroy_all()
        print("[ORCHESTRATOR] Matrix Processing Complete. Substrate shut down cleanly.")
        
    def generate_full_analysis(self, input_path, output_prefix):
        """
        Generates ONLY the mathematical representation for the entire file.
        Visuals are disabled for performance and storage optimization.
        """
        print(f"[ORCHESTRATOR] Running end-to-end analysis for: {input_path}")
        import librosa
        waveform, _ = librosa.load(input_path, sr=config.SAMPLE_RATE, mono=(self.channels==1))
        waveform = waveform.astype(config.DTYPE_AUDIO)
        
        # Save the mathematical array directly (Waveform)
        # This is the most efficient representation for AGI latent conversion
        save_mathematical_representation(waveform, output_prefix)
        
        return output_prefix
