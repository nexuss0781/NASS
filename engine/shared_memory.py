import numpy as np
from multiprocessing import shared_memory
import sys
import os

# Import the physics laws
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

"""
AGI AUDIO SUBSTRATE - ZERO-COPY MEMORY BRIDGE
---------------------------------------------
Bypasses the Python Global Interpreter Lock (GIL) and multiprocessing 
serialization overhead. CPU cores read/write to identical RAM addresses.
Uses Float16 for 50% memory reduction compared to Float32.
"""

class SharedTensor:
    def __init__(self, name: str, shape: tuple, dtype: type, is_creator: bool = False):
        """
        Maps a NumPy array directly onto a shared hardware memory block.
        
        Args:
            name: The OS-level identifier for the memory block.
            shape: The dimensions of the tensor (e.g., (480000,) for a 10s chunk).
            dtype: The precision standard (config.DTYPE_AUDIO or config.DTYPE_COMPLEX).
            is_creator: True if this process is allocating the RAM. 
                        False if this is a worker core just attaching to it.
        """
        self.name = name
        self.shape = shape
        self.dtype = np.dtype(dtype)
        
        # Calculate absolute byte size required in RAM
        self.num_elements = int(np.prod(self.shape))
        self.byte_size = self.num_elements * self.dtype.itemsize

        if is_creator:
            # Main Process: Ask the OS for a contiguous block of RAM
            try:
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.byte_size)
            except FileExistsError:
                # If a previous crash left the memory orphaned, forcefully adopt and wipe it
                self.shm = shared_memory.SharedMemory(name=self.name, create=False)
                self.shm.unlink()
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.byte_size)
        else:
            # Worker Process: Attach directly to the existing memory address
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)

        # STATE FREEZE: Wrap the raw memory buffer in a NumPy array.
        # Any mathematical operation on 'self.data' instantly modifies the RAM across all cores.
        self.data = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def wipe(self):
        """
        Overwrites the memory block with absolute zero. 
        Critical for preventing audio ghosting between chunk cycles.
        """
        self.data.fill(0)

    def close(self):
        """
        Disconnects this specific process from the memory block.
        """
        self.shm.close()

    def destroy(self):
        """
        Main Process Only: Tells the OS to permanently free the RAM.
        """
        self.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass


class MemoryPool:
    """
    Manages a pool of SharedTensors so the Orchestrator doesn't have to 
    allocate/deallocate RAM dynamically (which causes fragmentation and latency).
    Memory is allocated ONCE at boot.
    """
    def __init__(self, num_blocks: int, chunk_shape: tuple, dtype: type, prefix: str = "agi_audio"):
        self.num_blocks = num_blocks
        self.blocks = []
        
        print(f"[MEMORY] Allocating {num_blocks} Zero-Copy hardware blocks...")
        for i in range(num_blocks):
            name = f"{prefix}_block_{i}"
            block = SharedTensor(name=name, shape=chunk_shape, dtype=dtype, is_creator=True)
            block.wipe()
            self.blocks.append(block)

    def get_block(self, index: int) -> SharedTensor:
        return self.blocks[index]

    def destroy_all(self):
        for block in self.blocks:
            block.destroy()
        print("[MEMORY] All hardware blocks freed.")