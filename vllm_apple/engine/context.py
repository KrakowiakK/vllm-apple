# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metal Engine Context for vLLM-Apple Metal Engine v2.0.

This module provides the process-wide Metal context that manages:
- MTLDevice (singleton)
- MTLCommandQueue
- MTLLibrary cache (compiled kernels)
- MTLComputePipelineState cache
- Scratch buffer pool

The context is designed to be created once at startup and reused across
all engine execution. Unlike the v1.x MetalDevice, this context does NOT
call torch.mps.synchronize() internally - that would violate the v2.0
"no PyTorch-MPS in hot path" invariant.

Usage:
    context = MetalEngineContext.get_instance()

    # Create command buffer for step execution
    cmd_buffer = context.new_command_buffer()

    # Get compiled pipeline for kernel
    pipeline = context.get_pipeline("rmsnorm", head_size=128)

    # Allocate scratch buffer
    scratch = context.allocate_scratch(num_bytes)
"""

import os
import ctypes
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import threading

from vllm.logger import init_logger
from .config import get_engine_config, get_resource_path

logger = init_logger(__name__)

# Try to import Metal Python bindings
try:
    import Metal
    from Metal import (
        MTLCreateSystemDefaultDevice,
        MTLCompileOptions,
        MTLSize,
        MTLResourceStorageModeShared,
        MTLResourceStorageModePrivate,
        MTLFunctionConstantValues,
        MTLBarrierScopeBuffers,
    )
    import objc
    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    Metal = None


@dataclass
class PipelineKey:
    """Key for pipeline cache lookup."""
    library_name: str
    function_name: str
    constants: Optional[Tuple[Tuple[int, int], ...]] = None  # ((idx, value), ...)

    def __hash__(self):
        return hash((self.library_name, self.function_name, self.constants))

    def __eq__(self, other):
        if not isinstance(other, PipelineKey):
            return False
        return (self.library_name == other.library_name and
                self.function_name == other.function_name and
                self.constants == other.constants)


@dataclass
class ScratchBuffer:
    """A scratch buffer from the pool."""
    buffer: Any  # MTLBuffer
    size: int
    in_use: bool = False
    last_step_id: int = -1
    last_generation: int = -1  # Generation (layer index) when last used


class EngineScratchPool:
    """Pool of scratch buffers for temporary allocations.

    Scratch buffers are used for intermediate results during engine execution.
    The pool pre-allocates buffers to avoid allocation overhead in hot path.

    GENERATION-BASED REUSE (v2.1):
    Scratch buffers can be reused across layers within a step. Each layer:
    1. Advances the generation counter at layer start
    2. Allocates scratch buffers (tagged with current generation)
    3. Releases buffers from that generation at layer end

    This enables cross-layer reuse for MLP intermediates and other
    layer-local temporaries, dramatically reducing memory requirements.

    IMPORTANT: Buffers are only reused when explicitly released (in_use=False).
    Generation tracking is for lifetime management, NOT for bypassing in_use.
    """

    def __init__(self, device: Any, pool_size_mb: int = 512):
        """Initialize scratch pool.

        Args:
            device: MTLDevice for buffer creation
            pool_size_mb: Total pool size in megabytes
        """
        self.device = device
        self.pool_size_bytes = pool_size_mb * 1024 * 1024
        self._buffers: List[ScratchBuffer] = []
        self._lock = threading.Lock()
        self._allocated_bytes = 0
        self._current_generation = 0

        # Stats for debugging
        self._reuse_hits = 0
        self._new_allocs = 0
        self._peak_in_use_bytes = 0
        self._current_in_use_bytes = 0

    def allocate(self, size: int, step_id: int = -1, generation: int = -1) -> Any:
        """Allocate a scratch buffer of at least the given size.

        SAFETY: Only reuses buffers that are NOT in_use. Generation tracking
        is for lifetime management (knowing which buffers to release), not
        for bypassing the in_use flag.

        Args:
            size: Minimum buffer size in bytes
            step_id: Current step ID for tracking
            generation: Current generation (layer index) for lifetime tracking.

        Returns:
            MTLBuffer

        Raises:
            RuntimeError: If pool is exhausted
        """
        with self._lock:
            # Use pool's current generation if not specified
            if generation < 0:
                generation = self._current_generation

            # Find best-fit buffer that is NOT in use
            # CRITICAL: Do NOT reuse buffers that are still in_use, even if
            # from older generation. The GPU may still need them.
            best_fit = None
            best_fit_size = float('inf')

            for buf in self._buffers:
                if buf.size >= size and not buf.in_use:
                    if buf.size < best_fit_size:
                        best_fit = buf
                        best_fit_size = buf.size

            if best_fit is not None:
                best_fit.in_use = True
                best_fit.last_step_id = step_id
                best_fit.last_generation = generation
                self._reuse_hits += 1
                self._current_in_use_bytes += best_fit.size
                self._peak_in_use_bytes = max(self._peak_in_use_bytes, self._current_in_use_bytes)
                return best_fit.buffer

            # Need to allocate new buffer
            # Round up to 64KB alignment for Metal efficiency
            aligned_size = (size + 65535) & ~65535

            if self._allocated_bytes + aligned_size > self.pool_size_bytes:
                # Provide diagnostic info
                in_use_count = sum(1 for b in self._buffers if b.in_use)
                in_use_bytes = sum(b.size for b in self._buffers if b.in_use)
                raise RuntimeError(
                    f"Scratch pool exhausted: requested {size} bytes, "
                    f"allocated {self._allocated_bytes}/{self.pool_size_bytes}, "
                    f"in_use {in_use_count} buffers ({in_use_bytes} bytes), "
                    f"generation {generation}, current_gen {self._current_generation}"
                )

            buffer = self.device.newBufferWithLength_options_(
                aligned_size,
                MTLResourceStorageModeShared
            )

            if buffer is None:
                raise RuntimeError(f"Failed to allocate scratch buffer of {aligned_size} bytes")

            scratch = ScratchBuffer(
                buffer=buffer,
                size=aligned_size,
                in_use=True,
                last_step_id=step_id,
                last_generation=generation,
            )
            self._buffers.append(scratch)
            self._allocated_bytes += aligned_size
            self._new_allocs += 1
            self._current_in_use_bytes += aligned_size
            self._peak_in_use_bytes = max(self._peak_in_use_bytes, self._current_in_use_bytes)

            logger.debug(f"Allocated scratch buffer: {aligned_size} bytes, total: {self._allocated_bytes}")
            return buffer

    def advance_generation(self) -> int:
        """Advance to the next generation (typically called per layer).

        Returns:
            The new generation number.
        """
        with self._lock:
            self._current_generation += 1
            return self._current_generation

    def get_generation(self) -> int:
        """Get the current generation number."""
        with self._lock:
            return self._current_generation

    def reset_generation(self) -> None:
        """Reset generation counter (typically called at step start)."""
        with self._lock:
            self._current_generation = 0

    def release(self, buffer: Any) -> None:
        """Release a scratch buffer back to the pool.

        Args:
            buffer: MTLBuffer to release
        """
        with self._lock:
            for buf in self._buffers:
                if buf.buffer is buffer:
                    if buf.in_use:
                        self._current_in_use_bytes -= buf.size
                    buf.in_use = False
                    return

    def release_all_from_step(self, step_id: int) -> None:
        """Release all buffers used in a specific step.

        Args:
            step_id: Step ID whose buffers should be released
        """
        with self._lock:
            for buf in self._buffers:
                if buf.in_use and buf.last_step_id == step_id:
                    self._current_in_use_bytes -= buf.size
                    buf.in_use = False

    def release_generation(self, generation: int) -> int:
        """Release all buffers from a specific generation.

        Args:
            generation: Generation whose buffers should be released

        Returns:
            Number of buffers released
        """
        released = 0
        with self._lock:
            for buf in self._buffers:
                if buf.in_use and buf.last_generation == generation:
                    self._current_in_use_bytes -= buf.size
                    buf.in_use = False
                    released += 1
        return released

    def release_up_to_generation(self, generation: int) -> int:
        """Release all buffers from generations <= given generation.

        Useful for recovery or cleanup if a generation was skipped.

        Args:
            generation: Release all buffers with last_generation <= this

        Returns:
            Number of buffers released
        """
        released = 0
        with self._lock:
            for buf in self._buffers:
                if buf.in_use and buf.last_generation <= generation:
                    self._current_in_use_bytes -= buf.size
                    buf.in_use = False
                    released += 1
        return released

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            in_use = sum(1 for b in self._buffers if b.in_use)
            in_use_bytes = sum(b.size for b in self._buffers if b.in_use)

            # Per-generation breakdown
            gen_stats = {}
            for buf in self._buffers:
                if buf.in_use:
                    gen = buf.last_generation
                    if gen not in gen_stats:
                        gen_stats[gen] = {"count": 0, "bytes": 0}
                    gen_stats[gen]["count"] += 1
                    gen_stats[gen]["bytes"] += buf.size

            return {
                "total_buffers": len(self._buffers),
                "in_use_buffers": in_use,
                "allocated_bytes": self._allocated_bytes,
                "in_use_bytes": in_use_bytes,
                "pool_size_bytes": self.pool_size_bytes,
                "current_generation": self._current_generation,
                "reuse_hits": self._reuse_hits,
                "new_allocs": self._new_allocs,
                "peak_in_use_bytes": self._peak_in_use_bytes,
                "per_generation": gen_stats,
            }

    def reset_stats(self) -> None:
        """Reset allocation statistics (call at step boundaries)."""
        with self._lock:
            self._reuse_hits = 0
            self._new_allocs = 0
            self._peak_in_use_bytes = self._current_in_use_bytes


class MetalEngineContext:
    """Process-wide Metal context for engine execution.

    This class manages all Metal resources needed for engine execution:
    - MTLDevice: The default Metal device
    - MTLCommandQueue: For command buffer creation
    - Library cache: Compiled MTLLibrary objects
    - Pipeline cache: MTLComputePipelineState objects
    - Scratch pool: Temporary buffers for intermediate results

    The context is a singleton to ensure resources are shared across
    all engine components.

    Important: This class does NOT call torch.mps.synchronize() or any
    other PyTorch-MPS operations. The v2.0 architecture requires that
    the engine operates independently of PyTorch-MPS.
    """

    _instance: Optional["MetalEngineContext"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetalEngineContext":
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the Metal context."""
        if self._initialized:
            return

        if not HAS_METAL:
            raise RuntimeError(
                "Metal Python bindings not available. "
                "Install with: pip install pyobjc-framework-Metal pyobjc-framework-Foundation"
            )

        # Create Metal device
        self._device = MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("Failed to create Metal device. Is MPS available?")

        # Create command queue
        self._command_queue = self._device.newCommandQueue()
        if self._command_queue is None:
            raise RuntimeError("Failed to create Metal command queue")

        # Initialize caches
        self._libraries: Dict[str, Any] = {}
        self._pipelines: Dict[PipelineKey, Any] = {}
        self._library_lock = threading.Lock()
        self._pipeline_lock = threading.Lock()

        # Initialize scratch pool
        config = get_engine_config()
        self._scratch_pool = EngineScratchPool(
            self._device,
            pool_size_mb=config.scratch_pool_size_mb
        )

        # Determine kernel resource path
        self._resource_path = self._find_resource_path()

        self._initialized = True
        logger.info(f"MetalEngineContext initialized: {self._device.name()}")

    @classmethod
    def get_instance(cls) -> "MetalEngineContext":
        """Get the singleton instance.

        Returns:
            MetalEngineContext singleton
        """
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def _find_resource_path(self) -> Path:
        """Find the path to Metal kernel resources.

        Returns:
            Path to kernels directory
        """
        # Check environment override
        custom_path = get_resource_path()
        if custom_path:
            path = Path(custom_path)
            if path.exists():
                return path
            logger.warning(f"Custom resource path not found: {custom_path}")

        # Default: relative to this module
        module_dir = Path(__file__).parent
        kernels_dir = module_dir / "kernels"
        if kernels_dir.exists():
            return kernels_dir

        # Fallback: check metal/kernels
        metal_kernels = module_dir.parent / "metal" / "kernels"
        if metal_kernels.exists():
            return metal_kernels

        # Last resort: just use the module directory
        logger.warning("Could not find kernels directory, using module directory")
        return module_dir

    @property
    def device(self) -> Any:
        """Get the MTLDevice."""
        return self._device

    @property
    def command_queue(self) -> Any:
        """Get the MTLCommandQueue."""
        return self._command_queue

    @property
    def scratch_pool(self) -> EngineScratchPool:
        """Get the scratch buffer pool."""
        return self._scratch_pool

    def new_command_buffer(self) -> Any:
        """Create a new command buffer.

        Returns:
            MTLCommandBuffer for encoding commands
        """
        cmd_buffer = self._command_queue.commandBuffer()
        if cmd_buffer is None:
            raise RuntimeError("Failed to create command buffer")
        return cmd_buffer

    def load_library(
        self,
        name: str,
        source_path: Optional[str] = None,
        source_code: Optional[str] = None,
        preprocessor_macros: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Load or compile MTLLibrary, preferring precompiled .metallib.

        Per METAL_PLAN.md section 1.1:
        - Load precompiled .metallib from package resources when available
        - Fall back to runtime compilation from .metal source for debug
        - Resource path override via VLLM_METAL_PATH_RESOURCES

        The priority order is:
        1. If VLLM_METAL_PATH_RESOURCES is set, compile from source (debug mode)
        2. Try loading precompiled {name}.metallib
        3. Fall back to compiling from {name}.metal or provided source

        Args:
            name: Library name for caching
            source_path: Path to .metal source file (optional)
            source_code: Direct source code string (optional)
            preprocessor_macros: Dict of macro definitions for compilation

        Returns:
            MTLLibrary

        Raises:
            RuntimeError: If loading/compilation fails
        """
        with self._library_lock:
            if name in self._libraries:
                return self._libraries[name]

            library = None
            loaded_from = None

            # Check if resource path override is set (debug mode)
            force_source = get_resource_path() is not None

            if not force_source:
                # Try loading precompiled .metallib first
                metallib_path = self._resource_path / f"{name}.metallib"
                if metallib_path.exists():
                    library = self._load_metallib(metallib_path)
                    if library is not None:
                        loaded_from = f"precompiled metallib: {metallib_path}"

            # Fall back to source compilation
            if library is None:
                library, loaded_from = self._compile_from_source(
                    name, source_path, source_code, preprocessor_macros
                )

            if library is None:
                raise RuntimeError(f"Failed to load or compile Metal library '{name}'")

            self._libraries[name] = library
            logger.info(f"Loaded Metal library '{name}' from {loaded_from}")
            return library

    def _load_metallib(self, path: Path) -> Optional[Any]:
        """Load a precompiled .metallib file.

        Args:
            path: Path to .metallib file

        Returns:
            MTLLibrary or None if loading fails
        """
        try:
            # Create NSURL for the metallib file
            from Foundation import NSURL
            url = NSURL.fileURLWithPath_(str(path))

            # Load the library
            library, error = self._device.newLibraryWithURL_error_(url, None)

            if library is None or error is not None:
                logger.warning(f"Failed to load metallib {path}: {error}")
                return None

            return library
        except Exception as e:
            logger.warning(f"Exception loading metallib {path}: {e}")
            return None

    def _compile_from_source(
        self,
        name: str,
        source_path: Optional[str],
        source_code: Optional[str],
        preprocessor_macros: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Any], str]:
        """Compile MTLLibrary from source.

        Args:
            name: Library name
            source_path: Path to .metal source file
            source_code: Direct source code string
            preprocessor_macros: Dict of macro definitions

        Returns:
            Tuple of (MTLLibrary or None, description of source)
        """
        # Determine source
        if source_code is not None:
            loaded_from = "inline source"
        elif source_path is not None:
            path = Path(source_path)
            if not path.is_absolute():
                path = self._resource_path / path
            if not path.exists():
                raise FileNotFoundError(f"Metal shader not found: {path}")
            with open(path, "r") as f:
                source_code = f.read()
            loaded_from = f"source file: {path}"
        else:
            # Try default .metal file
            metal_path = self._resource_path / f"{name}.metal"
            if metal_path.exists():
                with open(metal_path, "r") as f:
                    source_code = f.read()
                loaded_from = f"source file: {metal_path}"
            else:
                raise FileNotFoundError(
                    f"No source found for library '{name}': tried {metal_path}"
                )

        # Create compile options
        options = MTLCompileOptions.alloc().init()

        # Set preprocessor macros if provided
        # METAL_PLAN.md: "use MTLCompileOptions.preprocessorMacros for device
        # capability gating (e.g., BF16/tensor features)"
        if preprocessor_macros:
            from Foundation import NSDictionary, NSNumber, NSString
            macro_dict = {}
            for key, value in preprocessor_macros.items():
                if isinstance(value, bool):
                    macro_dict[key] = NSNumber.numberWithBool_(value)
                elif isinstance(value, int):
                    macro_dict[key] = NSNumber.numberWithInt_(value)
                else:
                    macro_dict[key] = NSString.stringWithString_(str(value))
            options.setPreprocessorMacros_(NSDictionary.dictionaryWithDictionary_(macro_dict))

        # Compile
        library, error = self._device.newLibraryWithSource_options_error_(
            source_code, options, None
        )

        if library is None or error is not None:
            raise RuntimeError(f"Failed to compile Metal shader '{name}': {error}")

        return library, loaded_from

    def compile_library(
        self,
        name: str,
        source_path: Optional[str] = None,
        source_code: Optional[str] = None,
    ) -> Any:
        """Compile or get cached MTLLibrary.

        This is a compatibility wrapper for load_library(). New code should
        use load_library() directly to benefit from precompiled metallib loading.

        Args:
            name: Library name for caching
            source_path: Path to .metal source file
            source_code: Direct source code string

        Returns:
            MTLLibrary

        Raises:
            RuntimeError: If compilation fails
        """
        return self.load_library(name, source_path, source_code)

    def get_pipeline(
        self,
        library_name: str,
        function_name: str,
        function_constants: Optional[Dict[int, int]] = None,
    ) -> Any:
        """Get or create compute pipeline.

        Args:
            library_name: Name of the compiled library
            function_name: Name of the kernel function
            function_constants: Optional dict mapping constant index to value

        Returns:
            MTLComputePipelineState
        """
        # Create cache key
        constants_tuple = None
        if function_constants:
            constants_tuple = tuple(sorted(function_constants.items()))
        key = PipelineKey(library_name, function_name, constants_tuple)

        with self._pipeline_lock:
            if key in self._pipelines:
                return self._pipelines[key]

            # Get library
            if library_name not in self._libraries:
                raise RuntimeError(f"Library '{library_name}' not compiled")
            library = self._libraries[library_name]

            # Create function (with or without constants)
            if function_constants and MTLFunctionConstantValues is not None:
                constants = MTLFunctionConstantValues.alloc().init()
                for idx, value in function_constants.items():
                    data = struct.pack("I", value)
                    constants.setConstantValue_type_atIndex_(
                        data,
                        Metal.MTLDataTypeUInt,
                        idx
                    )
                # PyObjC pattern: pass None, get (result, error) tuple
                function, fn_error = library.newFunctionWithName_constantValues_error_(
                    function_name, constants, None
                )
                if function is None:
                    logger.warning(
                        f"Failed to create specialized function '{function_name}': {fn_error}. "
                        f"Falling back to non-specialized."
                    )
                    function = library.newFunctionWithName_(function_name)
            else:
                function = library.newFunctionWithName_(function_name)

            if function is None:
                available = [str(f) for f in library.functionNames()]
                raise RuntimeError(
                    f"Function '{function_name}' not found in library '{library_name}'. "
                    f"Available: {available}"
                )

            # Create pipeline - PyObjC pattern: pass None, get (result, error) tuple
            pipeline, pipeline_error = self._device.newComputePipelineStateWithFunction_error_(
                function, None
            )

            if pipeline is None:
                raise RuntimeError(
                    f"Failed to create pipeline for '{function_name}': {pipeline_error}"
                )

            self._pipelines[key] = pipeline
            logger.debug(f"Created pipeline: {library_name}::{function_name}")
            return pipeline

    def create_buffer(
        self,
        size: int,
        storage_mode: str = "shared",
    ) -> Any:
        """Create a new MTLBuffer.

        Args:
            size: Buffer size in bytes
            storage_mode: "shared" (CPU+GPU) or "private" (GPU only)

        Returns:
            MTLBuffer
        """
        if storage_mode == "shared":
            mode = MTLResourceStorageModeShared
        elif storage_mode == "private":
            mode = MTLResourceStorageModePrivate
        else:
            raise ValueError(f"Unknown storage mode: {storage_mode}")

        buffer = self._device.newBufferWithLength_options_(size, mode)
        if buffer is None:
            raise RuntimeError(f"Failed to create buffer of {size} bytes")
        return buffer

    def create_buffer_from_bytes(
        self,
        data: bytes,
        storage_mode: str = "shared",
    ) -> Any:
        """Create MTLBuffer initialized with data.

        Args:
            data: Bytes to copy into buffer
            storage_mode: "shared" or "private"

        Returns:
            MTLBuffer
        """
        if storage_mode == "shared":
            mode = MTLResourceStorageModeShared
        else:
            mode = MTLResourceStorageModePrivate

        buffer = self._device.newBufferWithBytes_length_options_(
            data, len(data), mode
        )
        if buffer is None:
            raise RuntimeError(f"Failed to create buffer from {len(data)} bytes")
        return buffer

    def allocate_scratch(self, size: int, step_id: int = -1) -> Any:
        """Allocate a scratch buffer from the pool.

        Args:
            size: Minimum buffer size
            step_id: Current step ID

        Returns:
            MTLBuffer
        """
        return self._scratch_pool.allocate(size, step_id)

    def release_scratch(self, buffer: Any) -> None:
        """Release a scratch buffer back to the pool.

        Args:
            buffer: MTLBuffer to release
        """
        self._scratch_pool.release(buffer)

    def release_scratch_for_step(self, step_id: int) -> None:
        """Release all scratch buffers used in a specific step.

        Args:
            step_id: Step ID whose buffers should be released
        """
        self._scratch_pool.release_all_from_step(step_id)

    def advance_scratch_generation(self) -> int:
        """Advance scratch pool generation (call per layer).

        Returns:
            The new generation number.
        """
        return self._scratch_pool.advance_generation()

    def get_scratch_generation(self) -> int:
        """Get current scratch generation."""
        return self._scratch_pool.get_generation()

    def reset_scratch_generation(self) -> None:
        """Reset scratch generation (call at step start)."""
        self._scratch_pool.reset_generation()

    def release_scratch_generation(self, generation: int) -> int:
        """Release all scratch buffers from a specific generation.

        Call this at layer scope exit to enable cross-layer buffer reuse.

        Args:
            generation: Generation (layer index) to release

        Returns:
            Number of buffers released
        """
        return self._scratch_pool.release_generation(generation)

    def release_scratch_up_to_generation(self, generation: int) -> int:
        """Release all scratch buffers from generations <= given.

        Useful for cleanup or recovery.

        Args:
            generation: Release all buffers with last_generation <= this

        Returns:
            Number of buffers released
        """
        return self._scratch_pool.release_up_to_generation(generation)

    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics.

        Returns:
            Dict with statistics about libraries, pipelines, and scratch pool
        """
        return {
            "device_name": self._device.name(),
            "num_libraries": len(self._libraries),
            "num_pipelines": len(self._pipelines),
            "scratch_pool": self._scratch_pool.get_stats(),
        }

    def blit_buffer_to_staging(
        self,
        source_buffer: Any,
        source_offset: int,
        size: int,
        staging_buffer: Optional[Any] = None,
    ) -> Tuple[Any, memoryview]:
        """Copy data from any buffer to a staging buffer for CPU readback.

        IMPORTANT: This method creates its own command buffer and waits for
        completion. It is ONLY allowed during READBACK or IDLE phases.
        Using this during ENCODE or SUBMIT would break the "one command buffer
        per step" invariant.

        For hot-path usage, use encode_blit_copy() to encode blits into the
        step's command buffer, then read after the step's waitUntilCompleted().

        Usage:
            # For Private buffer readback at step boundary:
            with EngineHotPathGuard.readback_phase():
                staging, mv = ctx.blit_buffer_to_staging(private_buf, 0, size)
                data = bytes(mv)  # Now safe to read

        Args:
            source_buffer: Source MTLBuffer (can be Private or Shared)
            source_offset: Byte offset in source
            size: Bytes to copy
            staging_buffer: Optional pre-allocated staging buffer (Shared mode).
                           If None, one will be allocated.

        Returns:
            Tuple of (staging_buffer, memoryview)
            - staging_buffer: The Shared buffer containing the data
            - memoryview: View into the staging buffer data

        Raises:
            RuntimeError: If called during ENCODE or SUBMIT phase
        """
        from .guards import EngineHotPathGuard, EnginePhase

        # Phase guard: this method breaks "one CB per step" if used in hot path
        phase = EngineHotPathGuard.get_phase()
        if phase in (EnginePhase.ENCODE, EnginePhase.SUBMIT):
            raise RuntimeError(
                f"blit_buffer_to_staging() called during {phase.value} phase. "
                f"This method creates its own command buffer and waits, which "
                f"breaks the 'one command buffer per step' invariant. "
                f"Use encode_blit_copy() for hot-path blits, or call this "
                f"only during READBACK/IDLE phase."
            )

        # Allocate staging buffer if needed
        if staging_buffer is None:
            staging_buffer = self.create_buffer(size, storage_mode="shared")

        # Validate staging buffer size
        if staging_buffer.length() < size:
            raise ValueError(
                f"Staging buffer too small: need {size}, have {staging_buffer.length()}"
            )

        # Create command buffer for blit
        cmd_buffer = self._command_queue.commandBuffer()
        if cmd_buffer is None:
            raise RuntimeError("Failed to create command buffer for blit")

        # Create blit encoder
        blit_encoder = cmd_buffer.blitCommandEncoder()
        if blit_encoder is None:
            raise RuntimeError("Failed to create blit encoder")

        # Encode copy
        blit_encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
            source_buffer, source_offset, staging_buffer, 0, size
        )

        # End encoding and commit
        blit_encoder.endEncoding()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()

        # Return staging buffer and memoryview
        contents = staging_buffer.contents()
        mv = contents.as_buffer(size)
        return staging_buffer, mv

    def encode_blit_copy(
        self,
        command_buffer: Any,
        source_buffer: Any,
        source_offset: int,
        dest_buffer: Any,
        dest_offset: int,
        size: int,
    ) -> None:
        """Encode a blit copy to an existing command buffer.

        This is for encoding blit copies within a step's command buffer,
        allowing the copy to be batched with other operations.

        Args:
            command_buffer: MTLCommandBuffer to encode to
            source_buffer: Source MTLBuffer
            source_offset: Byte offset in source
            dest_buffer: Destination MTLBuffer
            dest_offset: Byte offset in destination
            size: Bytes to copy
        """
        # Create blit encoder
        blit_encoder = command_buffer.blitCommandEncoder()
        if blit_encoder is None:
            raise RuntimeError("Failed to create blit encoder")

        # Encode copy
        blit_encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
            source_buffer, source_offset, dest_buffer, dest_offset, size
        )

        # End encoding
        blit_encoder.endEncoding()


def get_engine_context() -> MetalEngineContext:
    """Get the global engine context.

    Convenience function for accessing the singleton.

    Returns:
        MetalEngineContext instance
    """
    return MetalEngineContext.get_instance()
