"""Metal MoE kernel Python bindings.

Uses PyObjC to interface with Metal compute shaders for optimized
Mixture of Experts computation on Apple Silicon.

Requirements:
- PyObjC (pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders)
- macOS 12.0+

This is a proof-of-concept implementation. For production use,
consider using MLX which has better Metal integration.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch

# Try to import Metal bindings
try:
    import Metal
    import Foundation
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    print("Warning: PyObjC Metal bindings not available. Install with:")
    print("  pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders")


class MetalMoEKernel:
    """Metal-accelerated MoE kernel for Apple Silicon."""

    def __init__(self, device_id: int = 0):
        """Initialize Metal device and compile kernels.

        Args:
            device_id: Metal device index (0 for default GPU)
        """
        if not METAL_AVAILABLE:
            raise RuntimeError("Metal bindings not available")

        # Get Metal device
        devices = Metal.MTLCopyAllDevices()
        if not devices or device_id >= len(devices):
            raise RuntimeError(f"No Metal device found at index {device_id}")

        self.device = devices[device_id]
        print(f"Using Metal device: {self.device.name()}")

        # Create command queue
        self.command_queue = self.device.newCommandQueue()

        # Load and compile kernel
        self._compile_kernel()

    def _compile_kernel(self) -> None:
        """Compile Metal kernel from source."""
        # Find kernel source file
        kernel_path = Path(__file__).parent / "moe_kernel.metal"
        if not kernel_path.exists():
            raise FileNotFoundError(f"Kernel source not found: {kernel_path}")

        with open(kernel_path, "r") as f:
            source = f.read()

        # Compile shader
        options = Metal.MTLCompileOptions.new()

        # newLibraryWithSource returns (library, error) tuple
        result = self.device.newLibraryWithSource_options_error_(
            source, options, None
        )

        # Handle result - could be tuple (library, error) or just library
        if isinstance(result, tuple):
            self.library, error = result
            if error is not None:
                raise RuntimeError(f"Failed to compile Metal kernel: {error}")
        else:
            self.library = result

        if self.library is None:
            raise RuntimeError("Failed to compile Metal kernel: library is None")

        # Get kernel functions
        self.fused_moe_fn = self.library.newFunctionWithName_("fused_moe_kernel")
        self.gate_up_fn = self.library.newFunctionWithName_("moe_gate_up_kernel")

        if self.fused_moe_fn is None and self.gate_up_fn is None:
            raise RuntimeError("Failed to load kernel functions")

        # Create compute pipeline
        if self.gate_up_fn:
            result = self.device.newComputePipelineStateWithFunction_error_(
                self.gate_up_fn, None
            )
            if isinstance(result, tuple):
                self.gate_up_pipeline, error = result
            else:
                self.gate_up_pipeline = result

        print("Metal MoE kernel compiled successfully")

    def fused_experts(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str = "silu",
    ) -> torch.Tensor:
        """Execute fused MoE computation using Metal.

        Args:
            hidden_states: [num_tokens, hidden_size]
            w1: [num_experts, intermediate_size * 2, hidden_size]
            w2: [num_experts, hidden_size, intermediate_size]
            topk_weights: [num_tokens, topk]
            topk_ids: [num_tokens, topk]
            activation: Activation function ("silu" or "gelu")

        Returns:
            output: [num_tokens, hidden_size]
        """
        # For now, fall back to PyTorch implementation
        # This is a placeholder for the Metal kernel integration
        return self._pytorch_fallback(
            hidden_states, w1, w2, topk_weights, topk_ids, activation
        )

    def _pytorch_fallback(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str = "silu",
    ) -> torch.Tensor:
        """PyTorch fallback for MoE computation."""
        num_tokens, hidden_size = hidden_states.shape
        topk = topk_ids.shape[1]
        intermediate_size = w1.shape[1] // 2

        flat_ids = topk_ids.view(-1).long()
        flat_w = topk_weights.view(-1).to(hidden_states.dtype)
        expanded = hidden_states.unsqueeze(1).expand(-1, topk, -1).reshape(-1, hidden_size)

        w1_sel = w1[flat_ids]
        w2_sel = w2[flat_ids]

        gate_up = torch.bmm(expanded.unsqueeze(1), w1_sel.transpose(1, 2)).squeeze(1)
        gate = gate_up[:, :intermediate_size]
        up = gate_up[:, intermediate_size:]

        if activation == "silu":
            activated = torch.nn.functional.silu(gate) * up
        elif activation == "gelu":
            activated = torch.nn.functional.gelu(gate) * up
        else:
            activated = gate * up

        out = torch.bmm(activated.unsqueeze(1), w2_sel.transpose(1, 2)).squeeze(1)
        out = out * flat_w.unsqueeze(-1)

        return out.view(num_tokens, topk, hidden_size).sum(dim=1)


def get_metal_moe_kernel() -> Optional[MetalMoEKernel]:
    """Get Metal MoE kernel if available.

    Returns:
        MetalMoEKernel instance or None if Metal is not available
    """
    if not METAL_AVAILABLE:
        return None

    try:
        return MetalMoEKernel()
    except Exception as e:
        print(f"Failed to initialize Metal MoE kernel: {e}")
        return None


# Test function
def test_metal_moe():
    """Test Metal MoE kernel."""
    print("Testing Metal MoE kernel...")

    kernel = get_metal_moe_kernel()
    if kernel is None:
        print("Metal kernel not available, skipping test")
        return

    # Test dimensions (Qwen3-30B-A3B)
    num_tokens = 1
    hidden_size = 2048
    intermediate_size = 768
    num_experts = 128
    topk = 8

    device = "mps"
    dtype = torch.float16

    hidden = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    w1 = torch.randn(num_experts, intermediate_size * 2, hidden_size, dtype=dtype, device=device) * 0.01
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, dtype=dtype, device=device) * 0.01
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device)
    topk_w = torch.randn(num_tokens, topk, dtype=dtype, device=device).softmax(dim=-1)

    # Run kernel
    output = kernel.fused_experts(hidden, w1, w2, topk_w, topk_ids)
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0, :5]}")


if __name__ == "__main__":
    test_metal_moe()
