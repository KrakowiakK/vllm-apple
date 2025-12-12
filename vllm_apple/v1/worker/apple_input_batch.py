# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apple input batch implementation for vLLM V1 engine.

This module provides batch management for Apple Silicon (MPS) inference,
handling request tracking, tensor storage, and sampling parameter management.
"""

from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import torch

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.collection_utils import swap_dict_values
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.utils import copy_slice
from vllm.v1.worker.block_table import MultiGroupBlockTable

_SAMPLING_EPS = 1e-5


@dataclass
class CachedRequestState:
    """Cached state for a single request in the batch.

    This stores all necessary information about a request including
    prompt tokens, sampling parameters, KV cache block assignments,
    and generation state.
    """
    req_id: str
    prompt_token_ids: Optional[list[int]]
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: Optional[SamplingParams]
    generator: Optional[torch.Generator]

    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]

    lora_request: Optional[LoRARequest] = None
    prompt_embeds: Optional[torch.Tensor] = None

    # For pooling models
    pooling_params: Optional[PoolingParams] = None

    def __post_init__(self):
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            self.prompt_token_ids, self.prompt_embeds
        )

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)


class AppleInputBatch:
    """Input batch manager for Apple Silicon (MPS) inference.

    This class manages batched requests for model execution on MPS devices.
    It handles:
    - Request tracking and indexing
    - Token ID storage and management
    - Block table management for KV cache
    - Sampling parameter storage
    - Tensor transfers between CPU and MPS

    The design is optimized for MPS unified memory architecture while
    maintaining compatibility with the vLLM V1 API.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],
    ):
        """Initialize the Apple input batch.

        Args:
            max_num_reqs: Maximum number of requests in a batch
            max_model_len: Maximum sequence length
            max_num_batched_tokens: Maximum total tokens in a batch
            device: Target device (should be MPS)
            pin_memory: Whether to pin CPU memory (Note: MPS uses unified memory)
            vocab_size: Size of the model vocabulary
            block_sizes: Block size for each KV cache group
        """
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device
        self.pin_memory = pin_memory
        self.vocab_size = vocab_size

        # Request tracking
        self._req_ids: list[Optional[str]] = []
        self.req_id_to_index: dict[str, int] = {}

        # Token storage
        # Note: For MPS unified memory, we keep CPU tensors and transfer as needed
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,  # MPS uses unified memory
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()

        # Token counts
        self.num_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.num_computed_tokens_cpu = self.num_computed_tokens_cpu_tensor.numpy()

        # Block table for KV cache management
        # Use simplified kernel_block_sizes (same as block_sizes for Apple)
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            block_sizes=block_sizes,
            kernel_block_sizes=block_sizes,  # Apple doesn't need separate kernel sizes
        )

        # Sampling parameters - GPU tensors
        self.temperature = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device=device
        )
        self.temperature_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory
        )
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()
        self.greedy_reqs: set[str] = set()
        self.random_reqs: set[str] = set()

        self.top_p = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)
        self.top_p_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory
        )
        self.top_p_cpu = self.top_p_cpu_tensor.numpy()
        self.top_p_reqs: set[str] = set()

        self.top_k = torch.empty((max_num_reqs,), dtype=torch.int32, device=device)
        self.top_k_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
        )
        self.top_k_cpu = self.top_k_cpu_tensor.numpy()
        self.top_k_reqs: set[str] = set()

        self.min_p = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)
        self.min_p_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory
        )
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()
        self.min_p_reqs: set[str] = set()

        # Penalty parameters
        self.frequency_penalties = torch.empty(
            (max_num_reqs,), dtype=torch.float, device=device
        )
        self.frequency_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.frequency_penalties_cpu = self.frequency_penalties_cpu_tensor.numpy()
        self.frequency_penalties_reqs: set[str] = set()

        self.presence_penalties = torch.empty(
            (max_num_reqs,), dtype=torch.float, device=device
        )
        self.presence_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.presence_penalties_cpu = self.presence_penalties_cpu_tensor.numpy()
        self.presence_penalties_reqs: set[str] = set()

        self.repetition_penalties = torch.empty(
            (max_num_reqs,), dtype=torch.float, device=device
        )
        self.repetition_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.repetition_penalties_cpu = self.repetition_penalties_cpu_tensor.numpy()
        self.repetition_penalties_reqs: set[str] = set()

        # Min tokens tracking
        self.min_tokens: dict[int, tuple[int, set[int]]] = {}

        # LoRA support
        self.request_lora_mapping = np.zeros((self.max_num_reqs,), dtype=np.int64)
        self.lora_id_to_request_ids: dict[int, set[str]] = {}
        self.lora_id_to_lora_request: dict[int, LoRARequest] = {}

        # Random number generators
        self.generators: dict[int, torch.Generator] = {}

        # Logprobs tracking
        self.num_logprobs: dict[str, int] = {}
        self.in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors] = {}

        # Logit bias and allowed token IDs
        self.logit_bias: list[Optional[dict[int, float]]] = [None] * max_num_reqs
        self.has_allowed_token_ids: set[str] = set()
        self.allowed_token_ids_mask: Optional[torch.Tensor] = None
        self.allowed_token_ids_mask_cpu_tensor: Optional[torch.Tensor] = None

        # Bad words filtering
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

        # Output token IDs tracking
        self.req_output_token_ids: list[Optional[list[int]]] = []

    @property
    def req_ids(self) -> list[str]:
        """Get list of active request IDs.

        None elements should only be present transiently while performing
        state updates to the batch.
        """
        return cast(list[str], self._req_ids)

    def add_request(
        self,
        request: CachedRequestState,
        req_index: Optional[int] = None,
    ) -> int:
        """Add a request to the batch.

        Args:
            request: Request state to add
            req_index: Optional index to use (if None, appends to end)

        Returns:
            The index where the request was added
        """
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs, (
            f"Request index {req_index} exceeds max {self.max_num_reqs}"
        )

        req_id = request.req_id
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids

        self.req_id_to_index[req_id] = req_index

        # Copy prompt token IDs and output token IDs
        num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, request.prompt_embeds
        )
        self.num_prompt_tokens[req_index] = num_prompt_tokens

        if request.prompt_token_ids is not None:
            self.token_ids_cpu[req_index, :num_prompt_tokens] = request.prompt_token_ids

        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_cpu[req_index, start_idx:end_idx] = request.output_token_ids

        # Update token counts
        self.num_tokens[req_index] = request.num_tokens
        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens

        # Update block table
        self.block_table.add_row(request.block_ids, req_index)

        # Configure sampling parameters
        sampling_params = request.sampling_params
        if sampling_params is not None:
            self._add_sampling_params(req_index, req_id, sampling_params)

        # Handle random generator
        if request.generator is not None:
            self.generators[req_index] = request.generator

        # Handle LoRA
        if request.lora_request:
            lora_id = request.lora_request.lora_int_id
            if lora_id not in self.lora_id_to_request_ids:
                self.lora_id_to_request_ids[lora_id] = set()

            self.request_lora_mapping[req_index] = lora_id
            self.lora_id_to_request_ids[lora_id].add(request.req_id)
            self.lora_id_to_lora_request[lora_id] = request.lora_request
        else:
            self.request_lora_mapping[req_index] = 0

        return req_index

    def _add_sampling_params(
        self,
        req_index: int,
        req_id: str,
        sampling_params: SamplingParams,
    ) -> None:
        """Configure sampling parameters for a request.

        Args:
            req_index: Index of the request
            req_id: ID of the request
            sampling_params: Sampling parameters to configure
        """
        # Temperature and sampling type
        if sampling_params.sampling_type == SamplingType.GREEDY:
            self.temperature_cpu[req_index] = 0.0
            self.greedy_reqs.add(req_id)
        else:
            self.temperature_cpu[req_index] = sampling_params.temperature
            self.random_reqs.add(req_id)

        # Top-p sampling
        self.top_p_cpu[req_index] = sampling_params.top_p
        if sampling_params.top_p < 1:
            self.top_p_reqs.add(req_id)

        # Top-k sampling
        top_k = sampling_params.top_k
        if 0 < top_k < self.vocab_size:
            self.top_k_reqs.add(req_id)
        else:
            top_k = self.vocab_size
        self.top_k_cpu[req_index] = top_k

        # Min-p sampling
        self.min_p_cpu[req_index] = sampling_params.min_p
        if sampling_params.min_p > _SAMPLING_EPS:
            self.min_p_reqs.add(req_id)

        # Penalties
        self.frequency_penalties_cpu[req_index] = sampling_params.frequency_penalty
        if sampling_params.frequency_penalty != 0.0:
            self.frequency_penalties_reqs.add(req_id)

        self.presence_penalties_cpu[req_index] = sampling_params.presence_penalty
        if sampling_params.presence_penalty != 0.0:
            self.presence_penalties_reqs.add(req_id)

        self.repetition_penalties_cpu[req_index] = sampling_params.repetition_penalty
        if sampling_params.repetition_penalty != 1.0:
            self.repetition_penalties_reqs.add(req_id)

        # Min tokens
        if sampling_params.min_tokens:
            self.min_tokens[req_index] = (
                sampling_params.min_tokens,
                sampling_params.all_stop_token_ids,
            )

        # Logprobs
        if sampling_params.logprobs is not None:
            self.num_logprobs[req_id] = (
                self.vocab_size
                if sampling_params.logprobs == -1
                else sampling_params.logprobs
            )

        # Logit bias
        if sampling_params.logit_bias is not None:
            self.logit_bias[req_index] = sampling_params.logit_bias

        # Allowed token IDs
        if sampling_params.allowed_token_ids:
            self.has_allowed_token_ids.add(req_id)
            if self.allowed_token_ids_mask_cpu_tensor is None:
                # Lazy allocation
                self.allowed_token_ids_mask = torch.zeros(
                    self.max_num_reqs,
                    self.vocab_size,
                    dtype=torch.bool,
                    device=self.device,
                )
                self.allowed_token_ids_mask_cpu_tensor = torch.zeros(
                    self.max_num_reqs,
                    self.vocab_size,
                    dtype=torch.bool,
                    device="cpu",
                )
            self.allowed_token_ids_mask_cpu_tensor[req_index] = True
            self.allowed_token_ids_mask_cpu_tensor[req_index][
                sampling_params.allowed_token_ids
            ] = False

        # Bad words
        if sampling_params.bad_words_token_ids:
            self.bad_words_token_ids[req_index] = sampling_params.bad_words_token_ids

    def remove_request(self, req_id: str) -> Optional[int]:
        """Remove a request from the batch.

        This method must always be followed by a call to condense().

        Args:
            req_id: ID of the request to remove

        Returns:
            The removed request index, or None if not found
        """
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None

        # Clear request slot
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None

        # Clear sampling parameters
        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.min_p_reqs.discard(req_id)
        self.min_tokens.pop(req_index, None)
        self.frequency_penalties_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.in_progress_prompt_logprobs_cpu.pop(req_id, None)

        # Clear LoRA
        lora_id = self.request_lora_mapping[req_index]
        if lora_id != 0:
            self.lora_id_to_request_ids[lora_id].discard(req_id)
            if len(self.lora_id_to_request_ids[lora_id]) == 0:
                self.lora_id_to_request_ids.pop(lora_id)
                self.lora_id_to_lora_request.pop(lora_id)
            self.request_lora_mapping[req_index] = 0

        # Clear logit bias
        self.logit_bias[req_index] = None

        # Clear allowed tokens
        self.has_allowed_token_ids.discard(req_id)
        if self.allowed_token_ids_mask_cpu_tensor is not None:
            self.allowed_token_ids_mask_cpu_tensor[req_index].fill_(False)

        # Clear bad words
        self.bad_words_token_ids.pop(req_index, None)

        return req_index

    def condense_indices(self, indices: list[int]) -> None:
        """Compact the batch by removing empty slots.

        This method is called after remove_request() to consolidate
        the batch and remove gaps left by removed requests.

        Args:
            indices: List of indices that were removed (sorted descending)
        """
        num_reqs = self.num_reqs

        if not indices:
            return

        if num_reqs == 0:
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            return

        # Sort indices in descending order
        empty_indices = sorted(indices, reverse=True)
        last_req_index = num_reqs + len(empty_indices) - 1

        while empty_indices:
            # Find the largest non-empty index
            while last_req_index in empty_indices:
                last_req_index -= 1

            # Find the smallest empty index
            empty_index = empty_indices[-1]
            if empty_index >= last_req_index:
                break

            # Move active request down into empty index
            empty_indices.pop()
            req_id = self._req_ids[last_req_index]
            output_token_ids = self.req_output_token_ids[last_req_index]
            assert req_id is not None

            self._req_ids[empty_index] = req_id
            self._req_ids[last_req_index] = None
            self.req_output_token_ids[empty_index] = output_token_ids
            self.req_output_token_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            # Move token data
            num_tokens = self.num_tokens[last_req_index]
            self.token_ids_cpu[empty_index, :num_tokens] = self.token_ids_cpu[
                last_req_index, :num_tokens
            ]
            self.num_tokens[empty_index] = num_tokens
            self.num_prompt_tokens[empty_index] = self.num_prompt_tokens[last_req_index]
            self.num_computed_tokens_cpu[empty_index] = self.num_computed_tokens_cpu[
                last_req_index
            ]

            # Move block table
            self.block_table.move_row(last_req_index, empty_index)

            # Move LoRA mapping
            self.request_lora_mapping[empty_index] = self.request_lora_mapping[
                last_req_index
            ]

            # Move sampling parameters
            self.temperature_cpu[empty_index] = self.temperature_cpu[last_req_index]
            self.top_p_cpu[empty_index] = self.top_p_cpu[last_req_index]
            self.top_k_cpu[empty_index] = self.top_k_cpu[last_req_index]
            self.min_p_cpu[empty_index] = self.min_p_cpu[last_req_index]
            self.frequency_penalties_cpu[empty_index] = self.frequency_penalties_cpu[
                last_req_index
            ]
            self.presence_penalties_cpu[empty_index] = self.presence_penalties_cpu[
                last_req_index
            ]
            self.repetition_penalties_cpu[empty_index] = self.repetition_penalties_cpu[
                last_req_index
            ]

            # Move generator
            generator = self.generators.pop(last_req_index, None)
            if generator is not None:
                self.generators[empty_index] = generator

            # Move min_tokens
            min_tokens_data = self.min_tokens.pop(last_req_index, None)
            if min_tokens_data is not None:
                self.min_tokens[empty_index] = min_tokens_data

            # Move allowed token IDs mask
            if self.allowed_token_ids_mask_cpu_tensor is not None:
                self.allowed_token_ids_mask_cpu_tensor[empty_index] = (
                    self.allowed_token_ids_mask_cpu_tensor[last_req_index]
                )

            # Move bad words
            bad_words = self.bad_words_token_ids.pop(last_req_index, None)
            if bad_words is not None:
                self.bad_words_token_ids[empty_index] = bad_words

            # Move logit bias
            self.logit_bias[empty_index] = self.logit_bias[last_req_index]
            self.logit_bias[last_req_index] = None

            last_req_index -= 1

        # Trim lists to the batch size
        del self._req_ids[num_reqs:]
        del self.req_output_token_ids[num_reqs:]

    def get_model_inputs(self) -> dict:
        """Build inputs for the model forward pass.

        This prepares all tensors needed for model execution, including
        input IDs, positions, and attention metadata.

        Returns:
            Dictionary containing model input tensors
        """
        num_reqs = self.num_reqs
        if num_reqs == 0:
            return {}

        # Copy sampling parameters to device
        if not self.all_greedy:
            copy_slice(self.temperature_cpu_tensor, self.temperature, num_reqs)
        if not self.no_top_p:
            copy_slice(self.top_p_cpu_tensor, self.top_p, num_reqs)
        if not self.no_top_k:
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)
        if not self.no_min_p:
            copy_slice(self.min_p_cpu_tensor, self.min_p, num_reqs)
        if not self.no_penalties:
            copy_slice(
                self.frequency_penalties_cpu_tensor,
                self.frequency_penalties,
                num_reqs,
            )
            copy_slice(
                self.presence_penalties_cpu_tensor,
                self.presence_penalties,
                num_reqs,
            )
            copy_slice(
                self.repetition_penalties_cpu_tensor,
                self.repetition_penalties,
                num_reqs,
            )

        # Copy allowed token IDs mask if needed
        if not self.no_allowed_token_ids:
            assert self.allowed_token_ids_mask is not None
            copy_slice(
                self.allowed_token_ids_mask_cpu_tensor,
                self.allowed_token_ids_mask,
                num_reqs,
            )

        return {
            "num_reqs": num_reqs,
            "req_ids": self.req_ids,
        }

    def commit_step(self, sampled_token_ids: torch.Tensor) -> None:
        """Update batch state after a sampling step.

        This commits the newly sampled tokens to the batch state,
        updating output token lists and computed token counts.

        Args:
            sampled_token_ids: Tensor of sampled token IDs [num_reqs]
        """
        num_reqs = self.num_reqs
        if num_reqs == 0:
            return

        # Convert to CPU if needed
        if sampled_token_ids.device != torch.device("cpu"):
            sampled_token_ids_cpu = sampled_token_ids.cpu()
        else:
            sampled_token_ids_cpu = sampled_token_ids

        # Update output tokens
        sampled_ids = sampled_token_ids_cpu.squeeze(-1).tolist()
        for req_idx in range(num_reqs):
            req_id = self._req_ids[req_idx]
            if req_id is None:
                continue

            token_id = sampled_ids[req_idx]
            output_tokens = self.req_output_token_ids[req_idx]
            if output_tokens is not None:
                output_tokens.append(token_id)

            # Update token IDs
            num_tokens = self.num_tokens[req_idx]
            self.token_ids_cpu[req_idx, num_tokens] = token_id
            self.num_tokens[req_idx] = num_tokens + 1
            self.num_computed_tokens_cpu[req_idx] += 1

    def get_num_active_requests(self) -> int:
        """Get the number of active requests in the batch.

        Returns:
            Number of active requests
        """
        return self.num_reqs

    def make_sampling_metadata(self) -> SamplingMetadata:
        """Create sampling metadata for the current batch state.

        Returns:
            SamplingMetadata object for token sampling
        """
        num_reqs = self.num_reqs

        # Prepare temperature
        temperature = None
        if not self.all_greedy:
            temperature = copy_slice(
                self.temperature_cpu_tensor, self.temperature, num_reqs
            )

        # Prepare prompt token IDs if needed for penalties
        prompt_token_ids = None
        if not self.no_penalties:
            prompt_token_ids = self._make_prompt_token_ids_tensor()

        # Prepare output token IDs if needed
        output_token_ids = (
            cast(list[list[int]], self.req_output_token_ids)
            if not self.no_penalties or bool(self.bad_words_token_ids)
            else []
        )

        # Prepare allowed token IDs mask
        allowed_token_ids_mask = None
        if not self.no_allowed_token_ids:
            assert self.allowed_token_ids_mask is not None
            allowed_token_ids_mask = self.allowed_token_ids_mask[:num_reqs]

        return SamplingMetadata(
            temperature=temperature,
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=None if self.no_top_p else self.top_p[:num_reqs],
            top_k=None if self.no_top_k else self.top_k[:num_reqs],
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=self.frequency_penalties[:num_reqs],
            presence_penalties=self.presence_penalties[:num_reqs],
            repetition_penalties=self.repetition_penalties[:num_reqs],
            output_token_ids=output_token_ids,
            spec_token_ids=[],  # Apple doesn't support spec decode yet
            no_penalties=self.no_penalties,
            allowed_token_ids_mask=allowed_token_ids_mask,
            bad_words_token_ids=self.bad_words_token_ids,
            logitsprocs=None,  # Simplified for Apple
        )

    def _make_prompt_token_ids_tensor(self) -> torch.Tensor:
        """Create a tensor of prompt token IDs for the current batch.

        Returns:
            Tensor of shape [num_reqs, max_prompt_len]
        """
        num_reqs = self.num_reqs
        max_prompt_len = self.num_prompt_tokens[:num_reqs].max()

        prompt_token_ids_cpu_tensor = torch.empty(
            (num_reqs, max_prompt_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        prompt_token_ids = prompt_token_ids_cpu_tensor.numpy()
        prompt_token_ids[:] = self.token_ids_cpu[:num_reqs, :max_prompt_len]

        # Pad with vocab_size
        for i in range(num_reqs):
            prompt_token_ids[i, self.num_prompt_tokens[i]:] = self.vocab_size

        return prompt_token_ids_cpu_tensor.to(device=self.device, non_blocking=True)

    def make_lora_inputs(
        self,
        num_scheduled_tokens: np.ndarray,
        num_sampled_tokens: np.ndarray,
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        """Create LoRA inputs for the current batch.

        Args:
            num_scheduled_tokens: Number of scheduled tokens per request
            num_sampled_tokens: Number of sampled tokens per request

        Returns:
            Tuple of (prompt_lora_mapping, token_lora_mapping, lora_requests)
        """
        req_lora_mapping = self.request_lora_mapping[:self.num_reqs]
        prompt_lora_mapping = tuple(req_lora_mapping.repeat(num_sampled_tokens))
        token_lora_mapping = tuple(req_lora_mapping.repeat(num_scheduled_tokens))

        active_lora_requests: set[LoRARequest] = set(
            self.lora_id_to_lora_request.values()
        )

        return prompt_lora_mapping, token_lora_mapping, active_lora_requests

    # Properties for batch state queries

    @property
    def num_reqs(self) -> int:
        """Number of active requests."""
        return len(self.req_id_to_index)

    @property
    def all_greedy(self) -> bool:
        """Whether all requests use greedy sampling."""
        return len(self.random_reqs) == 0

    @property
    def all_random(self) -> bool:
        """Whether all requests use random sampling."""
        return len(self.greedy_reqs) == 0

    @property
    def no_top_p(self) -> bool:
        """Whether no requests use top-p sampling."""
        return len(self.top_p_reqs) == 0

    @property
    def no_top_k(self) -> bool:
        """Whether no requests use top-k sampling."""
        return len(self.top_k_reqs) == 0

    @property
    def no_min_p(self) -> bool:
        """Whether no requests use min-p sampling."""
        return len(self.min_p_reqs) == 0

    @property
    def no_penalties(self) -> bool:
        """Whether no requests use penalties."""
        return (
            len(self.presence_penalties_reqs) == 0
            and len(self.frequency_penalties_reqs) == 0
            and len(self.repetition_penalties_reqs) == 0
        )

    @property
    def max_num_logprobs(self) -> Optional[int]:
        """Maximum number of logprobs across all requests."""
        return max(self.num_logprobs.values()) if self.num_logprobs else None

    @property
    def no_allowed_token_ids(self) -> bool:
        """Whether no requests have allowed token ID restrictions."""
        return len(self.has_allowed_token_ids) == 0
