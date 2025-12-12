/**
 * Native KV cache write for Metal PagedAttention.
 *
 * This C function performs the KV cache update without Python overhead.
 * Compiled as a shared library and called via ctypes.
 *
 * Build: clang -shared -O3 -o libkv_write.dylib kv_write.c
 */

#include <stdint.h>
#include <string.h>

/**
 * Write KV data to MTLBuffer cache in strided layout.
 *
 * Layout: [num_blocks, num_kv_heads, block_size, head_size]
 * For (block_id, head_idx, token_offset):
 *     element_offset = block_id * stride_block + head_idx * stride_head + token_offset * stride_token
 *
 * @param key_dst       Destination pointer to key MTLBuffer
 * @param value_dst     Destination pointer to value MTLBuffer
 * @param key_src       Source key data [num_tokens, num_kv_heads, head_size] float16
 * @param value_src     Source value data [num_tokens, num_kv_heads, head_size] float16
 * @param block_ids     Block IDs for each token [num_tokens] int64
 * @param token_offsets Token offsets within blocks [num_tokens] int64
 * @param num_tokens    Number of tokens to write
 * @param num_kv_heads  Number of KV heads
 * @param head_size     Size of each head (in float16 elements)
 * @param stride_block  Elements per block
 * @param stride_head   Elements per head
 * @param stride_token  Elements per token position
 */
void metal_write_kv_batch(
    uint16_t* key_dst,
    uint16_t* value_dst,
    const uint16_t* key_src,
    const uint16_t* value_src,
    const int64_t* block_ids,
    const int64_t* token_offsets,
    int num_tokens,
    int num_kv_heads,
    int head_size,
    int stride_block,
    int stride_head,
    int stride_token
) {
    const size_t head_bytes = head_size * sizeof(uint16_t);
    const int src_token_stride = num_kv_heads * head_size;

    for (int t = 0; t < num_tokens; ++t) {
        int64_t block_id = block_ids[t];
        int64_t token_off = token_offsets[t];

        // Base offset for this (block, token)
        size_t base_offset = (size_t)block_id * stride_block + (size_t)token_off * stride_token;

        // Source base for this token
        const uint16_t* src_k_base = key_src + t * src_token_stride;
        const uint16_t* src_v_base = value_src + t * src_token_stride;

        for (int h = 0; h < num_kv_heads; ++h) {
            // Destination offset for this head
            size_t dst_offset = base_offset + (size_t)h * stride_head;

            // Source offset for this head (contiguous in source)
            const uint16_t* src_k = src_k_base + h * head_size;
            const uint16_t* src_v = src_v_base + h * head_size;

            // Destination pointers
            uint16_t* dst_k = key_dst + dst_offset;
            uint16_t* dst_v = value_dst + dst_offset;

            // Copy head data
            memcpy(dst_k, src_k, head_bytes);
            memcpy(dst_v, src_v, head_bytes);
        }
    }
}

/**
 * Optimized version for single token decode (num_tokens=1).
 * Unrolls the head loop for common head counts.
 */
void metal_write_kv_single(
    uint16_t* key_dst,
    uint16_t* value_dst,
    const uint16_t* key_src,
    const uint16_t* value_src,
    int64_t block_id,
    int64_t token_offset,
    int num_kv_heads,
    int head_size,
    int stride_block,
    int stride_head,
    int stride_token
) {
    const size_t head_bytes = head_size * sizeof(uint16_t);
    size_t base_offset = (size_t)block_id * stride_block + (size_t)token_offset * stride_token;

    for (int h = 0; h < num_kv_heads; ++h) {
        size_t dst_offset = base_offset + (size_t)h * stride_head;
        size_t src_offset = (size_t)h * head_size;

        memcpy(key_dst + dst_offset, key_src + src_offset, head_bytes);
        memcpy(value_dst + dst_offset, value_src + src_offset, head_bytes);
    }
}
