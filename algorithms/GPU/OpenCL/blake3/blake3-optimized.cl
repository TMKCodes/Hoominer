#define BLAKE3_VERSION_STRING "1.8.2"
#define BLAKE3_KEY_LEN 32
#define BLAKE3_OUT_LEN 32
#define BLAKE3_BLOCK_LEN 64
#define BLAKE3_CHUNK_LEN 1024
#define BLAKE3_MAX_DEPTH 54
#define BLAKE3_BLOCK_LEN_LOG2 6
#define BLAKE3_CHUNK_LEN_LOG2 10

// BLAKE3 initialization vector as macros
#define IV_0 0x6A09E667UL
#define IV_1 0xBB67AE85UL
#define IV_2 0x3C6EF372UL
#define IV_3 0xA54FF53AUL
#define IV_4 0x510E527FUL
#define IV_5 0x9B05688CUL
#define IV_6 0x1F83D9ABUL
#define IV_7 0x5BE0CD19UL

// BLAKE3 flags
enum blake3_flags {
    CHUNK_START         = 1 << 0,
    CHUNK_END           = 1 << 1,
    PARENT              = 1 << 2,
    ROOT                = 1 << 3,
    KEYED_HASH          = 1 << 4,
    DERIVE_KEY_CONTEXT  = 1 << 5,
    DERIVE_KEY_MATERIAL = 1 << 6,
};

// Message schedule for BLAKE3 rounds
__constant uchar MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

// Data structures
typedef struct {
    uint cv[8];
    ulong chunk_counter;
    uchar buf[BLAKE3_BLOCK_LEN];
    uchar buf_len;
    uchar blocks_compressed;
    uchar flags;
} blake3_chunk_state;

typedef struct {
    uint key[8];
    blake3_chunk_state chunk;
    uchar cv_stack_len;
    uchar cv_stack[(BLAKE3_MAX_DEPTH + 1) * BLAKE3_OUT_LEN];
} blake3_hasher;

typedef struct {
    uint input_cv[8];
    ulong counter;
    char block[BLAKE3_BLOCK_LEN];
    char block_len;
    char flags;
} output_t;

// Initialize chunk state
inline void chunk_state_init(blake3_chunk_state *self, const uint key[8], uchar flags) {
    vstore4(vload4(0, key), 0, self->cv);
    vstore4(vload4(1, key), 1, self->cv);
    self->chunk_counter = 0;
    self->buf_len = 0;
    self->blocks_compressed = 0;
    self->flags = flags;
    vstore16((uchar16)(0), 0, self->buf);
}

// Get current chunk length
inline size_t chunk_state_len(const blake3_chunk_state *self) {
    return (BLAKE3_BLOCK_LEN * (size_t)self->blocks_compressed) + ((size_t)self->buf_len);
}

// Determine if CHUNK_START flag should be set
inline uchar chunk_state_maybe_start_flag(const blake3_chunk_state *self) {
    return (self->blocks_compressed == 0) ? CHUNK_START : 0;
}

// Create output structure
inline output_t make_output(const uint input_cv[8], const uchar block[BLAKE3_BLOCK_LEN],
                          uchar block_len, ulong counter, uchar flags) {
    output_t ret;
    vstore4(vload4(0, input_cv), 0, ret.input_cv);
    vstore4(vload4(1, input_cv), 1, ret.input_cv);
    vstore16(vload16(0, block), 0, (uchar *)ret.block);
    ret.block_len = block_len;
    ret.counter = counter;
    ret.flags = flags;
    return ret;
}

// Get chunk state output
inline output_t chunk_state_output(const blake3_chunk_state *self) {
    uchar block_flags = self->flags | chunk_state_maybe_start_flag(self) | CHUNK_END;
    return make_output(self->cv, self->buf, self->buf_len, self->chunk_counter, block_flags);
}

// Utility functions
inline uint load32(const void *src) {
    return vload4(0, (const uint *)src).s0;
}

inline void store32(void *dst, uint w) {
    vstore4((uint4)(w, 0, 0, 0), 0, (uint *)dst);
}

inline uint counter_low(ulong counter) { return (uint)counter; }
inline uint counter_high(ulong counter) { return (uint)(counter >> 32); }
inline uint rotr32(uint w, uint c) { return rotate(w, (uint)(32 - c)); }

// BLAKE3 G function with vectorization
inline void g_vec(uint4 *state, uint4 msg0, uint4 msg1) {
    state[0] += state[1] + msg0;
    state[3] = rotate(state[3] ^ state[0], 16u);
    state[2] += state[3];
    state[1] = rotate(state[1] ^ state[2], 12u);
    state[0] += state[1] + msg1;
    state[3] = rotate(state[3] ^ state[0], 8u);
    state[2] += state[3];
    state[1] = rotate(state[1] ^ state[2], 7u);
}

// BLAKE3 round function with work-item parallelism
inline void round_fn(uint *state, const uint *msg, size_t round) {
    uint4 state_vec[4] = {vload4(0, state), vload4(1, state), vload4(2, state), vload4(3, state)};
    uint4 msg_vec[4] = {vload4(0, msg), vload4(1, msg), vload4(2, msg), vload4(3, msg)};
    uint lid = get_local_id(0);
    if (lid < 4) {
        uint4 msg_pair;
        msg_pair.xy = (lid & 1) ? msg_vec[lid / 2].zw : msg_vec[lid / 2].xy;
        msg_pair.zw = (lid & 1) ? msg_vec[lid / 2].xy : msg_vec[lid / 2].zw;
        g_vec(&state_vec[lid / 2], msg_pair, msg_pair);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 4) {
        vstore4(state_vec[lid], lid, state);
    }
}

// Core compression function
inline void compress_pre(uint4 state[4], const uint cv[8], const uchar block[BLAKE3_BLOCK_LEN],
                        uchar block_len, ulong counter, uchar flags) {
    block_len = min(block_len, BLAKE3_BLOCK_LEN);
    uint4 block_words[4];
    for (int i = 0; i < 4; i++) {
        block_words[i] = vload4(i, (const uint *)block);
    }
    state[0] = vload4(0, cv);
    state[1] = vload4(1, cv);
    state[2] = (uint4)(IV_0, IV_1, IV_2, IV_3);
    state[3] = (uint4)(IV_4, IV_5, IV_6, IV_7);
    state[2].s3 = counter_low(counter);
    state[3].s0 = counter_high(counter);
    state[3].s1 = (uint)block_len;
    state[3].s2 = (uint)flags;
    round_fn((uint *)state, (uint *)block_words, 0);
    round_fn((uint *)state, (uint *)block_words, 1);
    round_fn((uint *)state, (uint *)block_words, 2);
    round_fn((uint *)state, (uint *)block_words, 3);
    round_fn((uint *)state, (uint *)block_words, 4);
    round_fn((uint *)state, (uint *)block_words, 5);
    round_fn((uint *)state, (uint *)block_words, 6);
}

// Main compression function
inline void blake3_compress_in_place(uint cv[8], const uchar block[BLAKE3_BLOCK_LEN],
                                    uchar block_len, ulong counter, uchar flags) {
    uint4 state[4];
    compress_pre(state, cv, block, block_len, counter, flags);
    vstore4(state[0] ^ state[2], 0, cv);
    vstore4(state[1] ^ state[3], 1, cv);
}

// Extended output compression
inline void blake3_compress_xof(const uint cv[8], const uchar block[BLAKE3_BLOCK_LEN],
                               uchar block_len, ulong counter, uchar flags, uchar out[64]) {
    uint4 state[4];
    compress_pre(state, cv, block, block_len, counter, flags);
    vstore4(state[0] ^ state[2], 0, (uint *)out);
    vstore4(state[1] ^ state[3], 1, (uint *)out);
    vstore4(state[2] ^ vload4(0, cv), 2, (uint *)out);
    vstore4(state[3] ^ vload4(1, cv), 3, (uint *)out);
}

// Store chaining value
inline void store_cv_words(uchar bytes_out[32], uint cv_words[8]) {
    vstore4(vload4(0, cv_words), 0, (uint *)bytes_out);
    vstore4(vload4(1, cv_words), 1, (uint *)bytes_out);
}

// Output chaining value
inline void output_chaining_value(const output_t *self, uchar cv[32]) {
    uint cv_words[8];
    vstore4(vload4(0, self->input_cv), 0, cv_words);
    vstore4(vload4(1, self->input_cv), 1, cv_words);
    blake3_compress_in_place(cv_words, (const uchar *)self->block, self->block_len,
                           self->counter, self->flags);
    store_cv_words(cv, cv_words);
}

// Bit manipulation utilities
inline unsigned int highest_one(ulong x) {
    return sizeof(x) * 8 - clz(x);
}

inline ulong round_down_to_power_of_2(ulong x) {
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x - (x >> 1);
}

// Chunk state operations
inline size_t chunk_state_fill_buf(blake3_chunk_state *self, const uchar *input, size_t input_len) {
    size_t take = min(BLAKE3_BLOCK_LEN - (size_t)self->buf_len, input_len);
    vstore16(vload16(0, input), self->buf_len / 16, self->buf);
    self->buf_len += (uchar)take;
    return take;
}

void chunk_state_update(blake3_chunk_state *self, const uchar *input, size_t input_len) {
    size_t remaining = input_len;
    while (remaining > 0) {
        size_t take = chunk_state_fill_buf(self, input, remaining);
        input += take;
        remaining -= take;
        if (self->buf_len == BLAKE3_BLOCK_LEN) {
            blake3_compress_in_place(self->cv, self->buf, BLAKE3_BLOCK_LEN,
                                   self->chunk_counter, self->flags | chunk_state_maybe_start_flag(self));
            self->blocks_compressed++;
            self->buf_len = 0;
            vstore16((uchar16)(0), 0, self->buf);
        }
    }
}

void chunk_state_reset(blake3_chunk_state *self, const uint key[8], ulong chunk_counter) {
    vstore4(vload4(0, key), 0, self->cv);
    vstore4(vload4(1, key), 1, self->cv);
    self->chunk_counter = chunk_counter;
    self->blocks_compressed = 0;
    self->buf_len = 0;
    vstore16((uchar16)(0), 0, self->buf);
}

// Hasher operations
void hasher_merge_cv_stack(blake3_hasher *self, ulong total_len) {
    size_t post_merge_stack_len = (size_t)highest_one(total_len | 1);
    while (self->cv_stack_len > post_merge_stack_len) {
        uchar *parent_node = &self->cv_stack[(self->cv_stack_len - 2) * BLAKE3_OUT_LEN];
        output_t output = make_output(self->key, parent_node, BLAKE3_BLOCK_LEN, 0,
                                    self->chunk.flags | PARENT);
        output_chaining_value(&output, parent_node);
        self->cv_stack_len--;
    }
}

inline void hasher_push_cv(blake3_hasher *self, uchar new_cv[BLAKE3_OUT_LEN], ulong chunk_counter) {
    hasher_merge_cv_stack(self, chunk_counter);
    vstore8(vload8(0, new_cv), self->cv_stack_len * BLAKE3_OUT_LEN / 8, self->cv_stack);
    self->cv_stack_len++;
}

// Hash processing
inline void blake3_hash_many(const uchar *const *inputs, size_t num_inputs, size_t blocks,
                            const uint key[8], ulong counter, bool increment_counter,
                            uchar flags, uchar flags_start, uchar flags_end, uchar *out) {
    uint lid = get_local_id(0);
    uint gid = get_global_id(0);
    if (gid < num_inputs) {
        uint cv[8];
        vstore4(vload4(0, key), 0, cv);
        vstore4(vload4(1, key), 1, cv);
        uchar block_flags = flags | flags_start;
        const uchar *input = inputs[gid];
        size_t remaining_blocks = blocks;
        while (remaining_blocks > 0) {
            if (remaining_blocks == 1) block_flags |= flags_end;
            blake3_compress_in_place(cv, input, BLAKE3_BLOCK_LEN, counter + gid,
                                   block_flags);
            input += BLAKE3_BLOCK_LEN;
            remaining_blocks--;
            block_flags = flags;
        }
        vstore4(vload4(0, cv), 0, (uint *)(out + gid * BLAKE3_OUT_LEN));
        vstore4(vload4(1, cv), 1, (uint *)(out + gid * BLAKE3_OUT_LEN));
    }
}

size_t compress_chunks_parallel(const uchar *input, size_t input_len, const uint key[8],
                              ulong chunk_counter, uchar flags, uchar *out) {
    size_t chunks_array_len = 0;
    size_t input_position = 0;
    const uchar *chunks_array[4];
    uint lid = get_local_id(0);
    while (input_len - input_position >= BLAKE3_CHUNK_LEN && chunks_array_len < 4) {
        chunks_array[chunks_array_len] = &input[input_position];
        input_position += BLAKE3_CHUNK_LEN;
        chunks_array_len++;
    }
    if (lid < chunks_array_len) {
        blake3_hash_many(&chunks_array[lid], 1, BLAKE3_CHUNK_LEN / BLAKE3_BLOCK_LEN,
                        key, chunk_counter + lid, true, flags, CHUNK_START, CHUNK_END,
                        &out[lid * BLAKE3_OUT_LEN]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (input_len > input_position && lid == 0) {
        blake3_chunk_state chunk_state;
        chunk_state_init(&chunk_state, key, flags);
        chunk_state.chunk_counter = chunk_counter + chunks_array_len;
        chunk_state_update(&chunk_state, &input[input_position], input_len - input_position);
        output_t output = chunk_state_output(&chunk_state);
        output_chaining_value(&output, &out[chunks_array_len * BLAKE3_OUT_LEN]);
        chunks_array_len++;
    }
    return chunks_array_len;
}

size_t blake3_simd_degree(void) { return 4; }

size_t left_subtree_len(size_t input_len) {
    size_t full_chunks = (input_len - 1) / BLAKE3_CHUNK_LEN;
    return round_down_to_power_of_2(full_chunks) * BLAKE3_CHUNK_LEN;
}

size_t compress_parents_parallel(const uchar *child_chaining_values, size_t num_chaining_values,
                               const uint key[8], uchar flags, uchar *out) {
    const uchar *parents_array[2];
    size_t parents_array_len = 0;
    while (num_chaining_values - (2 * parents_array_len) >= 2) {
        parents_array[parents_array_len] = &child_chaining_values[2 * parents_array_len * BLAKE3_OUT_LEN];
        parents_array_len++;
    }
    blake3_hash_many(parents_array, parents_array_len, 1, key, 0, false,
                    flags | PARENT, 0, 0, out);
    if (num_chaining_values > 2 * parents_array_len) {
        vstore8(vload8(2 * parents_array_len, child_chaining_values),
                parents_array_len * BLAKE3_OUT_LEN / 8, out);
        return parents_array_len + 1;
    }
    return parents_array_len;
}

size_t blake3_compress_subtree_wide(const uchar *input, size_t input_len,
                                  const uint key[8], ulong chunk_counter,
                                  uchar flags, uchar *out, bool use_tbb) {
    if (input_len <= blake3_simd_degree() * BLAKE3_CHUNK_LEN) {
        return compress_chunks_parallel(input, input_len, key, chunk_counter, flags, out);
    }
    size_t left_input_len = left_subtree_len(input_len);
    size_t right_input_len = input_len - left_input_len;
    const uchar *right_input = &input[left_input_len];
    ulong right_chunk_counter = chunk_counter + (left_input_len / BLAKE3_CHUNK_LEN);
    uchar cv_array[4 * BLAKE3_OUT_LEN];
    size_t degree = blake3_simd_degree();
    if (left_input_len > BLAKE3_CHUNK_LEN && degree == 1) {
        degree = 2;
    }
    uchar *right_cvs = &cv_array[degree * BLAKE3_OUT_LEN];
    size_t left_n = blake3_compress_subtree_wide(input, left_input_len, key, chunk_counter,
                                               flags, cv_array, use_tbb);
    size_t right_n = blake3_compress_subtree_wide(right_input, right_input_len, key,
                                                right_chunk_counter, flags, right_cvs,
                                                use_tbb);
    if (left_n == 1) {
        vstore16(vload16(0, cv_array), 0, out);
        return 2;
    }
    return compress_parents_parallel(cv_array, left_n + right_n, key, flags, out);
}

inline void compress_subtree_to_parent_node(const uchar *input, size_t input_len,
                                          const uint key[8], ulong chunk_counter,
                                          uchar flags, uchar out[2 * BLAKE3_OUT_LEN],
                                          bool use_tbb) {
    uchar cv_array[BLAKE3_OUT_LEN];
    size_t num_cvs = blake3_compress_subtree_wide(input, input_len, key, chunk_counter,
                                                flags, cv_array, use_tbb);
    vstore16(vload16(0, cv_array), 0, out);
}

void blake3_hasher_update(blake3_hasher *self, const uchar *input, size_t input_len) {
    if (input_len == 0) return;
    size_t remaining_len = input_len;
    const uchar *current_input = input;
    if (chunk_state_len(&self->chunk) > 0) {
        size_t take = min(BLAKE3_CHUNK_LEN - chunk_state_len(&self->chunk), input_len);
        chunk_state_update(&self->chunk, current_input, take);
        current_input += take;
        remaining_len -= take;
        if (remaining_len > 0) {
            output_t output = chunk_state_output(&self->chunk);
            uchar chunk_cv[BLAKE3_OUT_LEN];
            output_chaining_value(&output, chunk_cv);
            hasher_push_cv(self, chunk_cv, self->chunk.chunk_counter);
            chunk_state_reset(&self->chunk, self->key, self->chunk.chunk_counter + 1);
        }
    }
    while (remaining_len > BLAKE3_CHUNK_LEN) {
        size_t subtree_len = round_down_to_power_of_2(remaining_len);
        ulong count_so_far = self->chunk.chunk_counter * BLAKE3_CHUNK_LEN;
        while ((subtree_len - 1) & count_so_far) {
            subtree_len /= 2;
        }
        uchar cv_pair[2 * BLAKE3_OUT_LEN];
        size_t num_cvs = compress_chunks_parallel(current_input, subtree_len, self->key,
                                               self->chunk.chunk_counter, self->chunk.flags,
                                               cv_pair);
        for (size_t i = 0; i < num_cvs; i++) {
            hasher_push_cv(self, &cv_pair[i * BLAKE3_OUT_LEN],
                          self->chunk.chunk_counter + i * (subtree_len / BLAKE3_CHUNK_LEN / num_cvs));
        }
        self->chunk.chunk_counter += subtree_len / BLAKE3_CHUNK_LEN;
        current_input += subtree_len;
        remaining_len -= subtree_len;
    }
    if (remaining_len > 0) {
        chunk_state_update(&self->chunk, current_input, remaining_len);
        hasher_merge_cv_stack(self, self->chunk.chunk_counter);
    }
}

inline void blake3_hasher_init(blake3_hasher *self) {
    self->key[0] = IV_0;
    self->key[1] = IV_1;
    self->key[2] = IV_2;
    self->key[3] = IV_3;
    self->key[4] = IV_4;
    self->key[5] = IV_5;
    self->key[6] = IV_6;
    self->key[7] = IV_7;
    chunk_state_init(&self->chunk, self->key, 0);
    self->cv_stack_len = 0;
}

void blake3_xof_many(const uint cv[8], const uchar block[BLAKE3_BLOCK_LEN],
                    uchar block_len, ulong counter, uchar flags,
                    __global uchar out[64], size_t outblocks) {
    uint lid = get_local_id(0);
    if (lid < outblocks) {
        blake3_compress_xof(cv, block, block_len, counter + lid, flags,
                           &out[lid * 64]);
    }
}

inline void output_root_bytes(const output_t *self, ulong seek, __global uchar *out,
                             size_t out_len) {
    if (out_len == 0) return;
    ulong output_block_counter = seek >> 6;
    size_t offset_within_block = seek & 63;
    uchar wide_buf[64];
    if (offset_within_block) {
        blake3_compress_xof(self->input_cv, (const uchar *)self->block,
                           self->block_len, output_block_counter, self->flags | ROOT,
                           wide_buf);
        size_t bytes = min(out_len, 64 - offset_within_block);
        vstore16(vload16(0, wide_buf + offset_within_block), 0, out);
        out += bytes;
        out_len -= bytes;
        output_block_counter++;
    }
    if (out_len / 64) {
        blake3_xof_many(self->input_cv, (const uchar *)self->block, self->block_len,
                       output_block_counter, self->flags | ROOT, out, out_len / 64);
        output_block_counter += out_len / 64;
        out += out_len & -64;
        out_len -= out_len & -64;
    }
    if (out_len) {
        blake3_compress_xof(self->input_cv, (const uchar *)self->block,
                           self->block_len, output_block_counter, self->flags | ROOT,
                           wide_buf);
        vstore8(vload8(0, wide_buf), 0, out);
    }
}

void blake3_hasher_finalize_seek(const blake3_hasher *self, ulong seek,
                                __global uchar *out, size_t out_len) {
    if (out_len == 0) return;
    if (self->cv_stack_len == 0) {
        output_t output = chunk_state_output(&self->chunk);
        output_root_bytes(&output, seek, out, out_len);
        return;
    }
    output_t output;
    size_t cvs_remaining;
    if (chunk_state_len(&self->chunk) > 0) {
        cvs_remaining = self->cv_stack_len;
        output = chunk_state_output(&self->chunk);
    } else {
        cvs_remaining = self->cv_stack_len - 2;
        output = make_output(self->key, &self->cv_stack[cvs_remaining * BLAKE3_OUT_LEN],
                           BLAKE3_BLOCK_LEN, 0, self->chunk.flags | PARENT);
    }
    while (cvs_remaining > 0) {
        cvs_remaining--;
        uchar parent_block[BLAKE3_BLOCK_LEN];
        vstore8(vload8(cvs_remaining * BLAKE3_OUT_LEN / 8, self->cv_stack), 0, parent_block);
        output_chaining_value(&output, &parent_block[BLAKE3_OUT_LEN]);
        output = make_output(self->key, parent_block, BLAKE3_BLOCK_LEN, 0,
                           self->chunk.flags | PARENT);
    }
    output_root_bytes(&output, seek, out, out_len);
}

void blake3_hasher_finalize(const blake3_hasher *self, __global uchar *out,
                           size_t out_len) {
    blake3_hasher_finalize_seek(self, 0, out, out_len);
}

__kernel void blake3_hash(__global const uchar *input, ulong input_len,
                         __global uchar *out, ulong out_len) {
    // Input validation
    if (input_len == 0 || out_len == 0 || out_len > BLAKE3_OUT_LEN) {
        return;
    }
    
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, input, input_len);
    blake3_hasher_finalize(&hasher, out, out_len);
}

__kernel void blake3_test(__global const uchar *input, ulong input_len,
                         __global uchar *out, ulong out_len) {
    if (input_len < 3 || out_len < BLAKE3_OUT_LEN) {
        return;
    }
    
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, input, min(input_len, 3UL));
    
    if (out_len >= 96) {
        output_t output = chunk_state_output(&hasher.chunk);
        uint4 state[4];
        compress_pre(state, output.input_cv, (const uchar *)output.block,
                    output.block_len, output.counter, output.flags | ROOT);
        vstore16(vload16(0, (uint *)state), 2, out);
    }
    
    blake3_hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
}