#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL SELECT_ROUNDING_MODE rte
#pragma OPENCL FP_CONTRACT OFF

#define BLAKE3_VERSION_STRING "1.8.2"
#define BLAKE3_KEY_LEN 32
#define BLAKE3_OUT_LEN 32
#define BLAKE3_BLOCK_LEN 64
#define BLAKE3_CHUNK_LEN 1024
#define BLAKE3_MAX_DEPTH 54
#define BLAKE3_BLOCK_LEN_LOG2 6
#define BLAKE3_CHUNK_LEN_LOG2 10
#define DOMAIN_HASH_SIZE 32
#define RANDOM_TYPE_LEAN 0
#define RANDOM_TYPE_XOSHIRO 1
#define COMPLEX_TRANSFORM_MULTIPLIER 0.000001
#define PI 3.14159265358979323846

__constant uint IV[8] = {0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL,
                         0xA54FF53AUL, 0x510E527FUL, 0x9B05688CUL,
                         0x1F83D9ABUL, 0x5BE0CD19UL};

enum blake3_flags {
  CHUNK_START = 1 << 0,
  CHUNK_END = 1 << 1,
  PARENT = 1 << 2,
  ROOT = 1 << 3,
  KEYED_HASH = 1 << 4,
  DERIVE_KEY_CONTEXT = 1 << 5,
  DERIVE_KEY_MATERIAL = 1 << 6,
};

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
  uchar block[BLAKE3_BLOCK_LEN];
  uchar block_len;
  uchar flags;
} output_t;

__constant uchar MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

uint clz_own(ulong x) {
  if (x == 0)
    return 64;
  uint n = 0;
  while (!(x & (1UL << 63))) {
    x <<= 1;
    n++;
  }
  return n;
}

void chunk_state_init(blake3_chunk_state *self, const uint key[8],
                      uchar flags) {
  self->cv[0] = key[0];
  self->cv[1] = key[1];
  self->cv[2] = key[2];
  self->cv[3] = key[3];
  self->cv[4] = key[4];
  self->cv[5] = key[5];
  self->cv[6] = key[6];
  self->cv[7] = key[7];
  self->chunk_counter = 0;
  for (int i = 0; i < BLAKE3_BLOCK_LEN; i++) {
    self->buf[i] = 0;
  }
  self->buf_len = 0;
  self->blocks_compressed = 0;
  self->flags = flags;
}

size_t chunk_state_len(const blake3_chunk_state *self) {
  return (BLAKE3_BLOCK_LEN * (size_t)self->blocks_compressed) +
         ((size_t)self->buf_len);
}

uchar chunk_state_maybe_start_flag(const blake3_chunk_state *self) {
  return (self->blocks_compressed == 0) ? CHUNK_START : 0;
}

output_t make_output(const uint input_cv[8],
                     const uchar block[BLAKE3_BLOCK_LEN], uchar block_len,
                     ulong counter, uchar flags) {
  output_t ret;
  for (int i = 0; i < 8; i++) {
    ret.input_cv[i] = input_cv[i];
  }
  for (int i = 0; i < BLAKE3_BLOCK_LEN; i++) {
    ret.block[i] = block[i];
  }
  ret.block_len = block_len;
  ret.counter = counter;
  ret.flags = flags;
  return ret;
}

output_t chunk_state_output(const blake3_chunk_state *self) {
  uchar block_flags =
      self->flags | chunk_state_maybe_start_flag(self) | CHUNK_END;
  return make_output(self->cv, self->buf, self->buf_len, self->chunk_counter,
                     block_flags);
}

uint load32(const void *src) {
  const uchar *p = (const uchar *)src;
  return ((uint)(p[0]) << 0) | ((uint)(p[1]) << 8) | ((uint)(p[2]) << 16) |
         ((uint)(p[3]) << 24);
}
// __attribute__((noinline))
uint counter_low(ulong counter) { return (uint)counter; }

// __attribute__((noinline))
uint counter_high(ulong counter) { return (uint)(counter >> 32); }

uint rotr32(uint w, uint c) { return (w >> c) | (w << (32 - c)); }

void g(uint *state, size_t a, size_t b, size_t c, size_t d, uint x, uint y) {
  state[a] = state[a] + state[b] + x;
  state[d] = rotr32(state[d] ^ state[a], 16);
  state[c] = state[c] + state[d];
  state[b] = rotr32(state[b] ^ state[c], 12);
  state[a] = state[a] + state[b] + y;
  state[d] = rotr32(state[d] ^ state[a], 8);
  state[c] = state[c] + state[d];
  state[b] = rotr32(state[b] ^ state[c], 7);
}

void round_fn(uint state[16], const uint *msg, size_t round) {
  g(state, 0, 4, 8, 12, msg[MSG_SCHEDULE[round][0]],
    msg[MSG_SCHEDULE[round][1]]);
  g(state, 1, 5, 9, 13, msg[MSG_SCHEDULE[round][2]],
    msg[MSG_SCHEDULE[round][3]]);
  g(state, 2, 6, 10, 14, msg[MSG_SCHEDULE[round][4]],
    msg[MSG_SCHEDULE[round][5]]);
  g(state, 3, 7, 11, 15, msg[MSG_SCHEDULE[round][6]],
    msg[MSG_SCHEDULE[round][7]]);
  g(state, 0, 5, 10, 15, msg[MSG_SCHEDULE[round][8]],
    msg[MSG_SCHEDULE[round][9]]);
  g(state, 1, 6, 11, 12, msg[MSG_SCHEDULE[round][10]],
    msg[MSG_SCHEDULE[round][11]]);
  g(state, 2, 7, 8, 13, msg[MSG_SCHEDULE[round][12]],
    msg[MSG_SCHEDULE[round][13]]);
  g(state, 3, 4, 9, 14, msg[MSG_SCHEDULE[round][14]],
    msg[MSG_SCHEDULE[round][15]]);
}

void compress_pre(uint state[16], const uint cv[8],
                  const uchar block[BLAKE3_BLOCK_LEN], uchar block_len,
                  ulong counter, uchar flags) {
  if (block_len > BLAKE3_BLOCK_LEN) {
    block_len = BLAKE3_BLOCK_LEN;
  }
  uint block_words[16];
  for (int i = 0; i < 16; i++) {
    block_words[i] = load32(block + 4 * i);
  }
  for (int i = 0; i < 8; i++) {
    state[i] = cv[i];
    state[i + 8] = IV[i];
  }
  state[12] = counter_low(counter);
  state[13] = counter_high(counter);
  state[14] = (uint)block_len;
  state[15] = (uint)flags;
  for (int i = 0; i < 7; i++) {
    round_fn(state, block_words, i);
  }
}

void blake3_compress_in_place_portable(uint cv[8],
                                       const uchar block[BLAKE3_BLOCK_LEN],
                                       uchar block_len, ulong counter,
                                       uchar flags) {
  uint state[16];
  compress_pre(state, cv, block, block_len, counter, flags);
  for (int i = 0; i < 8; i++) {
    cv[i] = state[i] ^ state[i + 8];
  }
}

output_t parent_output(const uchar block[BLAKE3_BLOCK_LEN], const uint key[8],
                       uchar flags) {
  return make_output(key, block, BLAKE3_BLOCK_LEN, 0, flags | PARENT);
}

void blake3_compress_in_place(uint cv[8], const uchar block[BLAKE3_BLOCK_LEN],
                              uchar block_len, ulong counter, uchar flags) {
  blake3_compress_in_place_portable(cv, block, block_len, counter, flags);
}

void store32(uchar *dst, uint w) {
  dst[0] = (uchar)(w >> 0);
  dst[1] = (uchar)(w >> 8);
  dst[2] = (uchar)(w >> 16);
  dst[3] = (uchar)(w >> 24);
}

void store_cv_words(uchar bytes_out[32], uint cv_words[8]) {
  for (int i = 0; i < 8; i++) {
    store32(&bytes_out[i * 4], cv_words[i]);
  }
}

void output_chaining_value(const output_t *self, uchar cv[32]) {
  uint cv_words[8];
  for (int i = 0; i < 8; i++) {
    cv_words[i] = self->input_cv[i];
  }
  blake3_compress_in_place(cv_words, self->block, self->block_len,
                           self->counter, self->flags);
  store_cv_words(cv, cv_words);
}
// __attribute__((noinline))
uint highest_one(ulong x) {
  if (x == 0ul)
    return 0u;
  // clz(x) is defined for x!=0; return index (1..64)
  return 64u - (uint)clz_own(x);
}

void hasher_merge_cv_stack(blake3_hasher *self, ulong total_len) {
  size_t post_merge_stack_len = (size_t)highest_one(total_len | 1);
  while (self->cv_stack_len > post_merge_stack_len) {
    uchar *parent_node =
        &self->cv_stack[(self->cv_stack_len - 2) * BLAKE3_OUT_LEN];
    output_t output = parent_output(parent_node, self->key, self->chunk.flags);
    output_chaining_value(&output, parent_node);
    self->cv_stack_len -= 1;
  }
}

void hasher_push_cv(blake3_hasher *self, uchar new_cv[BLAKE3_OUT_LEN],
                    ulong chunk_counter) {
  hasher_merge_cv_stack(self, chunk_counter);
  for (int i = 0; i < BLAKE3_OUT_LEN; i++) {
    self->cv_stack[self->cv_stack_len * BLAKE3_OUT_LEN + i] = new_cv[i];
  }
  self->cv_stack_len += 1;
}

size_t chunk_state_fill_buf(blake3_chunk_state *self, const uchar *input,
                            size_t input_len) {
  size_t take = BLAKE3_BLOCK_LEN - ((size_t)self->buf_len);
  if (take > input_len) {
    take = input_len;
  }
  uchar *dest = self->buf + ((size_t)self->buf_len);
  for (size_t i = 0; i < take; i++) {
    dest[i] = input[i];
  }
  self->buf_len += (uchar)take;
  return take;
}

void chunk_state_update(blake3_chunk_state *self, const uchar *input,
                        size_t input_len) {
  if (self->buf_len > 0) {
    size_t take = chunk_state_fill_buf(self, (const uchar *)input, input_len);
    input += take;
    input_len -= take;
    if (input_len > 0) {
      blake3_compress_in_place(
          self->cv, self->buf, BLAKE3_BLOCK_LEN, self->chunk_counter,
          self->flags | chunk_state_maybe_start_flag(self));
      self->blocks_compressed += 1;
      self->buf_len = 0;
      for (int i = 0; i < BLAKE3_BLOCK_LEN; i++) {
        self->buf[i] = 0;
      }
    }
  }
  while (input_len > BLAKE3_BLOCK_LEN) {
    blake3_compress_in_place(self->cv, input, BLAKE3_BLOCK_LEN,
                             self->chunk_counter,
                             self->flags | chunk_state_maybe_start_flag(self));
    self->blocks_compressed += 1;
    input += BLAKE3_BLOCK_LEN;
    input_len -= BLAKE3_BLOCK_LEN;
  }
  chunk_state_fill_buf(self, (const uchar *)input, input_len);
}

void chunk_state_reset(blake3_chunk_state *self, const uint key[8],
                       ulong chunk_counter) {
  self->cv[0] = key[0];
  self->cv[1] = key[1];
  self->cv[2] = key[2];
  self->cv[3] = key[3];
  self->cv[4] = key[4];
  self->cv[5] = key[5];
  self->cv[6] = key[6];
  self->cv[7] = key[7];
  self->chunk_counter = chunk_counter;
  self->blocks_compressed = 0;
  for (int i = 0; i < BLAKE3_BLOCK_LEN; i++) {
    self->buf[i] = 0;
  }
  self->buf_len = 0;
}
// __attribute__((noinline))
ulong round_down_to_power_of_2(ulong x) {
  if (x == 0ul)
    return 0ul;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  return x - (x >> 1);
}

void hash_one_portable(const uchar *input, size_t blocks, const uint key[8],
                       ulong counter, uchar flags, uchar flags_start,
                       uchar flags_end, uchar out[BLAKE3_OUT_LEN]) {
  uint cv[8];
  for (int i = 0; i < 8; i++) {
    cv[i] = key[i];
  }
  uchar block_flags = flags | flags_start;
  while (blocks > 0) {
    if (blocks == 1) {
      block_flags |= flags_end;
    }
    blake3_compress_in_place_portable(cv, input, BLAKE3_BLOCK_LEN, counter,
                                      block_flags);
    input = &input[BLAKE3_BLOCK_LEN];
    blocks -= 1;
    block_flags = flags;
  }
  store_cv_words(out, cv);
}

void blake3_hash_many_portable(const uchar *const *inputs, size_t num_inputs,
                               size_t blocks, const uint key[8], ulong counter,
                               bool increment_counter, uchar flags,
                               uchar flags_start, uchar flags_end, uchar *out) {
  while (num_inputs > 0) {
    hash_one_portable(inputs[0], blocks, key, counter, flags, flags_start,
                      flags_end, out);
    if (increment_counter) {
      counter += 1;
    }
    inputs += 1;
    num_inputs -= 1;
    out = &out[BLAKE3_OUT_LEN];
  }
}

void blake3_hash_many(const uchar *const *inputs, size_t num_inputs,
                      size_t blocks, const uint key[8], ulong counter,
                      bool increment_counter, uchar flags, uchar flags_start,
                      uchar flags_end, uchar *out) {
  blake3_hash_many_portable(inputs, num_inputs, blocks, key, counter,
                            increment_counter, flags, flags_start, flags_end,
                            out);
}

size_t compress_parents_parallel(const uchar *child_chaining_values,
                                 size_t num_chaining_values, const uint key[8],
                                 uchar flags, uchar *out) {
  const uchar *parents_array[2];
  size_t parents_array_len = 0;
  while (num_chaining_values - (2 * parents_array_len) >= 2) {
    parents_array[parents_array_len] =
        &child_chaining_values[2 * parents_array_len * BLAKE3_OUT_LEN];
    parents_array_len += 1;
  }
  blake3_hash_many(parents_array, parents_array_len, 1, key, 0, false,
                   flags | PARENT, 0, 0, out);
  if (num_chaining_values > 2 * parents_array_len) {
    for (size_t i = 0; i < BLAKE3_OUT_LEN; i++) {
      out[parents_array_len * BLAKE3_OUT_LEN + i] =
          child_chaining_values[2 * parents_array_len * BLAKE3_OUT_LEN + i];
    }
    return parents_array_len + 1;
  }
  return parents_array_len;
}

size_t blake3_simd_degree(void) { return 1; }

size_t compress_chunks_parallel(const uchar *input, size_t input_len,
                                const uint key[8], ulong chunk_counter,
                                uchar flags, uchar *out) {
  const uchar *chunks_array[1];
  size_t input_position = 0;
  size_t chunks_array_len = 0;
  while (input_len - input_position >= BLAKE3_CHUNK_LEN) {
    chunks_array[chunks_array_len] = &input[input_position];
    input_position += BLAKE3_CHUNK_LEN;
    chunks_array_len += 1;
  }
  blake3_hash_many(chunks_array, chunks_array_len,
                   BLAKE3_CHUNK_LEN / BLAKE3_BLOCK_LEN, key, chunk_counter,
                   true, flags, CHUNK_START, CHUNK_END, out);
  if (input_len > input_position) {
    ulong counter = chunk_counter + (ulong)chunks_array_len;
    blake3_chunk_state chunk_state;
    chunk_state_init(&chunk_state, key, flags);
    chunk_state.chunk_counter = counter;
    chunk_state_update(&chunk_state, &input[input_position],
                       input_len - input_position);
    output_t output = chunk_state_output(&chunk_state);
    output_chaining_value(&output, &out[chunks_array_len * BLAKE3_OUT_LEN]);
    return chunks_array_len + 1;
  }
  return chunks_array_len;
}

size_t left_subtree_len(size_t input_len) {
  size_t full_chunks = (input_len - 1) / BLAKE3_CHUNK_LEN;
  return round_down_to_power_of_2(full_chunks) * BLAKE3_CHUNK_LEN;
}

size_t blake3_compress_subtree_wide(const uchar *input, size_t input_len,
                                    const uint key[8], ulong chunk_counter,
                                    uchar flags, uchar *out, bool use_tbb) {
  if (input_len <= blake3_simd_degree() * BLAKE3_CHUNK_LEN) {
    return compress_chunks_parallel(input, input_len, key, chunk_counter, flags,
                                    out);
  }
  size_t left_input_len = left_subtree_len(input_len);
  size_t right_input_len = input_len - left_input_len;
  const uchar *right_input = &input[left_input_len];
  ulong right_chunk_counter =
      chunk_counter + (ulong)(left_input_len / BLAKE3_CHUNK_LEN);
  uchar cv_array[2 * 2 * BLAKE3_OUT_LEN];
  size_t degree = blake3_simd_degree();
  if (left_input_len > BLAKE3_CHUNK_LEN && degree == 1) {
    degree = 2;
  }
  uchar *right_cvs = &cv_array[degree * BLAKE3_OUT_LEN];
  size_t left_n = blake3_compress_subtree_wide(
      input, left_input_len, key, chunk_counter, flags, cv_array, use_tbb);
  size_t right_n = blake3_compress_subtree_wide(right_input, right_input_len,
                                                key, right_chunk_counter, flags,
                                                right_cvs, use_tbb);
  if (left_n == 1) {
    for (size_t i = 0; i < 2 * BLAKE3_OUT_LEN; i++) {
      out[i] = cv_array[i];
    }
    return 2;
  }
  size_t num_chaining_values = left_n + right_n;
  return compress_parents_parallel(cv_array, num_chaining_values, key, flags,
                                   out);
}

void compress_subtree_to_parent_node(const uchar *input, size_t input_len,
                                     const uint key[8], ulong chunk_counter,
                                     uchar flags, uchar out[2 * BLAKE3_OUT_LEN],
                                     bool use_tbb) {
  uchar cv_array[2 * BLAKE3_OUT_LEN];
  size_t num_cvs = blake3_compress_subtree_wide(
      input, input_len, key, chunk_counter, flags, cv_array, use_tbb);
  for (size_t i = 0; i < 2 * BLAKE3_OUT_LEN; i++) {
    out[i] = cv_array[i];
  }
}

void blake3_hasher_update_base(blake3_hasher *self, __private void *input,
                               size_t input_len, bool use_tbb) {
  if (input_len == 0) {
    return;
  }
  __private uchar *private_input_bytes = (__private uchar *)input;
  if (chunk_state_len(&self->chunk) > 0) {
    size_t take = BLAKE3_CHUNK_LEN - chunk_state_len(&self->chunk);
    if (take > input_len) {
      take = input_len;
    }
    chunk_state_update(&self->chunk, (const uchar *)private_input_bytes, take);
    private_input_bytes += take;
    input_len -= take;
    if (input_len > 0) {
      output_t output = chunk_state_output(&self->chunk);
      uchar chunk_cv[32];
      output_chaining_value(&output, chunk_cv);
      hasher_push_cv(self, chunk_cv, self->chunk.chunk_counter);
      chunk_state_reset(&self->chunk, self->key, self->chunk.chunk_counter + 1);
    } else {
      return;
    }
  }
  while (input_len > BLAKE3_CHUNK_LEN) {
    size_t subtree_len = round_down_to_power_of_2(input_len);
    ulong count_so_far = self->chunk.chunk_counter * BLAKE3_CHUNK_LEN;
    while ((((ulong)(subtree_len - 1)) & count_so_far) != 0) {
      subtree_len /= 2;
    }
    size_t subtree_chunks = subtree_len >> BLAKE3_CHUNK_LEN_LOG2;
    if (subtree_len <= BLAKE3_CHUNK_LEN) {
      blake3_chunk_state chunk_state;
      chunk_state_init(&chunk_state, self->key, self->chunk.flags);
      chunk_state.chunk_counter = self->chunk.chunk_counter;
      chunk_state_update(&chunk_state, (const uchar *)private_input_bytes,
                         subtree_len);
      output_t output = chunk_state_output(&chunk_state);
      uchar cv[BLAKE3_OUT_LEN];
      output_chaining_value(&output, cv);
      hasher_push_cv(self, cv, chunk_state.chunk_counter);
    } else {
      uchar cv_pair[2 * BLAKE3_OUT_LEN];
      compress_subtree_to_parent_node(
          (const uchar *)private_input_bytes, subtree_len, self->key,
          self->chunk.chunk_counter, self->chunk.flags, (uchar *)cv_pair,
          use_tbb);
      hasher_push_cv(self, (uchar *)cv_pair, self->chunk.chunk_counter);
      hasher_push_cv(self, (uchar *)&cv_pair[BLAKE3_OUT_LEN],
                     self->chunk.chunk_counter + (subtree_chunks / 2));
    }
    self->chunk.chunk_counter += subtree_chunks;
    private_input_bytes += subtree_len;
    input_len -= subtree_len;
  }
  if (input_len > 0) {
    chunk_state_update(&self->chunk, (const uchar *)private_input_bytes,
                       input_len);
    hasher_merge_cv_stack(self, self->chunk.chunk_counter);
  }
}

void hasher_init_base(blake3_hasher *self, uint key[8], uchar flags) {
  for (int i = 0; i < 8; i++) {
    self->key[i] = key[i];
  }
  chunk_state_init(&self->chunk, key, flags);
  self->cv_stack_len = 0;
}

void blake3_hasher_init(blake3_hasher *self) {
  uint local_IV[8];
  for (int i = 0; i < 8; i++) {
    local_IV[i] = IV[i];
  }
  hasher_init_base(self, local_IV, 0);
}

void blake3_hasher_update(blake3_hasher *self, __private void *input,
                          ulong input_len) {
  bool use_tbb = false;
  blake3_hasher_update_base(self, input, input_len, use_tbb);
}

void blake3_compress_xof_portable(uint cv[8], uchar block[BLAKE3_BLOCK_LEN],
                                  uchar block_len, ulong counter, uchar flags,
                                  uchar out[64]) {
  uint state[16];
  compress_pre(state, cv, block, block_len, counter, flags);
  for (int i = 0; i < 8; i++) {
    store32(&out[i * 4], state[i] ^ state[i + 8]);
    store32(&out[(i + 8) * 4], state[i + 8] ^ cv[i]);
  }
}

void blake3_compress_xof(uint cv[8], uchar block[BLAKE3_BLOCK_LEN],
                         uchar block_len, ulong counter, uchar flags,
                         uchar out[64]) {
  blake3_compress_xof_portable(cv, block, block_len, counter, flags, out);
}

void blake3_xof_many(uint cv[8], uchar block[BLAKE3_BLOCK_LEN], uchar block_len,
                     ulong counter, uchar flags, uchar out[64],
                     size_t outblocks) {
  if (outblocks == 0) {
    return;
  }
  for (size_t i = 0; i < outblocks; ++i) {
    uchar private_out[64];
    blake3_compress_xof(cv, block, block_len, counter + i, flags, private_out);
    for (int j = 0; j < 64; j++) {
      out[64 * i + j] = private_out[j];
    }
  }
}

void output_root_bytes(output_t *self, ulong seek, uchar *out, size_t out_len) {
  if (out_len == 0) {
    return;
  }
  ulong output_block_counter = seek >> 6;
  size_t offset_within_block = (size_t)(seek & 63ul);
  uchar wide_buf[64];
  if (offset_within_block) {
    blake3_compress_xof(self->input_cv, (uchar *)self->block, self->block_len,
                        output_block_counter, self->flags | ROOT, wide_buf);
    size_t available_bytes = 64 - offset_within_block;
    size_t bytes = out_len > available_bytes ? available_bytes : out_len;
    for (size_t i = 0; i < bytes; i++) {
      out[i] = wide_buf[offset_within_block + i];
    }
    out += bytes;
    out_len -= bytes;
    output_block_counter += 1;
  }
  if (out_len / 64) {
    blake3_xof_many(self->input_cv, (uchar *)self->block, self->block_len,
                    output_block_counter, self->flags | ROOT, out,
                    out_len / 64);
  }
  output_block_counter += out_len / 64;
  out += out_len & -64;
  out_len -= out_len & -64;
  if (out_len) {
    blake3_compress_xof(self->input_cv, (uchar *)self->block, self->block_len,
                        output_block_counter, self->flags | ROOT, wide_buf);
    for (size_t i = 0; i < out_len; i++) {
      out[i] = wide_buf[i];
    }
  }
}

void blake3_hasher_finalize_seek(blake3_hasher *self, ulong seek, uchar *out,
                                 size_t out_len) {
  if (out_len == 0) {
    return;
  }
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
    output = parent_output(&self->cv_stack[cvs_remaining * 32], self->key,
                           self->chunk.flags);
  }
  while (cvs_remaining > 0) {
    cvs_remaining -= 1;
    uchar parent_block[BLAKE3_BLOCK_LEN];
    for (int i = 0; i < 32; i++) {
      parent_block[i] = self->cv_stack[cvs_remaining * 32 + i];
    }
    output_chaining_value(&output, &parent_block[32]);
    output = parent_output(parent_block, self->key, self->chunk.flags);
  }
  output_root_bytes(&output, seek, out, out_len);
}

void blake3_hasher_finalize(blake3_hasher *self, uchar *out, size_t out_len) {
  blake3_hasher_finalize_seek(self, 0, out, out_len);
}

void ConvertBytesToUint32Array(uint *H, uchar *bytes) {
  for (int i = 0; i < 8; i++) {
    H[i] = ((uint)bytes[i * 4] << 24) | ((uint)bytes[i * 4 + 1] << 16) |
           ((uint)bytes[i * 4 + 2] << 8) | (uint)bytes[i * 4 + 3];
  }
}

double MediumComplexNonLinear(double x) {
  double sin_x, cos_x;
  sin_x = sincos(x, &cos_x); // OpenCL's sincos function
  return exp(sin_x + cos_x);
}

double IntermediateComplexNonLinear(double x) {
  if (x == PI / 2 || x == 3 * PI / 2) {
    return 0; // Avoid singularity
  }
  return sin(x) * sin(x);
}

double HighComplexNonLinear(double x) { return 1.0 / sqrt(fabs(x) + 1); }

double ComplexNonLinear(double x) {
  double transformFactorOne = fmod(x * COMPLEX_TRANSFORM_MULTIPLIER, 8) / 8;
  double transformFactorTwo = fmod(x * COMPLEX_TRANSFORM_MULTIPLIER, 4) / 4;
  if (transformFactorOne < 0.33) {
    if (transformFactorTwo < 0.25) {
      return MediumComplexNonLinear(x + (1 + transformFactorTwo));
    } else if (transformFactorTwo < 0.5) {
      return MediumComplexNonLinear(x - (1 + transformFactorTwo));
    } else if (transformFactorTwo < 0.75) {
      return MediumComplexNonLinear(x * (1 + transformFactorTwo));
    } else {
      return MediumComplexNonLinear(x / (1 + transformFactorTwo));
    }
  } else if (transformFactorOne < 0.66) {
    if (transformFactorTwo < 0.25) {
      return IntermediateComplexNonLinear(x + (1 + transformFactorTwo));
    } else if (transformFactorTwo < 0.5) {
      return IntermediateComplexNonLinear(x - (1 + transformFactorTwo));
    } else if (transformFactorTwo < 0.75) {
      return IntermediateComplexNonLinear(x * (1 + transformFactorTwo));
    } else {
      return IntermediateComplexNonLinear(x / (1 + transformFactorTwo));
    }
  } else {
    if (transformFactorTwo < 0.25) {
      return HighComplexNonLinear(x + (1 + transformFactorTwo));
    } else if (transformFactorTwo < 0.5) {
      return HighComplexNonLinear(x - (1 + transformFactorTwo));
    } else if (transformFactorTwo < 0.75) {
      return HighComplexNonLinear(x * (1 + transformFactorTwo));
    } else {
      return HighComplexNonLinear(x / (1 + transformFactorTwo));
    }
  }
}

double TransformFactor(double x) {
  const double granularity = 1024.0;
  return fmod(x, granularity) / granularity;
}

double ForComplex(double forComplex) {
  double complexValue;
  double rounds = 1.0;

  complexValue = ComplexNonLinear(forComplex);
  while (complexValue != complexValue) {
    forComplex *= 0.1;
    if (forComplex <= 0.1) {
      return 0.0 * rounds;
    }
    rounds += 1.0;
    complexValue = ComplexNonLinear(forComplex);
  }

  return complexValue * rounds;
}

ulong rotl(const ulong x, int k) { return (x << k) | (x >> (64 - k)); }

ulong xoshiro256_next(__global ulong4 *s) {
  // Unpack the ulong4 state
  ulong s0 = s->x;
  ulong s1 = s->y;
  ulong s2 = s->z;
  ulong s3 = s->w;

  const ulong result = rotl(s1 * 5, 7) * 9;

  const ulong t = s1 << 17;

  s2 ^= s0;
  s3 ^= s1;
  s1 ^= s2;
  s0 ^= s3;
  s2 ^= t;
  s3 = rotl(s3, 45);

  s->x = s0;
  s->y = s1;
  s->z = s2;
  s->w = s3;

  return result;
}

void print_hash(uchar *hash) {
  printf("%02x%02x%02x%02x%02x%02x%02x%02x"
         "%02x%02x%02x%02x%02x%02x%02x%02x"
         "%02x%02x%02x%02x%02x%02x%02x%02x"
         "%02x%02x%02x%02x%02x%02x%02x%02x\n",
         hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7],
         hash[8], hash[9], hash[10], hash[11], hash[12], hash[13], hash[14],
         hash[15], hash[16], hash[17], hash[18], hash[19], hash[20], hash[21],
         hash[22], hash[23], hash[24], hash[25], hash[26], hash[27], hash[28],
         hash[29], hash[30], hash[31]);
}

int compare_target(uchar *hash, __global uchar *target) {
  for (size_t i = 0; i < DOMAIN_HASH_SIZE; i++) {
    if (hash[i] > target[i])
      return 1;
    if (hash[i] < target[i])
      return -1;
  }
  return 0;
}

void HoohashMatrixMultiplication(__global double *mat, const uchar *hashBytes,
                                 uchar *output, ulong nonce) {
  uchar vector[64] = {0};
  double product[64] = {0};
  uchar scaledValues[32] = {0};
  __private uchar result[32] = {0};
  uint H[8] = {0};
  for (int i = 0; i < 8; i++) {
    H[i] = ((uint)hashBytes[i * 4] << 24) | ((uint)hashBytes[i * 4 + 1] << 16) |
           ((uint)hashBytes[i * 4 + 2] << 8) | (uint)hashBytes[i * 4 + 3];
  }
  double hashMod =
      (double)(H[0] ^ H[1] ^ H[2] ^ H[3] ^ H[4] ^ H[5] ^ H[6] ^ H[7]);
  double nonceMod = (nonce & 0xFF);
  double divider = 0.0001;
  double multiplier = 1234;
  for (int i = 0; i < 32; i++) {
    vector[2 * i] = hashBytes[i] >> 4;
    vector[2 * i + 1] = hashBytes[i] & 0x0F;
  }

  double sw = 0.0;
  for (int i = 0; i < 64; i++) {
    for (int j = 0; j < 64; j++) {
      if (sw <= 0.02) {
        double matrixScaledByHash = mat[i * 64 + j] * hashMod;
        double matrixScaledByHashAndVector =
            matrixScaledByHash * (double)vector[j];
        double input = matrixScaledByHashAndVector + nonceMod;
        double transformedInput = ForComplex(input);
        double scaledByVector = transformedInput * (double)vector[j];
        double output = scaledByVector * multiplier;
        product[i] += output;
      } else {
        double matrixElementScaledByDivider = mat[i * 64 + j] * divider;
        double output = matrixElementScaledByDivider * (double)vector[j];
        product[i] += output;
      }
      sw = TransformFactor(product[i]);
    }
  }
  for (int i = 0; i < 64; i += 2) {
    ulong pval = (ulong)product[i] + (ulong)product[i + 1];
    scaledValues[i / 2] = (uchar)(pval & 0xFF);
  }
  for (int i = 0; i < 32; i++) {
    result[i] = hashBytes[i] ^ scaledValues[i];
  }
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, result, DOMAIN_HASH_SIZE);
  blake3_hasher_finalize(&hasher, output, DOMAIN_HASH_SIZE);
}

typedef struct {
  ulong nonce;
  uchar hash[32];
} Result;

__kernel void Hoohash_hash(const ulong local_size, const ulong start_nonce,
                           __global uchar *previous_header,
                           __global long *timestamp, __global double *matrix,
                           __global uchar *target,
                           volatile __global Result *result) {
#if defined(PAL)
  int nonceId = get_group_id(0) * local_size + get_local_id(0);
#else
  int nonceId = get_global_id(0);
#endif

  ulong nonce = start_nonce + nonceId;
  // printf("Trying nonce %lu\n", nonce);

  // Compute BLAKE3 hash
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);

  __private uchar private_previous_header[DOMAIN_HASH_SIZE];
  for (int i = 0; i < DOMAIN_HASH_SIZE; i++) {
    private_previous_header[i] = previous_header[i];
  }
  blake3_hasher_update(&hasher, private_previous_header, DOMAIN_HASH_SIZE);
  __private long private_timestamp[8];
  for (int i = 0; i < 8; i++) {
    private_timestamp[i] = timestamp[i];
  }
  blake3_hasher_update(&hasher, private_timestamp, 8);
  uchar zeroes[DOMAIN_HASH_SIZE] = {0};
  blake3_hasher_update(&hasher, zeroes, DOMAIN_HASH_SIZE);
  blake3_hasher_update(&hasher, &nonce, sizeof(nonce));
  uchar first_pass[DOMAIN_HASH_SIZE];
  blake3_hasher_finalize(&hasher, first_pass, DOMAIN_HASH_SIZE);

  // Matrix multiplication
  uchar final_hash[DOMAIN_HASH_SIZE];
  uchar vector[64] = {0};
  double product[64] = {0};
  HoohashMatrixMultiplication(matrix, first_pass, final_hash, nonce);
  // printf("Work item %d incremented nonces_processed\n", nonceId);
  uchar reversed_hash[DOMAIN_HASH_SIZE];
  for (size_t i = 0; i < DOMAIN_HASH_SIZE; i++) {
    reversed_hash[i] = final_hash[DOMAIN_HASH_SIZE - 1 - i];
  }
  if (compare_target(reversed_hash, target) <= 0) {
    if (atom_cmpxchg(&result->nonce, 0, nonce) == 0) {
      for (int i = 0; i < 32; i++) {
        result->hash[i] = final_hash[i];
      }
    }
  }
}