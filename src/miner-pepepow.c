/*
 * PEPEPOW CPU and GPU mining support for Hoominer.
 *
 * PEPEPOW uses Bitcoin-style 80-byte block headers with standard stratum v1.
 * The pool sends: prevhash, coinbase1, coinbase2, merkle_branches, version,
 * nbits, ntime.  Hoominer builds the 80-byte "endiandata" header from those
 * fields, exactly as cpuminer does, then hashes it with the Hoohash PoW.
 *
 * Header construction (endiandata):
 *   [0..3]   = BE-encoded version   (each 4-byte field byte-reversed)
 *   [4..35]  = prevhash with each 4-byte word byte-reversed
 *   [36..67] = sha256d(coinbase) merkle root, iterated with branches (no swap)
 *   [68..71] = BE-encoded ntime
 *   [72..75] = BE-encoded nbits
 *   [76..79] = BE-encoded nonce
 *
 * PoW hash computation (matches reference miner hoohash_pepew_hash):
 *   1. firstPass  = BLAKE3(full 80-byte endiandata with nonce at [76..79])
 *   2. matrixSeed = BLAKE3(endiandata with nonce zeroed)
 *   3. matrix     = generateHoohashMatrix(matrixSeed)  -- constant per job
 *   4. nonce_val  = read_uint32_le(endiandata + 76)    -- BE bytes → LE read
 *   5. outhash    = HoohashMatrixMultiplication(matrix, firstPass, nonce_val)
 *   6. output     = reverse(outhash)                   -- reference miner reverses
 *
 * Submission (standard Bitcoin stratum v1):
 *   mining.submit [worker, job_id, extranonce2_hex, ntime_hex, nonce_LE_hex]
 *   where nonce_LE_hex = little-endian bytes of the nonce value as hex
 */

#include "miner-pepepow.h"
#include "opencl-host.h"
#include "hoohash_cl.h"
#include <blake3.h>
#include <openssl/sha.h>
#include <time.h>
#include <inttypes.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef _WIN32
#include <malloc.h>
#endif
#include "platform_compat.h"

#ifdef _WIN32
#include <windows.h>
static void clock_gettime_monotonic_pp(struct timespec *ts)
{
  static LARGE_INTEGER freq = {0};
  LARGE_INTEGER now;
  if (!freq.QuadPart)
    QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&now);
  long double seconds = (long double)now.QuadPart / (long double)freq.QuadPart;
  ts->tv_sec  = (time_t)seconds;
  ts->tv_nsec = (long)((seconds - ts->tv_sec) * 1e9);
}
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
#define clock_gettime(id, ts) clock_gettime_monotonic_pp(ts)
#endif /* _WIN32 */

/* ---------------------------------------------------------------------------
 * sha256d: double-SHA256 (SHA256(SHA256(data)))
 * -------------------------------------------------------------------------*/
static void sha256d(uint8_t *out, const uint8_t *data, size_t len)
{
  uint8_t tmp[32];
  SHA256(data, len, tmp);
  SHA256(tmp, 32, out);
}

/* ---------------------------------------------------------------------------
 * hex_decode: decode a hex string into bytes.
 * Returns 0 on success, -1 on error.
 * -------------------------------------------------------------------------*/
static int hex_decode(const char *hex, uint8_t *out, size_t out_len)
{
  size_t hex_len = strlen(hex);
  if (hex_len != out_len * 2)
    return -1;
  for (size_t i = 0; i < out_len; i++)
  {
    unsigned int b;
    if (sscanf(hex + i * 2, "%02x", &b) != 1)
      return -1;
    out[i] = (uint8_t)b;
  }
  return 0;
}

/* ---------------------------------------------------------------------------
 * be32enc / be32dec helpers (byte-swap a 4-byte field).
 * -------------------------------------------------------------------------*/
static inline void be32enc(uint8_t *p, uint32_t v)
{
  p[0] = (uint8_t)(v >> 24);
  p[1] = (uint8_t)(v >> 16);
  p[2] = (uint8_t)(v >> 8);
  p[3] = (uint8_t)(v);
}

static inline uint32_t le32dec(const uint8_t *p)
{
  return (uint32_t)p[0]
       | ((uint32_t)p[1] << 8)
       | ((uint32_t)p[2] << 16)
       | ((uint32_t)p[3] << 24);
}

static inline uint32_t be32dec_h(const uint8_t *p)
{
  return ((uint32_t)p[0] << 24)
       | ((uint32_t)p[1] << 16)
       | ((uint32_t)p[2] << 8)
       |  (uint32_t)p[3];
}

/* ---------------------------------------------------------------------------
 * pepepow_build_endiandata
 *
 * Build the 80-byte "endiandata" (BE-encoded block header) from the stratum
 * job fields.  This exactly matches cpuminer's stratum_gen_work + the
 * scanhash_hoohash_pepew pre-encoding step.
 *
 * version_hex, prevhash_hex, nbits_hex, ntime_hex: 8-char hex strings
 * merkle_root: 32 bytes from sha256d(coinbase) + branch processing
 * out: 80-byte output buffer (nonce field at [76..79] is zeroed)
 * -------------------------------------------------------------------------*/
static void pepepow_build_endiandata(
    const char *version_hex,
    const uint8_t *prevhash,        /* 32 raw bytes */
    const uint8_t *merkle_root,     /* 32 raw bytes */
    const char *ntime_hex,
    const char *nbits_hex,
    uint8_t out[PEPEPOW_HEADER_SIZE])
{
  memset(out, 0, PEPEPOW_HEADER_SIZE);

  /* Version: le32dec then be32enc = byte-reverse the 4 bytes */
  uint8_t ver_bytes[4];
  if (hex_decode(version_hex, ver_bytes, 4) == 0)
    be32enc(out + 0, le32dec(ver_bytes));

  /* Prevhash: each 4-byte word byte-reversed */
  for (int i = 0; i < 8; i++)
  {
    uint32_t w = le32dec(prevhash + i * 4);
    be32enc(out + 4 + i * 4, w);
  }

  /* Merkle root: be32dec → be32enc = identity (no byte swap) */
  memcpy(out + 36, merkle_root, 32);

  /* Ntime: byte-reverse */
  uint8_t nt_bytes[4];
  if (hex_decode(ntime_hex, nt_bytes, 4) == 0)
    be32enc(out + 68, le32dec(nt_bytes));

  /* Nbits: byte-reverse */
  uint8_t nb_bytes[4];
  if (hex_decode(nbits_hex, nb_bytes, 4) == 0)
    be32enc(out + 72, le32dec(nb_bytes));

  /* Nonce at [76..79] left as 0 */
}

/* ---------------------------------------------------------------------------
 * hoohashv110_compute
 *
 * Compute the PEPEPOW PoW hash for a given 80-byte endiandata header that
 * already has the candidate nonce written at bytes [76..79] as big-endian
 * (matching reference miner's be32enc(&endiandata[19], nonce)).
 *
 * mat    - pre-computed matrix (generated from nonce-zeroed endiandata)
 * hdr    - 80-byte BE-encoded header with nonce at offset 76
 * output - 32-byte result (reversed, ready for compare_target)
 * -------------------------------------------------------------------------*/
static void hoohashv110_compute(double mat[64][64],
                                const uint8_t hdr[PEPEPOW_HEADER_SIZE],
                                uint8_t output[DOMAIN_HASH_SIZE])
{
  blake3_hasher hasher;
  uint8_t firstPass[DOMAIN_HASH_SIZE];

  /* 1) BLAKE3(full 80-byte endiandata including nonce) */
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, hdr, PEPEPOW_HEADER_SIZE);
  blake3_hasher_finalize(&hasher, firstPass, DOMAIN_HASH_SIZE);

  /*
   * 2) Read the nonce: the reference miner stores nonce as BE at [76..79]
   * then reads it with read_uint32_le, giving the byte-swapped value.
   * We do the same: read_uint32_le of a BE-stored nonce value N = bswap(N).
   */
  const uint64_t nonce_val = (uint64_t)le32dec(hdr + PEPEPOW_NONCE_OFFSET);

  /* 3) Matrix multiply */
  uint8_t outhash[DOMAIN_HASH_SIZE];
  HoohashMatrixMultiplication(mat, firstPass, outhash, nonce_val);

  /* 4) Reverse output (matches reference miner's final reversal step) */
  for (int i = 0; i < DOMAIN_HASH_SIZE; i++)
    output[i] = outhash[DOMAIN_HASH_SIZE - 1 - i];
}

/* ---------------------------------------------------------------------------
 * submit_pepepow_solution
 *
 * Send a PEPEPOW share using standard Bitcoin stratum v1 format:
 *   mining.submit [worker, job_id, extranonce2_hex, ntime_hex, nonce_LE_hex]
 *
 * nonce is submitted as LE-bytes-as-hex (e.g. nonce=0xf803c995 → "95c903f8")
 * -------------------------------------------------------------------------*/
int submit_pepepow_solution(int sockfd, const char *worker, const char *job_id,
                            uint32_t nonce,
                            const uint8_t *extranonce2, int extranonce2_len,
                            const uint8_t *ntime,
                            MiningState *ms, StratumContext *ctx,
                            int reporting_index)
{
  if (!worker || !job_id || !extranonce2 || !ntime)
  {
    fprintf(stderr, "submit_pepepow_solution: invalid parameters\n");
    return -1;
  }

  /* Nonce as LE-bytes-as-hex: nonce value N → bytes [N&0xFF, N>>8, N>>16, N>>24] */
  char nonce_hex[9];
  snprintf(nonce_hex, sizeof(nonce_hex), "%02x%02x%02x%02x",
           nonce & 0xFF, (nonce >> 8) & 0xFF,
           (nonce >> 16) & 0xFF, (nonce >> 24) & 0xFF);

  /* extranonce2 as hex */
  char en2_hex[33] = {0};
  for (int i = 0; i < extranonce2_len && i < 16; i++)
    snprintf(en2_hex + i * 2, 3, "%02x", extranonce2[i]);

  /* ntime as hex (raw bytes, same as received from pool) */
  char ntime_hex[9];
  snprintf(ntime_hex, sizeof(ntime_hex), "%02x%02x%02x%02x",
           ntime[0], ntime[1], ntime[2], ntime[3]);

  printf("PEPEPOW solution found: job=%s nonce=%s ntime=%s en2=%s\n",
         job_id, nonce_hex, ntime_hex, en2_hex);

  char buf[512];
  int written = snprintf(buf, sizeof(buf),
    "{\"method\": \"mining.submit\", \"params\": [\"%s\", \"%s\", \"%s\", \"%s\", \"%s\"], \"id\":4}\n",
    worker, job_id, en2_hex, ntime_hex, nonce_hex);
  if (written < 0 || (size_t)written >= sizeof(buf))
  {
    fprintf(stderr, "submit_pepepow_solution: JSON buffer too small\n");
    return -1;
  }

  int ret = send(sockfd, buf, (size_t)written, 0);
  enqueue_int_fifo(&ctx->mining_submit_fifo, reporting_index);
  return ret < 0 ? -1 : 0;
}

/* ---------------------------------------------------------------------------
 * mining_cpu_thread_pepepow
 *
 * CPU mining thread for PEPEPOW.  Iterates nonces starting at thread_index,
 * stepping by num_cpu_threads.
 * -------------------------------------------------------------------------*/
void *mining_cpu_thread_pepepow(void *arg)
{
  MiningThread *mt    = (MiningThread *)arg;
  StratumContext *ctx = mt->ctx;
  const int thread_index = mt->threadIndex;
  free(mt);

  MiningState *ms   = ctx->ms;
  char *current_job_id = NULL;

  int reporting_index = 0;
  ReportingDevice *cpu_reporting_device = ctx->hd->devices[reporting_index];

  uint32_t nonce = (uint32_t)thread_index;
  const uint32_t step = (uint32_t)ms->num_cpu_threads;

  while (ctx->running)
  {
    QueuedJob current_job = {0};

    pthread_mutex_lock(&ms->job_queue.queue_mutex);
    int job_available = (ms->job_queue.head != ms->job_queue.tail);
    if (job_available)
    {
      int tail_idx = (ms->job_queue.tail - 1 + JOB_QUEUE_SIZE) % JOB_QUEUE_SIZE;
      current_job = ms->job_queue.jobs[tail_idx];
      if (current_job_id == NULL ||
          strcmp(current_job_id, current_job.job_id) != 0)
      {
        free(current_job_id);
        current_job_id = strdup(current_job.job_id);
        if (!current_job_id)
        {
          pthread_mutex_unlock(&ms->job_queue.queue_mutex);
          fprintf(stderr, "PEPEPOW: memory allocation failed for job_id\n");
          sleep_ms(100);
          continue;
        }
      }
      job_available = current_job.running && !current_job.completed;
    }
    pthread_mutex_unlock(&ms->job_queue.queue_mutex);

    if (!job_available)
    {
      sleep_ms(100);
      continue;
    }

    /* Snapshot the job's matrix and endiandata template (nonce zeroed). */
    double mat[64][64];
    uint8_t hdr_template[PEPEPOW_HEADER_SIZE];
    uint8_t job_ntime[4];
    uint8_t job_extranonce2[16];
    int job_extranonce2_len;

    memcpy(mat, current_job.matrix, sizeof(double) * 64 * 64);
    memcpy(hdr_template, current_job.pepepow_header, PEPEPOW_HEADER_SIZE);
    memcpy(job_ntime, current_job.ntime, 4);
    job_extranonce2_len = current_job.extranonce2_len;
    if (job_extranonce2_len > (int)sizeof(job_extranonce2))
      job_extranonce2_len = (int)sizeof(job_extranonce2);
    memcpy(job_extranonce2, current_job.extranonce2, job_extranonce2_len);

    struct timespec start_time, end_time;
    if (ctx->config->debug == 1)
    {
      clock_gettime(CLOCK_MONOTONIC, &start_time);
      printf("PEPEPOW: starting job %s\n", current_job_id);
    }

    int nonces_processed = 0;

    while (ctx->running)
    {
      /* Check that this is still the latest job. */
      pthread_mutex_lock(&ms->job_queue.queue_mutex);
      int job_valid = 0;
      int tail_idx  = (ms->job_queue.tail - 1 + JOB_QUEUE_SIZE) % JOB_QUEUE_SIZE;
      if (ms->job_queue.head != ms->job_queue.tail &&
          ms->job_queue.jobs[tail_idx].job_id &&
          strcmp(ms->job_queue.jobs[tail_idx].job_id, current_job_id) == 0 &&
          ms->job_queue.jobs[tail_idx].running &&
          !ms->job_queue.jobs[tail_idx].completed)
      {
        job_valid = 1;
      }
      pthread_mutex_unlock(&ms->job_queue.queue_mutex);

      if (!job_valid)
        break;

      /*
       * Write nonce as big-endian at offset 76, matching reference miner's
       * be32enc(&endiandata[19], nonce).
       */
      uint8_t hdr[PEPEPOW_HEADER_SIZE];
      memcpy(hdr, hdr_template, PEPEPOW_HEADER_SIZE);
      be32enc(hdr + PEPEPOW_NONCE_OFFSET, nonce);

      uint8_t result[DOMAIN_HASH_SIZE];
      hoohashv110_compute(mat, hdr, result);

      pthread_mutex_lock(&ctx->hd->hashrate_mutex);
      cpu_reporting_device->nonces_processed++;
      nonces_processed++;
      pthread_mutex_unlock(&ctx->hd->hashrate_mutex);

      pthread_mutex_lock(&ms->target_mutex);
      int meets_target = compare_target(result, ms->global_target, DOMAIN_HASH_SIZE);
      pthread_mutex_unlock(&ms->target_mutex);

      if (meets_target <= 0)
      {
        if (ctx->config->debug == 1)
          printf("PEPEPOW: submitting solution for job %s nonce=0x%08x\n",
                 current_job_id, nonce);
        submit_pepepow_solution(ctx->sockfd, ctx->worker, current_job_id,
                                nonce, job_extranonce2, job_extranonce2_len,
                                job_ntime, ms, ctx, reporting_index);
        /* Do not break — continue mining the same job (submit multiple shares
         * like a standard Bitcoin stratum miner does). */
      }

      nonce += step;
    }

    if (ctx->config->debug == 1)
    {
      clock_gettime(CLOCK_MONOTONIC, &end_time);
      long elapsed_ns = (end_time.tv_sec  - start_time.tv_sec)  * 1000000000L
                      + (end_time.tv_nsec - start_time.tv_nsec);
      double elapsed_ms = elapsed_ns / 1e6;
      printf("PEPEPOW: job %s: %.3f ms, %d nonces\n",
             current_job_id, elapsed_ms, nonces_processed);
    }
  }

  free(current_job_id);
  return NULL;
}

/* ---------------------------------------------------------------------------
 * pepepow_build_job
 *
 * Parse standard Bitcoin stratum mining.notify parameters and populate a
 * QueuedJob for PEPEPOW mining.
 *
 * Params (in order from mining.notify):
 *   job_id_str, prevhash_hex (64), coinb1_hex, coinb2_hex,
 *   merkle_branches (JSON array), version_hex (8), nbits_hex (8),
 *   ntime_hex (8)
 *
 * extranonce1_hex: from subscribe response (e.g. "700016c5")
 * extranonce2:     current extranonce2 bytes
 * extranonce2_len: byte length of extranonce2
 *
 * Returns 0 on success, -1 on error.
 * -------------------------------------------------------------------------*/
int pepepow_build_job(QueuedJob *job,
                      const char *job_id_str,
                      const char *prevhash_hex,
                      const char *coinb1_hex,
                      const char *coinb2_hex,
                      json_object *merkle_arr,
                      const char *version_hex,
                      const char *nbits_hex,
                      const char *ntime_hex,
                      const char *extranonce1_hex,
                      const uint8_t *extranonce2,
                      int extranonce2_len)
{
  /* Validate required string lengths */
  if (!prevhash_hex || strlen(prevhash_hex) != 64 ||
      !version_hex  || strlen(version_hex)  != 8  ||
      !nbits_hex    || strlen(nbits_hex)    != 8  ||
      !ntime_hex    || strlen(ntime_hex)    != 8)
    return -1;

  /* Decode prevhash */
  uint8_t prevhash[32];
  if (hex_decode(prevhash_hex, prevhash, 32) != 0)
    return -1;

  /* Build coinbase: coinb1 + extranonce1 + extranonce2 + coinb2 */
  size_t coinb1_len = strlen(coinb1_hex) / 2;
  size_t coinb2_len = strlen(coinb2_hex) / 2;
  size_t en1_len    = extranonce1_hex ? strlen(extranonce1_hex) / 2 : 0;

  /* Guard against unreasonably large inputs to prevent overflow */
  if (coinb1_len > 65536 || coinb2_len > 65536 || en1_len > 256 ||
      (size_t)extranonce2_len > 256)
    return -1;

  size_t coinbase_len = coinb1_len + en1_len + (size_t)extranonce2_len + coinb2_len;
  uint8_t *coinbase = malloc(coinbase_len);
  if (!coinbase)
    return -1;

  uint8_t *p = coinbase;

  /* coinb1 */
  for (size_t i = 0; i < coinb1_len; i++, p++)
  {
    unsigned int b;
    if (sscanf(coinb1_hex + i * 2, "%02x", &b) != 1) { free(coinbase); return -1; }
    *p = (uint8_t)b;
  }

  /* extranonce1 */
  for (size_t i = 0; i < en1_len; i++, p++)
  {
    unsigned int b;
    if (sscanf(extranonce1_hex + i * 2, "%02x", &b) != 1) { free(coinbase); return -1; }
    *p = (uint8_t)b;
  }

  /* extranonce2 */
  memcpy(p, extranonce2, extranonce2_len);
  p += extranonce2_len;

  /* coinb2 */
  for (size_t i = 0; i < coinb2_len; i++, p++)
  {
    unsigned int b;
    if (sscanf(coinb2_hex + i * 2, "%02x", &b) != 1) { free(coinbase); return -1; }
    *p = (uint8_t)b;
  }

  /* Compute merkle root: sha256d(coinbase), then iterate branches */
  uint8_t merkle_root[32];
  sha256d(merkle_root, coinbase, coinbase_len);
  free(coinbase);

  size_t branch_count = merkle_arr ? json_object_array_length(merkle_arr) : 0;
  for (size_t i = 0; i < branch_count; i++)
  {
    json_object *branch_obj = json_object_array_get_idx(merkle_arr, i);
    const char *branch_hex  = json_object_get_string(branch_obj);
    if (!branch_hex || strlen(branch_hex) != 64)
      return -1;
    uint8_t branch[32];
    if (hex_decode(branch_hex, branch, 32) != 0)
      return -1;

    uint8_t concat[64];
    memcpy(concat, merkle_root, 32);
    memcpy(concat + 32, branch, 32);
    sha256d(merkle_root, concat, 64);
  }

  /* Build the 80-byte BE-encoded endiandata (nonce zeroed) */
  pepepow_build_endiandata(version_hex, prevhash, merkle_root,
                           ntime_hex, nbits_hex, job->pepepow_header);

  /* Compute matrix seed: BLAKE3(endiandata with nonce zeroed) */
  /* nonce is already 0 in pepepow_header */
  blake3_hasher hasher;
  uint8_t matrix_seed[DOMAIN_HASH_SIZE];
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, job->pepepow_header, PEPEPOW_HEADER_SIZE);
  blake3_hasher_finalize(&hasher, matrix_seed, DOMAIN_HASH_SIZE);

  generateHoohashMatrix(matrix_seed, job->matrix);

  /* Store ntime bytes for submission */
  hex_decode(ntime_hex, job->ntime, 4);

  /* Store extranonce2 for submission */
  if (extranonce2_len > (int)sizeof(job->extranonce2))
    extranonce2_len = (int)sizeof(job->extranonce2);
  memcpy(job->extranonce2, extranonce2, extranonce2_len);
  job->extranonce2_len = extranonce2_len;

  /* Job metadata */
  free(job->job_id);
  job->job_id = strdup(job_id_str);
  job->timestamp = (long long)time(NULL) * 1000;
  job->running   = 1;
  job->completed = 0;

  return job->job_id ? 0 : -1;
}

/* ---------------------------------------------------------------------------
 * mining_opencl_thread_pepepow
 *
 * OpenCL GPU mining thread for PEPEPOW.  Each GPU device iterates its own
 * partition of the 32-bit nonce space, submitting via submit_pepepow_solution
 * when the GPU kernel finds a hash that meets the target.
 * -------------------------------------------------------------------------*/
void *mining_opencl_thread_pepepow(void *arg)
{
  MiningThread *mt    = (MiningThread *)arg;
  StratumContext *ctx = mt->ctx;
  const int thread_index = mt->threadIndex;
  free(mt);

  MiningState *ms = ctx->ms;
  char *current_job_id = NULL;
  unsigned int trim_counter = 0;

  cl_ulong local_work_size  = ctx->opencl_resources[thread_index].max_work_group_size;
  cl_ulong global_work_size = ctx->opencl_resources[thread_index].max_global_work_size;
  time_t last_opencl_reset  = time(NULL);

  /* Partition the 32-bit nonce space evenly across GPU devices. */
  int num_devices = ms->num_opencl_threads > 0 ? ms->num_opencl_threads : 1;
  uint32_t nonce_stride = (uint32_t)(0x100000000ULL / (uint64_t)num_devices);
  uint32_t start_nonce32 = (uint32_t)thread_index * nonce_stride;

  int reporting_index = ctx->cpu_device_count + thread_index;
  if (reporting_index >= ctx->hd->device_count)
  {
    fprintf(stderr, "PEPEPOW OpenCL reporting index %d exceeds device count %d\n",
            reporting_index, ctx->hd->device_count);
    return NULL;
  }
  ReportingDevice *opencl_reporting_device = ctx->hd->devices[reporting_index];

  while (ctx->running)
  {
    /* Fetch the latest job from the queue. */
    QueuedJob current_job = {0};
    pthread_mutex_lock(&ms->job_queue.queue_mutex);
    int job_available = (ms->job_queue.head != ms->job_queue.tail);
    if (job_available)
    {
      int tail_idx = (ms->job_queue.tail - 1 + JOB_QUEUE_SIZE) % JOB_QUEUE_SIZE;
      current_job = ms->job_queue.jobs[tail_idx];
      if (current_job_id == NULL ||
          strcmp(current_job_id, current_job.job_id) != 0)
      {
        free(current_job_id);
        current_job_id = strdup(current_job.job_id);
        if (!current_job_id)
        {
          pthread_mutex_unlock(&ms->job_queue.queue_mutex);
          fprintf(stderr, "PEPEPOW OpenCL: memory allocation failed for job_id\n");
          sleep_ms(100);
          continue;
        }
      }
      job_available = current_job.running && !current_job.completed;
    }
    pthread_mutex_unlock(&ms->job_queue.queue_mutex);

    if (!job_available)
    {
      sleep_ms(100);
      continue;
    }

    /* Snapshot job data. */
    double mat[64][64];
    uint8_t hdr_template[PEPEPOW_HEADER_SIZE];
    uint8_t job_ntime[4];
    uint8_t job_extranonce2[16];
    int job_extranonce2_len;

    memcpy(mat, current_job.matrix, sizeof(double) * 64 * 64);
    memcpy(hdr_template, current_job.pepepow_header, PEPEPOW_HEADER_SIZE);
    memcpy(job_ntime, current_job.ntime, 4);
    job_extranonce2_len = current_job.extranonce2_len;
    if (job_extranonce2_len > (int)sizeof(job_extranonce2))
      job_extranonce2_len = (int)sizeof(job_extranonce2);
    memcpy(job_extranonce2, current_job.extranonce2, job_extranonce2_len);

    while (ctx->running)
    {
      /* Check that this is still the latest job. */
      pthread_mutex_lock(&ms->job_queue.queue_mutex);
      int job_valid = 0;
      int tail_idx  = (ms->job_queue.tail - 1 + JOB_QUEUE_SIZE) % JOB_QUEUE_SIZE;
      if (ms->job_queue.head != ms->job_queue.tail &&
          ms->job_queue.jobs[tail_idx].job_id &&
          strcmp(ms->job_queue.jobs[tail_idx].job_id, current_job_id) == 0 &&
          ms->job_queue.jobs[tail_idx].running &&
          !ms->job_queue.jobs[tail_idx].completed)
      {
        job_valid = 1;
      }
      pthread_mutex_unlock(&ms->job_queue.queue_mutex);

      if (!job_valid)
        break;

      if (ctx->config->opencl_reset_interval > 0 &&
          time(NULL) - last_opencl_reset >= ctx->config->opencl_reset_interval)
      {
        const char *required_extensions[] = {"cl_khr_fp64"};
        printf("Resetting OpenCL device %d (interval %d s)\n",
               thread_index, ctx->config->opencl_reset_interval);
        cl_int reset_err = opencl_reinit_device(ctx, &ctx->opencl_resources[thread_index],
                                                Hoohash_cl, Hoohash_cl_len, "Pepepow_hash",
                                                required_extensions, 1);
        if (reset_err != CL_SUCCESS)
        {
          fprintf(stderr, "PEPEPOW OpenCL reset failed for device %d: %d\n",
                  thread_index, reset_err);
          sleep_ms(1000);
        }
        local_work_size  = ctx->opencl_resources[thread_index].max_work_group_size;
        global_work_size = ctx->opencl_resources[thread_index].max_global_work_size;
        last_opencl_reset = time(NULL);
      }

      pthread_mutex_lock(&ms->target_mutex);
      uint8_t local_target[DOMAIN_HASH_SIZE];
      memcpy(local_target, ms->global_target, DOMAIN_HASH_SIZE);
      pthread_mutex_unlock(&ms->target_mutex);

      OpenCLResult result = {0};
      cl_int status = run_opencl_pepepow_kernel(&ctx->opencl_resources[thread_index],
                                                global_work_size, local_work_size,
                                                hdr_template, local_target, mat,
                                                (cl_ulong)start_nonce32, &result);

#ifndef _WIN32
      if (++trim_counter >= 1000)
      {
        malloc_trim(0);
        trim_counter = 0;
      }
#endif

      pthread_mutex_lock(&ctx->hd->hashrate_mutex);
      opencl_reporting_device->nonces_processed += global_work_size;
      pthread_mutex_unlock(&ctx->hd->hashrate_mutex);

      if (status != CL_SUCCESS)
      {
        if (status == -54)
        {
          fprintf(stderr, "PEPEPOW device %d: CL_INVALID_WORK_GROUP_SIZE; halving local work size\n",
                  thread_index);
          local_work_size /= 2;
        }
        else
        {
          fprintf(stderr, "PEPEPOW device %d: kernel execution failed: %d\n",
                  thread_index, status);
        }
        start_nonce32 += (uint32_t)global_work_size;
        break;
      }

      if (result.nonce != 0)
      {
        pthread_mutex_lock(&ms->target_mutex);
        int meets_target = compare_target(result.hash, ms->global_target, DOMAIN_HASH_SIZE);
        pthread_mutex_unlock(&ms->target_mutex);

        if (meets_target <= 0)
        {
          uint32_t winning_nonce = (uint32_t)result.nonce;
          if (ctx->config->debug == 1)
            printf("PEPEPOW GPU solution found: job=%s nonce=0x%08x\n",
                   current_job_id, winning_nonce);
          submit_pepepow_solution(ctx->sockfd, ctx->worker, current_job_id,
                                  winning_nonce, job_extranonce2, job_extranonce2_len,
                                  job_ntime, ms, ctx, reporting_index);
          /* Do not break — continue mining the same job (Bitcoin stratum allows
           * multiple shares per job; keep hashing until the pool sends a new job). */
        }
      }

      start_nonce32 += (uint32_t)global_work_size;
    }
    current_job.running = 0;
  }

  free(current_job_id);
  return NULL;
}
