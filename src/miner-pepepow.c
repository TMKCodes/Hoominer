/*
 * PEPEPOW CPU mining support for Hoominer.
 *
 * PEPEPOW uses Bitcoin-style 80-byte block headers.  The proof-of-work
 * computation differs from standard Hoohash as follows:
 *
 *   1. firstPass  = BLAKE3(full 80-byte header)
 *   2. matrixSeed = BLAKE3(header with bytes [76..79] zeroed)
 *   3. matrix     = generateHoohashMatrix(matrixSeed)   -- constant per job
 *   4. nonce      = little-endian uint32 at header[76]
 *   5. output     = HoohashMatrixMultiplication(matrix, firstPass, nonce)
 *
 * The matrix is precomputed once per job (nonce-independent) and cached in
 * QueuedJob.matrix.  The mining thread only needs to update the four nonce
 * bytes in a local header copy and call hoohashv110_compute() for each
 * candidate nonce.
 */

#include "miner-pepepow.h"
#include <blake3.h>
#include <time.h>
#include <inttypes.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
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
 * Helper: write a little-endian uint32 into a byte buffer.
 * -------------------------------------------------------------------------*/
static inline void write_uint32_le(uint8_t *p, uint32_t v)
{
  p[0] = (uint8_t)(v);
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

/* ---------------------------------------------------------------------------
 * Helper: read a little-endian uint32 from a byte buffer.
 * -------------------------------------------------------------------------*/
static inline uint32_t read_uint32_le(const uint8_t *p)
{
  return (uint32_t)p[0]
       | ((uint32_t)p[1] << 8)
       | ((uint32_t)p[2] << 16)
       | ((uint32_t)p[3] << 24);
}

/* ---------------------------------------------------------------------------
 * hoohashv110_compute
 *
 * Compute the PEPEPOW PoW hash for a given 80-byte header that already
 * has the candidate nonce written at bytes [76..79] (LE uint32).
 *
 * mat    - the pre-computed matrix (generated from the nonce-zeroed header)
 * hdr    - 80-byte header with nonce at offset 76
 * output - 32-byte result
 * -------------------------------------------------------------------------*/
static void hoohashv110_compute(double mat[64][64],
                                const uint8_t hdr[PEPEPOW_HEADER_SIZE],
                                uint8_t output[DOMAIN_HASH_SIZE])
{
  blake3_hasher hasher;
  uint8_t firstPass[DOMAIN_HASH_SIZE];

  /* 1) Nonce-dependent first pass: BLAKE3(full 80-byte header) */
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, hdr, PEPEPOW_HEADER_SIZE);
  blake3_hasher_finalize(&hasher, firstPass, DOMAIN_HASH_SIZE);

  /* 2) Read the nonce from header[76..79] (LE uint32 → uint64 for matrix mul) */
  const uint64_t nonce = (uint64_t)read_uint32_le(hdr + PEPEPOW_NONCE_OFFSET);

  /* 3) Final PoW: matrix multiply using precomputed matrix */
  HoohashMatrixMultiplication(mat, firstPass, output, nonce);
}

/* ---------------------------------------------------------------------------
 * submit_pepepow_solution
 *
 * Send a PEPEPOW share to the pool using Bitcoin stratum v1 5-parameter format:
 *   ["worker", "job_id", "extranonce2", "ntime", "nonce"]
 * The nonce is encoded as an 8-character lower-case hex string representing
 * the uint32 value (big-endian display, e.g. nonce=0x9EFE8071 → "9efe8071").
 * -------------------------------------------------------------------------*/
int submit_pepepow_solution(int sockfd, const char *worker, const char *job_id,
                            uint32_t nonce, const char *ntime_hex,
                            const char *extranonce2_hex, MiningState *ms,
                            StratumContext *ctx, int reporting_index)
{
  if (!worker || !job_id || !ntime_hex || !extranonce2_hex)
  {
    fprintf(stderr, "submit_pepepow_solution: invalid parameters\n");
    return -1;
  }

  pthread_mutex_lock(&ms->job_queue.queue_mutex);
  for (int i = 0; i < JOB_QUEUE_SIZE; i++)
  {
    int idx = (ms->job_queue.head + i) % JOB_QUEUE_SIZE;
    if (ms->job_queue.jobs[idx].job_id &&
        strcmp(ms->job_queue.jobs[idx].job_id, job_id) == 0)
    {
      ms->job_queue.jobs[idx].completed = 1;
      ms->job_queue.jobs[idx].running   = 0;
      if (idx == ms->job_queue.head)
        ms->job_queue.head = (ms->job_queue.head + 1) % JOB_QUEUE_SIZE;
      break;
    }
  }
  pthread_cond_broadcast(&ms->job_queue.queue_cond);
  pthread_mutex_unlock(&ms->job_queue.queue_mutex);

  /* Format nonce as 8-char lower-case hex string of the uint32 value
   * (e.g. nonce=0x9EFE8071 → "9efe8071").  This is the numeric representation
   * of the nonce; the pool interprets it as a uint32 and writes it as
   * little-endian bytes at offset 76 of the reconstructed header. */
  char nonce_hex[9];
  snprintf(nonce_hex, sizeof(nonce_hex), "%08x", nonce);

  printf("PEPEPOW solution found, Nonce: %" PRIu32 " (0x%s)\n",
         nonce, nonce_hex);

  /* Bitcoin stratum v1 submit: 5 params, no PoW hash in the message.
   * Format: {"method":"mining.submit","params":["worker","job_id","extranonce2","ntime","nonce"],"id":N} */
  char buf[4096];
  int written = snprintf(buf, sizeof(buf),
    "{\"method\":\"mining.submit\","
    "\"params\":[\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"],"
    "\"id\":4}\n",
    worker, job_id, extranonce2_hex, ntime_hex, nonce_hex);
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
 * CPU mining thread for PEPEPOW.  Each thread iterates uint32 nonces
 * starting at its thread index and stepping by the total number of CPU
 * threads, wrapping around when the uint32 space is exhausted.
 * -------------------------------------------------------------------------*/
void *mining_cpu_thread_pepepow(void *arg)
{
  MiningThread *mt  = (MiningThread *)arg;
  StratumContext *ctx = mt->ctx;
  const int thread_index = mt->threadIndex;
  free(mt);

  MiningState *ms   = ctx->ms;
  char *current_job_id = NULL;

  int reporting_index = 0;
  ReportingDevice *cpu_reporting_device = ctx->hd->devices[reporting_index];

  /* Each thread uses a different nonce starting point, stepping by thread
   * count so threads never duplicate work.  We wrap modulo UINT32_MAX+1. */
  uint32_t nonce = (uint32_t)thread_index;
  const uint32_t step = (ms->num_cpu_threads > 0) ? (uint32_t)ms->num_cpu_threads : 1;

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

    /* Snapshot the job's precomputed matrix, header template, and submit fields. */
    double mat[64][64];
    uint8_t hdr_template[PEPEPOW_HEADER_SIZE];
    char ntime_hex[16];
    char extranonce2_hex[32];
    memcpy(mat, current_job.matrix, sizeof(double) * 64 * 64);
    memcpy(hdr_template, current_job.pepepow_header, PEPEPOW_HEADER_SIZE);
    strncpy(ntime_hex, current_job.ntime_hex, sizeof(ntime_hex) - 1);
    ntime_hex[sizeof(ntime_hex) - 1] = '\0';
    strncpy(extranonce2_hex, current_job.extranonce2_hex, sizeof(extranonce2_hex) - 1);
    extranonce2_hex[sizeof(extranonce2_hex) - 1] = '\0';

    struct timespec start_time, end_time;
    if (ctx->config->debug == 1)
    {
      clock_gettime(CLOCK_MONOTONIC, &start_time);
      printf("PEPEPOW: starting job %s\n", current_job_id);
    }

    int nonces_processed = 0;

    while (ctx->running)
    {
      /* Check that the job is still the latest one. */
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

      /* Write the candidate nonce into a local header copy and hash it. */
      uint8_t hdr[PEPEPOW_HEADER_SIZE];
      memcpy(hdr, hdr_template, PEPEPOW_HEADER_SIZE);
      write_uint32_le(hdr + PEPEPOW_NONCE_OFFSET, nonce);

      uint8_t result[DOMAIN_HASH_SIZE];
      hoohashv110_compute(mat, hdr, result);

      /* PEPEPOW uses big-endian hash comparison (Bitcoin convention): the
       * most-significant byte is at index 0.  The CPU compare_target()
       * function treats the hash as little-endian (MSB at index 31), so we
       * must reverse the hash bytes before comparing -- identical to what
       * the GPU kernels do explicitly (see hoohash.cu / Hoohash.cl). */
      uint8_t reversed_hash[DOMAIN_HASH_SIZE];
      for (int i = 0; i < DOMAIN_HASH_SIZE; i++)
        reversed_hash[i] = result[DOMAIN_HASH_SIZE - 1 - i];

      pthread_mutex_lock(&ctx->hd->hashrate_mutex);
      cpu_reporting_device->nonces_processed++;
      nonces_processed++;
      pthread_mutex_unlock(&ctx->hd->hashrate_mutex);

      pthread_mutex_lock(&ms->target_mutex);
      int meets_target = compare_target(reversed_hash, ms->global_target, DOMAIN_HASH_SIZE);
      pthread_mutex_unlock(&ms->target_mutex);

      if (meets_target <= 0)
      {
        if (ctx->config->debug == 1)
          printf("PEPEPOW: submitting solution for job %s\n", current_job_id);
        /* submit_pepepow_solution marks the job completed in the queue. */
        submit_pepepow_solution(ctx->sockfd, ctx->worker, current_job_id,
                                nonce, ntime_hex, extranonce2_hex,
                                ms, ctx, reporting_index);
        break;
      }

      nonce += step;
    }
    /* Inner loop exited: either job was superseded or a solution was found. */

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
