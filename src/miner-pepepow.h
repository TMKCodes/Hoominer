#ifndef MINER_PEPEPOW_H
#define MINER_PEPEPOW_H

#include "miner-hoohash.h"

#define PEPEPOW_HEADER_SIZE 80
#define PEPEPOW_NONCE_OFFSET 76

/*
 * Build a QueuedJob from standard Bitcoin stratum mining.notify fields.
 * Returns 0 on success, -1 on error.
 */
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
                      int extranonce2_len);

/*
 * Submit a PEPEPOW mining solution using standard Bitcoin stratum v1 format.
 * nonce           - the uint32 nonce value that solved the block
 * extranonce2     - extranonce2 bytes used when building this job
 * extranonce2_len - length of extranonce2 in bytes
 * ntime           - 4-byte ntime field from the job
 */
int submit_pepepow_solution(int sockfd, const char *worker, const char *job_id,
                            uint32_t nonce,
                            const uint8_t *extranonce2, int extranonce2_len,
                            const uint8_t *ntime,
                            MiningState *ms, StratumContext *ctx,
                            int reporting_index);

/*
 * CPU mining thread for PEPEPOW.
 * arg must be a heap-allocated MiningThread *.
 */
void *mining_cpu_thread_pepepow(void *arg);

#endif /* MINER_PEPEPOW_H */
