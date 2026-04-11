#ifndef MINER_PEPEPOW_H
#define MINER_PEPEPOW_H

#include "miner-hoohash.h"

#define PEPEPOW_HEADER_SIZE 80
#define PEPEPOW_NONCE_OFFSET 76

/*
 * Submit a PEPEPOW mining solution using Bitcoin stratum v1 5-parameter format:
 *   ["worker", "job_id", "extranonce2", "ntime", "nonce"]
 * nonce          - the uint32 nonce found at header offset 76
 * ntime_hex      - ntime as 8-char hex string from the job (e.g. "69da10c4")
 * extranonce2_hex - extranonce2 as hex string for this job
 */
int submit_pepepow_solution(int sockfd, const char *worker, const char *job_id,
                            uint32_t nonce, const char *ntime_hex,
                            const char *extranonce2_hex, MiningState *ms,
                            StratumContext *ctx, int reporting_index);

/*
 * CPU mining thread for PEPEPOW.
 * arg must be a heap-allocated MiningThread *.
 */
void *mining_cpu_thread_pepepow(void *arg);

#endif /* MINER_PEPEPOW_H */
