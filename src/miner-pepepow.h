#ifndef MINER_PEPEPOW_H
#define MINER_PEPEPOW_H

#include "miner-hoohash.h"

#define PEPEPOW_HEADER_SIZE 80
#define PEPEPOW_NONCE_OFFSET 76

/*
 * Submit a PEPEPOW mining solution.
 * nonce    - the uint32 nonce that was used (from offset 76 of the header)
 * hash     - the 32-byte PoW hash
 */
int submit_pepepow_solution(int sockfd, const char *worker, const char *job_id,
                            uint32_t nonce, uint8_t *hash, MiningState *ms,
                            StratumContext *ctx, int reporting_index);

/*
 * CPU mining thread for PEPEPOW.
 * arg must be a heap-allocated MiningThread *.
 */
void *mining_cpu_thread_pepepow(void *arg);

#endif /* MINER_PEPEPOW_H */
