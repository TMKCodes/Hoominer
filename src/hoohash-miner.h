#ifndef HOOHASH_MINER_H
#define HOOHASH_MINER_H
#include <json-c/json.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/socket.h>
#include <signal.h>
#include <endian.h>
#include "target.h"
#include "../algorithms/hoohash/hoohash.h"

typedef struct MiningState MiningState;
typedef struct StratumContext StratumContext;

MiningState *init_mining_state();
void cleanup_mining_state(MiningState *state);
void uint64_to_little_endian(uint64_t value, uint8_t *buffer);
uint64_t little_endian_to_uint64(const uint8_t *buffer);
void smallJobHeader(const uint64_t *ids, uint8_t *headerData);
int hex_to_bytes(const char *hex, uint8_t *bytes, size_t len);
void print_hex(const char *label, const uint8_t *data, size_t len);
int submit_mining_solution(int sockfd, const char *worker, const char *job_id, uint64_t nonce, uint8_t *hash);
void *mining_thread(void *arg);
void cleanup_job(MiningState *ms);
void cleanup_mining_threads(MiningState *ms);
int start_mining_threads(StratumContext *ctx, MiningState *ms);
void process_stratum_message(json_object *message, StratumContext *ctx, MiningState *ms);

#endif