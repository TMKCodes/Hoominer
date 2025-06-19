#ifndef REPORTING_H
#define REPORTING_H
#include <pthread.h>
#include <stdint.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include "stratum.h"

typedef struct HashrateDisplay HashrateDisplay;

typedef struct StratumContext StratumContext;

struct HashrateDisplay
{
  uint64_t nonces_processed;
  uint64_t accepted;
  uint64_t rejected;
  pthread_mutex_t hashrate_mutex;
  sig_atomic_t running;
};

HashrateDisplay *init_hashrate_display();
void cleanup_hashrate_display(StratumContext *ctx);
void *hashrate_display_thread(void *arg);

#endif