// reporting.h
#ifndef REPORTING_H
#define REPORTING_H
#include <pthread.h>
#include <stdint.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "stratum.h"

typedef struct ReportingDevice ReportingDevice;
typedef struct HashrateDisplay HashrateDisplay;
typedef struct StratumContext StratumContext;

struct ReportingDevice
{
  uint32_t device_id;
  char *device_name;
  uint64_t nonces_processed;
  uint64_t accepted;
  uint64_t stales;
  uint64_t rejected;
  pthread_mutex_t device_mutex;
};

struct HashrateDisplay
{
  ReportingDevice **devices;
  uint32_t device_count;
  uint32_t device_capacity;
  pthread_mutex_t hashrate_mutex;
  pthread_t display_thread;
  sig_atomic_t running;
};

ReportingDevice *init_reporting_device(uint32_t device_id, char *device_name);
HashrateDisplay *init_hashrate_display(uint32_t initial_capacity);
int add_reporting_device(HashrateDisplay *hd, ReportingDevice *device);
void free_hashrate_display(HashrateDisplay *hd);
void *hashrate_display_thread(void *arg);

#endif