#ifndef STRATUM_H
#define STRATUM_H
#include <sys/select.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <json-c/json.h>
#include <stdbool.h>
#include "datatypes.h"
#include "hoohash-miner.h"

#define BUFFER_SIZE 8192

typedef struct StratumContext StratumContext;
typedef struct OpenCLResources OpenCLResources;
typedef struct HashrateDisplay HashrateDisplay;
typedef struct MiningState MiningState;
typedef struct CudaResources CudaResources;

struct StratumContext
{
  volatile int running;
  int sockfd;
  const char *worker;
  int disable_cpu;
  int disable_gpu;
  unsigned int cpu_device_count;
  unsigned int opencl_device_count;
  unsigned int cuda_device_count;
  OpenCLResources *opencl_resources;
  CudaResources *cuda_resources;
  HashrateDisplay *hd;
  MiningState *ms;
  pthread_t recv_thread;
  IntFifo mining_submit_fifo;
};

StratumContext *init_stratum_context();
void *stratum_receive_thread(void *arg);
int stratum_subscribe(int sockfd);
int stratum_authenticate(int sockfd, const char *username, const char *password);
int connect_to_stratum_server(const char *hostname, int port);

#endif