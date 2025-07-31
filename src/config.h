#ifndef CONFIG_H
#define CONFIG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

struct HoominerConfig
{
  char *pool_ip;
  int pool_port;
  char *username;
  char *password;
  char *algorithm;
  bool disable_cpu;
  bool disable_gpu;
  bool disable_opencl;
  bool disable_cuda;
  int cpu_threads;
  char *selected_gpus_str;
  bool ssl_enabled;
  bool list_gpus;
  bool debug;
  int selected_gpus[256];
  int selected_gpus_num;
};

void parse_args(int argc, char **argv, struct HoominerConfig *config);

#endif