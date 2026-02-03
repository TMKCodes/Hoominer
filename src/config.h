#ifndef CONFIG_H
#define CONFIG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX_STRATUM_URLS 16

struct StratumConfig
{
  char *pool_ip;
  int pool_port;
  bool ssl_enabled;
};

struct HoominerConfig
{
  struct StratumConfig stratum_urls[MAX_STRATUM_URLS];
  int api_port;
  bool api_enabled;
  int stratum_urls_num;
  char *username;
  char *password;
  char *algorithm;
  int opencl_optimization_level;
  int gpu_work_multiplier;
  bool disable_cpu;
  bool disable_gpu;
  bool disable_opencl;
  bool disable_cuda;
  bool disable_opencl_cache;
  int opencl_reset_interval;
  int cpu_threads;
  char *selected_gpus_str;
  char *build_options;
  bool list_gpus;
  bool debug;
  int selected_gpus[256];
  int selected_gpus_num;
};

void parse_args(int argc, char **argv, struct HoominerConfig *config);
struct StratumConfig *get_stratum(struct HoominerConfig *config, int current_index);
void show_config(char **argv);

#endif