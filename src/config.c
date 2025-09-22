#include "config.h"

void parse_args(int argc, char **argv, struct HoominerConfig *config)
{
  config->disable_cpu = false;
  config->disable_gpu = false;
  config->disable_opencl = false;
  config->disable_cuda = false;
  config->list_gpus = false;
  config->cpu_threads = 0;
  config->debug = false;
  config->opencl_optimization_level = 2;
  config->gpu_work_multiplier = 1;
  config->selected_gpus_num = 0;
  config->build_options = NULL;
  config->stratum_urls_num = 0;
  bool gpus_selected = false;

  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "--user") && i + 1 < argc)
      config->username = argv[++i];
    else if (!strcmp(argv[i], "--pass") && i + 1 < argc)
      config->password = argv[++i];
    else if (!strcmp(argv[i], "--algorithm") && i + 1 < argc)
      config->algorithm = argv[++i];
    else if (!strcmp(argv[i], "--opencl-o") && i + 1 < argc)
      config->opencl_optimization_level = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--gpu-work-multiplier") && i + 1 < argc)
      config->gpu_work_multiplier = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--disable-cpu"))
      config->disable_cpu = true;
    else if (!strcmp(argv[i], "--disable-gpu"))
      config->disable_gpu = true;
    else if (!strcmp(argv[i], "--disable-opencl"))
      config->disable_gpu = true;
    else if (!strcmp(argv[i], "--disable-cuda"))
      config->disable_gpu = true;
    else if (!strcmp(argv[i], "--list-gpus"))
      config->list_gpus = true;
    else if (!strcmp(argv[i], "--debug"))
      config->debug = true;
    else if (!strcmp(argv[i], "--opencl-build-options") && i + 1 < argc)
    {
      config->build_options = strdup(argv[++i]);
      if (!config->build_options)
      {
        fprintf(stderr, "Memory allocation failed for build options\n");
        exit(1);
      }
      printf("Parsed OpenCL build options: %s\n", config->build_options);
    }
    else if (!strcmp(argv[i], "--cpu-threads") && i + 1 < argc)
      config->cpu_threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--gpu-ids") && i + 1 < argc)
    {
      config->selected_gpus_str = argv[++i];
      gpus_selected = true;
      printf("Selected GPUs: %s\n", config->selected_gpus_str);
    }
    else if (!strcmp(argv[i], "--stratum") && i + 1 < argc)
    {
      char *stratum_urls_str = strdup(argv[++i]);
      if (!stratum_urls_str)
      {
        fprintf(stderr, "Memory allocation failed for stratum URLs\n");
        exit(1);
      }

      char *token = strtok(stratum_urls_str, " ");
      while (token && config->stratum_urls_num < MAX_STRATUM_URLS)
      {
        // Check if token is a flag (starts with --)
        if (strncmp(token, "--", 2) == 0)
        {
          // Put the flag back into argv and adjust i to process it in the next iteration
          i--;
          argv[i] = token;
          break;
        }

        struct StratumConfig *stratum = &config->stratum_urls[config->stratum_urls_num];

        if (strncmp(token, "stratum+tcp://", 14) == 0)
        {
          stratum->ssl_enabled = false;
        }
        else if (strncmp(token, "stratum+ssl://", 14) == 0 || strncmp(token, "stratum+tls://", 14) == 0)
        {
          stratum->ssl_enabled = true;
        }
        else
        {
          fprintf(stderr, "Invalid stratum URL format: must start with stratum+tcp://, stratum+ssl://, or stratum+tls://\n");
          free(stratum_urls_str);
          exit(1);
        }

        const char *url_part = token + 14;
        char *url = malloc(strlen(url_part) + 1);
        if (!url)
        {
          fprintf(stderr, "Memory allocation failed\n");
          free(stratum_urls_str);
          exit(1);
        }
        strcpy(url, url_part);

        char *colon = strchr(url, ':');
        if (!colon)
        {
          fprintf(stderr, "Stratum URL missing port\n");
          free(url);
          free(stratum_urls_str);
          exit(1);
        }
        *colon = '\0';

        stratum->pool_ip = malloc(strlen(url) + 1);
        if (!stratum->pool_ip)
        {
          fprintf(stderr, "Memory allocation failed\n");
          free(url);
          free(stratum_urls_str);
          exit(1);
        }
        strcpy(stratum->pool_ip, url);

        stratum->pool_port = atoi(colon + 1);
        config->stratum_urls_num++;
        free(url);
        token = strtok(NULL, " ");
      }
      free(stratum_urls_str);
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("Usage: %s [--stratum <stratum+tcp://domain:port> [stratum+tcp://domain:port ...]] [--user <user>] [--pass <pass>]\n", argv[0]);
      printf("\nGeneral parameters: \n");
      printf("--algorithm <algorithm>\t\tThe algorithm to mine, by default 'hoohash'.\n");
      printf("--stratum <stratum+tcp://domain:port [stratum+tcp://domain:port ...]>\t\tThe stratum protocol://address:port to specify connection points (stratum+ssl:// or stratum+tls:// to enable SSL/TLS). Multiple URLs separated by spaces.\n");
      printf("--user <user>\t\t\t\tStratum username (Usually mining wallet address).\n");
      printf("--password <password>\t\t\tStratum password (Usually not required or used as additional stratum parameters).\n");
      printf("--disable-cpu\t\t\t\tDisable CPU mining completely.\n");
      printf("--disable-gpu\t\t\t\tDisable GPU mining completely.\n");
      printf("--disable-opencl\t\t\tDisable OpenCL mining.\n");
      printf("--disable-cuda\t\t\t\tDisable CUDA mining.\n");
      printf("--debug\t\t\t\t\tMore information displayed.\n");
      printf("\nCPU parameters: \n");
      printf("--cpu-threads <thread-count>\t\tSelect how many CPU threads to create.\n");
      printf("\nGPU parameters: \n");
      printf("--list-gpus\t\t\t\tList gpu bus id's.\n");
      printf("--gpu-ids <bus-id list>\t\t\tSelect which GPU's to use, separate bus id's with comma (if not specified all devices will be used).\n");
      printf("--opencl-o <level>\t\t\tSelect OpenCL compile time optimization level.\n");
      printf("--gpu-work-multiplier <level>\t\tSelect multiplier for OpenCL global work size or Nvidia blocks size.\n");
      exit(0);
    }
  }
  if (gpus_selected == true)
  {
    config->selected_gpus_num = 0;
    char *gpu_ids_str = strdup(config->selected_gpus_str);
    char *token = strtok(gpu_ids_str, ",");

    while (token && config->selected_gpus_num < 16)
    {
      config->selected_gpus[config->selected_gpus_num++] = atoi(token);
      token = strtok(NULL, ",");
    }
    free(gpu_ids_str);
  }
}


struct StratumConfig *get_next_stratum(struct HoominerConfig *config)
{
  if (config->stratum_urls_num == 0)
  {
    return NULL;
  }

  static int current_index = -1;
  current_index = (current_index + 1) % config->stratum_urls_num;
  return &config->stratum_urls[current_index];
}