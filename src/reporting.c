#include "reporting.h"
#include "platform_compat.h"

ReportingDevice *init_reporting_device(uint32_t device_id, char *device_name)
{
  ReportingDevice *device = malloc(sizeof(ReportingDevice));
  if (!device)
  {
    return NULL;
  }

  device->device_id = device_id;
  device->device_name = device_name;
  device->nonces_processed = 0;
  device->accepted = 0;
  device->rejected = 0;
  device->stales = 0;

  if (pthread_mutex_init(&device->device_mutex, NULL) != 0)
  {
    free(device);
    return NULL;
  }

  return device;
}

HashrateDisplay *init_hashrate_display(uint32_t initial_capacity)
{
  HashrateDisplay *hd = malloc(sizeof(HashrateDisplay));
  if (!hd)
  {
    return NULL;
  }

  hd->devices = calloc(initial_capacity, sizeof(ReportingDevice *));
  if (!hd->devices)
  {
    free(hd);
    return NULL;
  }

  hd->device_count = 0;
  hd->device_capacity = initial_capacity;
  hd->running = 1;

  if (pthread_mutex_init(&hd->hashrate_mutex, NULL) != 0)
  {
    free(hd->devices);
    free(hd);
    return NULL;
  }

  return hd;
}

int add_reporting_device(HashrateDisplay *hd, ReportingDevice *device)
{
  if (!hd || !device)
  {
    return -1;
  }

  if (hd->device_count >= hd->device_capacity)
  {
    uint32_t new_capacity = hd->device_capacity ? hd->device_capacity * 2 : 4;
    ReportingDevice **new_devices = realloc(hd->devices, new_capacity * sizeof(ReportingDevice *));
    if (!new_devices)
    {
      return -1;
    }
    hd->devices = new_devices;
    hd->device_capacity = new_capacity;
  }

  hd->devices[hd->device_count++] = device;
  return 0;
}

void free_hashrate_display(HashrateDisplay *hd)
{
  if (!hd)
  {
    return;
  }

  pthread_mutex_lock(&hd->hashrate_mutex);
  for (uint32_t i = 0; i < hd->device_count; i++)
  {
    pthread_mutex_destroy(&hd->devices[i]->device_mutex);
    free(hd->devices[i]);
  }
  free(hd->devices);
  pthread_mutex_unlock(&hd->hashrate_mutex);
  pthread_mutex_destroy(&hd->hashrate_mutex);
  free(hd);
}

void *hashrate_display_thread(void *arg)
{
  StratumContext *ctx = (StratumContext *)arg;
  HashrateDisplay *hd = ctx->hd;
  const int seconds = 1;

  while (ctx->running)
  {
    sleep_ms(seconds * 1000);
    pthread_mutex_lock(&hd->hashrate_mutex);
    if (hd->device_count == 0)
    {
      pthread_mutex_unlock(&hd->hashrate_mutex);
      continue;
    }

    time_t t = time(NULL);
    struct tm tm;
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char time_str[9];
    strftime(time_str, sizeof(time_str), "%H:%M:%S", &tm);

    printf("[%-8s] ==================================================================================================== \n", time_str);
    printf("[%-8s] | Device ID \t\t | Hashrate \t\t| Accepted shares | Stale shares    | Rejected shares |\n", time_str);

    for (uint32_t i = 0; i < hd->device_count; i++)
    {
      ReportingDevice *device = hd->devices[i];
      pthread_mutex_lock(&device->device_mutex);
      device->hashrate = device->nonces_processed / (double)seconds;
      device->nonces_processed = 0;
      pthread_mutex_unlock(&device->device_mutex);
      if (strcmp(device->device_name, "CPU") == 0)
      {
        printf("[%-8s] | %s \t\t | ", time_str, device->device_name);
      }
      else
      {
        printf("[%-8s] | %s\t | ", time_str, device->device_name);
      }

      if (device->hashrate > 1000000000000.0)
      {
        printf("%-6.2f TH/s \t\t| ", device->hashrate / 1000000000000.0);
      }
      else if (device->hashrate > 1000000000.0)
      {
        printf("%-6.2f GH/s \t\t| ", device->hashrate / 1000000000.0);
      }
      else if (device->hashrate > 1000000.0)
      {
        printf("%-6.2f MH/s \t\t| ", device->hashrate / 1000000.0);
      }
      else if (device->hashrate > 1000.0)
      {
        printf("%-6.2f KH/s \t\t| ", device->hashrate / 1000.0);
      }
      else
      {
        printf("%-6.2f H/s  \t\t| ", device->hashrate);
      }

      printf("%I64d \t\t  | %I64d  \t\t    | %I64d\t\t      |\n", device->accepted, device->stales, device->rejected);
    }

    pthread_mutex_unlock(&hd->hashrate_mutex);
    printf("[%-8s] ====================================================================================================\n", time_str);
  }

  return NULL;
}

void list_gpus(StratumContext *ctx)
{
  time_t t = time(NULL);
  struct tm tm;
#ifdef _WIN32
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  char time_str[9];
  strftime(time_str, sizeof(time_str), "%H:%M:%S", &tm);
  printf("[%-8s] =======================================================================\n", time_str);
  printf("[%-8s] | Device Name \t\t\t | Device Index | Device BUS id  |\n", time_str);
  if (ctx->opencl_device_count > 0)
  {
    for (uint32_t i = 0; i < ctx->opencl_device_count; i++)
    {
      printf("[%-8s] | %s\t\t\t\t | %d\t\t| %d\t\t |\n", time_str, ctx->opencl_resources[i].device_name, ctx->cpu_device_count + i, ctx->opencl_resources[i].pci_bus_id);
    }
  }
  if (ctx->cuda_device_count > 0)
  {
    for (uint32_t i = 0; i < ctx->cuda_device_count; i++)
    {
      printf("[%-8s] | %s\t\t | %d\t\t| %d\t\t |\n", time_str, ctx->cuda_resources[i].device_name, ctx->cpu_device_count + ctx->opencl_device_count + i, ctx->cuda_resources[i].pci_bus_id);
    }
  }

  printf("[%-8s] =======================================================================\n", time_str);
}