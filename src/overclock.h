#ifndef OVERCLOCKS_H
#define OVERCLOCKS_H
#include <stdio.h>
#include <dlfcn.h>

typedef int (*nvmlInit_t)(void);
typedef int (*nvmlShutdown_t)(void);
typedef const char *(*nvmlErrorString_t)(int);
typedef int (*nvmlDeviceGetHandleByIndex_t)(unsigned int, void **);
typedef int (*nvmlDeviceSetApplicationsClocks_t)(void *, unsigned int, unsigned int);
typedef int (*nvmlDeviceResetApplicationsClocks_t)(void *);
typedef int (*nvmlDeviceSetPowerManagementLimit_t)(void *, unsigned int);
typedef int (*nvmlDeviceGetPowerManagementLimitConstraints_t)(void *, unsigned int *, unsigned int *);
typedef int (*nvmlDeviceSetFanSpeed_t)(void *, unsigned int); // Note: NVML fan API is limited, might not exist in all versions

// Load NVML and resolve symbols (similar to your code)
struct nvmlHandles
{
  void *nvml;
  nvmlInit_t nvmlInit;
  nvmlShutdown_t nvmlShutdown;
  nvmlErrorString_t nvmlErrorString;
  nvmlDeviceGetHandleByIndex_t nvmlDeviceGetHandleByIndex;
  nvmlDeviceSetApplicationsClocks_t nvmlDeviceSetApplicationsClocks;
  nvmlDeviceResetApplicationsClocks_t nvmlDeviceResetApplicationsClocks;
  nvmlDeviceSetPowerManagementLimit_t nvmlDeviceSetPowerManagementLimit;
  nvmlDeviceGetPowerManagementLimitConstraints_t nvmlDeviceGetPowerManagementLimitConstraints;
};

#endif