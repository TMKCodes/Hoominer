#include "overclock.h"

int load_nvml(struct nvmlHandles *h)
{
#ifdef _WIN32
  HMODULE lib = LoadLibraryA("nvml.dll");
  if (!lib)
    return -1;
  h->nvml = lib;
#else
  h->nvml = dlopen("libnvidia-ml.so", RTLD_LAZY);
  if (!h->nvml)
    return -1;
#endif

  #ifdef _WIN32
  h->nvmlInit = (nvmlInit_t)GetProcAddress((HMODULE)h->nvml, "nvmlInit");
  h->nvmlShutdown = (nvmlShutdown_t)GetProcAddress((HMODULE)h->nvml, "nvmlShutdown");
  h->nvmlErrorString = (nvmlErrorString_t)GetProcAddress((HMODULE)h->nvml, "nvmlErrorString");
  h->nvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_t)GetProcAddress((HMODULE)h->nvml, "nvmlDeviceGetHandleByIndex");
  h->nvmlDeviceSetApplicationsClocks = (nvmlDeviceSetApplicationsClocks_t)GetProcAddress((HMODULE)h->nvml, "nvmlDeviceSetApplicationsClocks");
  h->nvmlDeviceResetApplicationsClocks = (nvmlDeviceResetApplicationsClocks_t)GetProcAddress((HMODULE)h->nvml, "nvmlDeviceResetApplicationsClocks");
  h->nvmlDeviceSetPowerManagementLimit = (nvmlDeviceSetPowerManagementLimit_t)GetProcAddress((HMODULE)h->nvml, "nvmlDeviceSetPowerManagementLimit");
  h->nvmlDeviceGetPowerManagementLimitConstraints = (nvmlDeviceGetPowerManagementLimitConstraints_t)GetProcAddress((HMODULE)h->nvml, "nvmlDeviceGetPowerManagementLimitConstraints");
  #else
  h->nvmlInit = dlsym(h->nvml, "nvmlInit");
  h->nvmlShutdown = dlsym(h->nvml, "nvmlShutdown");
  h->nvmlErrorString = dlsym(h->nvml, "nvmlErrorString");
  h->nvmlDeviceGetHandleByIndex = dlsym(h->nvml, "nvmlDeviceGetHandleByIndex");
  h->nvmlDeviceSetApplicationsClocks = dlsym(h->nvml, "nvmlDeviceSetApplicationsClocks");
  h->nvmlDeviceResetApplicationsClocks = dlsym(h->nvml, "nvmlDeviceResetApplicationsClocks");
  h->nvmlDeviceSetPowerManagementLimit = dlsym(h->nvml, "nvmlDeviceSetPowerManagementLimit");
  h->nvmlDeviceGetPowerManagementLimitConstraints = dlsym(h->nvml, "nvmlDeviceGetPowerManagementLimitConstraints");
  #endif

  if (!h->nvmlInit || !h->nvmlShutdown || !h->nvmlErrorString || !h->nvmlDeviceGetHandleByIndex ||
      !h->nvmlDeviceSetApplicationsClocks || !h->nvmlDeviceResetApplicationsClocks ||
      !h->nvmlDeviceSetPowerManagementLimit || !h->nvmlDeviceGetPowerManagementLimitConstraints)
    return -2;

  return 0;
}

int get_clocks(struct nvmlHandles *h, unsigned int gpuIndex, unsigned int *memMHz, unsigned int *coreMHz)
{
  typedef int (*nvmlDeviceGetApplicationsClock_t)(void *, unsigned int, unsigned int *);
#ifdef _WIN32
  nvmlDeviceGetApplicationsClock_t nvmlDeviceGetApplicationsClock =
      (nvmlDeviceGetApplicationsClock_t)GetProcAddress((HMODULE)h->nvml, "nvmlDeviceGetApplicationsClock");
#else
  nvmlDeviceGetApplicationsClock_t nvmlDeviceGetApplicationsClock = dlsym(h->nvml, "nvmlDeviceGetApplicationsClock");
#endif
  if (!nvmlDeviceGetApplicationsClock)
    return -1;

  void *device;
  int ret = h->nvmlDeviceGetHandleByIndex(gpuIndex, &device);
  if (ret)
    return ret;

  // NVML clock IDs (from nvml.h)
  const unsigned int NVML_CLOCK_MEM = 2;
  const unsigned int NVML_CLOCK_GRAPHICS = 0;

  ret = nvmlDeviceGetApplicationsClock(device, NVML_CLOCK_MEM, memMHz);
  if (ret)
    return ret;

  ret = nvmlDeviceGetApplicationsClock(device, NVML_CLOCK_GRAPHICS, coreMHz);
  return ret;
}

int set_fixed_clocks(struct nvmlHandles *h, unsigned int gpuIndex, unsigned int memMHz, unsigned int coreMHz)
{
  void *device;
  int ret = h->nvmlDeviceGetHandleByIndex(gpuIndex, &device);
  if (ret)
    return ret;

  return h->nvmlDeviceSetApplicationsClocks(device, memMHz, coreMHz);
}

int reset_clocks(struct nvmlHandles *h, unsigned int gpuIndex)
{
  void *device;
  int ret = h->nvmlDeviceGetHandleByIndex(gpuIndex, &device);
  if (ret)
    return ret;

  return h->nvmlDeviceResetApplicationsClocks(device);
}

int set_power_limit(struct nvmlHandles *h, unsigned int gpuIndex, unsigned int powerLimitMilliWatts)
{
  void *device;
  int ret = h->nvmlDeviceGetHandleByIndex(gpuIndex, &device);
  if (ret)
    return ret;

  // Check limits first
  unsigned int minLimit, maxLimit;
  ret = h->nvmlDeviceGetPowerManagementLimitConstraints(device, &minLimit, &maxLimit);
  if (ret)
    return ret;

  if (powerLimitMilliWatts < minLimit || powerLimitMilliWatts > maxLimit)
  {
    fprintf(stderr, "Power limit %u mW outside allowed range [%u, %u]\n", powerLimitMilliWatts, minLimit, maxLimit);
    return -1;
  }

  return h->nvmlDeviceSetPowerManagementLimit(device, powerLimitMilliWatts);
}