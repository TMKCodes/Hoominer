#include "overclocks.h"

int load_nvml(struct nvmlHandles *h)
{
  h->nvml = dlopen("libnvidia-ml.so", RTLD_LAZY);
  if (!h->nvml)
    return -1;

  h->nvmlInit = dlsym(h->nvml, "nvmlInit");
  h->nvmlShutdown = dlsym(h->nvml, "nvmlShutdown");
  h->nvmlErrorString = dlsym(h->nvml, "nvmlErrorString");
  h->nvmlDeviceGetHandleByIndex = dlsym(h->nvml, "nvmlDeviceGetHandleByIndex");
  h->nvmlDeviceSetApplicationsClocks = dlsym(h->nvml, "nvmlDeviceSetApplicationsClocks");
  h->nvmlDeviceResetApplicationsClocks = dlsym(h->nvml, "nvmlDeviceResetApplicationsClocks");
  h->nvmlDeviceSetPowerManagementLimit = dlsym(h->nvml, "nvmlDeviceSetPowerManagementLimit");
  h->nvmlDeviceGetPowerManagementLimitConstraints = dlsym(h->nvml, "nvmlDeviceGetPowerManagementLimitConstraints");

  if (!h->nvmlInit || !h->nvmlShutdown || !h->nvmlErrorString || !h->nvmlDeviceGetHandleByIndex ||
      !h->nvmlDeviceSetApplicationsClocks || !h->nvmlDeviceResetApplicationsClocks ||
      !h->nvmlDeviceSetPowerManagementLimit || !h->nvmlDeviceGetPowerManagementLimitConstraints)
    return -2;

  return 0;
}

int get_clocks(struct nvmlHandles *h, unsigned int gpuIndex, unsigned int *memMHz, unsigned int *coreMHz)
{
  typedef int (*nvmlDeviceGetApplicationsClock_t)(void *, unsigned int, unsigned int *);
  nvmlDeviceGetApplicationsClock_t nvmlDeviceGetApplicationsClock = dlsym(h->nvml, "nvmlDeviceGetApplicationsClock");
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