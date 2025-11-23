#include "cuda-host.h"

// Function pointers for CUDA Driver API
typedef CUresult (*CUInit_t)(unsigned int);
typedef CUresult (*CUDeviceGetCount_t)(int *);
typedef CUresult (*CUDeviceGet_t)(CUdevice *, int);
typedef CUresult (*CUDeviceGetName_t)(char *, int, CUdevice);
typedef CUresult (*CUDeviceGetAttribute_t)(int *, CUdevice_attribute, CUdevice);
typedef CUresult (*CUModuleLoadData_t)(CUmodule *, const void *);
typedef CUresult (*CUModuleGetFunction_t)(CUfunction *, CUmodule, const char *);
typedef CUresult (*CULaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int,
                                     unsigned int, cudaStream_t, void **, void **);
typedef CUresult (*CUModuleUnload_t)(CUmodule);

// Global handles for dynamic loading
void *cuda_lib_handle = NULL;
CUInit_t p_cuInit = NULL;
CUDeviceGetCount_t p_cuDeviceGetCount = NULL;
CUDeviceGet_t p_cuDeviceGet = NULL;
CUDeviceGetName_t p_cuDeviceGetName = NULL;
CUDeviceGetAttribute_t p_cuDeviceGetAttribute = NULL;
CUModuleLoadData_t p_cuModuleLoadData = NULL;
CUModuleGetFunction_t p_cuModuleGetFunction = NULL;
CULaunchKernel_t p_cuLaunchKernel = NULL;
CUModuleUnload_t p_cuModuleUnload = NULL;

// Load CUDA driver library dynamically
int load_cuda_library()
{
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("nvcuda.dll");
    if (!lib)
    {
        fprintf(stderr, "Failed to load nvcuda.dll\n");
        return 0;
    }
    cuda_lib_handle = lib;

    p_cuInit = (CUInit_t)GetProcAddress(lib, "cuInit");
    p_cuDeviceGetCount = (CUDeviceGetCount_t)GetProcAddress(lib, "cuDeviceGetCount");
    p_cuDeviceGet = (CUDeviceGet_t)GetProcAddress(lib, "cuDeviceGet");
    p_cuDeviceGetName = (CUDeviceGetName_t)GetProcAddress(lib, "cuDeviceGetName");
    p_cuDeviceGetAttribute = (CUDeviceGetAttribute_t)GetProcAddress(lib, "cuDeviceGetAttribute");
    p_cuModuleLoadData = (CUModuleLoadData_t)GetProcAddress(lib, "cuModuleLoadData");
    p_cuModuleGetFunction = (CUModuleGetFunction_t)GetProcAddress(lib, "cuModuleGetFunction");
    p_cuLaunchKernel = (CULaunchKernel_t)GetProcAddress(lib, "cuLaunchKernel");
    p_cuModuleUnload = (CUModuleUnload_t)GetProcAddress(lib, "cuModuleUnload");
#else
    cuda_lib_handle = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!cuda_lib_handle)
    {
        fprintf(stderr, "Failed to load libcuda.so.1: %s\n", dlerror());
        return 0; // Indicate failure
    }

    // Resolve CUDA Driver API functions
    p_cuInit = (CUInit_t)dlsym(cuda_lib_handle, "cuInit");
    p_cuDeviceGetCount = (CUDeviceGetCount_t)dlsym(cuda_lib_handle, "cuDeviceGetCount");
    p_cuDeviceGet = (CUDeviceGet_t)dlsym(cuda_lib_handle, "cuDeviceGet");
    p_cuDeviceGetName = (CUDeviceGetName_t)dlsym(cuda_lib_handle, "cuDeviceGetName");
    p_cuDeviceGetAttribute = (CUDeviceGetAttribute_t)dlsym(cuda_lib_handle, "cuDeviceGetAttribute");
    p_cuModuleLoadData = (CUModuleLoadData_t)dlsym(cuda_lib_handle, "cuModuleLoadData");
    p_cuModuleGetFunction = (CUModuleGetFunction_t)dlsym(cuda_lib_handle, "cuModuleGetFunction");
    p_cuLaunchKernel = (CULaunchKernel_t)dlsym(cuda_lib_handle, "cuLaunchKernel");
    p_cuModuleUnload = (CUModuleUnload_t)dlsym(cuda_lib_handle, "cuModuleUnload");
#endif

    if (!p_cuInit || !p_cuDeviceGetCount || !p_cuDeviceGet || !p_cuDeviceGetName ||
        !p_cuDeviceGetAttribute || !p_cuModuleLoadData || !p_cuModuleGetFunction ||
        !p_cuLaunchKernel || !p_cuModuleUnload)
    {
#ifdef _WIN32
        fprintf(stderr, "Failed to resolve CUDA Driver API functions\n");
        FreeLibrary((HMODULE)cuda_lib_handle);
#else
        fprintf(stderr, "Failed to resolve CUDA Driver API functions: %s\n", dlerror());
        dlclose(cuda_lib_handle);
#endif
        cuda_lib_handle = NULL;
        return 0; // Indicate failure
    }

    // Initialize CUDA Driver API
    CUresult cu_err = p_cuInit(0);
    if (cu_err != CUDA_SUCCESS)
    {
        // Note: cuGetErrorString is not dynamically loaded here for simplicity
        fprintf(stderr, "cuInit failed: %d\n", cu_err);
#ifdef _WIN32
        FreeLibrary((HMODULE)cuda_lib_handle);
#else
        dlclose(cuda_lib_handle);
#endif
        cuda_lib_handle = NULL;
        return 0;
    }

    return 1; // Indicate success
}

// Function to calculate optimal grid and block dimensions
static void calculate_optimal_dimensions(CudaResources *resource, int work_multiplier)
{
    const int warp_size = 32;
    int sm_count = resource->device_prop.multiProcessorCount;
    int max_threads_per_block = resource->device_prop.maxThreadsPerBlock;
    int max_threads_per_sm = resource->device_prop.maxThreadsPerMultiProcessor;

    // Start with a typical block size
    int threads_per_block = 256;
    int max_active_blocks_per_sm = 0;

    // Estimate optimal blocks per SM
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_per_sm, resource->kernel, threads_per_block, 0);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Occupancy calc failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
        max_active_blocks_per_sm = 2;
    }

    // Choose target blocks per SM
    int target_blocks_per_sm = max_active_blocks_per_sm >= 2 ? max_active_blocks_per_sm : 2;

    // Adjust threads per block to stay within per-SM thread limit
    int max_possible_threads = max_threads_per_sm / target_blocks_per_sm;
    threads_per_block = (threads_per_block > max_possible_threads) ? max_possible_threads : threads_per_block;
    threads_per_block = (threads_per_block / warp_size) * warp_size;

    // Clamp to device limits
    if (threads_per_block > max_threads_per_block)
        threads_per_block = max_threads_per_block - (max_threads_per_block % warp_size);
    if (threads_per_block < warp_size)
        threads_per_block = warp_size;

    // Compute total grid size
    int grid_size = sm_count * target_blocks_per_sm;
    if (grid_size > resource->device_prop.maxGridSize[0])
        grid_size = resource->device_prop.maxGridSize[0];

    resource->optimal_block_size = threads_per_block * work_multiplier;
    resource->optimal_grid_size = grid_size;

    printf("Calculated for %s: block_size=%d, grid_size=%d\n",
           resource->device_name, threads_per_block, grid_size);
}

int compare_pci_bus_id(const void *a, const void *b)
{
    const CudaResources *ra = (const CudaResources *)a;
    const CudaResources *rb = (const CudaResources *)b;
    return (int)(ra->pci_bus_id - rb->pci_bus_id);
}

CudaResources *initialize_all_cuda_gpus(unsigned int *device_count, int selected_gpus[256], int selected_gpus_num)
{
    cudaError_t err = cudaSuccess;
    int num_devices, devices_found;

    *device_count = 0;
    devices_found = 0;

    // Check if CUDA library is loaded
    if (!load_cuda_library())
    {
        fprintf(stderr, "CUDA library loading failed, checking for AMD GPUs...\n");
        // TODO: Add OpenCL initialization for AMD GPUs here
        return NULL;
    }

    // Get device count using dynamically loaded function
    CUresult cu_err = p_cuDeviceGetCount(&num_devices);
    if (cu_err != CUDA_SUCCESS || num_devices == 0)
    {
        fprintf(stderr, "No CUDA devices found or cuDeviceGetCount failed: %d\n", cu_err);
#ifdef _WIN32
        FreeLibrary((HMODULE)cuda_lib_handle);
#else
        dlclose(cuda_lib_handle);
#endif
        cuda_lib_handle = NULL;
        return NULL;
    }

    // Allocate memory for the maximum possible number of valid devices
    CudaResources *res = (CudaResources *)calloc(num_devices, sizeof(CudaResources));
    if (!res)
    {
        fprintf(stderr, "Memory allocation failed\n");
#ifdef _WIN32
        FreeLibrary((HMODULE)cuda_lib_handle);
#else
        dlclose(cuda_lib_handle);
#endif
        cuda_lib_handle = NULL;
        return NULL;
    }

    for (unsigned int i = 0; i < (unsigned int)num_devices; i++)
    {
        CUdevice device;
        cu_err = p_cuDeviceGet(&device, i);
        if (cu_err != CUDA_SUCCESS)
        {
            fprintf(stderr, "cuDeviceGet failed for device %u: %d\n", i, cu_err);
            continue;
        }

        err = cudaSetDevice(i);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Set device %u failed: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        // Temporary device properties for checking
        cudaDeviceProp temp_prop;
        err = cudaGetDeviceProperties(&temp_prop, i);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Device %u properties query failed: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        int compute_mode = 0;
        // Query compute mode via Driver API to avoid relying on deprecated runtime field
        cu_err = p_cuDeviceGetAttribute(&compute_mode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device);
        if (cu_err != CUDA_SUCCESS)
        {
            fprintf(stderr, "Device %u compute mode query failed: %d\n", i, cu_err);
            compute_mode = 0; // assume default
        }
        if (compute_mode == CU_COMPUTEMODE_PROHIBITED || temp_prop.major < 2)
        {
            fprintf(stderr, "Device %u (PCI-BUS-ID: %u, %s) lacks sufficient compute capability\n",
                    i, temp_prop.pciBusID, temp_prop.name);
            continue;
        }

        // Check if the device is in the selected_gpus list (if any)
        if (selected_gpus_num > 0)
        {
            int found = 0;
            for (int x = 0; x < selected_gpus_num; x++)
            {
                if (temp_prop.pciBusID == selected_gpus[x])
                {
                    found = 1;
                    break;
                }
            }
            if (found == 0)
            {
                continue;
            }
        }
        else
        {
            printf("Using device %d\n", temp_prop.pciBusID);
        }

        // Populate the res array only for valid devices
        strncpy(res[devices_found].device_name, temp_prop.name, sizeof(res[devices_found].device_name) - 1);
        res[devices_found].device_prop = temp_prop;
        res[devices_found].device_id = i;
        res[devices_found].pci_bus_id = temp_prop.pciBusID;

        err = cudaStreamCreate(&res[devices_found].stream);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Device %u (PCI-BUS-ID: %u) stream creation failed: %s\n",
                    i, res[devices_found].pci_bus_id, cudaGetErrorString(err));
            continue;
        }

        err = cudaMalloc(&res[devices_found].previous_header, DOMAIN_HASH_SIZE);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Device %u (PCI-BUS-ID: %u) previous_header allocation failed: %s\n",
                    i, res[devices_found].pci_bus_id, cudaGetErrorString(err));
            cudaStreamDestroy(res[devices_found].stream);
            continue;
        }
        err = cudaMalloc(&res[devices_found].timestamp, sizeof(int64_t));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Device %u (PCI-BUS-ID: %u) timestamp allocation failed: %s\n",
                    i, res[devices_found].pci_bus_id, cudaGetErrorString(err));
            cudaStreamDestroy(res[devices_found].stream);
            cudaFree(res[devices_found].previous_header);
            continue;
        }
        err = cudaMalloc(&res[devices_found].matrix, 64 * 64 * sizeof(double));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Device %u (PCI-BUS-ID: %u) matrix allocation failed: %s\n",
                    i, res[devices_found].pci_bus_id, cudaGetErrorString(err));
            cudaStreamDestroy(res[devices_found].stream);
            cudaFree(res[devices_found].previous_header);
            cudaFree(res[devices_found].timestamp);
            continue;
        }
        err = cudaMalloc(&res[devices_found].target, DOMAIN_HASH_SIZE);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Device %u (PCI-BUS-ID: %u) target allocation failed: %s\n",
                    i, res[devices_found].pci_bus_id, cudaGetErrorString(err));
            cudaStreamDestroy(res[devices_found].stream);
            cudaFree(res[devices_found].previous_header);
            cudaFree(res[devices_found].timestamp);
            cudaFree(res[devices_found].matrix);
            continue;
        }
        err = cudaMalloc(&res[devices_found].result, sizeof(CudaResult));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Device %u (PCI-BUS-ID: %u) result allocation failed: %s\n",
                    i, res[devices_found].pci_bus_id, cudaGetErrorString(err));
            cudaStreamDestroy(res[devices_found].stream);
            cudaFree(res[devices_found].previous_header);
            cudaFree(res[devices_found].timestamp);
            cudaFree(res[devices_found].matrix);
            cudaFree(res[devices_found].target);
            continue;
        }
        err = cudaMalloc(&res[devices_found].nonces_processed, sizeof(uint64_t));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Device %u (PCI-BUS-ID: %u) nonces_processed allocation failed: %s\n",
                    i, res[devices_found].pci_bus_id, cudaGetErrorString(err));
            cudaStreamDestroy(res[devices_found].stream);
            cudaFree(res[devices_found].previous_header);
            cudaFree(res[devices_found].timestamp);
            cudaFree(res[devices_found].matrix);
            cudaFree(res[devices_found].target);
            cudaFree(res[devices_found].result);
            continue;
        }

        devices_found++;
    }

    if (devices_found == 0)
    {
        free(res);
#ifdef _WIN32
        FreeLibrary((HMODULE)cuda_lib_handle);
#else
        dlclose(cuda_lib_handle);
#endif
        cuda_lib_handle = NULL;
        return NULL;
    }

    // Reallocate to the exact number of devices found
    res = (CudaResources *)realloc(res, devices_found * sizeof(CudaResources));
    if (!res)
    {
        fprintf(stderr, "Memory reallocation failed\n");
#ifdef _WIN32
        FreeLibrary((HMODULE)cuda_lib_handle);
#else
        dlclose(cuda_lib_handle);
#endif
        cuda_lib_handle = NULL;
        return NULL;
    }

    qsort(res, devices_found, sizeof(CudaResources), compare_pci_bus_id);
    *device_count = devices_found;
    return res;
}

bool load_cuda_kernel_binary(CudaResources *resource, const char *cubin_filename, const char *kernel_name, int work_multiplier)
{
    cudaError_t err;
    CUresult cu_err;

    if (!cuda_lib_handle)
    {
        fprintf(stderr, "CUDA library not loaded for %s\n", resource->device_name);
        return false;
    }

    FILE *file = fopen(cubin_filename, "rb");
    if (!file)
    {
        fprintf(stderr, "Failed to open %s: %s\n", cubin_filename, strerror(errno));
        return false;
    }

    fseek(file, 0, SEEK_END);
    size_t binary_size = ftell(file);
    rewind(file);
    char *binary = (char *)malloc(binary_size);
    if (!binary || fread(binary, 1, binary_size, file) != binary_size)
    {
        fprintf(stderr, "Failed to read %s\n", cubin_filename);
        free(binary);
        fclose(file);
        return cudaErrorInvalidValue;
    }
    fclose(file);

    err = cudaSetDevice(resource->device_id);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Set device failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
        free(binary);
        return err;
    }

    cu_err = p_cuModuleLoadData(&resource->module, binary);
    free(binary);
    if (cu_err != CUDA_SUCCESS)
    {
        // Note: cuGetErrorString is not dynamically loaded for simplicity
        fprintf(stderr, "Module load failed for %s: %d\n", resource->device_name, cu_err);
        return false;
    }

    cu_err = p_cuModuleGetFunction(&resource->kernel, resource->module, kernel_name);
    if (cu_err != CUDA_SUCCESS)
    {
        fprintf(stderr, "Kernel %s creation failed for %s: %d\n", kernel_name, resource->device_name, cu_err);
        p_cuModuleUnload(resource->module);
        return false;
    }

    // Calculate optimal grid and block dimensions after kernel is loaded
    calculate_optimal_dimensions(resource, work_multiplier);

    printf("Kernel %s loaded for %s\n", kernel_name, resource->device_name);

    return cudaSuccess;
}

cudaError_t run_cuda_hoohash_kernel(CudaResources *resource, unsigned char *previous_header, unsigned char *target, double matrix[64][64],
                                    int64_t timestamp, uint64_t start_nonce, CudaResult *result)
{
    cudaError_t err;

    if (!resource || !previous_header || !target || !matrix || !result)
    {
        fprintf(stderr, "Invalid input pointers for %s\n", resource ? resource->device_name : "unknown");
        return cudaErrorInvalidValue;
    }

    if (!cuda_lib_handle)
    {
        fprintf(stderr, "CUDA library not loaded for %s\n", resource->device_name);
        return cudaErrorInitializationError;
    }

    err = cudaSetDevice(resource->device_id);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Set device failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
        return err;
    }

    err = cudaMemcpyAsync(resource->previous_header, previous_header, DOMAIN_HASH_SIZE, cudaMemcpyHostToDevice, resource->stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Memory copy to previous_header failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
        return err;
    }

    err = cudaMemcpyAsync(resource->timestamp, &timestamp, sizeof(int64_t), cudaMemcpyHostToDevice, resource->stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Memory copy to timestamp failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
        return err;
    }

    err = cudaMemcpyAsync(resource->matrix, matrix, 64 * 64 * sizeof(double), cudaMemcpyHostToDevice, resource->stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Memory copy to matrix failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
        return err;
    }

    err = cudaMemcpyAsync(resource->target, target, DOMAIN_HASH_SIZE, cudaMemcpyHostToDevice, resource->stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Memory copy to target failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
        return err;
    }

    CudaResult init_result = {0};
    err = cudaMemcpyAsync(resource->result, &init_result, sizeof(CudaResult), cudaMemcpyHostToDevice, resource->stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Result buffer initialization failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
        return err;
    }

    void *args[] = {
        &start_nonce,
        &resource->previous_header,
        &resource->timestamp,
        &resource->matrix,
        &resource->target,
        &resource->result,
        &resource->nonces_processed};

    CUresult cu_err = p_cuLaunchKernel(resource->kernel,
                                       resource->optimal_grid_size, 1, 1,
                                       resource->optimal_block_size, 1, 1,
                                       0,
                                       resource->stream,
                                       args,
                                       NULL);
    if (cu_err != CUDA_SUCCESS)
    {
        fprintf(stderr, "Kernel launch failed for %s: %d\n", resource->device_name, cu_err);
        return cudaErrorLaunchFailure;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch error for %s: %s\n", resource->device_name, cudaGetErrorString(err));
        return err;
    }

    err = cudaMemcpyAsync(result, resource->result, sizeof(CudaResult), cudaMemcpyDeviceToHost, resource->stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Result copy failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
        return err;
    }

    // Check if the matrix is correct
    double matrix_back[64][64];
    err = cudaMemcpyAsync(matrix_back, resource->matrix, 64 * 64 * sizeof(double), cudaMemcpyDeviceToHost, resource->stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "matrix_back copy failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
        return err;
    }

    err = cudaStreamSynchronize(resource->stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Stream synchronization failed for %s: %s\n", resource->device_name, cudaGetErrorString(err));
        return err;
    }

    // Check if the matrix is correct
    for (size_t r = 0; r < 64; ++r)
    {
        for (size_t c = 0; c < 64; ++c)
        {
            if (matrix_back[r][c] != matrix[r][c])
            {
                fprintf(stderr, "Matrix mismatch at [%zu][%zu]: %f vs %f\n", r, c, matrix_back[r][c], matrix[r][c]);
            }
        }
    }

    return cudaSuccess;
}

void cleanup_cuda_resources(CudaResources *resource)
{
    if (!resource)
        return;

    cudaSetDevice(resource->device_id);
    cudaFree(resource->previous_header);
    cudaFree(resource->timestamp);
    cudaFree(resource->matrix);
    cudaFree(resource->target);
    cudaFree(resource->result);
    if (resource->module && p_cuModuleUnload)
    {
        p_cuModuleUnload(resource->module);
    }
    cudaStreamDestroy(resource->stream);
}

void cleanup_all_cuda_gpus(CudaResources *resources, unsigned int device_count)
{
    if (!resources)
        return;
    for (unsigned int i = 0; i < device_count; i++)
    {
        cleanup_cuda_resources(&resources[i]);
    }
    free(resources);
    if (cuda_lib_handle)
    {
#ifdef _WIN32
        FreeLibrary((HMODULE)cuda_lib_handle);
#else
        dlclose(cuda_lib_handle);
#endif
        cuda_lib_handle = NULL;
    }
}