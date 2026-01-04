#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define CHK(call) \
do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
cudaGetErrorString(err)); \
assert(0); \
} \
} while(0)


__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Automatic GPU selection based on MPI rank
    int deviceCount;
    CHK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found on rank %d\n", rank);
        MPI_Finalize();
        return -1;
    }

    // Assign GPU: rank modulo number of available GPUs
    int device = rank % deviceCount;
    CHK(cudaSetDevice(device));

    // Get device properties for confirmation
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, device));

    printf("Rank %d/%d using GPU %d (%s) with %d multiprocessors\n",
        rank, size, device, prop.name, prop.multiProcessorCount);

    // Problem size (each process handles a portion)
    const int n = 1000000;
    int base = n / size;
    int rem = n % size;
    int count = base + (rank < rem); // distribute remainder
    int offset = rank * base + (rank < rem ? rank : rem);

    // Allocate host memory per rank
    float* h_a = (float*)malloc(count * sizeof(float));
    float* h_b = (float*)malloc(count * sizeof(float));
    float* h_c = (float*)malloc(count * sizeof(float));

    // Initialize data
    for (int i = 0; i < count; i++) {
        h_a[i] = offset + i;
        h_b[i] = (offset + i) * 2.0f;
        // thus, h_c[i] should be: (start_idx + i) * 3.0f
    }

    // Allocate device memory
    float* d_a, * d_b, * d_c;
    CHK(cudaMalloc(&d_a, count * sizeof(float)));
    CHK(cudaMalloc(&d_b, count * sizeof(float)));
    CHK(cudaMalloc(&d_c, count * sizeof(float)));

    // Copy data to GPU
    CHK(cudaMemcpy(d_a, h_a, count * sizeof(float), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_b, h_b, count * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (count + blockSize - 1) / blockSize;

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));

    CHK(cudaEventRecord(start));
    vector_add << <gridSize, blockSize >> > (d_a, d_b, d_c, count);
    CHK(cudaEventRecord(stop));
    CHK(cudaEventSynchronize(stop));          // not DeviceSynchronize
    CHK(cudaEventElapsedTime(&milliseconds, start, stop));

    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());

    // Copy result back
    CHK(cudaMemcpy(h_c, d_c, count * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few results for this rank
    bool success = true;
    for (int i = 0; i < 10 && i < count; i++) {
        float expected = h_a[i] + h_b[i];
        if (fabsf(h_c[i] - expected) > 1e-5) {
            success = false;
            break;
        }
    }

    printf("Rank %d: Processed %d elements in %.2f ms on GPU %d - %s\n",
        rank, count, milliseconds, device, success ? "SUCCESS" : "FAILED");


    // Verify sum using reduce
    float local_sum = 0;
    for (int i = 0; i < count; i++) {
        local_sum += h_c[i];
    }

    //
    float global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        float nf = n;
        if (fabsf(global_sum - 3.0 * nf * (nf - 1) / 2) < 0.001) {
            printf("global sum check FAILED: %f\n", global_sum);
        }
        else {
            printf("global sum check SUCEEDED: %f\n", global_sum);
        }
    }

    // Cleanup
    CHK(cudaEventDestroy(start));
    CHK(cudaEventDestroy(stop));
    CHK(cudaFree(d_a));
    CHK(cudaFree(d_b));
    CHK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    MPI_Finalize();
    return 0;
}
