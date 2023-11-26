#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1024
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}

int main(){
    float *a, *b, *out;
    float *d_a, *d_b, *d_out; 
    cudaEvent_t start, stop;
    float time;

    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory 
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start recording the time for host to device data transfer
    cudaEventRecord(start, 0);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Stop recording the time and calculate the elapsed time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for host to device data transfer: %f ms\n", time);

    // Executing kernel 
    int block_size = 256;
    int grid_size = ((N + block_size - 1) / block_size);

    // Start recording the time for kernel execution
    cudaEventRecord(start, 0);

    vector_add<<<grid_size,block_size>>>(d_out, d_a, d_b, N);

    // Stop recording the time and calculate the elapsed time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for CUDA kernel execution: %f ms\n", time);

    // Start recording the time for device to host data transfer
    cudaEventRecord(start, 0);

    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Stop recording the time and calculate the elapsed time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for device to host data transfer: %f ms\n", time);

    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


