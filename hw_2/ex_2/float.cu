#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DataType float  // Changed from double to float

// Kernel function to compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows, int numAColumns, int numBRows, int numBColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < numARows && col < numBColumns) {
        DataType sum = 0;
        for (int i = 0; i < numAColumns; i++) {
            sum += A[row * numAColumns + i] * B[i * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}

int main(int argc, char **argv) {
    DataType *hostA; // The A matrix
    DataType *hostB; // The B matrix
    DataType *hostC; // The output C matrix
    DataType *deviceA;
    DataType *deviceB;
    DataType *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B

    // Reading matrix dimensions from arguments
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <numARows> <numAColumns> <numBRows> <numBColumns>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    numARows = atoi(argv[1]);
    numAColumns = atoi(argv[2]);
    numBRows = atoi(argv[3]);
    numBColumns = atoi(argv[4]);

    if (numAColumns != numBRows) {
        fprintf(stderr, "Error: number of columns in A must equal number of rows in B\n");
        exit(EXIT_FAILURE);
    }

    int numCRows = numARows;
    int numCColumns = numBColumns;

    // Allocate Host memory
    size_t sizeA = numARows * numAColumns * sizeof(DataType);
    size_t sizeB = numBRows * numBColumns * sizeof(DataType);
    size_t sizeC = numCRows * numCColumns * sizeof(DataType);
    hostA = (DataType *)malloc(sizeA);
    hostB = (DataType *)malloc(sizeB);
    hostC = (DataType *)malloc(sizeC);

    // Initialize matrices A and B with random numbers
    srand(0);
    for (int i = 0; i < numARows * numAColumns; i++) {
        hostA[i] = static_cast<DataType>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < numBRows * numBColumns; i++) {
        hostB[i] = static_cast<DataType>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    cudaMalloc((void **)&deviceA, sizeA);
    cudaMalloc((void **)&deviceB, sizeB);
    cudaMalloc((void **)&deviceC, sizeC);

    // Create CUDA events for timing
    cudaEvent_t start, stop, startKernel, stopKernel;
    float timeCopyToDevice, timeKernel, timeCopyToHost;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);

    // Start the timer for memory copy to device
    cudaEventRecord(start);

    // Copy memory to the GPU
    cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);

    // Stop the timer for memory copy to device and start the kernel timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeCopyToDevice, start, stop);
    cudaEventRecord(startKernel);

    // Initialize the grid and block dimensions
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((numBColumns + blockDim.x - 1) / blockDim.x, (numARows + blockDim.y - 1) / blockDim.y, 1);

    // Launch the GPU Kernel
    gemm<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);

    // Stop the kernel timer and start the memory copy to host timer
    cudaEventRecord(stopKernel);
    cudaEventSynchronize(stopKernel);
    cudaEventElapsedTime(&timeKernel, startKernel, stopKernel);
    cudaEventRecord(start);

    // Copy the GPU memory back to the CPU
    cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

    // Stop the timer for memory copy to host
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeCopyToHost, start, stop);

    // Print timing information
    printf("Time to copy data from host to device: %f ms\n", timeCopyToDevice);
    printf("Time for CUDA kernel execution: %f ms\n", timeKernel);
    printf("Time to copy data from device to host: %f ms\n", timeCopyToHost);

    // Free the GPU memory
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    // Free the CPU memory
    free(hostA);
    free(hostB);
    free(hostC);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);

    return 0;
}

