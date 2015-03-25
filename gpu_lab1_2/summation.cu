#include "utils.h"
#include <stdlib.h>

struct results
{
	float sum;
};

#include "summation_kernel.cu"

// CPU implementation
float log2_series(int n)
{
	int i;
	float sum = 0;
	for (i = 0; i < n; i++) {
		if (i % 2 == 0)
			sum += 1.0 / (i + 1);
		else
			sum -= 1.0 / (i + 1);
	}
	return sum;
}

float reverse_log2_series(int n)
{
	int i;
	float sum = 0;
	for (i = n-1; i >= 0; i--) {
		if (i % 2 == 0)
			sum += 1.0 / (i + 1);
		else
			sum -= 1.0 / (i + 1);
	}
	return sum;
}

int main(int argc, char ** argv)
{
    int data_size = 1024 * 1024 * 128, i;

    // Run CPU version
    double start_time = getclock();
    float log2 = log2_series(data_size);
    double end_time = getclock();

    double r_start_time = getclock();
    float r_log2 = reverse_log2_series(data_size);
    double r_end_time = getclock();
    
    printf(" log(2)=\n%.15f\n", log(2.0));
    printf("CPU result:\n%.15f\n", log2);
    printf("CPU result (reverse):\n%.15f\n", r_log2);
    printf(" time=%fs\n", end_time - start_time);
    printf(" time=%fs (reverse)\n", r_end_time - r_start_time);
    
    // Parameter definition
    int threads_per_block = 16 * 32;
    int blocks_in_grid = 8;
    
    int num_threads = threads_per_block * blocks_in_grid;

    // Timer initialization
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

	// Allocating output data on GPU
    float *d_C;
	cudaMalloc((void **)&d_C, num_threads * sizeof(float));

    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // Execute kernel
    summation_kernel<<<blocks_in_grid,threads_per_block>>>(data_size, d_C);

    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    // Get results back
    float* res = (float*)malloc(num_threads * sizeof(float));
	cudaMemcpy(res, d_C, num_threads * sizeof(float), cudaMemcpyDeviceToHost);
	
    // Finish reduction
	float sum = 0.;
	for(i = 0; i < num_threads; i ++) {
		sum += res[i];
	}
    
    // Cleanup
	cudaFree(d_C);
	free(res);
    
    printf("GPU results:\n");
    printf(" Sum: %.15f\n", sum);
    
    float elapsedTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	// In ms

    double total_time = elapsedTime / 1000.;	// s
    double time_per_iter = total_time / (double)data_size;
    double bandwidth = sizeof(float) / time_per_iter; // B/s
    
    printf(" Total time: %g s,\n Per iteration: %g ns\n Throughput: %g GB/s\n",
    	total_time,
    	time_per_iter * 1.e9,
    	bandwidth / 1.e9);
  
    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));
    return 0;
}

