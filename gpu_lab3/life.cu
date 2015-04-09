
#include "utils.h"
#include <stdlib.h>

#include "life_kernel.cu"


int main(int argc, char ** argv)
{
    // Definition of parameters
    int domain_x = 128;	// Multiple of threads_per_block * cells_per_word
    int domain_y = 128;
    
    int cells_per_word = 1;
    
    int steps = 2;
    
	int threads_x = 16;
	int threads_y = 16;
    int threads_per_block = threads_x * threads_y;
    int blocks_x = domain_x / (threads_x * cells_per_word);
    int blocks_y = domain_y / (threads_y * cells_per_word);
    
    dim3  grid(blocks_x, blocks_y);	// CUDA grid dimensions
	dim3  gridInit(1, domain_y);
    dim3  threads(threads_x, threads_y); // CUDA block dimensions
	dim3  threadsInit(128);

    // Allocation of arrays
    int * domain_gpu[2] = {NULL, NULL};

	// Arrays of dimensions domain.x * domain.y
	size_t domain_size = domain_x * domain_y / cells_per_word * sizeof(int);
	CUDA_SAFE_CALL(cudaMalloc((void**)&domain_gpu[0], domain_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&domain_gpu[1], domain_size));

	init_kernel<<< gridInit, threadsInit, 0 >>>(domain_gpu[0], domain_x);

    // Timer initialization
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // Kernel execution
    int shared_mem_size = (threads_x + 2) * (threads_y + 2) * sizeof(int); // cellules "actives" + cellules "bordures"
    for(int i = 0; i < steps; i++) {
	    life_kernel<<< grid, threads, shared_mem_size >>>(domain_gpu[i%2],
	    	domain_gpu[(i+1)%2], domain_x, domain_y);
	}

    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    
    float elapsedTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	// In ms
    printf("GPU time: %f ms\n", elapsedTime);

    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));

    // Get results back
    int * domain_cpu = (int*)malloc(domain_size);
    CUDA_SAFE_CALL(cudaMemcpy(domain_cpu, domain_gpu[steps%2], domain_size, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(domain_gpu[0]));
    CUDA_SAFE_CALL(cudaFree(domain_gpu[1]));

    // Count colors
    int red = 0;
    int blue = 0;
    for(int y = 0; y < domain_y; y++)
    {
    	for(int x = 0; x < domain_x; x++)
    	{
    		int cell = domain_cpu[y * domain_x + x];
    		//printf("%u", cell);
    		if(cell == 1) {
    			red++;
    		}
    		else if(cell == 2) {
    			blue++;
    		}
    	}
    	//printf("\n");
    }

    printf("Red/Blue cells: %d/%d\n", red, blue);
    
    free(domain_cpu);
  
    return 0;
}

