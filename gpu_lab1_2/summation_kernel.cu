
// GPU kernel
__global__ void summation_kernel(int data_size, float * data_out)
{
	int nbThread = gridDim.x * blockDim.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	float sum = 0;
	for(i = data_size - nbThread + idx; i >= idx; i -= nbThread) {
		if (i % 2 == 0)
			sum += 1.0 / (i + 1);
		else
			sum -= 1.0 / (i + 1);
	}
	data_out[idx] = sum;
}

// GPU kernel
__global__ void summation_kernel2(int data_size, float * data_out)
{
	int nbThread = gridDim.x * blockDim.x;
	int nbElts = data_size / nbThread;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int firstElt = idx * nbElts;
	int lastElt = (idx + 1) * nbElts;
	int i;
	float sum = 0;
	for(i = firstElt; i < lastElt; i++) {
		if (i % 2 == 0)
			sum += 1.0 / (i + 1);
		else
			sum -= 1.0 / (i + 1);
	}
	data_out[idx] = sum;
}


