
//data_out : tableau avec une case pour le résultat de chacun des threads
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

//data_out : tableau avec une case pour le résultat de chacun des threads
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

//data_out : tableau avec une case pour le résultat de chacun des blocks
__global__ void reduced_summation_kernel(int data_size, float * data_out)
{
	extern __shared__ float sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	// Summation
	int nbThread = gridDim.x * blockDim.x;
	int nbElts = data_size / nbThread;
	int firstElt = id * nbElts;
	int lastElt = (id + 1) * nbElts;
	int i;
	float sum = 0;
	for(i = firstElt; i < lastElt; i++) {
		if (i % 2 == 0)
			sum += 1.0 / (i + 1);
		else
			sum -= 1.0 / (i + 1);
	}
	sdata[tid] = sum;

	// Attente des résulatts de chaque thread dans sdata
	__syncthreads();

	// Reduction
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if(index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		// Attente de la fin des calculs avant de passer au niveau de réduction suivant
		__syncthreads();
	}

	// Quand la réduction est terminé, le résultat se trouve dans la première case de la mémoire partagée
	// Un thread par block écrit le résultat
	if (tid == 0) 
		data_out[blockIdx.x] = sdata[0];
}