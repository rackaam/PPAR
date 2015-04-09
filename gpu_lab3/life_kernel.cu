

#define EMPTY	0
#define RED		1
#define BLUE	2

#define CENTER			0
#define TOP_LEFT		1
#define TOP				2
#define TOP_RIGHT		3
#define RIGHT			4
#define BOTTOM_RIGHT	5
#define	BOTTOM			6
#define BOTTOM_LEFT		7
#define LEFT			8


__global__ void init_kernel(int * domain, int domain_x)
{
	// Dummy initialization
	domain[blockIdx.y * domain_x + blockIdx.x * blockDim.x + threadIdx.x]
		= (1664525ul * (blockIdx.x + threadIdx.y + threadIdx.x) + 1013904223ul) % 3;
}

// Reads a cell at (x+dx, y+dy)
__device__ int read_cell(int * source_domain, int x, int y, int dx, int dy,
    unsigned int domain_x, unsigned int domain_y)
{
    x = (unsigned int)(x + dx) % domain_x;	// Wrap around
    y = (unsigned int)(y + dy) % domain_y;
    return source_domain[y * domain_x + x];
}

__device__ int read_shared_cell(int * sdata, int x, int y, int dx, int dy, int sdataW)
{
    x = (unsigned int)(x + dx + 1);
    y = (unsigned int)(y + dy + 1);
    return sdata[y * sdataW + x];
}

// Write value in a cell at (x+dx, y+dy)
__device__ void write_cell(int * dest_domain, int x, int y, int dx, int dy, int value,
    unsigned int domain_x, unsigned int domain_y)
{
    x = (unsigned int)(x + dx) % domain_x;	// Wrap around
    y = (unsigned int)(y + dy) % domain_y;
    dest_domain[y * domain_x + x]=  value;
}


// Compute kernel
__global__ void life_kernel(int * source_domain, int * dest_domain,
    int domain_x, int domain_y)
{
	extern __shared__ int sdata[];
	int sdataw = blockDim.x + 2;
	int sdatah = blockDim.y + 2;

	int i;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
	
	int sdataIdx = (threadIdx.y+1) * sdataw + (threadIdx.x+1);
	sdata[sdataIdx] = read_cell(source_domain, tx, ty, 0, 0, domain_x, domain_y);
	if(threadIdx.y == 0){
		// Bordure supérieure
		sdata[threadIdx.x + 1] = read_cell(source_domain, tx, ty, 0, 1, domain_x, domain_y);
	}else if(threadIdx.y == blockDim.y - 1) {
		// Bordure inférieure
		sdata[(sdatah-1) * sdataw + threadIdx.x + 1] = read_cell(source_domain, tx, ty, 0, -1, domain_x, domain_y);
	}else if(threadIdx.x == 0) {
		// Bordure gauche
		sdata[(threadIdx.y+1) * sdataw] =  read_cell(source_domain, tx, ty, -1, 0, domain_x, domain_y);
	}else if(threadIdx.x == blockDim.x - 1) {
		// Bordure droite
		sdata[(threadIdx.y+1) * sdataw + sdataw-1] =  read_cell(source_domain, tx, ty, 1, 0, domain_x, domain_y);
	}
	if(threadIdx.x == 0 && threadIdx.y == 0){
		sdata[0] =  read_cell(source_domain, tx, ty, -1, 1, domain_x, domain_y);
	}
	if(threadIdx.x == blockDim.x-1 && threadIdx.y == 0){
		sdata[sdataw] =  read_cell(source_domain, tx, ty, 1, 1, domain_x, domain_y);
	}
	if(threadIdx.x == 0 && threadIdx.y == blockDim.y-1){
		sdata[(sdatah-1)*sdataw] =  read_cell(source_domain, tx, ty, -1, -1, domain_x, domain_y);
	}
	if(threadIdx.x == blockDim.x-1 && threadIdx.y == blockDim.y-1){
		sdata[(sdatah-1)*sdataw + sdataw-1] =  read_cell(source_domain, tx, ty, 1, -1, domain_x, domain_y);
	}
    
    // Read cell
	int cells[9];
    cells[CENTER] = read_shared_cell(sdata, tx, ty, 0, 0, sdataw);
    cells[TOP] = read_shared_cell(sdata, tx, ty, 0, 1, sdataw); 
    cells[TOP_LEFT] = read_shared_cell(sdata, tx, ty, -1, 1, sdataw);
    cells[TOP_RIGHT] = read_shared_cell(sdata, tx, ty, 1, 1, sdataw);
	cells[LEFT] = read_shared_cell(sdata, tx, ty, -1, 0, sdataw);
	cells[RIGHT] = read_shared_cell(sdata, tx, ty, 1, 0, sdataw);
	cells[BOTTOM] = read_shared_cell(sdata, tx, ty, 0, -1, sdataw);
	cells[BOTTOM_LEFT] = read_shared_cell(sdata, tx, ty, -1, -1, sdataw);
	cells[BOTTOM_RIGHT] = read_shared_cell(sdata, tx, ty, 1, -1, sdataw);

	int neighbors = 0;
	for(i = 1; i < 9; i++)
		if(cells[i] != 0)
			neighbors++;

	if(cells[CENTER] != EMPTY) {
		if(neighbors < 2 || neighbors > 3) {
			write_cell(dest_domain, tx, ty, 0, 0, EMPTY, domain_x, domain_y);
		}
	} else if(neighbors == 3) {	
		int redCount = 0, blueCount = 0;
		for(i = 1; i < 9; i++)
			if(cells[i] == BLUE)
				blueCount++;
			else if(cells[i] == RED)
				redCount++;
		int newColor = redCount > blueCount ? RED : BLUE;
		write_cell(dest_domain, tx, ty, 0, 0, newColor, domain_x, domain_y);
	}
}

