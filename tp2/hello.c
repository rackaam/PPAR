#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

int main(int argc, char ** argv)
{
	int rank,n;
	char hostname[128];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n);

	gethostname(hostname,128);
	
	if (rank == 0)
	{
		printf("I am the master: %s\n",hostname);
	} else {
		printf("I am a worker: %s (rank%d/%d)\n",hostname,rank,n-1);
	}

	MPI_Finalize();
	return 0;
}
