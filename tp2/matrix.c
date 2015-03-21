#include <stdio.h>
#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>

#define N 4

int main(int argc, char ** argv)
{
	int rank,n,i,j;
	int a[N][N];
	int recv1[N][N];
	int recv2[N][N];

	srand (time(NULL));

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n);
	
	if (rank == 0)
	{	
		printf("Matrice avant transpo :\n");
		// Génère une matrice aléatoire
		for(i = 0; i < N; i++)
		{
			for(j = 0; j < N; j++)
			{
				a[i][j] = rand() % 10;
			}
		}

		// Affiche Matrice avant modif
		for(i = 0; i < N; i++)
		{
			for(j = 0; j < N; j++)
			{
				printf("%d ", a[i][j]);
			}
		 	printf("\n");
		}
	} 

	MPI_Scatter(a, N*N/n, MPI_INT, recv1, N*N/n, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Alltoall(recv1,1,MPI_INT,recv2,1,MPI_INT, MPI_COMM_WORLD);

	MPI_Gather(recv2, N*N/n, MPI_INT, a, N*N/n, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		printf("\nMatrice après transpo :\n");
		// Affiche la matrice modifié
		for(i = 0; i < N; i++)
		{
			for(j = 0; j < N; j++)
			{
				printf("%d ", a[i][j]);
			}
		 	printf("\n");
		}
	} 

	MPI_Finalize();
	return 0;
}
