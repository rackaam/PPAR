#include <stdio.h>
#include <omp.h>
#include <sys/time.h>

#define N 750

int main()
{
	omp_set_num_threads(4);
	int me = 0;

	
	// float a[N][N] = {
	// 	{1, 2, 3, 4},
	// 	{2,3,3,4},
	// 	{5,5,6,7},
	// 	{1,5,9,7}
	// };


	// float b[N][N] = {
	// 	{7,8,5,9},
	// 	{4,8,6,4},
	// 	{1,9,0,6},
	// 	{4,8,6,8}
	// };
	
	float a[N][N], b[N][N];

	float res[N][N];

    float temp;
	int i, j, h;

	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			a[i][j] = i * j;
			b[i][j] = i + j;
		}
	}

	struct timeval tv0, tv1;
	gettimeofday(&tv0, 0);

	// Version séquentielle
	// for(i = 0; i < N; i++)
	// {
	// 	for(j = 0; j < N; j++)
	// 	{
	// 		temp = 0;
	// 		for(h = 0; h < N; h++)
	// 		{
	// 			temp += a[i][h] * b[h][j];
	// 		}
	// 		res[i][j] = temp;
	// 	}		
	// }

	
	// Version 1
	// #pragma omp parallel shared(a, b, res) private(j, h, temp, me)
	// {
	// 	me = omp_get_thread_num();
	// 	// printf("me=%d\n", me);
	// 	for(j = 0; j < N; j++)
	// 	{
	// 		temp = 0;
	// 		for(h = 0; h < N; h++)
	// 		{
	// 			temp += a[me][h] * b[h][j];
	// 		}
	// 		res[me][j] = temp;
	// 		// printf("temp=%f\n", temp);
	// 	}
	// }


	//Version 2
	#pragma omp parallel shared(a, b, res) private(i, j, h, temp, me)
	{
		#pragma omp for schedule(static) 
		for(i = 0; i < N; i++)
		{
			for(j = 0; j < N; j++)
			{
				temp = 0;
				for(h = 0; h < N; h++)
				{
					temp += a[i][h] * b[h][j];
				}
				res[i][j] = temp;
			}
		}
	}

	gettimeofday(&tv1, 0);
	double elapsed = (double)((tv1.tv_sec-tv0.tv_sec)*1000000 +
	tv1.tv_usec-tv0.tv_usec) / 1000000;
	printf("Elapsed time: %.7f\n", elapsed);


	// Printf
	// for(i = 0; i < N; i++)
	// {
	// 	for(j = 0; j < N; j++)
	// 	{
	// 		printf("%f ", res[i][j]);
	// 	}
	// 	printf("\n");
	// }
	printf("Termine\n");
}

