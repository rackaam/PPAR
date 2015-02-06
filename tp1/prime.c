#include <stdio.h>
#include <omp.h>

#define N 100000
#define N_ROOT 1000
#define NB_THREAD 4

int main()
{
	omp_set_num_threads(NB_THREAD);
	int i,j=2, nb_prime = 0;


	int tab[N];

	#pragma omp parallel shared(tab) private(i)
	{
		#pragma omp for schedule(static)
		for(i=0; i<N; i++){
			tab[i]=1;
		}
	}

	i = 2;

	#pragma omp parallel shared(tab, i) private(j)
	{
		int nb = omp_get_thread_num();
		int h;
		while((h = i + nb) < N_ROOT)
		{
			if(tab[h] == 1)
			{
				for(j=2; j * h < N; j++)
				{
					tab[h*j]= 0;	
				}
			}
			#pragma omp master
			{
				i += omp_get_num_threads();
			}
		}
	}

#pragma omp parallel shared(tab) private(i)
	{
		#pragma omp for schedule(static)
		for(i=2; i<N; i++)
		{
			if(tab[i] == 1)
			{
				#pragma omp critical 
				{
					nb_prime++;
				}
			}
		}	
	}

	printf("Total : %d\n", nb_prime);
}

	
	


