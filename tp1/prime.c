#include <stdio.h>
#include <omp.h>

#define N 1200000
#define N_ROOT 1096

int main()
{
	int i,j=2, nb_prime = 0;


	int tab[N];

	for(i=0;i<N;i++){
		tab[i]=1;
	}

	i = 2;
	while(i < N_ROOT)
	{
		if(tab[i] == 1)
		{
			nb_prime++;
			for(j=2; j * i < N; j++)
			{
				tab[i*j]= 0;
			}	
		}
		i++;
	}

	printf("Total : %d\n", nb_prime);
}

	
	


