#include <stdio.h>
#include <omp.h>

int main()
{
	int me = 0;
	int nb = 4;
	omp_set_num_threads(4);

	#pragma omp parallel private(nb, me)
	{
		nb = omp_get_num_threads();
		me = omp_get_thread_num();

		printf("Hello from thread %d\n",me);

		if(me == 0)
		{
			printf("Total=%d threads\n", nb);
		}
	}
}