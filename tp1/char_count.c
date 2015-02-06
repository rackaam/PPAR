#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <time.h>

#define N 750
#define OFFSET 32

int main()
{
	omp_set_num_threads(4);
	char tab[N];
	int i, vowels = 0, consonants = 0;
  	srand (time(NULL));

	for(i = 0; i < N; i++)
	{
			tab[i] = rand() % 59 + OFFSET;
	}
	
	int amount[26] = {0};
	for(i = 0; i < N; i++)
	{
		if(tab[i] > 64) {
			amount[tab[i] - 64]++;
			
			if ( tab[i] == 'A' | tab[i] == 'E' | tab[i] == 'I' 
				| tab[i] == 'O' | tab[i] == 'U' | tab[i] =='Y') {
				vowels++;
			} else {
				consonants++;
			}
		}
	}

	for(i = 0; i < 26; i++)
	{
		printf("%c:%d\n", i + 65, amount[i]);
	}

	printf("Cons:%d\nVoy:%d\n", consonants, vowels);
}