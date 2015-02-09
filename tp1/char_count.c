#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <time.h>

#define N 20
#define OFFSET 32

int main()
{
	omp_set_num_threads(4);
	const char* tab[N];
	int i, j,vowels = 0, consonants = 0;
	int amount[26] = {0};
  	srand (time(NULL));

  	/* Genere N mots contenant des lettres, chiffres et signes de ponctuations */
  	#pragma omp parallel shared(tab) private(i, j)
	{
		// Partage l'initialisation du tableau de mots entre les 4 threads
		#pragma omp for schedule(static) 
		for(i = 0; i < N; i++)
		{	
				char *randomString = NULL;
				int length = rand() % 22 + 1;
				randomString = malloc(sizeof(char) * length);
				for(j = 0; j < length - 1; j++)
				{
					// Valeurs ASCII de ESPACE a Z
					randomString[j] = rand() % 59 + OFFSET; 
				}
				randomString[length - 1] = '\0';
				tab[i] = randomString;
		}
	}

	#pragma omp parallel shared(tab, amount, vowels, consonants) private(i, j)
	{
		// Chacun des treads traite 1/4 des mots du tableau
		#pragma omp for schedule(static) 
		for(i = 0; i < N; i++)
		{
			const char* word = tab[i];
			j = 0;
			do {
				if(word[j] > 64) // Si c'est une lettre 
				{
					// La modification d'une variable partagée est effectuée dans
					// une section critique
					#pragma omp critical
					{
						amount[word[j] - 65]++;// Incremente le compteur correspondant a la lettre
					}
					
					if ( word[j] == 'A' | word[j] == 'E' | word[j] == 'I' 
					   | word[j] == 'O' | word[j] == 'U' | word[j] == 'Y')
					{
						// L'incrémentation peut être effectuée de façon atomique
						#pragma omp atomic
							vowels++;
					} 
					else 
					{
						#pragma omp atomic
							consonants++;
					}
				}
			} while(word[j++] != '\0');		
		}
	}

	for(i = 0; i < 26; i++)
	{
		printf("%c:%d\n", i + 65, amount[i]);
	}

	printf("Cons:%d\nVoy:%d\nTotal:%d\n", consonants, vowels, consonants + vowels);

	return 0;
}