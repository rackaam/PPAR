#include <stdio.h>
#include <omp.h>

#define NB 5 // Nombre de philosophes (et de fourchettes)

omp_lock_t fourchettes[NB]; // Les fourchettes sont representées par des locks

void penser(int num){
	printf("%d pense\n", num);
	sleep(rand() % 2);
}

void manger(int num){
	printf("%d mange\n", num);
	sleep(rand() % 3);
}

omp_lock_t* prendreFourchetteG(int i){
	// printf("%d gauche\n", i);
	return &fourchettes[(i - 1 + NB) % NB];
}

omp_lock_t* prendreFourchetteD(int i) {
	// printf("%d droite\n", i);
	return &fourchettes[i];
}

int main(){
	// Autant de thread que de philosophe
	omp_set_num_threads(NB);
	#pragma omp parallel
	{
		// Chaque philosophe est identifié par le numéro de son thread
		// Les philosophes ont donc une place fixe qui leur est assigné.
		int i, num = omp_get_thread_num();
		// Chaque philosophe mange 10x
		for(i = 0; i < 10; i++)
		{
			penser(num);
			printf("%d veut manger\n", num);
			// Tous les philosophes prennent les fourchettes dans le même sens sauf 1 (pour éviter un blocage)
			if (num % 2 == 1) 
			{
				// Suspend l'execution du thread tant que le lock n'est pas pris
				 omp_set_lock(prendreFourchetteG(num));
				 omp_set_lock(prendreFourchetteD(num));
			} 
			else 
			{
				 omp_set_lock(prendreFourchetteD(num));
				 omp_set_lock(prendreFourchetteG(num));
			} 
			manger(num);
			printf("%d libere sa place\n", num);
			// Libère les locks, permet à certains threads bloqués de continuer leur execution
			omp_unset_lock(prendreFourchetteG(num));
	 		omp_unset_lock(prendreFourchetteD(num));
		}
	}

	return 0;
}