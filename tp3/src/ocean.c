// Matthias Gradaive GL 1.2
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stddef.h>
#include <mpi.h>
#include "ocean.h"

//#define LOG // Enable printf

/* constants for the ocean */
#define N 40 // rows
#define M 20 // columns
#define WALL 100
#define STEP 20 // Nombre de pas de simulation
#define RATIO 10

// TAG pour MPI_Isend/Recv
#define S_FROM_SOUTH 1
#define T_FROM_SOUTH 2
#define S_FROM_NORTH 3
#define T_FROM_NORTH 4

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);
    int i, n, rank;
    fish_t * ocean = NULL;

    MPI_Datatype MPI_FISH = createFishDatatype();

    // Init com
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n);

    if (rank == 0) {
        // Seul le process root initialise un ocean complet
        ocean = (fish_t *)malloc(N * M * sizeof(fish_t));
        init_ocean(ocean, N, M, RATIO);
        printf(CLS "\n");
        display_ocean(ocean, N, M);
    }

    // Chaque process alloue un tableau pour stocker une sous-partie de l'ocean.
    // Les tailles de chacune des sous-parties sont égales : (nombre de colonnes * nombre de lignes) / nombre de process
    fish_t *ocean_part = (fish_t *) malloc(N * M / n * sizeof(fish_t));

    // Le process root envoie une partie de l'ocean à chacun de process (lui y compris)
    MPI_Scatter(ocean, N * M / n, MPI_FISH, ocean_part, N * M / n, MPI_FISH, 0, MPI_COMM_WORLD);

    // Boucle de simulation
    for(i = 0; i < STEP; i++) {
        // Mise a jour des sous-partie de l'ocean et recuperation du nombre de requins et thons
        // a envoyer aux process voisins.
        int ns_north = 0, nt_north = 0, ns_south = 0, nt_south = 0;
        update_ocean_part(ocean_part, N/n, M, &ns_north, &nt_north, &ns_south, &nt_south);

        #ifdef LOG
        printf("%d sends %d:%d:%d:%d\n",rank, ns_north, nt_north, ns_south, nt_south);
        #endif

        int recv_s_from_north = 0, recv_t_from_north = 0, recv_s_from_south = 0, recv_t_from_south = 0;
        MPI_Request request_s_north, request_t_north, request_s_south, request_t_south;
        MPI_Status status_s_north, status_t_north, status_s_south, status_t_south;

        // Calcul de l'identifiant des process voisins
        int northDest = rank - 1;
        if(northDest == -1) northDest = n - 1;
        int southDest = (rank + 1) % n;

        // Envoie asynchrone du nombres de thon/requins a transferer
        // Chaque envoie utilise un TAG different correspondant aux donnees qu'il envoie
        MPI_Isend(&ns_north, 1, MPI_INT, northDest, S_FROM_SOUTH, MPI_COMM_WORLD, &request_s_north);
        MPI_Isend(&nt_north, 1, MPI_INT, northDest, T_FROM_SOUTH, MPI_COMM_WORLD, &request_t_north);
        MPI_Isend(&ns_south, 1, MPI_INT, southDest, S_FROM_NORTH, MPI_COMM_WORLD, &request_s_south);
        MPI_Isend(&nt_south, 1, MPI_INT, southDest, T_FROM_NORTH, MPI_COMM_WORLD, &request_t_south);

        // Reception des donnees envoyees par les process voisins
        MPI_Recv(&recv_s_from_south, 1, MPI_INT, southDest, S_FROM_SOUTH, MPI_COMM_WORLD, &status_s_north);
        MPI_Recv(&recv_t_from_south, 1, MPI_INT, southDest, T_FROM_SOUTH, MPI_COMM_WORLD, &status_t_north);
        MPI_Recv(&recv_s_from_north, 1, MPI_INT, northDest, S_FROM_NORTH, MPI_COMM_WORLD, &status_s_south);
        MPI_Recv(&recv_t_from_north, 1, MPI_INT, northDest, T_FROM_NORTH, MPI_COMM_WORLD, &status_t_south);

        // Attente la fin des transfere de donnees
        MPI_Wait(&request_s_north, &status_s_north);
        MPI_Wait(&request_t_north, &status_t_north);
        MPI_Wait(&request_s_south, &status_s_south);
        MPI_Wait(&request_t_south, &status_t_south);

        // Injection des requins/thons en fonction des donnees recues
        inject_ocean(ocean_part, N/n, M, recv_s_from_north + recv_s_from_south, recv_t_from_north + recv_t_from_south);

        #ifdef LOG
        printf("%d recv %d:%d:%d:%d\n", rank, recv_s_from_south, recv_t_from_south, recv_s_from_north, recv_t_from_north);
        #endif
    }

    // Rassemblement des sous-parties en un seul ocean
    MPI_Gather(ocean_part, N * M / n, MPI_FISH, ocean, N * M / n, MPI_FISH, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        display_ocean(ocean, N, M);
    }

    free(ocean);
    free(ocean_part);
    MPI_Type_free(&MPI_FISH);
    MPI_Finalize();
    return 0;
}

MPI_Datatype createFishDatatype()
{
    MPI_Datatype MPI_FISH;

    // Description of the type fish_t
    int fish_lengths[2];
    fish_lengths[0] = sizeof(char);
    fish_lengths[1] = sizeof(char);
    // Initialization
    MPI_Aint fish_offsets[2];
    fish_offsets[0] = offsetof(fish_t, type);
    fish_offsets[1] = offsetof(fish_t, moved);

    MPI_Datatype fish_types[2] = {MPI_CHAR, MPI_CHAR};

    // Creation of the type MPI_FISH
    MPI_Type_create_struct(2, fish_lengths, fish_offsets, fish_types, &MPI_FISH);
    MPI_Type_commit(&MPI_FISH);
    return MPI_FISH;
}
