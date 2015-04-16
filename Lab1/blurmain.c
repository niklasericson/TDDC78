#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "ppmio.h"
#include "blurfilter.h"
#include "gaussw.h"

#ifdef __MACH__
    #include <mach/clock.h>
    #include <mach/mach.h>
#endif

#define MAX_RAD 1000

int main (int argc, char ** argv) {
    
    /* Definitions */

    int radius;
    int xsize, ysize, colmax, lsize_last_core, rows_per_core, remaining_rows;
    int size[1], myid, np, lsize;
    pixel src[MAX_PIXELS];
    struct timespec stime, etime;


    MPI_Comm com, scatter_com;

    com = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(com, &np);

    MPI_Status status;

    /* Putting all processes in scatter_com exepct last. This is so that scatter
        will scatter the image evenly even if ysize/np isn't an integer. */

    int color;
    int key;

    color = (myid == (np - 1));
    key = myid;

    MPI_Comm_split(com, color, key, &scatter_com);

    double w[MAX_RAD];

    /* Take care of the arguments */

    if (argc != 4) {
        fprintf(stderr, "Usage: %s radius infile outfile\n", argv[0]);
        exit(1);
    }
    radius = atoi(argv[1]);
    
    if((radius > MAX_RAD) || (radius < 1)) {
        fprintf(stderr, "Radius (%d) must be greater than zero and less then %d\n", radius, MAX_RAD);
        exit(1);
    }

    if (myid == 0) {
    /* read file */
        if(read_ppm (argv[2], &xsize, &ysize, &colmax, (char *) src) != 0) {
             exit(2);
        }

        if (colmax > 255) {
            fprintf(stderr, "Too large maximum color-component value\n");
            exit(1);
        }
        printf("Has read the image, generating coefficients\n");
        
        get_gauss_weights(radius, w);
    }

    /* filter */

    if (myid == 0) {
        printf("Calling filter\n");
    }

    #ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
        clock_serv_t cclock;
        mach_timespec_t mts;
        host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
        clock_get_time(cclock, &mts);
        mach_port_deallocate(mach_task_self(), cclock);
        stime.tv_sec = mts.tv_sec;
        stime.tv_nsec = mts.tv_nsec;

    #else
        clock_gettime(CLOCK_REALTIME, &stime);
    #endif

    // Create array of xsize and ysize and broadcast to all cores.

    size[0] = xsize;
    size[1] = ysize;

    MPI_Bcast(size, 2, MPI_INT, 0, com);

    xsize = size[0];
    ysize = size[1];


    /* lsize = Number of rows for every process execpt np - 1, wich will contain
        the remaining rows (since ysize/(np - 1) might not be whole number. */
    rows_per_core = ysize/np;

    /* Reamining rows for last core to process
        (Performance anlysis: remaining_rows < rows_per_core + np) */
    remaining_rows = ysize % np + rows_per_core;

    lsize = rows_per_core*xsize;
    lsize_last_core = remaining_rows*xsize;

    register int last_start_pos = lsize*(np - 1);

    pixel* local;

    local = (pixel *)malloc((lsize_last_core + 2*radius)*sizeof(pixel));

    MPI_Scatter(src, lsize*sizeof(pixel), MPI_CHAR, local,
                lsize*sizeof(pixel), MPI_CHAR,0,scatter_com);

    if (myid == 0) {
        MPI_Send(src + last_start_pos, lsize_last_core*sizeof(pixel),
                    MPI_CHAR, np-1, 10, com);
    }
    else if (myid == (np - 1)) {
        MPI_Recv(local, lsize_last_core*sizeof(pixel), 
                    MPI_CHAR, 0, 10, com, &status);
    }

    /* Filtering goes here */
    //splitblur(xsize, ysize, src, radius, w);


    MPI_Gather(local, lsize*sizeof(pixel), MPI_CHAR, src, lsize*sizeof(pixel),
                MPI_CHAR, 0, scatter_com);

    if (myid == 0) {
        MPI_Recv(src + last_start_pos, lsize_last_core*sizeof(pixel), 
                    MPI_CHAR, np-1, 20, com, &status);
    }
    else if (myid == np-1) {
        MPI_Send(local, lsize_last_core*sizeof(pixel), 
                    MPI_CHAR, 0, 20, com);
    }


    #ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
        host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
        clock_get_time(cclock, &mts);
        mach_port_deallocate(mach_task_self(), cclock);
        etime.tv_sec = mts.tv_sec;
        etime.tv_nsec = mts.tv_nsec;
    #else
        clock_gettime(CLOCK_REALTIME, &etime);
    #endif


    /* write result */
    
    if (myid == 0) {
        printf("Filtering took: %g secs\n", (etime.tv_sec  - stime.tv_sec) +
	           1e-9*(etime.tv_nsec  - stime.tv_nsec)) ;
        
        printf("Writing output file\n");
        
        if(write_ppm (argv[3], xsize, ysize, (char *)src) != 0) {
            printf("Write failed writing src\n");
            exit(1);
        }
    }

    MPI_Finalize();
    return(0);
}
