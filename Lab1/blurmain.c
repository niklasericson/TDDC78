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


int main (int argc, char ** argv) {
   int radius;
    int xsize, ysize, colmax;
    pixel src[MAX_PIXELS];
    struct timespec stime, etime;
#define MAX_RAD 1000

    MPI_Comm com = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);

    int myid, np, lsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(com, &np);

    MPI_Status status;

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
             exit(1);    
        }

        if (colmax > 255) {
            fprintf(stderr, "Too large maximum color-component value\n");
            exit(1);
        }
        printf("Has read the image, generating coefficients\n");
        
        get_gauss_weights(radius, w);
    }

    /* filter */


    //printf("Calling filter\n");

    //clock_gettime(CLOCK_REALTIME, &stime);

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

    int size[1];
    size[0] = xsize;
    size[1] = ysize;

    MPI_Bcast(size, 2, MPI_INT, 0, com);

    xsize = size[0];
    ysize = size[1];

    // lsize = Number of pixels for every process.
    int rows_per_core = ysize/(np - 1);

    lsize = rows_per_core*xsize;

    printf("lsize: %d\n", lsize);
    printf("Number of pixels: %d\n", xsize*ysize);

    pixel* local;

    local = (pixel *)malloc((lsize + 2*radius)*sizeof(pixel));


    // local + radius
    MPI_Scatter(src, lsize*sizeof(pixel), MPI_CHAR, local,
                lsize*sizeof(pixel), MPI_CHAR,0,com);

    printf("ID: %d\n",myid);

    if (myid == 0) {
        if (write_ppm ("im1procces0.ppm", xsize, ysize/np, (char *) local) != 0) {
            printf("Write failed in scatter check\n");
        }
    }
    else if (myid == 1) {
        if (write_ppm ("im1procces1.ppm", xsize, ysize/np, (char *) local) != 0) {
            printf("Write failed in scatter check\n");
        }
    }
    else if (myid == 2) {
        if (write_ppm ("im1procces2.ppm", xsize, ysize/np, (char *) local) != 0) {
            printf("Write failed in scatter check\n");
        }
    }
    else if (myid == 3) {
        if (write_ppm ("im1procces3.ppm", xsize, ysize/np, (char *) local) != 0) {
            printf("Write failed in scatter check\n");
        }
    }

    //splitblur(xsize, ysize, src, radius, w);

    //clock_gettime(CLOCK_REALTIME, &etime);

    #ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
        //clock_serv_t cclock;
        //mach_timespec_t mts;
        host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
        clock_get_time(cclock, &mts);
        mach_port_deallocate(mach_task_self(), cclock);
        etime.tv_sec = mts.tv_sec;
        etime.tv_nsec = mts.tv_nsec;

    #else
        clock_gettime(CLOCK_REALTIME, &etime);
    #endif

    printf("Filtering took: %g secs\n", (etime.tv_sec  - stime.tv_sec) +
	   1e-9*(etime.tv_nsec  - stime.tv_nsec)) ;

    /* write result */
    printf("Writing output file\n");
    
    if(write_ppm (argv[3], xsize, ysize, (char *)src) != 0) {
      printf("Write failed writing src\n");
      exit(1);
    }

    MPI_Finalize();
    return(0);
}
