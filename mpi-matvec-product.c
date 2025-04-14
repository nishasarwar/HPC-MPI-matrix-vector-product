#include <assert.h>
#include <stdio.h>
#include <mpi.h>
#include <papi.h>
#include <stdlib.h>

// Compute a 1D distribution offset
//
// @param N the total number of rows or columns
// @param P the number of partitions
// @param i the partition id
long offset(long N, int P, int i);

int main(int argc, char* argv[argc])
{
    // Check that we have the right number of parameters.
    if (argc != 4) {
        fprintf(stderr, "Usage: exe [total_rows] [total_columns] [n_samples]\n");
        return EXIT_FAILURE;
    }

    // Read our command line variables.
    long const total_rows = atol(argv[1]);
    long const total_cols = atol(argv[2]);
    int const n_samples = atoi(argv[3]);

    // Initialize our two libraries.
    PAPI_library_init(PAPI_VER_CURRENT);
    MPI_Init(&argc, &argv);

    int n_ranks = 0;
    int rank = -1;

    /// [TODO] get the number of ranks and the current rank using MPI.
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Compute a 1D column distribution. The element in cols[i[ is the minimum
    // row assigned to rank i.
    long cols[n_ranks + 1];
    for (int i = 0; i < n_ranks + 1; ++i) {
        cols[i] = offset(total_cols, n_ranks, i);
    }

    // Figure out which columns in A I own.
    long const my_min_col = cols[rank];
    long const my_max_col = cols[rank + 1];

    // Figure out how many rows and columns of A I own.
    long const my_n_rows = total_rows;
    long const my_n_cols = my_max_col - my_min_col;

    // [TODO] Allocate the local part of A, and initialize it as A[i] = i.
    float *A = NULL;
    A = malloc(sizeof(float) * my_n_rows * my_n_cols);
    assert(A != NULL);
    for (long i = 0; i < my_n_rows * my_n_cols; ++i) {
        A[i] = (float)i;
    }

    // [TODO] Allocate the local part of x, and initialize it as x[i] = i.
    float *x = NULL;
    x = malloc(sizeof(float) * my_n_cols);  // my_n_cols is the number of elements in x this rank will hold
    assert(x != NULL); 
    for (long j = 0; j < my_n_cols; ++j) {
        x[j] = (float)(my_min_col + j);  // Set each element to its global index
    }


    // [TODO] Allocate the local version of y and initialize it with 0 data.
    float *y = NULL;
    y = malloc(sizeof(float) * my_n_rows);
    assert(y != NULL);
    for (long i = 0; i < my_n_rows; ++i) {
        y[i] = 0.0f;
    }

    // Statistic we are accumulating.
    long total_compute_time = 0;
    long total_communicate_time = 0;

    // Print an output header.
    if (rank == 0) {
        printf("sample   type          min_μs    max_μs    avg_μs\n");
    }

    // For each sample.
    for (int n = 0; n < n_samples; ++n)
    {
        // Make sure that everyone has reached this point.
        MPI_Barrier(MPI_COMM_WORLD);

        // Start timing.
        long const start_time = PAPI_get_virt_usec();

        /// [TODO] Perform your local matrix-vector product, Ax = y. Use either
        ///        row major or column major indexing.
        for (long i = 0; i < my_n_rows; ++i) {
            for (long j = 0; j < my_n_cols; ++j) {
                y[i] += A[i * my_n_cols + j] * x[j];
            }
        }

        // End compute timing.
        long const compute_end = PAPI_get_virt_usec();
        long const compute_time = compute_end - start_time;

        // [TODO] Reduce the partial sums of y so that everyone has the total
        //        using MPI.
        MPI_Allreduce(MPI_IN_PLACE, y, my_n_rows, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        // End communication timing.
        long const communicate_end = PAPI_get_virt_usec();
        long const communicate_time = communicate_end - compute_end;

        // The min, max, and average compute time across all of the ranks.
        long min_compute_time = compute_time;
        long max_compute_time = compute_time;
        long avg_compute_time = compute_time;

        // [TODO] Collect the min, max, and average compute time across all of 
        //        the ranks into the above variables using MPI.
        MPI_Allreduce(MPI_IN_PLACE, &min_compute_time, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &max_compute_time, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &avg_compute_time, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

        avg_compute_time /= n_ranks;

        if (rank == 0) {
            printf("%-8dcompute       %-10ld%-10ld%-10ld\n", n, min_compute_time, max_compute_time, avg_compute_time);
        }

        // The min, max, and average communcation time.
        long min_communicate_time = communicate_time;
        long max_communicate_time = communicate_time;
        long avg_communicate_time = communicate_time;

        // [TODO] Collect the min, max, and average communication time across
        //        all of the ranks into the above variables using MPI.
        MPI_Allreduce(MPI_IN_PLACE, &min_communicate_time, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &max_communicate_time, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &avg_communicate_time, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

        avg_communicate_time /= n_ranks;

        if (rank == 0) {
            printf("%-8dcommunicate   %-10ld%-10ld%-10ld\n", n, min_communicate_time, max_communicate_time, avg_communicate_time);
        }

        // Accumulate the total times.
        total_compute_time += compute_time;
        total_communicate_time += communicate_time;
    }

    // [TODO] Sum the total compute and communication time across the 
    //        ranks using MPI.
    MPI_Allreduce(MPI_IN_PLACE, &total_compute_time, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &total_communicate_time, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);


    total_compute_time /= (n_ranks * n_samples);
    total_communicate_time /= (n_ranks * n_samples);

    if (rank == 0) {
        printf("\n");
        printf("Totals:\n");
        printf("average compute μs: %ld\n", total_compute_time);
        printf("average communicate μs: %ld\n\n",  total_communicate_time);
    }

    // Cleanup and exit.
    free(y);
    free(x);
    free(A);

    MPI_Finalize();

    return EXIT_SUCCESS;
}

long offset(long const N, int const ranks, int const rank)
{
    long const d = N / ranks;
    long const r = N % ranks;
    return d * rank + (r < rank ? r : rank);
}