#include "coo_spmv_kernel.h"
#include <omp.h>
#include <stdio.h>
#include <likwid.h>

/*
 * m = number of cols
 * A_cols = col pointers, m+1 of them, each element with the start of the nonZero's in the columns array
 * A_cols_idx = column indices of the nonZero values
 * A_values = the actual values in the nonZero locations
 * B = the vector
 * C = the result
 */
void coo_spmv(int nz, const int *A_rows, const int *A_cols, const float *A_values, const float *B, float *C)
{
    int i;

    for (i = 0; i < nz; i++)
    {
        C[A_rows[i]] += A_values[i] * B[A_cols[i]];
    }
}

void coo_spmv_parallel(int nz, const int *A_rows, const int *A_cols, const float *A_values, const float *B, float *C)
{
    int i;
    omp_set_dynamic(0);
    omp_set_num_threads(32);
    LIKWID_MARKER_INIT;
#pragma omp parallel
    {
        LIKWID_MARKER_THREADINIT;
    }
#pragma omp parallel
    {
        LIKWID_MARKER_START("coo");
#pragma omp for
        for (i = 0; i < nz; i++)
        {
#pragma omp atomic update
            C[A_rows[i]] += A_values[i] * B[A_cols[i]];
        }
        LIKWID_MARKER_STOP("coo");
    }
    LIKWID_MARKER_CLOSE;
}