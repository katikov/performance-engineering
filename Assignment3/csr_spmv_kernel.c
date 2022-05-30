#include "csr_spmv_kernel.h"
#include <omp.h>
#include <likwid.h>

/*
 * m = number of rows
 * A_rows = row pointers, m+1 of them, each element with the start of the nonZero's in the columns array
 * A_cols_idx = column indices of the nonZero values
 * A_values = the actual values in the nonZero locations
 * B = the vector
 * C = the result
 */

void csr_spmv(int m, const int *A_rows, const int *A_cols_idx, const float *A_values, const float *B, float *C)
{
    int i, j;
    int row_start, row_end;

    for (i = 0; i < m; i++)
    {
        float tmp = 0.0f;
        row_start = A_rows[i];
        row_end = A_rows[i + 1];

        for (j = row_start; j < row_end; j++)
        {
            tmp += A_values[j] * B[A_cols_idx[j]];
        }

        C[i] = tmp;
    }
}

void csr_spmv_parallel(int m, const int *A_rows, const int *A_cols_idx, const float *A_values, const float *B, float *C)
{
    int i, j;
    int row_start, row_end;
    omp_set_num_threads(32);
    LIKWID_MARKER_INIT;
#pragma omp parallel
    {
        LIKWID_MARKER_THREADINIT;
    }
#pragma omp parallel
    {
        LIKWID_MARKER_START("csr");
#pragma omp for private(i, j, row_start, row_end)
        for (i = 0; i < m; i++)
        {
            float tmp = 0.0f;
            row_start = A_rows[i];
            row_end = A_rows[i + 1];

            for (j = row_start; j < row_end; j++)
            {
                tmp += A_values[j] * B[A_cols_idx[j]];
            }

            C[i] = tmp;
        }
        LIKWID_MARKER_STOP("csr");
    }
    LIKWID_MARKER_CLOSE;
}
