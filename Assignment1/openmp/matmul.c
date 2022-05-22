#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "mmio.h"

#define N 512
#define M 512
#define P 512

#define REP 10

double inline get_density(int m, int n, float *A) {
   int i, j;
   long long total, zero=0;
   total = m * n;
   for (i = 0; i < m; i++)
   {
      for(j=0; j<n; j++) {
         if (A[i*n+j] !=0){
            zero++;
         }
      }
   }
   printf("%d, %d\n", total, zero);
   return 1.0 - zero * 1.0 / total;
}

void inline matrix_mult_basic(int m, int n, int p, float *A, float *B, float *C)
{
   int i, j, k;

   for (i = 0; i < m; i++)
   {
      for (j = 0; j < p; j++)
      {
         // float res = 0;
         C[i * p + j] = 0;
         for (k = 0; k < n; k++)
         {
            C[i * p + j] += A[i * n + k] * B[k * p + j];
         }
      }
   }
}

void inline matrix_mult_better(int m, int n, int p, float *A, float *B, float *C)
{
   int i, j, k;
#pragma omp parallel for private(i)
   for (int i = 0; i < m * p; i++)
      C[i] = 0;
#pragma omp parallel for private(i, j, k)
   for (i = 0; i < m; i++)
      for (k = 0; k < n; k++)
#pragma omp simd
         for (j = 0; j < p; j++)
            C[i * p + j] += A[i * n + k] * B[k * p + j];
}

void inline matrix_mult(int m, int n, int p, float *A, float *B, float *C)
{
   int i, j, k, iInner, jInner, kInner;
#pragma omp parallel for private(i)
   for (int i = 0; i < m * p; i++)
      C[i] = 0;
   constexpr int blockSize = 8;
   const int m_floor = m / blockSize * blockSize;
   const int n_floor = n / blockSize * blockSize;
   const int p_floor = p / blockSize * blockSize;
   //#pragma vector aligned
#pragma omp parallel for private(i, j, k, iInner, jInner, kInner)
   for (i = 0; i < m_floor; i += blockSize)
      for (k = 0; k < n_floor; k += blockSize)
         for (j = 0; j < p_floor; j += blockSize)
            for (iInner = i; iInner < i + blockSize; iInner++)
               for (kInner = k; kInner < k + blockSize; kInner++)
//#pragma vector aligned
// pragma simd required
#pragma omp simd
                  for (jInner = j; jInner < j + blockSize; jInner++)
                     C[iInner * p + jInner] += A[iInner * n + kInner] * B[kInner * p + jInner];

                     // last few rows & cols, O(n^2) computation
#pragma omp parallel for private(i, j, k)
   for (i = m_floor; i < m; i++)
      for (k = 0; k < n; k++)
         //# pragma omp parallel for private(j)
         for (j = 0; j < p; j++)
            C[i * p + j] += A[i * n + k] * B[k * p + j];

#pragma omp parallel for private(i, j, k)
   for (i = 0; i < m_floor; i++)
      for (k = n_floor; k < n; k++)
         for (j = 0; j < p; j++)
            C[i * p + j] += A[i * n + k] * B[k * p + j];

#pragma omp parallel for private(i, j, k)
   for (i = 0; i < m_floor; i++)
      for (k = 0; k < n_floor; k++)
         for (j = p_floor; j < p; j++)
            C[i * p + j] += A[i * n + k] * B[k * p + j];
}

void cmp(float *A, float *B, int length)
{
   for (int i = 0; i < length; i++)
   {
      if (A[i] != B[i])
      {
         printf("error in results!\n");
         return;
      }
   }
   printf("results OK!\n");
}

void generate_mat(int m, int n, int p, float *A, float *B, float density)
{
   int i;
   float random = 0;
   for (i = 0; i < (m * n); i++)
   {
      random = (float)rand()/(float)(RAND_MAX);
      if (random > density)
      {
         A[i] = 0; // i/10;
      }
      else
      {
         A[i] = (float)rand() / (float)(RAND_MAX);
      }
   }
   for (i = 0; i < (n * p); i++)
   {
      random = (float)rand()/(float)(RAND_MAX);
      if (random > density)
      {
         B[i] = 0; // i/10;
      }
      else
      {
         B[i] = (float)rand() / (float)(RAND_MAX);
      }
   }}

void read_sparse(FILE *f, int m, int n, int nz, float *A)
{
   int i, row, col;
   float val;

   /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
   /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
   /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

   for (i = 0; i < nz; i++)
   {
      fscanf(f, "%d %d %f\n", &row, &col, &val);
      A[(row - 1) * n + col - 1] = val; /* adjust from 1-based to 0-based */
   }
}

void write_sparse(FILE *f, int m, int p, const float *C)
{
   int i, nz = 0;
   MM_typecode matcode;

   for (i = 0; i < m * p; i++)
      if (C[i] != 0.0)
         nz++;

   mm_initialize_typecode(&matcode);
   mm_set_matrix(&matcode);
   mm_set_coordinate(&matcode);
   mm_set_real(&matcode);

   mm_write_banner(f, matcode);
   mm_write_mtx_crd_size(f, m, p, nz);

   for (i = 0; i < m * p; i++)
   {
      if (C[i] != 0.0)
         fprintf(f, "%d %d %f\n", i / p + 1, i % p + 1, C[i]);
   }
}

void read_dense(FILE *f, int m, int n, float *A)
{
   int row, col;

   for (row = 0; row < m; row++)
   {
      for (col = 0; col < n; col++)
      {
         fscanf(f, "%f ", &A[row * n + col]);
         //	printf("%20.19f \n", A[row*(*n)+col]);
      }
   }
}

int read_mat(int *m, int *n, int *p, int *nzA, int *nzB, FILE *fa, FILE *fb)
{
   MM_typecode ta, tb;
   int ret_code;
   int n1;

   if (mm_read_banner(fa, &ta) != 0)
   {
      printf("Could not process Matrix Market banneri for A.\n");
      return -3;
   }
   if (mm_read_banner(fb, &tb) != 0)
   {
      printf("Could not process Matrix Market banner for B.\n");
      return -4;
   }

   if (mm_is_complex(ta))
      return -6;
   if (mm_is_complex(tb))
      return -7;

   if (mm_is_matrix(ta) && mm_is_sparse(ta))
   {
      if ((ret_code = mm_read_mtx_crd_size(fa, m, n, nzA)) != 0)
         return -10;
   }
   else if (mm_is_matrix(ta) && mm_is_array(ta))
   {
      *nzA = 0;
      if ((ret_code = mm_read_mtx_array_size(fa, m, n)) != 0)
         return -11;
   }
   else
      return -8;

   if (mm_is_matrix(tb) && mm_is_sparse(tb))
   {
      if ((ret_code = mm_read_mtx_crd_size(fb, &n1, p, nzB)) != 0)
         return -10;
   }
   else if (mm_is_matrix(tb) && mm_is_array(tb))
   {
      *nzB = 0;
      if ((ret_code = mm_read_mtx_array_size(fb, &n1, p)) != 0)
         return -11;
   }
   else
      return -9;

   if (*n != n1)
      return -15;

   return 0;
   /* find out size of sparse matrix .... */
}

int main(int argc, char **argv)
{
   float *A, *B, *C;
#ifdef TIMING
   struct timeval before, after;
#endif
   int m, n, p, r, err;
   float density;
   int nzA = 0, nzB = 0;
   FILE *fa, *fb, *fc;
   int proc_count = 1;
#ifdef GENERATE
   if (argc < 5)
   {
      fprintf(stderr, "Usage: %s [m] [n] [p] [density] [proc_count]\n", argv[0]);
      exit(1);
   }
   else
   {
      m = atoi(argv[1]);
      n = atoi(argv[2]);
      p = atoi(argv[3]);
      density = atof(argv[4]);
      proc_count = atoi(argv[5]);
   }
#else
   if (argc < 4)
   {
      fprintf(stderr, "Usage: %s [martix1] [matrix2] [resultmatrix] [proc_count]\n", argv[0]);
      exit(1);
   }
   else
   {
      if ((fa = fopen(argv[1], "rt")) == NULL)
         exit(1);
      if ((fb = fopen(argv[2], "rt")) == NULL)
         exit(2);
      err = read_mat(&m, &n, &p, &nzA, &nzB, fa, fb);
      if (err == -15)
      {
         printf("Matrices are incompatible! \n");
         fclose(fa);
         fclose(fb);
         exit(1);
      }
      proc_count = atoi(argv[4]);
   }
#endif
   omp_set_num_threads(proc_count);
   // omp_set_schedule(omp_heat_parallel_type + 1, chunk_size);

   A = (float *)calloc(m * n, sizeof(float));
   if (A == NULL)
   {
      printf("Out of memory A! \n");
      exit(1);
   }
   B = (float *)calloc(n * p, sizeof(float));
   if (B == NULL)
   {
      printf("Out of memory B! \n");
      exit(1);
   }

#ifdef GENERATE
   generate_mat(m, n, p, A, B, density);
#else
   if (nzA > 0)
      read_sparse(fa, m, n, nzA, A);
   else
      read_dense(fa, m, n, A);
   if (nzB > 0)
      read_sparse(fb, n, p, nzB, B);
   else
      read_dense(fb, n, p, B);
   fclose(fa);
   fclose(fb);
#endif

   C = (float *)calloc(m * p, sizeof(float));
   if (C == NULL)
   {
      printf("Out of memory C1! \n");
      exit(1);
   }
   float *D = (float *)calloc(m * p, sizeof(float));
   if (D == NULL)
   {
      printf("Out of memory C1! \n");
      exit(1);
   }
// C2 = (float *)calloc(N*P,sizeof(float));
// if (C2==NULL) {printf("Out of memory C2! \n"); exit(1);}

// naive implementation
#ifdef TIMING
   gettimeofday(&before, NULL);
#endif

   for (r = 0; r < REP; r++)
      matrix_mult(m, n, p, A, B, C);

   unsigned long long int num = ((unsigned long long int)m) * n * p;
#ifdef TIMING
   gettimeofday(&after, NULL);
  printf("%d, %d, %d, %d,%f, %llu, %f, %.6f\n", proc_count, m, n, p, m*1.0/n,num, density, ((after.tv_sec + (after.tv_usec / 1000000.0)) -
            (before.tv_sec + (before.tv_usec / 1000000.0)))/REP);

#endif
   // matrix_mult_basic(m,n,p,A,B,D);
   // cmp(C,D, m*p);

#ifdef GENERATE
   if ((fc = fopen("gen_result.mtx", "wt")) == NULL)
      exit(3);
#else
   if ((fc = fopen(argv[3], "wt")) == NULL)
      exit(3);
#endif
   write_sparse(fc, m, p, C);
   fclose(fc);

   free(A);
   free(B);
   free(C);
   // free(C2);
}
