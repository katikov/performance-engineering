#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mmio.h"

#define N  512
#define M  512
#define P  512

#define REP 10
constexpr int threadBlockSize = 128;

__global__ void matrix_mult_kernel(int m, int n, int p, float *A, float *B, float *C)
{
    // Calculate global thread ID
   unsigned int j = blockIdx.y * threadBlockSize + threadIdx.x;
   unsigned int i = blockIdx.x;
   if(j<p){
      float res = 0.0;
      for(int k=0;k<n;k++)
         res += A[i*n+k]*B[k*p+j];
      C[i*p+j] = res;
   }
}



void inline matrix_mult_cuda(int m, int n, int p, float *A, float *B, float *C) {
   // int i, j, k;
   dim3 numBlocks(m,(p+threadBlockSize-1)/threadBlockSize);

   float *A_device, *B_device, *C_device;
struct timeval before, after;
   cudaMalloc((void **)&A_device, m*n*sizeof(float));
   cudaMalloc((void **)&B_device, n*p*sizeof(float));
   cudaMalloc((void **)&C_device, m*p*sizeof(float));
   // cudaMemset(C_device, 0, m*p*sizeof(float));
   cudaMemcpy(A_device, A, m*n*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(B_device, B, n*p*sizeof(float), cudaMemcpyHostToDevice);
gettimeofday(&before, NULL);
   matrix_mult_kernel<<<numBlocks, threadBlockSize>>>(m,n,p,A_device,B_device,C_device);
   cudaDeviceSynchronize();
gettimeofday(&after, NULL);
printf("Computation time: %10.2f seconds \n", ((after.tv_sec + (after.tv_usec / 1000000.0)) -
            (before.tv_sec + (before.tv_usec / 1000000.0))));
   cudaMemcpy(C, C_device, m*p*sizeof(float), cudaMemcpyDeviceToHost);
   cudaFree(A_device);
   cudaFree(B_device);
   cudaFree(C_device);
}

constexpr int blockWidth = 16;

__global__ void matrix_mult_block_kernel(int m, int n, int p, float *A, float *B, float *C)
{
    __shared__ float block_A[blockWidth][blockWidth];
    __shared__ float block_B[blockWidth][blockWidth];
   unsigned int j = blockIdx.x * blockWidth + threadIdx.x;
   unsigned int i = blockIdx.y * blockWidth + threadIdx.y;
   
   float res = 0.0;
   int n_floor = n/blockWidth * blockWidth;
   for(int b = 0; b < n_floor; b+=blockWidth){
      if(i<m){// && (b+threadIdx.y)<n){
         block_A[threadIdx.y][threadIdx.x] = A[i*n + b + threadIdx.x];
      }else{
        block_A[threadIdx.y][threadIdx.x] = 0;
      }

      //if((b+threadIdx.x)<n && j<p){
      if(j<p){
         block_B[threadIdx.y][threadIdx.x] = B[(b+threadIdx.y)*p + j];
      }else{
         block_B[threadIdx.y][threadIdx.x] = 0;
      }
      __syncthreads();
      for(int k=0; k<blockWidth; ++k){
         res += block_A[threadIdx.y][k] * block_B[k][threadIdx.x];
      }
      __syncthreads();
   }

   if(i<m && j<p){
      for(int k=n_floor;k<n;k++)
         res += A[i*n+k]*B[k*p+j];
      C[i*p + j] = res;
   }
}


void inline matrix_mult(int m, int n, int p, float *A, float *B, float *C) {
   // int i, j, k;
   dim3 numThreads(blockWidth, blockWidth);
   dim3 numBlocks((p+blockWidth-1)/blockWidth, (m+blockWidth-1)/blockWidth);

   float *A_device, *B_device, *C_device;
// struct timeval before, after;
   cudaMalloc((void **)&A_device, m*n*sizeof(float));
   cudaMalloc((void **)&B_device, n*p*sizeof(float));
   cudaMalloc((void **)&C_device, m*p*sizeof(float));
   // cudaMemset(C_device, 0, m*p*sizeof(float));
   cudaMemcpy(A_device, A, m*n*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(B_device, B, n*p*sizeof(float), cudaMemcpyHostToDevice);
// gettimeofday(&before, NULL);
   matrix_mult_block_kernel<<<numBlocks, numThreads>>>(m, n, p, A_device,B_device,C_device);

   cudaDeviceSynchronize();
// gettimeofday(&after, NULL);
// printf("Computation time: %10.2f seconds \n", ((after.tv_sec + (after.tv_usec / 1000000.0)) -
//             (before.tv_sec + (before.tv_usec / 1000000.0))));
   cudaMemcpy(C, C_device, m*p*sizeof(float), cudaMemcpyDeviceToHost);
   cudaFree(A_device);
   cudaFree(B_device);
   cudaFree(C_device);
}


void inline matrix_mult_basic(int m, int n, int p, float *A, float *B, float *C) {
   int i, j, k;

   for(i=0; i<m; i++) {
      for(j=0; j<p; j++) {
	      //float res = 0;
         C[i*p+j]=0;
        for(k=0; k<n; k++) {
                C[i*p+j] += A[i*n+k]*B[k*p+j];
            }
      }
   }
}

// already SIMD, 16 bytes(4 floats) in one instruction
void inline matrix_mult_better(int m, int n, int p, float *A, float *B, float *C) {
   int i, j, k;
   for(int i=0;i<m*p;i++) C[i]=0;
// double CC[3000*3000];
// for(int i=0;i<m*p;i++) CC[i]=0;
   for(i=0; i<m; i++) 
      for(k=0; k<n; k++) 
         for(j=0; j<p; j++) 
            C[i*p+j] += A[i*n+k]*B[k*p+j];

// CC[i*p+j] += double(1)*A[i*n+k]*B[k*p+j];
// for(int i=0;i<m*p;i++) C[i]=CC[i];
}

void inline matrix_mult_block(int m, int n, int p, float *A, float *B, float *C) {
   int i, j, k, iInner, jInner, kInner ;
   for(int i=0;i<m*p;i++) C[i]=0;
   constexpr int blockSize = 4;
   const int m_floor = m/4*4;
   const int n_floor = n/4*4;
   const int p_floor = p/4*4;
//#pragma vector aligned
   for (i = 0; i < m_floor; i+=blockSize)
      for (k = 0 ; k < n_floor; k+=blockSize)
         for (j=0; j<p_floor ; j+= blockSize)
            for (iInner = i; iInner<i+blockSize; iInner++)
               for (kInner = k ; kInner<k+blockSize ; kInner++)
//#pragma vector aligned
// pragma simd required
#pragma omp simd
                  for (jInner = j ; jInner<j+blockSize; jInner++)
                        C[iInner*p + jInner] += A[iInner*n + kInner] * B[kInner*p + jInner] ;

   // last few rows & cols, O(n^2) computation
   for(i=m_floor; i<m; i++) 
         for(k=0; k<n; k++) 
            for(j=0; j<p; j++) 
               C[i*p+j] += A[i*n+k]*B[k*p+j];

   for(i=0; i<m_floor; i++) 
         for(k=n_floor; k<n; k++) 
            for(j=0; j<p; j++) 
               C[i*p+j] += A[i*n+k]*B[k*p+j];

   for(i=0; i<m_floor; i++) 
         for(k=0; k<n_floor; k++) 
            for(j=p_floor; j<p; j++) 
               C[i*p+j] += A[i*n+k]*B[k*p+j];

}

void cmp(float* A, float* B, int length){
   for(int i=0;i<length;i++){
      if(fabs((A[i]-B[i])/B[i])>1e-6){
      //if(A[i]!=B[i]){
         printf("error in results!\n");
         printf("%f %f\n", A[i], B[i]);
         return;
      }
   }
   printf("results OK!\n");
}

void generate_mat(int m, int n, int p, float *A, float *B) {
  int i;

  for (i=0; i<(m*n); i++) A[i] = 1; //i/10; 
  for (i=0; i<(n*p); i++) B[i] = 1; //i/5;

}

void read_sparse(FILE *f, int m, int n, int nz, float *A) {
  int i, row, col;
  float val;  
 
    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %f\n", &row, &col, &val);
        A[(row-1)*n+col-1] = val;   /* adjust from 1-based to 0-based */
    }

}

void write_sparse(FILE *f, int m, int p, const float *C) {
   int i, nz=0; 
   MM_typecode matcode;

   for (i=0; i<m*p; i++) if (C[i] != 0.0) nz++; 

    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_coordinate(&matcode);
    mm_set_real(&matcode);

    mm_write_banner(f, matcode); 
    mm_write_mtx_crd_size(f, m, p, nz);

    for (i=0; i<m*p; i++) {
	if (C[i] != 0.0) 
          fprintf(f, "%d %d %f\n", i/p+1, i%p+1, C[i]);
    }

}

void read_dense(FILE *f, int m, int n, float *A) {
  int row, col;

  for(row=0; row<m; row++) { 
     for (col=0; col<n; col++) {
        fscanf(f, "%f ", &A[row*n+col]); 
//	printf("%20.19f \n", A[row*(*n)+col]);
     }
  } 
}


int read_mat(int *m, int *n, int *p, int *nzA, int *nzB, FILE* fa, FILE *fb) {
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

  if (mm_is_complex(ta)) return -6;
  if (mm_is_complex(tb)) return -7; 

  if (mm_is_matrix(ta) && mm_is_sparse(ta))
    {
        if ((ret_code = mm_read_mtx_crd_size(fa, m, n, nzA)) !=0)
           return -10;
    }
  else if (mm_is_matrix(ta) && mm_is_array(ta)) {
	*nzA = 0;
        if ((ret_code = mm_read_mtx_array_size(fa, m, n)) !=0)
           return -11;

    }
  else return -8; 


  if (mm_is_matrix(tb) && mm_is_sparse(tb))
    {
        if ((ret_code = mm_read_mtx_crd_size(fb, &n1, p, nzB)) !=0)
           return -10;
    }
  else if (mm_is_matrix(tb) && mm_is_array(tb)) {
	*nzB = 0;
        if ((ret_code = mm_read_mtx_array_size(fb, &n1, p)) !=0)
           return -11;
  
    }
  else return -9;
  
  if (*n!=n1) return -15;
  
  return 0;
    /* find out size of sparse matrix .... */
}

int main (int argc, char** argv) {
 float *A, *B, *C;
#ifdef TIMING
 struct timeval before, after;
#endif
 int m, n, p, r, err;
 int nzA=0, nzB=0;
 FILE *fa, *fb, *fc; 
 
#ifdef GENERATE 
 m=M; n=N; p=P;
#else 
 if (argc < 3) {
    fprintf(stderr, "Usage: %s [martix1] [matrix2] [resultmatrix] \n", argv[0]);
    exit(1);
 }
 else {
    if ((fa = fopen(argv[1], "rt")) == NULL) exit(1);
    if ((fb = fopen(argv[2], "rt")) == NULL) exit(2);
    err = read_mat(&m, &n, &p, &nzA, &nzB, fa,fb);    
    if (err == -15) {
	printf("Matrices are incompatible! \n");
	fclose(fa); fclose(fb); 
	exit(1);
    }
 }
#endif

 A = (float *)calloc(m*n,sizeof(float));
 if (A==NULL) {printf("Out of memory A! \n"); exit(1);}
 B = (float *)calloc(n*p,sizeof(float));
 if (B==NULL) {printf("Out of memory B! \n"); exit(1);}

#ifdef GENERATE
   generate_mat(m,n,p,A,B);
#else 
   if (nzA>0)
	read_sparse(fa, m,n,nzA, A);
   else 
	read_dense(fa, m,n, A);
   if (nzB>0)
        read_sparse(fb, n,p, nzB, B);
   else
        read_dense(fb, n,p, B); 
   fclose(fa); 
   fclose(fb);
#endif

 C = (float *)calloc(m*p,sizeof(float));
 if (C==NULL) {printf("Out of memory C1! \n"); exit(1);}
 float *D = (float *)calloc(m*p,sizeof(float));
 if (D==NULL) {printf("Out of memory C1! \n"); exit(1);}
// C2 = (float *)calloc(N*P,sizeof(float));
// if (C2==NULL) {printf("Out of memory C2! \n"); exit(1);}

//naive implementation 
#ifdef TIMING
  gettimeofday(&before, NULL); 
#endif

for (r=0; r<REP; r++) 
   matrix_mult(m,n,p,A,B,C);
#ifdef TIMING
  gettimeofday(&after, NULL);
  printf("Reference code: %10.2f seconds \n", ((after.tv_sec + (after.tv_usec / 1000000.0)) -
            (before.tv_sec + (before.tv_usec / 1000000.0)))/REP);

#endif

// floating point error different in cpu and gpu. check algorithm by result is not feasible
// double is more accurate than float. no fp error when using double.
//matrix_mult_better(m,n,p,A,B,D);
//cmp(C,D, m*p);

#ifdef GENERATE
 if ((fc = fopen("gen_result.mtx", "wt")) == NULL) exit(3); 
#else 
 if ((fc = fopen(argv[3], "wt")) == NULL) exit(3); 
#endif   
 write_sparse(fc,m,p,C);
 fclose(fc);  

 free(A);
 free(B);
 free(C);
// free(C2);

}

