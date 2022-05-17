#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "fft_cpu.h"


#define REP 100


double computation_time = 0;
void cmp(double* A, double* B, int length){
   for(int i=0;i<length;i++){
      if(fabs(A[i]-B[i])>1e-6 && fabs((A[i]-B[i])/B[i])>1e-6){
      //if(A[i]!=B[i]){
         printf("error in results!\n");
         printf("%f %f\n", A[i], B[i]);
         return;
      }
   }
   printf("results OK!\n");
}

int* generate_mat(int n, int m) {
   srand(42); // fixed seed
   int size = m*n;
   int* img = (int*)malloc(size*sizeof(int));
   if(img==NULL){
      printf("Out of memory! \n");
      exit(-1);
    }
   for(int i=0;i<size;i++){
      img[i] = rand()%256;
   }
   return img;
}
__constant__ int _prime[3] = {2,3,5};
__constant__ double pi = 3.141592653589793;
const double pi_cpu = 3.141592653589793;
constexpr int prime_num = 3;
__global__ void cuda_fft_init(int n, int* radix, int* ex_bit_reversal, Complex* wn_pows){
   int start=0, length = n;
   int r = blockIdx.x * blockDim.x + threadIdx.x;
   int i=r;
   if(r<n){
      for(int t=prime_num-1;t>=0;t--){
         int p = _prime[t];
         int cnt = radix[t];
         for(int _=0;_<cnt;_++){
            int id = i % p;
            i = i/p;
            length /= p;
            start += id*length; 
         }
      }
      ex_bit_reversal[r] = start;
      wn_pows[r] = Complex{cos(2*pi*r/n), -sin(2*pi*r/n)};
   }
}

__forceinline__ __device__ Complex operator *(const Complex a, const Complex b){
    return Complex{a.real*b.real-a.imag*b.imag, a.real*b.imag+a.imag*b.real};
} 

__forceinline__ __device__ Complex operator *(const Complex a, double b){
    return Complex{a.real*b, a.imag*b};
} 

__forceinline__ __device__ Complex operator +(const Complex a, const Complex b){
    return Complex{a.real + b.real, a.imag + b.imag};
} 

__forceinline__ __device__ Complex operator -(const Complex a, const Complex b){
    return Complex{a.real - b.real, a.imag - b.imag};
}


__forceinline__ __device__ Complex wm(int k, int m){
   //k%=m;
   return Complex{cos(2*k*pi/m), -sin(2*k*pi/m)};
} 


__global__ void fft_cuda_basic_kernel(int* image, Complex* dft_image, int* ex_bit_reversal, 
             int* radix, Complex* wm_pows, int n, int m)
{
   unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int threads = blockDim.y;
   unsigned int tid = threadIdx.y;
   if(r < n){
      Complex* dft_line = dft_image + r * m;
      int* image_line = image + r*m;
      for(int j=tid;j<m;j+=threads) dft_line[j] = Complex{(double)image_line[ex_bit_reversal[j]], 0};
      
      int len=1;
      Complex temp[5], temp2[5];
      __syncthreads();
      for(int t=2;t>=0;t--){
         int cnt = radix[t];
         int p = _prime[t];
         while(cnt--){
            for(int i=len*p*tid;i<m;i+=len*p*threads){
                  for(int j=0;j<len;j++){
                     for(int v=0; v<p; v++) temp[v] = dft_line[i+j+v*len] * wm_pows[m*v*j/(len*p)];
                     for(int v=0; v<p; v++){
                        temp2[v] = temp[0];
                        for(int w=1;w<p;w++) temp2[v] = temp2[v] + wm_pows[(w*v*m/p)%m] * temp[w];
                     }
                     for(int v=0; v<p; v++) dft_line[i+j+v*len] = temp2[v];
                  }
            }
            len *= p;
            __syncthreads();
         }
      }

   }
}

__global__ void fft_cuda_basic_kernel_col(Complex* col_temp, Complex* dft_image, int* ex_bit_reversal, 
             int* radix, Complex* wm_pows, int n, int m)
{
   unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int threads = blockDim.y;
   unsigned int tid = threadIdx.y;
   if(r < m){
      Complex* dft_col = dft_image + r;
      Complex *col = col_temp + r*n;
      for(int i=tid;i<n;i+=threads) col[i] = dft_col[ex_bit_reversal[i]*m];
      __syncthreads();
      int len=1;
      Complex temp[5], temp2[5];
      for(int t=2;t>=0;t--){
         int cnt = radix[t];
         int p = _prime[t];
         while(cnt--){
            for(int i=len*p*tid;i<n;i+=len*p*threads){
                  for(int j=0;j<len;j++){
                     for(int v=0; v<p; v++) temp[v] = col[i+j+v*len] * wm_pows[n*v*j/(len*p)];
                     for(int v=0; v<p; v++){
                        temp2[v] = temp[0];
                        for(int w=1;w<p;w++) temp2[v] = temp2[v] + wm_pows[(w*v*n/p)%n] * temp[w];
                     }
                     for(int v=0; v<p; v++) col[i+j+v*len] = temp2[v];
                  }
            }
            __syncthreads();
            len *= p;
         }
      }
      for(int i=tid;i<n;i+=threads){
         dft_col[i*m] = col[i];
      }
   }
}


void inline fft2_cuda_basic(int* image, Complex* dft_image, int* image_device, Complex* dft_device, int n, int m) {
   int numThreads = 32;
   int numBlocksRow = (n+numThreads-1)/numThreads;
   int numBlocksCol = (m+numThreads-1)/numThreads;
   struct timeval before, after;
   //for(int i=0;i<n;i++)printf("%d ", image[i]); printf("\n");
   // cudaMemset(C_device, 0, m*p*sizeof(float));
   cudaMemcpy(image_device, image, m*n*sizeof(int), cudaMemcpyHostToDevice);
   int* radix_device; cudaMalloc((void **)&radix_device, 3*sizeof(int));
   int* ex_bit_reversal; cudaMalloc((void **)&ex_bit_reversal, max(m,n)*sizeof(int));
   Complex* wm_pows; cudaMalloc((void **)&wm_pows, max(m,n)*sizeof(Complex));
   Complex* col_temp; cudaMalloc((void **)&col_temp, m*n*sizeof(Complex));
   gettimeofday(&before, NULL);
   int radix[3];
   assert(getradix(m, radix)==1);
   cudaMemcpy(radix_device, radix, 3*sizeof(int), cudaMemcpyHostToDevice);
   cuda_fft_init<<<numBlocksCol ,numThreads>>>(m, radix_device, ex_bit_reversal, wm_pows);
   //fft_cuda_basic_kernel<<<numBlocksRow, numThreads>>>(image_device, dft_device, ex_bit_reversal, radix_device, wm_pows, n, m);
   fft_cuda_basic_kernel<<<(n+4-1)/4, dim3(4,8)>>>(image_device, dft_device, ex_bit_reversal, radix_device, wm_pows, n, m);

   assert(getradix(n, radix)==1);
   cudaMemcpy(radix_device, radix, 3*sizeof(int), cudaMemcpyHostToDevice);
   cuda_fft_init<<<numBlocksRow ,numThreads>>>(n, radix_device, ex_bit_reversal, wm_pows);
   // cudaMemcpy(dft_image, wm_pows, m*n*sizeof(Complex), cudaMemcpyDeviceToHost);
   // for(int i=0;i<n;i++)printf("%lf+%lf ", dft_image[i].real, dft_image[i].imag); printf("\n");
   fft_cuda_basic_kernel_col<<<(m+4-1)/4, dim3(4,8)>>>(col_temp, dft_device, ex_bit_reversal, radix_device, wm_pows, n, m);


   cudaDeviceSynchronize();
   gettimeofday(&after, NULL);
   computation_time += (after.tv_sec + (after.tv_usec / 1000000.0)) -
                      (before.tv_sec + (before.tv_usec / 1000000.0));

// printf("Computation time: %10.2f seconds \n", ));
   cudaMemcpy(dft_image, dft_device, m*n*sizeof(Complex), cudaMemcpyDeviceToHost);
   cudaFree(radix_device);
   cudaFree(ex_bit_reversal);
   cudaFree(wm_pows);
   cudaFree(col_temp);

   //for(int i=0;i<n;i++)printf("%lf+%lfj ", dft_image[i].real, dft_image[i].imag); printf("\n");
}

/*******************************************************************************************************************/
__device__ __forceinline__ void fft1_kernel(Complex *dft_line, Complex* wm_pows, int* radix, int m, int tid, int threads){
   int len=1;
   Complex temp[5], temp2[5];
   for(int t=2;t>=0;t--){
      int cnt = radix[t];
      int p = _prime[t];
      while(cnt--){
         if(len >= threads)
            for(int i=0;i<m;i+=len*p){
               for(int j=tid;j<len;j+=threads){
                  for(int v=0; v<p; v++) temp[v] = dft_line[i+j+v*len] * wm_pows[m*v*j/(len*p)];
                  for(int v=0; v<p; v++){
                     temp2[v] = temp[0];
                     for(int w=1;w<p;w++) temp2[v] = temp2[v] + wm_pows[(w*v*m/p)%m] * temp[w];
                  }
                  for(int v=0; v<p; v++) dft_line[i+j+v*len] = temp2[v];
               }
            }
         else
            for(int i=len*p*tid;i<m;i+=len*p*threads){
                  for(int j=0;j<len;j++){
                     for(int v=0; v<p; v++) temp[v] = dft_line[i+j+v*len] * wm_pows[m*v*j/(len*p)];
                     for(int v=0; v<p; v++){
                        temp2[v] = temp[0];
                        for(int w=1;w<p;w++) temp2[v] = temp2[v] + wm_pows[(w*v*m/p)%m] * temp[w];
                     }
                     for(int v=0; v<p; v++) dft_line[i+j+v*len] = temp2[v];
                  }
            }
         len *= p;
         __syncthreads();
      }
   }
} 

// __constant__ Complex w3_1 = Complex{-0.5,-0.8660254037844386}, w3_2 = Complex{-0.5,0.8660254037844386};
// __constant__ Complex w5_1 = Complex{0.30901699437494745, -0.9510565162951535},
//                   w5_2 = Complex{-0.8090169943749473, -0.5877852522924732},
//                   w5_3 = Complex{-0.8090169943749475, 0.587785252292473},
//                   w5_4 = Complex{0.30901699437494723, 0.9510565162951536};

__device__ __forceinline__ void fft1_kernel_unroll(Complex *dft_line, Complex* wm_pows, int* radix, int m, int tid, int threads){
   const Complex w3_1 = Complex{-0.5,-0.8660254037844386}, w3_2 = Complex{-0.5,0.8660254037844386};
   const Complex w5_1 = Complex{0.30901699437494745, -0.9510565162951535},
                     w5_2 = Complex{-0.8090169943749473, -0.5877852522924732},
                     w5_3 = Complex{-0.8090169943749475, 0.587785252292473},
                     w5_4 = Complex{0.30901699437494723, 0.9510565162951536};
   
      
   int len=1;
   Complex temp[5];
   int cnt = radix[2];
   int p = 5;
   while(cnt--){
      if(len >= threads)
         for(int i=0;i<m;i+=len*p){
            for(int j=tid;j<len;j+=threads){
               temp[0] = dft_line[i+j];
               temp[1] = dft_line[i+j+1*len] * wm_pows[m/(len*p)*j];
               temp[2] = dft_line[i+j+2*len] * wm_pows[m/(len*p)*j*2];
               temp[3] = dft_line[i+j+3*len] * wm_pows[m/(len*p)*j*3];
               temp[4] = dft_line[i+j+4*len] * wm_pows[m/(len*p)*j*4];

               dft_line[i+j] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4];
               dft_line[i+j+len] = temp[0] + temp[1]*w5_1 + temp[2]*w5_2 + temp[3]*w5_3 + temp[4]*w5_4;
               dft_line[i+j+2*len] = temp[0] + temp[1]*w5_2 + temp[2]*w5_4 + temp[3]*w5_1 + temp[4]*w5_3;
               dft_line[i+j+3*len] = temp[0] + temp[1]*w5_3 + temp[2]*w5_1 + temp[3]*w5_4 + temp[4]*w5_2;
               dft_line[i+j+4*len] = temp[0] + temp[1]*w5_4 + temp[2]*w5_3 + temp[3]*w5_2 + temp[4]*w5_1;
            }
         }
      else
         for(int i=len*p*tid;i<m;i+=len*p*threads){
            for(int j=0;j<len;j++){
               temp[0] = dft_line[i+j];
               temp[1] = dft_line[i+j+1*len] * wm_pows[m/(len*p)*j];
               temp[2] = dft_line[i+j+2*len] * wm_pows[m/(len*p)*j*2];
               temp[3] = dft_line[i+j+3*len] * wm_pows[m/(len*p)*j*3];
               temp[4] = dft_line[i+j+4*len] * wm_pows[m/(len*p)*j*4];

               dft_line[i+j] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4];
               dft_line[i+j+len] = temp[0] + temp[1]*w5_1 + temp[2]*w5_2 + temp[3]*w5_3 + temp[4]*w5_4;
               dft_line[i+j+2*len] = temp[0] + temp[1]*w5_2 + temp[2]*w5_4 + temp[3]*w5_1 + temp[4]*w5_3;
               dft_line[i+j+3*len] = temp[0] + temp[1]*w5_3 + temp[2]*w5_1 + temp[3]*w5_4 + temp[4]*w5_2;
               dft_line[i+j+4*len] = temp[0] + temp[1]*w5_4 + temp[2]*w5_3 + temp[3]*w5_2 + temp[4]*w5_1;
            }
         }
      len *= p;
      __syncthreads();
   }
   
   cnt = radix[1];
   p = 3;
   while(cnt--){
      if(len >= threads)
         for(int i=0;i<m;i+=len*p){
            for(int j=tid;j<len;j+=threads){
               temp[0] = dft_line[i+j];
               temp[1] = dft_line[i+j+len] * wm_pows[m/(len*p)*j];
               temp[2] = dft_line[i+j+2*len] * wm_pows[m/(len*p)*j*2];
               dft_line[i+j] = temp[0] + temp[1] + temp[2];
               dft_line[i+j+len] = temp[0] + temp[1]*w3_1 + temp[2]*w3_2;
               dft_line[i+j+2*len] = temp[0] + temp[1]*w3_2 + temp[2]*w3_1;
            }
         }
      else
         for(int i=len*p*tid;i<m;i+=len*p*threads){
            for(int j=0;j<len;j++){
               temp[0] = dft_line[i+j];
               temp[1] = dft_line[i+j+len] * wm_pows[m/(len*p)*j];
               temp[2] = dft_line[i+j+2*len] * wm_pows[m/(len*p)*j*2];
               dft_line[i+j] = temp[0] + temp[1] + temp[2];
               dft_line[i+j+len] = temp[0] + temp[1]*w3_1 + temp[2]*w3_2;
               dft_line[i+j+2*len] = temp[0] + temp[1]*w3_2 + temp[2]*w3_1;
            }
         }
      len *= p;
      __syncthreads();
   }
   


   cnt = radix[0];
   p = 2;
   while(cnt--){
      if(len >= threads)
         for(int i=0;i<m;i+=len*p){
            for(int j=tid;j<len;j+=threads){
               temp[0] = dft_line[i+j];
               temp[1] = dft_line[i+j+len] * wm_pows[m/(len*p)*j];
               dft_line[i+j] = temp[0] + temp[1];
               dft_line[i+j+len] = temp[0] - temp[1];
            }
         }
      else
         for(int i=len*p*tid;i<m;i+=len*p*threads){
            for(int j=0;j<len;j++){
               temp[0] = dft_line[i+j];
               temp[1] = dft_line[i+j+len] * wm_pows[m/(len*p)*j];
               dft_line[i+j] = temp[0] + temp[1];
               dft_line[i+j+len] = temp[0] - temp[1];
            }
         }
      len *= p;
      __syncthreads();
   }
} 

__global__ void fft_cuda_kernel(int* image, Complex* dft_image, int* ex_bit_reversal, 
             int* radix, Complex* wm_pows, int n, int m)
{
   unsigned int r = blockIdx.x;
   unsigned int threads = blockDim.x;
   unsigned int tid = threadIdx.x;


   extern __shared__ Complex dft_line[];
   Complex* dft_res = dft_image + r * m;
   int* image_line = image + r*m;
   for(int j=tid;j<m;j+=threads) dft_line[j] = Complex{(double)image_line[ex_bit_reversal[j]], 0};
   __syncthreads();
   fft1_kernel(dft_line, wm_pows, radix, m, tid, threads);
   for(int j=tid;j<m;j+=threads) dft_res[j] = dft_line[j];
   
}


__global__ void fft_cuda_kernel_col(Complex* dft_image, int* ex_bit_reversal, 
             int* radix, Complex* wm_pows, int n, int m)
{
   unsigned int r = blockIdx.x;
   unsigned int threads = blockDim.x;
   unsigned int tid = threadIdx.x;

   extern __shared__ Complex col[];
   Complex* dft_col = dft_image + r;
   for(int i=tid;i<n;i+=threads) col[i] = dft_col[ex_bit_reversal[i]*m];
   __syncthreads();
   fft1_kernel(col, wm_pows, radix, n, tid, threads);
   for(int i=tid;i<n;i+=threads){
      dft_col[i*m] = col[i];
   }
}


void inline fft2_cuda(int* image, Complex* dft_image, int* image_device, Complex* dft_device, int n, int m) {
   int numThreads = 32;
   int numBlocksRow = (n+numThreads-1)/numThreads;
   int numBlocksCol = (m+numThreads-1)/numThreads;
   struct timeval before, after;

   //for(int i=0;i<m;i++)printf("%d ", image[i]); printf("\n");
   // cudaMemset(C_device, 0, m*p*sizeof(float));
   //gettimeofday(&before, NULL);
   cudaMemcpy(image_device, image, m*n*sizeof(int), cudaMemcpyHostToDevice);
   //gettimeofday(&after, NULL);
   int* radix_device; cudaMalloc((void **)&radix_device, 3*sizeof(int));
   int* ex_bit_reversal; cudaMalloc((void **)&ex_bit_reversal, max(m,n)*sizeof(int));
   Complex* wm_pows; cudaMalloc((void **)&wm_pows, max(m,n)*sizeof(Complex));
   //Complex* col_temp; cudaMalloc((void **)&col_temp, m*n*sizeof(Complex));
   
   gettimeofday(&before, NULL);
   int radix[3];
   assert(getradix(m, radix)==1);
   cudaMemcpy(radix_device, radix, 3*sizeof(int), cudaMemcpyHostToDevice);
   cuda_fft_init<<<numBlocksCol ,numThreads>>>(m, radix_device, ex_bit_reversal, wm_pows);
   //fft_cuda_basic_kernel<<<(n+4-1)/4, dim3(4,8)>>>(image_device, dft_device, ex_bit_reversal, radix_device, wm_pows, n, m);
   fft_cuda_kernel<<<n, 64, m*sizeof(Complex)>>>(image_device, dft_device, ex_bit_reversal, radix_device, wm_pows, n, m);

   assert(getradix(n, radix)==1);
   cudaMemcpy(radix_device, radix, 3*sizeof(int), cudaMemcpyHostToDevice);
   cuda_fft_init<<<numBlocksRow ,numThreads>>>(n, radix_device, ex_bit_reversal, wm_pows);
   
   //cudaMemcpy(dft_image, dft_device, m*n*sizeof(Complex), cudaMemcpyDeviceToHost);
   //for(int i=0;i<m;i++)printf("%lf+%lf ", dft_image[i].real, dft_image[i].imag); printf("\n");
   //fft_cuda_basic_kernel_col<<<(m+4-1)/4, dim3(4,8)>>>(col_temp, dft_device, ex_bit_reversal, radix_device, wm_pows, n, m);
   fft_cuda_kernel_col<<<m, 64, n*sizeof(Complex)>>>(dft_device, ex_bit_reversal, radix_device, wm_pows, n, m);
   //fft_cuda_kernel_swap<<<m, 64, n*sizeof(Complex)*2>>>(dft_device, radix_device, wm_pows, n, m);

   cudaDeviceSynchronize();
   gettimeofday(&after, NULL);
   computation_time += (after.tv_sec + (after.tv_usec / 1000000.0)) -
                      (before.tv_sec + (before.tv_usec / 1000000.0));

// printf("Computation time: %10.2f seconds \n", ));
   cudaMemcpy(dft_image, dft_device, m*n*sizeof(Complex), cudaMemcpyDeviceToHost);
   cudaFree(radix_device);
   cudaFree(ex_bit_reversal);
   cudaFree(wm_pows);
   //cudaFree(col_temp);

   //for(int i=0;i<n;i++)printf("%lf+%lfj ", dft_image[i].real, dft_image[i].imag); printf("\n");
}


// // void fft2_cufft(int* image, Complex* dft_image, int* image_device, cufftHandle *plan, int n, int m) {
// void fft2_cufft(cufftDoubleComplex* idata, cufftDoubleComplex* odata, cufftDoubleComplex* CompData, cufftHandle *plan, Complex* dft_image, int n, int m) {
   
//    // cufftDoubleComplex *idata, *odata;
//    // cudaMalloc((void**)&idata, sizeof(cufftDoubleComplex)*m*n);
//    // cudaMalloc((void**)&odata, sizeof(cufftDoubleComplex)*m*n);
//    // //cudaMalloc((void**)&tdata, sizeof(cufftDoubleComplex)*m*n);

//    // cufftDoubleComplex *CompData = (cufftDoubleComplex*) malloc(m*n*sizeof(cufftDoubleComplex));
//    // for(int i=0; i<m*n; i++){
//    //    CompData[i].x = image[i]; CompData[i].y=0;
//    // }
//    // cudaMemcpy(idata, CompData, m*n * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
   
//    //cufftExecZ2Z(*plan, idata, odata, CUFFT_FORWARD);
//    //cufftExecZ2Z(plan, idata, tdata, CUFFT_FORWARD);
   

//    cudaMemcpy(dft_image, odata, m*n*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//    //for(int i=0;i<m*n;i++) dft_image[i] = Complex{CompData[i].x, CompData[i].y};
   
   
// }
/*********************************************************************************************************/

__global__ void fft_cuda_kernel_unroll(int* image, Complex* dft_image, int* ex_bit_reversal, 
             int* radix, Complex* wm_pows, int n, int m)
{
   unsigned int r = blockIdx.x;
   unsigned int threads = blockDim.x;
   unsigned int tid = threadIdx.x;


   extern __shared__ Complex dft_line[];
   Complex* dft_res = dft_image + r * m;
   int* image_line = image + r*m;
   for(int j=tid;j<m;j+=threads) dft_line[j] = Complex{(double)image_line[ex_bit_reversal[j]], 0};
   __syncthreads();
   fft1_kernel_unroll(dft_line, wm_pows, radix, m, tid, threads);
   for(int j=tid;j<m;j+=threads) dft_res[j] = dft_line[j];
   
}

__global__ void fft_cuda_kernel_col_unroll(Complex* dft_image, int* ex_bit_reversal, 
             int* radix, Complex* wm_pows, int n, int m)
{
   unsigned int r = blockIdx.x;
   unsigned int threads = blockDim.x;
   unsigned int tid = threadIdx.x;

   extern __shared__ Complex col[];

   Complex* dft_col = dft_image + r;
   for(int i=tid;i<n;i+=threads) col[i] = dft_col[ex_bit_reversal[i]*m];
   __syncthreads();
   fft1_kernel_unroll(col, wm_pows, radix, n, tid, threads);
   for(int i=tid;i<n;i+=threads){
      dft_col[i*m] = col[i];
   }
}

void inline fft2_cuda_unroll(int* image, Complex* dft_image, int* image_device, Complex* dft_device, int n, int m) {
   int numThreads = 32;
   int numBlocksRow = (n+numThreads-1)/numThreads;
   int numBlocksCol = (m+numThreads-1)/numThreads;
   struct timeval before, after;

   cudaMemcpy(image_device, image, m*n*sizeof(int), cudaMemcpyHostToDevice);
   //gettimeofday(&after, NULL);
   int* radix_device; cudaMalloc((void **)&radix_device, 3*sizeof(int));
   int* ex_bit_reversal; cudaMalloc((void **)&ex_bit_reversal, max(m,n)*sizeof(int));
   Complex* wm_pows; cudaMalloc((void **)&wm_pows, max(m,n)*sizeof(Complex));
   //Complex* col_temp; cudaMalloc((void **)&col_temp, m*n*sizeof(Complex));
   
   gettimeofday(&before, NULL);
   int radix[3];
   assert(getradix(m, radix)==1);
   cudaMemcpy(radix_device, radix, 3*sizeof(int), cudaMemcpyHostToDevice);
   cuda_fft_init<<<numBlocksCol ,numThreads>>>(m, radix_device, ex_bit_reversal, wm_pows);
   //fft_cuda_basic_kernel<<<(n+4-1)/4, dim3(4,8)>>>(image_device, dft_device, ex_bit_reversal, radix_device, wm_pows, n, m);
   fft_cuda_kernel_unroll<<<n, 64, m*sizeof(Complex)>>>(image_device, dft_device, ex_bit_reversal, radix_device, wm_pows, n, m);

   assert(getradix(n, radix)==1);
   cudaMemcpy(radix_device, radix, 3*sizeof(int), cudaMemcpyHostToDevice);
   cuda_fft_init<<<numBlocksRow ,numThreads>>>(n, radix_device, ex_bit_reversal, wm_pows);
   fft_cuda_kernel_col_unroll<<<m, 64, n*sizeof(Complex)>>>(dft_device, ex_bit_reversal, radix_device, wm_pows, n, m);
   cudaDeviceSynchronize();
   gettimeofday(&after, NULL);
   computation_time += (after.tv_sec + (after.tv_usec / 1000000.0)) -
                      (before.tv_sec + (before.tv_usec / 1000000.0));

   cudaMemcpy(dft_image, dft_device, m*n*sizeof(Complex), cudaMemcpyDeviceToHost);
   cudaFree(radix_device);cudaFree(ex_bit_reversal);cudaFree(wm_pows);

}
/*******************************************************************************************************/



constexpr int DEFAULT_M = 1920;
constexpr int DEFAULT_N = 1080;
/*

__constant__ int _radix_m[3] = {7,1,1}; // 1920
__constant__ int _radix_n[3] = {3,3,1}; // 1080
__constant__ int _ex_bit_reversal_n[DEFAULT_N];
__constant__ int _ex_bit_reversal_m[DEFAULT_M];

__constant__ Complex _wm_pows[DEFAULT_M];
__constant__ Complex _wn_pows[DEFAULT_N];

inline int getrev(int i, int n, int* radix){
   const static int prime[] = {2,3,5};
    int start=0, length = n;
    for(int t=prime_num-1;t>=0;t--){
        int p = prime[t];
        int cnt = radix[t];
        for(int _=0;_<cnt;_++){
            int id = i % p;
            i = i/p;
            length /= p;
            start += id*length; 
        }
    }
    return start;
}

void  fft_cuda_constant_init(){
   int ex_bit_reversal[DEFAULT_M];
   Complex w_pows[DEFAULT_M];
   int radix_m[] = {7,1,1};
   int radix_n[] = {3,3,1};
   for(int i=0;i<DEFAULT_M;i++) ex_bit_reversal[i] = getrev(i, DEFAULT_M, radix_m);
   cudaMemcpyToSymbol(_ex_bit_reversal_m, ex_bit_reversal, sizeof(int)*DEFAULT_M);
   for(int i=0;i<DEFAULT_N;i++) ex_bit_reversal[i] = getrev(i, DEFAULT_N, radix_n);
   cudaMemcpyToSymbol(_ex_bit_reversal_n, ex_bit_reversal, sizeof(int)*DEFAULT_N);

   for(int i=0;i<DEFAULT_M;i++) w_pows[i] = Complex{cos(2*i*pi_cpu/DEFAULT_M), -sin(2*i*pi_cpu/DEFAULT_M)};
   cudaMemcpyToSymbol(_wm_pows, w_pows, sizeof(Complex)*DEFAULT_M);
   for(int i=0;i<DEFAULT_N;i++) w_pows[i] = Complex{cos(2*i*pi_cpu/DEFAULT_N), -sin(2*i*pi_cpu/DEFAULT_N)};
   cudaMemcpyToSymbol(_wn_pows, w_pows, sizeof(Complex)*DEFAULT_N);

}



__global__ void fft_cuda_kernel_constant(int* image, Complex* dft_image)
{
   int m = DEFAULT_M;
   unsigned int r = blockIdx.x;
   unsigned int threads = blockDim.x;
   unsigned int tid = threadIdx.x;


   extern __shared__ Complex dft_line[];
   Complex* dft_res = dft_image + r * m;
   int* image_line = image + r*m;
   for(int j=tid;j<m;j+=threads) dft_line[j] = Complex{(double)image_line[_ex_bit_reversal_m[j]], 0};
   //for(int j=tid;j<m;j+=threads) dft_line[j] = _wm_pows[j];
   __syncthreads();
   fft1_kernel_unroll(dft_line, _wm_pows, _radix_m, m, tid, threads);
   for(int j=tid;j<m;j+=threads) dft_res[j] = dft_line[j];
   
}

__global__ void fft_cuda_kernel_col_constant(Complex* dft_image)
{
   int m = DEFAULT_M, n = DEFAULT_N;
   unsigned int r = blockIdx.x;
   unsigned int threads = blockDim.x;
   unsigned int tid = threadIdx.x;

   extern __shared__ Complex col[];
   Complex* dft_col = dft_image + r;
   for(int i=tid;i<n;i+=threads) col[i] = dft_col[_ex_bit_reversal_n[i]*m];
   __syncthreads();
   fft1_kernel_unroll(col, _wn_pows, _radix_n, n, tid, threads);
   for(int i=tid;i<n;i+=threads){
      dft_col[i*m] = col[i];
   }
}

void  fft2_cuda_constant(int* image, Complex* dft_image, int* image_device, Complex* dft_device) {
   const int n = DEFAULT_N, m = DEFAULT_M;
   struct timeval before, after;

   cudaMemcpy(image_device, image, m*n*sizeof(int), cudaMemcpyHostToDevice);

   // for(int i=0;i<m;i++)printf("%d ", image[i]); printf("\n");

   gettimeofday(&before, NULL);

   //fft_cuda_kernel<<<n, 64, m*sizeof(Complex)>>>(image_device, dft_device, _ex_bit_reversal_m, _radix_m, _wm_pows, n, m);
   fft_cuda_kernel_constant<<<n, 64, m*sizeof(Complex)>>>(image_device, dft_device);
   // cudaMemcpy(dft_image, dft_device, m*n*sizeof(Complex), cudaMemcpyDeviceToHost);
   // for(int i=0;i<m;i++)printf("%lf+%lf ", dft_image[i].real, dft_image[i].imag); printf("\n");

   // fft_cuda_kernel_col<<<m, 64, n*sizeof(Complex)>>>(dft_device, _ex_bit_reversal_n, _radix_n, _wn_pows, n, m);
   fft_cuda_kernel_col_constant<<<m, 64, n*sizeof(Complex)>>>(dft_device);
   cudaDeviceSynchronize();
   
   gettimeofday(&after, NULL);
   computation_time += (after.tv_sec + (after.tv_usec / 1000000.0)) -
                      (before.tv_sec + (before.tv_usec / 1000000.0));


   cudaMemcpy(dft_image, dft_device, m*n*sizeof(Complex), cudaMemcpyDeviceToHost);

}

*/

void inline fft2_cuda_stream(int* image, Complex* dft_image, int* image_device, Complex* dft_device, int n, int m) {
   struct timeval before, after;
   //fft_cuda_constant_init();

   const int nstreams = 2;
   cudaStream_t streams[nstreams];
   int* image_device_stream[nstreams];
   Complex* dft_image_stream[nstreams];
   Complex* dft_image_device_stream[nstreams];
   for(int i=0;i<nstreams;i++) {
      cudaStreamCreate(&streams[i]);
      cudaMalloc((void **)&dft_image_device_stream[i], m*n*sizeof(Complex));
      cudaMalloc((void **)&image_device_stream[i], m*n*sizeof(int));
      dft_image_stream[i] = (Complex *)malloc(m*n*sizeof(Complex));
   }

   int numThreads = 32;
   int numBlocksRow = (n+numThreads-1)/numThreads;
   int numBlocksCol = (m+numThreads-1)/numThreads;
   
   //gettimeofday(&after, NULL);
   int* radix_device_m; cudaMalloc((void **)&radix_device_m, 3*sizeof(int));
   int* ex_bit_reversal_m; cudaMalloc((void **)&ex_bit_reversal_m, m*sizeof(int));
   Complex* wm_pows; cudaMalloc((void **)&wm_pows, m*sizeof(Complex));
   int* radix_device_n; cudaMalloc((void **)&radix_device_n, 3*sizeof(int));
   int* ex_bit_reversal_n; cudaMalloc((void **)&ex_bit_reversal_n, n*sizeof(int));
   Complex* wn_pows; cudaMalloc((void **)&wn_pows, n*sizeof(Complex));

   int radix[3];
   assert(getradix(m, radix)==1);
   cudaMemcpy(radix_device_m, radix, 3*sizeof(int), cudaMemcpyHostToDevice);
   cuda_fft_init<<<numBlocksCol ,numThreads>>>(m, radix_device_m, ex_bit_reversal_m, wm_pows);

   assert(getradix(n, radix)==1);
   cudaMemcpy(radix_device_n, radix, 3*sizeof(int), cudaMemcpyHostToDevice);
   cuda_fft_init<<<numBlocksRow ,numThreads>>>(n, radix_device_n, ex_bit_reversal_n, wn_pows);   
   cudaDeviceSynchronize();

gettimeofday(&before, NULL);

   for(int i=0;i<REP;i++){
      int stream_id = i%nstreams;
      //cudaStreamSynchronize(streams[stream_id]);
      // if(i>=nstreams)
      //    cudaMemcpyAsync(dft_image_stream[i-nstreams], dft_image_device_stream[stream_id], m*n*sizeof(Complex), 
      //                 cudaMemcpyDeviceToHost, streams[stream_id]);
      cudaMemcpyAsync(dft_image_stream[stream_id], dft_image_device_stream[stream_id], m*n*sizeof(Complex), 
                      cudaMemcpyDeviceToHost, streams[stream_id]);

      cudaMemcpyAsync(image_device_stream[stream_id], image, m*n*sizeof(int), cudaMemcpyHostToDevice, streams[stream_id]);
      
      fft_cuda_kernel_unroll<<<n, 64, m*sizeof(Complex), streams[stream_id]>>>(image_device_stream[stream_id], 
                     dft_image_device_stream[stream_id], ex_bit_reversal_m, radix_device_m, wm_pows, n, m);
      fft_cuda_kernel_col_unroll<<<m, 64, n*sizeof(Complex), streams[stream_id]>>>(dft_image_device_stream[stream_id], 
                     ex_bit_reversal_n, radix_device_n, wn_pows, n, m);
      // cudaMemcpyAsync(dft_image_stream[stream_id], dft_image_device_stream[stream_id], m*n*sizeof(Complex), 
      //                cudaMemcpyDeviceToHost, streams[stream_id]);
   }
   for(int i=0;i<nstreams; i++){
      int stream_id = (REP+i)%nstreams;
      // cudaMemcpyAsync(dft_image_stream[REP+i-nstreams], dft_image_device_stream[stream_id], m*n*sizeof(Complex), 
      //                 cudaMemcpyDeviceToHost, streams[stream_id]);
      cudaMemcpyAsync(dft_image_stream[stream_id], dft_image_device_stream[stream_id], m*n*sizeof(Complex), 
                      cudaMemcpyDeviceToHost, streams[stream_id]);
   }
   cudaDeviceSynchronize();
   
   gettimeofday(&after, NULL);
   computation_time += (after.tv_sec + (after.tv_usec / 1000000.0)) -
                      (before.tv_sec + (before.tv_usec / 1000000.0));
   
   cudaFree(radix_device_m);cudaFree(ex_bit_reversal_m);cudaFree(wm_pows);
   cudaFree(radix_device_n);cudaFree(ex_bit_reversal_n);cudaFree(wn_pows);
   memcpy(dft_image, dft_image_stream[0], m*n*sizeof(Complex));
   for(int i=0;i<nstreams;i++) {
      //cudaStreamCreate(&streams[i]);
      free(dft_image_stream[i]);
      cudaFree(dft_image_device_stream[i]);
      cudaFree(image_device_stream[i]);
   }
   
}


int main (int argc, char** argv) {
   struct timeval before, after;
   int m, n;
   int* image = NULL;
   double* GT = NULL;
 
   if (argc > 3) {
      fprintf(stderr, "Usage: %s [martix1] [groundtruthmatrix] \n", argv[0]);
      exit(1);
   }
   else {
      if(argc >= 2){
         image = readImage(argv[1], &n, &m);
      }else{
         n = DEFAULT_N; m = DEFAULT_M;
         // n = 1; m = 120;
         image = generate_mat(n, m);
      }
      if(argc==3){
         int _m, _n;
         GT = readGT(argv[2], &_n, &_m);
         if(n!=_n  or m!=_m){
            printf("Size error! \n"); 
            exit(1);
         }
      }
   }

   Complex* dft_image = (Complex *)malloc(m*n*sizeof(Complex));
   Complex* dft_image2 = (Complex *)malloc(m*n*sizeof(Complex));
   if(dft_image == NULL){
      printf("Out of memory! \n");
      exit(-1);
   }
   Complex* dft_device;
   int* image_device;
   cudaMalloc((void **)&dft_device, m*n*sizeof(Complex));
   cudaMalloc((void **)&image_device, m*n*sizeof(int));


//    cufftHandle plan;
//    cufftPlan2d(&plan, n, m, CUFFT_Z2Z);

//    cufftDoubleComplex *idata, *odata;
//    cudaMalloc((void**)&idata, sizeof(cufftDoubleComplex)*m*n);
//    cudaMalloc((void**)&odata, sizeof(cufftDoubleComplex)*m*n);
//    cufftDoubleComplex *CompData = (cufftDoubleComplex*) malloc(m*n*sizeof(cufftDoubleComplex));
//    for(int i=0; i<m*n; i++){
//       CompData[i].x = image[i]; CompData[i].y=0;
//    }
// cudaMemcpy(idata, CompData, m*n * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
   gettimeofday(&before, NULL); 
   // for(int i=0;i<REP; i++)
   //    fft2_cufft(idata, odata, CompData, &plan,dft_image2, n, m);

   // for(int i=0;i<REP; i++)
   //    fft2_cuda_unroll(image, dft_image2, image_device, dft_device, n, m);

   fft2_cuda_stream(image, dft_image2, image_device, dft_device, n, m);

   // for(int i=0;i<REP; i++)
   //    fft2_cuda(image, dft_image2, image_device, dft_device, n, m);

   // fft_cuda_constant_init();
   // for(int i=0;i<REP; i++)
   //    fft2_cuda_constant(image, dft_image2, image_device, dft_device);

   //fft2_cpu(image, dft_image2, n, m);
   gettimeofday(&after, NULL);
//    cufftDestroy(plan);
// cudaFree(idata);
//    cudaFree(odata);
//    free(CompData);

   fft2_cpu(image, dft_image, n, m);

   // printf("GT:\n");
   // for(int i=0;i<m;i++)printf("%lf+%lfj ", dft_image[i].real, dft_image[i].imag); printf("\n");

   cmp((double*)dft_image, (double*)dft_image2, n*m*2);
   if(argc == 3){
      cmp((double*)dft_image, GT, n*m*2);
      free(GT);
   }

   printf("Total exec  time: %.6f seconds \n", ((after.tv_sec + (after.tv_usec / 1000000.0)) -
               (before.tv_sec + (before.tv_usec / 1000000.0)))/REP);
   printf("Computation time: %.6f seconds \n", computation_time/REP);
   free(dft_image);
   free(dft_image2);
   free(image);
   cudaFree(image_device);
   cudaFree(dft_device);
   return 0;
}

