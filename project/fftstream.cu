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
         printf("error in results!\n");
         printf("%f %f\n", A[i], B[i]);
         return;
      }
   }
   printf("results OK!\n");
}

unsigned char* generate_mat(int n, int m) {
   srand(42); // fixed seed
   int size = m*n;
   unsigned char* img = (unsigned char*)malloc(size*sizeof(char));
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
   return Complex{cos(2*k*pi/m), -sin(2*k*pi/m)};
} 


__global__ void fft_cuda_basic_kernel(unsigned char* image, Complex* dft_image, int* ex_bit_reversal, 
             int* radix, Complex* wm_pows, int n, int m)
{
   unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int threads = blockDim.y;
   unsigned int tid = threadIdx.y;
   if(r < n){
      Complex* dft_line = dft_image + r * m;
      unsigned char* image_line = image + r*m;
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


void inline fft2_cuda_basic(unsigned char* image, Complex* dft_image, unsigned char* image_device, Complex* dft_device, int n, int m) {
   int numThreads = 32;
   int numBlocksRow = (n+numThreads-1)/numThreads;
   int numBlocksCol = (m+numThreads-1)/numThreads;
   struct timeval before, after;
   cudaMemcpy(image_device, image, m*n*sizeof(unsigned char), cudaMemcpyHostToDevice);
   int* radix_device; cudaMalloc((void **)&radix_device, 3*sizeof(int));
   int* ex_bit_reversal; cudaMalloc((void **)&ex_bit_reversal, max(m,n)*sizeof(int));
   Complex* wm_pows; cudaMalloc((void **)&wm_pows, max(m,n)*sizeof(Complex));
   Complex* col_temp; cudaMalloc((void **)&col_temp, m*n*sizeof(Complex));
   gettimeofday(&before, NULL);
   int radix[3];
   assert(getradix(m, radix)==1);
   cudaMemcpy(radix_device, radix, 3*sizeof(int), cudaMemcpyHostToDevice);
   cuda_fft_init<<<numBlocksCol ,numThreads>>>(m, radix_device, ex_bit_reversal, wm_pows);
   fft_cuda_basic_kernel<<<(n+4-1)/4, dim3(4,8)>>>(image_device, dft_device, ex_bit_reversal, radix_device, wm_pows, n, m);

   assert(getradix(n, radix)==1);
   cudaMemcpy(radix_device, radix, 3*sizeof(int), cudaMemcpyHostToDevice);
   cuda_fft_init<<<numBlocksRow ,numThreads>>>(n, radix_device, ex_bit_reversal, wm_pows);
   fft_cuda_basic_kernel_col<<<(m+4-1)/4, dim3(4,8)>>>(col_temp, dft_device, ex_bit_reversal, radix_device, wm_pows, n, m);


   cudaDeviceSynchronize();
   gettimeofday(&after, NULL);
   computation_time += (after.tv_sec + (after.tv_usec / 1000000.0)) -
                      (before.tv_sec + (before.tv_usec / 1000000.0));
   cudaMemcpy(dft_image, dft_device, m*n*sizeof(Complex), cudaMemcpyDeviceToHost);
   cudaFree(radix_device);
   cudaFree(ex_bit_reversal);
   cudaFree(wm_pows);
   cudaFree(col_temp);
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

__global__ void fft_cuda_kernel(unsigned char* image, Complex* dft_image, int* ex_bit_reversal, 
             int* radix, Complex* wm_pows, int n, int m)
{
   unsigned int r = blockIdx.x;
   unsigned int threads = blockDim.x;
   unsigned int tid = threadIdx.x;


   extern __shared__ Complex dft_line[];
   Complex* dft_res = dft_image + r * m;
   unsigned char* image_line = image + r*m;
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


void inline fft2_cuda(unsigned char* image, Complex* dft_image, unsigned char* image_device, Complex* dft_device, int n, int m) {
   int numThreads = 32;
   int numBlocksRow = (n+numThreads-1)/numThreads;
   int numBlocksCol = (m+numThreads-1)/numThreads;
   struct timeval before, after;
   cudaMemcpy(image_device, image, m*n*sizeof(char), cudaMemcpyHostToDevice);
   int* radix_device; cudaMalloc((void **)&radix_device, 3*sizeof(int));
   int* ex_bit_reversal; cudaMalloc((void **)&ex_bit_reversal, max(m,n)*sizeof(int));
   Complex* wm_pows; cudaMalloc((void **)&wm_pows, max(m,n)*sizeof(Complex));
   
   gettimeofday(&before, NULL);
   int radix[3];
   assert(getradix(m, radix)==1);
   cudaMemcpy(radix_device, radix, 3*sizeof(int), cudaMemcpyHostToDevice);
   cuda_fft_init<<<numBlocksCol ,numThreads>>>(m, radix_device, ex_bit_reversal, wm_pows);
   fft_cuda_kernel<<<n, 64, m*sizeof(Complex)>>>(image_device, dft_device, ex_bit_reversal, radix_device, wm_pows, n, m);

   assert(getradix(n, radix)==1);
   cudaMemcpy(radix_device, radix, 3*sizeof(int), cudaMemcpyHostToDevice);
   cuda_fft_init<<<numBlocksRow ,numThreads>>>(n, radix_device, ex_bit_reversal, wm_pows);
   fft_cuda_kernel_col<<<m, 64, n*sizeof(Complex)>>>(dft_device, ex_bit_reversal, radix_device, wm_pows, n, m);

   cudaDeviceSynchronize();
   gettimeofday(&after, NULL);
   computation_time += (after.tv_sec + (after.tv_usec / 1000000.0)) -
                      (before.tv_sec + (before.tv_usec / 1000000.0));
   cudaMemcpy(dft_image, dft_device, m*n*sizeof(Complex), cudaMemcpyDeviceToHost);
   cudaFree(radix_device);
   cudaFree(ex_bit_reversal);
   cudaFree(wm_pows);
}
/*********************************************************************************************************/

__global__ void fft_cuda_kernel_unroll(unsigned char* image, Complex* dft_image, int* ex_bit_reversal, 
             int* radix, Complex* wm_pows, int n, int m)
{
   unsigned int r = blockIdx.x;
   unsigned int threads = blockDim.x;
   unsigned int tid = threadIdx.x;


   extern __shared__ Complex dft_line[];
   Complex* dft_res = dft_image + r * m;
   unsigned char* image_line = image + r*m;
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

void inline fft2_cuda_unroll(unsigned char* image, Complex* dft_image, unsigned char* image_device, Complex* dft_device, int n, int m) {
   int numThreads = 32;
   int numBlocksRow = (n+numThreads-1)/numThreads;
   int numBlocksCol = (m+numThreads-1)/numThreads;
   struct timeval before, after;

   cudaMemcpy(image_device, image, m*n*sizeof(char), cudaMemcpyHostToDevice);
   int* radix_device; cudaMalloc((void **)&radix_device, 3*sizeof(int));
   int* ex_bit_reversal; cudaMalloc((void **)&ex_bit_reversal, max(m,n)*sizeof(int));
   Complex* wm_pows; cudaMalloc((void **)&wm_pows, max(m,n)*sizeof(Complex));
   
   gettimeofday(&before, NULL);
   int radix[3];
   assert(getradix(m, radix)==1);
   cudaMemcpy(radix_device, radix, 3*sizeof(int), cudaMemcpyHostToDevice);
   cuda_fft_init<<<numBlocksCol ,numThreads>>>(m, radix_device, ex_bit_reversal, wm_pows);
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

unsigned char frame_HW3[DEFAULT_N][DEFAULT_M][3];
unsigned char frame_3HW[3][DEFAULT_N][DEFAULT_M];

const int nstreams = 3;
cudaStream_t streams[nstreams];
int inline fft2_cuda_stream() {
   const int n = 1080, m = 1920;
   const int H=n, W=m;
   struct timeval before, after;
   unsigned char* image_device_stream[nstreams];
   Complex* dft_image_device_stream[nstreams];
   Complex* dft_image_stream[nstreams];

   for(int i=0;i<nstreams;i++) {
      cudaMalloc((void **)&dft_image_device_stream[i], m*n*sizeof(Complex));
      cudaMalloc((void **)&image_device_stream[i], m*n*sizeof(char));
      dft_image_stream[i] = (Complex *)malloc(m*n*sizeof(Complex));
   }

   int numThreads = 32;
   int numBlocksRow = (n+numThreads-1)/numThreads;
   int numBlocksCol = (m+numThreads-1)/numThreads;
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
    int x, y, count, frames=0;
    // Open an input pipe from ffmpeg and an output pipe to a second instance of ffmpeg
    FILE* pipein = popen("ffmpeg -i \"./validation/Validation (15).mp4\" -f image2pipe -pix_fmt rgb24 -vcodec rawvideo - 2> /dev/null", "r");
    FILE* pipeout = popen("ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt rgb24 -s 1920x1080 -r 24 -i - -f mp4 -q:v 5 -an -vcodec mpeg4 output.mp4", "w");

    // Process video frames
    
    while (1)
    {
        count = fread(frame_HW3, 1, H * W * 3, pipein);
        // If we didn't get a frame of video, we're probably at the end
        if (count != H * W * 3) break;
        for(int i=0;i<H;i++) for(int j=0;j<W;j++){
           frame_3HW[0][i][j] =  frame_HW3[i][j][0];
           frame_3HW[1][i][j] =  frame_HW3[i][j][1];
           frame_3HW[2][i][j] =  frame_HW3[i][j][2];
        }
        cudaDeviceSynchronize();
        for(int i=0;i<3;i++){
            cudaMemcpyAsync(image_device_stream[i], frame_3HW[i], m*n*sizeof(char), cudaMemcpyHostToDevice, streams[i]);
            fft_cuda_kernel_unroll<<<n, 64, m*sizeof(Complex), streams[i]>>>(image_device_stream[i], 
                           dft_image_device_stream[i], ex_bit_reversal_m, radix_device_m, wm_pows, n, m);
            fft_cuda_kernel_col_unroll<<<m, 64, n*sizeof(Complex), streams[i]>>>(dft_image_device_stream[i], 
                           ex_bit_reversal_n, radix_device_n, wn_pows, n, m);

            // retrieving back
            // cudaMemcpyAsync(dft_image_stream[i], dft_image_device_stream[i], m*n*sizeof(Complex), 
            //                cudaMemcpyDeviceToHost, streams[i]);
        }
        frames++;
    }
    // Flush and close input and output pipes

   cudaDeviceSynchronize();
   
   gettimeofday(&after, NULL);
   computation_time += (after.tv_sec + (after.tv_usec / 1000000.0)) -
                      (before.tv_sec + (before.tv_usec / 1000000.0));
   
   cudaFree(radix_device_m);cudaFree(ex_bit_reversal_m);cudaFree(wm_pows);
   cudaFree(radix_device_n);cudaFree(ex_bit_reversal_n);cudaFree(wn_pows);

   for(int i=0;i<nstreams;i++) {
      free(dft_image_stream[i]);
      cudaFree(dft_image_device_stream[i]);
      cudaFree(image_device_stream[i]);
   }
    fflush(pipein);
    pclose(pipein);
    fflush(pipeout);
    pclose(pipeout);
    return frames;
}


int main (int argc, char** argv) {
   struct timeval before, after;

   gettimeofday(&before, NULL); 

   int frames = fft2_cuda_stream();
   gettimeofday(&after, NULL);
   printf("Total frames: %d\n", frames);
   printf("Computation time: %.6f seconds \n", ((after.tv_sec + (after.tv_usec / 1000000.0)) -
               (before.tv_sec + (before.tv_usec / 1000000.0))));
   return 0;
}

