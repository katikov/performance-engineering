#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include "fft_cpu.h"


#define REP 10

constexpr int DEFAULT_M = 1920;
constexpr int DEFAULT_N = 1080;
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

inline __device__ Complex operator *(const Complex a, const Complex b){
    return Complex{a.real*b.real-a.imag*b.imag, a.real*b.imag+a.imag*b.real};
} 

inline __device__ Complex operator *(const Complex a, double b){
    return Complex{a.real*b, a.imag*b};
} 

inline __device__ Complex operator +(const Complex a, const Complex b){
    return Complex{a.real + b.real, a.imag + b.imag};
} 

inline __device__ Complex operator -(const Complex a, const Complex b){
    return Complex{a.real - b.real, a.imag - b.imag};
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
                  __syncthreads();
            }
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

int main()
    int x, y, count;

    // Open an input pipe from ffmpeg and an output pipe to a second instance of ffmpeg
    FILE* pipein = popen("ffmpeg -i clouds.mp4 -f image2pipe -pix_fmt rgb24 -vcodec rawvideo -", "rb");
    FILE* pipeout = popen("ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt rgb24 -s 1920x1080 -r 24 -i - -f mp4 -q:v 5 -an -vcodec mpeg4 output.mp4", "wb");

    // Process video frames
    while (1)
    {
        // Read a frame from the input pipe into the buffer
        count = fread(frame, 1, H * W * 3, pipein);

        // If we didn't get a frame of video, we're probably at the end
        if (count != H * W * 3) break;

        // Process this frame
        for (y = 0; y < H; ++y) for (x = 0; x < W; ++x)
        {
            // Invert each colour component in every pixel
            frame[y][x][0] = 255 - frame[y][x][0]; // red
            frame[y][x][1] = 255 - frame[y][x][1]; // green
            frame[y][x][2] = 255 - frame[y][x][2]; // blue
        }

        //Write this frame to the output pipe
        fwrite(frame, 1, H * W * 3, pipeout);
    }

    // Flush and close input and output pipes
    fflush(pipein);
    pclose(pipein);
    fflush(pipeout);
    pclose(pipeout);
    return 0;
}

int main2 (int argc, char** argv) {
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
         //n = 6; m = 10;
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

   gettimeofday(&before, NULL); 
   //fft2_basic(image, dft_image2, n, m);
   for(int i=0;i<REP; i++)
      fft2_cuda_basic(image, dft_image2, image_device, dft_device, n, m);
   gettimeofday(&after, NULL);


   fft2_cpu(image, dft_image, n, m);
   //printf("GT:\n");
   //for(int i=0;i<n;i++)printf("%lf+%lfj ", dft_image[i].real, dft_image[i].imag); printf("\n");
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
   cudaFree(image_device);
   cudaFree(dft_device);
   return 0;
}

