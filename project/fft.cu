#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "fft_cpu.h"


#define REP 1

constexpr int DEFAULT_M = 1920;
constexpr int DEFAULT_N = 1080;

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
   if(dft_image == NULL){
      printf("Out of memory! \n");
      exit(-1);
   }
   gettimeofday(&before, NULL); 
   fft2_basic(image, dft_image, n, m);

   gettimeofday(&after, NULL);
   if(argc == 3){
      cmp((double*)dft_image, GT, n*m*2);
      free(GT);
   }

   printf("Total exec  time: %.6f seconds \n", ((after.tv_sec + (after.tv_usec / 1000000.0)) -
               (before.tv_sec + (before.tv_usec / 1000000.0)))/REP);
   free(dft_image);
   return 0;
}

