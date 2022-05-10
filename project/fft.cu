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

void generate_mat(int m, int n, int p, float *A, float *B) {
  int i;

  for (i=0; i<(m*n); i++) A[i] = 1; //i/10; 
  for (i=0; i<(n*p); i++) B[i] = 1; //i/5;

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
         // TODO: generate image
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

   cmp((double*)dft_image, GT, n*m*2);


   printf("Total exec  time: %.6f seconds \n", ((after.tv_sec + (after.tv_usec / 1000000.0)) -
               (before.tv_sec + (before.tv_usec / 1000000.0)))/REP);
   free(dft_image);
   return 0;
}

