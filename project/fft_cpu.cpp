/* 
*   Matrix Market I/O library for ANSI C
*
*   See http://math.nist.gov/MatrixMarket for details.
*
*
*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <sys/time.h>
#include <math.h>
#include "fft_cpu.h"

const double pi = acos(-1);

int* readImage(const char* filename, int *N, int *M){
    int* img;
    FILE* fp = fopen(filename, "r");
    if(fp==NULL){
        printf("image file error!\n");
        exit(-1);
    }
    if(fscanf(fp, "%d%d",N, M)<=1){
        exit(-1);
    }
    int size = (*N) * (*M);
    img = (int*)malloc(size*sizeof(int));
    if(img==NULL){
        printf("Out of memory! \n");
        exit(-1);
    }
    for(int i=0;i<size; i++){
        if(fscanf(fp, "%d", &img[i])<=0){
            exit(-1);
        }
    }
    fclose(fp);
    
    return img;
}

double* readGT(const char* filename, int *N, int *M){
    double* img;
    FILE* fp = fopen(filename, "r");
    if(fp==NULL){
        printf("GT file error!\n");
        exit(-1);
    }
    if(fscanf(fp, "%d%d",N, M)<=1){
        exit(-1);
    }

    int size = (*N) * (*M) * 2;
    img = (double*)malloc(size*sizeof(double));
    if(img==NULL){
        printf("Out of memory! \n");
        exit(-1);
    }
    for(int i=0;i<size; i++){
        if(fscanf(fp, "%lf", &img[i])<=0){
            exit(-1);
        }
    }
    fclose(fp);
    
    return img;
}

inline Complex operator *(const Complex a, const Complex b){
    return Complex{a.real*b.real-a.imag*b.imag, a.real*b.imag+a.imag*b.real};
} 

inline Complex operator *(const Complex a, double b){
    return Complex{a.real*b, a.imag*b};
} 

inline Complex operator +(const Complex a, const Complex b){
    return Complex{a.real + b.real, a.imag + b.imag};
} 

inline Complex wx(int x){
    return Complex{cos(2*pi/x), -sin(2*pi/x)};
}

void fft2_basic(int* image, Complex* dft_image, int n, int m){
    Complex wm = wx(m), wn = wx(n);
    Complex wm_pows[m];
    Complex wn_pows[n];
    wm_pows[0] = wn_pows[0] = Complex{1,0};
    for(int i=1;i<m;i++) wm_pows[i] = wm_pows[i-1]*wm;
    for(int i=1;i<n;i++) wn_pows[i] = wn_pows[i-1]*wn;

    for(int i=0;i<n;i++){
        int* image_line = image + i*m;
        Complex* dft_line = dft_image + i*m;
        for(int j=0;j<m;j++){
            Complex res{0,0};
            for(int k=0;k<m;k++){
                res = res + wm_pows[j*k%m]*image_line[k];
            }
            dft_line[j] = res;
        }
        
    }
    Complex col[n];
    for(int j=0;j<m;j++){
        Complex* dft_col = dft_image + j;
        for(int i=0;i<n;i++) col[i] = dft_col[i*m];
        for(int i=0;i<n;i++){
            Complex res{0,0};
            for(int k=0;k<n;k++){
                res = res + wn_pows[i*k%n]*col[k];
            }
            dft_col[i*m] = res;
        }
    }

}


void fft2_cpu(int* image, Complex* dft_image, int n, int m){
    Complex wm = wx(m), wn = wx(n);
    Complex wm_pows[m];
    Complex wn_pows[n];
    wm_pows[0] = wn_pows[0] = Complex{1,0};
    for(int i=1;i<m;i++) wm_pows[i] = wm_pows[i-1]*wm;
    for(int i=1;i<n;i++) wn_pows[i] = wn_pows[i-1]*wn;

    for(int i=0;i<n;i++){
        int* image_line = image + i*m;
        Complex* dft_line = dft_image + i*m;
        for(int j=0;j<m;j++){
            Complex res{0,0};
            for(int k=0;k<m;k++){
                res = res + wm_pows[j*k%m]*image_line[k];
            }
            dft_line[j] = res;
        }
        
    }
    Complex col[n];
    for(int j=0;j<m;j++){
        Complex* dft_col = dft_image + j;
        for(int i=0;i<n;i++) col[i] = dft_col[i*m];
        for(int i=0;i<n;i++){
            Complex res{0,0};
            for(int k=0;k<n;k++){
                res = res + wn_pows[i*k%n]*col[k];
            }
            dft_col[i*m] = res;
        }
    }

}
