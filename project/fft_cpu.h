/* 
*   Matrix Market I/O library for ANSI C
*
*   See http://math.nist.gov/MatrixMarket for details.
*
*
*/

#ifndef FFT_CPU_H
#define FFT_CPU_H
struct Complex{
	double real, imag;
};

int* readImage(const char* filename, int *N, int *M);
double* readGT(const char* filename, int *N, int *M);

void fft2_basic(int* image, Complex* dft_image, int n, int m);
void fft2_cpu(int* image, Complex* dft_image, int n, int m);

#endif
