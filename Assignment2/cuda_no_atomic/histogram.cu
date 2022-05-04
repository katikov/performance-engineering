#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "timer.h"
#include <unistd.h>
#include <getopt.h>

using namespace std;

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
void die(const char *msg){
    if (errno != 0) 
        perror(msg);
    else
        fprintf(stderr, "error: %s\n", msg);
    exit(1);
}   

void generate_image(int num_rows, int num_cols, unsigned char * image){
    for (int i = 0; i < num_cols * num_rows; ++i)
    {
        image[i] = (unsigned char) (rand() % 256); //255 + 1 for num bins
    }
}

void read_image(const char * image_path, int num_rows, int num_cols, unsigned char * image){
	char format[3];
    FILE *f;
    unsigned imgw, imgh, maxv, v;
    size_t i;

	printf("Reading PGM data from %s...\n", image_path);

	if (!(f = fopen(image_path, "r"))) die("fopen");

	fscanf(f, "%2s", format);
    if (format[0] != 'P' || format[1] != '2') die("only ASCII PGM input is supported");
    
    if (fscanf(f, "%u", &imgw) != 1 ||
        fscanf(f, "%u", &imgh) != 1 ||
        fscanf(f, "%u", &maxv) != 1) die("invalid input");

    if (imgw != num_cols || imgh != num_rows) {
        fprintf(stderr, "input data size (%ux%u) does not match cylinder size (%zux%zu)\n",
                imgw, imgh, num_cols, num_rows);
        die("invalid input");
    }

    for (i = 0; i < num_cols * num_rows; ++i)
    {
        if (fscanf(f, "%u", &v) != 1) die("invalid data");
        image[i] = (unsigned char) (((int)v * 255) / maxv); //255 for num bins
    }
    fclose(f);
}

static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}

constexpr int threadBlockSize = 32;
constexpr int hist_size = 256;
__global__ void histogramKernelLayout2(unsigned char* image, long img_size, unsigned int* histogram) {
// insert operation here
    int blockThreadIdx = threadIdx.x;
    int imageThreadIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int step = (gridDim.x * blockDim.x);
    __shared__ unsigned short myHisto[hist_size/2][threadBlockSize][2]; // 256

    for(int i=0;i<hist_size/2;i++){
        myHisto[i][blockThreadIdx][0] = 0;
        myHisto[i][blockThreadIdx][1] = 0;

    }
    for (int i = imageThreadIdx; i < img_size; i += step)
        ++myHisto[image[i]/2][blockThreadIdx][image[i]%2];
    __syncthreads();
    for(int s = blockDim.x/2; s>0; s/=2){
        if(blockThreadIdx<s){
            for(int i=0;i<hist_size/2; i++){
                myHisto[i][blockThreadIdx][0] += myHisto[i][blockThreadIdx+s][0];
                myHisto[i][blockThreadIdx][1] += myHisto[i][blockThreadIdx+s][1];
            }
        }
        __syncthreads();
    }
    histogram = histogram + hist_size * blockIdx.x;
    for(int i=blockThreadIdx;i<hist_size/2;i+=threadBlockSize){
        histogram[i*2] = myHisto[i][0][0];
        histogram[i*2+1] = myHisto[i][0][1];
    }

}

__global__ void histogramKernel(unsigned char* image, long img_size, unsigned int* histogram) {
// insert operation here
    int blockThreadIdx = threadIdx.x;
    int imageThreadIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int step = (gridDim.x * blockDim.x);
    __shared__ unsigned short myHisto[hist_size][threadBlockSize]; // 256

    for(int i=0;i<hist_size;i++){
        myHisto[i][blockThreadIdx] = 0;
    }
    for (int i = imageThreadIdx; i < img_size; i += step)
        ++myHisto[image[i]][blockThreadIdx];
    __syncthreads();
    for(int s = blockDim.x/2; s>0; s/=2){
        if(blockThreadIdx<s){
            for(int i=0;i<hist_size; i++){
                myHisto[i][blockThreadIdx] += myHisto[i][blockThreadIdx+s];
            }
        }
        __syncthreads();
    }
    histogram = histogram + hist_size * blockIdx.x;
    for(int i=blockThreadIdx;i<hist_size;i+=threadBlockSize)
        histogram[i] = myHisto[i][0];

}


__global__ void reduceHistogramKernel(unsigned int* histogram, int stride, int endBlock) {
    long blockThreadIdx = threadIdx.x;
    if(stride + blockIdx.x < endBlock) {
        int currThread = hist_size * blockIdx.x + blockThreadIdx;
        int nextThread = hist_size * ( blockIdx.x + stride) + blockThreadIdx;
        histogram[currThread] += histogram[nextThread];
    }
}

void histogramCuda(unsigned char* image, long img_size, unsigned int* histogram, int hist_size) {
    int pixelPerThread = 512;
    int numOfBlock = ((img_size + threadBlockSize-1)/threadBlockSize + pixelPerThread - 1)/pixelPerThread;
    // allocate the vectors on the GPU
    unsigned char* deviceImage = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceImage, img_size * sizeof(unsigned char)));
    if (deviceImage == NULL) {
        cout << "could not allocate memory!" << endl;
        return;
    }
    unsigned int* deviceHisto = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceHisto, numOfBlock*hist_size * sizeof(unsigned int)));
    if (deviceHisto == NULL) {
        checkCudaCall(cudaFree(deviceImage));
        cout << "could not allocate memory!" << endl;
        return;
    }

    timer kernelTime1 = timer("kernelTime1");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceImage, image, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    histogramKernel<<<numOfBlock, threadBlockSize>>>(deviceImage, img_size, deviceHisto);
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());
    /* reduce multiple histograms to one histogram */
    int stride = ceil((double)numOfBlock/2);
    int endBlock = numOfBlock;
    kernelTime1.start();
    while(endBlock > 1){
        reduceHistogramKernel<<<endBlock-stride, hist_size>>>(deviceHisto, stride, endBlock);
        cudaDeviceSynchronize();
        checkCudaCall(cudaGetLastError());
        endBlock = stride;
        stride = ceil((double)stride/2);
    }
    kernelTime1.stop();
    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(histogram, deviceHisto, hist_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceImage));
    checkCudaCall(cudaFree(deviceHisto));

    cout << "histogram (kernel): \t\t" << kernelTime1  << endl;
    cout << "histogram (memory): \t\t" << memoryTime << endl;
    cout << "histogram (total):  \t\t  = " << (kernelTime1.getTimeInSeconds() + memoryTime.getTimeInSeconds()) << " seconds" << endl;
}

void histogramSeq(unsigned char* image, long img_size, unsigned int* histogram, int hist_size) {
  int i; 

  timer sequentialTime = timer("Sequential");
  
  for (i=0; i<hist_size; i++) histogram[i]=0;

  sequentialTime.start();
  for (i=0; i<img_size; i++) {
	histogram[image[i]]++;
  }
  sequentialTime.stop();
  
  cout << "histogram (sequential): \t\t" << sequentialTime << endl;

}

int main(int argc, char* argv[]) {
    int c;
    int seed = 42;
    const char *image_path = 0;
    image_path ="../../../images/pat1_100x150.pgm";
    int gen_image = 0;
    int debug = 0;

    int num_rows = 150;
    int num_cols = 100;

    /* Read command-line options. */
    while((c = getopt(argc, argv, "s:i:rp:n:m:g")) != -1) {
        switch(c) {
            case 's':
                seed = atoi(optarg);
                break;
            case 'i':
            	image_path = optarg;
            	break;
            case 'r':
            	gen_image = 1;
            	break;
            case 'n':
            	num_rows = strtol(optarg, 0, 10);
            	break;
            case 'm':
				num_cols = strtol(optarg, 0, 10);
				break;
			case 'g':
				debug = 1;
				break;
            case '?':
                fprintf(stderr, "Unknown option character '\\x%x'.\n", optopt);
                return -1;
            default:
                return -1;
        }
    }

    int hist_size = 256;
    long img_size = num_rows*num_cols;

    unsigned char *image = (unsigned char *)malloc(img_size * sizeof(unsigned char)); 
    unsigned int *histogramS = (unsigned int *)malloc(hist_size * sizeof(unsigned int));     
    unsigned int *histogram = (unsigned int *)malloc(hist_size * sizeof(unsigned int));

    /* Seed such that we can always reproduce the same random vector */
    if (gen_image){
    	srand(seed);
    	generate_image(num_rows, num_cols, image);
    }else{
    	read_image(image_path,num_rows, num_cols, image);
    }

    histogramSeq(image, img_size, histogramS, hist_size);
    histogramCuda(image, img_size, histogram, hist_size);
    
    // verify the resuls
    for(int i=0; i<hist_size; i++) {
	  if (histogram[i]!=histogramS[i]) {
            cout << "error in results! Bin " << i << " is "<< histogram[i] << ", but should be " << histogramS[i] << endl; 
            exit(1);
        }
    }
    cout << "results OK!" << endl;
     
    free(image);
    free(histogram);
    free(histogramS);         
    
    return 0;
}