#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define N_EXP 30
#define SAMPLE 100000000.0
#define ARRARY_SIZE 10

void inline atomic_add(size_t n, float a)
{
  printf("#atomic_add (ns)\n");
  struct timespec before, after;
  for (size_t i = 0; i < n; i++)
  {

    timespec_get(&before, TIME_UTC);
#pragma omp parallel
    {
      for (size_t j = 0; j < SAMPLE; j++)
      {
#pragma omp atomic
        a++;
      }
    }
    timespec_get(&after, TIME_UTC);
    long long sec = (after.tv_sec - before.tv_sec) * 1000000000 + (after.tv_nsec - before.tv_nsec);
    printf("%f\n", sec / SAMPLE);
  }
}

void inline cache(size_t n, float *arrary)
{
  printf("#cache (ns)\n");
  float a = 0;
  struct timespec before, after;
  for (size_t i = 0; i < n; i++)
  {
    for (size_t stride = 1; stride < ARRARY_SIZE; stride = stride * 2)
    {
      timespec_get(&before, TIME_UTC);
      for (size_t j = 0; j < SAMPLE; j++)
      {
        a = arrary[stride * 0];
        a = arrary[stride * 1];
        a = arrary[stride * 2];
        a = arrary[stride * 3];
        a = arrary[stride * 4];
        a = arrary[stride * 5];
        a = arrary[stride * 6];
        a = arrary[stride * 7];
        a = arrary[stride * 8];
        a = arrary[stride * 9];
        a = arrary[stride * 10];
      }
      timespec_get(&after, TIME_UTC);
      long long sec = (after.tv_sec - before.tv_sec) * 1000000000 + (after.tv_nsec - before.tv_nsec);
      printf("%d,%f\n", stride, sec / SAMPLE);
    }
  }
  printf("%f\n", arrary[0]);
}

void inline add_single(size_t n, float a)
{
  printf("#add_single (ns)\n");
  struct timespec before, after;
  for (size_t i = 0; i < n; i++)
  {

    timespec_get(&before, TIME_UTC);
    for (size_t j = 0; j < SAMPLE; j++)
    {
      a += a*a;
    }
    timespec_get(&after, TIME_UTC);
    long long sec = (after.tv_sec - before.tv_sec) * 1000000000 + (after.tv_nsec - before.tv_nsec);
    printf("%f\n", sec / SAMPLE);
  }
}

void inline add_multiple(size_t n, float a, float b, float c, float d)
{
  printf("#add_multiple (ns)\n");
  struct timespec before, after;
  for (size_t i = 0; i < n; i++)
  {

    timespec_get(&before, TIME_UTC);
    for (size_t j = 0; j < SAMPLE; j++)
    {
      a += a * a;
      b += b * b;
      c += c * c;
      d += d * d;
    }
    timespec_get(&after, TIME_UTC);
    long long sec = (after.tv_sec - before.tv_sec) * 1000000000 + (after.tv_nsec - before.tv_nsec);
    printf("%f\n", sec / SAMPLE);
  }
}

int main(int argc, char **argv)
{
  printf("benchmark start");
  float a = atof(argv[1]);
  float b = atof(argv[2]);
  float c = atof(argv[3]);
  float d = atof(argv[4]);
  float array[ARRARY_SIZE];
  for (size_t i = 0; i < ARRARY_SIZE; i++)
  {
    array[i] = 2;
  }

  add_multiple(N_EXP, a, b, c, d);
  add_single(N_EXP, a);
  atomic_add(N_EXP, a);
}
