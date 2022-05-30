#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>



#define REP 100

#define N 1000000

int inline check_prime(long long n){
    for(long long i=2;i*i<=n; i++){
        if(n%i==0) return 0;
    }
    return 1;
}

int main (int argc, char** argv) {
    int proc_count = 1;
    struct timeval before, after;
    if (argc < 2) {
        fprintf(stderr, "Usage: %s [thread_num] [length]\n", argv[0]);
        exit(1);
    }
    long long n = N;
    proc_count =  atoi(argv[1]);
    if(argc >= 3) n = strtoll(argv[2], NULL, 10);
    omp_set_dynamic(0);
    omp_set_num_threads(proc_count);
    long long cnt = 0;

    gettimeofday(&before, NULL); 
    // fix: # pragma omp parallel for reduction(+:cnt) schedule(static, 32)
    # pragma omp parallel for reduction(+:cnt)
    for(long long i=2;i<=n;i++){
        if(check_prime(i)) cnt++;
    }
    gettimeofday(&after, NULL);
    printf("Number of primes in range 1 - %lld: %lld\n", n, cnt);

    printf("Reference code: %10.6f seconds \n", ((after.tv_sec + (after.tv_usec / 1000000.0)) -
                (before.tv_sec + (before.tv_usec / 1000000.0))));

    return 0;
}

