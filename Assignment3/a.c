
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>
#include <likwid.h>
#define SLEEPTIME 2
int main(int argc, char* argv[])
{
    int i, g;
    int nevents = 10;
    double events[10];
    double time;
    int count;
    LIKWID_MARKER_INIT;
    #pragma omp parallel
    {
        LIKWID_MARKER_THREADINIT;
        LIKWID_MARKER_REGISTER("example");
    }
    for (g=0;g < 1; g++)
    {
        #pragma omp parallel
        {
            printf("Thread %d sleeps now for %d seconds\n", omp_get_thread_num(), SLEEPTIME);
            LIKWID_MARKER_START("example");
            sleep(SLEEPTIME);
            LIKWID_MARKER_STOP("example");
            printf("Thread %d wakes up again\n", omp_get_thread_num());
            LIKWID_MARKER_GET("example", &nevents, events, &time, &count);
            printf("Region example measures %d events, total measurement time is %f\n", nevents, time);
            printf("The region was called %d times\n", count);
            for (i = 0; i < nevents; i++)
            {
                printf("Event %d: %f\n", i, events[i]);
            }
            LIKWID_MARKER_SWITCH;
        }
    }
    LIKWID_MARKER_CLOSE;
    return 0;
}
