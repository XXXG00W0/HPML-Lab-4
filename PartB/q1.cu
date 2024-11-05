#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "addKernel.h"

int main(int argc, char** argv){
    long K;

    if (argc != 2) {
        printf("Usage: %s <K> \n", argv[0]);
        printf("Example: %s 1\n", argv[0]);
        printf("<K>: K million of elements in the array\n");
        exit(1);
    } else {
        sscanf(argv[1], "%ld", &K);
    }

    long arrLength = K * 1000000;
    float* vectorA = (float*) malloc(arrLength * sizeof(float));
    float* vectorB = (float*) malloc(arrLength * sizeof(float));

    long i;
    for (i = 0; i < arrLength; ++i){
        vectorA[i] = (float)i;
        vectorB[i] = (float)(arrLength - i);
    }
    
    float* vectorC = (float*) malloc(arrLength * sizeof(float));

    // Warm up
    arrayAddHost(vectorA, vectorB, vectorC, arrLength);

    // Measure time
    initialize_timer();
    start_timer();
    arrayAddHost(vectorA, vectorB, vectorC, arrLength);
    stop_timer();
    printf("Question 1: Array length: %ld, Elapsed time: %g s\n", arrLength, elapsed_time());

    // Verify the result
    for (i = 0; i < arrLength; ++i){
        if (fabs(vectorC[i] - arrLength) > 1e-5){
            printf("Error: vectorC[%ld] = %f\n", i, vectorC[i]);
            break;
        }
    }
    printf("Test %s \n", (i == arrLength) ? "PASSED" : "FAILED");
    
    free(vectorA);
    free(vectorB);
    free(vectorC);

}