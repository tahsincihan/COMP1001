/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP1001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/


#include <stdio.h>
#include <time.h>
#include <pmmintrin.h>
#include <process.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <omp.h>

#define M 1024*512
#define ARITHMETIC_OPERATIONS1 3*M
#define TIMES1 1

#define N 8192
#define ARITHMETIC_OPERATIONS2 4*N*N
#define TIMES2 1

#define EPSILON 1e-6


//function declaration
void initialize();
void routine1(float alpha, float beta);
void routine1_vec(float alpha, float beta);

void routine2(float alpha, float beta);
void routine2_vec(float alpha, float beta);

unsigned short int Compare_Routine1();
unsigned short int Compare_Routine2();

__declspec(align(64)) float  y[M], z[M] ;
__declspec(align(64)) float A[N][N], x[N], w[N];

float y_reference[M];
float z_reference[M];

float w_reference[N];
float A_reference[N][N];
float x_reference[N];

void copyArraysForReference() {
    memcpy(y_reference, y, sizeof(float) * M);
    memcpy(z_reference, z, sizeof(float) * M);
    memcpy(w_reference, w, sizeof(float) * N);
    memcpy(A_reference, A, sizeof(float) * N * N);
    memcpy(x_reference, x, sizeof(float) * N);
}


int main() {

    float alpha = 0.023f, beta = 0.045f;
    double run_time, start_time;
    unsigned int t;

    initialize();
    copyArraysForReference();


    printf("\nRoutine1:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES1; t++)
        routine1(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));

    if(Compare_Routine1() == 0)
        printf("\n\n\r ----- output of routine 1 vec is correct -----\n\r");
    else
        printf("\n\n\r ----- output of routine 1 vac is incorrect -----\n\r");

    printf("\nRoutine2:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES2; t++)
        routine2(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS2) / ((double)run_time / TIMES2));

    if (Compare_Routine2() == 0)
        printf("\n\n\r ----- output of routine 1 vec is correct -----\n\r");
    else
        printf("\n\n\r ----- output of routine 1 vac is incorrect -----\n\r");


    return 0;
}

void initialize() {

    unsigned int i, j;

    //initialize routine2 arrays
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = (i % 99) + (j % 14) + 0.013f;
        }

    //initialize routine1 arrays
    for (i = 0; i < N; i++) {
        x[i] = (i % 19) - 0.01f;
        w[i] = (i % 5) - 0.002f;
    }

    //initialize routine1 arrays
    for (i = 0; i < M; i++) {
        z[i] = (i % 9) - 0.08f;
        y[i] = (i % 19) + 0.07f;
    }


}




void routine1(float alpha, float beta) {

    unsigned int i;


    for (i = 0; i < M; i++)
        y[i] = alpha * y[i] + beta * z[i];

}

void routine1_vec(float alpha, float beta) {

    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);

    for (unsigned int i = 0; i < M; i += 8) {

        //load 8 elements of y and z
        __m256 y_vec = _mm256_loadu_ps(&y[i]);
        __m256 z_vec = _mm256_loadu_ps(&z[i]);

        //perform vectorization
        __m256 alpha_times_y = _mm256_mul_ps(alpha_vec, y_vec);
        __m256 beta_times_z = _mm256_mul_ps(beta_vec, z_vec);
        y_vec = _mm256_add_ps(alpha_times_y, beta_times_z);

        _mm256_storeu_ps(&y[i], y_vec);
    }
}

void routine2(float alpha, float beta) {

    unsigned int i, j;


    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            w[i] = w[i] - beta + alpha * A[i][j] * x[j];


}

void routine2_vec(float alpha, float beta) {

    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);

    for (unsigned int i = 0; i < N; i++) {

        __m256 w_vec = _mm256_loadu_ps(&w[i]);

        for (unsigned int j = 0; j < N; j += 8) {

            __m256 A_vec = _mm256_loadu_ps(&A[i][j]);
            __m256 x_vec = _mm256_loadu_ps(&x[j]);

            __m256 w_sub_beta = _mm256_sub_ps(w_vec, beta_vec);
            A_vec = _mm256_mul_ps(A_vec, x_vec);
            alpha_vec = _mm256_mul_ps(alpha_vec, A_vec);
            w_vec = _mm256_add_ps(w_sub_beta, alpha_vec);
        }
        _mm256_storeu_ps(&w[i], w_vec);
    }

}

unsigned short int Compare_Routine1() {
    for (unsigned int i = 0; i < M; i++) {
        if (equal(y[i], y_reference[i]) == 1) {
            printf("\n Error in Routine1_vec! Mismatch at index %u \n", i);
            return 1;
        }
    }
    return 0;
}

unsigned short int Compare_Routine2() {
    for (unsigned int i = 0; i < N; i++) {
        if (equal(w[i], w_reference[i]) == 1) {
            printf("\n Error in Routine2_vec! Mismatch at index %u \n", i);
            return 1;
        }
    }
    return 0;
}


unsigned short int equal(float a, float b) {
    float temp = a - b;
    if ((fabs(temp) / fabs(b) < EPSILON))
        return 0;
    else
        return 1;
    
}
