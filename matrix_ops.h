#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ITERATIONS 10000
#define TOLERANCE 1e-10

typedef struct {
    int rows;
    int cols;
    double** data;
} Matrix;

Matrix* create_matrix(int rows, int cols);
void free_matrix(Matrix* m);
void print_matrix(Matrix* m);
Matrix* copy_matrix(Matrix* src);
Matrix* multiply_matrices(Matrix* A, Matrix* B);
void qr_decomposition_householder(Matrix* A, Matrix* Q_out, Matrix* R_out);
void qr_algorithm(Matrix* A, double eigenvals[], Matrix* eigenvecs);
void numerical_solve(Matrix* A, double* y, int n, double t_start, double t_end, double dt);

#endif
