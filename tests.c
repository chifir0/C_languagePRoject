#include "tests.h"
#include "matrix_ops.h"
#include <assert.h>

void test_multiply_matrices() {
    Matrix* A = create_matrix(2, 2);
    A->data[0][0] = 1.0; A->data[0][1] = 2.0;
    A->data[1][0] = 3.0; A->data[1][1] = 4.0;

    Matrix* B = create_matrix(2, 2);
    B->data[0][0] = 2.0; B->data[0][1] = 0.0;
    B->data[1][0] = 1.0; B->data[1][1] = 2.0;

    Matrix* C = multiply_matrices(A, B);

    assert(fabs(C->data[0][0] - 4.0) < TOLERANCE);  // 1*2 + 2*1 = 4
    assert(fabs(C->data[0][1] - 4.0) < TOLERANCE);  // 1*0 + 2*2 = 4
    assert(fabs(C->data[1][0] - 10.0) < TOLERANCE); // 3*2 + 4*1 = 10
    assert(fabs(C->data[1][1] - 8.0) < TOLERANCE);  // 3*0 + 4*2 = 8

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    Matrix* D = create_matrix(2, 2);
    D->data[0][0] = 1.0; D->data[0][1] = 2.0;
    D->data[1][0] = 3.0; D->data[1][1] = 4.0;

    Matrix* E = create_matrix(2, 2);
    E->data[0][0] = 0.0; E->data[0][1] = 0.0;
    E->data[1][0] = 0.0; E->data[1][1] = 0.0;

    Matrix* F = multiply_matrices(D, E);

    assert(fabs(F->data[0][0]) < TOLERANCE); 
    assert(fabs(F->data[0][1]) < TOLERANCE); 
    assert(fabs(F->data[1][0]) < TOLERANCE); 
    assert(fabs(F->data[1][1]) < TOLERANCE);  

    free_matrix(D);
    free_matrix(E);
    free_matrix(F);

    printf("test_multiply_matrices passed.\n");
}

void test_qr_decomposition() {
    Matrix* A = create_matrix(2, 2);
    A->data[0][0] = 1.0; A->data[0][1] = 2.0;
    A->data[1][0] = 3.0; A->data[1][1] = 4.0;

    Matrix* Q = create_matrix(2, 2);
    Matrix* R = create_matrix(2, 2);

    qr_decomposition_householder(A, Q, R);

    Matrix* A_reconstructed = multiply_matrices(Q, R);

    assert(fabs(A_reconstructed->data[0][0] - A->data[0][0]) < TOLERANCE);
    assert(fabs(A_reconstructed->data[0][1] - A->data[0][1]) < TOLERANCE);
    assert(fabs(A_reconstructed->data[1][0] - A->data[1][0]) < TOLERANCE);
    assert(fabs(A_reconstructed->data[1][1] - A->data[1][1]) < TOLERANCE);

    free_matrix(A);
    free_matrix(Q);
    free_matrix(R);
    free_matrix(A_reconstructed);
    printf("test_qr_decomposition passed.\n");
}

void test_qr_algorithm_simple() {
    Matrix* A = create_matrix(2, 2);
    A->data[0][0] = 4.0; A->data[0][1] = 2.0;
    A->data[1][0] = 2.0; A->data[1][1] = 1.0;

    double eigenvals[2];
    Matrix* eigenvecs = create_matrix(2, 2);

    qr_algorithm(A, eigenvals, eigenvecs);

    // Собственные значения 5, 0
    int found_5 = 0, found_0 = 0;
    for (int i = 0; i < 2; i++) {
        if (fabs(eigenvals[i] - 5.0) < 0.1) found_5 = 1;
        if (fabs(eigenvals[i] - 0.0) < 0.1) found_0 = 1;
    }
    assert(found_5 && found_0);

    free_matrix(A);
    free_matrix(eigenvecs);
    printf("test_qr_algorithm_simple passed.\n");
}

void test_qr_algorithm_identity() {
    Matrix* A = create_matrix(2, 2);
    A->data[0][0] = 1.0; A->data[0][1] = 0.0;
    A->data[1][0] = 0.0; A->data[1][1] = 1.0;

    double eigenvals[2];
    Matrix* eigenvecs = create_matrix(2, 2);

    qr_algorithm(A, eigenvals, eigenvecs);

    assert(fabs(eigenvals[0] - 1.0) < TOLERANCE);
    assert(fabs(eigenvals[1] - 1.0) < TOLERANCE);

    free_matrix(A);
    free_matrix(eigenvecs);
    printf("test_qr_algorithm_identity passed.\n");
}

void test_qr_algorithm_diagonal() {
    Matrix* A = create_matrix(3, 3);
    A->data[0][0] = 2.0; A->data[0][1] = 0.0; A->data[0][2] = 0.0;
    A->data[1][0] = 0.0; A->data[1][1] = -1.0; A->data[1][2] = 0.0;
    A->data[2][0] = 0.0; A->data[2][1] = 0.0; A->data[2][2] = 3.0;

    double eigenvals[3];
    Matrix* eigenvecs = create_matrix(3, 3);

    qr_algorithm(A, eigenvals, eigenvecs);

    int found_2 = 0, found_neg1 = 0, found_3 = 0;
    for (int i = 0; i < 3; i++) {
        if (fabs(eigenvals[i] - 2.0) < TOLERANCE) found_2 = 1;
        if (fabs(eigenvals[i] - (-1.0)) < TOLERANCE) found_neg1 = 1;
        if (fabs(eigenvals[i] - 3.0) < TOLERANCE) found_3 = 1;
    }
    assert(found_2 && found_neg1 && found_3);

    free_matrix(A);
    free_matrix(eigenvecs);
    printf("test_qr_algorithm_diagonal passed.\n");
}
