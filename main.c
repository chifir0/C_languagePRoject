#include "matrix_ops.h"
#include "tests.h"

void run_all_tests() {
    test_multiply_matrices();
    test_qr_decomposition();
    test_qr_algorithm_simple();
    test_qr_algorithm_identity();
    test_qr_algorithm_diagonal();
    printf("\n");
}

int main() {

    run_all_tests();

    printf("Solving a system of differential equations using the QR algorithm\n");
    printf("dx/dt = Ax\n");

    int n;
    printf("Enter n - matrix A size (n x n): ");
    scanf_s("%d", &n);

    if (n <= 0) {
        fprintf(stderr, "Size must be > 0\n");
        return 1;
    }

    Matrix* A = create_matrix(n, n);
    if (!A) {
        fprintf(stderr, "Memory allocation failed for matrix A.\n");
        return 1;
    }

    printf("Enter matrix elements %dx%d:\n", n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("A[%d][%d]: ", i, j);
            scanf_s("%lf", &A->data[i][j]);
        }
    }

    printf("\nEntered matrix A:\n");
    print_matrix(A);

    printf("\n--- ANALYTICAL SOLUTION (QR Algorithm) ---\n");

    double* eigenvals = malloc(n * sizeof(double));
    if (!eigenvals) {
        fprintf(stderr, "Memory error for eigenvals\n");
        free_matrix(A);
        return 1;
    }

    Matrix* eigenvecs = create_matrix(n, n);
    if (!eigenvecs) {
        fprintf(stderr, "Memory error for eigenvecs\n");
        free(eigenvals);
        free_matrix(A);
        return 1;
    }

    qr_algorithm(A, eigenvals, eigenvecs);

    printf("Eigenvalues:\n");
    for (int i = 0; i < n; i++) {
        printf("lambda[%d] = %.5f\n", i, eigenvals[i]);
    }

    printf("\nEigenvectors (columns):\n");
    for (int j = 0; j < n; j++) {
        printf("v[%d] = [", j);
        for (int i = 0; i < n; i++) {
            printf("%.5f", eigenvecs->data[i][j]);
            if (i < n - 1) printf(", ");
        }
        printf("]^T\n");
    }

    printf("\n--- FORM OF THE GENERAL SOLUTION WITH SUBSTITUTED VALUES ---\n");
    printf("x(t) = c1 * [");
    for (int i = 0; i < n; i++) {
        printf("%.5f", eigenvecs->data[i][0]);
        if (i < n - 1) printf(", ");
    }
    printf("]^T * exp(%.2f*t)", eigenvals[0]);

    for (int j = 1; j < n; j++) {
        printf(" + c%d * [", j + 1);
        for (int i = 0; i < n; i++) {
            printf("%.2f", eigenvecs->data[i][j]);
            if (i < n - 1) printf(", ");
        }
        printf("]^T * exp(%.2f*t)", eigenvals[j]);
    }
    printf("\n");

    free(eigenvals);
    free_matrix(eigenvecs);

    printf("\n*** NUMERICAL SOLUTION (Euler's method) ***\n");

    double* y = malloc(n * sizeof(double));
    if (!y) {
        fprintf(stderr, "Memory error for y\n");
        free_matrix(A);
        return 1;
    }

    printf("Enter initial conditions (x1(0), x2(0), ..., xn(0)):\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d](0): ", i);
        scanf_s("%lf", &y[i]);
    }

    double t_start = 0.0, t_end = 2.0, dt = 0.1;
    printf("\nNumerical solution on [%.2f, %.2f] with step of %.3f:\n", t_start, t_end, dt);
    numerical_solve(A, y, n, t_start, t_end, dt);

    free(y);
    free_matrix(A);

    return 0;
}
