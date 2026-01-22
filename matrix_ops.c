#include "matrix_ops.h"

Matrix* create_matrix(int rows, int cols) {
    Matrix* m = malloc(sizeof(Matrix));
    if (!m) return NULL;

    m->rows = rows;
    m->cols = cols;
    m->data = malloc(rows * sizeof(double*));
    if (!m->data) {
        free(m);
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        m->data[i] = calloc(cols, sizeof(double));
        if (!m->data[i]) {
            for (int j = 0; j < i; j++) {
                free(m->data[j]);
            }
            free(m->data);
            free(m);
            return NULL;
        }
    }
    return m;
}

void free_matrix(Matrix* m) {
    if (m) {
        for (int i = 0; i < m->rows; i++) {
            free(m->data[i]);
        }
        free(m->data);
        free(m);
    }
}

void print_matrix(Matrix* m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%10.5f ", m->data[i][j]);
        }
        printf("\n");
    }
}

Matrix* copy_matrix(Matrix* src) {
    Matrix* dst = create_matrix(src->rows, src->cols);
    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            dst->data[i][j] = src->data[i][j];
        }
    }
    return dst;
}

Matrix* multiply_matrices(Matrix* A, Matrix* B) {
    if (A->cols != B->rows) return NULL;

    Matrix* C = create_matrix(A->rows, B->cols);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            C->data[i][j] = 0.0;
            for (int k = 0; k < A->cols; k++) {
                C->data[i][j] += A->data[i][k] * B->data[k][j];
            }
        }
    }
    return C;
}

void qr_decomposition_householder(Matrix* A, Matrix* Q_out, Matrix* R_out) {
    int n = A->rows;
    Matrix* Q_total = create_matrix(n, n);
    Matrix* A_current = copy_matrix(A);

    for (int i = 0; i < n; i++) Q_total->data[i][i] = 1.0;

    for (int k = 0; k < n - 1; k++) {
        double norm_x = 0.0;
        for (int i = k; i < n; i++) {
            norm_x += A_current->data[i][k] * A_current->data[i][k];
        }
        norm_x = sqrt(norm_x);

        if (norm_x == 0.0) continue;

        double alpha = (A_current->data[k][k] >= 0) ? -norm_x : norm_x;

        double* v = calloc(n, sizeof(double));
        v[k] = A_current->data[k][k] - alpha;
        for (int i = k + 1; i < n; i++) {
            v[i] = A_current->data[i][k];
        }

        double norm_v = 0.0;
        for (int i = k; i < n; i++) {
            norm_v += v[i] * v[i];
        }
        norm_v = sqrt(norm_v);
        if (norm_v != 0.0) {
            for (int i = k; i < n; i++) {
                v[i] /= norm_v;
            }
        }

        Matrix* H = create_matrix(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                H->data[i][j] = (i == j) ? 1.0 : 0.0;
                H->data[i][j] -= 2.0 * v[i] * v[j];
            }
        }

        Matrix* A_new = multiply_matrices(H, A_current);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A_current->data[i][j] = A_new->data[i][j];
            }
        }

        Matrix* Q_updated = multiply_matrices(Q_total, H);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Q_total->data[i][j] = Q_updated->data[i][j];
            }
        }

        free_matrix(H);
        free_matrix(A_new);
        free_matrix(Q_updated);
        free(v);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j < i) R_out->data[i][j] = 0.0;
            else R_out->data[i][j] = A_current->data[i][j];
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Q_out->data[i][j] = Q_total->data[i][j];
        }
    }

    free_matrix(A_current);
    free_matrix(Q_total);
}

void qr_algorithm(Matrix* A, double eigenvals[], Matrix* eigenvecs) {
    int n = A->rows;
    Matrix* Ak = copy_matrix(A);
    Matrix* Q_total = create_matrix(n, n);
    for (int i = 0; i < n; i++) Q_total->data[i][i] = 1.0;

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        Matrix* Qk = create_matrix(n, n);
        Matrix* Rk = create_matrix(n, n);

        qr_decomposition_householder(Ak, Qk, Rk);

        Matrix* Ak_new = multiply_matrices(Rk, Qk);

        Matrix* Q_mult = multiply_matrices(Q_total, Qk);

        double diff = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                diff += fabs(Ak_new->data[i][j] - Ak->data[i][j]);
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Ak->data[i][j] = Ak_new->data[i][j];
                Q_total->data[i][j] = Q_mult->data[i][j];
            }
        }

        if (diff < TOLERANCE) {
            break;
        }

        free_matrix(Qk);
        free_matrix(Rk);
        free_matrix(Ak_new);
        free_matrix(Q_mult);
    }

    for (int i = 0; i < n; i++) {
        eigenvals[i] = Ak->data[i][i];
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            eigenvecs->data[i][j] = Q_total->data[i][j];
        }
    }

    free_matrix(Ak);
    free_matrix(Q_total);
}

void numerical_solve(Matrix* A, double* y, int n, double t_start, double t_end, double dt) {
    int steps = (int)((t_end - t_start) / dt);
    double* k = malloc(n * sizeof(double));
    if (!k) {
        fprintf(stderr, "Memory error\n");
        return;
    }

    printf(" t\t");
    for (int i = 0; i < n; i++) {
        printf("x[%d]\t\t", i);
    }
    printf("\n");

    for (int step = 0; step <= steps; step++) {
        printf("%.3f\t", t_start);
        for (int i = 0; i < n; i++) {
            printf("%.5f\t", y[i]);
        }
        printf("\n");

        // k = A * y
        for (int i = 0; i < n; i++) {
            k[i] = 0.0;
            for (int j = 0; j < n; j++) {
                k[i] += A->data[i][j] * y[j];
            }
        }

        // y = y + dt * k
        for (int i = 0; i < n; i++) {
            y[i] += dt * k[i];
        }

        t_start += dt;
    }

    free(k);
}
