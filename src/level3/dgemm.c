#include <ulmblas.h>
#include <auxiliary/xerbla.h>
#include <level3/dgemm_nn.h>

void
ULMBLAS(dgemm)(enum Trans     transA,
               enum Trans     transB,
               int            m,
               int            n,
               int            k,
               double         alpha,
               const double   *A,
               int            ldA,
               const double   *B,
               int            ldB,
               double         beta,
               double         *C,
               int            ldC)
{
//
//  Local scalars
//
    int     i, j;

//
//  Quick return if possible.
//
    if (m==0 || n==0 || ((alpha==0.0 || k==0) && (beta==1.0))) {
        return;
    }

//
//  And if alpha is exactly zero
//
    if (alpha==0.0) {
        if (beta==0.0) {
            for (j=0; j<n; ++j) {
                for (i=0; i<m; ++i) {
                    C[i+j*ldC] = 0.0;
                }
            }
        } else {
            for (j=0; j<n; ++j) {
                for (i=0; i<m; ++i) {
                    C[i+j*ldC] *= beta;
                }
            }
        }
        return;
    }

//
//  Start the operations.
//
    if (transB==NoTrans || transB==Conj) {
        if (transA==NoTrans || transA==Conj) {
//
//          Form  C := alpha*A*B + beta*C.
//
            dgemm_nn(m, n, k,
                     alpha,
                     A, 1, ldA,
                     B, 1, ldB,
                     beta,
                     C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B + beta*C
//
            dgemm_nn(m, n, k,
                     alpha,
                     A, ldA, 1,
                     B, 1, ldB,
                     beta,
                     C, 1, ldC);
        }
    } else {
        if (transA==NoTrans || transA==Conj) {
//
//          Form  C := alpha*A*B**T + beta*C
//
            dgemm_nn(m, n, k,
                     alpha,
                     A, 1, ldA,
                     B, ldB, 1,
                     beta,
                     C, 1, ldC);
        } else {
//
//          Form  C := alpha*A**T*B**T + beta*C
//
            dgemm_nn(m, n, k,
                     alpha,
                     A, ldA, 1,
                     B, ldB, 1,
                     beta,
                     C, 1, ldC);
        }
    }
}

void
F77BLAS(dgemm)(const char     *_transA,
               const char     *_transB,
               const int      *_m,
               const int      *_n,
               const int      *_k,
               const double   *_alpha,
               const double   *A,
               const int      *_ldA,
               const double   *B,
               const int      *_ldB,
               const double   *_beta,
               double         *C,
               const int      *_ldC)
{
//
//  Dereference scalar parameters
//
    enum Trans transA = charToTranspose(*_transA);
    enum Trans transB = charToTranspose(*_transB);
    int m             = *_m;
    int n             = *_n;
    int k             = *_k;
    double alpha      = *_alpha;
    int ldA           = *_ldA;
    int ldB           = *_ldB;
    double beta       = *_beta;
    int ldC           = *_ldC;

//
//  Set  numRowsA and numRowsB as the number of rows of A and B
//
    int numRowsA = (transA==NoTrans || transA==Conj) ? m : k;
    int numRowsB = (transB==NoTrans || transB==Conj) ? k : n;

//
//  Test the input parameters
//
    int info = 0;
    if (transA==0) {
        info = 1;
    } else if (transB==0) {
        info = 2;
    } else if (m<0) {
        info = 3;
    } else if (n<0) {
        info = 4;
    } else if (k<0) {
        info = 5;
    } else if (ldA<max(1,numRowsA)) {
        info = 8;
    } else if (ldB<max(1,numRowsB)) {
        info = 10;
    } else if (ldC<max(1,m)) {
        info = 13;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DGEMM ", &info);
    }

    ULMBLAS(dgemm)(transA, transB,
                   m, n, k,
                   alpha,
                   A, ldA,
                   B, ldB,
                   beta,
                   C, ldC);
}
