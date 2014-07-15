#include <ulmblas.h>
#include <math.h>
#include <level1/dgescal.h>

#define MC 16

//
//  Solve A*X = alpha*B
//
//  where A is a upper triangular mxm matrix with unit or non-unit diagonal
//  and B is a general mxn matrix.
//

void
dtrsm_u(enum Diag       diag,
        int             m,
        int             n,
        double          alpha,
        const double    *A,
        int             incRowA,
        int             incColA,
        double          *B,
        int             incRowB,
        int             incColB)
{
    //all but the last diagonal block
    int I = (m-1)/MC;
    for (int i=0; i<I; ++i) {
        dtrsb_u(diag, MC, n, alpha, A, incRowA, incColA, B, incRowB, incColB);
        dgemm_nn();
        A += MC*(incRowA+incColA);
        B += MC*incRowB;
    }
    I = m % MC;
    dtrsb_u(diag, I? I:MC, n, alpha, A, incRowA, incColA, B, incRowB, incColB);
}

void
dtrsb_u(enum Diag       diag,
        int             m,
        int             n,
        double          alpha,
        const double    *A,
        int             incRowA,
        int             incColA,
        double          *B,
        int             incRowB,
        int             incColB)
{
    // Your implementation goes here
    int i,j,k;

    //nothing to be done
    if (n==0 || m==0) return;

    dgescal(m,n,alpha,B,incRowB, incColB);

    //for alpha == 0, we return X = 0
    if (alpha==0.0) return;

    for (j=0; j<n; ++j) {
        for (k=m-1; k>=0; --k) {
            //empty entry: x_{jk} = 0
            if (B[k*incRowB + j*incColB] == 0.0) continue;
            if (diag!=Unit) {
                B[k*incRowB + j*incColB] /= A[k*incRowA +k*incColA];
            }
            for (i=0; i<k; ++i) {
                B[i*incRowB + j*incColB] -= B[k*incRowB + j*incColB]
                                          * A[i*incRowA +k*incColA];

            }
        }
    }
}
