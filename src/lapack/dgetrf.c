#include <ulmblas.h>
#include <float.h>
#include <math.h>
#include <level1/dscal.h>
#include <level1/dswap.h>
#include <level1/idamax.h>
#include <level2/dger.h>

#include <stdio.h>

static double
safeMin()
{
    double eps = DBL_EPSILON * 0.5;
    double sMin  = DBL_MIN;
    double small = 1.0 / DBL_MAX;

    if (small>=sMin) {
//
//      Use SMALL plus a bit, to avoid the possibility of rounding
//      causing overflow when computing  1/sfmin.
//
        sMin = small*(1.0+eps);
    }
    return sMin;
}

int
dgetf2(int     m,
       int     n,
       double  *A,
       int     incRowA,
       int     incColA,
       int     *piv)
{
    int i, j, jp, info;

    double sMin = safeMin();

    info = 0;

    if (m==0 || n==0) {
        return info;
    }

    for (j=0; j<min(m,n); ++j) {
        jp = j+idamax(m-j, &A[j*incRowA+j*incColA], incRowA);
        piv[j] = jp;

        if (j != jp) {
            dswap(n, A+j*incRowA, incRowA, A+jp*incRowA, incRowA);
        }
        double ajj = A[j*incRowA+j*incColA];
        if (fabs(ajj) > sMin) {
            dscal(n-j, 1/ajj, A+(j+1)*incColA, incColA);
        } else {
            for (double *a21 = A+(j+1)*incColA; a21 < A+m*incColA ; a21 += incColA) {
                a21 /= ajj;
            } 
        }
        dgemm_nn(m-j, n-j, 1, -1, 
                 A+(j+1)*incColA, incRowA, incColA, 
                 A+(j+1)*incRowA, incRowA, incColA,
                 A+(j+1)*incRowA+j*incColA, incRowA, incColA);
    }
    return info;
}

int
ULMBLAS(dgetrf)(enum Order  order,
                int         m,
                int         n,
                double      *A,
                int         ldA,
                int         *piv)
{
    if (order==ColMajor) {
        return dgetf2(m, n, A, 1, ldA, piv);
    } else {
        return dgetf2(m, n, A, ldA, 1, piv);
    }
}
