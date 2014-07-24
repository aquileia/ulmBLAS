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
    int j, jp, info;

    double *a21, sMin = safeMin();

    info = 0;

    if (m==0 || n==0) {
        return info;
    }

    for (j=0; j<min(m,n); ++j) {
        jp = j+idamax(m-j, &A[j*incRowA+j*incColA], incRowA);
        piv[j] = jp;
        if (A[jp*incRowA+j*incColA] == 0.0){
            info = info?info:j+1;
            continue;
        }
        if (j != jp) {
            dswap(n, A+j*incRowA, incColA, A+jp*incRowA, incColA);
        }
        double ajj = A[j*incRowA+j*incColA];
        if (fabs(ajj) > sMin) {
            dscal(m-j-1, 1/ajj, A+(j+1)*incRowA+j*incColA, incRowA);
        } else {
            for (a21 = A+(j+1)*incRowA+j*incColA; a21 < A+m*incRowA ; a21 += incRowA) {
                *a21 /= ajj;
            } 
        }
        if (j<min(m,n)-1){
            dger(m-j-1, n-j-1, -1.0, 
                     A+(j+1)*incRowA+  j  *incColA, incRowA,
                     A+  j  *incRowA+(j+1)*incColA, incColA, 
                     A+(j+1)*incRowA+(j+1)*incColA, incRowA, incColA);
        }
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
