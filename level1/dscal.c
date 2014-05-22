#include <ulmblas.h>
#include <emmintrin.h>

void
ULMBLAS(dscal)(const int    n,
               const double alpha,
               double       *x,
               const int    incX)
{
//  Quick return if possible
//
    if (n<=0 || incX<=0 || alpha == 1.0) {
        return;
    }
    if (incX==1) {
        //remaining elements
        int r = n;
        //fix alignment if necessary
        if (!IS_ALIGNED(x,16)) {
            x[0] *= alpha;
            r--;
            x++;
        }
	__m128d x01, a = _mm_load_pd1(&alpha);
	for (; r>1; x+=2, r-=2) {
	    x01 = _mm_load_pd(x);
	    x01 = _mm_mul_pd(a, x01);
	    _mm_store_pd(x,x01);
	}
        //last element done?
	if (r) {
	    (*x) *= alpha;
        }
    } else {
//
//      Code for increment not equal to 1
//
        int i;
        for (i=0; i<n; ++i, x+=incX) {
            (*x) *= alpha;
        }
    }
}

void
F77BLAS(dscal)(const int    *_n,
               const double *_alpha,
               double       *x,
               const int    *_incX)
{
//
//  Dereference scalar parameters
//
    int n        = *_n;
    double alpha = *_alpha;
    int incX     = *_incX;

    ULMBLAS(dscal)(n, alpha, x, incX);
}
