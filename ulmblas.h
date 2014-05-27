#define F77BLAS(x) x##_


#ifdef FAKE_ATLAS
#   define ULMBLAS(x) ATL_##x
#else
#   define ULMBLAS(x) ULM_##x
#endif

#define IS_ALIGNED(X,N) ( ! ((size_t) X & N) )
