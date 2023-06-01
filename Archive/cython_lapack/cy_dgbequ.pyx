# cython: wraparound = False
# cython: boundscheck = False
# cython: language_level=3

import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_lapack as cython_lapack

np.import_array()
ctypedef np.float64_t REAL_t

def equilibrate(
    int kl, 
    int ku, 
    np.ndarray[REAL_t, ndim=2] mat not None
):
    cdef int ldmat = mat.shape[0]
    cdef int N     = mat.shape[1]
    cdef int i, ib, j, info
    cdef double rowcnd, colcnd, amax

    # Pointers definition
    cdef REAL_t* mat_pointer = <REAL_t*> np.PyArray_DATA(mat)
    if not mat_pointer:
        raise MemoryError()

    # Internal definition for scaling coefficients
    R = np.ones(N)
    C = np.ones(N)
    cdef REAL_t* R_pointer = <REAL_t*> np.PyArray_DATA(R)
    cdef REAL_t* C_pointer = <REAL_t*> np.PyArray_DATA(C)

    # Actual call to DGBEQU (cf. LAPACK documentation)
    cython_lapack.dgbequ(
        &N, &N, &kl, &ku, mat_pointer, &ldmat, R_pointer, 
        C_pointer, &rowcnd, &colcnd, &amax, &info
    )

    # Scaling the matrix
    if info == 0 :
        for j in range(N) :
            for ib in range(ldmat) :
                i = ib + j - ku
                if ((i < 0) or (i > N-1)) : continue
                mat[ib, j] = mat[ib, j] * R[i] * C[j]
    else :
        if info < 0 :
            raise ValueError(f"The {info-1}-th argument had an illegal value")
        else : 
            if info <= N :
                raise ValueError(f"The {info-1}-th row of the matrix is exactly zero")
            else : 
                raise ValueError(f"The ({info-1-N})-th column of the matrix is exactly zero")
    
    return mat, R, C, rowcnd, colcnd, amax
