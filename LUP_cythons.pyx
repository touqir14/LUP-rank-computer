cimport numpy as np
import numpy as np
import cython
from libc.stdint cimport uint64_t, int64_t

cdef extern from "complex.h":
    double cabs(double complex x) nogil

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
cpdef (uint64_t, uint64_t) sort_rows_loop(uint64_t[:] scores, double[:,:] u, double threshold, int64_t adder):
    
    """
    Sorts rows of input numpy array u such that for any two rows i,j of the sorted array, if i>j
    then the pivot column index of row i will not be smaller than that of row j.

    Computes the column pivot index of each row within the upper triangular matrix u.

    Parameters
    __________

    u: (2D float64 numpy array)
    u is the upper triangular matrix produced from
    the LUP factorization.

    scores: (unsigned 64-bit 1D numpy integer array)
    After the function executes, scores contains 
    the pivot column index of each row so that the
    callee can use column index information of each 
    row of u.
    
    threshold: (double)
    All entries of u with the absolute magnitude larger 
    than threshold are classified as non-zero entries while
    the rest are classified as zero entries which appear to 
    be non-zeros due to numerical approximation error. 

    adder: (64-bit Integer)
    adder can be used to skip collumns that are surely 
    known to contain zero entries since the search for
    pivot entries start from the collumn index denoted by
    adder.


    Returns
    __________

    num_zero_rows: (unsigned 64-bit Integer)
    num_zero_rows denotes the number of rows with 
    all the entries classified as zeros.

    min_piv_idx: (unsigned 64-bit Integer)
    min_piv_idx denotes the smallest column pivot index 
    that has been found. 
    """

    cdef uint64_t cols = u.shape[1], min_shape = min((u.shape[0], u.shape[1]))
    cdef uint64_t num_zero_rows = 0, min_j = cols+adder, j = 0, min_piv_idx = 0

    for i in range(min_shape):
        j = i
        while (-threshold <= u[i, j] <= threshold):
            if j == cols - 1:
                j += 1
                num_zero_rows += 1
                break
            j += 1

        j += adder
        if min_j > j:
            min_j = j
        scores[i] = j

    min_piv_idx = min_j 
    return num_zero_rows, min_piv_idx

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
cpdef (uint64_t, uint64_t) sort_rows_loop_complex(uint64_t[:] scores, double complex[:,:] u, double threshold, int64_t adder):
    
    """
    Sorts rows of input numpy array u such that for any two rows i,j of the sorted array, if i>j
    then the pivot column index of row i will not be smaller than that of row j.

    Computes the column pivot index of each row within the upper triangular matrix u.

    Parameters
    __________

    u: (2D complex128 numpy array)
    See documentation of the function "sort_rows_loop".


    Returns
    __________

    num_zero_rows: (unsigned 64-bit Integer)
    num_zero_rows denotes the number of rows with 
    all the entries classified as zeros.

    min_piv_idx: (unsigned 64-bit Integer)
    min_piv_idx denotes the smallest column pivot index 
    that has been found. 
    """

    cdef uint64_t cols = u.shape[1], min_shape = min((u.shape[0], u.shape[1]))
    cdef uint64_t num_zero_rows = 0, min_j = cols+adder, j = 0, min_piv_idx = 0

    for i in range(min_shape):
        j = i
        while (cabs(u[i, j]) <= threshold):
            if j == cols - 1:
                j += 1
                num_zero_rows += 1
                break
            j += 1

        j += adder
        if min_j > j:
            min_j = j
        scores[i] = j

    min_piv_idx = min_j 
    return num_zero_rows, min_piv_idx


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
cpdef void merge_sorted(uint64_t[:] sorted_scores_1, uint64_t[:] sorted_scores_2, uint64_t[:] new_sorted_scores, uint64_t[:] order_1, uint64_t[:] order_2, uint64_t[:] new_order):

    """
    Merges two sorted arrays to form a sorted array in linear time.

    Parameters
    __________

    ALL params: (unsigned 64-bit 1D numpy integer array)
    
    sorted_scores_1 AND order_1: 
    sorted_scores_1[i] contains the column pivot index of
    the row indicated by order_1[i]. sorted_scores_1 contains
    the column pivot indices in a sorted order. This is the 
    first pair that encodes the 1st sorted array.

    sorted_scores_2 AND order_2:
    This is the 2nd pair that encodes the 2nd sorted array.
    The definitions given for the 1st pair holds for this pair
    too.

    new_sorted_scores AND new_order:
    This is the 3rd pair that encodes the new merged sorted array.
    The definitions given for the 1st pair holds for this pair
    too.


    Returns
    __________

    None
    """

    cdef uint64_t new_size = new_order.shape[0], size_1 = order_1.shape[0], size_2 = order_2.shape[0], p1 = 0, p2 = 0
    cdef uint64_t to_compare = 1
    
    for i in range(new_size):
        if to_compare == 1:
            if sorted_scores_1[p1] > sorted_scores_2[p2]:
                new_sorted_scores[i] = sorted_scores_2[p2]
                new_order[i] = order_2[p2]
                p2 += 1
                if p2 == size_2: to_compare = 2  
            else:
                new_sorted_scores[i] = sorted_scores_1[p1]
                new_order[i] = order_1[p1]
                p1 += 1
                if p1 == size_1: to_compare = 3  

        elif to_compare == 2:
            new_sorted_scores[i] = sorted_scores_1[p1]
            new_order[i] = order_1[p1]
            p1 += 1

        else:
            new_sorted_scores[i] = sorted_scores_2[p2]
            new_order[i] = order_2[p2]
            p2 += 1

    return



