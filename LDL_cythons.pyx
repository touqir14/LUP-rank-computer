cimport numpy as np
import numpy as np
import cython
from libc.stdint cimport uint64_t, int64_t

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
cpdef (uint64_t, uint64_t) sort_rows_loop(uint64_t[:] scores, double[:,:] u, double threshold, int64_t adder):
    cdef uint64_t cols = u.shape[1], min_shape = min((u.shape[0], u.shape[1]))
    cdef uint64_t num_zero_rows = 0, min_j = cols+adder, j = 0

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

    return num_zero_rows, min_j


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
cpdef void merge_sorted(uint64_t[:] sorted_scores_1, uint64_t[:] sorted_scores_2, uint64_t[:] new_sorted_scores, uint64_t[:] order_1, uint64_t[:] order_2, uint64_t[:] new_order):

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



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
cpdef void print_vector(uint64_t[:] A):
    print("Printing..")

    for i in range(A.shape[0]):
        print(A[i])        

    return


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
cpdef void copy_array(double[:] source, double[:] destination):
    
    cdef int size = source.shape[0]

    for i in range(size):

        destination[i] = source[i]

    return
