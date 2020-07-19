import numpy as np
from scipy.linalg import lu
from LUP_cythons import *
from numpy.core import finfo


def sort_rows_fast(u, threshold, remove_zero_columns_rows=True, adder=0):
    """
    Sorts rows of input matrix u such that for any two rows i,j of the sorted matrix, if i>j
    then the pivot column index of row i will not be smaller than that of row j.


    Parameters
    __________

    u: (2D float64 numpy array)
    u is the upper triangular matrix produced from
    the LUP factorization.

    remove_zero_columns_rows: (Boolean)
    If True, this flag instructs the function to
    return a sorted matrix such that it contains
    no column and row with all entries classified 
    as zeros. 

    threshold: (Float)
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

    u_sorted: (2D float64 numpy array)
    u_sorted is u with rows shuffled such that for any i>j, 
    the pivot column index of u_sorted[i,:] will not be smaller
    than that of u_sorted[j,:].

    sorted_scores AND order: (unsigned 64-bit 1D numpy Integer array)
    sorted_scores[i] contains the column pivot index of
    the row indicated by order[i]. sorted_scores contains
    the column pivot indices in a sorted order.

    """

    min_shape = min(u.shape)
    scores = np.empty(min_shape, dtype=np.uint64)

    if remove_zero_columns_rows:
        num_zero_rows, min_j = sort_rows_loop(scores, u, threshold, 0)
        order = np.argsort(scores).astype(np.uint64)
        u_sorted = u[order]
        sorted_scores = scores[order]

        if num_zero_rows == 0:
            return u_sorted[:, min_j:], sorted_scores-min_j, order
        else:
            return u_sorted[:-num_zero_rows, min_j:], sorted_scores[:-num_zero_rows]-min_j, order[:-num_zero_rows]
    else:
        num_zero_rows, min_j = sort_rows_loop(scores, u, threshold, adder)
        order = np.argsort(scores).astype(np.uint64)

        u_sorted = np.copy(u)
        u_sorted[np.arange(min_shape)] = u[order]
        sorted_scores = np.ones(u.shape[0], dtype=np.uint64)*min_shape
        sorted_scores[np.arange(min_shape)] = scores[order]
        return u_sorted, sorted_scores, order


def merge_sorted_arrays(sorted_scores_1, sorted_scores_2, order_1, order_2):
    """
    Merges two sorted arrays to form a sorted array in linear time.


    Parameters
    __________

    ALL: (unsigned 64-bit 1D numpy Integer array)

    sorted_scores_1 AND order_1: 
    sorted_scores_1[i] contains the column pivot index of
    the row indicated by order_1[i]. sorted_scores_1 contains
    the column pivot indices in a sorted order. This is the 
    first pair that encodes the 1st sorted array.

    sorted_scores_2 AND order_2:
    This is the 2nd pair that encodes the 2nd sorted array.
    The definitions given for the 1st pair holds for this pair
    too.

    Returns
    __________

    new_sorted_scores AND new_order:
    This is the 3rd pair that encodes the new merged sorted array.
    The definitions given for the 1st pair holds for this pair
    too.

    """

    if len(order_1) == 0:
        return sorted_scores_2, order_2
    if len(order_2) == 0:
        return sorted_scores_1, order_1

    new_size = len(order_1) + len(order_2)
    new_sorted_scores = np.empty(new_size, dtype=np.uint64)
    new_order = np.empty(new_size, dtype=np.uint64)

    merge_sorted(sorted_scores_1, sorted_scores_2,
                 new_sorted_scores, order_1, order_2, new_order)
    return new_sorted_scores, new_order


def compute_ref_LUP_fast(u, threshold):
    """
    Function for computing Row Echelon Form of u which is produced from LUP factorization.

    Parameters
    __________

    threshold: (Float)
    See documentation of the function "sort_rows_fast"


    Returns
    __________

    u[order]: (2D float64 numpy array)
    Modified u array that is in Row Echelon Form.

    """

    _, sorted_scores, order = sort_rows_fast(
        u, threshold, remove_zero_columns_rows=False)
    rows, cols = u.shape
    halt = False
    i, j = 0, 0
    in_ref = True
    not_ref_pos = [-1, -1]
    last_row = False

    while not halt:
        while (-threshold <= u[order[i], j] <= threshold) or last_row:
            in_ref = True
            if not_ref_pos[0] != -1:
                start, end = not_ref_pos
                _, u_prime = lu(u[order[start:end+1], j:], permute_l=True)
                # u_prime = u_prime / np.abs(u_prime).max() #
                sorted_scores_t = np.ones(end-start+1, dtype=np.uint64)*cols
                order_t = np.arange(end-start+1, dtype=np.uint64)
                u_sorted_t, sorted_scores_t[:u_prime.shape[0]], order_t[:u_prime.shape[0]] = sort_rows_fast(
                    u_prime, threshold, remove_zero_columns_rows=False, adder=j)
                idxs_1, idxs_2 = order[start:start+u_prime.shape[0]
                                       ], order[start+u_prime.shape[0]:end+1]
                u[idxs_1, j:] = u_sorted_t
                u[idxs_2, j:] = 0
                sorted_scores[start:], order[start:] = merge_sorted_arrays(
                    sorted_scores_t, sorted_scores[end+1:], np.append(idxs_1, idxs_2)[order_t], order[end+1:])
                i = start
                not_ref_pos[0], not_ref_pos[1] = -1, -1
                last_row = False
                continue

            if j == cols - 1 or last_row:
                halt = True
                break

            j += 1

        if not in_ref:
            if not_ref_pos[0] == -1:
                not_ref_pos[0] = i - 1
            not_ref_pos[1] = i

        i += 1
        in_ref = False

        if i == rows:
            last_row = True
            i -= 1

    return u[order]


def compute_rank_ref(ref, threshold):
    """
    Function for computing the rank from the Row Echelon Form of u.

    Parameters
    __________

    ref: (2D float64 numpy array)
    The input Row Echelon form of u.

    threshold: (Float)
    See documentation of the function "sort_rows_fast".


    Returns
    __________

    rank: (Integer)
    rank of u computed from its Row Echelon Form.

    """

    rows, cols = ref.shape
    j = 0
    rank = 0
    halt = False

    for i in range(rows):
        while (-threshold <= ref[i, j] <= threshold):
            if j == cols - 1:
                halt = True
                break
            j += 1

        if halt:
            break

        rank += 1

    return rank


def rank_revealing_LUP(A, threshold=10**-10):
    """
    It returns the rank of A using solely CPU

    Parameters
    __________

    A: (2D float64 Numpy Array)
    Input array for which rank needs to be computed

    threshold: (Float)
    See documentation of the function "sort_rows_fast"


    Returns:
    _________

    rank: (Integer)
    rank of A

    """

    if A.shape[0] < A.shape[1]:
        A = A.T

    A = A / np.abs(A).max()
    _, u = lu(A, permute_l=True)
    threshold = np.abs(u).max() * np.finfo(np.float64).eps * np.max(u.shape) * 300
    ref = compute_ref_LUP_fast(u, threshold)
    rank = compute_rank_ref(ref, threshold)

    return rank



def rank_revealing_LUP_GPU(A, threshold=10**-10):
    """
    It returns the rank of A using GPU acceleration. This is based on PyTorch's LUP implementation.
    
    Parameters
    __________

    A: (2D float64 Numpy Array)
    Input array for which rank needs to be computed

    threshold: (Float)
    See documentation of the function "sort_rows_fast"


    Returns:
    _________

    rank: (Integer)
    rank of A

    """

    try:
        import torch
    except ImportError as e:
        print(str(e))
        print("rank_revealing_LUP_GPU requires PyTorch to run.")

    A = A / np.abs(A).max()
    A_tensor = torch.from_numpy(A).cuda()    
    A_LU, pivots = torch.lu(A_tensor, get_infos=False)
    _, _, u = torch.lu_unpack(A_LU, pivots, unpack_pivots=False)
    u = u.cpu().numpy()
    ref = compute_ref_LUP_fast(u, threshold)
    rank = compute_rank_ref(ref, threshold)

    return rank
