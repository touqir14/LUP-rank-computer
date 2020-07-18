import numpy as np
from numpy.linalg import matrix_rank
import LUP_rank
import other_rank_computers
import statistics as stat
from time import perf_counter
import argparse


RANK_COMPUTER = None

def run_all_tests(params, threshold, diagnostics, runs):
    """
    It runs all testing functions given within params along with the parameters to pass.

    Parameters
    ___________

    params: (Dict)
    params[k] = [param_0, .... param_i, function_k]
    param_i is the i'th param to pass to function_k and k is a label used.

    threshold: (Float)
    All entries of u (produced from LUP factorization) with the absolute 
    magnitude larger than threshold are classified as non-zero entries while
    the rest are classified as zero entries which appear to be non-zeros due
    to numerical approximation error.

    diagnostics: (Boolean)
    If True and in the event of a failure, it instructs the functions to print
    out details such as ranks computed from SVD and LUP.

    runs: (Integer)
    Number of times to repeat an experimental configuration.

    """

    for k, v in params.items():
        fn = v[-1]
        args = list(zip(*[[*v[:-1], threshold, diagnostics]]*runs))
        result_values = list(map(fn, *args))
        params[k] += compute_mean_values(result_values)
        params[k].append("For {0}: success_rate:{1}, lup_time_avg:{2}, svd_time_avg:{3}, speed_gain:{4}".format(
            k, params[k][-3]*100, params[k][-2], params[k][-1], params[k][-1]/params[k][-2]))

    for k in params:
        print(params[k][-1])

    return


def tests_random_cond_rank(row, col, cond, rank=None, threshold=10**-10, diagnostics=False):
    """
    This function randomly generates a matrix with cond (See _build_matrix_rank_k documentation) and rank.

    Parameters
    __________

    See _build_matrix_rank_k and run_all_tests documentation.


    Returns
    __________

    success: (Boolean)
    If LUP rank and SVD rank matches then this will True, else False

    lup_time: (Float)
    Time to Compute rank using the LUP method

    svd_time: (Float)
    Time to Compute rank using the SVD method.

    """

    if rank is None:
        rank = np.random.randint(low=1, high=min(row, col)+1)

    success = True
    A = _build_matrix_rank_k(row, col, rank, cond, symmetric=False)
    # A = _build_matrix_rank_k_FAST(row, col, rank, really_fast=True)

    t0 = perf_counter()
    A_rank_svd = matrix_rank(A)
    svd_time = perf_counter() - t0

    t1 = perf_counter()
    A_rank_lup = RANK_COMPUTER(A, threshold)
    lup_time = perf_counter() - t1

    if A_rank_svd != A_rank_lup:
        success = False
        if diagnostics:
            print("*****")
            print("Failed! From tests_random_cond_rank with row:{0},column:{1},threshold:{2},cond:{3},rank:{4}".format(
                row, col, threshold, cond, rank))
            print("LUP_rank:{0}, SVD_rank:{1}".format(A_rank_lup, A_rank_svd))

    return success, lup_time, svd_time


def tests_random_uniform(row, col, interval, threshold=10**-10, diagnostics=False):
    """
    This function randomly (uniform distribution) generates a matrix of shape (row, col) 
    within given interval. See run_all_tests and tests_random_cond_rank documentation for 
    information on "threshold" and "diagnostics" along with the return values.

    """

    A = np.random.uniform(interval[0], interval[1], size=(row, col))
    success = True

    t0 = perf_counter()
    A_rank_svd = matrix_rank(A)
    svd_time = perf_counter() - t0

    t1 = perf_counter()
    A_rank_lup = RANK_COMPUTER(A, threshold)
    lup_time = perf_counter() - t1

    if A_rank_svd != A_rank_lup:
        success = False
        if diagnostics:
            print("*****")
            print("Failed! From tests_random_uniform with row:{0},column:{1},threshold:{2},interval:{3}".format(
                row, col, threshold, interval))
            print("LUP_rank:{0}, SVD_rank:{1}".format(A_rank_lup, A_rank_svd))

    return success, lup_time, svd_time


def tests_random_gaussian(row, col, mean=0, scale=1, threshold=10**-10, diagnostics=False):
    """
    This function randomly (gaussian distribution) generates a matrix of shape (row, col) 
    with mean and scale. See run_all_tests and tests_random_cond_rank documentation for 
    information on "threshold" and "diagnostics" along with the return values.

    """

    A = np.random.normal(loc=mean, scale=scale, size=(row, col))
    success = True

    t0 = perf_counter()
    A_rank_svd = matrix_rank(A)
    svd_time = perf_counter() - t0

    t1 = perf_counter()
    A_rank_lup = RANK_COMPUTER(A, threshold)
    lup_time = perf_counter() - t1

    if A_rank_svd != A_rank_lup:
        success = False
        if diagnostics:
            print("*****")
            print("Failed! From tests_random_gaussian with row:{0},column:{1},threshold:{2},mean:{3},scale:{4}".format(
                row, col, threshold, mean, scale))
            print("LUP_rank:{0}, SVD_rank:{1}".format(A_rank_lup, A_rank_svd))

    return success, lup_time, svd_time


def tests_random_binomial(row, col, n=1, p=0.1, threshold=10**-10, diagnostics=False):
    """
    This function randomly generates a binomial matrix of shape (row, col) with n and p.
    See run_all_tests and tests_random_cond_rank documentation for information on 
    "threshold" and "diagnostics" along with the return values.

    """

    A = np.random.binomial(n=n, p=p, size=(row, col))
    success = True

    t0 = perf_counter()
    A_rank_svd = matrix_rank(A)
    svd_time = perf_counter() - t0

    t1 = perf_counter()
    A_rank_lup = RANK_COMPUTER(A, threshold)
    lup_time = perf_counter() - t1

    if A_rank_svd != A_rank_lup:
        success = False
        if diagnostics:
            print("*****")
            print("Failed! From tests_random_binomial with row:{0},column:{1},threshold:{2},n:{3},p:{4}".format(
                row, col, threshold, n, p))
            print("LUP_rank:{0}, SVD_rank:{1}".format(A_rank_lup, A_rank_svd))

    return success, lup_time, svd_time


def compute_mean_values(X):
    """
    For input X: [[x_0_0, ...],...,[x_j_0,...]], it computes avg_i : average of [x_i_0, .....].

    Returns
    _________

    mean_values: (List)
    mean_values contains the avg_i values such that mean_values = [avg_0,....,avg_j].

    """

    unzipped_X = zip(*X)
    mean_values = list(map(stat.mean, unzipped_X))
    return mean_values


def _build_matrix_rank_k(row, col, k, cond=1, symmetric=False):
    """
    Builds a random matrix A (2D numpy array) of shape=(row,col) with rank k.

    Parameters
    ----------

    row: (Integer)
    Number of rows of A

    col: (Integer)
    Number of columns of A

    k: (Integer)
    Target rank of A.

    cond: (Integer)
    Ratio of the largest singular value to the smallest non-zero singular value of A.

    symmetric: (Boolean)
    If True returns a symmetric matrix.


    Returns
    -------

    A : (2D array)
    Random matrix with rank k of shape=(row,col).

    """

    a = np.random.rand(row, col)
    if symmetric:
        a = a + a.T
        eigvals, u = np.linalg.eigh(a)
        s = np.flip(eigvals)
    else:
        u, s, vh = np.linalg.svd(a, full_matrices=False)
    max_singular = 10**7
    min_singular = 10**7 / cond
    # singular_values = np.linspace(min_singular, max_singular, k)
    singular_values = np.geomspace(min_singular, max_singular, k)
    s[:k] = singular_values
    s[k:] = 0

    if symmetric:
        u_s = u*s
        u_s = u_s / np.abs(u_s).mean()
        A = np.dot(u_s, u.T)
    else:
        u_s = u*s
        u_s = u_s / np.abs(u_s).mean()
        A = np.dot(u_s, vh)
    return A


def _build_matrix_rank_k_FAST(row, col, k, really_fast=False):
    """
    Builds a random matrix A (2D numpy array) of shape=(row,col) with rank k.
    This is a faster way of producing rank reduced random matrices than the SVD
    approach.


    Parameters
    ----------

    row: (Integer)
    Number of rows of A

    col: (Integer)
    Number of columns of A

    k: (Integer)
    Target rank of A.


    Returns
    -------

    A : (2D array)
    Random matrix with rank k of shape=(row,col).

    """

    A = np.random.normal(size=(row, col))
    if k >= min(col, row):
        return A
    else:
        if really_fast:
            k -= 1
            A[:, k:] = 1.1
        else:
            for i in range(k, col):
                A[:, i] = A[:, :k].dot(np.random.normal(size=(k)))

    return A


if __name__ == '__main__':
    row, col = 1000, 1000
    threshold, diagnostics = 10**-10, True
    runs = 30
    cond = 10**8

    RANK_COMPUTER = LUP_rank.rank_revealing_LUP
    # RANK_COMPUTER = LUP_rank.rank_revealing_LUP_GPU
    # RANK_COMPUTER = other_rank_computers.rank_torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", help="Matrix dimensions. Eg: '--shape 10,10' means row=10 and col=10")
    parser.add_argument("--runs", help="Number of runs over which the results will be averaged")
    parser.add_argument("--cond", help="Ratio of largest singular value to the smallest non-zero singular value for test_cond_rank. Eg: '1e3' is interpreted as '1000'")
    parser.add_argument("--threshold", help="A threshold parameter. Eg: '1e3' is interpreted as '1000'. 10^-10 is used by default")
    parser.add_argument("--diagnostics", help="0 for turning off diagnostics and 1 for turning on diagnostics. By default diagnostics is set to True")
    parser.add_argument("--test_mode", help="0 for CPU, 1 for GPU, 2 for Torch's matrix_rank")
    args = parser.parse_args()
    
    if args.shape is not None:
        row, col = args.shape.split(',')
        row, col = int(row), int(col)
    if args.runs is not None:
        runs = int(args.runs)
    if args.cond is not None:
        cond = float(args.cond)
    if args.threshold is not None:
        threshold = float(args.threshold)
    if args.diagnostics is not None:
        diagnostics = bool(int(args.diagnostics))
    if args.test_mode is not None:
        if args.test_mode == '0':
            RANK_COMPUTER = LUP_rank.rank_revealing_LUP
        elif args.test_mode == '1':
            RANK_COMPUTER = LUP_rank.rank_revealing_LUP_GPU
        elif args.test_mode == '2':
            RANK_COMPUTER = other_rank_computers.rank_torch

    print("Testing:", RANK_COMPUTER.__name__)
    print("row:",row,"col:",col,"runs:",runs,"cond:",cond,"threshold:",threshold,"diagnostics:",diagnostics)

    params = {}
    params['test_uniform'] = [row, col, [-100, 100], tests_random_uniform]
    # params['test_gaussian'] = [row, col, 0, 10, tests_random_gaussian]
    # params['test_binomial_01'] = [row, col, 1, 0.1, tests_random_binomial]
    # params['test_binomial_05'] = [row, col, 1, 0.5, tests_random_binomial]
    # params['test_binomial_09'] = [row, col, 1, 0.9, tests_random_binomial]
    # params['test_cond_rank'] = [row, col, cond, None, tests_random_cond_rank]

    run_all_tests(params, threshold, diagnostics, runs)
