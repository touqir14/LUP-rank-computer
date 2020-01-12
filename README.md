# LUP-rank-computer
An LUP based algorithm for Computing numerical rank of a matrix in Python.

### Why another rank computing algorithm?
The most common way to compute the numerical rank of a matrix in Python is to use Numpy's [matrix_rank](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.matrix_rank.html) function which is based on the SVD method. However, even for matrices with moderately large dimensions the "matrix_rank" function takes noticeably long to execute. For example, it can take upto a few minutes for a matrix with dimension 10000 by 10000 in most modern desktop machines. My original goal was to devise a reasonably accurate method that works much faster specially for large matrices.

### How fast is the LUP-rank-computer?
I have benchmarked the algorithm on randomly generated (UAR) matrices of various dimensions and compared its execution time against matrix_rank's execution time. The results were averaged over 100 runs. [Here](https://github.com/touqir14/LUP-rank-computer/blob/master/test_LUP_rank.py) is the benchmarking source code. The experiments were carried on a linux machine with python 3.8, cython 0.29.14 and gcc 9.2.0

| Matrix Size | Speedup |
|-------------|---------|
| 100x100     | 1.411x  |
| 100x1000    | 3.109x  |
| 100x10000   | 5.373x  |
| 1000x1000   | 6.758x  |
| 1000x10000  | 5.417x  |
| 10000x10000 | 23.607x |

### How should I run the algorithm?
Before calling the function, make sure you have cython and a C compiler installed. With cython installed, you will need to first compile the code using cython: 
```
python cython_setup.py build_ext --inplace
```

Now you can compute the rank of matrix A using the following code:
```python
from LUP_rank import rank_revealing_LUP
import numpy as np

A = np.random.rand(1000,1000)
rank, _ = rank_revealing_LUP(A, 10**-10)
```
Feel free to contact me if you have any questions or comments.
