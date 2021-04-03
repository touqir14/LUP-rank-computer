<!---
For embedding latex, read this https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b
-->

# LUP-rank-computer
An LUP factorization based algorithm for Computing the numerical rank of a matrix in Python.

### Why another rank computing algorithm?
The most common way to compute the numerical rank of a matrix in Python is to use Numpy's [matrix_rank](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html) function which is based on the SVD method. However, even for matrices with moderately large dimensions the "matrix_rank" function takes noticeably long to execute. For example, it can take upto a few minutes for a matrix with dimension 10000 by 10000 in most modern desktop machines. My original goal was to devise a reasonably accurate method that works much faster specially for large matrices.

### How fast is the LUP-rank-computer?
I have benchmarked the algorithm on randomly generated (UAR) matrices of various dimensions and compared its execution time against matrix_rank's execution time. The results were averaged over 100 runs. [Here](https://github.com/touqir14/LUP-rank-computer/blob/master/test_LUP_rank.py) is the benchmarking source code. The experiments were carried on a linux machine with python 3.8, cython 0.29.14 and gcc 9.2.0. Below, Speedup is defined as <img src="https://render.githubusercontent.com/render/math?math=\dfrac{\text{Average runtime of Numpy's matrix_rank}}{\text{Average runtime of LUP-rank-computer}}">.

| Matrix Size | Speedup |
|-------------|---------|
| 100x100     | 1.411x  |
| 100x1000    | 3.109x  |
| 100x10000   | 5.373x  |
| 1000x1000   | 6.758x  |
| 1000x10000  | 5.417x  |
| 10000x10000 | 23.607x |
| 15000x15000 | 25.146x |

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
rank = rank_revealing_LUP(A)
```
If you have a CUDA compatible GPU and PyTorch >= v1.1.0, you can also use GPU supported acceleration:
```python
from LUP_rank import rank_revealing_LUP
import numpy as np

A = np.random.rand(1000,1000)
rank = rank_revealing_LUP_GPU(A)
```

The following table shows the benchmarks for the GPU accelerated LUP-rank-computer along with PyTorch's GPU accelerated [matrix_rank](https://pytorch.org/docs/master/generated/torch.matrix_rank.html) when compared with Numpy's matrix_rank. Experiments were carried out on Google Colab using a Tesla K80 GPU.

| Matrix Size | Speedup (GPU accelerated LUP-rank-computer) | Speedup (PyTorch's matrix_rank) |
|-------------|---------------------------------------------|---------------------------------|
| 100x100     | 0.0054x  | 0.0069x |
| 100x1000    | 0.0369x  | 0.0377x |
| 100x10000   | 0.2569x  | 0.2365x |
| 1000x1000   | 5.612x   | 0.8191x |
| 1000x10000  | 4.794x   | 2.9842x |
| 10000x10000 | 176.3x   | 9.8972x |
| 15000x15000 | 228.5x   | 10.986x |

Feel free to contact me if you have any questions or comments.
