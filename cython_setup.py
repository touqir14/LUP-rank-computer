from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
import os

folder_path = os.path.dirname(os.path.abspath(__file__))
LUP_cython_path = os.path.join(folder_path, 'LUP_cythons.pyx')

print(LUP_cython_path)

os.environ['CFLAGS'] = '-O4'
setup(
	ext_modules = cythonize(LUP_cython_path),
	include_dirs = [np.get_include()]
)
