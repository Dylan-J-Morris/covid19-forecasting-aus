from distutils.core import setup
from Cython.Build import cythonize
import cython
import numpy

setup(ext_modules = cythonize("model/sim_class_cython.pyx", language_level=3), include_dirs=[numpy.get_include()])