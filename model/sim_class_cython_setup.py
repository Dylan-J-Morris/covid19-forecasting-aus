#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy
import sys, os

sys.argv = ["compile.py", "build_ext", "--inplace"]

from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize

import glob, os

for ff in ("*.c", "*.html"):
    for f in glob.glob(ff):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

ext_modules = [Extension("sim_class_cython", ["sim_class_cython.pyx"])]

os.chdir('model')

setup(name="sim_class_cython", 
      ext_modules=cythonize(ext_modules, 
                            annotate=True, 
                            language_level = 3), 
      include_dirs=[numpy.get_include()])