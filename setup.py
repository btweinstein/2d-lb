from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension("LB_D2Q9.cython_dimensionless",
              sources=["src/LB_D2Q9/cython_dimensionless.pyx"],
              include_dirs = [np.get_include()]),

    Extension("LB_D2Q9_OLD.pipe_cython",
              sources=["src/LB_D2Q9_OLD/pipe_cython.pyx"],
              include_dirs = [np.get_include()])
]

setup(
    name='2d-lb',
    version='0.01',
    package_dir={'':'src'},
    packages=['LB_D2Q9', 'LB_D2Q9_OLD'],
    url='',
    license='',
    author='Bryan Weinstein, Matheus C. Fernandes',
    author_email='bweinstein@seas.harvard.edu',
    description='',
    ext_modules = cythonize(extensions, annotate=True, reload_support=True),
    include_dirs = [np.get_include()]
)
