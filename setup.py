from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension("LB_D2Q9.dimensionless.cython_dimensionless",
              sources=["LB_D2Q9/dimensionless/cython_dimensionless.pyx"],
              include_dirs = [np.get_include()]),

    Extension("LB_D2Q9.OLD.pipe_cython",
              sources=["LB_D2Q9/OLD/pipe_cython.pyx"],
              include_dirs = [np.get_include()])
]

setup(
    name='2d-lb',
    version='0.01',
    packages=['LB_D2Q9', 'LB_D2Q9.OLD', 'LB_D2Q9.dimensionless'],
    include_package_data=True,
    package_data={'':["*.cl"]},
    url='',
    license='',
    author='Bryan Weinstein, Matheus C. Fernandes',
    author_email='bweinstein@seas.harvard.edu',
    description='',
    ext_modules = cythonize(extensions, annotate=True),
    include_dirs = [np.get_include()]
)
