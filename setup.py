"""
Created by G. Peter Lepage (Cornell University) on 9/2011.
Copyright (c) 2011-17 G. Peter Lepage.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version (see <http://www.gnu.org/licenses/>).

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

# from setuptools import setup, Extension

from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
import numpy

GVAR_VERSION = '8.3.1'

# create gvar/_version.py so gvar knows its version number
with open("src/gvar/_version.py","w") as version_file:
    version_file.write(
        "# File created by lsqfit setup.py\nversion = '%s'\n"
        % GVAR_VERSION
        )

# extension modules
# Add explicit directories to the ..._dirs variables if
# the build process has trouble finding the gsl library
# or the numpy headers. This should not be necessary if
# gsl and numpy are installed in standard locations.
ext_args = dict(
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    runtime_library_dirs=[],
    extra_link_args=[]
    )

ext_modules = [
    Extension("gvar._gvarcore", ["src/gvar/_gvarcore.pyx"], **ext_args),
    Extension( "gvar._svec_smat", ["src/gvar/_svec_smat.pyx"], **ext_args),
    Extension("gvar._utilities", ["src/gvar/_utilities.pyx"], **ext_args),
    Extension("gvar.dataset", ["src/gvar/dataset.pyx"], **ext_args),
    Extension("gvar._bufferdict", ["src/gvar/_bufferdict.pyx"], **ext_args),
    ]

# packages
packages = ["gvar"]
package_dir = dict(gvar="src/gvar")
package_data = dict(gvar=['../gvar.pxd', '_svec_smat.pxd', '_gvarcore.pxd'])

setup(name='gvar',
    version=GVAR_VERSION,
    description='Utilities for manipulating correlated Gaussian random variables.',
    author='G. Peter Lepage',
    author_email='g.p.lepage@cornell.edu',
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
    ext_modules= cythonize(ext_modules),
    install_requires=['cython>=0.17', 'numpy>=1.7'], # for pip (distutils ignores)
    requires=['cython (>=0.17)', 'numpy (>=1.7)'],   # for distutils
    url="https://github.com/gplepage/gvar.git",
    license='GPLv3+',
    platforms='Any',
    long_description="""\
    This package facilitates the creation and manipulation of arbitrarily
    complicated (correlated) multi-dimensional Gaussian random variables.
    The random variables are represented by a new data type that can be used
    in arithmetic expressions and pure Python functions. Such
    expressions/functions create new Gaussian random variables
    while automatically tracking statistical correlations between the new
    and old variables. This data type is useful for simple error propagation
    but also is heavily used by the Bayesian least-squares fitting module
    lsqfit.py (to define priors and specify fit results, while accounting
    for correlations between all variables).

    This package uses numpy for efficient array arithmetic, and cython to
    compile efficient core routines and interface code.
    """
    ,
    classifiers = [                     #
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering'
        ],
)