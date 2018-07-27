"""
Created by G. Peter Lepage (Cornell University) on 9/2011.
Copyright (c) 2011-18 G. Peter Lepage.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version (see <http://www.gnu.org/licenses/>).

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

GVAR_VERSION = '9.0'

from distutils.core import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext as _build_ext
from distutils.command.build_py import build_py as _build_py

# compile from existing .c files if USE_CYTHON is False
USE_CYTHON = False

class build_ext(_build_ext):
    # delays using numpy and cython until they are installed;
    # cython is optional (set USE_CYTHON)
    # this code adapted from https://github.com/pandas-dev/pandas setup.py
    def build_extensions(self):
        import numpy
        if USE_CYTHON:
            from Cython.Build import cythonize
            self.extensions = cythonize(self.extensions)
        numpy_include = numpy.get_include()
        for ext in self.extensions:
            ext.include_dirs.append(numpy_include)
        _build_ext.build_extensions(self)

class build_py(_build_py):
    # adds version info
    def run(self):
        """ Append version number to gvar/__init__.py """
        with open('src/gvar/__init__.py', 'a') as gvfile:
            gvfile.write("\n__version__ = '%s'\n" % GVAR_VERSION)
        _build_py.run(self)

# extension modules
# Add explicit directories to the ..._dirs variables if
# the build process has trouble finding the gsl library
# or the numpy headers. This should not be necessary if
# gsl and numpy are installed in standard locations.
ext_args = dict(
    include_dirs=[],
    library_dirs=[],
    runtime_library_dirs=[],
    extra_link_args=[]
    )

ext = '.pyx' if USE_CYTHON else '.c'

ext_modules = [
    Extension("gvar._gvarcore", ["src/gvar/_gvarcore" + ext], **ext_args),
    Extension( "gvar._svec_smat", ["src/gvar/_svec_smat" + ext], **ext_args),
    Extension("gvar._utilities", ["src/gvar/_utilities" + ext], **ext_args),
    Extension("gvar.dataset", ["src/gvar/dataset" + ext], **ext_args),
    Extension("gvar._bufferdict", ["src/gvar/_bufferdict" + ext], **ext_args),
    ]

# packages
packages = ["gvar"]
package_dir = dict(gvar="src/gvar")
package_data = dict(gvar=['../gvar.pxd', '_svec_smat.pxd', '_gvarcore.pxd'])

# pypi
with open('README.rst', 'r') as file:
    long_description = file.read()

setup(name='gvar',
    version=GVAR_VERSION,
    description='Utilities for manipulating correlated Gaussian random variables.',
    author='G. Peter Lepage',
    author_email='g.p.lepage@cornell.edu',
    cmdclass={'build_ext':build_ext, 'build_py':build_py},
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
    ext_modules= ext_modules,
    # for pip (distutils ignores):
    install_requires=['cython>=0.17', 'numpy>=1.7'] if USE_CYTHON else ['numpy>=1.7'],
    # for distutils (pip ignores):
    requires=['cython (>=0.17)', 'numpy (>=1.7)'] if USE_CYTHON else ['numpy (>=1.7)'],
    url="https://github.com/gplepage/gvar.git",
    license='GPLv3+',
    platforms='Any',
    long_description=long_description,
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