from setuptools import setup, Extension
import numpy

ext_args = dict(
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    runtime_library_dirs=[],
    extra_link_args=[]
    )

ext_modules = [
    Extension(name="gvar._gvarcore", sources=["src/gvar/_gvarcore.pyx"], **ext_args),
    Extension(name= "gvar._svec_smat", sources=["src/gvar/_svec_smat.pyx"], **ext_args),
    Extension(name="gvar._utilities", sources=["src/gvar/_utilities.pyx"], **ext_args),
    Extension(name="gvar.dataset", sources=["src/gvar/dataset.pyx"], **ext_args),
    Extension(name="gvar._bufferdict", sources=["src/gvar/_bufferdict.pyx"], **ext_args),
    ]

setup(ext_modules=ext_modules)
