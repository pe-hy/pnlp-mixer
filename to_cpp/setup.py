from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
import os
import sys

def find_mkl():
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        mkl_include = os.path.join(conda_prefix, 'include')
        mkl_lib = os.path.join(conda_prefix, 'lib')
        if os.path.exists(os.path.join(mkl_include, 'mkl.h')):
            return mkl_include, mkl_lib

    # Check common locations
    common_locations = [
        '/opt/intel/mkl',
        '/usr/local/intel/mkl',
        '/usr/include/mkl',
    ]
    for loc in common_locations:
        if os.path.exists(os.path.join(loc, 'include', 'mkl.h')):
            return os.path.join(loc, 'include'), os.path.join(loc, 'lib')

    print("MKL not found. Please install MKL or set MKLROOT environment variable.")
    sys.exit(1)

mkl_include, mkl_lib = find_mkl()

setup(
    name='FFFTorch',
    ext_modules=[
        cpp_extension.CppExtension(
            'fff_extension', 
            ['FFF/fff_extension.cpp'],
            include_dirs=[mkl_include],
            library_dirs=[mkl_lib],
            libraries=['mkl_rt'],
            extra_compile_args=['-fopenmp', f'-I{mkl_include}'],
            extra_link_args=['-Wl,-rpath,' + mkl_lib, '-lmkl_rt', '-liomp5']
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    packages=find_packages(),
    version='0.1'
)