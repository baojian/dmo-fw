# -*- coding: utf-8 -*-
"""
This is a wrapper of head and tail projection. To generate sparse_module.so
file, please use the following command (suppose you have Linux/MacOS/MacBook):
    python setup.py build_ext --inplace
I didn't know how to use GCC compiler under Windows. But if you have a C
compile, you can definitely generate above mentioned file in some way.
"""

import os
import numpy
from setuptools import setup
from distutils.core import Extension

here = os.path.abspath(os.path.dirname(__file__))

src_files = ['algo_wrapper/c/main_wrapper.c',
             'algo_wrapper/c/head_tail_proj.c',
             'algo_wrapper/c/fast_pcst.c',
             'algo_wrapper/c/sort.c']
compile_args = ['-shared', '-Wall', '-g', '-O3', '-fPIC', '-std=c11', '-lpython2.7', '-lm']
# calling the setup function
setup(
    name='sparse_module',
    version='0.0.1',
    description='----',
    long_description='----',
    url='----',
    author='----',
    author_email='----',
    include_dirs=[numpy.get_include()],
    license='MIT',
    packages=['sparse_modules'],
    classifiers=("Programming Language :: Python :: 2",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: POSIX :: Linux",),
    install_requires=['numpy', 'matplotlib', 'networkx', 'scipy'],
    headers=['algo_wrapper/c/head_tail_proj.h',
             'algo_wrapper/c/fast_pcst.h',
             'algo_wrapper/c/sort.h'],
    ext_modules=[Extension('sparse_module',
                           sources=src_files,
                           language="C",
                           extra_compile_args=compile_args,
                           include_dirs=[numpy.get_include()])],
    keywords='sparse learning, structure sparsity, head/tail projection')
