#!/bin/bash
ROOT=/home/$USER/anaconda3/envs
PYTHON="${ROOT}/python-3.7"
PYTHON_INCLUDE="${PYTHON}/include/python3.7m"
NUMPY="/lib/python3.7/site-packages/numpy/core/include"
NUMPY_INCLUDE="${PYTHON}${NUMPY}"
PYTHON_LIB="${ROOT}/python-3.7/lib"
gcc -g -shared -Wall -Wextra -fPIC -std=c11 -O3 -lc -lm -lpthread \
-I${PYTHON_INCLUDE} -I${NUMPY_INCLUDE} -L${PYTHON_LIB} \
-o sparse_module.so main_wrapper.c head_tail_proj.c fast_pcst.c \
sort.c fast_pcst.h sort.h head_tail_proj.h -lpython3.7m -lgfortran
