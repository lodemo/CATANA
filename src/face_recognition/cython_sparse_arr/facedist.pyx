
# MIT License
# 
# Copyright (c) 2017 Moritz Lode
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import dok_matrix
cimport numpy as np
cimport cython

from cython.parallel cimport prange

# don't use np.sqrt - the sqrt function from the C standard library is much
# faster
from libc.math cimport sqrt



# disable checks that ensure that array indices don't go out of bounds. this is
# faster, but you'll get a segfault if you mess up your indexing.
@cython.boundscheck(False)
# this disables 'wraparound' indexing from the end of the array using negative
# indices.
@cython.wraparound(False)
def mean_dist(np.ndarray A):

    # declare C types for as many of our variables as possible. note that we
    # don't necessarily need to assign a value to them at declaration time.
    cdef:
        # Py_ssize_t is just a special platform-specific type for indices
        Py_ssize_t nrow = A.shape[0]
        Py_ssize_t ncol = A.shape[1]
        Py_ssize_t ii, jj, nn

        # this line is particularly expensive, since creating a numpy array
        # involves unavoidable Python API overhead
        np.ndarray[np.float64_t, ndim=1] D = np.zeros( (nrow*(nrow-1)/2), np.float64)
        
        np.float64_t rd
       

    # another advantage of using Cython rather than broadcasting is that we can
    # exploit the symmetry of D by only looping over its upper triangle

    #with nogil:
    for ii in prange(nrow, nogil=True, schedule='static', num_threads=24):
        for jj in range(ii + 1, nrow):

            with gil:
                rd = np.mean(cdist(A[ii], A[jj], metric='euclidean'))
                nn = ii+jj*(jj-1)/2

            D[nn] = rd

    return D
