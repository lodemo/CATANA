#import pyximport
#pyximport.install()
import fastdist
import numpy as np

import timeit

import facedist





FA = np.zeros((90000, 100))

facedist.mean_dist(FA)



A = np.random.randn(100, 200)


F = []
for i in range(1000):
    F.append(np.array([np.linspace(0.1,1.5,1700) for i in range(100)]))

F = np.array(F)

print F.shape


D1 = np.sqrt(np.square(A[np.newaxis,:,:]-A[:,np.newaxis,:]).sum(2))
D2 = fastdist.dist(A)

D3 = facedist.dist(F)

print np.allclose(D1, D2)
# True


#print timeit.timeit('np.sqrt(np.square(A[np.newaxis,:,:]-A[:,np.newaxis,:]).sum(2))', number=100, setup='import numpy as np; A = np.random.randn(100, 200)')

print timeit.timeit('fastdist.dist(A)', number=1, setup='import numpy as np; import fastdist; A = np.random.randn(1000, 2000)')



