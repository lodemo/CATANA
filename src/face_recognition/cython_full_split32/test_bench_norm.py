import numpy as np
from scipy.spatial.distance import cdist

import time


F = []
for i in range(1000):
    F.append(np.array([np.linspace(0.1,1.5,1700) for i in range(100)]))

F = np.array(F)

print F.shape

start = time.time()

nrow = F.shape[0]
ncol = F.shape[1]

D = np.zeros((nrow, nrow), np.double)

for ii in range(nrow):
    for jj in range(ii + 1, nrow):

        rd = np.mean(cdist(F[ii], F[jj], metric='euclidean'))

        D[ii, jj] = rd
        D[jj, ii] = rd  # because D is symmetric

print "Took:", time.time() - start
