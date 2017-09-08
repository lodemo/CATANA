import numpy as np
import time
import facedist

import time


F = []
for i in range(1000):
    F.append(np.array([np.linspace(0.1,1.5,1700) for i in range(100)]))

F = np.array(F)

start = time.time()

D3 = facedist.mean_dist(F)

end = time.time()
print "Took:", end - start

