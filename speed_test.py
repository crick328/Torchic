import numpy as np
import time


nums = np.array([0,33,-3,5,0,1])
#nums = np.zeros(200)

#t = time.time()
#
#print(f"Metodo 1. Tempo impiegato: {((time.time() - t)):.10f} secondi")


t = time.time()
#print(nums.nonzero()[0])
np.put(nums, nums.nonzero()[0], 1)
print(f"Metodo 2. Tempo impiegato: {((time.time() - t)):.10f} secondi")
print(nums)