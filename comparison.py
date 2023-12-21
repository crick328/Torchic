import numpy as np


x = np.array([1.,-2.]) #1x2
A = np.array([
    [3.,5.,-1],
    [1.,2.,3.]
])  #2x3

chain_rule_tensor = np.ones(3) #1x3

# SBAGLIATI in shape e valori => la derivata della matmul non ha una matmul
#d_dx = np.ones(x.shape) @ A
#d_dA = x @ np.ones(A.shape)

d_dx = A.T
d_dA = x.T

print(d_dx)
print(d_dA)

dL_dx = chain_rule_tensor @ d_dx
dL_dA = np.outer(x, chain_rule_tensor)

print(dL_dx)
print(dL_dA)

# NOTE: nella chain rule, il * può essere qualsiasi tipo di prodotto: outer product, matmul, ecc.



































#A(T)@a
#print(b.transpose(1,0).__matmul__(a))
# x@A
#print(a.__matmul__(b))

# 1x2 @ 2x3 = 1x3
#d/dx = 1x2
#d/dA = 2x3
#dL/dk

# dL/dk * d/dA = 1x3 * 2x3
# dL/dk * d/dx = 1x3.sum() (=/=1) * 1x2 (=/=1) = 1x2

#crt = np.array([1.,1.,1.]) #1x3
# d/dx
#d_dx = np.dot(1, b).sum(axis=1) # stesso shape di x
#dL_dx = crt * B.sum(axis=1) # stesso shape di x
#print(d_dx)
#print(d_dx.shape)
# d/dA (invariata)

'''
print(np.array_equiv(np.ones((3,2)), 1))
print(np.array_equiv(1, np.ones(2)))
print(np.array_equiv(np.ones(3), np.ones(2)))
'''
