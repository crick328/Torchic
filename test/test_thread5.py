import cupy as cp

kernel = cp.RawKernel(r'''
extern "C" __global__
void atomic_add_3d(float* grad, float* update, int dim0, int dim1, int dim2) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (b < dim0 && s < dim1 && d < dim2) {
        int index = b * dim1 * dim2 + s * dim2 + d;
        atomicAdd(&grad[index], update[index]);
    }
}
''', 'atomic_add_3d')


def atomic_add(grad: cp.ndarray, contrib: cp.ndarray):
    # Configura i blocchi e la griglia (multiplo di 32)
    dim0, dim1, dim2 = grad.shape

    block_shape = (8, 8, 16)  # numero di thread per blocco
    grid_shape = (dim0 // block_shape[0] + 1, dim1 // block_shape[1] + 1, dim2 // block_shape[2] + 1)

    # Esegui il kernel
    kernel(grid_shape, block_shape, (grad, contrib, dim0, dim1, dim2))

# Esempio di uso
SHAPE = (2,320,512)

# Inizializza i gradienti e gli aggiornamenti
grad = cp.ones(SHAPE, dtype=cp.float32)
contrib = cp.ones(SHAPE, dtype=cp.float32)


import time

start = time.time()
for _ in range(0, 100000):
    # Esegui l'operazione di atomic add
    atomic_add(grad, contrib)
print("Tempo:", time.time() - start)
print(grad)