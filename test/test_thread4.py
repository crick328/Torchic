import cupy as cp

kernel = cp.RawKernel(r'''
extern "C" __global__
void atomic_add_general(float* grad, float* grad_contrib, int* shape, int n_dims, int grad_size) {
    int flat_index = 0;
    int dim_multiplier = 1;
                      
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // controllo che ci sia un ingresso corrispondente da calcolare
    if (thread_id < grad_size) {
        // calcolo l'index su cui lavorera' il singolo thread
        // (offsettato in base all'index del thread)
        
        for (int i=0; i < n_dims; ++i) {
            int idx = thread_id / dim_multiplier % shape[i];
            flat_index += idx * dim_multiplier;
            dim_multiplier *= shape[i];
        }
        
        atomicAdd(&grad[thread_id], grad_contrib[thread_id]);
    }
}
''', 'atomic_add_general')


def atomic_add(grad: cp.ndarray, contrib: cp.ndarray):
    # Configura i blocchi e la griglia (multiplo di 32)
    block_size = 256
    grid_size = (grad.size + block_size - 1) // block_size

    # non posso passare una tupla
    shape = cp.array(grad.shape, dtype=cp.int32)

    # Esegui il kernel
    kernel((grid_size,), (block_size,), (grad, contrib, shape, grad.ndim, grad.size))

# Esempio di uso
SHAPE = (2,5,4)

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
#assert cp.all(grad == 2)