import cupy as cp
from cupyx.profiler import benchmark

# codice C
# il kernel viene distributo su piu' grid per essere eseguito.
# I thread sono unita' di esecuzione individuale che eseguono
# un singolo pezzettino del kernel (in questo caso un singolo
# ingresso del gradiente).
# I thread che si trovano nello stesso blocco pososno collaborare,
# e per questo ogni blocco ha un pezzettino di memoria condivisa
# da tutti i suoi thread, chiamata appunto shared memory.
# Il kernel viene distribuito su tutti i threads di tutti
# i blocchi di tutte le grid. E' in questo modo che la gpu
# raggiunge una enorme capacita' di parallelizzazione!
#
# Ogni thread riceve lo stesso identico kernel, che viene eseguito.
# Ognuno di questi thread possiede un identificatore univoco (tid) 
# calcolabile.

from collections import deque
puttanate = deque()
minchia = (1,None)

puttanate.append(minchia)
puttanate.append(minchia)
print(puttanate)

# kernel = gpu function
# grid = insieme di blocchi
# grid size = numero di blocchi
# block = insieme di threads
# block size = n di threads
# thread = runna il kernel 
# Definire il kernel CUDA per atomicAdd su float
accumulate_grad_kernel = cp.RawKernel(r'''
extern "C" __global__
void accumulate_grad(float* grad, const float* grad_contrib, int grad_size) {
    // calcolo l'ID del thread (flat index univoco)
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // controllo che ci sia un ingresso corrispondente da calcolare
    if (tid < grad_size) {
        atomicAdd(&grad[tid], grad_contrib[tid]);
    }
}
''', 'accumulate_grad')

# Funzione per accumulare gradiente con atomicAdd
def accumulate_grad(grad, grad_contrib):
    assert grad.shape == grad_contrib.shape, "Shapes of gradients must match"

    # dimensione del blocco (numero di threads al sui interno)
    block_size = 256
    grid_size = (grad.size + block_size - 1) // block_size  # Dimensione della griglia
    print(grid_size)

    # Lanciare il kernel CUDA
    accumulate_grad_kernel((grid_size,), (block_size,), (grad, grad_contrib, grad.size))

# Esempio d'uso
#grad = cp.zeros(1000000, dtype=cp.float32)  # Gradiente inizializzato a zero
#grad_contrib = cp.ones_like(grad)  # Contributo da accumulare

# Accumulare gradiente usando atomicAdd
#accumulate_grad(grad, grad_contrib)

# Verificare il risultato
#print(cp.allclose(grad, cp.ones_like(grad)))  # Deve stampare True


SHAPE = (2,320,513)

# Inizializza i gradienti e gli aggiornamenti
grad = cp.ones(SHAPE, dtype=cp.float32)
contrib = cp.ones(SHAPE, dtype=cp.float32)


import time

#start = time.time()
#for _ in range(0, 100000):
    # Esegui l'operazione di atomic add
accumulate_grad(grad, contrib)
#print("Tempo:", time.time() - start)
print(grad)
assert cp.all(grad == 2)