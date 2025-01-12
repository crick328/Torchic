#  I kernel sono delle funzioni (in questo caso scritta in C)
#  che vengono eseguite su GPU.
#  L'astrazione a livello software per gestire i calcoli
#  su GPU e' composta da grids, blocks e threads. Una grid
#  contiene piu' blocks, che a sua volta contiene e gestisce
#  piu' threads. Sono questi che effettuano i calcoli e 
#  permettono la parallelizzazione.
#  Il kernel viene distribuito alle grids, che a loro volta
#  lo distribuiscono ai loro blocks, che lo danno ai loro
#  threads. Ogni thread esegue il kernel in parallelo, usandolo
#  per calcolare solo l'ingresso che gli corrisponde. In questo
#  modo, la GPU e' in grado di fare un'enorme quantita' di
#  calcoli elementari in poco tempo!
#
#  Grid:
#  +----------------------------------------------------+
#  |    Blocks:                                         |
#  |   +------------+   +------------+   +------------+ |
#  |   | Block 0    |   | Block 1    |   | Block N    | |
#  |   | +--------+ |   | +--------+ |   | +--------+ | |
#  |   | |Thread 0| |   | |Thread 0| |   | |Thread 0| | |
#  |   | |Thread 1| |   | |Thread 1| |   | |Thread 1| | |
#  |   | |  ...   | |   | |  ...   | |   | |  ...   | | |
#  |   | |Thread M| |   | |Thread M| |   | |Thread M| | |
#  |   | +--------+ |   | +--------+ |   | +--------+ | |
#  |   +------------+   +------------+   +------------+ |
#  |                                                    |
#  |                                                    |
#  +----------------------------------------------------+
#  |                   Global Memory                    |
#  |  +----------------------------------------------+  |
#  |  |                                              |  |
#  |  |    Data and Variables for All Threads        |  |
#  |  |                                              |  |
#  |  +----------------------------------------------+  |
#  +----------------------------------------------------+
#  |                   Shared Memory                    |
#  |  +----------------------------------------------+  |
#  |  |   Data Accessible by Threads within a Block  |  |
#  |  +----------------------------------------------+  |
#  +----------------------------------------------------+
#
#  Ogni blocco possiede una shared memory. Questa permette
#  hai thread del blocco di collaborare.
#  Ogni thread possiede un index univoco (thread ID, tid)
#  calcolabile. Ogni thread riceve lo stesso identico kernel.
#
#  Riassumendo:
#  kernel = gpu function
#  grid = insieme di blocchi
#  grid size = numero di blocchi nella grid
#  block = insieme di threads
#  block size = n di threads nel block
#  thread = runna il kernel in parallelo
import cupy as cp


# il nome del kernel puo' anche non coincidere con quello
# della funzione python
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

# sfrutto cuda.AtomicAdd per avere garanzia dell'integrita'
# del gradiente
# NOTE: AtomicAdd supporta nativamente int, float e double. Per
# gli altri dtype, bisogna accumularli castandoli al tipo piu'
# vicino tra questi tre
def accumulate_grad(grad: cp.ndarray, grad_contrib: cp.ndarray) -> None:
    assert grad.shape == grad_contrib.shape, "Shapes of gradients must match"
    assert grad.dtype == grad_contrib.dtype, "Dtypes of gradients must match"
    assert grad.dtype == cp.float32, "Dtype must be float32"

    '''
    if cp.any(cp.isnan(grad)):
        raise ValueError("NaN detected in gradient")

    if cp.any(cp.isnan(grad_contrib)):
        print(grad_contrib, grad_contrib.shape)
        raise ValueError("NaN detected in gradient contribution")
    '''

    # dimensione del blocco (numero di threads al suo interno)
    block_size = 256 # <= 1024, deve essere una potenza di 2
    grid_size = (grad.size + block_size - 1) // block_size

    accumulate_grad_kernel((grid_size,), (block_size,), (grad, grad_contrib, grad.size))


# la soluzione finale sarebbe implementare un mutex, messo come
# attributo di BackwardFunction, per ogni cella del gradiente.
# Ogni thread deve prima riuscire ad acquisire il lock per poter fare il calcolo
# o rimane bloccato in un while. Non e' molto pulito, ma, se il problema deriva
# davvero dall'integrita' dei gradienti, allora dovrebbe risolverlo.
# Detto questo, secondo me l'intero approccio al parallelismo non va bene.
# Innanzitutto, devi decidere se lanciare tutto in asincrono o tenere la
# null stream come punto di riferimento e di sincronizzazione, poi devi
# strutturare una logica chiara che sa a quali rami punta e come riconoscerli
# .