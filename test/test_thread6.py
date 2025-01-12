import cupy as cp
import time


accumulate_grad_kernel = cp.RawKernel(r'''
extern "C" __global__
void accumulate_grad(float* grad, const float* grad_contrib, int grad_size, int* init_flag) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Controlla se il thread è dentro i limiti
    if (tid < grad_size) {
        // Inizializzazione atomica del gradiente se la flag è a 0
        if (atomicCAS(init_flag, 0, 1) == 0) {
            for (int i = 0; i < grad_size; i++) {
                grad[i] = grad_contrib[i];
            }
        } else {
            // Se è già stato inizializzato, accumula il gradiente
            atomicAdd(&grad[tid], grad_contrib[tid]);
        }
    }
}
''', 'accumulate_grad')

# Esempio di uso
grad = None  # Il gradiente è inizialmente None
init_flag = cp.zeros(1, dtype=cp.int32)  # Flag per gestire l'inizializzazione

# Creiamo una stream asincrona e chiamiamo il kernel
stream1 = cp.cuda.Stream(non_blocking=True)
stream2 = cp.cuda.Stream(non_blocking=True)

grad_contrib = cp.ones((20,20), dtype=cp.float32)
grad_size = grad_contrib.size

with stream1:
    accumulate_grad_kernel((32,), (32,), (grad, grad_contrib, grad_size, init_flag))

with stream2:
    accumulate_grad_kernel((32,), (32,), (grad, grad_contrib, grad_size, init_flag))

#stream1.synchronize()
#stream2.synchronize()
time.sleep(15)
print(grad)