import cupy as cp

# Inizializza il mutex a 0 (libero)
mutex = cp.zeros(1, dtype=cp.int32)

# Inizializza il dato che sarà condiviso tra i thread
data = cp.zeros(1, dtype=cp.int32)

# Kernel per aggiornare il valore di 'data' in modo sicuro usando il mutex
update_data_kernel = cp.RawKernel(r'''
extern "C" __global__ void update_data(int *data, int *mutex, int value) {
    // Funzione lock
    while (atomicCAS(mutex, 0, 1) != 0) {
        // Attendi finché non si libera
    }

    // Sezione critica: aggiornamento del valore di 'data'
    *data += value;

    // Funzione unlock: Rilascia il lock (mutex)
    atomicExch(mutex, 0);
}
''', 'update_data', options=('--std=c++11',))

# Numero di thread
num_threads = 256

# Lancia il kernel per aggiornare 'data' usando il mutex
for i in range(num_threads):
    update_data_kernel((1,), (1,), (data, mutex, i))

# Sincronizza per assicurarsi che tutti i thread abbiano completato
cp.cuda.Stream.null.synchronize()

# Stampa il risultato finale di 'data'
print("Valore finale di 'data':", data.get())
