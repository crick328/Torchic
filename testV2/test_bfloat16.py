import cupy as cp
import time
from preset import torchic
from torchic import Tensor

# Verifica che bfloat16 sia disponibile
try:
    # bfloat16 viene registrato tramite ml_dtypes all'interno di cupy
    import ml_dtypes
    bfloat16_dtype = cp.dtype('bfloat16')
    print("Supporto bfloat16 rilevato con successo!")
    test = cp.array([1.,1.,2.,3.], dtype=bfloat16_dtype)
    print(type(bfloat16_dtype))
    e = Tensor([1.,1.,2.,3.], dtype=bfloat16_dtype)
    e.to(bfloat16_dtype)
    #print(test.dtype.itemsize*8, test.dtype.kind, test.dtype)
    exit()
except Exception as e:#(ImportError, TypeError):
    print(e)
    print("Errore: assicurati di avere installato 'ml_dtypes' e di usare CuPy v14+.")
    exit(1)

def benchmark_matmul(dtype, size=4096, warmup=5, reps=20):
    # Generazione dei dati sulla GPU
    # Nota: bfloat16 potrebbe richiedere il casting esplicito da float32 o l'uso di ml_dtypes
    if dtype == bfloat16_dtype:
        a = cp.random.randn(size, size).astype(cp.float32).astype(bfloat16_dtype)
        b = cp.random.randn(size, size).astype(cp.float32).astype(bfloat16_dtype)
    else:
        a = cp.random.randn(size, size).astype(dtype)
        b = cp.random.randn(size, size).astype(dtype)

    # Crea gli eventi CUDA per una misurazione del tempo accurata sulla GPU
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    # 1. WARM-UP (necessario per far compilare i kernel a CUDA senza falsare i tempi)
    for _ in range(warmup):
        c = cp.matmul(a, b)
    
    cp.cuda.Stream.null.synchronize()

    # 2. BENCHMARK EFFETTIVO
    start_event.record()
    for _ in range(reps):
        c = cp.matmul(a, b)
    end_event.record()
    
    # Sincronizza per assicurarti che la GPU abbia terminato tutti i calcoli
    end_event.synchronize()

    # Calcola il tempo medio per singola operazione (in millisecondi)
    elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    avg_time_ms = elapsed_ms / reps

    # Calcolo dei TFLOPS teorici eseguiti
    # Operazioni in una matmul di matrici quadrate (N x N) @ (N x N) = 2 * N^3 operazioni
    num_ops = 2.0 * (size ** 3)
    tflops = (num_ops / (avg_time_ms / 1000.0)) / 1e12

    return avg_time_ms, tflops

if __name__ == "__main__":
    # Dimensione delle matrici (N x N)
    MATRIX_SIZE = 4096
    REPETITIONS = 30
    
    print(f"Esecuzione del Benchmark su matrici {MATRIX_SIZE}x{MATRIX_SIZE} ({REPETITIONS} ripetizioni)...")
    print("-" * 65)

    # Test con float16
    try:
        f16_time, f16_tflops = benchmark_matmul(cp.float16, size=MATRIX_SIZE, reps=REPETITIONS)
        print(f"FP16:       {f16_time:8.2f} ms per matmul | Performance: {f16_tflops:6.2f} TFLOPS")
    except Exception as e:
        print(f"Errore durante il test FP16: {e}")

    # Test con bfloat16
    try:
        bf16_time, bf16_tflops = benchmark_matmul(bfloat16_dtype, size=MATRIX_SIZE, reps=REPETITIONS)
        print(f"BF16:       {bf16_time:8.2f} ms per matmul | Performance: {bf16_tflops:6.2f} TFLOPS")
    except Exception as e:
        print(f"Errore durante il test BF16: {e}")
        print("Nota: Se usi schede video molto vecchie (es. pre-Ampere), il supporto hardware nativo per BF16 potrebbe mancare o essere emulato.")

    print("-" * 65)





import cupy as cp
import ml_dtypes

# Definiamo i tipi
bf16 = cp.dtype('bfloat16')
f16 = cp.float16
f32 = cp.float32

print("=== TEST 1: UNDERFLOW (Range Dinamico) ===")
# Scegliamo un numero molto piccolo, es. 1e-10 (10^-10)
# - FP16 non può rappresentarlo (diventa 0.0)
# - BF16 e FP32 possono rappresentarlo senza problemi
valore_piccolo = 1e-10

print(f"Valore di partenza: {valore_piccolo}")
print(f"In float16:  {cp.array([valore_piccolo], dtype=f16)[0]}")
print(f"In bfloat16: {cp.array([valore_piccolo], dtype=bf16)[0]}")
print(f"In float32:  {cp.array([valore_piccolo], dtype=f32)[0]}")

print("\n" + "="*40 + "\n")

print("=== TEST 2: PRECISIONE (Mantissa) ===")
# Proviamo a fare 1.0 + 0.001 (1e-3)
# - BF16 ha troppa poca precisione e arrotonderà a 1.0
# - FP16 e FP32 mostreranno correttamente ~1.001
valore_somma = 1.0 + 0.001

print("Operazione: 1.0 + 0.001")
print(f"In float16:  {cp.array([valore_somma], dtype=f16)[0]}")
print(f"In bfloat16: {cp.array([valore_somma], dtype=bf16)[0]}")
print(f"In float32:  {cp.array([valore_somma], dtype=f32)[0]}")