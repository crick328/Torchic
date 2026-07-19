import sys
import time
import cupy as cp
from preset import torchic, compare_tensors, create_random_tensors, backward_and_compare
from torchic import nn, draw_graph, Tensor
from torchic.settings import set_seed
from torchic import set_default_device, set_default_precision
import cupyx as cpx

# IL DEVICE NON E CUDA!
set_default_device("cuda")
set_default_precision(32)




def benchmark_tensor():
    # 1. Prepariamo un mock di dati CuPy a monte per non calcolarlo nel test
    raw_data = cp.ones((100, 100), dtype=cp.float32)
    

    print("\n--- BENCHMARK TEMPO (OVERHEAD PYTHON) ---")
    ITERATIONS = 10_000_000
    
    # Test Costruttore Standard
    start_time = time.perf_counter_ns()
    for _ in range(ITERATIONS):
        t = Tensor(raw_data, _fast_init=False)
    end_time = time.perf_counter_ns()
    
    total_time_normal_ms = (end_time - start_time) / 1_000_000
    avg_time_normal_ns = (end_time - start_time) / ITERATIONS
    
    print(f"[Standard] Tempo totale per {ITERATIONS:,} istanziazioni: {total_time_normal_ms:.2f} ms")
    print(f"[Standard] Tempo medio per singolo Tensor: {avg_time_normal_ns:.2f} ns")

    # Test Costruttore Fast-Path
    start_time = time.perf_counter_ns()
    for _ in range(ITERATIONS):
        t = Tensor(raw_data, _fast_init=True)
    end_time = time.perf_counter_ns()
    
    total_time_fast_ms = (end_time - start_time) / 1_000_000
    avg_time_fast_ns = (end_time - start_time) / ITERATIONS
    
    print(f"[Fast-Init] Tempo totale per {ITERATIONS:,} istanziazioni: {total_time_fast_ms:.2f} ms")
    print(f"[Fast-Init] Tempo medio per singolo Tensor: {avg_time_fast_ns:.2f} ns")
    
    # Guadagno percentuale
    speedup = (avg_time_normal_ns / avg_time_fast_ns - 1) * 100
    print(f"\n>> Il metodo Fast-Init è più veloce del {speedup:.1f}% rispetto al costruttore standard.")


def benchmark_matmul():
   
    
    shape = (1000, 1000)
    # 1. Prepariamo un mock di dati CuPy a monte per non calcolarlo nel test
    a = torchic.rand(shape, requires_grad=True)
    b = torchic.rand(shape, requires_grad=True)

    

    print("\n--- BENCHMARK ---")
    ITERATIONS = 10000
    
    # Test Costruttore Standard
    start_time = time.perf_counter_ns()
    for _ in range(ITERATIONS):
        t = a.matmul(b)
    end_time = time.perf_counter_ns()

    elapsed_time = cpx.profiler.benchmark(Tensor.matmul, (a, b), n_repeat=ITERATIONS, name=".matmul")
    elapsed_time = elapsed_time.gpu_times.mean()
    elapsed_time *= 1000    # converto in ms

    total_time_normal_ms = (end_time - start_time) / 1_000_000
    avg_time_normal_ns = (end_time - start_time) / ITERATIONS
    
    print(f"[Standard] Tempo totale per {ITERATIONS:,} .matmul: {total_time_normal_ms:.2f} ms")
    print(f"[Standard] Tempo medio CPU per singola matmul: {avg_time_normal_ns:.2f} ns")
    print(f"[Standard] Tempo medio GPU per singola matmul: {elapsed_time:.2f} ns")





def benchmark_matmul_overhead():
    shape = (1000, 1000)
    # Assumiamo che a e b siano istanze della tua classe Tensor
    a = torchic.rand(shape, requires_grad=True)
    b = torchic.rand(shape, requires_grad=True)

    ITERATIONS = 500
    
    # 1. MISURA TEMPO GPU REALE (usando il profiler nativo di CuPy che usa i CUDA Events)
    # n_warmup è fondamentale per svegliare i kernel CUDA prima di misurare
    gpu_perf = cpx.profiler.benchmark(a.matmul, (b,), n_warmup=10, n_repeat=ITERATIONS)
    avg_gpu_s = gpu_perf.gpu_times.mean() # Tempo in secondi
    avg_gpu_ns = avg_gpu_s * 1_000_000_000

    # 2. MISURA TEMPO TOTALE (CPU + GPU Sincronizzata)
    # Sincronizziamo prima di partire
    cp.cuda.Device().synchronize()
    
    start_time = time.perf_counter_ns()
    for _ in range(ITERATIONS):
        res = a.matmul(b)
        # Sincronizziamo AD OGNI GIRO. Costringiamo la CPU ad aspettare la GPU.
        # In questo modo: Tempo_Totale = Overhead_CPU + Tempo_GPU
        cp.cuda.Device().synchronize()
    end_time = time.perf_counter_ns()
    
    avg_total_ns = (end_time - start_time) / ITERATIONS
    
    # 3. CALCOLO OVERHEAD CPU PER SOTTRAZIONE
    # Overhead_CPU = Tempo_Totale - Tempo_GPU
    avg_cpu_overhead_ns = avg_total_ns - avg_gpu_ns

    print("--- RISULTATI BENCHMARK (Valori Medi) ---")
    print(f"Tempo totale (CPU + GPU): {avg_total_ns:,.2f} ns  ({avg_total_ns/1_000_000:.3f} ms)")
    print(f"Tempo puro GPU:          {avg_gpu_ns:,.2f} ns  ({avg_gpu_ns/1_000_000:.3f} ms)")
    print(f"Overhead CPU (Python):   {avg_cpu_overhead_ns:,.2f} ns  ({avg_cpu_overhead_ns/1_000_000:.3f} ms)")
    
    percent_overhead = (avg_cpu_overhead_ns / avg_total_ns) * 100
    print(f"\nLa CPU impiega il {percent_overhead:.1f}% del tempo totale solo per gestire l'operazione in Python.")

if __name__ == "__main__":
    benchmark_tensor()
    #benchmark_matmul()
    #benchmark_matmul_overhead()