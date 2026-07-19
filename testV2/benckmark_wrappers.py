import sys
import timeit
import inspect
from preset import torchic



# 1. FUNZIONE ORIGINALE DA INTERCETTARE
def somma_fun(a, b):
    return a + b

# ==========================================
# APPROCCIO 1: Wrapper Classico (*args, **kwargs)
# ==========================================
def wrapper_classico(func):
    def wrapper(*args, **kwargs):
        # Intercettazione (operazione minima per non falsare il benchmark)
        _ = func.__name__ 
        return func(*args, **kwargs)
    return wrapper

somma_classica = wrapper_classico(somma_fun)

# ==========================================
# APPROCCIO 2: Wrapper Dinamico con exec()
# ==========================================
def crea_wrapper_exec(func):
    sig = inspect.signature(func)
    params = ", ".join(sig.parameters.keys())
    
    codice = f"""
def wrapper_veloce({params}):
    _ = originale.__name__
    return originale({params})
"""
    spazio_nomi = {"originale": func}
    exec(codice, spazio_nomi)
    return spazio_nomi["wrapper_veloce"]

somma_exec = crea_wrapper_exec(somma_fun)

# ==========================================
# APPROCCIO 3: Intercettazione tramite Profiler (sys.settrace)
# ==========================================
def tracciatore(frame, event, arg):
    if event == 'call':
        _ = frame.f_code.co_name
    return tracciatore

def esegui_con_profiler():
    sys.settrace(tracciatore)
    somma_fun(5, 10)
    sys.settrace(None)

# ==========================================
# ESECUZIONE DEL BENCHMARK
# ==========================================
NUMERO_RIPETIZIONI = 1_000_000

print("ESECUZIONE DEL BENCHMARK")
NUMERO_RIPETIZIONI = 1_000_000

#a = torchic.Tensor()

print(f"Avvio del benchmark ({NUMERO_RIPETIZIONI:,} ripetizioni)... \n")

# Baseline (Funzione liscia, senza intercettazioni)
tempo_baseline = timeit.timeit(lambda: somma_fun(5, 10), number=NUMERO_RIPETIZIONI)

# Test Wrapper Classico
tempo_classico = timeit.timeit(lambda: somma_classica(5, 10), number=NUMERO_RIPETIZIONI)

# Test Wrapper Exec
tempo_exec = timeit.timeit(lambda: somma_exec(5, 10), number=NUMERO_RIPETIZIONI)

# Test Profiler (sys.settrace)
tempo_profiler = timeit.timeit(esegui_con_profiler, number=NUMERO_RIPETIZIONI)

# Baseline (Funzione liscia, senza intercettazioni)
tempo_cast_singoli = timeit.timeit(lambda: somma_fun(5, 10), number=NUMERO_RIPETIZIONI)

# ==========================================
# RISULTATI
# ==========================================
print(f"{'Approccio':<30} | {'Tempo Totale (s)':<18} | {'Rallentamento vs Baseline':<25}")
print("-" * 80)
print(f"{'1. Funzione Originale (Baseline)':<30} | {tempo_baseline:<18.4f} | {'1.0x (Riferimento)':<25}")
print(f"{'2. Wrapper Dinamico (exec)':<30} | {tempo_exec:<18.4f} | {tempo_exec/tempo_baseline:<23.2f}x")
print(f"{'3. Wrapper Classico (*args)':<30} | {tempo_classico:<18.4f} | {tempo_classico/tempo_baseline:<23.2f}x")
print(f"{'4. Profiler (sys.settrace)':<30} | {tempo_profiler:<18.4f} | {tempo_profiler/tempo_baseline:<23.2f}x")