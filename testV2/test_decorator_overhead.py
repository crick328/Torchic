import time

AMP_ENABLED = False

# 1. Decoratore DINAMICO (quello del test precedente)
def amp_dinamico(funct):
    def wrapper(*args, **kwargs):
        if AMP_ENABLED: pass
        return funct(*args, **kwargs)
    return wrapper

# 2. Decoratore OTTIMIZZATO (Firma fissa, evita la creazione di tuple/dict)
def amp_ottimizzato(funct):
    # Assumiamo una firma standard per operazioni Tensor: (self, other, dim, keepdim)
    # Impostiamo i default speculari a quelli delle funzioni reali
    def wrapper(self, other=None, dim=None, keepdim=False):
        if AMP_ENABLED: pass
        return funct(self, other, dim, keepdim)
    return wrapper

# 3. Funzione di controllo da chiamare IN LINEA
def check_amp_inline(tensor):
    if AMP_ENABLED: pass

# --- FUNZIONI DA TESTARE ---

@amp_dinamico
def op_dinamica(self, other=None, dim=None, keepdim=False):
    return self + (other if out is not None else 0)

@amp_ottimizzato
def op_ottimizzata(self, other=None, dim=None, keepdim=False):
    return self + (other if out is not None else 0)

def op_inline(self, other=None, dim=None, keepdim=False):
    check_amp_inline(self) # Chiamata esplicita in linea
    return self + (other if out is not None else 0)

# --- BENCHMARK ---
iterazioni = 10_000_000
out = 0

print("Avvio test di ottimizzazione...\n")

# Test Dinamico
start = time.perf_counter()
for _ in range(iterazioni):
    op_dinamica(5, 10, dim=0)
tempo_dinamico = time.perf_counter() - start
print(f"Decoratore Dinamico (*args): {tempo_dinamico:.4f} secondi")

# Test Ottimizzato (Firma fissa)
start = time.perf_counter()
for _ in range(iterazioni):
    op_ottimizzata(5, 10, dim=0)
tempo_ottimizzato = time.perf_counter() - start
print(f"Decoratore Firma Fissa:     {tempo_ottimizzato:.4f} secondi")

# Test Chiamata In Linea
start = time.perf_counter()
for _ in range(iterazioni):
    op_inline(5, 10, dim=0)
tempo_inline = time.perf_counter() - start
print(f"Chiamata in linea (Metodo):  {tempo_inline:.4f} secondi")