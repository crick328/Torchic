import inspect
import timeit

# --- 1. FUNZIONE DI INTERCETTAZIONE E FUNZIONI TARGET ---
def _elabora_primi_due(arg1, arg2):
    # Simuliamo un'operazione fulminea (es. un controllo o logging)
    pass

def somma_due(a, b):
    return a + b

def elabora_quattro(a, b, c, d):
    return a + b + c + d


# --- 2. WRAPPER CLASSICO (*args, **kwargs) ---
def intercetta_classico(func):
    def w(*args, **kwargs):
        # Per equità, estraiamo i primi due argomenti se presenti posizionalmente
        if len(args) >= 2:
            _elabora_primi_due(args[0], args[1])
        return func(*args, **kwargs)
    return w


# --- 3. WRAPPER OTTIMIZZATI (Firma Fissa) ---
def _wrapper_2_args(func):
    def w(p1, p2):
        _elabora_primi_due(p1, p2)
        return func(p1, p2)
    return w

def _wrapper_4_args(func):
    def w(p1, p2, p3, p4):
        _elabora_primi_due(p1, p2)
        return func(p1, p2, p3, p4)
    return w

# reindirizza solo ai wrapper giusti
def intercetta_ottimizzato(func):
    sig = inspect.signature(func)
    num_params = len(sig.parameters)
    
    if num_params == 2:
        return _wrapper_2_args(func)
    elif num_params == 4:
        return _wrapper_4_args(func)
    else:
        raise ValueError("Funzione fuori standard")


# --- 4. PREPARAZIONE DEI TEST ---
# Generiamo le versioni decorate
somma_classica = intercetta_classico(somma_due)
somma_ottimizzata = intercetta_ottimizzato(somma_due)

quattro_classica = intercetta_classico(elabora_quattro)
quattro_ottimizzata = intercetta_ottimizzato(elabora_quattro)


# --- 5. RUN DEL BENCHMARK ---
if __name__ == "__main__":
    REPETITIONS = 5_000_000
    print(f"Esecuzione di {REPETITIONS:,} cicli per test...\n")

    # --- Test 2 Argomenti ---
    tempo_classico_2 = timeit.timeit("somma_classica(10, 20)", globals=globals(), number=REPETITIONS)
    tempo_ottimizzato_2 = timeit.timeit("somma_ottimizzata(10, 20)", globals=globals(), number=REPETITIONS)
    
    print("=== TEST 2 ARGOMENTI ===")
    print(f"Wrapper Classico (*args):    {tempo_classico_2:.4f} secondi")
    print(f"Wrapper Ottimizzato (Fisso): {tempo_ottimizzato_2:.4f} secondi")
    guadagno_2 = ((tempo_classico_2 - tempo_ottimizzato_2) / tempo_classico_2) * 100
    print(f"Risparmio di tempo:          {guadagno_2:.1f}%\n")

    # --- Test 4 Argomenti ---
    tempo_classica_4 = timeit.timeit("quattro_classica(1, 2, 3, 4)", globals=globals(), number=REPETITIONS)
    tempo_ottimizzato_4 = timeit.timeit("quattro_ottimizzata(1, 2, 3, 4)", globals=globals(), number=REPETITIONS)
    
    print("=== TEST 4 ARGOMENTI ===")
    print(f"Wrapper Classico (*args):    {tempo_classica_4:.4f} secondi")
    print(f"Wrapper Ottimizzato (Fisso): {tempo_ottimizzato_4:.4f} secondi")
    guadagno_4 = ((tempo_classica_4 - tempo_ottimizzato_4) / tempo_classica_4) * 100
    print(f"Risparmio di tempo:          {guadagno_4:.1f}%")