import cupy as cp
import time

# Pool di stream limitato
class StreamPool:
    def __init__(self, max_streams):
        # Creiamo max_streams stream non-bloccanti
        self.available_streams = [cp.cuda.Stream(non_blocking=True) for _ in range(max_streams)]
        self.streams_in_use = set()

    def get_stream(self):
        if self.available_streams:
            stream = self.available_streams.pop()
            self.streams_in_use.add(stream)
            return stream
        else:
            raise RuntimeError("Nessuna stream disponibile!")

    def release_stream(self, stream):
        # Simula un rilascio della stream con attesa e sincronizzazione
        print("Rilasciando stream...")
        # Riproviamo a sincronizzare qui, replicando il potenziale deadlock
        #stream.synchronize()
        self.streams_in_use.remove(stream)
        self.available_streams.append(stream)
        print("Stream rilasciata.")

# Callback da lanciare quando una stream ha terminato
def process_and_release(merda):
    stream, pool = merda
    print("Processamento completato, rilasciando la stream.")
    pool.release_stream(stream)

# Esempio principale
pool = StreamPool(max_streams=2)  # Pool con 2 stream

# Funzione di esempio che simula il calcolo
def run_computation_on_stream(tensor, pool):
    # Ottieni una stream dal pool
    stream = pool.get_stream()
    with stream:
        # Operazione GPU
        result = cp.matmul(tensor, tensor)
        # Callback per il rilascio della stream alla fine
        stream.launch_host_func(process_and_release, (stream, pool))

# Test con diverse stream
tensor = cp.random.rand(10000, 10000)

# Eseguo più operazioni che sfruttano il pool di stream
try:
    for _ in range(5):
        run_computation_on_stream(tensor, pool)
        time.sleep(0.1)  # Attesa per simulare calcolo e debugging
except RuntimeError as e:
    print(e)
