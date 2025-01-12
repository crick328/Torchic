import cupy as cp
import time


# L'evento risulta correttamente non finito, ma
# la stream1 se ne frega e tira dritto

# Creazione di una stream asincrona
stream = cp.cuda.Stream(non_blocking=True)
stream1 = cp.cuda.Stream(non_blocking=True)


start = time.time()
with stream:
    count = 0

    while count < 3:
        tensor = cp.random.rand(1000,1000)
        c = cp.matmul(tensor, tensor)
        count += 1

    event = cp.cuda.Event()
    stream.record(event)
    print(event.done)

print(cp.max())

with stream1:
    event.synchronize()
    #stream1.wait_event(event)
    print(f"HA! {event.done}")

print("Tempo CPU:", time.time() - start)

# Sincronizzazione per verificare il tempo GPU
stream.synchronize()
print("Tempo totale (CPU+GPU):", time.time() - start)