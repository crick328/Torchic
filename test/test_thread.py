from queue import LifoQueue
from collections import deque
import cupy as cp
import time


# asincrono = lancio i calcoli e rido' subito il controllo
# al main thread senza aspettare la loro fine. All'apparenza
# si comportano come thread, ma in realta' schedulano tutte
# le operazioni al loro interno. Sono vere e proprie sequenze
# di istruzioni che vengono eseguite a parte, ma all'interno
# della stream viene rispettato l'ordine delle operazioni.
# La fregatura e' che ognuna di queste operazioni (cicli e condizioni
# incluse) si comportano come threads!
# CuPy sembra usare degli array vuoti "placeholder" se si tenta
# di accedere ad un risultato che ancora non e' pronto. Questo
# spiega perche' la rete riusciva a trainarsi senza imparare o
# dare errore di gradienti nulli.
# 
# Chiamare stream.synchronize() distrugge TUTTO. Il codice si
# comporta in modo sincrono e fare lo stream si rivela inutile
# perche' il codice va subito a chiamare la funzione di sync.
# Non c'e' condizione che salva perche' il codice e' come se
# fosse gia' stato eseguito tutto, solo che ha valori placeholders
# anziche' quelli corretti. Se una lista o una LifoQueue viene
# popolata solo dopo un'intero ciclo precedente, apparira' comunque
# subito con il numero finale di elementi.
# 
# devo anche accertarmi che lifoqueue e co non copino i tensori su cpu
# per effettuare il put/append
# 
# .

import numpy as np

ee = np.array(5)
print(ee==5)


class Dummy:
    def __init__(self) -> None:
        self.array = None



a = cp.random.rand(10000, 10000)
b = cp.random.rand(10000, 10000)
# Creazione della stream
stream = cp.cuda.Stream(non_blocking=True)
stack = deque()
out = deque()
merda = deque()
stack.append((a, b))


def merda(stream):
    print(stream)
    print("merda!")

d = {None: 2}
print(d)

device = cp.cuda.Device()
print(device.attributes)
print(device.compute_capability)
print(device.mem_info)

used_mem = cp.get_default_memory_pool().used_bytes()
total_mem = device.mem_info[1]
mem_usage_ratio = used_mem / total_mem
print(mem_usage_ratio*100)
print((device.mem_info[1] - device.mem_info[0])/8/1024/1024/1024)
print(used_mem/8/1024/1024/1024)
print(cp.cuda.runtime.getDeviceProperties(0))

# Misurazione asincrona
start = time.time()
with stream:
    while stack:
        print("ITER")
        a, b = stack.pop()
        c = cp.dot(a, b)
        out.append(c)
    
    # QUESTO sincronizza la stream a
    # prescindere da LifoQueue. Senza
    # pop()/get(), il while itera
    # all'infinito e blocca sia lo
    # stream sia il programma.
    # Provare a printare un tensore nel
    # secondo while fa sincronizzare tutto,
    # ma cio' non accade se lo metti in un'altra
    # lista e lo stampi da fuori (stampa tutto zero)

    # gli eventi vanno solo tra stream diverse
    event = cp.cuda.Event()
    stream.record(event)
    #event.wait()

    print("adsf")
    #event.synchronize()
    print("SODOMAYOOO!!!")

    # https://docs.cupy.dev/en/v13.3.0/reference/generated/cupy.cuda.Stream.html#cupy.cuda.Stream.launch_host_func
    stream.launch_host_func(merda, stream)

    print("AA")
print("BBB")
#tmp = out.get()
#print(merda.pop())
##print(tmp)
#out.put(tmp)

print("CPU sta facendo altro lavoro...")
time.sleep(1)  # Simulazione di altro lavoro CPU
print("Tempo CPU:", time.time() - start)

# Sincronizzazione per verificare il tempo GPU
stream.synchronize()
print("Tempo totale (CPU+GPU):", time.time() - start)
#print(out.get())
