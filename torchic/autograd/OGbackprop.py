import numpy as np
from typing import TYPE_CHECKING, Union, Optional
if TYPE_CHECKING:
    from torchic.cuda_tensor import Tensor

from torchic.settings import CUPY_AVAILABLE, CONFIG

USING_CUDA = False #True if CONFIG["default_device"] == "cuda" else False
if CUPY_AVAILABLE:
    import cupy as cp

ArrayType = Union[np.ndarray, Optional['cp.ndarray']]

# Questo approccio (2.0) permette di:
# - separare il codice;
# - evitare di "compilare" la backward function ogni volta che si chiama una
#   funzione (cosa che cambia le performance in inference)
# - selezionare cosa salvare per la backward e cosa no anziche' avere Python
#   che salva tutto (args inclusi)
# - creare un vero grafo;
#
# introduzione di _version in modo da evitare casini? Si potrebbe introdurre
# facendo si che .data abbia un setter. Ogni volta che viene triggerato,
# aumenta la version di uno.

# Ecco perché del self non ha funzionato: questo processo
# elimina solo la reference 'self' all'oggetto in memoria,
# non elimina l'oggetto stesso:
# https://stackoverflow.com/questions/293431/python-object-deleting-itself 

# L'approccio 3.0 permette il rilascio istantaneo delle grad_fn, 
# i print di errore ora sono perfettamente leggibili ed interpretabili,

# NOTE: se la backward si interrompe improvvisamente senza dare errori,
# prob. la lifo e' piena e il programma si ferma in attesa

#https://www.geeksforgeeks.org/stack-in-python/
# versione thread safe
from queue import LifoQueue


# macini calcoli perche' sai che, a prescindere dall'ordine,
# cio' che sta in cima allo stack e' pronto per essere calcolato.
# Prova a fare cosi': pusha solo cio' che puo' subito partire
# per il calcolo, altrimenti avvia i vari thread e
# aspettali tutti prima di aggiungere il nodo di accumulazione
from collections import deque
# STATS CPU
# 2.38 min con liste (4.7 GB)
# 2.44 min con deque (4.7 GB)
# 2.54 min con LifoQueue (4.6 GB) (2.47 min)
#  
# STATS GPU
# 3.53 min con liste (4.9 GB)
# 3.65 min con deque (5.0 GB)
# 3.60 min con LifoQueue (5.0 GB)
#
# i tre approcci sono perfettamente identici su transformer GPU
# sia per tempo impiegato che per memoria occupata

# APPROVATO! 
# Rifinisci un attimo la logica di backprop (ad
# alleggerire il tensor,
# rivedere la sua logica in .backward(),
# capire se la logica della funzione qua sotto di soddisfa o noti
# cazzate/possibili migliorie, ecc.).
# Infine, test con il transformer per vedere che e' cambiato

# pytorch parallelizza a livello kernel, non con threads



# di base avvierei tre streams: la principale (default, usata di base
# per interagire con la GPU) e altre due da usare all'occorrenza.
# Vanno avviate quando si avvia il programma. Quando riconosci la
# possibilita' di parallelizzare, semplicemente dai quel compito ad
# una stream diversa dalla main. GPT parla di eventi piusttosto
# che di sync espliciti. Credo sia una sorta di condizione che
# fa partire qualcosa (in questo caso mette in pausa lo stream)


stack = LifoQueue()


def backpropagation(root: 'Tensor', grad: ArrayType):
    '''
    Logica di base
    Ogni tensore puo' avere uno o due parenti, che occupano
    sempre le prime posizioni. Ogni BackwardFunction restituisce
    sempre due gradienti, che sono restituiti nello stesso ordine
    in cui i parent sono posizionati nel ctx. Quando manca un
    parent o uno dei due non richiede il gradiente, non viene
    calcolato e viene restituito un None placeholder.
    Quando questa funzione riceve un gradiente None, lo ignora,
    altrimenti, accoppia il gradiente con il parent e li mette
    in cima allo stack.
    Nello stack ci entrano solo i tensori che sono pronti per
    la computazione.
    '''
    # 173 it/min, 6.2
    # idem
    # stack LIFO (Last-In, First-Out)
    #stack = LifoQueue(maxsize=root.grad_fn.id//2)
    #stack = deque(maxlen=root.grad_fn.id//2)
    stack.put((root, grad))

    while not stack.empty():
    #while stack:
        tensor, grad = stack.get()
        #tensor, grad = stack.pop()

        if tensor.is_leaf:
            # aggiorno il gradiente
            if tensor._grad is not None:
                tensor._grad += grad
            else:
                tensor._grad = grad
        else:
            if tensor.grad_fn.ref_count > 1:
                # memorizzo temporaneamente il gradiente e
                # aspetto di averli tutti per continuare
                if tensor.grad_fn.temp_grad is not None:
                    tensor.grad_fn.temp_grad += grad
                else:
                    tensor.grad_fn.temp_grad = grad
                tensor.grad_fn.ref_count -= 1

            # se non devo aspettare la computazione di altri rami
            else:
                if tensor.grad_fn.temp_grad is not None:
                    grad += tensor.grad_fn.temp_grad
                    tensor.grad_fn.temp_grad = None

                # continuo la backprop
                grad_x, grad_y = tensor.grad_fn(grad)

                # se x ha richiesto il gradiente
                if grad_x is not None:
                    if grad_x.dtype != tensor.dtype:
                        grad_x = grad_x.astype(tensor.dtype, copy=False)
                    stack.put((tensor.grad_fn.ctx[0], grad_x))

                if grad_y is not None:
                    if grad_y.dtype != tensor.dtype:
                        grad_y = grad_y.astype(tensor.dtype, copy=False)
                    stack.put((tensor.grad_fn.ctx[1], grad_y))

                tensor.grad_fn = None