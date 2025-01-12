from typing import TYPE_CHECKING, Union, Optional
from torchic.settings import CUPY_AVAILABLE
from collections import deque
import numpy as np

if TYPE_CHECKING:
    from torchic.cuda_tensor import Tensor

if CUPY_AVAILABLE:
    import cupy as cp
    from torchic.autograd.cuda.stream_pool import StreamPool
    from torchic.autograd.cuda.funct_cost import worth_parallelizing

    pool = StreamPool()
    graph_cache = {}
    #collector_stream = cp.cuda.Stream(non_blocking=False)

ArrayType = Union[np.ndarray, Optional['cp.ndarray']]


# null stream: usata di default per fare calcoli su gpu.
# E' la stream principale ed e' sincrona rispetto alla cpu.
# Conviene fare le operazioni delicate (tensori condivisi) sulla null stream,
# in quanto non crea problemi di accesso.
# async stream: indipendenti e sbloccano la possibilita'
# di parallelizzare, ma questo dipende dal driver. La parallelizzazione
# tra stream e' possibile solo se ci sono abbastanza risorse disponibili
# sulla GPU, e questo lo valuta il driver.

# METTI piu' commenti e typehint appena la logica si stabilizza.
# Non ti preoccupare se parallelizza poco o non vedi un
# grande guadagno in termini di tempo, quello dipende
# dalle condizioni di parallelizzazione e si possono finetunare
# in qualsiasi momento. Per ora preoccupati del fatto che parallelizzi
# per davvero e che sia solida

# ----

# L'idea di base e' creare un sistema in cui le stream child dipendono
# e riconvergono alla stream che l'ha creata, con la default stream come
# creatrice di tutte le stream
#
# OCCHIO! Il programma ha subito sbarellato quando non riassegnavi sia branch_x
# che branch_y! Questo significa che o sei sicuro che la variabile verra'
# sovrascritta al pass dopo o devi settarla subito a None!!
# 
# Di tanto in tanto noto delle imprecisioni gravi nel codice. Occhio!
# 
# Potrei sommare in forward la size del ramo dalla
# prima diramazione fino al punto corrente in modo da poter valutare meglio
# quanto e' pesante il ramo. Di nuovo, PyTorch non sembra fare cosi', ma e'
# un'idea. Si puo' anche sfruttare del caching (cosa che pytorch fa) per fare questi
# conti una volta sola e poi tirare dritto
# .

# PS: il training serio fallo su kaggle con una batch size grossa!


# testato con null, ok (circa 1 ora e 30, poco piu')
# testato anche senza null, ok
main_stack = deque()


def start_backprop(tensor: 'Tensor', grad: ArrayType):
    # se non ho cuda disponibile o sto trainando su cpu
    if tensor.device == 'cpu':
        using_cuda = False
        stream = None
    else:
        # cosi' considero anche il dynamic
        using_cuda = True
        stream = pool.default_stream

    backpropagation(tensor, grad, stream, using_cuda=using_cuda)

    if using_cuda:
        stream.launch_host_func(pool.sync_all, None)


# NOTE: i frizzini (da confermare se avvengono per sovraccarico esterno e per il
# min_ms=0.5) gli ho notati da quando ho fuso compute_branch() e backpropagation().
# Forse l'if alla fine per il callback non e' gradito?
def backpropagation(tensor, grad, stream=None, event=None, using_cuda=True) -> None:
    if event:
       stream.wait_event(event)
       #event.synchronize()

    # se ho una stream che non e' la default
    if using_cuda and stream is not pool.default_stream:
        substack = deque()
    # se non sto usando stream (CPU) o sto usando la default
    else:
        substack = main_stack
    substack.append((tensor, grad))

    while len(substack) > 0:
        tensor, grad = substack.pop()

        if tensor.is_leaf:
            # aggiorno il suo gradiente
            if tensor._grad is not None:
                tensor._grad += grad
            else:
                tensor._grad = grad
        else:
            # continuo la backprop
            grad_fn = tensor.grad_fn

            # se e' un tensore critico (con piu' dipendenze)
            if grad_fn.ref_count > 1:
                grad_fn.accumulate_grad(grad, stream, using_cuda)

                if using_cuda:
                    with pool.null_stream:
                        grad_fn.ref_count -= 1
                else:
                    grad_fn.ref_count -= 1
            else:
                if grad_fn.grad_buffer:
                    if using_cuda:
                        # se non sono sulla stream di default
                        if stream is not pool.default_stream:
                            # gli passo il tensore e vado avanti
                            main_stack.append((tensor, grad))
                            continue

                        # sincronizzo tutte le streams che hanno lavorato sul tensor
                        # (se sta usando solo la null o CPU, sara' una chiamata vuota)
                        pool.sync_critical_streams(grad_fn.event_register)
                        grad_fn.event_register = None

                    # Sommo il contributo del gradiente accumulato e mi
                    # assicuro che l'operazione finisca prima di utilizzare
                    # il nuovo gradiente (NOTE: step necessario anche usando 
                    # la null SE i rami sono molto piccoli)
                    #with collector_stream:
                    for stream_key in grad_fn.grad_buffer:
                        grad += grad_fn.grad_buffer[stream_key]
                    #collector_stream.synchronize()
                    grad_fn.grad_buffer = None

                # calcolo i gradienti e li associo ai rispettivi tensori
                #branches = grad_fn.next_elems()
                #parent_x, parent_y = next(branches), next(branches, None)
                parent_x, parent_y = grad_fn.parents()
                grad_x, grad_y = grad_fn(grad)

                # if not tensor._retain_graph/CONFIG["retain_graph"]:
                tensor.grad_fn = None

                # se x ha richiesto il gradiente
                if grad_x is not None:
                    # mi assicuro che grad e tensor abbiano lo stesso dtype
                    if grad_x.dtype != parent_x.dtype:
                        grad_x = grad_x.astype(parent_x.dtype, copy=False)

                if grad_y is not None:
                    if grad_y.dtype != parent_y.dtype:
                        grad_y = grad_y.astype(parent_y.dtype, copy=False)

                # se sto usando cuda, ho streams a disposizione e ho due rami
                if using_cuda and parent_y is not None and pool.can_parallelize():
                    # posso parallelizzare; valuto se farlo
                    node_id = f"{grad_fn}_{tensor.shape()}"

                    if node_id in graph_cache:
                        # ho gia' valutato se ne vale la pena, controllo
                        result = graph_cache[node_id]
                    else:
                        # calcolo se vale la pena e salvo il risultato
                        result = worth_parallelizing(parent_x, parent_y)
                        graph_cache[node_id] = result

                    if result:
                        #print("PARALLELIZZO!")
                        # la stream null e' sincronizzata con tutte le
                        # streams, quindi non e' necessario
                        if stream is not pool.null_stream:
                            new_event = cp.cuda.Event()
                            stream.record(new_event)
                        else:
                            new_event=None
                        new_stream = pool.get_stream()

                        if result == -1:
                            # parallelizzo x
                            with new_stream:
                                backpropagation(parent_x, grad_x, new_stream, new_event)
                            # mantengo y sulla stream corrente
                            substack.append((parent_y, grad_y))
                        else:
                            # mantengo x sulla stream corrente
                            substack.append((parent_x, grad_x))
                            # parallelizzo y
                            with new_stream:
                                backpropagation(parent_y, grad_y, new_stream, new_event)
                        continue
                # se non uso la GPU o non vale la pena parallelizzare,
                # appendi allo stack principale
                if grad_x is not None:
                    # allora parent_x.requires_grad is True
                    substack.append((parent_x, grad_x))

                if grad_y is not None:
                    # allora ho un parent_y con requires_grad==True
                    # NOTE: il caso speciale degli embeddings passa per di qui
                    #print(grad_y is None, grad_fn, parent_y.shape())
                    substack.append((parent_y, grad_y))

    if using_cuda:
        if stream is not pool.default_stream:
            stream.launch_host_func(pool.release_stream, stream)
        #else:
            #stream.launch_host_func(pool.sync_all, None)