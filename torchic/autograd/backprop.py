from typing import TYPE_CHECKING, Union, Optional
from torchic.settings import CUPY_AVAILABLE
from collections import deque
import numpy as np

if TYPE_CHECKING:
    from torchic.cuda_tensor import Tensor
    from torchic.autograd.functions import BackwardFunction

if CUPY_AVAILABLE:
    import cupy as cp
    from torchic.autograd.cuda.cuda_pools import stream_ring_pool, event_pool
    from torchic.autograd.cuda.funct_cost import worth_parallelizing
    graph_cache = {}

ArrayType = Union[np.ndarray, Optional['cp.ndarray']]


# null stream: usata di default per fare calcoli su gpu.
# E' la stream principale ed e' sincrona rispetto alle
# altre streams => fa da barriera di sincronizzazione:
# - Regola 1 (Inizio): un'operazione accodata sulla null stream non può iniziare
#   finché tutti gli stream asincroni della GPU non hanno completato le
#   operazioni precedentemente accodate
# - Regola 2 (Fine): Nessuna operazione accodata sugli stream asincroni
#   può iniziare finché l'operazione sulla null stream non è completamente terminata.
# NOTE: le stream create con non_blocking=True ignorano completamente la barriera
# della null stream!
# Problema: non si riesce a parallelizzare con la null_stream tra i piedi
# proprio perche' sincronizza tutto. Da qui la mia scelta di NON usarla come
# default e di usare solo streams con non_blocking=True.
# 
# Esempio. S = worker blocking, N = NULL stream. Io schedulo
# S:   A
# N:       B
# S:           C
# -----------------> t
# CUDA impone implicitamente A → B → C. Quindi, appunto, fa da barriera: nel
# momento in cui schedulo B sulla null, tutto cio' che e' stato assegnato prima
# su tutte le streams deve finire prima che parta B. Dopo che B ha concluso,
# possono partire tutti i calcoli schedulati su qualsiasi streams dopo che
# e' stato schedulato B.
# 
# non-blocking stream: indipendenti e sbloccano la possibilita'
# di parallelizzare, ma questo dipende dal driver. La parallelizzazione
# tra stream e' possibile solo se ci sono abbastanza risorse disponibili
# sulla GPU, e questo lo valuta il driver.
# eventi: sono flag che scattano quando la stream arriva a quel punto. Sono
# utilissimi per sincronizzare in modo preciso le streams in modo che facciano
# i calcoli solo quando sono pronti i dati su cui operare (partenza del child,
# accumulo gradiente con contributi di piu' rami)
# NOTE: fare stream.record(event) o event.record(stream) e' indifferente,
# chiamano la stessa primitiva
# NOTE: registrare nuovamente un evento (sulla stessa stream o su una
# diversa) comporta la CANCELLAZIONE del marcatore precedente
# Diversi tipi di sincronizzazione:
# - stream.synchronize() significa mettere in attesa la CPU fin quando
#   la stream non ha finito tutti i calcoli inviati prima di quel sync
# - event.synchronize() vuol dire mettere in pausa la CPU in quando
#   non scatta quell'evento
# - stream.wait_event(event) mette solamente in attesa la stream. Quando
#   scatta, parte. La CPU NON e' coinvolta
# RULE OF THUMB: le streams hanno effetto SOLO sui cp.ndarray. Tutto
# il resto viene eseguito sempre dalla CPU

# ----

# La null stream fa da default stream ed esegue tutti i calcoli
# (dal forward alle computazioni dell'optimizer) -> non e' necesaria
# sincronizzazione esplicita da fwd e bwd, tra bwd e optim e per
# svuotare la amp_cache (Python rilascia il dict, il memory pool
# di CuPy registra i blocchi di memoria come usabili, ma non gli 
# cancella e i pointers rimangono validi. Io ho gia' chiamato
# null_stream.wait(eventi) e le queue delle stream sono FIFO,
# quindi quella memoria verra' scritta solo quando saranno terminate
# le operazioni prima del wait -> operazioni gia' fatte).
# Bisogna invece attendere le worker_stream, che sono rami paralleli
# asincroni rispetto a CPU e default stream. Vengono usate durante la
# backward (in certe condizioni) e vanno aspettate esplicitamente tramite
# eventi (event_pool.wait_workers()) se non si ricongiungono con la null
# durante la computazione (reunited_with_null=False). Negli altri casi,
# le workers si diramano dalla null e riconvergono ad essa: una volta 
# calcolato il loro contributo al tensore critico (tensor.grad_fn.ref_count > 1),
# ritornano tra le pool disponibili.
# NOTE: il pool di stream e di eventi e' sincrono rispetto alla CPU, non segue
# la GPU. Questo vuol dire che una stream viene reinserita nel pool prima
# che questa abbia effettivamente finito i calcoli accodati. Questa e' una
# scelta di design perche' l'alternativa sarebbe stata reinserire la stream
# a fine computazione con una callback_host_function(), che pero' dovrebbe
# chiede un context switch alla CPU e lancia la callback su un altro thread,
# richiedendo quindi sistemi thread-safe. Insomma, avrei piu' overhead
# per ottenere lo stesso identico risultato. Con questo approcio, invece, 
# la CPU schedula i calcoli e naviga nelle ricorsioni di backpropagation()
# mentre la GPU macina calcoli.
#
# OCCHIO! Il programma ha subito sbarellato quando non riassegnavi sia branch_x
# che branch_y! Questo significa che o sei sicuro che la variabile verra'
# sovrascritta al pass dopo o devi settarla subito a None!!


# ho scelto deque perche' le operazioni di append() e pop() sono
# atomiche (thread-safe)
main_stack = deque()




def start_backprop(tensor: 'Tensor', grad: ArrayType):
    # se non ho cuda disponibile o sto trainando su cpu
    if tensor.device == 'cpu':
        backpropagation(tensor, grad, stream=None)
        return

    if not CUPY_AVAILABLE:
        raise RuntimeError()

    # la stream di default/principale e' la stessa del forward e
    # di tutti gli altri calcoli su GPU, ovvero la null stream
    backpropagation(tensor, grad, stream=stream_ring_pool.null_stream)

    # DISASTRO! Se il sync_all() parte quando c'e' ancora
    # una stream in pool.streams_in_use() (perche' il Garbage
    # Collector su CPU non ha fatto in tempo a pulire), la
    # stream chiama .syncronize() dentro se' stessa (cosa vietata)
    # e CuPy lancia un errore (che non posso ne' vedere ne' gestire)
    #stream.launch_host_func(pool.sync_all, None)

    # mette in attesa la stream principale mentre la CPU
    # puo' avanzare e schedulare il lavoro successivo
    event_pool.wait_workers(stream_ring_pool.null_stream)
    event_pool.release_events()


def backpropagation(tensor, grad, stream=None, start_event=None) -> None:
    # === Setup ===
    # tensor_ready_event del parent (null_stream)
    if start_event:
        # ho una stream che non e' la default (-> sto parallelizzando)

        # aspetto che il parent calcoli il grad di partenza di questa stream
        stream.wait_event(start_event)
        # non e' la main stream, quindi devo aspettarla se
        # non si ricongiunge con essa
        reunited_with_null = False
        # creo uno stack tutto per lei in modo da evitare conflitti
        substack = deque()
    # se non sto usando stream (CPU) o sto usando la default
    else:
        substack = main_stack
    substack.append((tensor, grad))

    grad_fn: BackwardFunction
    
    # ==== Backprop Logic ====
    while len(substack) > 0:
        tensor, grad = substack.pop()

        if tensor.is_leaf:
            # aggiorno il suo gradiente
            if tensor._grad is not None:
                tensor._grad += grad
            else:
                tensor._grad = grad
            continue
        
        # else continuo la backprop
        grad_fn = tensor.grad_fn

        # se e' un tensore critico (con piu' dipendenze)
        if grad_fn.ref_count > 1:
            # metto un evento nel dict grad_fn.partial_grad_events
            # che scattera' quando questa stream avra' calcolato e messo
            # il suo contributo nel dict grad_fn.grad_buffer
            grad_fn.accumulate_grad(grad, stream)

            grad_fn.ref_count -= 1
            if stream and stream is not stream_ring_pool.null_stream:
                reunited_with_null = True
            continue

        if grad_fn.grad_buffer:
            if stream:
                # se non sono sulla stream di default
                if stream is not stream_ring_pool.null_stream:
                    # visto che voglio che prosegua solo lo stream
                    # di default,
                    # gli passo il tensore e finisco gli eventuali calcoli
                    # restanti su questa stream, che poi si estinguera'
                    main_stack.append((tensor, grad))
                    continue

                # sincronizzo tutte le streams che hanno lavorato sul tensor
                # aspettando che scattino tutti gli eventi che notificano
                # l'aggiunta del contributo del grad
                # (se sta usando solo la null o CPU, il dict ha una sola entry)
                for stream_key in grad_fn.partial_grad_events:
                    stream_ring_pool.null_stream.wait_event(grad_fn.partial_grad_events[stream_key])
                grad_fn.partial_grad_events = None

            # a questo punto sono sulla default stream.
            # Sommo il contributo del gradiente accumulato e proseguo
            # con la backprop
            for stream_key in grad_fn.grad_buffer:
                grad += grad_fn.grad_buffer[stream_key]
            grad_fn.grad_buffer = None

        # calcolo i gradienti e li associo ai rispettivi tensori
        parent_x, parent_y = grad_fn.parents()
        grad_x, grad_y = grad_fn(grad)

        # if not tensor._retain_graph/CONFIG["retain_graph"]:
        tensor._grad_fn = None

        # ==== GRADIENTS UPDATE ====
        # se x ha richiesto il gradiente
        # NOTE: questi casting non si verificano mai, i gradienti
        # matchano nativamente i dtype dei parent
        if grad_x is not None:
            # mi assicuro che grad e tensor abbiano lo stesso dtype
            if grad_x.dtype != parent_x.dtype:
                #print(f"Casting su grad_x di {grad_fn.__str__()}, {parent_x.dtype}")
                grad_x = grad_x.astype(parent_x.dtype, copy=False)

        if grad_y is not None:
            if grad_y.dtype != parent_y.dtype:
                #print(f"Casting su grad_y di {grad_fn.__str__()}")
                grad_y = grad_y.astype(parent_y.dtype, copy=False)

        # se sto usando cuda, ho streams a disposizione e ho due rami
        if stream and parent_y is not None and stream_ring_pool.can_parallelize():
            # posso parallelizzare; valuto se farlo
            node_id = f"{grad_fn}_{tensor.shape()}"

            if node_id not in graph_cache:
                # calcolo se vale la pena e salvo il risultato
                is_worth = worth_parallelizing(parent_x, parent_y)
                graph_cache[node_id] = is_worth
            # ho gia' valutato se ne vale la pena, controllo
            is_worth = graph_cache[node_id]

            if is_worth != 0:
                #print(f"PARALLELIZZO! {grad_fn.__str__()}")
                tensor_ready_event = event_pool.get_event()
                # metto una pressure plate sulla strada della stream:
                # Quando i calcoli raggiungeranno questo punto,
                # l'evento scattera'. Uso questo sistema per far
                # partire il child solo quando il parent gli ha
                # computato il suo gradiente di partenza
                stream.record(tensor_ready_event)

                worker = stream_ring_pool.get_stream()

                if is_worth == -1:
                    # parallelizzo x
                    # NOTE: e' una semplice chiamata ricorsiva CPU, SOLO
                    # i calcoli saranno accodati al worker!
                    with worker:
                        backpropagation(parent_x, grad_x, worker, tensor_ready_event)
                    # mantengo y sulla stream corrente
                    substack.append((parent_y, grad_y))
                else:
                    # mantengo x sulla stream corrente
                    substack.append((parent_x, grad_x))
                    # parallelizzo y
                    with worker:
                        backpropagation(parent_y, grad_y, worker, tensor_ready_event)
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

    # potrei sintetizzare chiedendo solo if start_event, ma temo si capisca poco
    if stream and stream is not stream_ring_pool.null_stream:
        # la CPU arrivera' subito a questo punto (e va bene)
        if not reunited_with_null:
            event_pool.register_worker_done_event(stream)
        
        stream_ring_pool.release_stream(stream)