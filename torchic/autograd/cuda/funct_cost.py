import cupy as cp
import cupyx as cpx
from torchic.settings import CONFIG

# potrei usare un approccio analitico, ma non so quanto sarebbe preciso.
# In questo modo, misuro un tempo piu' fedele possibile alla realta'
# perche' misurato sull'hardware, nel suo stato attuale e con l'implementazione
# della libreria (che sicuramente e' piu' ottimizzata rispetto
# alle implementazioni naive)

# cache
operations_time = {}


def back_funct_time_complexity(grad_fn, grad, n_repeat: int=10) -> float:
    '''
    Returns average gpu time in ms on n_repeat.
    '''
    # spero che tutto questo avvenga con la stream che ha avviato la funzione
    # di valutazione di ramificarsi e che non cambi costantemente streams...
    operation_id = f"{grad_fn.__class__.__name__}_{grad.shape}"
    if operation_id in operations_time:
        return operations_time[operation_id]
    
    #print(grad_fn)
    elapsed_time = cpx.profiler.benchmark(grad_fn, (grad, ), n_repeat=n_repeat, name=operation_id)
    #cpx.time.repeat()
    elapsed_time = elapsed_time.gpu_times.mean()
    elapsed_time *= 1000    # converto in ms

    operations_time[operation_id] = elapsed_time
    return elapsed_time


def branch_time_estimator(tensor) -> float:
    '''
    Returns the average gpu time (in ms) used to calculate the backwards
    from given tensor onward.
    '''
    if tensor.is_leaf:
        # non ha grad_fn, deve solo sommare il grad
        return 0

    assert tensor.device == "cuda", f"Time estimation is only available on CUDA. Given tensor is on {tensor.device}"
    
    # non avendo il gradiente, lo randomizzo
    grad = cp.random.rand(*tensor.shape(), dtype=tensor.dtype)
    cost = back_funct_time_complexity(tensor.grad_fn, grad)
    
    if tensor.grad_fn.ref_count > 1:
        # stoppo
        return cost

    parent_x, parent_y = tensor.grad_fn.parents()

    if parent_y is not None:
        # allora, assumendo che potro' parallelizzare (perche' e' il
        # caso "peggiore" per la leggerezza del ramo), devo capire quale dei
        # due rami e' il piu' piccolo perche' rimarra' su questa stream

        cost_x = branch_time_estimator(parent_x)
        cost_y = branch_time_estimator(parent_y)

        # sommo il costo gpu del ramo piu' piccolo
        if cost_x > cost_y:
            cost += cost_y
        else:
            cost += cost_x
    else:
        cost += branch_time_estimator(parent_x)
    return cost


def worth_parallelizing(parent_x, parent_y) -> bool|int:
    '''
        min_ms: minimum amount of milliseconds of gpu computation required
        in order to candidate a branch for asynchronous computation with
        streams. NOTE: a value smaller than 0.5 ms creates instabilities
        (loss spikes, cramming of streams) and errors (loss=nan), probably
        because the scheduler doesn't have enough room to work.
    '''
    # NOTE: attualmente i tuoi rami non superano manco i 5 MILLISECONDI
    # di tempo gpu! Ecco perche' usare le streams non conviene ora!

    # se is_leaf=False => requires_grad=True
    # se is_leaf=True => requires_grad=True or False
    #    is_leaf=False <==> grad_fn=BackwardFunction

    # non devono essere leaf o altrimenti sarebbe solo
    # un cambio di stream per il ramo maggiore
    # A prescindere che chiedano il grad, non converrebbe
    if parent_x.is_leaf or parent_y.is_leaf:
        return False

    # se uno dei due deve accumulare, non ha senso perche'
    # una stream deve subito sincronizzarsi per mettersi in
    # attesa dell'altra
    if parent_x.grad_fn.ref_count > 1 or parent_y.grad_fn.ref_count > 1:
        return False
    
    # condizioni soddisfatte. Verifico quale dei due rami e' piu'
    # dispendioso in termini di tempo impiegato
    cost_x = branch_time_estimator(parent_x)
    cost_y = branch_time_estimator(parent_y)

    # parallelizzo il piu'
    # grande e mantengo il piu' piccolo sul ramo corrente per
    # cercare di bilanciare il carico. Parallelizzo solo
    # se ne vale la pena, quindi solo se i calcoli chiedono
    # piu' di 5 ms
    min_ms = CONFIG['cuda']["time_threshold_ms"]
    if cost_x > cost_y:
        return -1 if cost_x > min_ms else False
    else:
        return 1 if cost_y > min_ms else False
    
# posso pensare ad un sistema che, al secondo backward pass, guarda i dati
# del grafo in cache e valuta se non runnare un ramo perche' il prossimo
# e' piu' worth