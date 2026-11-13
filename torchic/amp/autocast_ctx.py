# Automatic Mixed Precision (AMP)
# autocast to float16 with cupy in order to use Tensor Cores

# NOTE: sembra una funzione da usare con parsimonia. I Tensor Cores non si attivano
# sempre
# NOTE: un forward in fp16 richiede comunque l'uso del loss scaling, anche se la
# backward e' tutta fp32. Il loss scaling, cosi' come la mixed precision, richiedono
# controllo sull'optimizer
# Llama 3.1 e' stato trainato con float32 per tutta la backward...

from torchic.settings import CUPY_AVAILABLE, get_default_precision, get_default_device, is_mixed_precision_enabled, set_mixed_precision_enabled
from torchic.amp.wrappers import wrapper_single_param, wrapper_double_param, wrapper_act_funct, generic_wrapper
from torchic.amp.amp_stability import blacklist, whitelist, to_lower_precision
import sys




from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torchic.cuda_tensor import Tensor


'''
Come funziona Autocast in PyTorch (in teoria)
E' un meccanismo che casta gli *input* delle funzioni safe in float16.
Quando una funzione safe e' incontrata ed autocast e' attivo, l'engine
chiama direttamente *la stessa funzione in versione autocast*, che potrebbe
tranquillamente essere uguale identica alla versione normale, ma con qualche if
aggiuntivo per castare gli specifici args papabili.
Agiscono cosi' (anziche' come vuoi fare tu, cioe' intercettare la chiamata e wrappare)
perche' il loro approccio *non richiede iterazioni o verifiche di condizioni*:
visto che ogni funzione contiene i suoi if, sai gia' quali args controllare. Tu, invece
dovresti iterare ogni volta e poi castare. Questo puo' avere costi esorbitanti.

I parametri della rete rimangono sempre in float32, infatti anche il gradiente viene riportato
in float32 prima dell'aggiornamento.
Questo implica che c'e' un altro aspetto a cui prestare attenzione: quando posso castare
direttamente il tensore e quando no? Se il tensore richiede gradiente, deve rimanere in
memoria un master in float32 e il casting deve avvenire su una copia, altrimenti
ouoi castare direttamente

Consiglio: non replicare ciecamente pytorch, non e' questo l'obiettivo di queste modifiche
o di questa libreria. Anziche' rendere ogni operazione dinamica, perche' non cerchi di fare il
casting a piu' alto livello possibile? Ad esempio, perche' castare matmul, add e ReLU quando puoi farlo
direttamente su tutto il layer? Il codice si semplifica e le performance ringraziano
'''

'''
Come funziona il MIO autocast
Qualsiasi funzione puo' essere decorata con @autocast_operation. Le funzioni
con questo decoratore saranno analizzate all'avvio della libreria con
"is_safe_in_float16" e messe in white/gray/blacklist di conseguenza.
Non succede nient'altro, quindi nessun overhead senza AMP.

Quando si entra nel context manager di autocast, tutte le funzioni vengono
prese e temporaneamente sostituite con una loro versione gia' wrappata su
misura. In questo modo l'overhead e' minimo. A seconda della classificazione
della funzione, il wrap casta gli input a float16 o 32.
I parametri originiali dei modelli non vengono toccati e il gradiente viene
accumulato in float32 (perche' ricevono il gradiente tramite toCopyBackward
che glielo casta in fp32 prima di passarglielo) ed e' una cosa voluta.
Se conviene, la copia in float16 viene inserita in cache in modo da non dover
ricastare lo stesso identico tensore ad ogni forward (amp_cache)

E' stato difficile da impostare bene... Questa e' la soluzione migliore
che ho trovato per non avere overhead ad AMP spento, per non dover decorare
tutte le funzioni e per mantenere la sicurezza (niente eval())
'''

# NOTE: ho una falla di design: questo decoratore ha lo scopo di castare a 
# float16 tutti gli input delle funzioni in whitelist e assicurarsi che tutte
# le altre ricevano float32, ma allo stesso tempo NON voglio spammare questo
# decoratore perche' introduce overhead.
# Se un'operazione critica non ha il decoratore, potrebbe ricevere input in
# float16 ed esplodere. Che fare?






import operator
import inspect

# e' equivalente al ciclo di getattr, ma in C
#operator.attrgetter(parent_path)(modulo)

# per standard, le classi context manager vanno scritte minuscole
class autocast:
    '''
    EEEE
    '''
    def __init__(self):
        self.original_functions = {}

    def __enter__(self):
        if not CUPY_AVAILABLE:
            raise RuntimeError(
                "Unable to use autocast without CUDA because fp16"
                "operations on CPU are not convenient in Numpy"
            )

        if get_default_device() != "cuda":
            raise RuntimeError(
                "You have to use device='cuda' in order to use AMP"
            )

        # non serve con la nuova logica... Per ora e' solo un flag di
        # servizio... Ma allora il setter dovrebbe essere "nascosto"
        set_mixed_precision_enabled(True)
        #print(f"is_mixed_precision_enabled: {is_mixed_precision_enabled()}")
        #print(f"get_default_precision: {get_default_precision()}")

        # qui faccio partire il wrap delle funzioni. Senza di questo non
        # va nulla a prescindere dalle classificazioni fatte dalle altre parti.
        self._set_autocast_wrappers(blacklist, -1)
        # per ora in whitelist ho anche funzioni con score=0 (incerto)
        self._set_autocast_wrappers(whitelist, 1)
        self._set_autocast_wrappers(to_lower_precision, 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_mixed_precision_enabled(False)

        # ripristino i metodi originali
        for module in self.original_functions.keys():
            for name, func in self.original_functions[module].items():
                setattr(module, name, func)

    # mi sa che questo approccio genera troppo CPU overhead...
    # Dovrei evitare di modificare le funzioni originali e "semplicemente"
    # reinderizzare la chiamata (anche semplicemente ad una copia della
    # funzione ma wrappata) che pero' persiste in memoria
    def _set_autocast_wrappers(self, category: set, safeness: int):
        '''
        La safeness viene passata direttamente ai caster dei singoli wrappers
        '''
        for module_path, funct_qualname in category:
            # prendo il module
            module = sys.modules[module_path] 

            if "." in funct_qualname:
                # e' un metodo di una classe, quindi devo prenderne il nome
                class_name, funct_name = funct_qualname.rsplit(".", maxsplit=1)
            
                # navigo il module per arrivare alla classe
                cls = operator.attrgetter(class_name)(module)
                # NOTE: per ora non ho un nome di variabile che
                # rende l'idea che "module" puo' essere una classe
                module = cls
            else:
                # e' una funzione di un modulo, non appartiene a nessun oggetto

                # NOTE: anche qua da capire
                funct_name = funct_qualname

            funct = getattr(module, funct_name)

            # salvo la funzione originale in modo da poterla
            # ripristinare una volta uscito dal context
            if module not in self.original_functions.keys():
                self.original_functions[module] = {}
            self.original_functions[module][funct_name] = funct

            # se una funzione e' in graylist non la wrappo nemmeno

            if funct_name.endswith("__call__"):
                # e' la call di una classe, quindi ActivationFunction o
                # al massimo di un Linear
                # NO! Puo' essere anche una activation function!
                wrapper = wrapper_act_funct(funct, safeness)
            else:
                num_params = len(inspect.signature(funct).parameters)

                match num_params:
                    case 1:
                        wrapper = wrapper_single_param(funct, safeness)
                    case 2:
                        wrapper = wrapper_double_param(funct, safeness)
                    case _:
                        wrapper = generic_wrapper(funct, safeness)

            setattr(module, funct_name, wrapper)
            # da qui hai il modulo e la str dell'attributo per fare
            # getattr, fare il backup, wrappi e sovrascrivi. Metti
            # questi dati in una cache della classe magari, dipende da
            # quanto pesa
            #exit()

    def _reset_default_ops(self):
        pass



'''
def create_wrapper(funct):
    def casted_function(arg1, arg2=None, *args, **kwargs):
        ''
        Wrapper for `funct` (`Tensor`'s operation) that casts the args tensor(s)
        to mixed precision.

        Args:
            arg1 (Tensor): first arg of the function. It could be the first
                operand, the `self` arg of a class or don't care
            arg2 (Tensor): second arg of the function. It could be the second
                operand, the input of an `ActivationFunction`.__call__()
                or don't care

        It converts to float16 the tensors that:
        - are on a cuda device (otherwise there's no point because you can't
            use CUDA optimizations)
        - have a precision higher than fp16
        If the tensor does not require grad, an inplace conversion is made, otherwise it's first cloned
        and then converted. This is done in order to avoid any variation to the master weights (original
        parameters of the network in fp32) because their update have to be done in fp32.
        ''

        # NOTE: questo print parte anche fuori dal contesto!
        #print(f"Attivo wrapper su {funct.__qualname__}!")

        # NOTE: se l'operazione non e' safe, devi assicurarti che i tensori siano
        # in float32 (e il casting lo fa sempre questo wrapper)
        #if is_mixed_precision_enabled():
        target_dtype = get_mixed_precision_dtype()
        is_funct_in_whitelist = funct.__qualname__ in whitelist

        # opero solo sulle istanze di Tensor che sono su cuda
        # NOTE: non posso importare Tensor perche' andrei in import circolare,
        # quindi devo verificare la classe in sto modo
        if arg1.__class__.__name__ == "Tensor" and arg1.device == "cuda":
            _manage_arg_dtype(arg1, is_funct_in_whitelist, target_dtype)

        # (se c'e') se il secondo arg e' un Tensor ed e' su cuda
        #if getattr(arg2, "device", None) == "cuda":
        # NOTE: questa casistica gestisce anche arg2=None perche' e' di
        # classe NoneType
        if arg2.__class__.__name__ == "Tensor" and arg2.device== "cuda":
            _manage_arg_dtype(arg2, is_funct_in_whitelist, target_dtype)

        if arg2 is not None:
            return funct(arg1, arg2, *args, **kwargs)
        return funct(arg1, *args, **kwargs)
    return casted_function


def _manage_arg_dtype(arg: 'Tensor', is_blacklisted: bool, target_dtype: dtype) -> None:
    if not is_blacklisted:
        # posso lavorare in fp16

        # se il dtype e' piu' alto di fp16
        if arg.dtype > target_dtype:
            print(f"casto arg1! {arg.dtype} -> {target_dtype}")
            if arg.requires_grad:
                # mantengo i parametri originali in fp32 e faccio il calcolo
                # con una copia in fp16
                arg = arg.clone().to(target_dtype)
            else:
                # converto direttamente
                arg.to(target_dtype)
    else:
        # funzione in blacklist: devo accertarmi che sia in fp32

        arg.to(float32) if arg.dtype < float32 else arg
'''