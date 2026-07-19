# Automatic Mixed Precision (AMP)
# autocast to float16 with cupy in order to use Tensor Cores

# NOTE: sembra una funzione da usare con parsimonia. I Tensor Cores non si attivano
# sempre
# NOTE: un forward in fp16 richiede comunque l'uso del loss scaling, anche se la
# backward e' tutta fp32. Il loss scaling, cosi' come la mixed precision, richiedono
# controllo sull'optimizer
# Llama 3.1 e' stato trainato con float32 per tutta la backward...

from torchic.settings import CUPY_AVAILABLE, get_default_device, get_mixed_precision_dtype, is_mixed_precision_enabled, set_mixed_precision_enabled
from torchic.dtypes import float32
from torchic.amp.amp_stability import is_safe_in_float16

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torchic.cuda_tensor import Tensor


'''
Come funziona Autocast in PyTorch
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


seen_functions = set()
# qua potrei aggiungere le funzioni selezionate dall'utente...?
whitelist = set()

# non ho capito se devo scartare solo le operazioni sensibili o le loro intere funzioni

# i decorators sono dei wrapper che avvolgono la funzione presa in input
# e restituiscono una nuova funzione
def autocast_operation(funct):
    """
    Decorator used to mark a function that should be analyzed for
    mixed precision compatibility.

    Functions with @autocast_operation are not guaranteed to be executed in mixed precision.
    Internally, the amp module has a whitelist, a blacklist and an automatic classification system.
    Functions can be either manually added to each of the two lists or they can be evalued
    by the system, which chooses the classification based on the mathematical operations used
    inside the function.

    This piece of code is responsible for casting tensors to fp16 if 'funct' is elegbile,
    but it also has to convert back to fp32 if it isn't.

    Decoratore per applicare il casting agli argomenti delle operazioni.
    Intercetto la chiamata alla funzione e casto i suoi args prima che venga eseguita
    """

    if not is_mixed_precision_enabled():
        return funct

    # il codice esterno da def wrapper() viene eseguito solo all'avvio della
    # libreria! Di conseguenza, il check della flag va messo dentro wrapper

    # il __qualname__ e' il nome della funzione piu' la sua radice (serie di references ".")
    # una funzione con lo stesso nome ma presente in due parti differenti del codice
    # saranno trattate come diverse

    if funct.__qualname__ not in seen_functions:
        print(f"{funct.__qualname__} safe? {is_safe_in_float16(funct)}")
        # al momento mi baso solo sul classificatore automatico che
        # e' l'obiettivo finale, non voglio mica inserirle a mano.
        # Tuttavia, voglio anche ascoltare le richieste dell'utente

        # se e' safe la metto in whitelist
        if is_safe_in_float16(funct):
            whitelist.add(funct.__qualname__)
        seen_functions.add(funct.__qualname__)

    # idea: anziche' iterare args e kwargs, mettere in cache gli indici dei tensori
    # e cose strane, posso direttamente definirla cosi'! E' la svolta se funziona
    # (annota processo e soluzione di questo problema)
    def casted_function(self_tensor, other_tensor=None, *args, **kwargs):
        '''
        Wrapper for `funct` (`Tensor`'s operation) that casts the args tensor(s) to mixed precision.

        Args:
            self_tensor (Tensor): it's the `self` arg of each `Tensor` operation
            other_tensor (Tensor): `other` arg of some `Tensor` operation

        It converts to float16 the tensors that:
        - are on a cuda device
        - have a precision higher than fp16
        If the tensor does not require grad, an inplace conversion is made, otherwise it's first cloned
        and then converted. This is done in order to avoid any variation to the master weights (original
        parameters of the network in fp32) because their update have to be done in fp32.
        '''
        #print("Attivo wrapper!")

        # NOTE: se l'operazione non e' safe, devi assicurarti che i tensori siano
        # in float32 (e il caasting lo fa sempre questo wrapper)
        if is_mixed_precision_enabled() and funct.__qualname__ in whitelist:
            target_dtype = get_mixed_precision_dtype()
            
            # opero solo sulle istanze della classe Tensor che sono su cuda
            if self_tensor.device == "cuda":
                if funct.__qualname__ in whitelist:
                    # posso lavorare in fp16

                    # se il dtype e' piu' alto di fp16
                    if self_tensor.dtype > target_dtype:
                        print(f"casto self.tensor! {self_tensor.dtype} -> {target_dtype}")
                        if self_tensor.requires_grad:
                            # mantengo i parametri originali in fp32 e faccio il calcolo
                            # con una copia in fp16
                            self_tensor = self_tensor.clone().to(target_dtype)
                        else:
                            # converto direttamente
                            self_tensor.to(target_dtype)
                else:
                    # funzione in blacklist: devo accertarmi che gli input siano fp32

                    # other teoricamente e' sempre in float32
                    self_tensor.to(float32) if self_tensor.dtype < float32 else self_tensor

            '''
            if getattr(other_tensor, "device", None) == "cuda" and other_tensor.dtype > target_dtype:
                if other_tensor.requires_grad:
                    other_tensor = other_tensor.clone().to(target_dtype)
                else:
                    other_tensor.to(target_dtype)
            '''
            # se il secondo arg e' un tensor ed e' su cuda
            if getattr(other_tensor, "device", None) == "cuda":
                if funct.__qualname__ in whitelist:
                    if other_tensor.dtype > target_dtype:
                        if other_tensor.requires_grad:
                            other_tensor = other_tensor.clone().to(target_dtype)
                        else:
                            other_tensor.to(target_dtype)
                else:
                    # other teoricamente e' sempre in float32
                    if other_tensor.dtype < float32:
                        raise RuntimeError(
                            f"A quanto pare other non e' sempre float32... {other_tensor.dtype}"
                        )

        if other_tensor is not None:
            return funct(self_tensor, other_tensor, *args, **kwargs)
        return funct(self_tensor, *args, **kwargs)
    return casted_function


# per standard, le classi context manager vanno scritte minuscole
class autocast:
    '''
    EEEE
    '''
    def __init__(self):
        pass

    def __enter__(self):
        if not CUPY_AVAILABLE:
            raise RuntimeError()
        
        set_mixed_precision_enabled(True)
        print(f"is_mixed_precision_enabled: {is_mixed_precision_enabled()}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_mixed_precision_enabled(False)


