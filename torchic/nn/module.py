from torchic.cuda_tensor import Tensor
from collections.abc import Generator
import inspect

# arrivati a sto punto toglierei il supportoagli slots singoli (attr)
# GPT dice che la tupla singola vuole la virogla, boh


# Metaclass che forza l'uso di __slots__ e li estende automaticamente
class ForceSlots(type):
    # Aggiunge '_init_params' agli __slots__ esistenti, se presenti
    def __new__(cls, name, bases, classdict):
        classdict['__slots__'] = ('_init_params',) + classdict.get('__slots__', ())
        #slots = classdict.get('__slots__', ())
        #if isinstance(slots, str):
            #slots = (slots,)
        
        #classdict['__slots__'] = ('_init_params',) + slots
        return super().__new__(cls, name, bases, classdict)



class Module(metaclass=ForceSlots):
    '''
    Base class of every `nn` component that has trainable `Tensor`s.
    This class is used to identify the type of object which the code
    interacts with and is useful for accessibility.
    The attributes of a `Module` can be anything, even another `Module`.
    *Only* `Tensor` attributes are considered by the function `parameters()`.  
    
    ForceSlots automatically creates an empty __slots__ on self and
    parent classes. It's a reminder to use __slots__ and not __dict__.
    '''

    def parameters(self, trainable_only: bool=True, with_names: bool=False, _layer_name=""):
        '''
        Yields all trainable params by default, but can
        also yield all tensor params if trainable_only
        is False. If with_names is True, it will return
        a tuple of (name, param) instead of just param.
        '''

        for name, param in self:
            # Se il parametro è un tensor
            if isinstance(param, Tensor):
                if not trainable_only or param.requires_grad:
                    # NOTE: le parentesi della tupla sono fondamentali per non avere
                    # comportamenti errati!
                    yield param if not with_names else (f"{_layer_name}{name}", param)
                    #yield (f"{_layer_name}{name}", param) if with_names else param

            elif isinstance(param, Module):
                yield from param.parameters(trainable_only, with_names, _layer_name=f"{_layer_name}{name}.")


    def state_dict(self, layer_name: str) -> Generator:
        '''
        This method should be called only by the Sequential class
        or their child. It's supposed to return

        Args:
            layer_name (str): identifier of the layer inside the model

        Returns the state of the trainable params.
        Restituisce
        '''
        for name, param in self:
            if isinstance(param, Tensor) and param.requires_grad:
                yield f"{layer_name}.{name}", param.data

            elif isinstance(param, Module):
                yield from param.state_dict(f"{layer_name}.{name}")

            #elif isinstance(param, ModuleList):
                #for sublayer in param:
                    #yield from sublayer.state_dict(f"{layer_name}.{name}")

            
    def to(self, *args):
        for _, param in self:
            if isinstance(param, Tensor) or isinstance(param, Module):
                param.to(*args)

    def freeze(self):
        self._set_requires_grad(False)

    def unfreeze(self):
        # NOTE: in questo modo potresti settare a True
        # il gradiente di alcune componenti che non dovrebbero
        # essere trainabili!
        self._set_requires_grad(True)

    def _set_requires_grad(self, value: bool):
        for _, param in self:
            if isinstance(param, Tensor):
                param.requires_grad = value
            elif isinstance(param, Module):
                param._set_requires_grad(value)


    # ==== __MAGIC__ FUNCTIONS ====

    def __new__(cls, *args, **kwargs):
        # in pratica intercetta la creazione dell'oggetto
        # per salvare gli args
        instance = super().__new__(cls)
        # salvo (solo) i parametri di init passati dall'utente
        instance._init_params = (args, kwargs)
        return instance

    def __repr__(self):
        class_name = self.__class__.__name__
        # Non esce ModuleList,ma il wrapper
        args, kwargs = self._init_params
        # dizionario contenente tutti i parametri degli init
        # con relativi valori di default (se ci sono)
        init_signature = inspect.signature(self.__init__)
        
        # Ricostruzione della rappresentazione
        param_strs = []
        for i, (arg_name, default_value) in enumerate(init_signature.parameters.items()):
            # NOTE: !r nelle f-string e' un modo per chiedere la repr della
            # variabile nelle graffe

            # se e' un argomento posizionale
            if i < len(args):
                param_strs.append(f"{arg_name}={args[i]!r}")

            # se e' un argomento passato con keyword
            elif arg_name in kwargs:
                param_strs.append(f"{arg_name}={kwargs[arg_name]!r}")
                
            # allora e' un parametro che ha preso il valore di default
            else:
                param_strs.append(f"{arg_name}={default_value.default!r}")
        
        formatted_params = ", ".join(param_strs)
        return f"{class_name}({formatted_params})"

    def __call__(self, *args, **kwargs):
        try:
            return self.forward(*args, **kwargs)
        except Exception as e:
            print(e)
            raise NotImplementedError

    def __iter__(self):
        """Itera attraverso gli attributi definiti in __slots__."""
        
        if isinstance(self.__slots__, str):
            # ha un solo attributo, __slots__ e' una stringa
            yield self.__slots__, self.__getattribute__(self.__slots__)
        else:
            for attr_name in self.__slots__:
                yield attr_name, self.__getattribute__(attr_name)



class ModuleList(Module):
    __slots__ = ("modules",)

    def __init__(self, *args: list[Module] | tuple[Module]) -> None:
        self.modules = list(*args)

    def __iter__(self) -> Generator[Module, None, None]:
        yield from self.modules

    def __call__(self, x: Tensor) -> Tensor:
        # al momento non supporta i multi-input
        for module in self:
            x = module(x)
        return x
    
    def __len__(self) -> int:
        return len(self.modules)

    def __getattribute__(self, function_name):
        # chiamata ogni volta che si accede ad un attributo.
        # Questo mi permette di intercettare le chiamate alle
        # funzioni di Module per poter ciclare, senza doverle
        # ridefinire una ad una in ModuleList

        def function_call_wrapper(*args, **kwargs):
            # chiamo la funzione interpellata da ogni Module
            for module in self.modules:
                yield from getattr(module, function_name)(*args, **kwargs)
        
        # puo' essere un attributo o una funzione
        x = super(ModuleList, self).__getattribute__(function_name)

        # non wrappo i metodi __magic__ e gli attributi
        if function_name.startswith('__') and function_name.endswith('__') or not callable(x):
            return x
        else:
            return function_call_wrapper
    
    def __getitem__(self, index: int|slice):
        return self.modules[index]