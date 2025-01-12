from torchic.cuda_tensor import Tensor
from collections.abc import Generator
import inspect

# arrivati a sto punto toglierei il supportoagli slots singoli (attr)
# GPT dice che la tupla singola vuole la virogla, boh


# Metaclass che forza l'uso di __slots__ e li estende automaticamente
class ForceSlots(type):
    # Aggiunge '_init_params' agli __slots__ esistenti, se presenti
    def __new__(cls, name, bases, classdict):
        # prendo gli slots della classe che sto istanziando e aggiungo _init_params
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
    
    ForceSlots automatically creates an empty __slots__ on self and
    parent classes. It's a reminder to use __slots__ and not __dict__.
    '''

    def parameters(self, trainable_only: bool=True, with_names: bool=False, tensor_only: bool=True, _layer_name: str="") -> Generator[Tensor] | Generator[str, Tensor]:
        '''
        Yields all trainable `Tensor`s by default.
        
        Args:
            trainable_only (bool): yield all tensor params if is False
            with_names (bool): it will return a tuple of (name, param) instead of just param.
            tensor_only (bool):
            _layer_name (str): internal param used to correctly build the names' dependencies
        '''

        for name, param in self:
            # Se il parametro è un tensor
            if isinstance(param, Tensor):
                if not trainable_only or param.requires_grad:
                    # NOTE: le parentesi della tupla sono fondamentali per non avere
                    # comportamenti errati!
                    yield param if not with_names else (f"{_layer_name}.{name}", param)
                    #yield (f"{_layer_name}{name}", param) if with_names else param

            elif isinstance(param, Module):
                #print(f"ENTRO {name}")
                yield from param.parameters(trainable_only, with_names, tensor_only, _layer_name=f"{_layer_name}.{name}")

            elif not tensor_only:
                yield param if not with_names else (f"{_layer_name}.{name}", param)
            
    #def _state_dict(self, layer_name: str) -> Generator:
        #for name, param in self.parameters(True, True, True, _layer_name=layer_name):
            #yield name, param.detach().clone().to("cpu").data

    def state_dict(self, layer_name: str="") -> dict:
        '''
        Returns the state of the trainable params.
        
        Example::

            >>> print(model.state_dict())
            {'linear0.weights': array([[ 0.09000251,  0.03145568, -0.7834164 ],
                            [-0.4419116 , -0.02768573, -1.3669962 ],
                            [ 0.36691672, -0.23904948, -0.5104246 ],
                            [ 0.1395965 ,  0.5443452 , -0.47324494]], dtype=float32),
            'linear0.bias': array([-0.8548351 , -0.39590812,  0.35631013], dtype=float32),
            'linear2.weights': array([[ 0.4408978 , -0.26182708],
                            [ 0.7175888 , -0.998319  ],
                            [ 0.79341125, -0.5995235 ]], dtype=float32),
            'linear2.bias': array([1.2450343 , 0.93363947], dtype=float32),
            'linear4.weights': array([[0.8089074 ], [0.26144603]], dtype=float32),
            'linear4.bias': array([0.04495028], dtype=float32)}
        '''

        state = {}
        for name, param in self.parameters(True, True, True, _layer_name=layer_name):
            # aggiungo la coppia key-value al dizionario di stato
            state[name] = param.detach().clone().to("cpu").data
        return state
    
    def number_of_params(self, trainable_only: bool=True) -> int:
        n = 0
        for param in self.parameters(trainable_only):
            n += param.size()
        return n

    def to(self, *args) -> None:
        for param in self.parameters(trainable_only=False):
            param.to(*args)

    def train(self) -> None:
        self._set_training_mode(True)

    def eval(self) -> None:
        self._set_training_mode(False)

    def _set_training_mode(self, flag: bool):
        for name, elem in self:
            if name == "training":
                setattr(self, name, flag)
            
            elif isinstance(elem, Module):
                elem._set_training_mode(flag)
        

    # non hanno senso. Se non vuoi il gradiente a livello globale, ti basta
    # disattivarlo, altrimenti devi andare te a selezionare cosa freezare.
    # Non posso saperlo in anticipo
    def freeze(self) -> None:
        self._set_requires_grad(False)

    def unfreeze(self) -> None:
        # NOTE: in questo modo potresti settare a True
        # il gradiente di alcune componenti che non dovrebbero
        # essere trainabili!
        # C'e' bisogno di una cache
        self._set_requires_grad(True)

    def _set_requires_grad(self, value: bool) -> None:
        pass


    # ==== __MAGIC__ FUNCTIONS ====

    def __new__(cls, *args, **kwargs):
        # salvo (solo) i parametri di init passati dall'utente
        # in pratica intercetta la creazione dell'oggetto
        # per salvare gli args
        instance = super().__new__(cls)
        if not args:
            args = None
        if not kwargs:
            kwargs = None
        instance._init_params = (args, kwargs)
        return instance

    def __repr__(self):
        class_name = self.__class__.__name__
        # Non esce ModuleList,ma il wrapper
        args, kwargs = self._init_params
        # oggetto Signature contenente tutti i parametri degli init
        # con relativi valori di default (se ci sono)
        init_signature = inspect.signature(self.__init__)

        # Ricostruzione della rappresentazione
        param_strs = []

        if "args" in init_signature.parameters:
            assert len(init_signature.parameters.values()) == 1
            # ho ricevuto SOLO argomenti posizionali e
            # l'init non ha firma
            for arg in args:
                param_strs.append(f"{arg!r}")
        else:
            for i, (arg_name, default_value) in enumerate(init_signature.parameters.items()):
                # NOTE: !r nelle f-string e' un modo per chiedere la repr della
                # variabile nelle graffe

                # se e' un argomento posizionale
                if args and i < len(args):
                    param_strs.append(f"{arg_name}={args[i]!r}")

                # se e' un argomento passato con keyword
                elif kwargs and arg_name in kwargs:
                    param_strs.append(f"{arg_name}={kwargs[arg_name]!r}")
                    
                # allora e' un parametro che ha preso il valore di default
                else:
                    param_strs.append(f"{arg_name}={default_value.default!r}")
        
        formatted_params = ", ".join(param_strs)
        return f"nn.{class_name}({formatted_params})"

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __iter__(self):
        """Itera attraverso gli attributi definiti in __slots__."""
        
        if isinstance(self.__slots__, str):
            # ha un solo attributo, __slots__ e' una stringa
            yield self.__slots__, self.__getattribute__(self.__slots__)
        else:
            for attr_name in self.__slots__:
                yield attr_name, self.__getattribute__(attr_name)

    def __str__(self):
        # non posso chiamare __str__ a cascata perche'
        # non potrei passare elem e depth

        output = f"{self.__class__.__name__}(\n"
        for name, layer in self:#.parameters(trainable_only=False, with_names=True, tensor_only=False):
            output += f"  ({name}): " + Module._build_fstring(layer, 2)
        return output + ")"
    
    @staticmethod
    def _build_fstring(elem, depth: int):
        # incolonno bene
        tab = "  "
        tabs = tab * depth

        if isinstance(elem, ModuleList):
            output = f"{elem.__class__.__name__}(\n"

            # se ModuleList contiene layer tutti uguali
            if len(set(repr(module) for name, module in elem)) == 1:
                output += (
                    f"{tabs}(0-{len(elem) - 1}): {len(elem)} x "
                    + Module._build_fstring(elem[0], depth + 1)
                    )
            else:
                for i, module in enumerate(elem):
                    output += (
                        f"{tabs}({i}): "
                        + Module._build_fstring(module, depth + 1)
                        )
            return output + f"{tab * (depth - 1)})\n"
        
        elif isinstance(elem, Module):
            # se il Module non ha altri Module come attributi
            if all(not isinstance(param, Module) for _, param in elem):
                return f"{repr(elem)}\n"
            else:
                output = f"{elem.__class__.__name__}(\n"

                for name, param in elem:
                    if isinstance(param, Module):
                        output += (
                            f"{tabs}({name}): "
                            + Module._build_fstring(param, depth + 1)
                            )
                return output + f"{tab * (depth - 1)})\n"
        else:
            # e' una act funct
            return f"{elem.__class__.__name__}()\n"



class ModuleList(Module):
    __slots__ = ("modules",)

    def __init__(self, *args: list[Module] | tuple[Module]) -> None:
        self.modules = list(*args)

    def __iter__(self):
        for i in range(0, len(self.modules)):
            yield f"{i}", self.modules[i]

    #def __iter__(self) -> Generator[Module]:
        #yield from self.modules

    def __call__(self, x) -> Tensor:
        # al momento non supporta i multi-input
        for module in self:
            x = module(x)
        return x
    
    def __len__(self) -> int:
        return len(self.modules)

    def _tr__getattribute__(self, function_name):
        # chiamata ogni volta che si accede ad un attributo.
        # Questo mi permette di intercettare le chiamate alle
        # funzioni di Module per poter ciclare, senza doverle
        # ridefinire una ad una in ModuleList

        def function_call_wrapper(*args, **kwargs):
            print("CHIAMATA a function wrapper!")
            # chiamo la funzione interpellata da ogni Module
            for module in self.modules:
                # se la funzione chiamata e' parameters() e l'utente
                # sta chiedendo i nomi dei layers
                if function_name == "parameters" and "_layer_name" in kwargs:#kwargs["_layer_name"] != "":
                    last_char = kwargs["_layer_name"][-1]

                    # aggiungo o incremento l'index del ModuleList
                    if last_char.isdigit():
                        # incremento e sostituisco l'index
                        last_char = str(int(last_char) + 1)
                        kwargs["_layer_name"] = kwargs["_layer_name"][:-1] + last_char
                    else:
                        # aggiungo l'index
                        last_char = ".0"
                        kwargs["_layer_name"] = kwargs["_layer_name"] + last_char
                
                yield from getattr(module, function_name)(*args, **kwargs)
                #output = getattr(module, function_name)(*args, **kwargs)
                #if output is not None:
                    #print(type(output))
                    #yield from output
        
        # puo' essere un attributo o una funzione
        x = super(ModuleList, self).__getattribute__(function_name)

        if function_name == "_set_training_mode":
            print(f"sto per chiamare _set_training_mode()")

        # non wrappo i metodi __magic__ e gli attributi
        if function_name.startswith('__') and function_name.endswith('__') or not callable(x):
            return x
        else:
            print(f"Wrappo {function_name}")
            return function_call_wrapper
    
    def __getitem__(self, index: int|slice) -> Module:
        return self.modules[index]