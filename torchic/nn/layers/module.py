class ForceSlots(type):
    @classmethod
    def __prepare__(metaclass, name, bases, **kwds):
        # calling super is not strictly necessary because
        #  type.__prepare() simply returns an empty dict.
        # But if you plan to use metaclass-mixins then this is essential!
        super_prepared = super().__prepare__(metaclass, name, bases, **kwds)
        super_prepared['__slots__'] = ()
        print("CHIAMATO")
        return super_prepared
    

class PostInit(type):  # deriving from type is what makes a metaclass
   def __call__(cls, *args, **kw):
       instance = super().__call__(*args, **kw) # "instance" now contains the new instance, after __init__ is run
       
       # code to retrieve custom configurations and apply to the instance:
       
       return instance


class Module(metaclass=PostInit):
    __slots__ = ("merda", "supermerdone")

    # avra' attributi che sono tutti tensor
    # questa struttura richiede che le classi che ereditano
    # definiscono un nuovo __slots__...
    def __init__(self) -> None:
        self.merda=4
        self.supermerdone=5
        print("PASSO E CHIUDO")

    def __iter__(self):
        if isinstance(self.__slots__, str):
            yield self.__getattribute__(self.__slots__)
        else:
            for name in self.__slots__:
                print("INNER")
                yield self.__getattribute__(name)
    
    def forward(self, x):
        raise NotImplementedError("Devi implementare il metodo forward")
    
    def zero_grad(self):
        for tensor in self:
            
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

    def parameters(self):
        return self.parameters
    

class Merda(Module):
    def __init__(self) -> None:
        print("IONIT")
        super().__init__()

a = Module()

for x in a:
    print(x)
