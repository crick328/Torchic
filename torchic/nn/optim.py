from torchic.cuda_tensor import Tensor
from torchic.tools.helpers import zeros_like
from collections.abc import Generator
from torchic.amp import wrappers
import inspect
from torchic.dtypes import float32
from torchic.settings import CUPY_AVAILABLE
if CUPY_AVAILABLE:
    import cupy as cp


# non mi convince, potrebbe solo condividere zero_grad
class Optimizer:
    def step(self, *args, **kwargs):
        raise NotImplementedError

    def zero_grad(self, set_to_none=True) -> None:
        #if CUPY_AVAILABLE:
            #cp.cuda.runtime.deviceSynchronize()
        wrappers.amp_cache = {}
        
        for param in self.parameters:
            param._grad = None if set_to_none else param._device.zeros(param._grad.shape, param.data.dtype)

    def __repr__(self):
        param_info = ''
        
        for attr_name in self.__slots__:
            # non salvo i parametri, ma memorizzo tutti gli altri attributi
            if not attr_name.startswith("param"):
                param_info += f"{attr_name}={self.__getattribute__(attr_name)}, "

        return f"{self.__class__.__name__}({param_info[:-2]})"



class SGD:
    __slots__ = ("parameters", "lr")

    def __init__(self, trainable_params: Generator[Tensor], lr=1e-3, momentum=0):
        # NOTA: questo funziona perché i miei oggetti sono incapsulati
        # all'interno di un oggetto mutabile (lista)
        # necessario se yieldo
        self.parameters = list(trainable_params)
        self.lr = Tensor(lr).data
        
    def zero_grad(self, set_to_none=True):
        for param in self.parameters:
            #param._grad = None if set_to_none else param._device.zeros(param._grad.shape, param.data.dtype)
            param._grad = None if set_to_none else param._device.zeros_like(param._grad)

    def step(self): 
        for param in self.parameters:
            param.data -= self.lr*param._grad
    # NOTA: dentro questa classe credo che sei libero di usare i return!


# Adaptive Moment Estimation
class AdamV3:
    __slots__ = ("params", "lr", "beta1", "beta2", "eps", "iteration", "m", "v")
    # se vuoi capire come funziona devi prima vederti RMSPropr e Momentum

    # credo sia meglio tenere i suoi params su tensor per evitare problemi con cupy
    # (non sembra fare differenze per semplici scalari... Boh, prob e' o piccolo)
    def __init__(self, params: Generator[Tensor], lr: float=0.001, beta1: float=0.9, beta2: float=0.999, eps: float=1e-8, _iteration: int=None, _m: list=None, _v: list=None) -> None:
        '''
        Parameters:
            lr: stepsize
            beta1, beta2: exponential decay rates for the momentum estimates
        '''
        # lista di [param, m, v] e invece m e v sono vettori con stessa shape
        # forse sarebbe meglio creare tre array separati ed iterarli con zip...
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # n_esimo aggiornamento (t)
        # NOTE: OCCHIO che i Tensor che ricevono un np.array diventano tensori
        # su cpu a prescindere dai global settings! (ED E' GIUSTO COSI')
        self.iteration = 1 if _iteration is None else _iteration
        self.m = [Tensor(p._device.zeros(p.data.shape)) for p in self.params] if _m is None else _m
        self.v = [Tensor(p._device.zeros(p.data.shape)) for p in self.params] if _v is None else _v

    
    def zero_grad(self, set_to_none=True):
        for param in self.params:
            param._grad = None if set_to_none else param._device.zeros(param._grad.shape, param.data.dtype)


    def step(self):
        # precalcolo i termini fuori dal loop anziche' ad ogni iteraz.
        sub_beta1 = 1 - self.beta1
        sub_beta2 = 1 - self.beta2

        # NOTE: meglio fare device.power()?
        beta1_t = 1 - self.beta1**self.iteration
        beta2_t = 1 - self.beta2**self.iteration

        for param, m, v in zip(self.params, self.m, self.v):
            # update biased first moment estimate (m)
            m.data *= self.beta1
            m.data += sub_beta1 * param._grad

            # update biased second moment estimate (v)
            v.data *= self.beta2
            v.data += sub_beta2 * param._device.square(param._grad)

            # compute bias-corrected first and second moments
            m_hat = m.data / beta1_t
            v_hat = v.data / beta2_t

            #param.data -= self.lr * (m.data / beta1_t) / (v._device.sqrt(v.data / beta2_t) + self.eps)
            param.data -= self.lr * m_hat / (v._device.sqrt(v_hat) + self.eps)
        self.iteration += 1

    # metodo chiamato per save(). Non uso repr() come per il modello
    # perche' in input ha bisogno dei parametri del nuovo modello
    def state_dict(self):
        '''
        Returns the state of the object as a (serializable) dictionary which
        contains key-value pairs representing the attributes of the optimizer.

        Keys are formatted as:
        - "`category`.`attr_name`" for simple attributes (e.g., `str`, `int`, or `float`);
        - "`category`.`attr_name`<`i`>" for complex
        attributes (`list` of `Tensor`s). `i` is an integer which represent the
        original position of the data in the list.

        `category` can be `"arg"`, which means that it is a value that
        can be passed as argument when creating an istance, or `"state"`,
        meaning that it's an internal variable of the optimizer. 

        Returns
        -------
        dict
            A dictionary where:
            - Keys are strings in the format described above.
            - Values are scalars (e.g., `str`, `int`, `float`) or `np.ndarray`.

        Examples
        --------
        >>> adam.state_dict()
        {
            'arg.lr': 0.001,
            'arg.beta1': 0.9,
            'arg.beta2': 0.999,
            'arg.eps': 1e-08,
            'state.iteration': 1,
            'state.m<0>': array(..., , dtype=float32),
            'state.m<1>': array(..., , dtype=float32),
            ...
            'state.m<87>': array(..., , dtype=float32),
            'state.v<0>': array(..., , dtype=float32),
            'state.v<1>': array(..., , dtype=float32),
            ...
            'state.v<87>': array(..., , dtype=float32)
        }
        '''
        state_dict = {}
        init_args = inspect.signature(self.__init__).parameters.keys()

        for attr in self.__slots__:
            # i parmetri del modello gli riprendero' dalla rete in memoria
            if attr == "params":
                continue

            value = self.__getattribute__(attr)
            # se e' un parametro dell'__init__
            if attr in init_args:
                state_dict[f"arg.{attr}"] = value
            else:
                # se l'attributo e' una lista e contiene solo tensori
                if type(value) == list and all(item.__class__ == Tensor for item in value):
                    # per ogni elemento della lista
                    for i in range(0, len(value)):
                        # porto in sicurezza il tensore su cpu e lo memorizzo
                        state_dict[f"state.{attr}<{i}>"] = value[i].detach().clone().to('cpu').data
                else:
                    # e' un attributo interno che non e' una lista di tensors
                    state_dict[f"state.{attr}"] = value
        return state_dict
    



from numpy import float16
# Adaptive Moment Estimation
class AdamV4:
    __slots__ = ("params", "lr", "beta1", "beta2", "eps", "iteration", "m", "v")
    # se vuoi capire come funziona devi prima vederti RMSPropr e Momentum

    # credo sia meglio tenere i suoi params su tensor per evitare problemi con cupy
    # (non sembra fare differenze per semplici scalari... Boh, prob e' o piccolo)
    def __init__(self, params: Generator[Tensor], lr: float=0.001, beta1: float=0.9, beta2: float=0.999, eps: float=1e-8, _iteration: int=None, _m: list=None, _v: list=None) -> None:
        '''
        Parameters:
            lr: stepsize
            beta1, beta2: exponential decay rates for the momentum estimates
        '''
        # lista di [param, m, v] e invece m e v sono vettori con stessa shape
        # forse sarebbe meglio creare tre array separati ed iterarli con zip...
        self.params = list(params)
        self.lr = Tensor(lr).data
        self.beta1 = Tensor(beta1).data#, dtype=float16).data
        self.beta2 = Tensor(beta2).data#, dtype=float16).data
        self.eps = Tensor(eps).data

        # n_esimo aggiornamento (t)
        # NOTE: OCCHIO che i Tensor che ricevono un np.array diventano tensori
        # su cpu a prescindere dai global settings! (ED E' GIUSTO COSI')
        self.iteration = 1 if _iteration is None else _iteration
        self.m = [zeros_like(p).data for p in self.params] if _m is None else _m
        self.v = [zeros_like(p).data for p in self.params] if _v is None else _v

    
    def zero_grad(self, set_to_none=True):
        for param in self.params:
            param._grad = None if set_to_none else param._device.zeros(param._grad.shape, param.data.dtype)


    def step(self):
        # precalcolo i termini fuori dal loop anziche' ad ogni iteraz.
        sub_beta1 = 1 - self.beta1
        sub_beta2 = 1 - self.beta2

        # NOTE: meglio fare device.power()?
        beta1_t = 1 - self.beta1.__pow__(self.iteration)
        beta2_t = 1 - self.beta2.__pow__(self.iteration)

        for param, m, v in zip(self.params, self.m, self.v):
            # update biased first moment estimate (m)
            m *= self.beta1
            m += sub_beta1 * param._grad

            # update biased second moment estimate (v)
            v *= self.beta2
            v += sub_beta2 * param._device.square(param._grad)

            # compute bias-corrected first and second moments
            m_hat = m / beta1_t
            v_hat = v / beta2_t

            assert m_hat.dtype == float32
            assert v_hat.dtype == float32
            assert self.eps.dtype == float32

            #param.data -= self.lr * (m.data / beta1_t) / (v._device.sqrt(v.data / beta2_t) + self.eps)
            param.data -= self.lr * m_hat / (v_hat**0.5 + self.eps) #+1e-4)
        self.iteration += 1

    # metodo chiamato per save(). Non uso repr() come per il modello
    # perche' in input ha bisogno dei parametri del nuovo modello
    def state_dict(self):
        '''
        Returns the state of the object as a (serializable) dictionary which
        contains key-value pairs representing the attributes of the optimizer.

        Keys are formatted as:
        - "`category`.`attr_name`" for simple attributes (e.g., `str`, `int`, or `float`);
        - "`category`.`attr_name`<`i`>" for complex
        attributes (`list` of `Tensor`s). `i` is an integer which represent the
        original position of the data in the list.

        `category` can be `"arg"`, which means that it is a value that
        can be passed as argument when creating an istance, or `"state"`,
        meaning that it's an internal variable of the optimizer. 

        Returns
        -------
        dict
            A dictionary where:
            - Keys are strings in the format described above.
            - Values are scalars (e.g., `str`, `int`, `float`) or `np.ndarray`.

        Examples
        --------
        >>> adam.state_dict()
        {
            'arg.lr': 0.001,
            'arg.beta1': 0.9,
            'arg.beta2': 0.999,
            'arg.eps': 1e-08,
            'state.iteration': 1,
            'state.m<0>': array(..., , dtype=float32),
            'state.m<1>': array(..., , dtype=float32),
            ...
            'state.m<87>': array(..., , dtype=float32),
            'state.v<0>': array(..., , dtype=float32),
            'state.v<1>': array(..., , dtype=float32),
            ...
            'state.v<87>': array(..., , dtype=float32)
        }
        '''
        state_dict = {}
        init_args = inspect.signature(self.__init__).parameters.keys()

        for attr in self.__slots__:
            # i parmetri del modello gli riprendero' dalla rete in memoria
            if attr == "params":
                continue

            value = self.__getattribute__(attr)
            # se e' un parametro dell'__init__
            if attr in init_args:
                state_dict[f"arg.{attr}"] = value
            else:
                # se l'attributo e' una lista e contiene solo tensori
                if type(value) == list and all(item.__class__ == Tensor for item in value):
                    # per ogni elemento della lista
                    for i in range(0, len(value)):
                        # porto in sicurezza il tensore su cpu e lo memorizzo
                        state_dict[f"state.{attr}<{i}>"] = value[i].detach().clone().to('cpu').data
                else:
                    # e' un attributo interno che non e' una lista di tensors
                    state_dict[f"state.{attr}"] = value
        return state_dict


#https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c