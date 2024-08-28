from torchic.cuda_tensor import Tensor
from collections.abc import Generator


# non mi convince, potrebbe solo condividere zero_grad
class Optimizer:
    def step(self, *args, **kwargs):
        raise NotImplementedError

    def zero_grad(self, set_to_none=True) -> None:
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
        self.lr = lr
        
    def zero_grad(self, set_to_none=True):
        for param in self.parameters:
            param._grad = None if set_to_none else param._device.zeros(param._grad.shape, param.data.dtype)

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
    def __init__(self, params: Generator[Tensor], lr: float=0.001, beta1: float=0.9, beta2: float=0.999, eps: float=1e-8) -> None:
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
        self.iteration = 1
        self.m = [Tensor(p._device.zeros(p.data.shape)) for p in self.params]
        self.v = [Tensor(p._device.zeros(p.data.shape)) for p in self.params]

    
    def zero_grad(self, set_to_none=True):
        for param in self.params:
            param._grad = None if set_to_none else param._device.zeros(param._grad.shape, param.data.dtype)


    def step(self):
        # precalcolo i termini fuori dal loop anziche' ritrovarlo ad ogni iteraz.
        sub_beta1 = 1 - self.beta1
        sub_beta2 = 1 - self.beta2

        # NOTE: meglio fare device.power()?
        beta1_t = 1 - self.beta1**self.iteration
        beta2_t = 1 - self.beta2**self.iteration

        for param, m, v in zip(self.params, self.m, self.v):
            # ricalcolo m e v
            # NOTE: vedi se riesci a chiamare device.square() anziche' **
            #m.data = self.beta1 * m.data + (1 - self.beta1) * param._grad
            # le inplace op sono piu' veloci perche' non riallocano memoria
            m.data *= self.beta1
            m.data += sub_beta1 * param._grad

            v.data *= self.beta2
            v.data += sub_beta2 * param._device.square(param._grad)

            param.data -= self.lr * (m.data / beta1_t) / (v._device.sqrt(v.data / beta2_t) + self.eps)
        self.iteration += 1


#https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c