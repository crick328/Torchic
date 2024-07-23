from torchic.cuda_tensor import Tensor
import numpy as np


class SGD:
    __slots__ = ("parameters", "lr")

    def __init__(self, trainable_params: list[Tensor], lr=1e-3, momentum=0):
        # NOTA: questo funziona perché i miei oggetti sono incapsulati
        # all'interno di un oggetto mutabile (lista)
        self.parameters = trainable_params
        self.lr = lr
        
    def zero_grad(self, set_to_none=True):
        for param in self.parameters:
            param._grad = None if set_to_none else param._device.zeros(param._grad.shape, param.data.dtype)

    # qua ci perdo 97s... Conta che
    # outer chiede 52.5s, mentre
    # matmul 28.7s...

    def step(self): 
        for param in self.parameters:
            param.data -= self.lr*param._grad
            #print(param.grad)
    # NOTA: dentro questa classe credo che sei libero di usare i return!


# Adaptive Moment Estimation
class Adam:
    # se vuoi capire come funziona devi prima vederti RMSPropr e Momentum

    # credo sia meglio tenere i suoi params su tensor per evitare problemi con cupy
    # (non sembra fare differenze per semplici scalari... Boh, prob e' o piccolo)
    def __init__(self, params: list[Tensor], lr: float=0.001, beta1: float=0.9, beta2: float=0.999, eps: float=1e-8) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        # n_esimo aggiornamento
        self.iteration = 1

        # alternativamente, potrei provare a rendere params una lista bidimensionale
        # dove ogni ingresso e' [param, m, v] in modo da poter loopare facendo
        # for param, m, v in self.params (che a sto punto puo' cambiare nome)
        self.m = [0.0 for p in params]  # Inizializzazione della media mobile dei gradienti
        self.v = [0.0 for p in params]  # Inizializzazione della media mobile dei quadrati dei gradienti


    def zero_grad(self, set_to_none=True):
        for param in self.params:
            param._grad = None if set_to_none else param._device.zeros(param._grad.shape, param.data.dtype)
    

    def step(self):
        for i in range(0, len(self.params)):
            # ricalcolo m e v

            # Update biased first moment estimate.
            # m is the exponentially moving average of the gradients.
            # beta1 is the decay rate for the first moment.
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * self.params[i]._grad

            # Update biased second raw moment estimate.
            # v is the exponentially moving average of the squared gradients.
            # beta2 is the decay rate for the second moment.
            # NOTE: vedi se riesci a chiamare device.square() anziche' **
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * self.params[i]._grad**2

            # Compute bias-corrected first moment estimate.
            # This corrects the bias in the first moment caused by initialization at origin.
            m_hat = self.m[i] / (1 - self.beta1**self.iteration)

            # Compute bias-corrected second raw moment estimate.
            # This corrects the bias in the second moment caused by initialization at origin.
            v_hat = self.v[i] / (1 - self.beta2**self.iteration)

            # Update parameters.
            # Parameters are adjusted based on the learning rate, corrected first moment,
            # and the square root of the corrected second moment.
            # epsilon is a small number to avoid division by zero.
            self.params[i].data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            self.iteration += 1




# Adaptive Moment Estimation
class AdamV2:
    # se vuoi capire come funziona devi prima vederti RMSPropr e Momentum

    # credo sia meglio tenere i suoi params su tensor per evitare problemi con cupy
    # (non sembra fare differenze per semplici scalari... Boh, prob e' o piccolo)
    def __init__(self, params: list[Tensor], lr: float=0.001, beta1: float=0.9, beta2: float=0.999, eps: float=1e-8) -> None:
        '''
        Parameters:
            lr: stepsize
            beta1, beta2: exponential decay rates for the momentum estimates
        '''
        # lista di [param, m, v] e invece m e v sono vettori con stessa shape
        # forse sarebbe meglio creare tre array separati ed iterarli con zip...
        self.params: list[list[Tensor, Tensor, Tensor]]
        self.params = [[p, Tensor(np.zeros(p.data.shape)), Tensor(np.zeros(p.data.shape))] for p in params]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        # n_esimo aggiornamento (t)
        self.iteration = 1

    
    def zero_grad(self, set_to_none=True):
        for param, _, _ in self.params:
            param._grad = None if set_to_none else param._device.zeros(param._grad.shape, param.data.dtype)


    def step(self):
        #print(len(self.params))
        #print(f"EMME: {self.params[1][1]}")
        for param, m, v in self.params:
            # ricalcolo m e v
            # NOTE: vedi se riesci a chiamare device.square() anziche' **
            m.data = self.beta1 * m.data + (1 - self.beta1) * param._grad
            v.data = self.beta2 * v.data + (1 - self.beta2) * param._grad**2

            m_hat = m.data / (1 - self.beta1**self.iteration)
            v_hat = v.data / (1 - self.beta2**self.iteration)

            # Update parameters.
            param.data -= self.lr * m_hat / (v._device.sqrt(v_hat) + self.eps)
        self.iteration += 1





class AdamV3:
    # se vuoi capire come funziona devi prima vederti RMSPropr e Momentum

    # credo sia meglio tenere i suoi params su tensor per evitare problemi con cupy
    # (non sembra fare differenze per semplici scalari... Boh, prob e' o piccolo)
    def __init__(self, params: list[Tensor], lr: float=0.001, beta1: float=0.9, beta2: float=0.999, eps: float=1e-8) -> None:
        '''
        Parameters:
            lr: stepsize
            beta1, beta2: exponential decay rates for the momentum estimates
        '''
        # lista di [param, m, v] e invece m e v sono vettori con stessa shape
        # forse sarebbe meglio creare tre array separati ed iterarli con zip...
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        #self.betas = Tensor([beta1, beta2])
        #self.beta1 = Tensor(beta1, device=params[0].device).data
        #self.beta2 = Tensor(beta2, device=params[0].device).data
        self.eps = eps

        # n_esimo aggiornamento (t)
        # NOTE: OCCHIO che i Tensor che ricevono un np.array diventano tensori
        # su cpu a prescindere dai global settings! (ED E' GIUSTO COSI')
        self.iteration = 1
        self.m = [Tensor(np.zeros(p.data.shape), device=p.device) for p in params]
        self.v = [Tensor(np.zeros(p.data.shape), device=p.device) for p in params]

    
    def zero_grad(self, set_to_none=True):
        for param in self.params:
            param._grad = None if set_to_none else param._device.zeros(param._grad.shape, param.data.dtype)


    def step(self):
        # beta in tensor separati
        # lr e eps come tensor
        
        # precalcolo il termine fuori dal loop anziche' ritrovarlo ad ogni iteraz.
        sub_beta1 = 1 - self.beta1
        sub_beta2 = 1 - self.beta2

        # NOTE: meglio fare device.power()?
        beta1_t = 1 - self.beta1**self.iteration
        beta2_t = 1 - self.beta2**self.iteration
        #beta1_t = 1 - self.params[0]._device.power(self.beta1, self.iteration)
        #beta2_t = 1 - self.params[0]._device.power(self.beta2, self.iteration)

        for param, m, v in zip(self.params, self.m, self.v):
            # ricalcolo m e v
            # NOTE: vedi se riesci a chiamare device.square() anziche' **
            #m.data = self.beta1 * m.data + (1 - self.beta1) * param._grad
            # le inplace op sono piu' veloci perche' non riallocano memoria
            m.data *= self.beta1
            m.data += sub_beta1 * param._grad
            #m.data = self.beta1 * m.data + sub_beta1 * param._grad

            #v.data = self.beta2 * v.data + (1 - self.beta2) * param._grad**2
            v.data *= self.beta2
            #v.data += sub_beta2 * param._grad**2
            v.data += sub_beta2 * param._device.square(param._grad)

            #x_hat = x / (1 - beta^t)
            #m_hat = m.data / beta1_t
            #v_hat = v.data / beta2_t

            # Update parameters.
            #param.data -= self.lr * m_hat / (v._device.sqrt(v_hat) + self.eps)
            param.data -= self.lr * (m.data / beta1_t) / (v._device.sqrt(v.data / beta2_t) + self.eps)

            #print(m.data.dtype)
            #print(v.data.dtype)
            #print(param.data.dtype)
        self.iteration += 1


#https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c