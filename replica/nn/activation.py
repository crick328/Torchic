# RULE OF THUMB: se devi solo fare operazioni sui .data e non
# ti serve la backprop, chiama solo quelli anziché tirare sempre in ballo i Tensor!
# metodo interessante per mantenere nella cache i risultati: https://docs.python.org/3/library/functools.html#functools.lru_cache

from replica.cuda_tensor import CudaTensor
import numpy as np


class Sigmoid:
    # ho bisogno che sia una classe per usare __repr__
    '''
    Input: 1 dimensional Tensor
    Output: 1 dimensional Tensor having same shape as input
    '''
    def __call__(self, x: CudaTensor) -> CudaTensor:
        output = CudaTensor(1/(1 + x._device.exp(-x)))

        def _sigmoid_backward(chain_rule_tensor: CudaTensor) -> None:
            # d/dσ = σ(x) * (1 - σ(x)); dL/dσ = d/dσ * prodotto derivate parziali delle operazioni precedenti
            x.grad += self(x)*(1-self(x)) * chain_rule_tensor
            # non può essere leaf, quindi chiamo direttamente la grad_fn
            x.grad_fn(x.grad)
            # non è mai leaf, quindi pulisco il gradient
            x.grad = None
            # elimino il pointer in modo che il garbage collector possa cancellare le variabili inutili
            x.grad_fn = None

        if x.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.grad_fn = _sigmoid_backward
        return output
    
    def __repr__(self):
        return "Sigmoid()"


class ReLU:
    def __call__(self, x: CudaTensor) -> CudaTensor:
        output = CudaTensor(x._device.maximum(0, x.data))

        def _ReLU_backward(chain_rule_tensor: CudaTensor) -> None:
            #partial_derivative = []
            #for value in x.data:
                #partial_derivative.append( 1 if value >= 0 else 0 )
            #nums = np.where(nums > 0, 1, nums)

            #print(partial_derivative)
            #x.grad += partial_derivative * chain_rule_tensor
            # crea overflow dopo una cinquantina di epochs
            #x.grad += Tensor(np.where(x.data > 0, 1, x.data) * chain_rule_tensor.data)
            # OCCHIO: QUESTO METODO MODIFICA X!
            #self._xp.put(x.data, x.data.nonzero()[0], 1)
            #x.grad += CudaTensor(x.data * chain_rule_tensor.data)
            x.grad += CudaTensor(x._device.where(x.data > 0, 1, 0) * chain_rule_tensor.data)
            x.grad_fn(x.grad)

            x.grad = None
            x.grad_fn = None

        if x.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.grad_fn = _ReLU_backward
        return output
    

# prima versione della softmax. Non la cancello
# perché ha la derivata vanilla e comprensibile
class SoftmaxOG:
    def __call__(self, x: CudaTensor) -> CudaTensor:
        #output = []
        
        # questa singola linea di codice previene overflow senza modificare
        # l'efficacia del gradient.
        shiftx = x.data - x._device.max(x.data)#x.data.max() #self._xp.max(x.data)
        exps = x._device.exp(shiftx)
        output = CudaTensor(exps / x._device.sum(exps))

        
        def _softmax_backward(chain_rule_tensor: CudaTensor) -> None:
            '''
            la derivata della softmax è una matrice. Ogni colonna rappresenta una sua
            derivata completa w.r.t. un input
            '''
            shiftx = x.data - x._device.max(x.data) #x.max()
            exps = x._device.exp(shiftx)
            softmax_x = exps / x._device.sum(exps)

            #for i in range(0, len(x.data)):
                #vector = []
                #softmax_i = np.exp(shiftx[i])/denominatore
                #softmax_i = exps[i]/denominatore

                #for j in range(0, len(x.data)):
                    #if i == j:
                        # d_S(x[i])/d_x[i] = S(i)*(1-S(i))
                        #vector.append( (softmax_i) * (1 - softmax_i) )
                    #else:
                        # d_S(x[j])/d_x[i] = -S(j)*S(i)
                        #vector.append( -(softmax_i)*(exps[j]/denominatore))
                        #pass
                #jacobian.append(vector) #* chain_rule_tensor)

            SM = softmax_x.reshape((-1,1))
            jacobian = x._device.diagflat(softmax_x) - np.dot(SM, SM.T)

            # la matmul permette di passare da jacobian a vettore per
            # continuare la backward (oltre a propagare correttamente il grad ricevuto)
            x.grad += CudaTensor(jacobian).matmul(chain_rule_tensor)
            x.grad_fn(x.grad)

            x.grad = None
            x.grad_fn = None

        if x.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.grad_fn = _softmax_backward
        return output
    

class Softmax:
    # Può essere calcolata solo su una dimensione
    def __init__(self, dim=0) -> None:
        self.dim = dim

    def __call__(self, x: CudaTensor) -> CudaTensor:
        # questa singola linea di codice previene overflow senza modificare
        # l'efficacia del gradient.
        shiftx = x.data - x.data.max(axis=self.dim, keepdims=True)
        exps = x._device.exp(shiftx)
        output = CudaTensor(exps / x._device.sum(exps, axis=self.dim, keepdims=True))

        # NOTE: riciclare il calcolo della softmax è geniale! Fallo ovunque!
        def _softmax_backward(chain_rule_tensor: CudaTensor) -> None:
            '''
            la derivata della softmax è una matrice. Ogni colonna rappresenta una sua
            derivata completa w.r.t. un input
            '''
            # CHADGPT. Calcolo introvabile. ChatGPT l'ha scritto versione Torch
            # e io l'ho confrontato alla softmax di PyTorch (ed era PERFETTA al
            # PRIMO TENTATIVO) e riadattato
            x.grad += CudaTensor(output.data * (chain_rule_tensor.data - x._device.sum(chain_rule_tensor.data * output.data, axis=self.dim, keepdims=True)))

            if not x.is_leaf:
                x.grad_fn(x.grad)
                x.grad = None
                x.grad_fn = None

        if x.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.grad_fn = _softmax_backward
        return output
    

class Softmax_Spam_Numpy:
    def __init__(self, dim=None) -> None:
        self.dim = dim
        self.grad_fn = None


    def __call__(self, x: CudaTensor) -> CudaTensor:

        def _calc_softmax(x_data: CudaTensor.data):
            # questa singola linea di codice previene overflow senza modificare
            # l'efficacia del gradient.
            shiftx = x_data - x._device.max(x_data)
            exps = x._device.exp(shiftx)
            output = exps / x._device.sum(exps)

            # fa schifo, ma è l'unica
            # https://stackoverflow.com/questions/73836765/softmax-and-its-derivative-along-an-axis
            def _softmax_backward(chain_rule_tensor: CudaTensor) -> None:
                '''
                la derivata della softmax è una matrice. Ogni colonna rappresenta una sua
                derivata completa w.r.t. un input
                '''
                shiftx = x_data - x._device.max(x_data) #x.max()
                exps = x._device.exp(shiftx)
                softmax_x = exps / x._device.sum(exps)

                SM = softmax_x.reshape((-1,1))
                jacobian = x._device.diagflat(softmax_x) - np.dot(SM, SM.T)

                # adesso è corretto, MA la matmul dà errore perché non sa
                # che dim prendere
                print(softmax_x.shape)
                print(jacobian.shape)
                print(chain_rule_tensor.data.shape)
                print(chain_rule_tensor.data.swapaxes(0,1).shape)

                # la matmul permette di passare da jacobian a vettore per
                # continuare la backward (oltre a propagare correttamente il grad ricevuto)
                x.grad += CudaTensor(jacobian @ chain_rule_tensor.data)

                # NOTE: questa condizione andrebbe aggiunta in tutte le activation functions!
                if not x.is_leaf:
                    x.grad_fn(x.grad)

                    x.grad = None
                    x.grad_fn = None

            if x.requires_grad:
                # da qua in avanti, requires_grad non sarà più modificabile perché
                # non è più considerato leaf tensor
                # uso una variabile della classe per poter bassare la grad_fn
                self.grad_fn = _softmax_backward
            # qua posso returnare solo un ndarray perché
            # numpy riassembla nell'array dato
            return output
        
        output = x._device.apply_along_axis(_calc_softmax, self.dim, x)
        output = CudaTensor(output, requires_grad=x.requires_grad)
        output.grad_fn = self.grad_fn
        
        return output