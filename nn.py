from test13_tensor import Tensor
import numpy as np


# Bisogna aggiustare la loss function. Non può essere SE, ma così:
# https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
class SELoss:
    def __call__(self, predicted, target):
        # per ora è solo una potenza, basta solo fare il return dell'operazione con **
        '''
        Squared Error
        Input:
        - predicted: 1 dimensional Tensor object containing the output of the network
        - target: 1 dimensional Tensor object containing the desired output values
        '''
        #target = Tensor(target) if type(target) != Tensor else target
        loss = Tensor(np.power(predicted.data-target, 2))
        
        def _SELoss_backward(gradient):
            # non aggiorno il grad di loss perché la derivata di loss w.r.t. loss è 1
            predicted.grad_fn(Tensor(2*(predicted.data-target) * gradient.data))
            # rilascio il grafo computazionale in modo tale che il garbage collector possa cancellare le variabili inutili
            predicted.grad_fn = None

        # scritto per uniformità e flessibilità, si può valutare di rimuoverlo
        if predicted.requires_grad:
            loss.grad_fn=_SELoss_backward
        return loss


class Sigmoid:
    # ho bisogno che sia una classe per usare __repr__
    '''
    Input: 1 dimensional Tensor
    Output: 1 dimensional Tensor having same shape as input
    '''
    def __call__(self, x):
        output = Tensor(1/(1 + np.exp(-x)))

        def _sigmoid_backward(chain_rule_tensor):
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
    def __call__(self, x):
        output = Tensor(np.maximum(0, x.data))

        def _ReLU_backward(chain_rule_tensor):
            #partial_derivative = []
            #for value in x.data:
                #partial_derivative.append( 1 if value >= 0 else 0 )
            #nums = np.where(nums > 0, 1, nums)

            #print(partial_derivative)
            #x.grad += partial_derivative * chain_rule_tensor
            # crea overflow dopo una cinquantina di epochs
            #x.grad += Tensor(np.where(x.data > 0, 1, x.data) * chain_rule_tensor.data)
            # OCCHIO: QUESTO METODO MODIFICA X!
            np.put(x.data, x.data.nonzero()[0], 1)
            x.grad += Tensor(x.data * chain_rule_tensor.data)
            x.grad_fn(x.grad)

            x.grad = None
            x.grad_fn = None

        if x.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.grad_fn = _ReLU_backward
        return output

class Softmax:
    def __call__(self, x):
        output = []
        # questa singola linea di codice previene overflow senza modificare
        # l'efficacia del gradient.
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        output = Tensor(exps / np.sum(exps))

        def _softmax_backward(chain_rule_tensor):
            '''
            la derivata della softmax è una matrice. Ogni colonna rappresenta una sua
            derivata completa w.r.t. un input
            '''
            shiftx = x - np.max(x)
            exps = np.exp(shiftx)
            #denominatore = np.sum(np.exp(shiftx))
            denominatore = np.sum(exps)
            jacobian = []

            for i in range(0, len(x.data)):
                vector = []
                #softmax_i = np.exp(shiftx[i])/denominatore
                softmax_i = exps[i]/denominatore

                for j in range(0, len(x.data)):
                    if i == j:
                        # d_S(x[i])/d_x[i] = S(i)*(1-S(i))
                        vector.append( (softmax_i) * (1 - softmax_i) )
                    else:
                        # d_S(x[j])/d_x[i] = -S(j)*S(i)
                        vector.append( -(softmax_i)*(exps[j]/denominatore))
                        pass

                jacobian.append(vector) #* chain_rule_tensor)

            # la matmul permette di passare da jacobian a vettore per
            # continuare la backward (oltre a propagare correttamente il grad ricevuto)
            x.grad += Tensor(jacobian).matmul(chain_rule_tensor)
            x.grad_fn(x.grad)

            x.grad = None
            x.grad_fn = None

        if x.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.grad_fn = _softmax_backward
        return output   