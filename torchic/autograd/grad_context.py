from torchic.settings import set_grad_enabled, is_grad_enabled



class GradContextManager:
    def __init__(self):
        self.prev_state = is_grad_enabled()

    # entrambi i ctx reimpostano lo stato trovato
    def __exit__(self, exc_type, exc_val, exc_tb):
        set_grad_enabled(self.prev_state)


class no_grad(GradContextManager):
    '''
    ctx used to temporarily disable autograd
    '''
    def __enter__(self):
        set_grad_enabled(False)

class enable_grad(GradContextManager):
    def __enter__(self):
        set_grad_enabled(True)