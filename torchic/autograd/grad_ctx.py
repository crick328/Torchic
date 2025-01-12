from torchic.settings import set_grad_enabled, is_grad_enabled
import functools


class GradContextManager:
    def __init__(self):
        self.prev_state = is_grad_enabled()

    # entrambi i ctx reimpostano lo stato trovato
    def __exit__(self, exc_type, exc_val, exc_tb):
        set_grad_enabled(self.prev_state)


class no_grad(GradContextManager):
    def __enter__(self):
        set_grad_enabled(False)

    def __call__(self, func):
        print("EEE")
        @functools.wraps(func)
        def decorate_no_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_no_grad

class enable_grad(GradContextManager):
    def __enter__(self):
        set_grad_enabled(True)