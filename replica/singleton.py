import types

# non è necessario che sia usata esplicitamente,
# può anche essere chiamata/usata da .to() under the hood
class Device:
    def __init__(self, name: str = "cpu"):
        self._type = "cpu"
        self._xp = ""

    @property
    def type(self) -> str:
        return self._type
    
    @property
    def xp(self) -> types.ModuleType:
        return self._xp
    
    @type.setter
    def type(self, value):
        assert value == "cuda" or value == "cpu", ValueError(f"{value} is not a valid argument.")
        if value == "cuda":
            self.xp = cp
        else:
            self.xp = np

    @xp.setter
    def xp(self, value):
        # impedisco di cambiare la flag
        raise AttributeError ("attribute 'xp' of Device object is not writable.")

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Device, cls).__new__(cls)
        return cls.instance
    
    def __str__(self):
        # metodo chiamato con print()
        return self.type


import numpy as np
import cupy as cp

# la vera domanda è: ti serve un'istanza?

class StaticDeviceManager:
    op = np
    # è poco chiaro chiamare sdm.xp.matmul...
    # forse è meglio device.matmul...
    # sdm.engine.matmul
    # Device.op.matmul
    # dm.op.matmul

    @classmethod
    def device(cls, device):
        assert device == "cuda" or device == "cpu", ValueError(f"{device} is not a valid argument.")

        if device == "cuda":
            cls.op = cp
        else:
            cls.op = np

    @classmethod
    def current_device(cls):
        if cls.op == np:
            return "cpu"
        else:
            return "cuda"


def print_sdm():
    print(StaticDeviceManager.op)

def print_d():
    print(Device.type)

from replica import CudaTensor
Device.type="cuda"
a=CudaTensor([1,2], device="cpu")
