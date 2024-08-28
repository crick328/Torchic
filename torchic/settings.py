from numpy import dtype
import warnings


DEFAULT = {
    "device": "cpu",
    "precision": 32,
    "img_dir": "images",
}


def get_default_device() -> str:
    return DEFAULT["device"]

def get_default_precision() -> int:
    return DEFAULT["precision"]

def get_default_dtype() -> dtype:
    return DEFAULT["dtype"]

def get_default_img_dir() -> dtype:
    return DEFAULT["img_dir"]


def set_default_device(device: str) -> None:
    if device != "cpu" and device != "cuda":
        ValueError(f"Expected 'cuda' or 'cpu', got {device} instead.")              
    DEFAULT["device"] = device

def set_default_precision(precision: int) -> None:
    if precision == 16 or precision == 32 or precision == 64:
        DEFAULT["precision"] = precision
    elif precision == 8:
        warnings.warn(
            "Precision 8 is supported only for integer "
            "data type. Some operations may return float, "
            "so, if error raises, force the global dtype to int8."
            )
        DEFAULT["precision"] = precision
    else:
        raise ValueError(
            "Precision must be a power of 2 between 8 and 64, "
            f"received {precision} instead."
        )

#def set_default_dtype(type: dtype) -> None:
    #DEFAULT["dtype"] = type

def set_default_img_dir(dir: str) -> None:
    DEFAULT["img_dir"] = dir