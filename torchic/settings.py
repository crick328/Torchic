from numpy import dtype, float32

# idea presa da https://stackoverflow.com/questions/30556857/creating-a-static-class-with-no-instances

DEFAULT = {
    "device": "cpu",
    "dtype": float32,
    "img_dir": "images"
}


def get_default_device() -> str:
    return DEFAULT["device"]

def set_defalt_device(device: str) -> None:
    if device != "cpu" and device != "cuda":
        ValueError(f"Expected 'cuda' or 'cpu', got {device} instead.")
                   
    DEFAULT["device"] = device


def get_default_dtype() -> dtype:
    return DEFAULT["dtype"]

def set_defalt_dtype(type: dtype) -> None:
    DEFAULT["dtype"] = type


def get_default_img_dir() -> dtype:
    return DEFAULT["img_dir"]

def set_defalt_img_dir(dir: str) -> None:
    DEFAULT["img_dir"] = dir