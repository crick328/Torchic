import numpy as np
import warnings


try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import matplotlib
    import graphviz
    VISUAL_AVAILABLE = True
except ImportError:
    VISUAL_AVAILABLE = False


CONFIG = {
    "device": "cpu",
    "precision": 32,
    "use_cuda_if_available": False,
    "seed": None,
    "cuda": {
        "max_async_streams": 4,     # 0 = no use of streams
        "use_null": False,
        "time_threshold_ms": 5,
    },
    "grad_enabled": True,
    "mixed_precision": {
        "enabled": False,
        "dtype": np.float16
    }
}
# prendo anche qualcosa in piu' per ricordarmi che sono dati
# a mia disposizione
DEVICE_SPECS = {
    "totalGlobalMem": 0,        # mem della scheda grafica
    "sharedMemPerBlock": 0,     # mem condivisa a disposizione per i thread di un block
    "maxThreadsPerBlock": 0,
    "maxThreadsDim": (),
    "maxGridSize": (),
    "multiProcessorCount": 0,
    "concurrentKernels": 0,     # n max di kernel eseguibili in contemporanea
    "asyncEngineCount": 0,
    "computeCapability": 0      # dato importante; racchiude le capacita' tecniche della scheda
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
}


if CUPY_AVAILABLE:
    if cp.cuda.is_available():
        device_id = cp.cuda.Device().id

        specs = cp.cuda.runtime.getDeviceProperties(device_id)
        device_name = specs['name'].decode('utf-8')
        print(f"Found cuda capable device {device_name}")
        
        for key in specs:
            if key in DEVICE_SPECS:
                DEVICE_SPECS[key] = specs[key]

        if CONFIG["use_cuda_if_available"]:
            CONFIG["device"] = "cuda"
            print("GPU selected as default device.")
    else:
        warnings.warn(
                "CuPy is installed, but your GPU isn't cuda "
                "capable or you are missing the cuda library. "
            )
        CONFIG["device"] = "cpu"

def get_current_backend():
    return cp if CONFIG["device"] == "cuda" else np

def get_default_device() -> str:
    return CONFIG["device"]

def get_default_precision() -> int:
    return CONFIG["precision"]

def get_default_dtype() -> np.dtype:
    return CONFIG["dtype"]

def get_default_img_dir() -> np.dtype:
    return CONFIG["img_dir"]


def set_default_device(device: str) -> None:
    if device not in ("dynamic", "cpu", "cuda"):
        raise ValueError(
                f"Expected 'dynamic', 'cuda' or 'cpu', got {device} instead."
            )

    if device == "cupy" and not CUPY_AVAILABLE:
        raise RuntimeError(
            "Cannot set cuda as device because no cuda capable "
            "device was found."
        )
    if device == "dynamic":
        if not CUPY_AVAILABLE:
            warnings.warn(
                "Dynamic mode is unavailable without a cuda device. "
            )
            return
        else:
            warnings.warn(
                "Dynamic mode automatically inferes the device for each tensor "
                "based on the class of the received data (cp.ndarray = cuda, else "
                "cpu). Note that this mode requires consistency or the backprop "
                "process won't be fast. If you want to train and do not want to "
                "check every given data, force a global device."
            )
    CONFIG["device"] = device

def set_default_precision(precision: int) -> None:
    if precision == 16 or precision == 32 or precision == 64:
        CONFIG["precision"] = precision
    elif precision == 8:
        warnings.warn(
            "Precision 8 is supported only for integer "
            "data type. Some operations may return float, "
            "so, if error raises, force the global dtype to int8."
            )
        CONFIG["precision"] = precision
    else:
        raise ValueError(
            "Precision must be a power of 2 between 8 and 64, "
            f"received {precision} instead."
        )

#def set_default_dtype(type: dtype) -> None:
    #CONFIG["dtype"] = type

def set_default_img_dir(directory: str) -> None:
    CONFIG["img_dir"] = directory

def set_seed(seed: int) -> None:
    CONFIG["seed"] = seed
    
    np.random.seed(seed)
    if CUPY_AVAILABLE:
        cp.random.seed(seed)


# ==== CUDA settings ====
def get_max_async_streams() -> int:
    return CONFIG["cuda"]["max_async_streams"]

def get_use_null() -> bool:
    return CONFIG["cuda"]["use_null"]

def get_time_threshold_ms() -> float:
    return CONFIG["cuda"]["time_threshold_ms"]


def set_max_async_streams(i: int) -> None:
    CONFIG["cuda"]["max_async_streams"] = i

def set_use_null(flag: bool) -> None:
    CONFIG["cuda"]["use_null"] = flag

def set_time_threshold_ms(f: float) -> None:
    CONFIG["cuda"]["time_threshold_ms"] = f


def is_grad_enabled() -> bool:
    return CONFIG["grad_enabled"]

def set_grad_enabled(flag: bool) -> None:
    CONFIG["grad_enabled"] = flag

# ==== Mixed precision settings ====
def set_mixed_precision_dtype(dtype: cp.float16) -> None:
    CONFIG["mixed_precision"]["dtype"] = dtype

def get_mixed_precision_dtype() -> None:
    return CONFIG["mixed_precision"]["dtype"]

def is_mixed_precision_enabled() -> bool:
    return CONFIG["mixed_precision"]["enabled"]

def set_mixed_precision_enabled(flag: bool) -> None:
    CONFIG["mixed_precision"]["enabled"] = flag
