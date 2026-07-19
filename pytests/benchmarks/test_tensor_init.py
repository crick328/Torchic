import pytest
import numpy as np
import torch
import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 3.1")
import torchic
#from ..utils import create_random_tensors, backward_and_compare

# pip install pytest-benchmark
# pip install pytest-benchmark[histogram]

@pytest.mark.benchmark
@pytest.mark.parametrize("_fast_init", [True, False])
def test_performance_istanza_tensor(benchmark, _fast_init):
    # 1. Prepariamo un mock di dati CuPy a monte per non calcolarlo nel test
    #_, my_t = create_random_tensors((100, 100), dtype=torchic.fp32)
    raw_data = np.ones((100,100), dtype=np.float32)
    
    # 2. Definiamo l'operazione da cronometrare
    def crea_tensore():
        return 
        
    # 3. Avviamo il benchmark
    benchmark(lambda : torchic.Tensor(raw_data, _fast_init=_fast_init))



import pytest
import cupy as cp

@pytest.mark.benchmark
@pytest.mark.parametrize("dtype", [cp.float32, cp.float16])
def test_cupy_gpu_init(benchmark, dtype):
    raw_data = cp.ones((2000, 2000), dtype=dtype)

    def run_cupy():
        res = raw_data @ raw_data
        cp.cuda.Device().synchronize()  # <--- CRITICAL for CuPy
        return res

    #benchmark(run_cupy)
    # 2. Explicitly force 10 warmup rounds before calculating stats over 100 rounds
    benchmark.pedantic(
        run_cupy, 
        iterations=1, 
        rounds=100, 
        warmup_rounds=10
    )