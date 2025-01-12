import torch
import torch.amp

with torch.no_grad():
    pass

#with torch.amp.autocast_mode.autocast

# Funzione di test
def test_precision_comparison():
    # Genera valori spinosi, valori piccoli e grandi
    values_float16 = torch.tensor([1e-6, 1e-1, 1e3, -1e-4, -1e2, 5.5, -7.2], dtype=torch.float16)
    values_float32 = values_float16.float()

    # 1. Operazione di shift
    shift_float16 = values_float16 - values_float16.max()
    shift_float32 = values_float32 - values_float32.max()

    print("\nShift Test:")
    print("Shift float16:", shift_float16)
    print("Shift float32:", shift_float32)

    # 2. Esponenziale
    try:
        exp_float16 = torch.exp(values_float16)
        exp_float32 = torch.exp(values_float32)
        print("\nExp Test:")
        print("Exp float16:", exp_float16)
        print("Exp float32:", exp_float32)
    except Exception as e:
        print("Exp Test - FAIL", e)

    # 3. Softmax
    try:
        softmax_float16 = torch.nn.functional.softmax(values_float16, dim=0)
        softmax_float32 = torch.nn.functional.softmax(values_float32, dim=0)
        print("\nSoftmax Test:")
        print("Softmax float16:", softmax_float16)
        print("Softmax float32:", softmax_float32)
    except Exception as e:
        print("Softmax Test - FAIL", e)

    # 4. Confronto finale e verifica overflow/underflow
    print("\nComparison Check:")
    max_diff_exp = torch.max(torch.abs(exp_float16 - exp_float32))
    max_diff_softmax = torch.max(torch.abs(softmax_float16 - softmax_float32))

    print(f"Max diff in exp: {max_diff_exp}")
    print(f"Max diff in softmax: {max_diff_softmax}")

# Eseguiamo il test
test_precision_comparison()
