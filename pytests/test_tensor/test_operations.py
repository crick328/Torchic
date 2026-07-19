# file e funzioni da testare devono iniziare con test_
import pytest
from utils import create_random_tensors, backward_and_compare

# NOTE: se i parametri dichiarati con .parametrize non sono presenti nella
# firma del test o c'e' qualche altro problemino del genere, PyTest ignorera'
# completamente l'intero file nella sezione "Testing" di VSCode, dove
# NON comparira' proprio! => Se non trovi un file di test, controlla
# che tutti i parametri siano in regola. Al massimo forza il run da terminale
# in modo che ti dia errore e capisci cosa non gli piace.
# Fai cd pytests e poi pytest

# ===== MATMUL =====
@pytest.mark.parametrize("shape_a, shape_b", [
    ((5,), (5, 3)),    # Caso 1: vettore @ matrice
    ((5, 3), (3,)),    # Caso 2: matrice @ vettore
    ((5, 3), (3, 4)),  # Caso 3: matrice @ matrice
])
def test_matmul_variants(shape_a, shape_b):
    """Testa le varie combinazioni di moltiplicazione tra matrici e vettori"""
    
    # Genera i tensori in base alla forma passata dal parametro
    torch_a, my_a = create_random_tensors(shape_a)
    torch_b, my_b = create_random_tensors(shape_b)

    torch_res = torch_a @ torch_b
    my_res = my_a @ my_b

    backward_and_compare(
        [torch_res, torch_a, torch_b],
        [my_res, my_a, my_b]
    )

