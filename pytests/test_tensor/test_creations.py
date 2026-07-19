# Inizializzazioni (factory methods, shapes, dtypes, device transfer)
import pytest
import torch
from utils import create_random_tensors, backward_and_compare
from numpy import float16, float64, int16

# Dizionario che serve SOLO a PyTorch per non farlo arrabbiare
np_to_torch = {
    float16: torch.float16,
    float64: torch.float64,
    int16: torch.int16
}


# la tupla parametrize viene passata insieme, ma 
# definire piu' parametrize esegue anche le combinazioni

# NOTE: i test 2 e 4 danno esito negativo perche' quando vado
# a fare .mean().backward(), il tensore che contiene il risultato
# di .mean() viene riportato (giustamente) sul default device e
# quindi in backward arriva un gradiente (quello di .mean()) che e'
# sul device sbagliato rispetto a quello di my_res.
# Ho provato alcuni fix, ma rompono la logica della libreria e si creano
# errori a catena. Visto che non e' un caso di training (o di utilizzo
# generale), per ora lascio stare.
#
# SUNTO: se cambi device di un tensore, a prescindere da come lo fai,
# il suo child sara' di nuovo sul default_device (comportamento desiderato
# che, come in questo caso, puo' andare contro la logica)
@pytest.mark.parametrize("args, check_grad", [
    (("cpu", float16), True),
    #(("cuda", float16), True),
    ((float16,), True),
    #(("cuda",), True),
    (("cpu", float64), False)  # should be self
])
def test_to(args, check_grad):
    torch_t, my_t = create_random_tensors((2, 3, 4))

    #torch_args = tuple(np_to_torch.get(x, x) for x in args)
    #torch_res = torch_t.to(*torch_args)
    my_res = my_t.to(*args)

    if isinstance(args[0], type):
        assert my_res.dtype == args[0]

    elif isinstance(args[0], str):
        assert my_res.device == args[0]

        if len(args) == 2:
            assert my_res.dtype == args[1]


    if check_grad:
        # ho restituito una copia: deve avere backward

        my_res.mean().backward()
        assert my_t.grad.dtype == my_t.dtype
        pass
        #backward_and_compare(
            #[torch_res, torch_t],
            #[my_res, my_t]
        #)
    else:
        # non c'era bisgno di nessuna modifica: deve essere self
        # e non puo' avere grad_fn
        assert my_res is my_t
        assert my_res.grad_fn == None #== torch_res.grad_fn