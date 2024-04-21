import numpy as np
import cupy as cp
from replica import CudaTensor



def get_classes(labels: list):
    '''
    Mi dai in input tutte le annotazioni del dataset e io tiro
    fuori solo le classi ordinate.
    Es: labels = [0,1,4,2,3,1,1,4,5,3,1,2,0]
        return [0,1,2,3,4,5]
    '''
 
    # insert the list to the set
    list_set = set(labels)
    # convert the set to the list
    classes = list(list_set)
    classes.sort()
    return classes


def create_ohe_table(classes: list[int|str]) -> dict[int|str: str]:
    '''
    restituisco un dict del tipo:
    ohe_dict = {
        0: [1,0,0],
        1: [0,1,0],
        2: [0,0,1]
    }
    dove le keys sono prese da classes
    '''
    # crea una matrice identità e associo le righe alle labels
    ohe_dict = {}
    ohe_labels = np.identity(len(classes))

    for i, row in enumerate(ohe_labels):
        ohe_dict[classes[i]] = row

    return ohe_dict


def labels_to_OHE(labels: list, device: str="cuda") -> list[CudaTensor]:
    CLASSES = get_classes(labels)
    OHE_DICT = create_ohe_table(CLASSES)

    output = []
    for label in labels:
        output.append(CudaTensor(OHE_DICT[label], device=device))
    return output


def memory_estimator(cls, data_type, dim):
    '''
    cls: classe (np.array, cp.array, ecc.)
    data_type: float32, float16, np.float32, ecc.
    dim: 0, (1,1), (4,1), (30,30)

    all'interno devo avere un dict in cui è segnato il peso
    di base di ogni classe e tipo

    output: memoria occupata
    '''
    pass