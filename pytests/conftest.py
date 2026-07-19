'''
Collections of utilities function for testing

Usa una fixture (passata come argomento) se la funzione deve:
- Gestire un Setup e un Teardown (Pulizia): Se devi connetterti a
  un database, creare un file temporaneo sul disco, o inizializzare
  una risorsa pesante che alla fine del test va assolutamente chiusa
  o cancellata.
- Isolare lo stato del test: Se vuoi che ogni test parta da una
  situazione "pulita" (es. un'istanza nuova di zecca di un modello
  o di una classe di configurazione).
- Fornire dati globali implicitamente: Se hai una configurazione che
  serve a tutti i file di test del progetto e non vuoi scriverne l'import
  in ognuno di essi (usando il file conftest.py).

Un esempio pratico per capire la differenza: Immagina di voler
testare il caricamento di pesi sulla GPU.
- Funzione normale (Import): calcola_loss(y_pred, y_true) 
  -> È matematica pura. Fai l'import.
- Fixture: Una funzione che alloca temporaneamente memoria sulla GPU,
  ti passa il modello e, alla fine del test (anche se il test fallisce!),
  svuota la cache di CUDA con torch.cuda.empty_cache() per evitare
  errori di Out Of Memory nei test successivi. Questa deve essere una fixture.
'''
import pytest
import sys
import torch
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 3.1")
import torchic


torch.set_default_dtype(torch.float64)
torchic.set_default_precision(64)