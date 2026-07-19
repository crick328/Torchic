import numpy as np

class MioTensore:
    def __init__(self, data, dtype=None):
        # Incapsuliamo un array NumPy interno
        self._array = np.array(data, dtype=dtype)
        self.dtype = self._array.dtype

    def __repr__(self):
        return f"MioTensore({self._array.tolist()}, dtype={self.dtype})"

    # IL DISPATCHER NATIVO DI NUMPY
    def __array_function__(self, func, types, args, kwargs):
        """
        Ogni volta che qualcuno passa questo oggetto a una funzione NumPy 
        (es. np.mean(mio_tensore)), NumPy si ferma e chiama QUESTO metodo.
        """
        print("CHIAMATO!")
        # 1. Controlliamo se sappiamo gestire la funzione richiesta (es. la media)
        if func is np.mean:
            # Recuperiamo l'array NumPy nascosto dentro l'oggetto del nostro utente
            array_estratto = args[0]._array 
            parent_dtype = array_estratto.dtype

            # 2. Eseguiamo il calcolo ad alta precisione (es. float64)
            risultato_preciso = np.mean(array_estratto, dtype=np.float64, **kwargs)

            # 3. Riconvertiamo al dtype originale (il downgrade intenzionale)
            risultato_convertito = risultato_preciso.astype(parent_dtype)

            # 4. Impacchettiamo di nuovo il risultato nel nostro boccale custom "MioTensore"
            return MioTensore(risultato_convertito)
        
        # Se è una funzione che non abbiamo implementato, passiamo la palla a NumPy standard
        return NotImplemented
    
    def __array_ufunc__(self, *args, **kwargs):
        print("UIB")
    
    def mean(self):
        return self._array.mean()

# --- VERIFICA ---
t = MioTensore([1.2, 1.2], dtype=np.float32)
print("Input:", t)

# Usiamo la funzione standard di NumPy direttamente sul nostro oggetto custom!
risultato = t.mean()

print("Risultato della media:", risultato)