# Toolkit to evaulate if a function can receive the @autocast decorator.

# tutte librerie standard
import textwrap
import inspect
import re



# se la funzione non fa parte di quelle classificate a mano, lo faccio
# in automatico basandomi sulle singole operazioni all'interno della funzione
BASIC_OPS = {
    "safe": {
        # elementary op
        '+', '+=', 'add',
        '-', '-=', 'sub',
        '*', '*=', 'mul',
        # matrix op
        '@', 'matmul', 'dot', 'conv',
        #'softmax', # perche' eseguita con shift trick (non  e' vero mai)
        'pooling', 'maximum'
    },
    "critical": {
        'log', 'sqrt',
        # risk of numerical cancellation
        '/', '/=', 'divide',
        # trig
        'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
        # exponentials (risk of overflow)
        '**', 'pow', 'exp',
        # reductions
        'var', 'std', 'mean', 'sum'
    }
}



def extract_operations(code, with_obj: bool=False):
    """
    Estrae operazioni e chiamate dal codice sorgente usando regex.
    """
    patterns = {
        # chiamata di un metodo di classe: es. x.data.max()
        "method_calls": r"(?!.*Backward\b)(\w+)\.(\w+)\(",
        # Funzione globale: es. Tensor(...)
        "function_calls": r"\b(?!Tensor\b|.*Backward\b)(\w+)\(",
        # Operatori: +, -, -=, @, etc.
        "operators": r"[-+*/@]=?|==|!=|<=?|>=?",
    }
    operations = []

    # per ogni tipo di pattern
    for name, pattern in patterns.items():
        # prendo tutte le stringhe che matchano col pattern corrente
        matches = re.findall(pattern, code)

        if name == "method_calls":
            # ho sia l'oggetto che il metodo chiamato
            if with_obj:
                operations.extend([f"{obj}.{method}" for obj, method in matches])
            else:
                operations.extend([method for _obj, method in matches])
        else:
            # ho solo la funzione
            operations.extend(matches)

    return operations


def is_safe_in_float16(func) -> bool:
    # restituisce una stringa contenente il codice della funzione
    source = inspect.getsource(func)
    # rimuovo indentazioni extra (necessarie se la funzione è parte di una classe)
    dedented_source = textwrap.dedent(source)
    print(dedented_source)

    operations = set(extract_operations(dedented_source))

    return operations