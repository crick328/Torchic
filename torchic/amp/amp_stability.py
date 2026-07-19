# Toolkit to evaulate if a function can receive the @autocast decorator.

# tutte librerie standard
import warnings
import textwrap
import inspect
import re
import torchic



# se la funzione non fa parte di quelle classificate a mano, lo faccio
# in automatico basandomi sulle singole operazioni all'interno della funzione
# Accessibile e modificabile dall'utente
BASIC_OPS = {
    "safe": {
        # elementary op
        '+', '+=', 'add',
        '-', '-=', 'sub',
        '*', '*=', 'mul',
        # matrix op
        '@', 'matmul', 'dot', 'conv',
        'pooling', 'max', 'maximum'
        # si puo' aggiungere relu, gelu e layernom se scritte in modo stabile
    },
    "critical": {
        'log', 'sqrt',
        # risk of numerical cancellation
        '/', '/=', 'divide',
        # trig
        'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'tanh',
        # exponentials (risk of overflow)
        '**', 'pow', 'exp',
        # reductions
        'var', 'std', 'mean', 'sum'
        # softmax ci vuole a prescindere, anche se scritta in modo stabile
        # perche' exp() puo' andare in overflow
        # cross entropy
    },
    "ignore": {
        'register_backward_if_needed', 'ascontiguousarray'
    },
    "mixed_precision": {
        # lista di operazioni che vorresti eseguire a precisione ridotta.
        # Unica parte di questo dizionario che e' modificabile dall'utente
        '@', 'matmul', 'conv'
    }
}
# safe: relu, leaky relu, reshape, view, transpose
# non safe: layernorm, batchnorm, mean, sum, sqrt, rsqt, div, exp, log, sum, sigmoid

# non direttamente modificabili dall'utente, usate internamente per velocizzare
SAFE_OPS = set(BASIC_OPS['safe'])
CRITICAL_OPS = set(BASIC_OPS['critical'])
IGNORE_OPS = set(BASIC_OPS['ignore'])
MIXED_PRECISION_OPS = set(BASIC_OPS['mixed_precision'])



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


def is_safe_in_float16(func) -> int:
    '''
    Analyze `funct`'s source code looking for `SAFE_OPS` and `CRITICAL_OPS`.

    Args:
        funct (function): function to analyze

    Returns:
        int: value that represent the safeness of the function. It can be:
        - -1: contains _at least_ one `CRITICAL_OPS`.
        - 0: contains `SAFE_OPS` _and_ unclassified ops (which raises a warning).
        - 1: contans _only_`SAFE_OPS`.
        - 2: contains _only_ `SAFE_OPS` _and_ **should be executed in mixed precision**.
    
    A warning is raised if there are one or more operations that don't
    fall in neither `SAFE_OPS`, `CRITICAL_OPS` or `IGNORE_OPS`. Ideally you
    should classify them.
    '''

    # restituisce una stringa contenente il codice della funzione
    source_code = inspect.getsource(func)
    # rimuovo indentazioni extra (necessarie se la funzione è parte di una classe)
    source_code = textwrap.dedent(source_code)
    #print(dedented_source)

    # 2. Rimuovi i blocchi multilinea: '''...''' o """..."""
    source_code = re.sub(r"(?s)'''(.*?)'''", "", source_code)  # rimuove blocchi con '''
    source_code = re.sub(r'(?s)"""(.*?)"""', "", source_code)  # rimuove blocchi con """
    
    # rimuovo la prime due righe (decoratore e definizione della funzione)
    # e righe che iniziano con "# ..."
    source_code = "\n".join(
        line for line in source_code.splitlines()[2:]
        if not line.strip().startswith("#")
    )

    # rimuovi commenti in-linea (es. "x = 5  # commento")
    source_code = re.sub(r"#.*", "", source_code)
    #print(source_code)


    operations1 = set(extract_operations(source_code))
    operations = set(operations1)
    # assert messo perche' sono curioso di sapere se il secondo set serve
    assert operations == operations1
    
    unclassified_ops = operations - SAFE_OPS - CRITICAL_OPS - IGNORE_OPS
    if unclassified_ops:
        print(f"Warning in {func.__qualname__}: trovate operazioni non classificate: {unclassified_ops}")
        #warnings.warn(f"Warning in {func.__qualname__}: trovate operazioni non classificate: {unclassified_ops}")

    #print(operations.intersection(set(BASIC_OPS['critical'])))
    # se e' presente anche solo un'operazione critica
    if operations.intersection(CRITICAL_OPS):
        return -1
    
    # se ha solo safe_ops
    if not unclassified_ops:
        if operations.issubset(MIXED_PRECISION_OPS):
            # quest'operazione contiene SOLO operazionic che voglio in mixed
            return 2
        return 1
    
    # se ha sia safe ops sia ops neutrali
    return 0



seen_functions = set()
# qua potrei aggiungere le funzioni selezionate dall'utente...?
whitelist = set()
blacklist = set()
to_lower_precision = set()

# sovrascrivo il file precedente
#with open("torchic/amp/_non_elegible.md", "w") as f:
  #f.write("(This file was auto-generated for dev purposes)\n# Non-elegible mixed precision functions:\n")

# non ho capito se devo scartare solo le operazioni sensibili o le loro intere funzioni

# i decorators sono dei wrapper che avvolgono la funzione presa in input
# e restituiscono una nuova funzione
# NOTE: il decoratore sostituisce la funzione con quella
# wrappata al lancio del codice!
def autocast_operation(funct):
    """
    Decorator used to mark a function that should be analyzed for
    mixed precision compatibility.

    Functions with @autocast_operation are not guaranteed to be executed in mixed precision.
    Internally, the amp module has a whitelist, a blacklist and an automatic classification system.
    Functions can be either manually added to each of the two lists or they can be evalued
    by the system, which chooses the classification based on the mathematical operations used
    inside the function.

    This piece of code is responsible for casting tensors to fp16 if 'funct' is elegbile,
    but it also has to convert back to fp32 if it isn't.

    Decoratore per applicare il casting agli argomenti delle operazioni.
    Intercetto la chiamata alla funzione e casto i suoi args prima che venga eseguita
    """
    # NOTE: il codice "di mezzo" interno a autocast_operation()
    # e esterno a casted_function() viene ESEGUITO SOLO ALL'AVVIO della
    # libreria! Di conseguenza, il check is_mixed_precision_enabled() va
    # messo nella funzione piu' interna o altrimenti questa logica non
    # partira' mai (perche' di default la mixed_precision e' off)
    #if not is_mixed_precision_enabled():
        #return funct

    # il __qualname__ e' il nome della funzione piu' la sua radice (serie di references ".")
    # una funzione con lo stesso nome ma presente in due parti differenti del codice
    # saranno trattate come diverse

    # come detto prima, questo codice e' eseguito all'avvio
    if funct.__qualname__ not in seen_functions:
        print(f"{funct.__qualname__} safe? {is_safe_in_float16(funct)}")
        # al momento mi baso solo sul classificatore automatico che
        # e' l'obiettivo finale, non voglio mica inserirle a mano.
        # Tuttavia, voglio anche ascoltare le richieste dell'utente

        safeness = is_safe_in_float16(funct)

        match safeness:
            case -1:
                blacklist.add((funct.__module__, funct.__qualname__))
            case 0:
                # NOTE: per ora ignoro le funzioni in whitelis, tanto
                # non verranno wrappate per ora

                # qua in teoria lo faccio in base a quanto voglio rischiare
                # Potrei anche metterle in graylist, ma per ora faccio cosi'
                #whitelist.add((funct.__module__, funct.__qualname__))
                pass
            case 1:
                #whitelist.add((funct.__module__, funct.__qualname__))
                pass
            case 2:
                to_lower_precision.add((funct.__module__, funct.__qualname__))

        # potrei avvisare l'utente che puo' anche togliere il decoratore
        # da questa funzione, tanto non la eseguiro' mai in mixed se le
        # regole rimangono queste. Allo stesso tempo, non voglio
        # soverchiarlo con 120 avvisi
        #with open("torchic/amp/_non_elegible.md", "a") as f:
            #f.write(f"- {funct.__qualname__}\n")

        seen_functions.add(funct.__qualname__)

    # idea: anziche' iterare args e kwargs, mettere in cache gli indici dei tensori
    # e cose strane, posso direttamente definirla cosi'! E' la svolta se funziona
    # (annota processo e soluzione di questo problema)
    # NOTE: NON mathca i nomi degli args! "arg2" potrebbe essere un
    # arg qualsiasi!
    
    return funct