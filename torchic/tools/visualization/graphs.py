from torchic.nn.layers.linears import Linear
from torchic.nn.module import Module
from torchic.nn.activations import ActivationFunction
from torchic.cuda_tensor import Tensor
import warnings
from torchic.tools.helpers import rand, ones
from torchic.settings import VISUAL_AVAILABLE

if VISUAL_AVAILABLE:
    from graphviz import Digraph


# ogni nodo e' un tensore. Se op e' settata, e' un tensore leaf.
# Se backward e' True, e' un tensore frutto di una backward
class Node:
    def __init__(self, grad_fn, output_shape=None, grad_shape=None, is_leaf=False, requires_grad=False, max_ref_count=None, next_functions=None) -> None:
        self.grad_fn = grad_fn
        self.fwd_shape = output_shape
        self.in_grad_shape = grad_shape
        self.is_leaf = is_leaf
        self.requires_grad = requires_grad
        self.max_ref_count = max_ref_count
        self.next_functions = next_functions

    # usato per reperire il nodo dal set
    def __eq__(self, other):
        if isinstance(other, Node):
            return (self.grad_fn == other.grad_fn and
                    self.fwd_shape == other.fwd_shape and
                    self.in_grad_shape == other.in_grad_shape and
                    self.is_leaf == other.is_leaf and
                    self.requires_grad == other.requires_grad and
                    self.max_ref_count == other.max_ref_count and
                    self.next_functions == other.next_functions
                    )
        return False

    def __hash__(self):
        return hash(
            (self.grad_fn, self.fwd_shape, self.in_grad_shape, 
             self.is_leaf, self.requires_grad, self.max_ref_count,
             self.next_functions
            )
        )


def dummy_backward(root_tensor):
    substack = [(root_tensor, ones(root_tensor.shape()).data)]
    nodes = set()
    edges = set()
    counter = 0
    seen_tensors_ids = {}

    def get_or_create_node(tensor, grad, _counter):
        if id(tensor) not in seen_tensors_ids:
            # NOTE: potrebbe dare problemi in __eq__
            output_shape = tensor.data.shape if tensor.data.shape != () else "scalar"
            if grad is not None:
            #print(tensor)
                grad_shape = grad.shape if grad.shape != () else "scalar"
            else:
                grad_shape = None

            if tensor.is_leaf:
                # creo un nodo di accumulazione
                grad_fn_name = f"AccumulateGrad{_counter}"
                max_ref_count = None
                next_functions = None

                _counter += 1
            else:
                grad_fn_name = f"{tensor.grad_fn}"
                max_ref_count = tensor.grad_fn.ref_count
                next_functions = _format_next_functs(tensor)

            node = Node(
                    grad_fn=grad_fn_name,
                    output_shape=output_shape,
                    grad_shape=grad_shape,
                    is_leaf=tensor.is_leaf,
                    requires_grad=tensor.requires_grad,
                    max_ref_count=max_ref_count,
                    next_functions=next_functions
                )
            
            seen_tensors_ids[id(tensor)] = node
        else:
            # devo prendere l'istanza esatta dal set o il grafo
            # si distrugge
            node = seen_tensors_ids[id(tensor)]

            for seen_node in nodes:
                if node is seen_node:
                    node = seen_node
                    break
        
        return node, _counter

    while len(substack) > 0:
        tensor, grad = substack.pop()
        node, counter = get_or_create_node(tensor, grad, counter)

        # se e' un nodo nuovo
        if node not in nodes:
            nodes.add(node)
        
        # stessa logica di backward
        if not tensor.is_leaf:
            grad_fn = tensor.grad_fn

            if grad_fn.ref_count > 1:
                grad_fn.ref_count -= 1
                continue 

            for parent, parent_grad in zip(grad_fn.parents(), grad_fn(grad)):
                if parent is not None:
                    parent_node, counter = get_or_create_node(parent, parent_grad, counter)
  
                    # creo la connessione tra i nodi
                    edges.add((node, parent_node))
                    nodes.add(parent_node)

                    # stesso append della backward
                    if parent_grad is not None:
                        substack.append((parent, parent_grad))
            tensor.grad_fn=None

    return nodes, edges




def trace_backward(tensor):
    nodes, edges = set(), set()
    acc_grad_counter = 0

    def _build(tensor, prev_node=None):
        # NOTE: molto brutto!
        # Riferisce alla variabile acc_grad_counter definita nella funzione esterna
        nonlocal acc_grad_counter

        node = Node(
                    grad_fn=f"{tensor.grad_fn}",
                    shape=tensor.data.shape,
                    is_leaf=tensor.is_leaf,
                    requires_grad=tensor.requires_grad,
                    next_functions=_format_next_functs(tensor)
                )

        if node not in nodes:
            nodes.add(node)

            if prev_node:
                edges.add((prev_node, node))

            for next_tensor in tensor.grad_fn.next_elems():
                # se il grafo continua
                if next_tensor.grad_fn is not None:
                    _build(next_tensor, node)
                else:
                    acc_grad = Node(
                        # assegno un counter o altrimenti avrebbero lo stesso
                        # hash e non sarebbe aggiunto a nodes
                            grad_fn=f"AccumulateGrad{acc_grad_counter}",
                            shape=next_tensor.data.shape,
                            is_leaf=next_tensor.is_leaf,
                            requires_grad=next_tensor.requires_grad,
                        )
                    acc_grad_counter += 1
                    nodes.add(acc_grad)
                    edges.add((node, acc_grad))
        # se il nodo e' gia' nel set ma ha altre connessioni
        elif prev_node:
            # prendo la sua occorrenza nel set
            for n in nodes:
                if n == node:
                    # aggiungo la connessione
                    edges.add((prev_node, n))
                    break
    _build(tensor)

    return nodes, edges


# https://graphviz.org/doc/info/colors.html
def draw_graph(root_tensor: Tensor, format='png', rankdir='TB', complete_format=True, layout='dot',  dpi=600.0):
    '''
    Legenda:
    - azzurro: tensore che possiede backward
    - verde: tensor che richiede grad
    - rosso: tensor che non richiede grad
    Leaf e non leaf si vede dalla struttura del grafo.
    '''
    if not VISUAL_AVAILABLE:
        warnings.warn(
            "Visual tools (matplotlib) are not available. "
            "Please install the necessary packages to use this module."
        )
        return

    # se il modello ha fatto un forward, plotta anche il grafo di backward
    assert rankdir in ['LR', 'TB']
    assert layout in ['dot', 'neato', 'fdp', 'sfdp', 'twopi', 'circo']

    #dot = Digraph(format=format, graph_attr={'rankdir': rankdir, 'dpi': dpi})
    dot = Digraph(format=format, engine=layout, graph_attr={
        'rankdir': rankdir,
        'dpi': str(dpi),
        'splines': 'true',
        #'nodesep': '0.2', # incrementa la separazione tra i nodi
        #'ranksep': '0.2', # incrementa la separazione tra i ranghi
        'concentrate': 'true',
        'pack': 'true',
        'packmode': 'clust',
        'margin': '0.1'   # riduzione del margine
    })
    #nodes, edges = set(), set()

    if root_tensor.grad_fn is not None:
        #nodes, bw_edges = trace_backward(root_tensor)
        nodes, bw_edges = dummy_backward(root_tensor)
        #nodes.update(bw_nodes)
    else:
        raise RuntimeError(
            "Unable to trace the backward graph, the given root "
            "tensor does not have a backward function."
        )
    

    for node in nodes:
        dot.node(
            name=str(id(node)),
            label=_create_label([f"{key}: {value}" for key, value in node.__dict__.items()]) if complete_format else f"{node.grad_fn} {node.in_grad_shape}",
            shape="record", #"box" if not node.backward else "record",
            #fillcolor="olivedrab3" if not node.is_leaf else "violetred2" if not node.backward else "steelblue2",
            #fillcolor="olivedrab3" if not node.is_leaf else "steelblue2",
            fillcolor="steelblue2" if node.next_functions else "olivedrab3" if node.requires_grad else "violetred2",
            # colore del bordo
            #color="green" if not node.is_leaf else "firebrick4" if not node.backward else "blue",
            #color="green" if not node.is_leaf else "blue",
            color="blue" if node.next_functions else "green" if node.requires_grad else "firebrick4",
            style="filled" if node.is_leaf else "filled, rounded",
            dir="back"
            )
    
    # qua semplicemente crei i legami e capisci chi unire in base agli ID
    #for node1, node2 in edges:
        #dot.edge(
                #tail_grad_fn=str(id(node1)),
                #head_grad_fn=str(id(node2))
            #)
    for node1, node2 in bw_edges:
        dot.edge(
                tail_name=str(id(node1)),
                head_name=str(id(node2)),
                #dir="back"
                style="dashed",
                color="darkgray"
            )
    
    #dot.view()
    dot.render('images/complete.gv', format=format, view=True, renderer="cairo")
    return dot


def _format_next_functs(tensor: Tensor):
    f_string = "[ <BR/>"

    for elem in tensor.grad_fn.next_elems():
        f_string += f"{elem.grad_fn}<BR/>"
    return f"{f_string} ]"

def _create_label(data: list[str]):
    output = f'<<TABLE ALIGN="LEFT" BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD ALIGN="LEFT">{data[0]}</TD></TR><TR><TD><TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" BGCOLOR="white" COLOR="gray">'

    # se non ha next_functions, non metterle nemmeno
    #if data[-1] == "next_functions: None":
        #data = data[:-1]

    for line in data[1:]:
        output = output + f"<TR><TD ALIGN='LEFT'>{line}</TD></TR>"
    return output + "</TABLE></TD></TR></TABLE>>"