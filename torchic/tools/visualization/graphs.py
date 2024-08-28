from torchic.nn.layers.linears import Linear
from torchic.nn.module import Module
from torchic.nn.activations import ActivationFunction
from torchic.cuda_tensor import Tensor
from torchic.tools.helpers import rand
from graphviz import Digraph


# ogni nodo e' un tensore. Se op e' settata, e' un tensore leaf.
# Se backward e' True, e' un tensore frutto di una backward
class Node:
    def __init__(self, grad_fn, shape=None, is_leaf=False, requires_grad=False, next_functions=None) -> None:
        self.grad_fn = grad_fn
        self.shape = shape
        self.is_leaf = is_leaf
        self.requires_grad = requires_grad
        self.next_functions = next_functions

    def __eq__(self, other):
        if isinstance(other, Node):
            return (self.grad_fn == other.grad_fn and
                    self.shape == other.shape and
                    self.is_leaf == other.is_leaf and
                    self.requires_grad == other.requires_grad)
        return False

    def __hash__(self):
        return hash((self.grad_fn, self.shape, self.is_leaf, self.requires_grad))



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
def draw_graph(root_tensor, format='png', rankdir='TB', complete_format=True, layout='dot',  dpi=600.0):
    '''
    Legenda:
    - azzurro: tensore che possiede backward
    - verde: tensor che richiede grad
    - rosso: tensor che non richiede grad
    Leaf e non leaf si vede dalla struttura del grafo.
    '''
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
        nodes, bw_edges = trace_backward(root_tensor)
        #nodes.update(bw_nodes)
    else:
        raise RuntimeError(
            "Unable to trace the backward graph, the given root "
            "tensor does not have a backward function."
        )
    

    for node in nodes:
        dot.node(
            name=str(id(node)),
            label=_create_label([f"{key}: {value}" for key, value in node.__dict__.items()]) if complete_format else f"{node.grad_fn} {node.shape}",
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