from torchic.nn.layers.linears import Linear
from torchic.nn.activations import ActivationFunction
from graphviz import Digraph


# ogni nodo e' un tensore. Se op e' settata, e' un tensore leaf.
# Se backward e' True, e' un tensore frutto di una backward
class Node:
    def __init__(self, name, shape=None, op=None, backward=False, next_functions=None) -> None:
        self.name = name
        self.shape = shape
        self.op = op
        self.backward = backward
        self.next_functions = next_functions

    def __eq__(self, other):
        if isinstance(other, Node):
            return (self.name == other.name and
                    self.shape == other.shape and
                    self.op == other.op and
                    self.backward == other.backward)
        return False

    def __hash__(self):
        return hash((self.name, self.shape, self.op, self.backward))
    
    



def build_fw_graph(model):
    nodes, forward_edges = set(), set()
    prec_forward = Node(
            name="Input",
            shape=model[0].weights.shape()[0]
        )
                        #shape=list(model.__dict__.values())[0].weights.shape()[0])
    nodes.add(prec_forward)

    #https://youtu.be/MswxJw-8PvE?feature=shared&t=425
    

    for name, elem in model.__dict__.items():
        # linear e' la mia classe base dei layer
        if isinstance(elem, Linear):
            weights_node = Node(name=f"{name}.weights",
                                    shape=elem.weights.shape())
            matmul_node=Node(
                    name=f"{name}.partial_out",
                    shape=elem.weights.shape()[1],
                    op="matmul",
                )
            nodes.add(weights_node)
            nodes.add(matmul_node)

            forward_edges.add((prec_forward, matmul_node))
            forward_edges.add((weights_node, matmul_node))

            prec_forward=matmul_node

            if elem.has_bias:
                bias_node = Node(name=f"{name}.bias",
                                shape=elem.bias.shape())
                add_node=Node(
                    name=f"{name}.out",
                    shape=elem.bias.shape(),
                    op="add",
                )
                nodes.add(bias_node)
                nodes.add(add_node)

                forward_edges.add((prec_forward, add_node))
                forward_edges.add((bias_node, add_node))

                prec_forward=add_node


        elif isinstance(elem, ActivationFunction):
            node = Node(
                name=f"{name}_out",
                shape=prec_forward.shape,
                op=name,
            )
            #if node not in nodes:
            nodes.add(node)
            forward_edges.add((prec_forward, node))
            prec_forward=node

    return nodes, forward_edges


def build_bw_graph(tensor):
    nodes, edges = set(), set()
    acc_grad_counter = 0

    def _build(tensor, prev_node=None):
        # NOTE: molto brutto!
        # Riferisce alla variabile acc_grad_counter definita nella funzione esterna
        nonlocal acc_grad_counter
        f_string = "[ <BR/>"

        for elem in tensor.grad_fn.next_elems():
            f_string += f"{elem.grad_fn}<BR/>"
        #print(e1.grad_fn, e2.grad_fn)
        #next_fn = list(e.grad_fn for e in tensor.grad_fn.next_elems())
        #print(nf for nf in next_fn)
        node = Node(
                    name=f"{tensor.grad_fn.__str__()} Grad",
                    shape=tensor.data.shape,
                    op=f"{tensor.grad_fn.__str__()}",
                    backward=True,
                    next_functions=f"{f_string} ]"#f"{e1.grad_fn}, {e2.grad_fn}"
                )
        #print(f"{node.name}, {node not in nodes}, {hash(node)}")

        if node not in nodes:
            nodes.add(node)

            if prev_node:
                edges.add((prev_node, node))

            for next_tensor in tensor.grad_fn.next_elems():
                if next_tensor.grad_fn is not None:
                    _build(next_tensor, node)
                else:
                    acc_grad = Node(
                        # assegno un counter o altrimenti avrebbero lo stesso
                        # hash e non sarebbe aggiunto a nodes
                            name=f"AccumulateGrad{acc_grad_counter}",
                            shape=next_tensor.data.shape,
                            op=None,
                            backward=True
                        )
                    acc_grad_counter += 1
                    nodes.add(acc_grad)
                    edges.add((node, acc_grad))
        # se il nodo e' gia' nel set ma ha altre connessioni
        elif prev_node:
            # prendo la sua occorrenza nel set
            for n in nodes:
                if n == node:
                    # aggiungo la/le connessioni
                    edges.add((prev_node, n))
    _build(tensor)

    return nodes, edges


# https://graphviz.org/doc/info/colors.html
def draw_graph(model=None, root_tensor=None, format='png', rankdir='TB'):
    # se il modello ha fatto un forward, plotta anche il grafo di backward
    assert rankdir in ['LR', 'TB']

    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    if model:
        nodes, edges = build_fw_graph(model)
    else:
        nodes, edges = set(), set()

    if root_tensor and root_tensor.grad_fn is not None:
        bw_nodes, bw_edges = build_bw_graph(root_tensor)
        
        nodes.update(bw_nodes)
        #for edge in bw_edges:
            #edges.add(edge)
    

    for node in nodes:
        dot.node(
            name=str(id(node)),
            label = create_label([f"{key}: {value}" for key, value in node.__dict__.items()]),
            shape="box" if not node.backward else "record",
            fillcolor="olivedrab3" if not node.op else "violetred2" if not node.backward else "steelblue2",
            # colore del bordo
            color="green" if not node.op else "firebrick4" if not node.backward else "blue",
            style="filled" if not node.op else "filled, rounded",
            dir="back"
            )
    
    # qua semplicemente crei i legami e capisci chi unire in base agli ID
    for node1, node2 in edges:
        dot.edge(
                tail_name=str(id(node1)),
                head_name=str(id(node2))
            )
    for node1, node2 in bw_edges:
        dot.edge(
                tail_name=str(id(node1)),
                head_name=str(id(node2)),
                #dir="back"
                style="dashed",
                color="darkgray"
            )
    
    #dot.view()
    dot.render('images/complete.gv', format=format, view=True)
    return dot



def create_label(data: list[str]):
    output = f'<<TABLE ALIGN="LEFT" BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD ALIGN="LEFT">{data[0]}</TD></TR><TR><TD><TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" BGCOLOR="white" COLOR="gray">'

    # se non ha next_functions, non metterle nemmeno
    #if data[-1] == "next_functions: None":
        #data = data[:-1]

    for line in data[1:]:
        output = output + f"<TR><TD ALIGN='LEFT'>{line}</TD></TR>"
    return output + "</TABLE></TD></TR></TABLE>>"



































def create_graph_old(model, create_re=False):
    nodes = []
    ADD_REL_EDGES = create_re
    forward_edges, backward_edges, relation_edges = set(), set(), set()

    prec_forward = Node(name="Input",
                        shape=list(model.__dict__.values())[0].weights.shape()[0])
    prec_backward=None
    nodes.append(prec_forward)

    #https://youtu.be/MswxJw-8PvE?feature=shared&t=425
    

    for name, elem in model.__dict__.items():
        # linear e' la mia classe base dei layer
        if isinstance(elem, Linear):
            weights_node = Node(name=f"{name}.weights",
                                    shape=elem.weights.shape())
            matmul_node=Node(
                    name=f"{name}.partial_out",
                    shape=elem.weights.shape()[1],
                    op="matmul",
                )
            matmul_back_node=Node(
                name=f"{name}.weights.grad",
                shape=elem.weights.shape()[0],
                op=f"{name}.weights._matmul_backward",
                backward=True
            )
            nodes.append(weights_node)
            nodes.append(matmul_node)
            nodes.append(matmul_back_node)

            forward_edges.add((prec_forward, matmul_node))
            forward_edges.add((weights_node, matmul_node))

            if prec_backward:
                backward_edges.add((matmul_back_node, prec_backward))

            if ADD_REL_EDGES:
                relation_edges.add((matmul_back_node, weights_node))
                relation_edges.add((matmul_node, matmul_back_node))

            prec_forward=matmul_node
            prec_backward=matmul_back_node

            if elem.weights.requires_grad:
                acc_grad_node=Node(
                        name="matmul grad",
                        shape=elem.weights.shape(),
                        op="_accumulate_grad",
                        backward=True
                    )
                nodes.append(acc_grad_node)
                backward_edges.add((matmul_back_node, acc_grad_node))

                if ADD_REL_EDGES:
                    relation_edges.add((acc_grad_node, weights_node))
            
            if elem.has_bias:
                bias_node = Node(name=f"{name}.bias",
                                shape=elem.bias.shape())
                add_node=Node(
                    name=f"{name}.out",
                    shape=elem.bias.shape(),
                    op="add",
                )
                add_back_node=Node(
                    name=f"{name}.bias.grad",
                    shape=elem.bias.shape(),
                    op=f"{name}.bias._add_backward",
                    backward=True
                )
                nodes.append(bias_node)
                nodes.append(add_node)
                nodes.append(add_back_node)

                forward_edges.add((prec_forward, add_node))
                forward_edges.add((bias_node, add_node))

                backward_edges.add((add_back_node, prec_backward))

                if ADD_REL_EDGES:
                    relation_edges.add((add_back_node, bias_node))
                    relation_edges.add((add_node, add_back_node))

                prec_forward=add_node
                prec_backward=add_back_node

                if elem.bias.requires_grad:
                    acc_grad_node=Node(
                                name="bias grad",
                                shape=elem.bias.shape(),
                                op="_accumulate_grad",
                                backward=True
                            )
                    nodes.append(acc_grad_node)
                    backward_edges.add((add_back_node, acc_grad_node))
                    if ADD_REL_EDGES:
                        relation_edges.add((acc_grad_node, bias_node))

        elif isinstance(elem, ActivationFunction):
            node = Node(
                name=f"{name}_out",
                shape=prec_forward.shape,
                op=name,
            )
            backward_node = Node(
                name=f"{name}.grad",
                shape=prec_forward.shape,
                op=f"_{name}_backward",
                backward=True
            )
            #if node not in nodes:
            nodes.append(node)
            nodes.append(backward_node)

            forward_edges.add((prec_forward, node))
            backward_edges.add((backward_node, prec_backward))
            if ADD_REL_EDGES:
                relation_edges.add((node, backward_node))

            prec_forward=node
            prec_backward=backward_node

    return nodes, {"fw":forward_edges, "bw":backward_edges, "re":relation_edges}