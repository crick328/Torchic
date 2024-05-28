from torchic.nn.layers import Linear
from torchic.nn.activation import ActivationFunction
from graphviz import Digraph


# ogni nodo e' un tensore. Se op e' settata, e' un tensore leaf.
# Se backward e' True, e' un tensore frutto di una backward
class Node:
    def __init__(self, name, shape=None, op=None, backward=False) -> None:
        self.name = name
        self.shape = shape
        self.op = op
        self.backward = backward

    # fondamentale per il set, altrimenti li reinserisce
    def __hash__(self) -> int:
        output = 0
        for value in self.__dict__.values():
            output += hash(value)
        return output



def create_graph(model, create_re=False):
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


# https://graphviz.org/doc/info/colors.html
def draw_graph(model, format='png', rankdir='TB', create_rel_edges: bool=False):
    assert rankdir in ['LR', 'TB']

    nodes, edges = create_graph(model, create_re=create_rel_edges)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    for node in nodes:
        dot.node(
            name=str(id(node)),
            label = create_label([f"{key}: {value}" for key, value in node.__dict__.items()]),
            shape="box" if not node.backward else "record",
            fillcolor="olivedrab3" if not node.op else "violetred2" if not node.backward else "steelblue2",
            # colore del bordo
            color="green" if not node.op else "firebrick4" if not node.backward else "blue",
            style="filled" if not node.op else "filled, rounded"
            )
    
    # qua semplicemente crei i legami e capisci chi unire in base agli ID
    for edge_group, edges in edges.items():
        for node1, node2 in edges:
            dot.edge(
                    tail_name=str(id(node1)),
                    head_name=str(id(node2))
                )
    
    #dot.view()
    dot.render('images/complete.gv', format=format, view=True)
    return dot



def create_label(data: list[str]):
    output = f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD>{data[0]}</TD></TR><TR><TD><TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" BGCOLOR="white" COLOR="gray">'

    for line in data[1:]:
        output = output + f"<TR><TD>{line}</TD></TR>"
    return output + "</TABLE></TD></TR></TABLE>>"

