from torchic.cuda_tensor import Tensor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def show_confusion_matrix(predictions, ground_truth, normalize: bool=False, filename=None):
    actual = pd.Series(ground_truth, name='Actual')
    pred = pd.Series(predictions, name='Predicted')
    df_confusion = pd.crosstab(actual, pred)

    if normalize:
        df_confusion = df_confusion.div(df_confusion.sum(axis=1), axis="index")

    plt.matshow(df_confusion, cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()

    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)

    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

    if filename:
        plt.savefig(filename)
    plt.show()


def show_matrix_distribution(w_matrix: Tensor, graph_name: str=None):
    unique, counts = np.unique(w_matrix.data.round(2), return_counts=True)
    tmp_dict =dict(zip(unique, counts))

    plt.bar(tmp_dict.keys(), tmp_dict.values(), color='b')
    plt.title("Distrib")

    # NOTE: salva solo bianco... Forse il bar vuole altro
    # forse fixato: deve essere prima di show() perche' azzera tutto
    if graph_name:
        plt.savefig(graph_name)
    plt.show()


def show_out_connections(w_matrix: Tensor, graph_name: str=None):
    # 784 x 10 -> (10 x) 784 x 1
    device = w_matrix._device
    rows, cols = w_matrix.data.shape
    out_conn = device.split(w_matrix.data, cols, axis=1)
    target_shape = [int(device.sqrt(rows)), int(device.sqrt(rows))]

    fig, axs = plt.subplots(2, cols)
    fig.suptitle('Output connections')

    for i in range (0, len(out_conn)):
        # vmin e vmax sono gli estremi in cui normalizza i dati
        axs[0][i].matshow(out_conn[i].reshape(target_shape),
                    cmap=plt.cm.Spectral_r, interpolation='none', vmin=0, vmax=1)
        axs[0][i].set_title(i)

    for i in range (0, len(out_conn)):
        # vmin e vmax sono gli estremi in cui normalizza i dati
        axs[1][i].matshow(out_conn[i].reshape(target_shape), cmap=plt.cm.Blues)
        axs[1][i].set_title(i)

    if graph_name:
        fig.savefig(f"{graph_name}.png")
    
    fig.show()
    fig.waitforbuttonpress()
    

def show_matrix_values(w_matrix: Tensor, graph_name: str=None):
    fig, ax = plt.subplots(ncols=2)

    ax[0].matshow(w_matrix.data, cmap=plt.cm.Blues)
    ax[1].matshow(w_matrix.data, cmap=plt.cm.Spectral_r, interpolation='none', vmin=0, vmax=1)
    
    fig.show()
    fig.waitforbuttonpress()

    if graph_name:
        fig.savefig(f"{graph_name}.png")