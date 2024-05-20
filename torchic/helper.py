import numpy as np
from torchic import Tensor, nn
import torchic.optim
from typing import Self
import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime
import random
from torchic.settings import DEFAULT
from os import path


class Metrics:
    def __init__(self, n_epochs: int):
        self.training_id = f'training-{datetime.datetime.now().strftime("%d/%m/%Y-%H_%M")}'
        self.t_accuracy = []
        self.t_loss = []
        self.e_accuracy = []
        self.e_loss = []
        self.training_time = time.time()
        self.curr_epoch = 1
        self.epochs = n_epochs

    def save_metrics(self, filename: str=None, dir: str="metrics"):
        # csv o json?
        # credo che json sia meglio per salvare le varie acc e loss
        if filename is None:
            filename = f"{self.training_id}.json"

    def load_metrics(filename: str) -> Self:
        pass

    def show_metrics(self, filename=None):
        # plotta andamento metriche (con training_id)
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.suptitle(self.training_id)

        # train
        axs[0].plot(self.t_accuracy, color="blue", label="train acc")
        axs[0].plot(self.t_loss, color="red", label="train loss")
        axs[0].grid(True)

        # eval
        axs[1].plot(self.e_accuracy, color="cyan", label="eval acc")
        axs[1].plot(self.e_loss, color="orange", label="eval loss")
        axs[1].grid(True)
        
        fig.legend(loc="upper left")

        if filename:
            fig.savefig(path.join(DEFAULT["img_dir"], filename))

        fig.show()
        fig.waitforbuttonpress()

    def _calc_train_time(self) -> None:
        '''Calc training time in minutes'''
        self.training_time = (time.time() - self.training_time) / 60

    def _resume_training():
        pass

    def _print_metrics(self,
                        acc: float,
                        loss: float,
                        eval_acc: float=None,
                        eval_loss: float=None) -> None:
        
        self.t_accuracy.append(acc)
        self.t_loss.append(loss)

        info = f"Epoch {self.curr_epoch}/{self.epochs}, train acc: {acc :.1f}%, train loss: {loss :.1f}%"
        self.curr_epoch += 1

        if eval_acc:
            self.e_accuracy.append(eval_acc)
            self.e_loss.append(eval_loss)
            info += f", eval accuracy: {acc :.1f}%, eval loss: {loss :.1f}%"

        print(info)


def supervised_training(model: nn.Sequential,
                        train: list[Tensor],
                        train_labels: list[Tensor],
                        loss_fn: nn.SELoss,
                        optimizer: torchic.optim.SGD,
                        epochs: int,
                        batch_size: int,
                        eval: list[Tensor]=None,
                        eval_labels: list[Tensor]=None,
                        checkpoint_interval: int=None,
                        print_metrics: bool=True) -> Metrics:
    
    metrics = Metrics(epochs)
    print("Training iniziato!")

    for epoch in range(1, epochs+1):
        epoch_loss = 0
        current_iter = 0
        correct_predictions = 0
        t = time.time()

        while current_iter < len(train):
            for i in range(current_iter, current_iter + batch_size):
                predicted = model(train[i])

                loss = loss_fn(predicted, train_labels[i])
                loss = loss.mean()
                loss.backward()

                epoch_loss += loss.data

                if predicted.data.argmax() == train_labels[i].data.argmax():
                    correct_predictions += 1

            optimizer.step()
            optimizer.zero_grad()

            current_iter += batch_size

        # rimescolo il dataset con le labels
        temp = list(zip(train, train_labels))
        random.shuffle(temp)
        train, train_labels = zip(*temp)
        # TODO: scrivi su foglio le stats dell'epoch
        # TODO: introduci gli optimizers seri!
        #optimizer.lr = optimizer.lr * 0.1

        if checkpoint_interval:
            pass

        if eval:
            eval_preds = 0
            eval_loss = 0

            for i in range(0, len(eval)):
                predicted = model(eval[i])
                loss = loss_fn(predicted, eval_labels[i])
                eval_loss += loss.mean().data

                if predicted.data.argmax() == eval_labels[i].data.argmax():
                    eval_preds += 1

            # se vuoi l'eval do per scontato che vuoi saperne anche le performance
            metrics._print_metrics(acc=correct_predictions/current_iter * 100,
                                loss=epoch_loss/current_iter * 100,
                                eval_acc=eval_preds/len(eval) * 100,
                                eval_loss=eval_loss/len(eval) * 100)

        elif print_metrics:
            metrics._print_metrics(acc=correct_predictions/current_iter * 100,
                                loss=epoch_loss/current_iter * 100)

        epoch_min = (time.time() - t) / 60
        print(f"ETA: { epoch_min * (epochs - epoch) : .1f} minuti")

    metrics._calc_train_time()
    print(f"Fine training! Tempo impiegato: {metrics.training_time :.1f} minuti")
    return metrics


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


def preprocess_images(images, device: str="cpu") -> list[Tensor]:
    """Returns the size of a tensor.

    Args:
        input: A `Tensor` or `SparseTensor`.
        name: A name for the operation (optional).
        optimize: if true, encode the size as a constant when possible.
        out_type: (Optional) The specified non-quantized numeric output type of the
        operation. Defaults to `tf.int32`.

    Returns:
        A `Tensor` of type `out_type`. Defaults to `tf.int32`.
    """
    '''
    ### Parameters
    1. images: ArrayLike
    2. device: str
        "cpu" or "cuda"
    ### Returns
    list of tensors
    :return: a list of Tensor containing flattened and normalized images.
    '''
    output = []
    for image in images:
        output.append(Tensor(image.flatten()/255, device=device))
    return output


def training_loop(model, train_data, loss, lr) -> None:
    # idea, pero' il modello deve essere passato by reference
    # banalmente dovrebbe bastare incapsularlo all'interno di un
    # oggetto mutabile come una lista
    pass


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


def labels_to_OHE(labels: list, device: str="cuda") -> list[Tensor]:
    CLASSES = get_classes(labels)
    OHE_DICT = create_ohe_table(CLASSES)

    output = []
    for label in labels:
        output.append(Tensor(OHE_DICT[label], device=device))
    return output


def memory_estimator(cls, data_manager, dim):
    '''
    cls: classe (np.array, cp.array, ecc.)
    data_manager: float32, float16, np.float32, ecc.
    dim: 0, (1,1), (4,1), (30,30)

    all'interno devo avere un dict in cui è segnato il peso
    di base di ogni classe e tipo

    output: memoria occupata
    '''
    pass