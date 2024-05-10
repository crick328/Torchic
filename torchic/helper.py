import numpy as np
from torchic import Tensor, nn
import torchic.optim
from typing import Self
import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime
import random


class Metrics:
    def __init__(self):
        self.training_id = f'training-{datetime.datetime.now().strftime("%d/%m/%Y-%H_%M")}'
        self.t_accuracy = []
        self.t_loss = []
        self.e_accuracy = []
        self.e_loss = []
        self.training_time = time.time()

    def save_metrics(self, filename: str=None, dir: str="metrics"):
        # csv o json?
        if filename is None:
            filename = f"{self.training_id}.csv"

    def load_metrics(filename: str) -> Self:
        pass

    def plot_metrics(self):
        # plotta andamento metriche (con training_id)
        pass

    def _calc_train_time(self):
        self.training_time = time.time() - self.training_time

    def _resume_training():
        pass

    def _print_formatted_metrics(self,
                                curr_iter,
                                curr_epoch,
                                epochs,
                                batch_size,
                                n_batch,
                                acc,
                                loss,
                                is_eval: bool=False,
                                to_save: bool=False):
        
        print(f"Epoch {curr_epoch}/{epochs}, batch {(curr_iter/batch_size) :.0f}/{int(n_batch) :.0f}, accuracy: {acc :.1f}%, loss: {loss :.1f}%")
        
        if to_save:
            if not is_eval:
                self.t_accuracy.append(acc)
                self.t_loss.append(loss)
            else:
                self.e_accuracy.append(acc)
                self.e_loss.append(loss)


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
                        print_metrics: bool=True) -> None|Metrics:
    
    start_time = time.time()

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

            if print_metrics:
                # fallo inline possibilmente
                print(f"Epoch {epoch}/{epochs}, batch {(current_iter/batch_size) :.0f}/{int(len(train)/batch_size) :.0f}, loss: {(epoch_loss/current_iter*100):.1f}%, accuracy: {(correct_predictions/current_iter*100) :.1f}%")

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
            print(f"Eval accuracy: {eval_preds / len(eval) * 100 : .2f}%, Eval loss: {eval_loss / len(eval) * 100 : .2f}%")

        if print_metrics:
            epoch_min = (time.time() - t) / 60
            print(f"ETA: { epoch_min * (epochs - epoch) : .1f} minuti")

    print(f"Fine training! Tempo impiegato: {((time.time() - start_time)/60):.1f} minuti")


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
    
    fig.show()
    fig.waitforbuttonpress()

    if graph_name:
        fig.savefig(f"{graph_name}.png")
    

def show_matrix_values(w_matrix: Tensor, graph_name: str=None):
    fig, ax = plt.subplots(ncols=2)

    ax[0].matshow(w_matrix.data, cmap=plt.cm.Blues)
    ax[1].matshow(w_matrix.data, cmap=plt.cm.Spectral_r, interpolation='none', vmin=0, vmax=1)
    
    fig.show()
    fig.waitforbuttonpress()

    if graph_name:
        fig.savefig(f"{graph_name}.png")


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