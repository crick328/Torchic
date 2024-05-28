import torchic.nn as nn
import torchic.nn.optim
from torchic.cuda_tensor import Tensor
from torchic.settings import DEFAULT
from typing import Self
import matplotlib.pyplot as plt
from os import path
import time
import datetime
import random


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

        axs[0].set_xlabel("N batch")
        axs[0].set_ylabel("%")
        axs[0].set_ylim([None, 100])

        axs[1].set_xlabel("N batch")
        axs[1].set_ylabel("%")
        axs[1].set_ylim([None, 100])

        # train
        axs[0].plot(self.t_accuracy, color="blue", label="train acc. %")
        axs[0].plot(self.t_loss, color="red", label="train loss %")
        axs[0].grid(True)

        # eval
        axs[1].plot(self.e_accuracy, color="cyan", label="eval acc. %")
        axs[1].plot(self.e_loss, color="orange", label="eval loss %")
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
                        optimizer: torchic.nn.optim.SGD,
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


def auto_eval(model, eval_data: list[Tensor], eval_labels: list[Tensor]):
    predicted_list = []
    ground_truth = []
    correct_predictions = 0
    total_eval_samples = len(eval_labels)

    for index in range(0, len(eval_data)):
        predicted = model(eval_data[index])

        if predicted.data.argmax() == eval_labels[index].data.argmax():
            correct_predictions += 1

        predicted_list.append(predicted.data.argmax())
        ground_truth.append(eval_labels[index].data.argmax())

    print(f"Prediction corrette: {correct_predictions}/{total_eval_samples}."
          f"Test accuracy: {correct_predictions/total_eval_samples*100}%")
    
    #if confusion_matrix:
        #show_confusion_matrix(predicted_list, ground_truth, "confusion_matrix.png")