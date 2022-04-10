import copy
import math
import sys
import torch
import numpy as np

from niapy.task import Task, OptimizationType
from niapy.problems import Problem
from niapy.algorithms.basic import GreyWolfOptimizer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch import nn
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import mean_squared_error
from datetime import datetime
from nianet.autoencoder import Autoencoder

"""DEVICE USED IN TRAINING"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Custom made functions (scheduler_to, optimizer to). With the purpose to put tensors on a selected device"""
"""Pytorch --> Feature request --> Open: https://github.com/pytorch/pytorch/issues/8741"""


def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def create_dataset(df):
    sequences = df.astype(np.float32).tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features


def train_model(model, optimizer, train_dataset, n_epochs):
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    scheduler_to(scheduler, device)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000.0

    for epoch in range(1, n_epochs + 1):

        # print(f"Current epoch: {epoch}")
        """Training the model"""
        model = model.train()
        train_losses = []

        for seq_true in train_dataset:
            optimizer.zero_grad()
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            # https://pytorch.org/tutorials/beginner/former_torchies/autograd_tutorial.html
            seq_pred.requires_grad_(True)
            seq_true.requires_grad_(True)

            loss = criterion(seq_pred, seq_true)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        train_loss = np.mean(train_losses)
        history['train'].append(train_loss)

        if train_loss < best_loss:
            best_loss = train_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        # print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    model.load_state_dict(best_model_wts)
    return model.eval(), history


def predict(model, dataset):
    predictions, losses, original = [], [], []
    criterion = nn.L1Loss(reduction='sum').to(device)
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
            original.append(seq_true.cpu().numpy().flatten())
    return predictions, losses, original


class WorldProblem(Problem):

    def __init__(self, dimension, X_train, y_train, alpha=0.99):
        super().__init__(dimension=dimension, lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.iteration = 0

    def _evaluate(self, genome):
        print("=================================================================================================")
        print(f"ITERATION IS: {self.iteration}")
        self.iteration += 1

        X_train_sequence, seq_len, n_features = create_dataset(X_train)
        X_test_sequence, _, _ = create_dataset(X_test)

        dataset_shape = X_train.shape
        model = Autoencoder(genome, dataset_shape)

        """Punishing bad decisions"""
        if len(model.encoding_layers) == 0 or len(model.decoding_layers) == 0:
            fitness = sys.maxsize
            print(f"Fitness: {fitness}")
            return fitness

        model = model.to(device)

        torch.cuda.empty_cache()
        optimizer_to(model.optimizer, device)

        model, history = train_model(
            model,
            model.optimizer,
            X_train_sequence,
            n_epochs=model.epochs
        )

        # Known problem: https://discuss.pytorch.org/t/why-my-model-returns-nan/24329/5
        if math.isnan(min(history['train'])):
            fitness = sys.maxsize
            print(f"Fitness: {fitness}")
            return fitness

        else:
            predictions, losses, original = predict(model, X_test_sequence)
            MSE = mean_squared_error(original, predictions)
            fitness = (MSE * 1000) + (model.epochs ** 2) + (model.layers ** 3)
            print(f"Fitness: {fitness}")

            return fitness


if __name__ == '__main__':
    start = datetime.now().strftime("%H:%M:%S-%d/%m/%Y")
    print(f"Program start... {start}")

    DIMENSIONALITY = 7
    dataset = load_breast_cancer()
    data = dataset.data
    target = dataset.target
    feature_names = dataset.feature_names

    data = StandardScaler().fit_transform(data)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=1234)

    problem = WorldProblem(DIMENSIONALITY, X_train, y_train)
    task = Task(problem, max_iters=100, optimization_type=OptimizationType.MINIMIZATION)
    algorithm = GreyWolfOptimizer(population_size=10, seed=1234)
    best_genome, best_fitness = algorithm.run(task)

    model = Autoencoder(best_genome, X_train.shape)

    MODEL_PATH = f"model_{str(datetime.now())}_best_model.pth"
    torch.save(model, MODEL_PATH)
    print(f"Best fitness: {best_fitness}")
    print(f"Best AE genome: {model.__dict__}")

    end = datetime.now().strftime("%H:%M:%S-%d/%m/%Y")
    print(f"Program end... {end}")
