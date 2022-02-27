import copy

import pandas as pd
import torch
from niapy.task import Task, OptimizationType
from niapy.problems import Problem, Alpine1
from niapy.algorithms.basic import GreyWolfOptimizer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import *
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from matplotlib import cm
import seaborn as sns

# our custom Problem classes
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from torch import nn, optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataset(df):
    sequences = df.astype(np.float32).tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features


def predict(model, dataset):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(device)
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses


def train_model(model, train_dataset, val_dataset, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    for epoch in range(1, n_epochs + 1):

        # print(f"Current epoch: {epoch}")
        """Training the model"""
        model = model.train()
        train_losses = []

        # TODO Add functionality to learn on batch level
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

        """Validating the model"""
        val_losses = []
        model = model.eval()

        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        # print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    model.load_state_dict(best_model_wts)
    return model.eval(), history


class Autoencoder(nn.Module):
    def __init__(self, layers):
        super(Autoencoder, self).__init__()

        self.encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()

        i = 30
        z = 25

        while layers != 0:
            """Minimum depth reached"""
            if z < 1:
                self.encoding_layers.append(nn.Linear(in_features=i, out_features=z + 1))
                self.decoding_layers.insert(0, nn.Linear(in_features=z + 1, out_features=i))
                break

            self.encoding_layers.append(nn.Linear(in_features=i, out_features=z))
            self.decoding_layers.insert(0, nn.Linear(in_features=z, out_features=i))
            i = i - 5
            z = z - 5
            layers = layers - 1

    def forward(self, x):
        x = x.reshape(x.shape[1], x.shape[0])

        new_shape = (len(x), 1)
        x = x.view(new_shape)

        for layer in self.encoding_layers:
            x = F.relu(layer(x))

        for layer in self.decoding_layers:
            x = F.relu(layer(x))

        x = x.reshape(x.shape[0], x.shape[1])

        return x


class WorldProblem(Problem):

    def __init__(self, dimension, X_train, y_train, alpha=0.99):
        super().__init__(dimension=dimension, lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.iteration = 0

    def _evaluate(self, x):
        print(f"ITERATION IS: {self.iteration}")
        self.iteration += 1
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0

        model = Autoencoder(num_selected)
        model = model.to(device)

        X_train_sequence, seq_len, n_features = create_dataset(X_train)
        X_test_sequence, _, _ = create_dataset(X_test)

        model, history = train_model(
            model,
            X_train_sequence,
            X_test_sequence,
            n_epochs=50
        )

        fitness = sum(history['val'])

        return fitness


if __name__ == '__main__':
    DIMENSIONALITY = 6
    dataset = load_breast_cancer()
    data = dataset.data
    target = dataset.target
    feature_names = dataset.feature_names

    data = StandardScaler().fit_transform(data)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=1234)

    problem = WorldProblem(DIMENSIONALITY, X_train, y_train)
    task = Task(problem, max_iters=10)
    algorithm = GreyWolfOptimizer(population_size=10, seed=1234)
    best_layers, best_fitness = algorithm.run(task)

    selected_layers = best_layers > 0.5
    print('Best number of layers in AE:', selected_layers.sum())