import copy
import torch
import numpy as np

from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

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
