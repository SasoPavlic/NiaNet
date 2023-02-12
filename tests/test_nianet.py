import pandas as pd
import torch.optim.radam
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nianet.autoencoder import Autoencoder
import torch.nn.functional as F
from nianet import __version__
import tomli
import yaml


def test_toml_version():
    import os
    print(os.curdir)
    with open("pyproject.toml", mode="rb") as fp:
        try:
            config = tomli.load(fp)
        except tomli.TOMLDecodeError as exc:
            print(exc)
    assert __version__ == config['tool']['poetry']['version']


def test_citation_version():
    with open("CITATION.cff", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    assert __version__ == config['version']


def test_solution_encoding():
    dataset = load_diabetes()
    df_dataset = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    df_dataset['target'] = pd.Series(dataset.target)

    target = df_dataset.loc[:, df_dataset.columns == 'target']
    data = df_dataset.loc[:, df_dataset.columns != 'target']
    data = StandardScaler().fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=1234)
    dataset_shape = X_train.shape

    solution = [0.8841286965, 0.2160772639, 0.5232306181, 0.4203579002, 0.0, 0.1086783096, 0.5137225955]
    model = Autoencoder(solution, dataset_shape)

    assert model.shape == 'A-SYMMETRICAL'
    assert model.layer_step == 2
    assert model.layers == 2
    assert model.activation == F.rrelu
    assert model.epochs == 110
    assert model.learning_rate == 0.10900000000000008
    assert type(model.optimizer) == torch.optim.RAdam
    assert model.bottleneck_size == model.encoding_layers[0].out_features
