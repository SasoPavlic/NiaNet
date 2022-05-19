import torch
import random
import numpy as np
import torch.nn.functional as F

from torch import nn


class Autoencoder(nn.Module):

    def __init__(self, solution, dataset_shape):
        super(Autoencoder, self).__init__()

        self.encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()
        self.bottleneck_size = 0
        self.shape = self.get_shape(solution[0])
        self.layer_step = self.get_layer_step(solution[1], dataset_shape)
        self.layers = self.get_layers(solution[2], self.layer_step, dataset_shape)
        self.activation = self.get_activation(solution[3])
        self.epochs = self.get_epochs(solution[4])
        self.learning_rate = self.get_learning_rate(solution[5])

        self.generate_autoencoder(self.shape,
                                  self.layers,

                                  dataset_shape,
                                  self.layer_step)

        self.optimizer = self.get_optimizer(solution[6])

        print(
            f"Epochs:{self.epochs}\n"
            f"Shape:{self.shape}\n"
            f"Layer step:{self.layer_step}\n"
            f"Layers:{self.layers}\n"
            f"Activation function:{self.activation}\n"
            f"Encoder:{self.encoding_layers}\n"
            f"Decoder:{self.decoding_layers}\n"
            f"Bottleneck size:{self.bottleneck_size}\n"
            f"Optimizer: {self.optimizer}")

    def forward(self, x):
        """Flipping shape of tensors"""
        x = x.reshape(x.shape[1], x.shape[0])

        for layer in self.encoding_layers:
            x = self.activation(layer(x))

        for layer in self.decoding_layers:
            x = self.activation(layer(x))

        """Flipping back to original shape"""
        x = x.reshape(x.shape[1], x.shape[0])

        return x

    def get_shape(self, gene):
        gene = np.array([gene])
        bins = np.array([0.0, 0.5])
        inds = np.digitize(gene, bins)

        if inds[0] - 1 == 0:
            return "SYMMETRICAL"

        elif inds[0] - 1 == 1:
            return "A-SYMMETRICAL"

        else:
            raise ValueError(f"Value not between boundaries 0.0 and 1.0. Value is: {inds[0] - 1}")

    def get_layer_step(self, gene, dataset_shape):
        gene = np.array([gene])
        bins = []
        value = 1 / dataset_shape[1]
        step = value
        for col in range(0, dataset_shape[1]):
            bins.append(step)
            step += value
        bins[-1] = 1.01
        inds = np.digitize(gene, bins)
        return inds[0]

    def get_layers(self, gene, layer_step, dataset_shape):

        if layer_step == 0:
            max_layers = dataset_shape[1]
            return max_layers

        else:
            max_layers = round(dataset_shape[1] / layer_step)

        if max_layers == 1:
            return 1

        else:
            gene = np.array([gene])

            bins = []
            value = 1 / max_layers
            step = value
            for col in range(0, max_layers):
                bins.append(step)
                step += value
            bins[-1] = 1.01
            inds = np.digitize(gene, bins)

            return inds[0]

    def get_activation(self, gene):
        gene = np.array([gene])
        bins = np.array([0.0, 0.125, 0.25, 0.375, 0.500, 0.625, 0.750, 0.875, 1.01])
        inds = np.digitize(gene, bins)

        if inds[0] - 1 == 0:
            return F.elu

        elif inds[0] - 1 == 1:
            return F.relu

        elif inds[0] - 1 == 2:
            return F.leaky_relu

        elif inds[0] - 1 == 3:
            return F.rrelu

        elif inds[0] - 1 == 4:
            return F.selu

        elif inds[0] - 1 == 5:
            return F.celu

        elif inds[0] - 1 == 6:
            return F.gelu

        elif inds[0] - 1 == 7:
            return torch.tanh

        else:

            raise ValueError(f"Value not between boundaries 0.0 and 1.0. Value is: {inds[0] - 1}")

    def get_epochs(self, gene):
        gene = np.array([gene])
        bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.60, 0.7, 0.8, 0.9, 1.01])
        inds = np.digitize(gene, bins)

        return inds[0] * 10 + 100

    def get_learning_rate(self, gene):
        gene = np.array([gene])
        bins = []
        value = 1 / 1000
        step = value
        for col in range(0, 1000):
            bins.append(step)
            step += value
        bins[-1] = 1.01
        inds = np.digitize(gene, bins)
        lr = np.array(bins)[inds[0]]

        return lr

    def generate_autoencoder(self, shape, layers, dataset_shape, layer_step):

        if shape == "SYMMETRICAL":

            i = dataset_shape[1]
            z = dataset_shape[1] - layer_step

            while layers != 0:
                """Minimum depth reached"""
                if z < 1:
                    self.encoding_layers.append(nn.Linear(in_features=i, out_features=z + 1))
                    self.decoding_layers.insert(0, nn.Linear(in_features=z + 1, out_features=i))
                    self.bottleneck_size = z + 1
                    break

                self.encoding_layers.append(nn.Linear(in_features=i, out_features=z))
                self.decoding_layers.insert(0, nn.Linear(in_features=z, out_features=i))
                i = i - layer_step
                z = z - layer_step
                layers = layers - 1

            if len(self.encoding_layers) == 0:
                self.bottleneck_size = 0
            else:
                self.bottleneck_size = self.encoding_layers[-1].out_features

        elif shape == "A-SYMMETRICAL":
            i = dataset_shape[1]
            z = dataset_shape[1] - layer_step

            if layers == 1 or layers == 2:
                self.encoding_layers.append(nn.Linear(in_features=i, out_features=z))
                self.decoding_layers.insert(0, nn.Linear(in_features=z, out_features=i))

            if layers >= 3:
                layers_encoder = random.randint(1, layers)
                layers_decoder = layers - layers_encoder

                encoder_counter = layers_encoder
                decoder_counter = layers_decoder

                if layers_decoder == 0:
                    layers_encoder = layers_encoder - 1
                    layers_decoder = 1

                    encoder_counter = layers_encoder
                    decoder_counter = layers_decoder

                while encoder_counter != 0:

                    if z < 1:
                        self.encoding_layers.append(nn.Linear(in_features=i, out_features=z + 1))
                        self.bottleneck_size = z + 1
                        break

                    self.encoding_layers.append(nn.Linear(in_features=i, out_features=z))

                    i = i - layer_step
                    z = z - layer_step
                    encoder_counter = encoder_counter - 1

                while decoder_counter != 0:

                    if layers_decoder == 1:
                        self.decoding_layers.insert(0, nn.Linear(in_features=i, out_features=dataset_shape[1]))
                        break

                    layer_step = int((dataset_shape[1] - i) / decoder_counter)  # Make more complex logic
                    last_i = i
                    i = i + layer_step
                    z = z + layer_step
                    decoder_counter = decoder_counter - 1

                    self.decoding_layers.append(nn.Linear(in_features=last_i, out_features=i))

            if len(self.encoding_layers) == 0:
                self.bottleneck_size = 0
            else:
                self.bottleneck_size = self.encoding_layers[-1].out_features

    def get_optimizer(self, gene):
        gene = np.array([gene])
        bins = np.array([0.0, 0.167, 0.334, 0.50, 0.667, 0.834, 1.01])
        inds = np.digitize(gene, bins)

        """When AE does not have any layers"""
        if len(list(self.parameters())) == 0:
            return None

        if inds[0] - 1 == 0:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 1:
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 2:
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 3:
            return torch.optim.RAdam(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 4:
            return torch.optim.ASGD(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 5:
            return torch.optim.Rprop(self.parameters(), lr=self.learning_rate)

        else:
            raise ValueError(f"Value not between boundaries 0.0 and 1.0. Value is: {inds[0] - 1}")
