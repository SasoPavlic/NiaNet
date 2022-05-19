import math
import sys
import torch
import nianet.helper as helper

from niapy import Runner
from niapy.problems import Problem
from niapy.algorithms.basic import *
from niapy.algorithms.modified import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from nianet.autoencoder import Autoencoder


class AutoencoderArchitecture(Problem):

    def __init__(self, dimension, X_train, y_train, X_test, y_test, alpha=0.99):
        super().__init__(dimension=dimension, lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.alpha = alpha
        self.iteration = 0

    def _evaluate(self, solution):
        print("=================================================================================================")
        print(f"ITERATION IS: {self.iteration}")
        self.iteration += 1

        X_train_sequence, seq_len, n_features = helper.create_dataset(self.X_train)
        X_test_sequence, _, _ = helper.create_dataset(self.X_test)

        dataset_shape = self.X_train.shape
        model = Autoencoder(solution, dataset_shape)

        """Punishing bad decisions"""
        if len(model.encoding_layers) == 0 or len(model.decoding_layers) == 0:
            fitness = sys.maxsize
            print(f"Fitness: {fitness}")
            return fitness

        model = model.to(helper.device)

        torch.cuda.empty_cache()
        helper.optimizer_to(model.optimizer, helper.device)

        model, history = helper.train_model(
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
            predictions, losses, original = helper.predict(model, X_test_sequence)
            MSE = mean_squared_error(original, predictions)
            fitness = (MSE * 1000) + (model.epochs ** 2) + (model.layers ** 3) + (model.bottleneck_size * 100)
            print(f"Fitness: {fitness}")

            return fitness


def find_architecture(df_dataset):
    target = df_dataset.loc[:, df_dataset.columns == 'target']
    data = df_dataset.loc[:, df_dataset.columns != 'target']
    DIMENSIONALITY = data.shape[1]
    data = StandardScaler().fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=1234)

    runner = Runner(
        dimension=DIMENSIONALITY,
        max_evals=100,
        runs=2,
        algorithms=[
            ParticleSwarmAlgorithm(),
            DifferentialEvolution(),
            FireflyAlgorithm(),
            SelfAdaptiveDifferentialEvolution(),
            GeneticAlgorithm()
        ],
        problems=[
            AutoencoderArchitecture(DIMENSIONALITY, X_train, y_train, X_test, y_test)
        ]
    )

    print("=================================================================================================")
    final_solutions = runner.run(export='json', verbose=True)
    best_fitness = sys.maxsize
    best_solution = None

    for algorithm in final_solutions:
        fitness = final_solutions[algorithm]['AutoencoderArchitecture'][0][1]
        print(f"{algorithm}'s fitness: {fitness}")

        if best_fitness > fitness:
            best_fitness = fitness
            best_solution = final_solutions[algorithm]['AutoencoderArchitecture'][0][0]

    print("=================================================================================================")
    model = Autoencoder(best_solution, X_train.shape)
    print(f"Best fitness: {best_fitness}")
    print(f"Best AE genome: {model.__dict__}")

    return model
