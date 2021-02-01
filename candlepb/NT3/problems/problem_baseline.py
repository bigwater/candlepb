from candlepb.NT3.models.candle_conv_mlp_baseline import create_structure
from deephyper.benchmark import NaProblem

import pandas as pd
from candlepb.NT3.problems.load_data import load_data



Problem = NaProblem()

Problem.load_data(load_data)

Problem.search_space(create_structure)

Problem.hyperparameters(
    batch_size=20,
    learning_rate=0.01,
    optimizer='adam',
    num_epochs=1,
)

Problem.loss('categorical_crossentropy')

Problem.metrics(['acc'])

Problem.objective('val_acc__last')


if __name__ == '__main__':
    print(Problem)
