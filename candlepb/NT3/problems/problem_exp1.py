from candlepb.NT3.models.candle_conv_mlp_1 import create_structure
from deephyper.benchmark import NaProblem

from candlepb.NT3.problems.load_data import load_data

import os
import numpy as np
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from keras.utils import np_utils



Problem = NaProblem()

Problem.load_data(load_data)

Problem.search_space(create_structure)

Problem.hyperparameters(
    batch_size=20,
    learning_rate=0.01,
    optimizer='adam',
    num_epochs=1,
    ranks_per_node=1
)

Problem.loss('categorical_crossentropy')

Problem.metrics(['acc'])

Problem.objective('val_acc__last')

# Problem.post_training(
#     num_epochs=1000,
#     metrics=['acc'],
#     model_checkpoint={
#         'monitor': 'val_acc',
#         'mode': 'max',
#         'save_best_only': True,
#         'verbose': 1
#     },
#     early_stopping={
#         'monitor': 'val_acc',
#         'mode': 'max',
#         'verbose': 1,
#         'patience': 20
#     }
# )

if __name__ == '__main__':
    print(Problem)
