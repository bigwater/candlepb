from candlepb.NT3.models.candle_conv_mlp_1 import create_structure
from deephyper.benchmark import NaProblem

import os
import numpy as np
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from keras.utils import np_utils


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')

    dataset_path = '/home/hyliu/work/candle_benchmark/nt3/datasets'
    
    print('load data from ', os.path.join(dataset_path, 'nt_train2.csv'), os.path.join(dataset_path, 'nt_test2.csv'))

    df_train = pd.read_csv(os.path.join(dataset_path, 'nt_train2.csv'), header=None).values.astype('float32')
    df_test  = pd.read_csv(os.path.join(dataset_path, 'nt_test2.csv') , header=None).values.astype('float32')

    print(df_train.shape, ' ', df_test.shape)
    # df train and df test are numpy array

    df_y_train = df_train[:,0].astype('int')
    df_y_test = df_test[:,0].astype('int')

    print(df_y_train.shape)
    
    Y_train = np_utils.to_categorical(df_y_train, 2)
    Y_test = np_utils.to_categorical(df_y_test, 2)
    
    print(Y_train.shape)

    seqlen = df_train.shape[1]
    
    df_x_train = df_train[:, 1:seqlen].astype(np.float32)
    print(df_train.dtype, ' ', df_x_train.dtype)
    df_x_test = df_test[:, 1:seqlen].astype(np.float32)

    X_train = df_x_train
    X_test = df_x_test
    
    scaler = MaxAbsScaler()
    mat = np.concatenate((X_train, X_test), axis=0)
    mat = scaler.fit_transform(mat)
    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]

    # print(type(X_train), ' ', type(Y_train), ' ', type(X_test), ' ', type(Y_test))

    return (X_train, Y_train), (X_test, Y_test)




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
