
from deephyper.benchmark import NaProblem
from candlepb.Uno.structs.uno_hyliu1 import create_structure
from candlepb.Uno.uno_baseline_keras2 import load_data_multi_array



Problem = NaProblem()

Problem.load_data(load_data_multi_array)

Problem.search_space(create_structure, num_cells=3)

Problem.hyperparameters(
    batch_size=64,
    learning_rate=0.001,
    optimizer='adam',
    num_epochs=1,
)

Problem.loss('mse')

Problem.metrics(['r2'])

Problem.objective('val_r2__last')


if __name__ == '__main__':
    print(Problem)
