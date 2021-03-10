import tensorflow as tf

from deephyper.nas.space.node import ConstantNode, VariableNode
from deephyper.nas.space.op.merge import AddByPadding, Concatenate
from deephyper.nas.space.op.op1d import Dense, Dropout, Identity
from deephyper.nas.space.auto_keras_search_space import AutoKSearchSpace
# from deephyper.search.nas.model.space.struct import AutoOutputStructure


def create_structure(input_shape=[(1, ), (942, ), (5270, ), (2048, )], output_shape=(1,), num_cells=2, *args, **kwargs):

    struct = AutoKSearchSpace(input_shape, output_shape, regression=True)
    input_nodes = struct.input_nodes

    output_submodels = [input_nodes[0]]

    for i in range(1, 4):
        cnode1 = ConstantNode(name='N', op=Dense(1000, tf.nn.relu))
        struct.connect(input_nodes[i], cnode1)

        cnode2 = ConstantNode(name='N', op=Dense(1000, tf.nn.relu))
        struct.connect(cnode1, cnode2)

        cnode3 = ConstantNode(name='N', op=Dense(1000, tf.nn.relu))
        struct.connect(cnode2, cnode3)

        output_submodels.append(cnode3)

    merge1 = ConstantNode(name='Merge')
    # merge1.set_op(Concatenate(struct, merge1, output_submodels))
    merge1.set_op(Concatenate(struct,  output_submodels))

    cnode4 = ConstantNode(name='N', op=Dense(1000, tf.nn.relu))
    struct.connect(merge1, cnode4)

    prev = cnode4

    for i in range(num_cells):
        cnode = ConstantNode(name='N', op=Dense(1000, tf.nn.relu))
        struct.connect(prev, cnode)


        merge = ConstantNode(name='Merge')
        # merge.set_op(AddByPadding(struct, merge, [cnode, prev]))
        merge.set_op(AddByPadding(struct, [cnode, prev]))

        prev = merge


    return struct

def test_create_structure():
    from random import random, seed
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf
    # seed(10)
    shapes = [
        (1, ),
        (942, ),
        (5270, ),
        (2048, )
    ]
    structure = create_structure(shapes, (1,))

    ops = [random() for i in range(structure.num_nodes)]

    print('num ops: ', len(ops))
    print('size: ', structure.size)
    print(ops)
    structure.set_ops(ops)
    structure.draw_graphviz('uno_baseline.dot')

    model = structure.create_model()

    model = structure.create_model()
    plot_model(model, to_file='uno_baseline.png', show_shapes=True)


    model.summary()

if __name__ == '__main__':
    test_create_structure()
