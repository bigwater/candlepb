import tensorflow as tf

from deephyper.nas.space.node import ConstantNode, VariableNode
from deephyper.nas.space.op.merge import AddByPadding, Concatenate
from deephyper.nas.space.op.op1d import Dense, Dropout, Identity
from deephyper.nas.space.auto_keras_search_space import AutoKSearchSpace


def add_mlp_op_(node):
    node.add_op(Dense(100, tf.nn.relu))
    node.add_op(Dense(100, tf.nn.tanh))
    node.add_op(Dense(100, tf.nn.sigmoid))
    node.add_op(Dense(500, tf.nn.relu))
    node.add_op(Dense(500, tf.nn.tanh))
    node.add_op(Dense(500, tf.nn.sigmoid))
    node.add_op(Dense(1000, tf.nn.relu))
    node.add_op(Dense(1000, tf.nn.tanh))
    node.add_op(Dense(1000, tf.nn.sigmoid))


def create_structure(input_shape=[(1, ), (942, ), (5270, ), (2048, )], output_shape=(1,), num_cells=2, *args, **kwargs):
    struct = AutoKSearchSpace(input_shape, output_shape, regression=True)
    input_nodes = struct.input_nodes

    output_submodels = [input_nodes[0]]

    for i in range(1, 4):
        cnode1 = ConstantNode(name='N', op=Dense(1000, tf.nn.relu))
        struct.connect(input_nodes[i], cnode1)

        cnode2 = ConstantNode(name='N', op=Dense(1000, tf.nn.relu))
        struct.connect(cnode1, cnode2)

        vnode1 = VariableNode(name='N3') 
        add_mlp_op_(vnode1)
        struct.connect(cnode2, vnode1)

        output_submodels.append(vnode1)

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

    return model


def print000(x):
    if type(x) == list:
        print("(")
        for i in x:
            print(i.shape)
        print(")")
    else:
        print(x.shape)


def mae(y_true, y_pred):
    import keras
    return keras.metrics.mean_absolute_error(y_true, y_pred)


def basic_train(model, X_train, Y_train, X_test, Y_test):
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    history = model.fit(X_train, Y_train,
                                batch_size=64,
                                epochs=4,
                                validation_data=(X_test, Y_test))
    print(history.history)


def loop_train(model, X_train, Y_train, X_test, Y_test):
    from tensorflow import keras
    import tensorflow as tf
    
    batch_size = 64
    optimizer = keras.optimizers.Adam()
    
    # print(Y_train.shape)
    # x train is a list of ndarray

    # print(([*X_train], Y_train))

    train_dataset =  tf.data.Dataset.from_tensor_slices((*X_train, Y_train)).shuffle(buffer_size=2048).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((*X_test, Y_test)).batch(batch_size)

    loss_fn = keras.losses.MeanSquaredError()
    train_acc_metric = keras.metrics.MeanSquaredError()
    val_acc_metric   = keras.metrics.MeanSquaredError()
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y, logits)
        return loss_value
    

    @tf.function
    def test_step(x, y):
        val_logits = model(x, training=False)
        val_acc_metric.update_state(y, val_logits)
    
    for epoch in range(4):
        for step, XY in enumerate(train_dataset):
            x_batch_train = list(XY[:-1])
            y_batch_train = XY[-1]
            train_step(x_batch_train, y_batch_train)
        
        print('train mse = ', float(train_acc_metric.result()))
        train_acc_metric.reset_states()

        for XY in val_dataset:
            test_step(list(XY[:-1]), XY[-1])
        
        print('val mse = ', float(val_acc_metric.result()))

        

if __name__ == '__main__':
    model = test_create_structure()
    
    from candlepb.Uno.uno_baseline_keras2 import load_data_multi_array as load_data
    (X_train, Y_train), (X_test, Y_test) = load_data()

    print000(X_train)
    print000(Y_train)
    print000(X_test)
    print000(Y_test)

    loop_train(model, X_train, Y_train, X_test, Y_test)
    # basic_train(model, X_train, Y_train, X_test, Y_test)
    # model = test_create_structure()



### training loop . 


