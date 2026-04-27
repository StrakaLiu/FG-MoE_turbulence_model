import os

#set threads of tf= 1
os.environ["TF_NUM_INTEROP_THREADS"]= "1"
os.environ["TF_NUM_INTRAOP_THREADS"]="1"
#set threads of openmp = 1
os.environ["OMP_NUM_THREADS"]="1"


import numpy as np
import tensorflow as tf
import time


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

f = open('log.python_module', 'w+')
print('_________________________________________________________________',
      file=f)
print('Computing function of scalar invariants from Python module', file=f)
print('Tensorflow version', tf.__version__, file=f)
print('_________________________________________________________________',
      file=f)

tf.keras.backend.clear_session()
# load model
model_path = './nn-model-5input_3output_2layer_10nodes_203weights_relu_linear.h5'
model = tf.keras.models.load_model(model_path, compile=False)

# load weights
# get weights flatten
weights_flatten = np.loadtxt('nn_weights_flatten.dat')

# get model shape
shapes = []
for iw in model.trainable_variables:
    shapes.append(iw.shape)

# shapes to sizes
sizes = []
for shape in shapes:
    isize = 1
    for ishape in shape:
        isize *= ishape
    sizes.append(isize)

# reshape weights
w_reshaped = []
i = 0
for shape, size in zip(shapes, sizes):
    w_reshaped.append(weights_flatten[i:i + size].reshape(shape))
    i += size
model.set_weights(w_reshaped)

print(model.get_weights(), file=f)
print('Neural-network weights loaded successfully', file=f)


def ml_func(array):

    array_theta = array[:,:5]

    t1 = time.time()

    g_ = model(array_theta, training=False)

    g = np.array(g_).reshape(-1, 3).astype('double')
    scale = [1.8, 0.5555555555, 1.0]  
    init = [1.8, 0.5555555555, 0.0]  
    gmin = [1.0, 1e-8, -1]
    gmax = [1e8, 1e8, 1e8]
    for i in range(g.shape[1]):
        g[:, i] *= scale[i]
        g[:, i] += init[i]
        g[:, i] = np.clip(g[:, i], gmin[i], gmax[i])


    return g
