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
model_base = tf.keras.models.load_model(model_path, compile=False)
model_sep = tf.keras.models.load_model(model_path, compile=False)
model_sqr = tf.keras.models.load_model(model_path, compile=False)
model_asj = tf.keras.models.load_model(model_path, compile=False)


# get weights flatten
weights_flatten = np.loadtxt('nn_weights_flatten_BASE.dat')
shapes = []
for iw in model_base.trainable_variables:
    shapes.append(iw.shape)
sizes = []
for shape in shapes:
    isize = 1
    for ishape in shape:
        isize *= ishape
    sizes.append(isize)
w_reshaped = []
i = 0
for shape, size in zip(shapes, sizes):
    w_reshaped.append(weights_flatten[i:i + size].reshape(shape))
    i += size
model_base.set_weights(w_reshaped)
print(model_base.get_weights(), file=f)
print('Neural-network weights of BASE loaded successfully', file=f)


# get weights flatten
weights_flatten = np.loadtxt('nn_weights_flatten_SEP.dat')
shapes = []
for iw in model_sep.trainable_variables:
    shapes.append(iw.shape)
sizes = []
for shape in shapes:
    isize = 1
    for ishape in shape:
        isize *= ishape
    sizes.append(isize)
w_reshaped = []
i = 0
for shape, size in zip(shapes, sizes):
    w_reshaped.append(weights_flatten[i:i + size].reshape(shape))
    i += size
model_sep.set_weights(w_reshaped)
print(model_sep.get_weights(), file=f)
print('Neural-network weights of SEP loaded successfully', file=f)


# get weights flatten
weights_flatten = np.loadtxt('nn_weights_flatten_SQR.dat')
shapes = []
for iw in model_sqr.trainable_variables:
    shapes.append(iw.shape)
sizes = []
for shape in shapes:
    isize = 1
    for ishape in shape:
        isize *= ishape
    sizes.append(isize)
w_reshaped = []
i = 0
for shape, size in zip(shapes, sizes):
    w_reshaped.append(weights_flatten[i:i + size].reshape(shape))
    i += size
model_sqr.set_weights(w_reshaped)
print(model_sqr.get_weights(), file=f)
print('Neural-network weights of SQR loaded successfully', file=f)

# get weights flatten
weights_flatten = np.loadtxt('nn_weights_flatten_ASJ.dat')
shapes = []
for iw in model_asj.trainable_variables:
    shapes.append(iw.shape)
sizes = []
for shape in shapes:
    isize = 1
    for ishape in shape:
        isize *= ishape
    sizes.append(isize)
w_reshaped = []
i = 0
for shape, size in zip(shapes, sizes):
    w_reshaped.append(weights_flatten[i:i + size].reshape(shape))
    i += size
model_asj.set_weights(w_reshaped)
print(model_asj.get_weights(), file=f)
print('Neural-network weights of ASJ loaded successfully', file=f)




def ml_func(array):
    array_theta = array[:,:5]

    t1 = time.time()

    g_ = model_base(array_theta, training=False)
    g_base = np.array(g_).reshape(-1, 3).astype('double')
    g_ = model_sep(array_theta, training=False)
    g_sep = np.array(g_).reshape(-1, 3).astype('double')
    g_ = model_sqr(array_theta, training=False)
    g_sqr = np.array(g_).reshape(-1, 3).astype('double')
    g_ = model_asj(array_theta, training=False)
    g_asj = np.array(g_).reshape(-1, 3).astype('double')

    g = g_base*0.0

    q_w = array[:,5]
    q_nps = array[:,6]
    q_3D = array[:,7]
    F_nps = 1.0 / (1.0 + np.exp(-np.clip(40*(q_nps - 0.25),-20,20)))
    F_3D = 1.0 / (1.0 + np.exp(-np.clip(1000*(q_3D - 0.01),-20,20)))
    F_w = 1.0 / (1.0 + np.exp(-np.clip(50*(q_w - 0.2),-20,20)))

    for i in range(g_base.shape[1]):
        g[:, i] = F_w*F_nps*g_sep[:, i] + (1-F_nps)*F_3D*F_w*g_sqr[:, i] + F_3D*(1-F_w)*g_asj[:, i] + ((1-F_3D)*(1-F_w)+(1-F_nps)*(1-F_3D)*F_w)*g_base[:, i]

    scale = [1.8, 0.5555555555, 1.0]  
    init = [1.8, 0.5555555555, 0.0]  
    gmin = [1.0, 1e-8, -1]
    gmax = [1e8, 1e8, 1e8]
    for i in range(g.shape[1]):
        g[:, i] *= scale[i]
        g[:, i] += init[i]
        g[:, i] = np.clip(g[:, i], gmin[i], gmax[i])


    return g
