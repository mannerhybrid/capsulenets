#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 09:50:35 2018

@author: mnhakim
"""

# In[]
# Import Packages
import numpy as np
import tensorflow as tf
import keras.datasets.cifar10 as c10
import matplotlib.pyplot as plt
from six.moves import xrange
tf.reset_default_graph()
# In[]
# Form Dataset


def rgb2gray(rgblist):
    im=[]
    for i in range(len(rgblist)):
        rgb = rgblist[i]
        rgb = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        
        rgb = np.expand_dims(rgb, axis=-1)
        #print(rgb.shape)
        im.append(rgb)
    return im

def convert_int(a):
    mean = np.mean(a)
    
    print("mean: {}, max:{}, min: {}".format(mean, np.max(a), np.min(a)))
    a = a - mean
    print("zero-center max: {}, median: {}, min: {}".format(np.max(a),np.median(a), np.min(a)))
    var = np.std(a)
    print("var: {}".format(var))
    a = a / var
    print("max: {}".format(np.max(a)))
    a = a.astype(int)
    return a


data = c10.load_data()

    
images = {
        '10k': rgb2gray(data[1][0]),
        '50k': rgb2gray(data[0][0])
        }
labels = {
        '10k': data[1][1],
        '50k': data[0][1]
        }

def name_label(label):
    LABELS = ['Airplane', 'Automobile', 'Bird', 'Cat','Deer',
          'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    print(LABELS[int(label)] + '\n')
    return LABELS[int(label)]

# In[] Define Primary Capsules Parameters
    
n_maps= 32
n_dims= 8
n_caps= n_maps * 8 * 8 

# In[] Define Squash and Primary Capsule functions
def squash(s, axis=-1, epsilon=1e-7, name=None):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector

conv1_params = {
        "filters": 256,
        "kernel_size": 9,
        "strides": 1,
        "padding": "valid",
        "activation": tf.nn.relu
        }
        
conv2_params = {
        "filters": n_maps * n_dims, # 256 convolutional filters
        "kernel_size": 9,
        "strides": 2,
        "padding": "valid",
        "activation": tf.nn.relu
        }
        
#def pri_caps(
"""
        input_tensor,
        n_caps,
        n_dims=8,
        n_maps=32):
"""
       

        



# In[] Form Predictions of Digit Capsules

def form_predictions(input_tensor,
                     weight_tensor,
                     batch_size=1):

        caps1_output = input_tensor
        caps1_output = tf.expand_dims(caps1_output, axis=-1)
        caps1_output = tf.expand_dims(caps1_output, axis=2)
        caps1_output_tiled = tf.tile(caps1_output, 
                                     [1,1,n_j_caps,1,1],
                                     name="caps1_output_tiled")
       
        W = weight_tensor; 
        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="tiled_weights")
        predictions = tf.matmul(W_tiled, 
                                caps1_output_tiled,
                                name="caps2_predicted")
        return predictions

# In[] Define R.B.A and Digit Capsule functions

def RBA(input_tensor,
        rnd,
        batch_size,
        updated_routing_weights=None,
        n_caps = 2048,
        n_dims = 8,
        n_j_caps = 10,
        n_j_dims = 16
        ):
    
        b_ij = tf.zeros(shape=[batch_size, n_caps, n_j_caps, 1, 1],
                            dtype=tf.float32)
        u_j_pred = input_tensor
        if rnd > 0:
            u_j_pred = tf.tile(input_tensor, 
                             [1, n_caps, 1, 1, 1])
            b_ij = updated_routing_weights
        
        print("U_j_predicted \n\n" , u_j_pred, "\n")
        c_ij = tf.nn.softmax(b_ij)
        s_j = tf.reduce_sum(tf.multiply(c_ij, u_j_pred, name="weighted_predictions"),
                            axis=1,
                            keepdims=True,
                            name="weighted_sum")
        v_j = squash(s_j,
                     axis=-2,
                     name="output_j_caps")
        

        v_j_tiled = tf.tile(v_j, 
                             [1, n_caps, 1, 1, 1],
                             name="caps2_output_tiled")
        
        agreement = tf.matmul(u_j_pred,
                        v_j_tiled,
                        transpose_a=True,
                        name="agreement")
        b_ij_updated = tf.add(b_ij, agreement)
        print("Updated Raw Weights ", str(rnd), "\n\n", b_ij, "\n")
        
        return v_j, b_ij_updated

# In[]
        
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)
    
# In[]
batch_size = 50

image = images['10k'][:5]
image = np.expand_dims(image, axis=0)
image = image.astype(dtype='float32',casting='same_kind')

x = tf.placeholder(dtype=tf.float32, 
                   shape=[None,32,32,1],
                   name='X')



conv1 = tf.nn.conv2d(
        x,
        name="conv1", 
        filter = tf.truncated_normal([9,9,1,256]),
        use_cudnn_on_gpu=True,
        strides=[1,1,1,1],
        padding="VALID"
        )
conv2 = tf.nn.relu(tf.nn.conv2d(
        conv1,
        name="conv2",
        filter = tf.truncated_normal([9,9,int(conv1.shape[-1]),256]),
        use_cudnn_on_gpu = True,
        strides=[1,2,2,1],
        padding="VALID"
        ))

pri_caps_raw=tf.reshape(conv2,
                        [-1, n_caps, n_dims],
                        name="caps1_raw") 

pri_caps = squash(pri_caps_raw)

pc = pri_caps
print("Primary Capsules Output \n\n", pc, "\n")

# In[] Define Digit Capsules Parameters and W_ij

n_j_caps = 10
n_j_dims = 16
init_sigma = 0.1

W_init = tf.random_normal(
    shape=(1, n_caps, n_j_caps, n_j_dims, n_dims),
    stddev=init_sigma,
    dtype=tf.float32)
W_ij = tf.Variable(W_init, name="W")
#print("Elementwise Weights \n\n", W_ij, "\n")
W_tiled = tf.tile(
        W_ij,
        [batch_size, 1, 1, 1, 1], name="W_tiled"
        )

dc = form_predictions(pc, W_tiled)
print("Predictions formed \n\n", dc, "\n")

counter = 5
i=0
while i < counter:
    if i ==0:
        dc, B = RBA(dc,0, batch_size=batch_size)
        print("Digit Capsule Output ", str(i),"\n", dc, "\n")
        i=i+1
    else:
        dc, B = RBA(dc,i,updated_routing_weights=B, batch_size=batch_size)
        print("Digit Capsule Output ", str(i),"\n", dc, "\n")
        i=i+1
        
y_prob = safe_norm(dc, axis=-2, name="y_proba")
y_proba_argmax = tf.argmax(y_prob, axis=2, name="y_proba")
y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")

y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

T = tf.one_hot(y, depth= n_j_caps, name="T")

dc_norm = safe_norm(dc, axis=-2, keep_dims=True,
                              name="caps2_output_norm")
present_error_raw = tf.square(tf.maximum(0., m_plus - dc_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, 10),
                           name="present_error")
absent_error_raw = tf.square(tf.maximum(0., dc_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, 10),
                          name="absent_error")
L = tf.add(T * present_error, 
           lambda_ * (1.0 - T) * absent_error,
           name="L")
margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), 
                             name="margin_loss")
mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")
reconstruction_targets = tf.cond(mask_with_labels, # condition
                                 lambda: y,        # if True
                                 lambda: y_pred,   # if False
                                 name="reconstruction_targets")
reconstruction_mask = tf.one_hot(reconstruction_targets,
                                 depth=n_j_caps,
                                 name="reconstruction_mask")
reconstruction_mask_reshaped = tf.reshape(
    reconstruction_mask, [-1, 1, n_j_caps, 1, 1],
    name="reconstruction_mask_reshaped")
print(reconstruction_mask_reshaped )
caps2_output_masked = tf.multiply(
    dc, reconstruction_mask_reshaped,
    name="caps2_output_masked")
print(caps2_output_masked)
decoder_input = tf.reshape(caps2_output_masked,
                           [-1, n_j_caps * n_j_dims],
                           name="decoder_input")
print(decoder_input)

##decoder shit
n_hidden1 = 512
n_hidden2 = 1024
n_output = 32 * 32 * 1

with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu,
                              name="hidden1")
    print(hidden1)
    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu,
                              name="hidden2")
    print(hidden2)
    decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")
    print(decoder_output)
    
X_flat = tf.reshape(x, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output,
                               name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference,
                                    name="reconstruction_loss")

alpha = 0.0005

loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")
correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# END OF CONSTURCTION

n_epochs = 10
batch_size = 50
restore_checkpoint = True

n_iterations_per_epoch = 10000 // batch_size
n_iterations_validation = 10000 // batch_size
best_loss_val = np.infty
checkpoint_path = "./my_capsule_network_ON_CIFAR"

n_samples = batch_size
with tf.Session() as sess:
   
    init.run()
    #saver.restore(sess, checkpoint_path)
    sample_images = images['10k'][:n_samples]
    caps2_output_value, decoder_output_value, y_pred_value = sess.run(
            [dc, decoder_output, y_pred],
            feed_dict={x: sample_images,
                       y: np.array([], dtype=np.int64)})
            
#sample_images = sample_images.reshape([n_samples, 32, 32, 1])
reconstructions = decoder_output_value.reshape([n_samples, 32, 32, 1])
reconstructions = convert_int(reconstructions)
print(np.max(reconstructions))
plt.imshow(sample_images[0])
plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.imshow(sample_images[index])
    name = name_label(labels['10k'][index])
    plt.title("Label:" + name)
    plt.axis("off")

plt.show()

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.title("Predicted:" + str(y_pred_value[index]))
    plt.imshow(reconstructions[index], cmap="binary")
    plt.axis("off")
    
plt.show()

            

            
            