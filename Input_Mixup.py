import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import os
import warnings #Suppress all warnings 
warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Progbar
import tensorflow_probability as tfp
import time 
import random
tf.executing_eagerly()
tf.config.experimental_run_functions_eagerly(True)

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


class Mixup(tf.keras.layers.Layer):
    def __init__(self, lmbda):
        super(Mixup, self).__init__()
        #print(dir(super(Mixup, self)).__init__())
        
        self.lmbda = tf.Variable(lmbda)
        
    def mixup(self, lmbda, a, b):
        x1 = tf.math.multiply(lmbda, a)
        x2 = tf.math.multiply(1-lmbda, b)
        return tf.math.add(x1, x2)
    
    #@tf.function
    #def __call__(self, inputs, training=None):
    def call(self, inputs, training=None):
        #Mixup doesnt happen when predicting
        if training:
            try:
                inputs = tf.reshape(inputs, [2,inputs.shape[1],inputs.shape[2],inputs.shape[3],inputs.shape[4]])
            except:
                raise Exception("Must have batch size of 2 or be matrix rank 5")  #Mixup only works with batch size of 2 (for now)
            #Mixup inputs and convert to numpy
            x_mix = self.mixup(self.lmbda, inputs[0], inputs[1])
            if len(x_mix.shape) < 5:
                x_mix = tf.reshape(x_mix, [1, x_mix.shape[0], x_mix.shape[1], x_mix.shape[2], x_mix.shape[3]])
            inputs = tf.concat([inputs, x_mix], 0)
            
            return inputs
        return inputs


def simple_UNET(dim, filters=8):
    
    input = tf.keras.Input(shape = dim)
    
    conv1 = input
 
    conv1 = tf.keras.layers.Conv3D(filters, (3, 3, 3), activation = 'relu', padding = 'same')(conv1)
    conv1 = tf.keras.layers.Conv3D(filters*2, (3, 3, 3), activation = 'relu', padding = 'same')(conv1)
    pool1 = tf.keras.layers.MaxPooling3D()(conv1)
    
        
    conv2 = tf.keras.layers.BatchNormalization()(pool1)
    conv2 = tf.keras.layers.Conv3D(filters*2, (3, 3, 3), activation = 'relu', padding = 'same')(conv2)
    conv2 = tf.keras.layers.Conv3D(filters*4, (3, 3, 3), activation = 'relu', padding = 'same')(conv2)
    pool2 = tf.keras.layers.MaxPooling3D()(conv2)
       
    
    conv3 = tf.keras.layers.BatchNormalization()(pool2)
    conv3 = tf.keras.layers.Conv3D(filters*4, (3, 3, 3), activation = 'relu', padding = 'same')(conv3)
    conv3 = tf.keras.layers.Conv3D(filters*8, (3, 3, 3), activation = 'relu', padding = 'same')(conv3)
    pool3 = tf.keras.layers.MaxPooling3D()(conv3)
        
    
    conv4 = tf.keras.layers.BatchNormalization()(pool3)
    conv4 = tf.keras.layers.Conv3D(filters*8, (3, 3, 3), activation = 'relu', padding = 'same')(conv4)
    upsample4 = tf.keras.layers.UpSampling3D()(conv4)


    conv5 = tf.keras.layers.BatchNormalization()(upsample4)
    conv5 = tf.keras.layers.Conv3D(filters*16, (3, 3, 3), activation = 'relu', padding = 'same')(conv5) #might be filters*8
    concat5 = tf.keras.layers.concatenate([conv5, conv3], axis=-1)
    conv5 = tf.keras.layers.Conv3D(filters*8, (3, 3, 3), activation = 'relu', padding = 'same')(concat5)
    upsample5 = tf.keras.layers.UpSampling3D()(conv5)


    conv6 = tf.keras.layers.BatchNormalization()(upsample5)
    conv6 = tf.keras.layers.Conv3D(filters*8, (3, 3, 3), activation = 'relu', padding = 'same')(conv6)
    concat6 = tf.keras.layers.concatenate([conv6, conv2], axis=-1)
    conv6 = tf.keras.layers.Conv3D(filters*4, (3, 3, 3), activation = 'relu', padding = 'same')(concat6)
    upsample6 = tf.keras.layers.UpSampling3D()(conv6)    
    
    
    conv7 = tf.keras.layers.BatchNormalization()(upsample6)
    conv7 = tf.keras.layers.Conv3D(filters*4, (3, 3, 3), activation = 'relu', padding = 'same')(conv7)
    concat7 = tf.keras.layers.concatenate([conv7, conv1], axis=-1)
    conv7 = tf.keras.layers.Conv3D(filters*2, (3, 3, 3), activation = 'relu', padding = 'same')(concat7)
    conv7 = tf.keras.layers.Conv3D(filters*2, (3, 3, 3), activation = 'relu', padding = 'same')(conv7)   

    output = tf.keras.layers.Conv3D(1, (1, 1, 1), activation = 'sigmoid', padding = 'same')(conv7)

    #Finalize the model
    model = tf.keras.Model(inputs=[input], outputs=[output])

    return model



x_train = np.load(r'C:\Users\alochbiler\Desktop\Medical-Image-Segmentation-App-main\Images\bladder_image_tensor.npy')
y_train = np.load(r'C:\Users\alochbiler\Desktop\Medical-Image-Segmentation-App-main\Images\bladder_mask_tensor.npy')
#(130.00, 149.00) centre 64 on each side
print('Before', x_train.shape)
x_train=x_train[:,8:56,66:194,66:194,:]
y_train=y_train[:,8:56,66:194,66:194,:]
print('After', x_train.shape)
height = x_train.shape[1]
rows = x_train.shape[2]
cols = x_train.shape[3]

#model = simple_UNET((height, rows, cols, 1), 4, k=0)

def get_batch(x_data, y_data, batch_size):
    i = np.random.randint(0, len(x_data), batch_size)
    #tf.random.shuffle(tf.range(6)) #Use this for sample without Duplicate
    return x_data[i], y_data[i]

def loss_fn(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
#optimizer = tf.keras.optimizers.Adam()

num_epochs = 10
batch_size = 2
total_batch = int(len(x_train) / batch_size)
metrics_names = ['loss','dice'] 
weights = 0
p = 0
shape = (height, rows, cols, 1)
model = simple_UNET(shape, 8)

for i in range(num_epochs):
    avg_loss = 0
    avg_metric = 0
    print("\nepoch {}/{}".format(i+1,num_epochs))
    progbar = Progbar(len(x_train), stateful_metrics=metrics_names)
    
    for j in range(total_batch):
        
        #ADD MIXUP ENABLE DIASBLE
        batch_x, batch_y = get_batch(x_train, y_train, batch_size=batch_size)

        # create tensors    
        batch_x = tf.Variable(batch_x)
        batch_y = tf.Variable(batch_y)
        '''
        alpha = tf.constant(0.4)
        dist = tfp.distributions.Beta(alpha, alpha)
        lmbda = dist.sample(1)
        
        batch_y = Mixup(lmbda=lmbda)(batch_y, training=True)
        batch_x = Mixup(lmbda=lmbda)(batch_x, training=True)
        '''
        with tf.GradientTape() as tape:
            y_pred = model(batch_x, training=True)
            
            loss = loss_fn(batch_y, y_pred)
        metric = dice_coef(batch_y, y_pred)
        
        grads = tape.gradient(loss, model.trainable_variables) #backprop
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        avg_loss += loss / total_batch
        avg_metric += metric / total_batch
        
        weights = model.get_weights()
        
        values=[('loss',loss), ('dice',metric)]
        progbar.add(batch_size, values=values)
        #print(i, loss, metric)
        p = 1
        
    print(f'loss={avg_loss:.3f}, metric={avg_metric:.3f}')
    model.save(r'C:\Users\alochbiler\Desktop\Medical-Image-Segmentation-App-main\Models\modelcheck_crop.h5') 
print("\nTraining complete!")

model.save(r'C:\Users\alochbiler\Desktop\Medical-Image-Segmentation-App-main\Models\model_crop.h5') 







