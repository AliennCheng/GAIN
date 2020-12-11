'''GAIN function -- TensorFlow2 implementation.

Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
'''

# Necessary packages
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import uniform_sampler, binary_sampler, sample_batch_index

def gain (data_x, gain_parameters):
    '''Impute missing values in data_x

    Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyperparameter
        - iterations: Iterations
        
    Returns:
    - imputed_data: imputed data
    '''
    # Define mask matrix
    data_m = (1-np.isnan(data_x)).astype(float)

    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    # Other parameters
    no, dim = data_x.shape

    # Hidden state dimensions
    h_dim = int(dim)

    # Normalization
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    # parameter initialization
    X = tf.convert_to_tensor(norm_data_x)
    X = tf.dtypes.cast(X, tf.float32)
    M = tf.convert_to_tensor(data_m)
    M = tf.dtypes.cast(M, tf.float32)
    X_input = tf.concat(values=[X, M], axis=1)

    ## GAIN architecture
    # Generator
    class Generator(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.flatten = layers.Flatten(input_shape=[dim*2])
            self.dense1 = layers.Dense(h_dim, activation='relu')
            self.dense2 = layers.Dense(h_dim, activation='relu')
            self.dense_output = layers.Dense(dim, activation='sigmoid')
            return
            
        def call(self, inputs, training=None):
            x = self.flatten(inputs)
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.dense_output(x)
            return x

    # Discriminator
    class Discriminator(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.flatten = layers.Flatten(input_shape=[dim*2])
            self.dense1 = layers.Dense(h_dim, activation='relu')
            self.dense2 = layers.Dense(h_dim, activation='relu')
            self.dense_output = layers.Dense(dim, activation='sigmoid')
            return
            
        def call(self, inputs, training=None):
            x = self.flatten(inputs)
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.dense_output(x)
            return x

    ## GAIN loss
    # Generator
    def generator_loss(generator, discriminator, x, m):
        generator.trainable = True
        discriminator.trainable = False
        G_input = tf.concat(values=[x, m], axis=1)
        G_sample = generator(G_input)
        MSE_loss = tf.reduce_mean((m * x - m * G_sample)**2) / tf.reduce_mean(m)
        D_input = tf.concat(values=[G_sample, m], axis=1)
        D_prob = discriminator(D_input)
        G_loss_tmp = -tf.reduce_mean((1-m) * tf.math.log(D_prob + 1e-8))
        return G_loss_tmp + alpha * MSE_loss
    
    # Discriminator
    def discriminator_loss(generator, discriminator, x, m, h):
        generator.trainable = False
        discriminator.trainable = True
        G_input = tf.concat(values=[x, m], axis=1)
        G_sample = generator(G_input)
        x_hat = x * m + G_sample * (1-m)
        D_input = tf.concat(values=[x_hat, h], axis=1)
        D_prob = discriminator(D_input)
        return -tf.reduce_mean(m * tf.math.log(D_prob + 1e-8) \
                + (1-m) * tf.math.log(1. - D_prob + 1e-8))

    # Build
    generator = Generator()
    generator.build(input_shape=(None, 2*dim))
    g_optimizer = tf.keras.optimizers.Adam()
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 2*dim))
    d_optimizer = tf.keras.optimizers.Adam()

    # Training
    one_tensor = tf.constant(1., shape=(batch_size, dim), dtype=float)

    for _ in tqdm(range(iterations)):
        # Sample batch
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = tf.gather(X, batch_idx)
        M_mb = tf.gather(M, batch_idx)
        Z_mb = tf.convert_to_tensor(uniform_sampler(0, 0.01, batch_size, dim), dtype=float)
        H_mb_tmp = tf.convert_to_tensor(binary_sampler(hint_rate, batch_size, dim), dtype=float)
        H_mb = tf.math.multiply(M_mb, H_mb_tmp)

        # Combine random vectors with observed vectors
        # X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
        X_mb = tf.math.add(tf.math.multiply(M_mb, X_mb), \
                tf.math.multiply(tf.math.subtract(one_tensor, M_mb), Z_mb))

        # training Discriminator
        with tf.GradientTape() as tape:
            d_loss = discriminator_loss(generator, discriminator, X_mb, M_mb, H_mb)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # training Generator
        with tf.GradientTape() as tape:
            g_loss = generator_loss(generator, discriminator, X_mb, M_mb)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    ## Return imputed data
    imputed_data = np.array([]).reshape(0, dim)
    train_data = tf.data.Dataset.from_tensor_slices(X_input).batch(batch_size)
    train_data_iter = iter(train_data)
    while True:
        try:
            batch = next(train_data_iter)
        except StopIteration:
            break
        X_tmp = generator(batch).numpy()
        imputed_data = np.vstack([imputed_data, X_tmp])
    
    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)

    # Recovery
    imputed_data = data_m * np.nan_to_num(data_x) + (1-data_m) * imputed_data

    # Rounding
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data