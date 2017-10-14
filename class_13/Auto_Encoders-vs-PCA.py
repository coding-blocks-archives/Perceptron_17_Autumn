import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import keras
from keras.layers import Dense, Activation, Input
from keras.models import Model

ds = pd.read_csv('./train.csv')               # Place the path to MNIST digit dataset in argument
data = ds.values

X_data = data[:, 1:]
X_std = X_data/255.0
n_train = int(0.75*X_std.shape[0])
n_val = int(0.25*X_std.shape[0])

X_train = X_std[:n_train]
X_val = X_std[n_train:n_train+n_val]

print X_train.shape, X_val.shape

########## Preparing the Auto Encoder ##########

inp = Input(shape = (784, ))                  # Using Keras Functional API
embedding_dim = 64                            # Dimensions of hidden vector representation

fc1 = Dense(embedding_dim)(inp)
ac1 = Activation('tanh')(fc1)

fc2 = Dense(784)(ac1)
ac2 = Activation('sigmoid')(fc2)

autoencoder = Model(inputs = inp, outputs = ac2)

########## Listing the layers used in Auto Encoder to form Encoder/Decoder ##########

print '\n'
for layer in autoencoder.layers:
	print layer
print '\n'

########## Preparing the Encoder and Decoder from Auto-Encoder's layers ##########

encoder = Model(inputs = inp, outputs = ac1)

dec_inp = Input(shape=(embedding_dim,))
x = autoencoder.layers[3](dec_inp)
x = autoencoder.layers[4](x)

decoder = Model(inputs = dec_inp, outputs = x)

########## Compiling and fiting the Auto Encoder ##########

autoencoder.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])
hist = autoencoder.fit(X_train, X_train, epochs=25, batch_size=100, shuffle=True, validation_data=(X_val, X_val))

auto_encoder_encodes = encoder.predict(X_train)                  # Encoder generates a hidden-dimension (64 dim) representation of original data (784 dim)
auto_encoder_decodes = decoder.predict(auto_encoder_encodes)     # Decoder decodes hidden-representation (64 dim) given by encoder to dimensions of input data (784 dim)

########## Applying PCA from sklearn on the same data ##########

from sklearn.decomposition import PCA

pca = PCA(n_components=embedding_dim)
pca_dim_reducts = pca.fit_transform(X_std[:(n_train + n_val)])
pca_regenerations = pca.inverse_transform(pca_dim_reducts)

########## Comparing 5 regerations from PCA & Auto Encoders ##########

plt.figure(0)
for ix in range(5, 10):
    plt.subplot(5, 3, ((ix-5) * 3) + 1)
    plt.title('Original')
    plt.imshow(X_train[ix].reshape((28, 28)), cmap='gray')
    plt.subplot(5, 3, ((ix-5) * 3) + 2)
    plt.title('A-E Regen.')
    plt.imshow(auto_encoder_decodes[ix].reshape((28, 28)), cmap='gray')
    plt.subplot(5, 3, ((ix-5) * 3) + 3)
    plt.title('PCA Regen.')
    plt.imshow(pca_regenerations[ix].reshape((28, 28)), cmap='gray')
plt.show()
