#simple feedforward NN without dropout

from keras import models, layers
from keras import regularizers
import numpy as np

net1 = models.Sequential()
net1.add(layers.Dense(64, kernel_regularizer = regularizers.l2(0.001), activation = 'relu', input_shape = (10000, )))
net1.add(layers.Dense(64, kernel_regularizer = regularizers.l1_l2(l1 = 0.001, l2 = 0.001), activation = 'relu'))
net1.add(layers.Dense(46, activation = 'softmax'))

net1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

