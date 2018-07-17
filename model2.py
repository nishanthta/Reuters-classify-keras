#simple feedforward NN with dropout

from keras import models, layers
from keras import regularizers
import numpy as np

net2 = models.Sequential()
net2.add(layers.Dense(64, kernel_regularizer = regularizers.l2(0.001), activation = 'relu', input_shape = (10000, )))
net2.add(layers.Dropout(0.5))
net2.add(layers.Dense(64, kernel_regularizer = regularizers.l1_l2(l1 = 0.001, l2 = 0.001), activation = 'relu'))
net2.add(layers.Dropout(0.5))
net2.add(layers.Dense(46, activation = 'softmax'))

net2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

