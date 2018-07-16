from keras import models, layers
from keras.utils.np_utils import to_categorical
from keras.datasets import reuters
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data,test_labels) = reuters.load_data(num_words = 10000)

def vectorize(sequences, dim = 10000):
	results = np.zeros((len(sequences), dim))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1
	return results

vec_train_data = vectorize(train_data)
vec_test_data = vectorize(test_data)
oh_train_labels = to_categorical(train_labels)
oh_test_labels = to_categorical(test_labels)
val_data = vec_train_data[:1000]
val_labels = oh_train_labels[:1000]
x_train = vec_train_data[1000:]
y_train = oh_train_labels[1000:]

model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape = (10000, )))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(46, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train, y_train, epochs = 20, batch_size = 512, validation_data = (val_data, val_labels))

train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'ro', label = 'training loss')
plt.plot(epochs, val_loss, 'r', label = 'validation loss')
plt.title('Training vs validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('losses.jpg')

plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'ro', label = 'training accuracy')
plt.plot(epochs, val_acc, 'r', label = 'validation accuracy')
plt.title('Training vs validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc = 2)
plt.savefig('accuracy.jpg')