from model1 import net1
from model2 import net2
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

history1 = net1.fit(x_train, y_train, epochs = 20, batch_size = 512, validation_data = (val_data, val_labels))
history2 = net2.fit(x_train, y_train, epochs = 20, batch_size = 512, validation_data = (val_data, val_labels))

train_loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'ro', label = 'training loss')
plt.plot(epochs, val_loss, 'r', label = 'validation loss')
plt.title('Training vs validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('losses.jpg')

plt.clf()

acc = history1.history['acc']
val_acc = history1.history['val_acc']
plt.plot(epochs, acc, 'ro', label = 'training accuracy')
plt.plot(epochs, val_acc, 'r', label = 'validation accuracy')
plt.title('Training vs validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc = 2)
plt.savefig('accuracy.jpg')

plt.clf()

val_acc2 = history2.history['val_acc']
plt.plot(epochs, val_acc, 'ro', label = 'without dropout')
plt.plot(epochs, val_acc2, 'r', label = 'with dropout')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc = 2)
plt.savefig('dropout_effect.jpg')

test_acc = model.evaluate(vec_test_data, oh_test_labels)