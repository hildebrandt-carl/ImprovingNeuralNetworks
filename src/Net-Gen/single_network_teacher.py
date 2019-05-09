import sys
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Training Hyperparameters
batch_size = 128
num_classes = 10
epochs = 12

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Get the size of the images
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]

# Reshape the data
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Change the images to images between 0 -1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Print out the size of tahe data
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# We want to replace y_train with the teacher labels
y_train = np.load("../Net-Use/Results/teacher_short.npy")

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Write the architecture for saving later
architecture = []
architecture.append("Conv2D: 32\n")
architecture.append("Conv2D: 64\n")
architecture.append("MaxPooling2D\n")
architecture.append("Dropout: 0.25\n")
architecture.append("Flatten\n")
architecture.append("Dense: 128\n")
architecture.append("Dropout: 0.5\n")
architecture.append("Dense: 10")

# We want to find the line of best fit, so we use mean square error
model.compile(loss='mse',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Train the model
model.fit(x=x_train,
          y=y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the testing accuracy:
details_saver = open("TeacherNetworks/Teacher.txt","w")
details_saver.write("---------------------------\n")
details_saver.write("Model\n")
details_saver.write("---------------------------\n")
details_saver.writelines(architecture)
details_saver.write("\n---------------------------\n")
details_saver.write("Hyper-Parameters\n")
details_saver.write("---------------------------\n")
details_saver.write("Batch Size: " + str(batch_size) + "\n")
details_saver.write("Number Classes: " + str(num_classes) + "\n")
details_saver.write("Epochs: " + str(epochs) + "\n")
details_saver.write("---------------------------\n")
details_saver.write("Results\n")
details_saver.write("---------------------------\n")
details_saver.write("Test loss: " + str(score[0]) + "\n")
details_saver.write("Test accuracy: " + str(score[1]) + "\n")
details_saver.close()

# Save full Model
model.save("TeacherNetworks/Teacher.h5")
