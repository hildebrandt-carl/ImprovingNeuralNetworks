import keras
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

# Print out the size of the data
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Create the model
model = Sequential()
model.add(Conv2D( filters=16,
                  kernel_size=(3, 3),
                  activation='relu',
                  input_shape=input_shape))
model.add(Conv2D( filters=32,
                  kernel_size=(3, 3),
                  activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=128,
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=num_classes,
                activation='softmax'))

# For a categorial classification problem
model.compile(loss=keras.losses.categorical_crossentropy,
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

# Save the model
model_json = model.to_json()
with open("Networks/conv.json", "w") as json_file:
  json_file.write(model_json)
model.save_weights("Networks/conv.h5

# Save full Model
model.save('Networks/full_conv.h5')