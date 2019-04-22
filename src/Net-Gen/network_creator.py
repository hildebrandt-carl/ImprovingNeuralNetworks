import sys
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Check that an argument has been passed with the python script
if len(sys.argv) <= 1:
  print("This file requires whether or not to use TMR")
  exit()

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

# Read the text file describing the architecture
f = open("NetworkArchitecture/" + sys.argv[1],"r")
layernum = 0
architecture = []

for line in f:
  # Save the architecture to put in the result file
  architecture.append(line)

  # remove the last \n
  if line.endswith('\n'):
    line = line[:-1]
  # Break the line up
  line_details = line.split(', ')

  # Add a dense layer
  if line_details[0] == "dense":
    model.add(Dense( units=int(line_details[1]),
                     activation=line_details[2]))

  # Add a convolutional layer
  elif line_details[0] == "conv":
    if layernum == 0:
      model.add(Conv2D( filters=int(line_details[1]),
                        kernel_size=(3, 3),
                        activation=line_details[2],
                        input_shape=input_shape))
    else:
      model.add(Conv2D( filters=int(line_details[1]),
                        kernel_size=(3, 3),
                        activation=line_details[2]))

  # Add a dropout layer
  elif line_details[0] == "dropout":
    model.add(Dropout( rate=float(line_details[1])))

  # Add a maxpooling layer
  elif line_details[0] == "maxpooling":
    model.add(MaxPooling2D( pool_size=(2, 2)))

  # Add a flattern layer
  elif line_details[0] == "flattern":
    model.add(Flatten())

  # We dont know what the layer is
  else:
    print("unkown layer type")
    exit()

  layernum += 1
  
# Close the file
f.close()

# Add the final classification layer
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

# Save the testing accuracy:
details_saver = open("FinalNetworks/results_" + sys.argv[1],"w") 
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
network_name = sys.argv[1].split('.')
model.save("FinalNetworks/" + str(network_name[0]) + ".h5")