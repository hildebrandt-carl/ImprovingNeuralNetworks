import foolbox
import keras
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import model_from_json
from random import randint


image_number = randint(1, 10000)

# Set to test mode
keras.backend.set_learning_phase(0)

# Load the model
json_file = open('smalldense.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('smalldense.h5')

# Instantiate attack model
fmodel = foolbox.models.KerasModel(loaded_model, bounds=(0, 1))

# get source image and label
_, (images, labels) = mnist.load_data()
images = images.reshape(10000,28,28,1)
images = images.astype('float32')
images /= 255.0

# apply attack on source image
attack = foolbox.attacks.FGSM(fmodel)
adversarial = attack(images[image_number], labels[image_number])

# if the attack fails, adversarial will be None and a warning will be printed
if adversarial == None:
	print("No adversarial found")
	exit()

# Plot the images
plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original')
img_plot = images[image_number].reshape(28,28)
plt.imshow(img_plot)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Adversarial')
adv_plot = adversarial.reshape(28,28)
plt.imshow(adv_plot)  # ::-1 to convert BGR to RGB
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Difference')
difference = adv_plot - img_plot
plt.imshow(difference)
plt.axis('off')

plt.show()