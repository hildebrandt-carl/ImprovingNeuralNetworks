#!/usr/bin/env python3

# -------------------------------------------------------------------
# Generates a single adverserial image for a network which you specify
# -------------------------------------------------------------------

import foolbox
import keras
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import model_from_json
from random import randint

# Set to test mode
keras.backend.set_learning_phase(0)

# Load the model
json_file = open('../Net-Gen/Networks/small_dense.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('../Net-Gen/Networks/small_dense.h5')

# Instantiate attack model
fmodel = foolbox.models.KerasModel(loaded_model, bounds=(0, 1))

# get source image and label
_, (images, labels) = mnist.load_data()
images = images.reshape(images.shape[0],images.shape[1],images.shape[2],1)
images = images.astype('float32')
images /= 255.0

# Generate a random image number
image_number = randint(0, images.shape[0])

# apply attack on source image
attack = foolbox.attacks.FGSM(fmodel)
adversarial = attack(images[image_number], labels[image_number])

# if the attack fails, adversarial will be None and a warning will be printed
if adversarial is None:
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