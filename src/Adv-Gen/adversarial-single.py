#!/usr/bin/env python3

# -------------------------------------------------------------------
# Generates a single adverserial image for a network which you specify
# -------------------------------------------------------------------

import sys
import foolbox
import keras
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import load_model
from random import randint

# Check that an argument has been passed with the python script
if len(sys.argv) <= 2:
  print("This file requires a network name and attack type")
  exit()

# Networks which will be loaded
net_name = sys.argv[1]
attack_name = sys.argv[2]
num_images = 100

# Set to test mode
keras.backend.set_learning_phase(0)

# Load the model
loaded_model = load_model("../Net-Gen/FinalNetworks/" + net_name + ".h5")

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
attack = None
if attack_name == "GradientAttack":
  attack = foolbox.attacks.GradientAttack(fmodel)
if attack_name == "GradientSignAttack":
  attack = foolbox.attacks.FGSM(fmodel)
elif attack_name == "DeepFool":
  attack = foolbox.attacks.DeepFoolAttack(fmodel)
elif attack_name == "ADef":
  attack = foolbox.attacks.ADefAttack(fmodel)
elif attack_name == "SaliencyMap":
  attack = foolbox.attacks.SaliencyMapAttack(fmodel)
elif attack_name == "CarliniWagner":
  attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel)
elif attack_name == "Newton":
  attack = foolbox.attacks.NewtonFoolAttack(fmodel)
elif attack_name == "ProjectedGradient":
  attack = foolbox.attacks.RandomStartProjectedGradientDescentAttack(fmodel)
elif attack_name == "SLSQPAttack":
  attack = foolbox.attacks.SLSQPAttack(fmodel)
elif attack_name == "LBFGS":
  attack = foolbox.attacks.LBFGSAttack(fmodel)
else:
  print("Attack not known")

# Get the adversarial attack
adversarial = attack(images[image_number], labels[image_number])

# Get the Original label
org_label = labels[image_number]

# Get the Adversarial label
logics = loaded_model.predict(adversarial.reshape([1,adversarial.shape[0],adversarial.shape[1],1]))
adv_label = np.argmax(logics)

# if the attack fails, adversarial will be None and a warning will be printed
if adversarial is None:
	print("No adversarial found")
	exit()

# Plot the images
plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original - Class:' + str(org_label))
img_plot = images[image_number].reshape(28,28)
plt.imshow(img_plot, cmap='gray', vmin=0, vmax=1)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Ariginal - Class:' + str(adv_label))
adv_plot = adversarial.reshape(28,28)
plt.imshow(adv_plot, cmap='gray', vmin=0, vmax=1)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Difference')
difference = adv_plot - img_plot
plt.imshow(difference, cmap='gray', vmin=0, vmax=1)
plt.axis('off')

plt.show()