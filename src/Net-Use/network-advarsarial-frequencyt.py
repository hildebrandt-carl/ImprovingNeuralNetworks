#!/usr/bin/env python3

# -------------------------------------------------------------------
# Compares the student vs the no teacher networks
# -------------------------------------------------------------------

import sys
import keras
import foolbox
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from random import randint
from keras.datasets import mnist

# Check that an argument has been passed with the python script
if len(sys.argv) <= 1:
  print("This file requires a list of network names")
  exit()

# Save the testing accuracy:
details_saver = open("Results/network-advarsarial-frequency-teacher.txt","w")

# Number of images you want to try on
total_images_tried = 100

# Network names we will be testing
net_names = sys.argv[1].split(', ')

# Set to test mode
keras.backend.set_learning_phase(0)

# Create variable for each of the loaded models
loaded_model = []

# The dataset variables
images = None
original_labels = None

print("--------------------------------------------------")
print("---Loading Datasets and Networks Networks Start---")
print("--------------------------------------------------")

# Load each of the networks
for net_i in range(0,len(net_names)):
    # Load the model
    l_model = load_model("../Net-Gen/TeacherNetworks/" + net_names[net_i] + ".h5")
    print("Loaded network" + str(net_names[net_i]))
    details_saver.write("Loaded network" + str(net_names[net_i]) + "\n")
    loaded_model.append(l_model)

# Get the adversarial dataset
adv_images = np.load("../Adv-Gen/Datasets/adversarial_dataset.npy")
adv_images /= 255.0
adv_original_labels = np.load("../Adv-Gen/Datasets/original_labels.npy")

# Get Mnist data
_, (mnist_images, mnist_labels) = mnist.load_data()
mnist_images = mnist_images.reshape(mnist_images.shape[0],mnist_images.shape[1],mnist_images.shape[2],1)
mnist_images = mnist_images.astype('float32')
mnist_images /= 255.0

# Testing begins now
print(" ")
print("--------------------------------------------------")
print("----------Individual Networks Testing-------------")
print("--------------------------------------------------")
details_saver.write("--------------------------------------------------\n")
details_saver.write("---------Individual Networks Testing-------------\n")
details_saver.write("--------------------------------------------------\n")

# Get the individual network adversarial scores
for net_i in range(0, len(loaded_model)):

    print("Testing " + str(net_names[net_i]))
    details_saver.write("Testing " + str(net_names[net_i]) + "\n")

    print("Calculating the adversarial frequency")
    # Instantiate attack model
    fmodel = foolbox.models.KerasModel(loaded_model[net_i], bounds=(0, 1))

    # Create the attacks
    attacks = []
    attacks.append(foolbox.attacks.GradientAttack(fmodel))
    attacks.append(foolbox.attacks.FGSM(fmodel))
    attacks.append(foolbox.attacks.DeepFoolAttack(fmodel))
    attacks.append(foolbox.attacks.ADefAttack(fmodel))
    attacks.append(foolbox.attacks.SaliencyMapAttack(fmodel))
    attacks.append(foolbox.attacks.CarliniWagnerL2Attack(fmodel))
    attacks.append(foolbox.attacks.NewtonFoolAttack(fmodel))
    attacks.append(foolbox.attacks.RandomStartProjectedGradientDescentAttack(fmodel))
    #attacks.append(foolbox.attacks.SLSQPAttack(fmodel))
    attacks.append(foolbox.attacks.LBFGSAttack(fmodel))
    attack_names = ["GradientAttack",
                    "FGSM",
                    "DeepFoolAttack",
                    "ADefAttack",
                    "SaliencyMapAttack",
                    "CarliniWagnerL2Attack",
                    "NewtonFoolAttack",
                    "RandomStartProjectedGradientDescentAttack",
                    "LBFGSAttack"]

    # Check the adversarial frequency
    attack_count = {
            		"GradientAttack" : 0,
            		"FGSM" : 0,
            		"DeepFoolAttack" : 0,
                    "ADefAttack" : 0,
            		"SaliencyMapAttack" : 0,
            		"CarliniWagnerL2Attack" : 0,
            		"NewtonFoolAttack" : 0,
                    "RandomStartProjectedGradientDescentAttack" : 0,
                    "LBFGSAttack" : 0
                	}

    # For each of the images in the mnist test dataset
    for img_i in range(0,total_images_tried):
        print("Processing image: " + str(img_i + 1) + "/" + str(total_images_tried))

        attack_num = 0
        # For each of the attacks try use it to find an adversarial example
        for attack in attacks:
            adversarial = attack(mnist_images[img_i], mnist_labels[img_i])
            # Increment the attack number (Used to save what kind of attack was used)
            if adversarial is not None:
                attack_count[attack_names[attack_num]] += 1
            # Go to next attack
            attack_num += 1

    # Print out the accuracy
    print(" ")
    total_adversarial_success = 0
    for key in attack_count:
        print("Adversarial Frequency " + key + ": " + str(attack_count[key]) + "/" + str(total_images_tried))
        details_saver.write("Adversarial Frequency " + key + ": " + str(attack_count[key]) + "/" + str(total_images_tried) + "\n")
        total_adversarial_success += attack_count[key]
    print("Final Adversarial Frequency: " + str(total_adversarial_success) + "/" + str(total_images_tried * len(attack_names)))
    print("--------------------------------------------------")
    details_saver.write("Final Adversarial Frequency: " + str(total_adversarial_success) + "/" + str(total_images_tried * len(attack_names)) + "\n")
    details_saver.write("--------------------------------------------------\n")


# Close the file writer
details_saver.close()
