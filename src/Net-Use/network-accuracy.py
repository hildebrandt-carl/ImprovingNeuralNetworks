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
details_saver = open("Results/network-accuracy.txt","w")

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
    mnist_correct = 0
    adv_correct = 0

    print("Checking the accuracy on the adversarial dataset")
    # For each of the images in the adverserial dataset
    for img_i in range(0,len(adv_original_labels)):

        # Get the models prediction
        adv_shaped_img = adv_images[img_i].reshape([1,adv_images[img_i].shape[0],adv_images[img_i].shape[1],1])
        logics = loaded_model[net_i].predict(adv_shaped_img)
        label = np.argmax(logics)

        # Count the number of images the network correctly classified
        if (adv_original_labels[img_i] == label):
            adv_correct += 1

    print("Checking the mnist test accuracy")
    # For each of the images in the mnist test dataset
    for img_i in range(0,len(mnist_labels)):

        # Get the models prediction
        mnist_shaped_img = mnist_images[img_i].reshape([1,mnist_images[img_i].shape[0],mnist_images[img_i].shape[1],1])
        logics = loaded_model[net_i].predict(mnist_shaped_img)
        label = np.argmax(logics)

        # Count the number of images the network correctly classified
        if (mnist_labels[img_i] == label):
            mnist_correct += 1

    # Print out the accuracy
    print(" ")
    print("Total Adversarial Images from Dataset: " + str(len(adv_original_labels)))
    print("Correctly Identified Adversarial Images from Dataset: " + str(adv_correct))
    print("Adversary Accuracy: " + str(float(adv_correct) / len(adv_original_labels)))
    print(" ")
    print("Total Mnist Images: " + str(len(mnist_labels)))
    print("Correctly Identified Mnist Images: " + str(mnist_correct))
    print("Mnist Accuracy: " + str(float(mnist_correct) / len(mnist_labels)))
    print("--------------------------------------------------")
    details_saver.write("Total Adversarial Images from Dataset: " + str(len(adv_original_labels)) + "\n")
    details_saver.write("Correctly Identified Adversarial Images from Dataset: " + str(adv_correct) + "\n")
    details_saver.write("Adversary Accuracy: " + str(float(adv_correct) / len(adv_original_labels)) + "\n")
    details_saver.write("\n")
    details_saver.write("Total Mnist Images: " + str(len(mnist_labels)) + "\n")
    details_saver.write("Correctly Identified Mnist Images: " + str(mnist_correct) + "\n")
    details_saver.write("Mnist Accuracy: " + str(float(mnist_correct) / len(mnist_labels)) + "\n")
    details_saver.write("--------------------------------------------------\n")


# Close the file writer
details_saver.close()
