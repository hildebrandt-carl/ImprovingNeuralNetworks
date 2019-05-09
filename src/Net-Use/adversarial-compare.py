#!/usr/bin/env python3

# -------------------------------------------------------------------
# Compares how well each of the networks do on advarserial images
# -------------------------------------------------------------------

import sys
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from random import randint

# Check that an argument has been passed with the python script
if len(sys.argv) <= 1:
  print("This file requires the network names")
  exit()

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
    l_model = load_model("../Net-Gen/FinalNetworks/" + net_names[net_i] + ".h5")
    print("Loaded network" + str(net_names[net_i]))
    loaded_model.append(l_model)

# Get the dataset
images = np.load("../Adv-Gen/Datasets/adversarial_dataset.npy")
images /= 255.0
original_labels = np.load("../Adv-Gen/Datasets/original_labels.npy")

# Save the testing accuracy:
details_saver = open("Results/compare_results.txt","w")

# Just to remove our program from the keras lines
print(" ")
print("--------------------------------------------------")
print("-----------Individual Networks Start--------------")
print("--------------------------------------------------")
details_saver.write("--------------------------------------------------\n")
details_saver.write("-----------Individual Networks Start--------------\n")
details_saver.write("--------------------------------------------------\n")

# Get the individual network adversarial scores
for net_i in range(0, len(loaded_model)):

    print("Testing Network " + str(net_i + 1))
    details_saver.write("Testing Network " + str(net_i + 1) + "\n")
    correct = 0

    # For each of the images
    for img_i in range(0,len(original_labels)):

        # Get the models prediction
        shaped_img = images[img_i].reshape([1,images[img_i].shape[0],images[img_i].shape[1],1])
        logics = loaded_model[net_i].predict(shaped_img)
        adv_label = np.argmax(logics)

        # Count the number of images the network correctly classified
        if (original_labels[img_i] == adv_label):
            correct += 1

    print("Total Images: " + str(len(original_labels)))
    print("Correctly Identified: " + str(correct))
    print("Accuracy: " + str(float(correct) / len(original_labels)))
    print("--------------------------------------------------")
    details_saver.write("Total Images: " + str(len(original_labels)) + "\n")
    details_saver.write("Correctly Identified: " + str(correct) + "\n")
    details_saver.write("Accuracy: " + str(float(correct) / len(original_labels)) + "\n")
    details_saver.write("--------------------------------------------------\n")

print(" ")
print("--------------------------------------------------")
print("-----N-Version Programming Networks Start---------")
print("--------------------------------------------------")
details_saver.write("\n--------------------------------------------------\n")
details_saver.write("-----N-Version Programming Networks Start---------\n")
details_saver.write("--------------------------------------------------\n")

# We are going to load different amounts of N
for N in range(1,len(loaded_model) + 1):

    correct = 0
    # Testing a N-Version programming network
    print("Testing N-Version programming with N=" + str(N))
    details_saver.write("Testing N-Version programming with N=" + str(N) + "\n")

    # For each of the images
    for img_i in range(0,len(original_labels)):

        # Get the models prediction
        shaped_img = images[img_i].reshape([1,images[img_i].shape[0],images[img_i].shape[1],1])

        # Add up the logics of each network
        logits = 0
        for net_i in range(0, N):
            logits += loaded_model[net_i].predict(shaped_img)
        # Get the final logics
        logits /= float(N)
        adv_label = np.argmax(logits)

        # Count the number of images the network correctly classified
        if (original_labels[img_i] == adv_label):
            correct += 1

    print("Total Images: " + str(len(original_labels)))
    print("Correctly Identified: " + str(correct))
    print("Accuracy: " + str(float(correct) / len(original_labels)))
    print("--------------------------------------------------")
    details_saver.write("Total Images: " + str(len(original_labels)) + "\n")
    details_saver.write("Correctly Identified: " + str(correct) + "\n")
    details_saver.write("Accuracy: " + str(float(correct) / len(original_labels)) + "\n")
    details_saver.write("--------------------------------------------------\n")

# Close the file writer
details_saver.close()
