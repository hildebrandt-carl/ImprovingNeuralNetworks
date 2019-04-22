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
  print("This file requires whether or not to use TMR")
  exit()

# Total networks we will be testing
total_networks = 15
net_names = ['small_dense', 'large_dense', 'conv']
program_type = sys.argv[1]

print("herererererere")
print(program_type)

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

# Change which networks are loaded
if program_type == "nmr":
    # Load each of the networks
    for i in range(1, total_networks + 1):
        # Load the model
        l_model = load_model("../Net-Gen/FinalNetworks/network" + str(i) + ".h5")
        print("Loaded network" + str(i) )
        loaded_model.append(l_model)

    # Get the dataset
    images = np.load("../Adv-Gen/Datasets/nmr_adversarial_dataset.npy")
    images /= 255.0
    original_labels = np.load("../Adv-Gen/Datasets/nmr_original_labels.npy")
elif program_type == "tmr":
    # Load each of the networks
    for i in range(0, len(net_names)):
        # Load the model
        l_model = load_model("../Net-Gen/FinalNetworks/" + net_names[i] + ".h5")
        print("Loaded network " + net_names[i] )
        loaded_model.append(l_model)

    # Get the dataset
    images = np.load("../Adv-Gen/Datasets/tmr_adversarial_dataset.npy")
    images /= 255.0
    original_labels = np.load("../Adv-Gen/Datasets/tmr_original_labels.npy")
else:
    print("Program type is not known")
    exit()

# Save the testing accuracy:
details_saver = open("Results/results_" + program_type + ".txt","w") 

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
print("---------------NMR Networks Start-----------------")
print("--------------------------------------------------")
details_saver.write("\n--------------------------------------------------\n")
details_saver.write("---------------NMR Networks Start-----------------\n")
details_saver.write("--------------------------------------------------\n")

# We are going to load different amounts of N
for N in range(1,len(loaded_model) + 1):

    correct = 0
    # Testing a TMR network 
    print("Testing NMR with N=" + str(N))
    details_saver.write("Testing NMR with N=" + str(N) + "\n")

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