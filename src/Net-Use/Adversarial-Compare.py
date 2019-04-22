#!/usr/bin/env python3

# -------------------------------------------------------------------
# Compares how well each of the networks do on advarserial images
# -------------------------------------------------------------------

import sys
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from random import randint

# Check that an argument has been passed with the python script
if len(sys.argv) <= 1:
  print("This file requires whether or not to use TMR")
  exit()

# Total networks we will be testing
total_networks = 100
net_names = ['small_dense', 'large_dense', 'conv']
TMR = bool(sys.argv[1])

# Set to test mode
keras.backend.set_learning_phase(0)

# Create variable for each of the loaded models
loaded_model = []

# The dataset variables
images = None
original_labels = None

# Change which networks are loaded
if TMR == False:
    # Load each of the networks
    for i in range(1,total_networks + 1):
        # Load the model
        l_model = load_model("../Net-Gen/FinalNetworks/network" + str(i) + ".h5")
        loaded_model.append(l_model)

    # Get the dataset
    images = np.load("../Adv-Gen/Datasets/nmr-adversarial_dataset.npy")
    images /= 255.0
    original_labels = np.load("../Adv-Gen/Datasets/nmr-original_labels.npy")
else:
    # Load each of the networks
    for i in range(0,len(net_names)):
        # Load the model
        l_model = load_model("../Net-Gen/FinalNetworks/" + net_names[i] + ".h5")
        loaded_model.append(l_model)

    # Get the dataset
    images = np.load("../Adv-Gen/Datasets/tmr-adversarial_dataset.npy")
    images /= 255.0
    original_labels = np.load("../Adv-Gen/Datasets/tmr-original_labels.npy")


# Get the individual network adversarial scores
for net_i in range(0, len(loaded_model)):

    print("Testing Network " + str(net_i + 1))
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


correct = 0
# We are going to load different amounts of N
for N in range(0,len(loaded_model)):

    # Testing a TMR network 
    print("Testing NMR with N=" + str(N))

    # For each of the images
    for img_i in range(0,len(original_labels)):

        # Get the models prediction
        shaped_img = images[img_i].reshape([1,images[img_i].shape[0],images[img_i].shape[1],1])

        # Add up the logics of each network
        logits = 0
        for net_i in range(0, len(loaded_model)):
            logits += loaded_model[net_i].predict(shaped_img)
        # Get the final logics
        logits /= float(len(loaded_model))
        adv_label = np.argmax(logits)

        # Count the number of images the network correctly classified
        if (original_labels[img_i] == adv_label):
            correct += 1

    print("Total Images: " + str(len(original_labels)))
    print("Correctly Identified: " + str(correct))
    print("Accuracy: " + str(float(correct) / len(original_labels)))
    print("--------------------------------------------------")