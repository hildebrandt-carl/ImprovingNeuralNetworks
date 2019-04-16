#!/usr/bin/env python3

# -------------------------------------------------------------------
# Compares how well each of the networks do on advarserial images
# -------------------------------------------------------------------

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from random import randint

# Networks which will be loaded
net_names = ['small_dense', 'large_dense', 'conv']

# Create variable for each of the loaded models
loaded_model = []

# Load each of the networks
for i in range(0,len(net_names)):
    # Load the model
    json_file = open("../Net-Gen/Networks/" + net_names[i] + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    l_model = model_from_json(loaded_model_json)
    l_model.load_weights("../Net-Gen/Networks/" + net_names[i] + '.h5')
    loaded_model.append(l_model)

# Set to test mode
keras.backend.set_learning_phase(0)

# Get the dataset
images = np.load("../Adv-Gen/Datasets/adversarial_dataset.npy")
images /= 255.0

original_labels = np.load("../Adv-Gen/Datasets/original_labels.npy")

# For each of the networks
for net_i in range(0,len(net_names)):

    print("Testing Network " + net_names[net_i])
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

# Testing a TMR network 
print("Testing NMR with N=" + str(len(net_names)))

correct = 0

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