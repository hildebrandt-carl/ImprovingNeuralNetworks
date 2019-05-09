#!/usr/bin/env python3

# -------------------------------------------------------------------
# This generate teacher labels which can be used to train a student network
# -------------------------------------------------------------------

import sys
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from random import randint
from keras.datasets import mnist

# Check that an argument has been passed with the python script
if len(sys.argv) <= 1:
  print("This file requires the networks")
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
print("-----------------Loading Networks-----------------")
print("--------------------------------------------------")

# Load each of the networks
for net_i in range(0,len(net_names)):
    # Load the model
    l_model = load_model("../Net-Gen/FinalNetworks/" + net_names[net_i] + ".h5")
    print("Loaded network" + str(net_names[net_i]))
    loaded_model.append(l_model)

# Get the total number of networks
N = len(loaded_model)

# get test image and labels
(images, labels), _  = mnist.load_data()
images = images.reshape(images.shape[0],images.shape[1],images.shape[2],1)
images = images.astype('float32')
images /= 255.0

# Create the array to save the teacher logits
teacher_labels = []

details_saver = open("Results/teacher-labels.txt","w")

print(" ")
print("--------------------------------------------------")
print("-----N-Version Programming Networks Start---------")
print("--------------------------------------------------")
details_saver.write("--------------------------------------------------\n")
details_saver.write("-----N-Version Programming Networks Start---------\n")
details_saver.write("--------------------------------------------------\n")

# For each of the images
correct = 0
for img_i in range(0,len(labels)):

    # Get the models prediction
    shaped_img = images[img_i].reshape([1,images[img_i].shape[0],images[img_i].shape[1],1])

    # Add up the logics of each network
    logits = 0
    for net_i in range(0, N):
        logits += loaded_model[net_i].predict(shaped_img)

    # Get the final logics
    logits /= float(N)
    teacher_labels.append(logits)
    model_label = np.argmax(logits)

    # Count the number of images the network correctly classified
    if (labels[img_i] == model_label):
        correct += 1

    print("Completed image: " + str(img_i) + "/" + str(len(labels)))

# Print out information about the training
print("Total Images: " + str(len(labels)))
print("Correctly Identified: " + str(correct))
print("Accuracy: " + str(float(correct) / len(labels)))
print("--------------------------------------------------")
details_saver.write("Total Images: " + str(len(labels)) + "\n")
details_saver.write("Correctly Identified: " + str(correct) + "\n")
details_saver.write("Accuracy: " + str(float(correct) / len(labels)) + "\n")
details_saver.write("--------------------------------------------------\n")

# Reshape the labels to the correct dimensions
teacher_labels = np.concatenate(teacher_labels)

# Save the teacher labels
print("Saving teacher labels")
print("Teacher labels are of shape: " + str(np.shape(teacher_labels)))
details_saver.write("Saving teacher labels\n")
details_saver.write("Teacher labels are of shape: " + str(np.shape(teacher_labels)) + "\n")

np.save("Results/teacher_labels.npy", teacher_labels)

# Close the file writer
details_saver.close()
