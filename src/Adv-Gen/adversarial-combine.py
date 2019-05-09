#!/usr/bin/env python3

# -------------------------------------------------------------------
# Generates a adversarial dataset using the networks listed
# -------------------------------------------------------------------

import sys
import numpy as np

# Check that an argument has been passed with the python script
if len(sys.argv) <= 1:
  print("Please include the filename prefix")
  exit()

# Networks which will be loaded
net_names = sys.argv[1].split(', ')

# Create datasets
adversarial_dataset = []
original_labels = []
adversarial_labels = []

# For each of the networks
for net_i in range(0,len(net_names)):

    print("Working on Network " + net_names[net_i])

    # Load the individual datasets
    ad_name = "Datasets/IndividualNetworks/" + net_names[net_i] + "_adversarial.npy"
    single_adversarial_dataset = np.load(ad_name)
    orglab_name = "Datasets/IndividualNetworks/" + net_names[net_i] + "_original_labels.npy"
    single_original_labels = np.load(orglab_name)
    adlab_name = "Datasets/IndividualNetworks/" + net_names[net_i] + "_adversarial_labels.npy"
    single_adversarial_labels = np.load(adlab_name)

    # Append these to the final datasets
    adversarial_dataset.append(single_adversarial_dataset)
    original_labels.append(single_original_labels)
    adversarial_labels.append(single_adversarial_labels)

# Stack each of the lists
adversarial_dataset = np.vstack(adversarial_dataset)
original_labels = np.hstack(original_labels)
adversarial_labels = np.hstack(adversarial_labels)

# Print the shapes to make sure nothing went wrong
print("Dataset shapes are:")
print("Adversarial Dataset: " + str(adversarial_dataset.shape))
print("Original Labels Dataset: " + str(original_labels.shape))
print("Adversarial Labels Dataset: " + str(adversarial_labels.shape))

# We are done
np.save("Datasets/adversarial_dataset.npy", adversarial_dataset)
np.save("Datasets/original_labels.npy", original_labels)
np.save("Datasets/adversarial_labels.npy", adversarial_labels)
print("We are done")
