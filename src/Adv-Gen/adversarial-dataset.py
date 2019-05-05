#!/usr/bin/env python3

# -------------------------------------------------------------------
# Generates a adversarial dataset using the networks listed
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
  print("Please include network names and save prefix")
  exit()

# Networks which will be loaded
net_names = sys.argv[1].split(', ')

# This defines the number of times the networs are attacked by each of the attack methods
num_images = 50

# Set to test mode
keras.backend.set_learning_phase(0)

# get source image and label
_, (images, labels) = mnist.load_data()
images = images.reshape(images.shape[0],images.shape[1],images.shape[2],1)
images = images.astype('float32')
images /= 255.0

# Create datasets
adversarial_dataset = []
original_labels = []
adversarial_labels = []

# For each of the networks
for net_i in range(0,len(net_names)):

    print("Working on Network " + net_names[net_i])
    conversion_details = []

    # Create a single dataset
    single_adversarial_dataset = []
    single_original_labels = []
    single_adversarial_labels = []

    # Load the model
    loaded_model = load_model("../Net-Gen/FinalNetworks/" + net_names[net_i] + ".h5")

    # Instantiate attack model
    fmodel = foolbox.models.KerasModel(loaded_model, bounds=(0, 1))

    # Images already done
    img_index_used = set()

    # For each of the images
    img_i = 0 
    while img_i < num_images:

        print(str(net_names[net_i]) + " - All attacks ran on " + str(img_i+1) + " images")

        # Generate a random image number
        image_number = randint(0, images.shape[0]-1)

        # If we generate an image that has already been used restart
        if image_number in img_index_used:
            # Restart the while loop
            continue

        # Save that we have used the number before
        img_index_used.add(image_number)

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
        attacks.append(foolbox.attacks.SLSQPAttack(fmodel))
        attacks.append(foolbox.attacks.LBFGSAttack(fmodel))
        attack_names = ["GradientAttack",
                        "FGSM",
                        "DeepFoolAttack",
                        "ADefAttack",
                        "SaliencyMapAttack",
                        "CarliniWagnerL2Attack",
                        "NewtonFoolAttack",
                        "RandomStartProjectedGradientDescentAttack",
                        "SLSQPAttack",
                        "LBFGSAttack"]
        
        # Apply attacks to image
        attack_number = -1
        for attack in attacks:
            adversarial = attack(images[image_number], labels[image_number])

            # Increment the attack number (Used to save what kind of attack was used)
            attack_number += 1

            # If the attack fails, adversarial will be None and a warning will be printed
            if adversarial is None:
                # Go to next attack
                continue

            # Save the Original label
            single_original_labels.append(labels[image_number])

            # Save the Adversarial label
            logics = loaded_model.predict(adversarial.reshape([1,adversarial.shape[0],adversarial.shape[1],1]))
            adv_label = np.argmax(logics)
            single_adversarial_labels.append(adv_label)

            # Save the adversarial image
            adversarial *= 255.0
            adversarial = adversarial.reshape(adversarial.shape[0],adversarial.shape[1])
            single_adversarial_dataset.append(adversarial)

            # Save conversion details
            conversion_details.append(str(img_i) + ") Image Number: " + str(image_number) + \
                                                   " - Orig Label: " + str(labels[image_number]) + \
                                                   " - Adv Label: " + str(adv_label) + \
                                                   " - Attack Type: " + str(attack_names[attack_number]) + "\n")

        # Go to next image
        img_i += 1

    # Save the single dataset into the full dataset
    adversarial_dataset.append(single_adversarial_dataset)
    np.save("Datasets/IndividualNetworks/" + net_names[net_i] + "_adversarial.npy", single_adversarial_dataset)

    # Save the single labels into the large original labels
    original_labels.append(single_original_labels)
    np.save("Datasets/IndividualNetworks/" + net_names[net_i] + "_original_labels.npy", single_original_labels)

    # Save the adversarial labels into the large adversarial labels
    adversarial_labels.append(single_adversarial_labels)
    np.save("Datasets/IndividualNetworks/" + net_names[net_i] + "_adversarial_labels.npy", single_adversarial_labels)

    # Save the details of the images which were converted accuracy:
    details_saver = open("Datasets/IndividualNetworks/results_" + net_names[net_i] + ".txt","w") 
    details_saver.write("---------------------------\n")
    details_saver.write("Conversion Details\n")
    details_saver.write("---------------------------\n")
    details_saver.writelines(conversion_details)
    details_saver.write("\n---------------------------")
    details_saver.close() 

# Stack each of the lists vertically
adversarial_dataset = np.vstack(adversarial_dataset)
original_labels = np.hstack(original_labels)
adversarial_labels = np.hstack(adversarial_labels)

# Print the shapes to make sure nothing went wrong
print("Dataset shapes are:")
print("Adversarial Dataset: " + str(adversarial_dataset.shape))
print("Original Labels Dataset: " + str(original_labels.shape))
print("Adversarial Labels Dataset: " + str(adversarial_labels.shape))

# We are done
np.save("Datasets/" + str(sys.argv[2]) + "_adversarial_dataset.npy", adversarial_dataset)
np.save("Datasets/" + str(sys.argv[2]) + "_original_labels.npy", original_labels)
np.save("Datasets/" + str(sys.argv[2]) + "_adversarial_labels.npy", adversarial_labels)
print("We are done")