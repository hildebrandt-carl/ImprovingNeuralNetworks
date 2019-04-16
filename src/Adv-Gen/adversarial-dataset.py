#!/usr/bin/env python3

# -------------------------------------------------------------------
# Generates a adversarial dataset using the networks listed
# -------------------------------------------------------------------


import foolbox
import keras
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import model_from_json
from random import randint

# Networks which will be loaded
net_names = ['small_dense', 'large_dense', 'conv']
num_images = 100

# Set to test mode
keras.backend.set_learning_phase(0)

# get source image and label
_, (images, labels) = mnist.load_data()
images = images.reshape(images.shape[0],images.shape[1],images.shape[2],1)
images = images.astype('float32')
images /= 255.0

# Create datasets containing all zeros
adversarial_dataset = np.zeros((num_images * len(net_names),images.shape[1],images.shape[2]))
original_labels = np.zeros(num_images * len(net_names))
adversarial_labels = np.zeros(num_images * len(net_names))

# For each of the networks
for net_i in range(0,len(net_names)):

    print("Working on Network " + net_names[net_i])

    # Create a single dataset
    single_adversarial_dataset = np.zeros((num_images,images.shape[1],images.shape[2]))
    single_original_labels = np.zeros(num_images)
    single_adversarial_labels = np.zeros(num_images)

    # Load the model
    json_file = open("../Net-Gen/Networks/" + net_names[net_i] + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("../Net-Gen/Networks/" + net_names[net_i] + '.h5')

    # Instantiate attack model
    fmodel = foolbox.models.KerasModel(loaded_model, bounds=(0, 1))

    # Images already done
    img_index_used = set()

    # For each of the images
    img_i = 0 
    while img_i < num_images:

        # Generate a random image number
        image_number = randint(0, images.shape[0]-1)

        # If we generate an image that has already been used restart
        if image_number in img_index_used:
            # Restart the while loop
            continue

        # Save that we have used the number before
        img_index_used.add(image_number)

        # Apply attack on source image
        attack = foolbox.attacks.FGSM(fmodel)
        adversarial = attack(images[image_number], labels[image_number])

        # If the attack fails, adversarial will be None and a warning will be printed
        if adversarial is None:
            # Restart the while loop
            continue

        # Save the Original label
        single_original_labels[img_i] = labels[image_number]

        # Save the Adversarial label
        logics = loaded_model.predict(adversarial.reshape([1,adversarial.shape[0],adversarial.shape[1],1]))
        adv_label = np.argmax(logics)
        single_adversarial_labels[img_i] = adv_label

        # Save the adversarial image
        adversarial *= 255.0
        adversarial = adversarial.reshape(adversarial.shape[0],adversarial.shape[1])
        single_adversarial_dataset[img_i] = adversarial

        # Go to next image
        print(str(img_i) + ") Orig Label: " + str(labels[image_number]) + " Adv Label: " + str(adv_label))
        img_i += 1

    # Save the single dataset into the full dataset
    adversarial_dataset[num_images * net_i:num_images * (net_i + 1)] = single_adversarial_dataset
    np.save("Datasets/" + net_names[net_i] + "_adversarial.npy", single_adversarial_dataset)

    # Save the single labels into the large original labels
    original_labels[num_images * net_i:num_images * (net_i + 1)] = single_original_labels
    np.save("Datasets/" + net_names[net_i] + "_original_labels.npy", single_original_labels)

    # Save the adversarial labels into the large adversarial labels
    adversarial_labels[num_images * net_i:num_images * (net_i + 1)] = single_adversarial_labels
    np.save("Datasets/" + net_names[net_i] + "_adversarial_labels.npy", single_adversarial_labels)

# We are done
np.save("Datasets/adversarial_dataset.npy", adversarial_dataset)
np.save("Datasets/original_labels.npy", original_labels)
np.save("Datasets/adversarial_labels.npy", adversarial_labels)
print("We are done")