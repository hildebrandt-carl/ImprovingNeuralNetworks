import foolbox
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import foolbox.utils


# Set to test mode
keras.backend.set_learning_phase(0)

# Load the model
json_file = open('smalldense.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('smalldense.h5')

# Instantiate attack model
fmodel = foolbox.models.KerasModel(loaded_model, bounds=(0, 1))

# get source image and label
image, label = foolbox.utils.samples(dataset='mnist', shape=(28,28))
image = image / 255.0

# apply attack on source image
attack = foolbox.attacks.FGSM(fmodel)
adversarial = attack(image, label)
# if the attack fails, adversarial will be None and a warning will be printed


print(image.shape)

plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image[0])
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Adversarial')
plt.imshow(adversarial[:, :, ::-1] / 255)  # ::-1 to convert BGR to RGB
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Difference')
difference = adversarial[:, :, ::-1] - image
plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
plt.axis('off')

plt.show()