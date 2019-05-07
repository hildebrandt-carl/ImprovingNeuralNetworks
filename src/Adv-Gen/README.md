# Adversarial Generation

This folder contains all files used to generate adversarial examples. This includes visualization and dataset generation.

### Prerequists

These files require you to have foolbox installed. Please read more about this in the homepage Readme. If you want to run the single adversarial file you will need matplotlib to display the images.

```
$ pip3 install matplotlib
```

## Running

There are two files in this directory, the viewer and the dataset generation tool. To view adversarial examples you can run the following:

```
$ python3 adversarial-dataset.py "<NetworkName>" "<AttackType>"
```

It will search for the network name in `../Net-Gen/FinalNetworks/`. Please note it will search for a file with a .h5 file name. The attacks types can be chosen from a list:

* "GradientAttack" - Gradient Attack
* "GradientSignAttack" - FGSM
* "DeepFool" - Deep Fool Attack
* "ADef" - ADef Attack
* "SaliencyMap" - Saliency Map Attack
* CarliniWagner L2 Attack
* "Newton" - Newton Fool Attack
* "ProjectedGradient" - Random Start Projected Gradient Descent Attack
* "SLSQPAttack" - SLSQP Attack
* "LBFGS" - LBFGS Attack

An example of running the above command is:

```
$ python3 adversarial-dataset.py "network1" "ADef"
```

The final file is the adversarial dataset generation file. This fill will go all networks passed to it. For each network it will attempt to generate 12 images for each of the 10 attacks. For each network it will save an adversarial example dataset inside the `./Datasets/IndividualNetowks/` folder. It will also generate a complete adversarial dataset which contains the adversarial images for all networks in the `./Datasets/` folder.

You can run the dataset generation python file using:

```
$ python3 adversarial-dataset.py "<NetworkNames>" "<SavePrefix>"
```

Where network names are a comma-space seperated list. It will search for each of the networks in the `../Net-Gen/FinalNetworks/` folder. Save prefix is a prefix appended to the output file name. An example of running the file is shown below:

```
$ python3 adversarial-dataset.py "Network1, Network2, Network3" "run1"
```

You can also run the provided scripts to automatically run on multiple networks. To use it run:

```
$ ./run.sh
```

## Note

The SLSQP Attack was removed from the adversarial generation technique due to creating images which to the human eye looked more like noise than data.

## Authors

* **Carl Hildebrandt** - *Initial work* - [hildebrandt-carl](https://github.com/hildebrandt-carl)
