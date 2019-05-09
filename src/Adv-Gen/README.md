# Adversarial Generation

This folder contains all files used to generate adversarial examples. This includes visualization and dataset generation.

### Prerequists

These files require you to have foolbox installed. Please read more about this in the homepage Readme. If you want to run the single adversarial file you will need matplotlib to display the images.

```
$ pip3 install matplotlib
```

## Running

There are three files in this directory, the adversarial viewer, the dataset generation tool and the dataset combination tool. To view adversarial examples you can run the following:

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

The final file is the adversarial dataset generation file. This fill will go all networks passed to it. For each network it will attempt to generate 13 images for each of the 10 attacks. For each network it will save an adversarial example dataset inside the `./Datasets/IndividualNetowks/` folder.

You can run the dataset generation python file using:

```
$ python3 adversarial-dataset.py "<NetworkNames>"
```

Where network names are a comma-space separated list. It will search for each of the networks in the `../Net-Gen/FinalNetworks/` folder. An example of running the file is shown below:

```
$ python3 adversarial-dataset.py "Network1, Network2, Network3"
```

You can also run the provided scripts to automatically run on multiple networks. To use it run:

```
$ ./run-dataset.sh
```

Once you are done you can run the dataset combination tool. This takes all the individual networks and combines them into a single dataset. The final dataset will be saved in `./Datasets/`. To run the combination tool you can run the following command:

```
$ python3 adversarial-combine.py "<NetworkNames>"
```

It will look for the networks in the `./Datasets/IndividualNetowks/` folder.

## Note

The SLSQP Attack was removed from the adversarial generation technique due to creating images which to the human eye looked more like noise than data.

## Authors

* **Carl Hildebrandt** - *Initial work* - [hildebrandt-carl](https://github.com/hildebrandt-carl)
