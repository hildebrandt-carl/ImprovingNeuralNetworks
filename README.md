# ImprovingNeuralNetworks

This project is aimed at improving the Robustness of Convolutional Neural Networks. We will be doing this using N-Version programming to generate a more robust network. This network will then be used as a teacher to train new networks.

## Prerequists

We will also be using [Bethgelab's foolbox](https://github.com/bethgelab/foolbox) as well as IBM's [CNN-Cert](https://arxiv.org/abs/1811.12395). Each of these have dependencies which can be installed using the following commands:

```
sudo apt-get install build-essential autoconf libtool pkg-config python-opengl python-imaging python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev python3-tk python3-dev
```

TODO:
Confirmed
python3-tk

We will be using [pip](https://pypi.org/project/pip/), pythons package manager, to install the python packages we need. To install [pip](https://pypi.org/project/pip/) you can run:

```
$ sudo apt install python3-pip
```

Our networks were generated using Keras. Full installation guides can be found in the [Keras installation documentation](https://keras.io/#installation). You have to install one of three backend engines. We will be using the Tensorflow engine. More information about installing Tensorflow can be found in the [Tensorflow installation documentation](https://www.tensorflow.org/install/pip). You can install this using:

```
# GPU version of tensorflow
$ pip3 install tensorflow-gpu
# CPU version of tensorflow
$ pip3 install tensorflow
```

Once Tensorflow is installed we can install Keras:

```
$ pip3 install keras
```

We can test that Keras was installed correctly using:

```
$ python3
>>> import keras
```

We will need to install [Bethgelab's foolbox](https://github.com/bethgelab/foolbox). More information about installing foolbox can be found in [foolbox's installation documentation](https://foolbox.readthedocs.io/en/latest/user/installation.html). We are using the development version which can be installed using:

```
$ pip3 install https://github.com/bethgelab/foolbox/archive/master.zip
```

## Project Sections

* [Source](./src/) - This contains all the source code I used for training networks. Creating datasets. Creating N-version setup. Distilling the networks, and finally testing the networks.
* [Documents](./docs) - This contains the project proposal, the progress reports, the presentation and the report. 

## Video

A video of the project working can be found on youtube: [https://youtu.be/u_tLKoU_lro](https://youtu.be/u_tLKoU_lro)

## Acknowledgments

This code will be using versions of [Bethgelab's foolbox](https://github.com/bethgelab/foolbox) as well as IBM's [CNN-Cert](https://arxiv.org/abs/1811.12395).

## Authors

* **Carl Hildebrandt** - *Initial work* - [hildebrandt-carl](https://github.com/hildebrandt-carl)
