# Network Generation

This code is used to generate the networks. The models are loaded from a text file which the user creates. The networks are then trained using a predefined set of epochs and hyper-parameters.

## Prerequists

This code looks for a network specified as an argument in directory called NetworkArchitecture. The networks need to be described using any of the following layers:

* Convolutional layer: conv, filters, activation function - `conv, 8, relu`
* Dense layer: dense, neurons, activation function - `dense, 256, relu`
* Maxpooling2D layer: maxpooling - `maxpooling`
* Flattern layer: flattern - `flattern`

## Running the code

You can run the code on an individual network using:

```
$ python3 network_creator.py "{network file}"
```

You can also run the provided scripts to train large sets of networks. To use them run:

```
$ ./tmr-networks.sh
```

## Authors

* **Carl Hildebrandt** - *Initial work* - [hildebrandt-carl](https://github.com/hildebrandt-carl)