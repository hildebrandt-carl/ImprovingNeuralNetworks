# Network Uses

This file is used to generate the teacher labels from the N-Version programming network. This file also compares all networks against the adversarial data which was generated. This folder contains 3 files and each of the uses are listed below:

* adversarial-compare.py - Takes the adversarial dataset and passes it to each of the individual networks to see how well they perform. It also takes the adversarial data and passes it to the N-Version programming network to see how well it performs.
* teacher-labels.py - This takes the N-Version program and runs all training images through the network. It then generates a set of labels which can be used for training the student network.
* teacher-compare.py -


## Running

### adversarial compare

To run this file you can use:

```
$ python3 adversarial-compare.py "<NetworkNames>"
```

Where Network Names is a comma-space separated network list. You can also run this file using the sript we provided by running:

```
$ ./run-compare.sh
```

### teacher labels

To run this file you can use the command:

```
$ python3 teacher-labels.py "<NetworkNames>"
```

Where Network names is the list of networks you want to be part of the teacher. It will save the labels in the `./Results` folder. If you want all networks to be part of the teacher you can run:

```
$ ./run-labels.sh
```

### teacher compare



## Note



## Authors

* **Carl Hildebrandt** - *Initial work* - [hildebrandt-carl](https://github.com/hildebrandt-carl)
