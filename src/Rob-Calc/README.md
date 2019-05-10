# Robustness Calculation

This file is used to generate formal verification of networks. This file uses CNN-Cert which has been added to this document. IBM's CNN-Cert can be found at their [repository](https://github.com/CNN-Cert/CNN-Cert).

## Prerequists

You will need the following packages to run CNN-Cert. You can install them by running:

```
$ pip3 install numba
$ pip3 install pandas
$ pip3 install posix_ipc
```

You will also need to add CNN-Cert to your python path. You can do that by running:

```
$ cd ../CNN-Cert
$ export PYTHONPATH=`pwd`:$PYTHONPATH
$ cd -
```

## Running

You are able to run their tool using the following command:

```
$ python3 robustness.py "<NetworkName>" "<Metric>"
```

Where the metric is either:

* L1
* L2

Thus an example of running this program is:

```
$ python3 robustness.py "Teacher" "L1"
```

## Authors

* **Carl Hildebrandt** - *Initial work* - [hildebrandt-carl](https://github.com/hildebrandt-carl)
