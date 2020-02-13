# FDIA detection for DR

This repository contains reproducable demo for our CSR and CNN models.

#### Prerequisite packages
* numpy
* pandas
* scipy
* sklearn
* tensorflow
* tslearn

#### Files
1. [csr.py](csr.py) - Python implementation of our CSR model
2. [cnn.py](cnn.py) - Python implementation of our CNN model
3. train-forecasts.7z - Compressed set of demand forecasts used to train the models
3. test-forecasts.7z - Compressed set of demand forecasts used to test the models

### How to run
1. First unzip the compressed data files
2. To run the csr model `python csr.py` this will execute with k = 300. 
You can pass -k argument to change the value of k.
e.g. `python csr.py -k 100`
3. To run the cnn model `python cnn.py`
