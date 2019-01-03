# CPM-prescription-master
Chinese patent medicine prescription prediction system by deep learning 

## Prerequisites
* PyCharm:  https://www.jetbrains.com/pycharm/download/#section=windows
* Python 3.5.2:  https://www.python.org/downloads/release/python-352/
* Tensorflow 1.1.0:  https://www.tensorflow.org/install/
## Installing

* Copy this repo directly to an empty PyCharm project folder
* File->open 

## Training

Path parameters are in the `reader.py` file.

Training Script: `model2.py`

## Test and Result

Testing script: `model2.py`

## Details

For the multi-label classification problem, we use `n hidden layer+softmax` strategy to design the structure of network. Because the feature we used is one-hot vector, which is a set of standard orthogonal basis, they are already sufficiently distinguishable. Thus, there is no need to transform the features. 
We use Cross-Entropy as loss fuction

The network structure is as follows.

The third step, network training and some results 

We randomly select 10,000 samples of the overall data from training test, let batchnum = 10, iteration times = 1, and selected the drug, of which score greater than 0.8, as the output. The sample results are as follows.
