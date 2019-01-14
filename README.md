# Deep Natural Language Classification

This repository contains Python3 programs to solve natural language classification  tasks using Deep Neural Networks.  

To train a natural language classifier you only need two files: a training dataset and a model configuration file. Dataset files must conform to a custom format and are parsed automatically. Model configurations are simple JSON files specifying the architecture, inputs and targets of a neural network classifier.

## Features

* Create custom deep neural networks by stacking different building blocks like RNNs, CNNs, GCNs and more   
* Implemented in Tensorflow which automatically enables GPU acceleration
* Create multi-task models to perform several classification tasks with a single neural network
* Numerical representation of natural language using word vectors, part-of-speech tags and syntactic dependencies
* Websocket server that processes natural language sentences using one or more trained neural networks

## Requirements

* Python 3.*  
* Tensorflow (see [installation instructions](https://www.tensorflow.org/install/), for better performance GPU use is recommended)  

#### Optional:
* Stanford CoreNLP Server to compute Part-Of-Speech tags and syntactic dependencies (see section [Setting up a Stanford CoreNLP server](documentation/corenlp-setup.md))

## Documentation

* [Quickstart guide](documentation/quickstart.md)  
A simple example of how to train a neural network for natural language classification. 

* [Dataset format](documentation/data-format.md)  
Describes a JSON format for storing datasets containing natural language sentences with different types of annotations.

* [Data generation](documentation/data-generation.md)  
Fill natural language sentences containing placeholders with short phrases to generate complete sentences.

* [Setting up a Stanford CoreNLP Server for use with this project](documentation/corenlp-setup.md)

* [Representing natural language sentences numerically](documentation/numerical-representation.md)  
Programs to add word vector, POS tag and syntactic dependency annotations to a dataset.

* [Model configuration](documentation/model-configuration.md)  
Describes how to specify neural network models using a JSON model configuration file.

* [Training neural networks](documentation/training.md)  
How to train a neural network with a dataset and model configuration file.

* [Starting a query server](documentation/query.md)  
Explains how to start a websocket server that processes natural language sentences using a trained neural network.

## License

[MIT](LICENSE)
