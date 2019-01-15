# Training neural networks

To train a neural network you need a dataset file (see section [Data format](data-format.md)) and a model configuration file (see section [Model Configuration](model-configuration.md)). The model config specifies the neural network architecture and the annotations of the dataset to use as inputs and targets.

## Usage
> python3 train.py dataset model-config

## Arguments
`dataset` : dataset file  
`model-config` : model configuration file

## Options
`-bs`, `--batch_size` : batch size for training, default value 32  
`-s`, `--steps` : number of training steps to perform, default value 1000  
`-q`, `--quiet` : don't print current loss and accuracy during training  
`-pf`, `--print_frequency` : print current loss and accuracy every x training steps

## Splitting dataset for training and testing

By default 90% of the dataset will be used to train the model and the remaining 10% to test it. A different ratio can be specified with the option `--test_data_fraction`. For example to use 80% of the dataset for training and 20% for testing run the command:

> python3 train.py dataset model-config --test_data_fraction 0.2

With the option `--test_data_file` a separate dataset file can be used to test the trained model.  
With `--train_all_examples` the complete dataset can be used to train the model.  

> python3 train.py dataset_train model-config --test_data_file dataset_test

## Cross validation

The option `-cv` or `--cross_validate` can be used to run cross validation. This will split the dataset into k parts of equal size. For every choice of k-1 parts a model is trained and tested with the remaining part. The number of parts can be specified with the option `-np` or `--num_parts`. To run multiple iterations of cross validation use the option `-nr` or `--num_repetitions`.

> python3 train.py dataset model-config --cross_validate --num_parts 10 --num_repetitions 5

## Saving and loading models

It is also possible to save and load neural network models for retraining with the options `-sm` (`--save_model`) and `-lm` (`--load_model`) respectively.

> python3 train.py dataset model-config -bs 32 -s 1000 -sm saved_model

This will create a new folder called `saved_model` with the saved model in it. To load this model for further training simply run:

> python3 train.py dataset saved_model -bs 32 -s 1000 -lm
