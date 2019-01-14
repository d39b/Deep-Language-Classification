# Quickstart guide

Consider a simple sentiment analysis task in which you are given a short movie review as natural language text. The goal is to classify the text as either 'positive', 'negative' or 'neutral'.

Examples:
"I watched the new Star Wars movie yesterday and it was great." -> positive
"Honestly, the new Star Wars movie could not have been any worse." -> negative
"The new Star Wars movie was alright, I just wish the story was less predictable." -> neutral

## Creating a dataset

To solve this classificatoin task using a neural network, you need a training dataset containing sentences with the correct sentiment labels.
A dataset file must conform to the format specified in section [Data format](data-format.md) of the documentation. Although it might be necessary to transform existing data files into this format, this approach has two advantages: the program can parse dataset files automatically and it is easy for the user to specify which information from the dataset to use as inputs and targets of a neural network. The following listing shows an example dataset file in the correct format.

```javascript
{
    "data" : [
        {
            "text" : "I watched the new Star Wars movie yesterday and it was great.",
            "sentiment" : "positive"
        },
        {
            "text" : "Honestly, the new Star Wars movie could not have been any worse."
            "sentiment" : "negative"
        },
        {
            "text" : "The new Star Wars movie was alright, I just wish the story was less predictable.",
            "sentiment" : "nail"
        },
        ...
    ],
    "metadata" : {
        "sentence_length" : 16,
        "annotations" : [
            {
                "type" : "sentence_class",
                "name" : "sentiment"
            }
        ]
    }
}
```

## Numerical Representation

Neural networks require numerical inputs like real-valued vectors or matrices. Therefore, the natural language sentences in our dataset file must be represented numerically, before we can train a neural network. For this purpose the script `create_data.py` can be used to add word vector, part-of-speech and syntactic dependency annotations to a dataset file. These annotations can then be used as inputs for the neural network. For more information see section [Numerical representation](numerical-representation.md) of the documentation.

For example to add word vectors to a dataset file run:

> python3 create_data.py find-vectors dataset.json word-vectors.txt

A word vector file can for example be obtained from [Facebook FastText](https://github.com/facebookresearch/fastText).

For the example above, adding word vectors might yield a dataset file similar to the following.

```javascript
{
    "data" : [
        {
            "text" : "I watched the new Star Wars movie yesterday and it was great."
            "sentiment" : "positive",
            "wordVectors" : [
                1,
                2,
                0,
                ...
            ]
        },
        ...
    ],
    "metadata" : {
        "sentence_length" : 32,
        "annotations" : [
            {
                "type" : "sentence_class",
                "name" : "activity"
            },
            {
                "type" : "vector_sequence",
                "vector_length" : 4,
                "name" : "wordVectors"
            }
        ]
    },
    "wordVectors" : [
        ["the", [0.17, 1.80, 0.39, -0.43]],
        ["I", [0.5612, 4.89, -0.32309, 0.65]],
        ["watched", [3.141,-1891, 0.519, -1.017]],
        ...
    ]
}
```

## Neural Network Training

To train a neural network, the model architecture must be specified in a model configuration file. The following example configuration file `model_config.json` defines a neural network consisting of two RNN layers. It uses word vectors as input and predicts the annotation `sentiment` of the dataset file. For more information about defining neural network models see section [Model configuration](model-configuration.md) of the documentation.

```javascript
{
    "layers" : [
        {
            "type" : "rnn",
            "depth" : 2,
            "cell_type" : "lstm",
            "sizes" = [500,300],
            "activation" = "relu"
        }
    ],
    "optimizer" : {
        "type" : "adadelta",
        "learning_rate" : 1.0
    },
    "losses" : [
        {
            "type" : "l2_weight",
            "lambda" : 0.0005
        }
    ]
    "inputs" : [
        {
            "type" : "vector_sequence",
            "name" : "wordVectors"
        }
    ],
    "targets" : [
        {
            "type" : "sentence_class",
            "target" : "sentiment",
            "output_layer" : "dense",
            "loss_function" : "cross_entropy"
        }
    ]
}
```

To train a neural network with our dataset file `dataset.json` and this model configuration simply run:

> python3 train.py datset.json model_config.json -bs 32 -s 1000 --save_model saved_neural_network

The model is trained for 1000 steps with batches of size 32 and saved in a new folder `saved_neural_network`.

## Inference using a query server

To use a previously trained neural network for inference, you can start a query server that receives natural language sentences on a websocket connection and returns the sentiment predicted by the neural network.

To start the query server you need a configuration file like the following.

`query-config.json`

```javascript
{
    "models" : [
        "saved_neural_network"
    ],
    "wordvector_file" : "word-vectors.txt",
    "hostname" : "localhost",
    "port" : 8765
}
```

To start the server run:

> python3 query_server.py --query_config query-config.json

To send queries to the server run:
> python3 websocket_cli_client.py --hostname localhost --port 8765
> Query: "I really liked the new film by Christopher Nolan."

The query server might then return the following JSON output with the correct sentiment classification:

```javascript
{
    "text" : "I really liked the new film by Christopher Nolan.",
    "sentiment" : "positive"
}
```
