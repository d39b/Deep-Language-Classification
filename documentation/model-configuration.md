# Neural network model configuration

This section describes how to define the architecture, inputs and targets of a neural network for use with the `train.py` script.

A model configuration file is a JSON file that contains a single JSON object, which in turn contains the following required and optional key-value pairs:

Required:  
`layers` : description of the layers of the neural network  
`optimizer` : optimizer to use for training  
`inputs` : describes which annotations of the dataset file to use as inputs of the neural network  
`tasks` : describes which annotations of the dataset file to use as targets of the neural network

Optional:  
`losses` : additional regularization loss terms, e.g. L2 regularization  
`word_dropout` : word dropout regularization
`initializers` : weight and bias initialization method

After giving a simple example of a model configuration, the different model configuration options are described in more detail.

## Example

The following example show a model configuration for a model with 2 BiRNN and a GCN layer. It uses word dropout, L2 loss and the AdaDelta optimizer. The input to the neural network is the annotation `"wordVectors"` of the dataset file. The model predicts the annotation `"sentiment"` of a dataset file.

```javascript
{
    "layers" : [
        {
            "type" : "birnn",
            "depth" : 2,
            "cell_type" : "lstm",
            "sizes" = [500,300],
            "activation" = "relu"
        },
        {
            "type" : "gcn",
            "adjacency" : "dep",
            "use_gating" : true,
            "activation" : "relu"
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
    ],
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

## Layers

The `layers` key of the model configuration describes the neural network architecture of the model. Its value is an array of JSON objects, where the i-th object describes the i-th layer of the network. A JSON Object describing a layer always contains a key `type`.

When stacking different types of layers, the following constraints must be considered. The input and output of the layer types `rnn`, `birnn`, `gcn` are matrices of size sentence_length × vector_length. Since a `dense` layer always reshapes the input to be a vector. Therfore a `rnn`, `birnn` or `gcn layer` should not be stacked on top of a 'dense' layer. A `conv` layer takes a matrix as input and by default produces a vector as output. If the option `preserve_vectors` is set to `true` the output is a matrix. The `attention` layer always takes a matrix as input and outputs a vector.

The following sections describe the different layer types and their options in more detail.

#### RNN/BiRNN  

The keys for layers `rnn` and `birnn` are exactly the same.

required key | value | description
--- | --- | ---  
depth | integer | used to stack multiple RNN/BiRNN layers
cell_type | "lstm" or "gru" | RNN architecture
sizes | list of integers | output size for each RNN/BiRNN layer

optional key | value | description
--- | --- | ---
activation | "tanh", "sigmoid" or "relu" | name of an activation function to apply to the output
concat_internal_state | boolean | concat internal state to output vector, defaults to `false`
dropout_mode | "word" or "normal" | name of dropout mode to use
dropout_keep_prob | list of floats | list of keep probabilities, length of list must equal depth, values may be `null`

Example:  

```javascript
{
    "type" : "birnn",
    "depth" : 3,
    "cell_type" : "lstm",
    "sizes" : [500,500,300],
    "activation" : "relu",
    "concat_internal_state" : false,
    "dropout_mode" : "word",
    "dropout_keep_prob" : [0.95, null, 0.9]
}
```

#### GCN

required key | value | description
--- | --- | ---
adjacency | string | name of the `graph_structure` annotation of the dataset to use as input

optional key | value | description
--- | --- | ---
activation | "tanh", "sigmoid" or "relu" | name of an activation function to apply to the output
use_gating | boolean | use gating function based on edge types, defaults to `true`
dropout_mode | "word" or "normal" | name of dropout mode to use
dropout_keep_prob | float | keep probability for dropout layer, value may be `null`

Example:  

```javascript
{
    "type" : "gcn",
    "adjacency" : "dep",
    "activation" : "relu",
    "use_gating" : true,
    "dropout_mode" : "word",
    "dropout_keep_prob" : 0.95
}
```

#### Conv

required key | value | description
--- | --- | ---
num_filters | integer | number of filters per filter size
filter_sizes | list of integers | each value is number of consecutive words in a sliding window
use_max_pool | boolean | apply max pooling to output

optional key | value | description
--- | --- | ---
activation | "tanh", "sigmoid" or "relu" | name of an activation function to apply to the output
preserve_vectors | boolean | reshapes output to one vector per word, overrides `use_max_pool` to false
dropout_mode | "word" or "normal" | name of dropout mode to use
dropout_keep_prob | float | keep probability for dropout layer, value may be `null`

`preserve_vectors` must be set to `true` when trying to solve a `class_sequence` task.
`dropout_mode` cannot be set to `word` if `preserve_vectors` is set to `false`.

Example:  

```javascript
{
    "type" : "conv",
    "num_filters" : 64,
    "filter_sizes" : [2,3,4,5,6],
    "use_max_pool" : false,
    "activation" : "relu",
    "preserve_vectors" : true,
    "dropout_mode" : "word",
    "dropout_keep_prob" : 0.95
}
```

#### Dense

required key | value | description
--- | --- | ---
size | integer | number of neurons / output size

optional key | value | description
--- | --- | ---
activation | "tanh", "sigmoid" or "relu" | name of an activation function to apply to the output

Example:  

```javascript
{
    "type" : "dense",
    "size" : 512,
    "activation" : "relu"
}
```

#### Dense3

required key | value | description
--- | --- | ---
size | integer | number of neurons / output size

optional key | value | description
--- | --- | ---
activation | "tanh", "sigmoid" or "relu" | name of an activation function to apply to the output

Example:  

```javascript
{
    "type" : "dense3",
    "size" : 512,
    "activation" : "relu"
}
```

#### Attention

No required or optional keys. Cannot be used for `class_sequence` tasks.

Example:  

```javascript
{
    "type" : "attention",
}
```

## Optimizer

The `optimizer` key describes the optimization method used to minimize the loss function of the neural network. Its value is a JSON-Object with required keys `type` and `learning_rate`. Possible types are `adadelta` and `adam`. The value of `learning_rate` is a floating point number. The key `clip_gradients` also takes a floating point value and is optional. If it is supplied, the norm of each weight vector/matrix gradient is clipped to the value before updating the weights.

Example:  

```javascript
"optimizer" : {
    "type" : "adadelta",
    "learning_rate" : 1.0,
    "clip_gradients" : 3.0
}
```

## Losses

Specifies additional terms to be added to the neural network loss function. The value of this key is an array of JSON objects each of which describes a loss term. An object must contain a key `type`. The only possible type is `l2_weight`, which corresponds to L2 regularization over all the weights in the model. For this type a key `lambda` with a floating point value is required.

Example:  

```javascript
"losses" : [
    {
        "type" : "l2_weight",
        "lambda" : 0.005
    }
]
```

## Word Dropout

This key is optional and is used to apply word dropout to the input of the neural network. The value of the key is a JSON object that contains a single key `keep_prob`. The value of `keep_prob` must be a floating point number describing the probability to keep words of the input.

Example:  

```javascript
"word_dropout" : {
    "keep_prob" : 0.95
}
```

## Inputs

The `inputs` key specifies which annotations of the dataset to use as inputs for the neural network model. Its value is a list of JSON objects, each of which has a key `name` and `type` referring to the name and type of an annotation in the dataset. Only annotations of type `vector_sequence`, `class_sequence` and `graph_structure` can be used as inputs.  
An object for an input of type `class_sequence` must contain a key `embedding_size` with an integer value.
Each element of the sequence is embedded as a vector of size `embedding_size` which yields a matrix of size sentence_length × embedding_size. Inputs of type `vector_sequence` already correspond to matrices of size sentence_length × vector_length. The training script will concatenate all the matrices of `vector_sequence` and `class_sequence` inputs along the second dimension to produce a single input matrix for the neural network.  
Inputs of type `graph_structure` can be used as inputs for GCN layers. In the definition of the GCN layer the value of the key `adjacency` is the name of `graph_structure` annotation.

```javascript
"inputs" : [
    {
        "type" : "vector_sequence",
        "name" : "wordVectors"
    },
    {
        "type" : "class_sequence",
        "name" : "pos",
        "embedding_size" : 32
    },
    {
        "type" : "graph_structure",
        "name" : "dep"
    }
]
```

## Tasks

The `tasks` key specifies with annotations of the dataset to use as targets for the neural network model to predict. It is possible to specify multiple targets to train a so-called Multi-Task model. A separate output layer and loss function is added to the model for each task.  
The value of `tasks` is an array of JSON objects each describing a task. A task description contains keys `target` and `type` referring to the name and type of the corresponding annotations in the dataset. Only annotations of type `sentence_class`, `class_sequence` and `fixed_length_class_sequence` can be used as neural network targets.
Furthermore the task description must contain keys `output_layer` and `loss` specifying what output layer type and loss function to use for this task. Possible output layers are `dense` and `dense3`. `Dense` reshapes the output of the last layer of the network to a vector if necessary and then applies a matrix multiplication to produce a single output vector. `Dense3` on the other hand can only be used if the output of the last layer is 2-dimensional, i.e. one vector for each word. Then a matrix multiplication is applied to each word vector to produce one output vector for each word in the input sentence. For annotations of type `sentence_class` and `fixed_length_class_sequence` an output layer `dense` must be used, while an annotation of type `class_sequence` requires an `dense3` output layer. The softmax function is always applied to output vectors to yield a valid probability distribution.
Possible loss functions are `squared_error`, `cross_entropy` and `weighted_cross_entropy`. `weighted_cross_entropy` can be used for `class_sequence` tasks where non zero target values are sparse. Via the floating point parameters `weight_zero` and `weight_non_zero` it is possible to get higher loss for incorrect non-zero target values.

```javascript
"targets" : [
    {
        "type" : "sentence_class",
        "name" : "sentiment",
        "output_layer" : "dense",
        "loss" : "cross_entropy"
    },
    {
        "type" : "fixed_length_class_sequence",
        "name" : "intents",
        "output_layer" : "dense",
        "loss" : "squared_error"
    },
    {
        "type" : "class_sequence",
        "name" : "entityLabels"
        "output_layer" : "dense3",
        "loss" : "weighted_cross_entropy",
        "weight_zero" : 0.05,
        "weight_non_zero" : 0.95
    }
]
```

## Initializers

The `initializers` key specifies how to initialize model variables. The value of this key is a dictionary that can contain a key `weight` and/or `bias` to specify weight and/or bias variable initializers respectively. Possible initializers are `zeros`, `random_normal`, `variance_scaling`, `glorot_uniform`, `glorot_normal` and `uniform_unit_scaling`. The default weight initializer is `variance_scaling` and the default bias initializer is `zeros`.

```javascript
"initializers" : {
    "weight" : "glorot_uniform",
    "bias" : "zeros"
}
```
