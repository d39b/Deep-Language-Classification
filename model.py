import numpy as np
import tensorflow as tf
import json
from pathlib import Path

class Model:
    """
    A class used to build neural network models.

    Contains functions to add different neural network layers (like fully connected, CNN, RNN, GCN, etc.),
    regularization, loss functions and optimizers. Also contains function to train and test a model.

    To build concrete neural networks, you should subclass this class. Add your train/test/query
    operations to the train_operations/test_operations/query_operations lists.

    Attributes
    ----------
    graph : tensorflow.Graph
        Tensorflow graph containing the operations of a model
    session : tensorflow.Session
        Tensorflow session used to run the computations defined by a model
    variables : List
        list of trainable Tensorflow variables of a model
    variables_initialized : bool
        truth value denoting whether the variables of the Tensorflow graph have been initialized
    losses : List
        list of Tensorflow operations representing loss functions
    dropout_placeholders : dict
        dictionary with dropout placeholder tensors as keys and float keep probabilities as values
    train_operations : List
        list of Tensoflow operations to run for training
    test_operations : List
        list of Tensorflow operations to run for testing
    query_operations : List
        list of Tensorflow operations to run for querying
    """


    activations = {
        "relu" : tf.nn.relu,
        "sigmoid" : tf.sigmoid,
        "tanh" : tf.tanh
    }

    def __init__(self):
        self.graph = tf.Graph()
        self.variables = []
        self.session = tf.Session(graph=self.graph)
        self.num_rnn_cells = 0
        self.variables_initialized = False
        self.next_name_id = 0
        self.losses = []
        self.dropout_placeholders = {}
        self.train_operations = {}
        self.test_operations = {}
        self.query_operations = {}
        self.DEFAULT_WEIGHT_INITIALIZER = "variance_scaling"
        self.DEFAULT_BIAS_INITIALIZER = "zeros"

    def get_initial_value(self, name, shape):
        """
        Returns an initialization operation.

        Parameters
        ----------
        name : str
            string denoting the initialization method to be used
        shape: List
            list of integers defining the shape of the initial value
        """

        with self.graph.as_default():
            if name == "zeros":
                return tf.zeros(shape)
            elif name == "random_normal":
                return tf.random_normal(shape)
            elif name == "variance_scaling":
                return tf.variance_scaling_initializer()(shape)
            elif name == "glorot_uniform":
                return tf.glorot_uniform_initializer()(shape)
            elif name == "glorot_normal":
                return tf.glorot_normal_initializer()(shape)
            elif name == "uniform_unit_scaling":
                return tf.uniform_unit_scaling_initializer()(shape)
            else:
                print("Error: unknown initializer")


    def add_dense_layer(self,inputs,size,use_bias=True,activation=None,weight_initializer=None,bias_initializer=None):
        """
        Adds a dense (or fully connected) layer to a neural network model.

        Reshapes the input to 2 dimensions. If no weight or bias initializer is passed, the initializers defined by self.DEFAULT_WEIGHT_INITIALIZER or self.DEFAULT_BIAS_INITIALIZER are used.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            input of the dense layer to be added
        size : int
            number of units in the dense layer
        use_bias : bool, optional
            if true, the units use a bias term (default is True)
        activation : str, optional
            string denoting a activation function (default is None)
        weight_initializer : str, optional
            string denoting a weight initialization function (default is None)
        bias_initializer : str, optional
            string denoting a bias initialization function (default is None)
        """

        with self.graph.as_default():
            if weight_initializer is None:
                weight_initializer = self.DEFAULT_WEIGHT_INITIALIZER
            if bias_initializer is None:
                bias_initializer = self.DEFAULT_BIAS_INITIALIZER

            input_shape = self.get_tensor_shape(inputs)
            #reshape input to [batch_size,x], i.e. one vector per example in the batch
            if len(input_shape) > 2:
                #vector size is product of the sizes of all dimensions (except first)
                new_size = 1
                for i in range(1,len(input_shape)):
                    new_size *= input_shape[i]

                input_shape = [-1,new_size]
                inputs = tf.reshape(inputs,input_shape)

            #create weight matrix and bias vector and initialization operations
            shape = [input_shape[-1],size]
            bias_shape = [size]
            weights = tf.Variable(self.get_initial_value(weight_initializer,shape),name=self.get_name("dense_weights"))
            bias = tf.Variable(self.get_initial_value(bias_initializer,bias_shape),name=self.get_name("dense_bias"))

            #add weight matrix and bias vector to variable list
            self.add_variable(weights)
            self.add_variable(bias)

            output = tf.nn.bias_add(tf.matmul(inputs,weights,name=self.get_name("dense_matmul")),bias,name=self.get_name("dense_bias_add"))
            return self.add_activation(output,activation)


    def add_activation(self,inputs,activation):
        """
        Applies the given activation function to the input tensor.

        Parameters
        ----------
        inputs: tensorflow.Tensor
            tensor to apply activation function to
        activation: str or None
            string denoting the name of the activation function
        """

        with self.graph.as_default():
            if activation is None:
                return inputs
            elif activation not in Model.activations:
                print("Warning: unknown activation function {}".format(activation))
                return inputs
            else:
                return Model.activations[activation](inputs,name=self.get_name(activation))


    def add_dense3_layer(self,inputs,size,use_bias=True,activation=None,weight_initializer=None,bias_initializer=None):
        """
        Adds a dense (or fully connected) layer to a neural network model.

        Use this method for 3 dimensional inputs.
        If no weight or bias initializer is passed, the initializers defined by self.DEFAULT_WEIGHT_INITIALIZER or self.DEFAULT_BIAS_INITIALIZER are used.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            input of the dense layer to be added
        size : int
            number of units in the dense layer
        use_bias : bool, optional
            if true, the units use a bias term (default is True)
        activation : str, optional
            string denoting a activation function (default is None)
        weight_initializer : str, optional
            string denoting a weight initialization function (default is None)
        bias_initializer : str, optional
            string denoting a bias initialization function (default is None)
        """

        with self.graph.as_default():
            if weight_initializer is None:
                weight_initializer = self.DEFAULT_WEIGHT_INITIALIZER
            if bias_initializer is None:
                bias_initializer = self.DEFAULT_BIAS_INITIALIZER

            input_shape = self.get_tensor_shape(inputs)

            shape = [input_shape[-1],size]
            bias_shape = [size]
            weights = tf.Variable(self.get_initial_value(weight_initializer,shape),name=self.get_name("dense3_weights"))
            bias = tf.Variable(self.get_initial_value(bias_initializer,bias_shape),name=self.get_name("dense3_bias"))
            self.add_variable(weights)
            self.add_variable(bias)
            output = tf.nn.bias_add(tf.tensordot(inputs,weights,[[2],[0]],name=self.get_name("dense3_tensordot")),bias,name=self.get_name("dense3_bias_add"))
            return self.add_activation(output,activation)

    #Assumes input is 3-dimensional: batch_size x sentence_length x embedding_size
    def add_attention_layer(self,inputs,weight_initializer=None,bias_initializer=None):
        """
        Adds an attention layer to a neural network model.

        This method expects 3-dimensinal input.
        If no weight or bias initializer is passed, the initializers defined
        by self.DEFAULT_WEIGHT_INITIALIZER or self.DEFAULT_BIAS_INITIALIZER are used.

        Returns a 2-dimensional tensor.

        Parameters
        ----------
        inputs : Tensor
            input tensor
        weight_initializer : str, optional
            string denoting a weight initialization function (default is None)
        bias_initializer : str, optional
            string denoting a bias initialization function (default is None)
        """

        with self.graph.as_default():
            if weight_initializer is None:
                weight_initializer = self.DEFAULT_WEIGHT_INITIALIZER
            if bias_initializer is None:
                bias_initializer = self.DEFAULT_BIAS_INITIALIZER

            m = tf.tanh(inputs)
            attention = self.add_dense3_layer(m,1,"tanh",weight_initializer=weight_initializer,bias_initializer=bias_initializer)
            attention = tf.squeeze(attention,axis=2,name=self.get_name("attention_squeeze"))
            attention = tf.nn.softmax(attention,axis=1,name=self.get_name("attention_softmax"))
            #attention value is batch_size x sentence_length
            #sum input vectors weighted by attention values
            r = tf.einsum('ijl,ij->il',inputs,attention)
            h = tf.tanh(r)
            return h


    def add_cnn_layer(self,inputs,num_filters,filter_sizes,strides=[1,1,1,1],activation=None,use_max_pool=True,preserve_vectors=False,kernel_initializer=None,bias_initializer=None):
        """
        Adds a convolutional layer to a neural network model.

        Special version of convolution for 3 dimensional inputs in NLP tasks.
        The input is seen as a batch of sentences, where each sentences is a sequence of word vectors.
        For each filter size k, windows of k consecutive word vectors are convolved.
        If no weight or bias initializer is passed, the initializers defined by self.DEFAULT_WEIGHT_INITIALIZER or self.DEFAULT_BIAS_INITIALIZER are used.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            input of the convolutional layer to be added
        num_filters : int
            number of filters to use
        filter_sizes : List
            list of integers indicating the sizes of sliding windows
        strides : List
            list of integers indicating the strides for each window dimension
        activation : str, optional
            string denoting a activation function (default is None)
        use_max_pool : bool, optional
            if true, max pooling is applied after convolution (default is True)
        kernel_initializer : str, optional
            string denoting a weight initialization function (default is None)
        bias_initializer : str, optional
            string denoting a bias initialization function (default is None)
        """

        with self.graph.as_default():
            if kernel_initializer is None:
                kernel_initializer = self.DEFAULT_WEIGHT_INITIALIZER
            if bias_initializer is None:
                bias_initializer = self.DEFAULT_BIAS_INITIALIZER

            input_shape = self.get_tensor_shape(inputs)
            if len(input_shape) == 3:
                inputs = tf.expand_dims(inputs,3)

            sentence_length = input_shape[1]
            embedding_size = input_shape[2]

            outputs = []
            for filter_size in filter_sizes:
                scope_name = self.get_name("conv")+"_fs"+str(filter_size)
                with tf.name_scope(scope_name) as scope:
                    padded_inputs = inputs
                    if preserve_vectors and filter_size > 1:
                        #pad inputs to make sure that outputs for different filter sizes have same dimensions
                        #"SAME" padding scheme cannot be used, because it produces the same size across all dimensions, i.e. also across dim 3 = word_vector_length
                        #what we want is to keep dim only along dimension 2
                        padding_sizes = None
                        num_pads = filter_size // 2
                        if filter_size % 2 == 1:
                            padding_sizes = tf.constant([[0,0],[num_pads,num_pads],[0,0],[0,0]])
                            padded_inputs = tf.pad(inputs,padding_sizes,"CONSTANT")
                        else:
                            padding_sizes = tf.constant([[0,0],[num_pads-1,num_pads],[0,0],[0,0]])
                            padded_inputs = tf.pad(inputs,padding_sizes,"CONSTANT")

                    filter_shape = [filter_size,embedding_size,1,num_filters]
                    bias_shape = [num_filters]
                    filter_weights = tf.Variable(self.get_initial_value(kernel_initializer,filter_shape),name=self.get_name("filter_weights"))
                    bias = tf.Variable(self.get_initial_value(bias_initializer,bias_shape),name=self.get_name("filter_bias"))
                    self.add_variable(filter_weights)
                    self.add_variable(bias)

                    conv = tf.nn.conv2d(padded_inputs,filter_weights,strides=strides,padding="VALID",name="conv")
                    activ = self.add_activation(tf.nn.bias_add(conv,bias,name="bias_add"),activation)
                    if not preserve_vectors and use_max_pool:
                        activ = tf.nn.max_pool(activ,
                                ksize=[1,sentence_length-filter_size+1,1,1],
                                strides=[1,1,1,1],
                                padding="VALID",
                                name="max_pool")
                    outputs.append(activ)

            total_num_filters = num_filters*len(filter_sizes)
            pool = None
            if not preserve_vectors:
                if use_max_pool:
                    pool = tf.concat(outputs,3,name="concat")
                    pool = tf.reshape(pool,[-1,total_num_filters],name="reshape")
                else:
                    pool = tf.concat(outputs,1,name="concat")
                    pool = tf.squeeze(pool,axis=2,name="squeeze")
            else:
                pool = tf.concat(outputs,3,name="concat")
                pool = tf.squeeze(pool,axis=2,name="squeeze")

            return pool


    def get_rnn_cell(self,cell_type,hidden_size):
        """
        Returns a RNN cell object used to create RNN layers.

        Parameters
        ----------
        cell_type : str
            cell type to be used, possible values are 'lstm' or 'gru'
        hidden_size : int
            output size of the RNN cell
        """

        name = cell_type + "_cell_" + str(self.num_rnn_cells)
        cell = None

        with self.graph.as_default():
            if cell_type == "lstm":
                cell = tf.nn.rnn_cell.LSTMCell(hidden_size,name=name)
            elif cell_type == "gru":
                cell = tf.nn.rnn_cell.GRUCell(hidden_size,name=name)

        self.num_rnn_cells += 1
        return cell

    def static_rnn(self,cell,inputs,dtype=tf.float32):
        """
        Creates tensor operations that compute RNN output for a given cell and input tensor.

        Returns a tuple (outputs, state). 'outputs' and 'state' are lists of output and state tensors respecitvely.

        Parameters
        ----------
        cell : tf.nn.rnn_cell.*
            cell used for RNN computation
        inputs : List of Tensor
            sequence of tensors used as input to the RNN
        """

        with self.graph.as_default():
            batch_size = tf.shape(inputs[0])[0]
            state = cell.zero_state(batch_size,dtype)
            outputs = []
            states = []
            for i in inputs:
                output, state = cell(i,state)
                outputs.append(output)
                #LSTM returns a tuple (c,h) where h corresponds to the last output, GRU just returns the internal state
                if isinstance(state,tf.nn.rnn_cell.LSTMStateTuple):
                    states.append(state[0])
                else:
                    states.append(state)

            return (outputs,states)

    def add_rnn_layer(self, inputs, cell_type, hidden_size, activation=None,unstack_input=True,stack_output=True,stack_axis=1,concat_internal_state=False,dropout_mode=None,dropout_keep_prob=None):
        """
        Adds a RNN layer to a neural network model.

        Input should be 3-dimensional as a single tensor. To use a list of 2-dimensional tensors as input set 'unstack_input' to false.
        The input is seen as a batch of sentences, where each sentences is a sequence of word vectors.

        Returns a 3-dimensional output tensor or a list of 2-dimensional tensors if 'stack_output' is set to false.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            input of RNN layer
        cell_type : str
            RNN cell type to use, possible values 'lstm' or 'gru'
        hidden_size : int
            output size of RNN
        activation : str, optional
            string denoting a activation function (default is None)
        unstack_input : bool, optional
            if true, input is unstacked along 'stack_axis' to produce list of tensors (default is True)
        stack_output : bool, optional
            if ture, list of output tensors is stacked along 'stack_axis' (default is True)
        stack_axis : int, optional
            axis along which to unstack/stack input/output tensors (default is 1)
        concat_internal_state : bool, optional
            if true, state tensors of the RNN are concatenated to output tensors (default is False)
        dropout_mode : str, optional
            name of dropout mode to use, possible values are "word" and "normal" (default is None)
        dropout_keep_prob: float or None, optional
            dropout keep probability, values may be None to indicate no dropout usage (default is None)
        """

        with self.graph.as_default():
            inputs_unstacked = inputs
            if unstack_input:
                inputs_unstacked = tf.unstack(inputs,axis=stack_axis,name=self.get_name("rnn_unstack"))

            cell = self.get_rnn_cell(cell_type,hidden_size)
            outputs = None
            state = None
            if not concat_internal_state:
                outputs, state = tf.nn.static_rnn(cell,inputs_unstacked,dtype=tf.float32)
            else:
                #concat internal state and output vectors at each time step
                outputs, state = self.static_rnn(cell,inputs_unstacked,dtype=tf.float32)
                new_outputs = []
                for i in range(len(inputs_unstacked)):
                    new_outputs.append(tf.concat([outputs[i],state[i]],-1))
                outputs = new_outputs

            self.add_variables(self.get_rnn_cell_variables(cell))

            if stack_output:
                outputs = self.add_activation(tf.stack(outputs,axis=stack_axis,name=self.get_name("rnn_stack")),activation)
            else:
                for i in range(len(outputs)):
                    outputs[i] = self.add_activation(outputs[i],activation)

            outputs = self.add_dropout_layer(outputs,dropout_keep_prob,mode=dropout_mode)

            return outputs

    def add_multi_rnn_layer(self,inputs,depth,cell_type,hidden_sizes,activation=None,stack_output=True,stack_axis=1,concat_internal_state=False,dropout_mode=None,dropout_keep_prob=None):
        """
        Adds multiple RNN layers to a neural network model.

        Input should be 3-dimensional tensor.
        The input is seen as a batch of sentences, where each sentences is a sequence of word vectors.

        Returns a 3-dimensional output tensor or a list of 2-dimensional tensors if 'stack_output' is set to false.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            input of RNN layer
        depth : int
            number of RNN layers to add
        cell_type : str
            RNN cell type to use, possible values 'lstm' or 'gru'
        hidden_sizes : List of int
            output sizes of the RNN layers
        activation : str, optional
            string denoting a activation function (default is None)
        stack_output : bool, optional
            if ture, list of output tensors is stacked along 'stack_axis' (default is True)
        stack_axis : int, optional
            axis along which to unstack/stack input/output tensors (default is 1)
        concat_internal_state : bool, optional
            if true, state tensors of the RNN are concatenated to output tensors (default is False)
        dropout_mode : str, optional
            name of dropout mode to use, possible values are "word" and "normal" (default is None)
        dropout_keep_prob: None or list, optional
            list of dropout keep probabilities for each layer, values may be None to indicate no dropout usage (default is None)
        """

        with self.graph.as_default():
            #correctly parse dropout arguments
            dropout_mode, dropout_keep_prob = self.check_dropout_arguments(dropout_mode, dropout_keep_prob, depth)

            current_output = self.add_rnn_layer(inputs,cell_type,hidden_sizes[0],activation=activation,stack_output=False,stack_axis=stack_axis,concat_internal_state=concat_internal_state,dropout_mode=dropout_mode,dropout_keep_prob=dropout_keep_prob[0])
            for i in range(depth-1):
                current_output = self.add_rnn_layer(current_output,cell_type,hidden_sizes[i+1],activation=activation,unstack_input=False,stack_output=False,stack_axis=stack_axis,concat_internal_state=concat_internal_state,dropout_mode=dropout_mode,dropout_keep_prob=dropout_keep_prob[i+1])

            if stack_output:
                current_output = tf.stack(current_output,axis=stack_axis,name=self.get_name("rnn_stack"))

            return current_output

    def add_birnn_layer(self, inputs, cell_type, hidden_size, activation=None,unstack_input=True,stack_output=True,stack_axis=1,concat_internal_state=False,dropout_mode=None,dropout_keep_prob=None):
        """
        Adds a BiRNN layer to a neural network model.

        Input should be 3-dimensional as a single tensor. To use a list of 2-dimensional tensors as input set 'unstack_input' to false.
        The input is seen as a batch of sentences, where each sentences is a sequence of word vectors.

        Returns a 3-dimensional output tensor or a list of 2-dimensional tensors if 'stack_output' is set to false.


        Parameters
        ----------
        inputs : tensorflow.Tensor
            input of BiRNN layer
        cell_type : str
            RNN cell type to use, possible values 'lstm' or 'gru'
        hidden_size : int
            output size of RNN
        activation : str, optional
            string denoting a activation function (default is None)
        unstack_input : bool, optional
            if true, input is unstacked along 'stack_axis' to produce list of tensors (default is True)
        stack_output : bool, optional
            if ture, list of output tensors is stacked along 'stack_axis' (default is True)
        stack_axis : int, optional
            axis along which to unstack/stack input/output tensors (default is 1)
        concat_internal_state : bool, optional
            if true, state tensors of the RNN are concatenated to output tensors (default is False)
        dropout_mode : str, optional
            name of dropout mode to use, possible values are "word" and "normal" (default is None)
        dropout_keep_prob: float or None, optional
            dropout keep probability, values may be None to indicate no dropout usage (default is None)
        """

        with self.graph.as_default():
            inputs_unstacked = inputs
            if unstack_input:
                inputs_unstacked = tf.unstack(inputs,axis=stack_axis,name=self.get_name("birnn_unstack"))

            forward_cell = self.get_rnn_cell(cell_type, hidden_size)
            backward_cell = self.get_rnn_cell(cell_type, hidden_size)

            inputs_reversed = []
            length = len(inputs_unstacked)
            for i in range(length):
                inputs_reversed.append(inputs_unstacked[length-i-1])

            f_outputs = None
            f_state = None
            b_outputs = None
            b_state = None

            if not concat_internal_state:
                f_outputs, f_state = tf.nn.static_rnn(forward_cell,inputs_unstacked,dtype=tf.float32)
                b_outputs, b_state = tf.nn.static_rnn(backward_cell,inputs_reversed,dtype=tf.float32)
            else:
                f_outputs, f_state = self.static_rnn(forward_cell,inputs_unstacked,dtype=tf.float32)
                b_outputs, b_state = self.static_rnn(backward_cell,inputs_reversed,dtype=tf.float32)

            self.add_variables(self.get_rnn_cell_variables(forward_cell))
            self.add_variables(self.get_rnn_cell_variables(backward_cell))

            outputs = []
            for i in range(length):
                if not concat_internal_state:
                    outputs.append(self.add_activation(tf.concat([f_outputs[i],b_outputs[length-i-1]],axis=1,name=self.get_name("birnn_concat")),activation))
                else:
                    outputs.append(self.add_activation(tf.concat([f_outputs[i],b_outputs[length-i-1],f_state[i],b_state[length-i-1]],axis=1,name=self.get_name("birnn_concat")),activation))

            if stack_output:
                outputs = tf.stack(outputs,axis=stack_axis,name=self.get_name("birnn_stack"))

            outputs = self.add_dropout_layer(outputs,dropout_keep_prob,mode=dropout_mode)

            return outputs

    def add_multi_birnn_layer(self,inputs,depth,cell_type,hidden_sizes,activation=None,stack_output=True,stack_axis=1,concat_internal_state=False,dropout_mode=None,dropout_keep_prob=None):
        """
        Adds multiple BiRNN layers to a neural network model.

        Input should be 3-dimensional tensor.
        The input is seen as a batch of sentences, where each sentences is a sequence of word vectors.

        Returns a 3-dimensional output tensor or a list of 2-dimensional tensors if 'stack_output' is set to false.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            input of RNN layer
        depth : int
            number of RNN layers to add
        cell_type : str
            RNN cell type to use, possible values 'lstm' or 'gru'
        hidden_sizes : List of int
            output sizes of the RNN layers
        activation : str, optional
            string denoting a activation function (default is None)
        stack_output : bool, optional
            if ture, list of output tensors is stacked along 'stack_axis' (default is True)
        stack_axis : int, optional
            axis along which to unstack/stack input/output tensors (default is 1)
        concat_internal_state : bool, optional
            if true, state tensors of the RNN are concatenated to output tensors (default is False)
        dropout_mode : str, optional
            name of dropout mode to use, possible values are "word" and "normal" (default is None)
        dropout_keep_prob: None or list, optional
            list of dropout keep probabilities for each layer, values may be None to indicate no dropout usage (default is None)
        """

        with self.graph.as_default():
            #correctly parse dropout arguments
            dropout_mode, dropout_keep_prob = self.check_dropout_arguments(dropout_mode,dropout_keep_prob,depth)

            current_output = self.add_birnn_layer(inputs,cell_type,hidden_sizes[0],activation=activation,stack_output=False,stack_axis=stack_axis,concat_internal_state=concat_internal_state,dropout_mode=dropout_mode,dropout_keep_prob=dropout_keep_prob[0])
            for i in range(depth-1):
                current_output = self.add_birnn_layer(current_output,cell_type,hidden_sizes[i+1],activation=activation,unstack_input=False,stack_output=False,stack_axis=stack_axis,concat_internal_state=concat_internal_state,dropout_mode=dropout_mode,dropout_keep_prob=dropout_keep_prob[i+1])

            if stack_output:
                current_output = tf.stack(current_output,axis=stack_axis)

            return current_output

    def add_gcn_layer(self,inputs,adjacency,num_tags,activation=None,weight_initializer=None,bias_initializer=None,use_gating=True):
        """
        Adds a Graph Convolutional Network layer to a neural network model.

        First input should be 3-dimensional tensor.
        The input is seen as a batch of sentences, where each sentences is a sequence of word vectors.
        Second input is a 3-dimensional batch of graph adjacency matrices. An adjacency matrix describes
        dependencies between different words of a sentence. If the adjacecny matrix has value t at position (i,j),
        this corresponds to a dependency of type t between word i and word j. The value t can be positive or negative
        depending on wether i is the head or the dependent of the word dependency.
        The number of possible types must be given via the 'num_tags' argument.

        Returns a 3-dimensional output tensor.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            3-dimensional input tensor for GCN layer
        adjacency : tensorflow.Tensor
            batch of graph adjacency matrices
        num_tags : int
            number of possible edge types.
        activation : str, optional
            string denoting a activation function (default is None)
        weight_initializer : str, optional
            string denoting a weight initialization function (default is None)
        bias_initializer : str, optional
            string denoting a bias initialization function (default is None)
        use_gating : bool, optional
            if true, apply gating function based on all edge types (default is True)
        """

        with self.graph.as_default():
            if weight_initializer is None:
                weight_initializer = self.DEFAULT_WEIGHT_INITIALIZER
            if bias_initializer is None:
                bias_initializer = self.DEFAULT_BIAS_INITIALIZER

            inputs_shape = self.get_tensor_shape(inputs)
            sentence_length = inputs_shape[1]
            embedding_size = inputs_shape[2]

            #split adjacency matrix into two matrices for incoming and outgoing edges
            adjacency_tags_d1 = tf.maximum(adjacency,0)
            adjacency_tags_d2 = tf.abs(tf.minimum(adjacency,0))

            #create binary adjacency matrices for incoming and outgoing edges
            adjacency_d1 = tf.cast(tf.minimum(adjacency_tags_d1,1),dtype=tf.float32)
            adjacency_d2 = tf.cast(tf.minimum(adjacency_tags_d2,1),dtype=tf.float32)

            #adjacency matrix with positive type indices and 1 on the diagonal for loop edges
            adjacency_tags = tf.add(adjacency_tags_d1,adjacency_tags_d2)
            adjacency_tags = tf.add(adjacency_tags,tf.eye(sentence_length,batch_shape=[tf.shape(adjacency)[0]],dtype=adjacency.dtype))

            # mask used to later reset bias vector of edge type 0 (i.e. no edge) to 0
            adjacency_tags_mask = tf.tile(tf.expand_dims(tf.add(adjacency_d1,adjacency_d2),-1),[1,1,1,embedding_size])

            #create weight and bias matrices for simple GCN
            weights_loop = tf.Variable(self.get_initial_value(weight_initializer,[embedding_size,embedding_size]), name=self.get_name("gcn_weights_loop"))
            weights_d1 = tf.Variable(self.get_initial_value(weight_initializer,[embedding_size,embedding_size]), name=self.get_name("gcn_weights_d1"))
            weights_d2 = tf.Variable(self.get_initial_value(weight_initializer,[embedding_size,embedding_size]), name=self.get_name("gcn_weights_d2"))
            tag_biases = tf.Variable(self.get_initial_value(bias_initializer,[num_tags,embedding_size]), name=self.get_name("gcn_tag_biases"))
            self.add_variables([weights_loop,weights_d1,weights_d2,tag_biases])

            gated_adjacency_d1 = adjacency_d1
            gated_adjacency_d2 = adjacency_d2
            gated_inputs_loop = inputs

            #if gating mechanism is not used, a new vector for word i is simply the sum of
            # 1. vector for word i multiplied with weights_loop matrix
            # 2. sum of vectors j with edge (i,j) in graph multiplied with weights_d1
            # 3. sum of vectors j with edge (j,i) in graph multiplied with weights_d2
            # 4. for each incoming or outgoing edge a bias vector according to the type of the edge
            if use_gating:
                #create weight vectors and bias terms
                gate_weights_loop = tf.Variable(self.get_initial_value(weight_initializer,[embedding_size]), name=self.get_name("gcn_gate_weights_loop"))
                gate_weights_d1 = tf.Variable(self.get_initial_value(weight_initializer,[embedding_size]), name=self.get_name("gcn_gate_weights_d1"))
                gate_weights_d2 = tf.Variable(self.get_initial_value(weight_initializer,[embedding_size]), name=self.get_name("gcn_gate_weights_d2"))
                gate_biases = tf.Variable(self.get_initial_value(bias_initializer,[num_tags]), name=self.get_name("gcn_gate_biases"))
                self.add_variables([gate_weights_loop,gate_weights_d1,gate_weights_d2,gate_biases])

                gate_loop = tf.tensordot(inputs,gate_weights_loop,[[2],[0]])
                gate_d1 = tf.tensordot(inputs,gate_weights_d1,[[2],[0]])
                gate_d2 = tf.tensordot(inputs,gate_weights_d2,[[2],[0]])

                gate_bias = tf.nn.embedding_lookup(gate_biases,adjacency_tags)
                gate_loop = tf.tile(tf.expand_dims(tf.add(gate_loop,tf.matrix_diag_part(gate_bias)),axis=-1),[1,1,embedding_size])
                gate_d1 = tf.tile(tf.expand_dims(gate_d1,axis=-1),[1,1,sentence_length])
                gate_d2 = tf.tile(tf.expand_dims(gate_d2,axis=-1),[1,1,sentence_length])
                gate_d1 = tf.multiply(tf.add(gate_d1,gate_bias),adjacency_d1)
                gate_d2 = tf.multiply(tf.add(gate_d2,gate_bias),adjacency_d2)
                gate_d1 = tf.sigmoid(gate_d1)
                gate_d2 = tf.sigmoid(gate_d2)
                gate_loop = tf.sigmoid(gate_loop)

                #this is necessary because sigmoid function at 0 is 1/2, therefore word pairs without dependency have to be reset to value 0
                gated_adjacency_d1 = tf.multiply(adjacency_d1,gate_d1)
                gated_adjacency_d2 = tf.multiply(adjacency_d2,gate_d2)
                gated_inputs_loop = tf.multiply(inputs,gate_loop)

            #add the vectors of all neighbours together
            output_d1 = tf.einsum('ijl,ilk->ijk',gated_adjacency_d1,inputs)
            output_d2 = tf.einsum('ijl,ilk->ijk',gated_adjacency_d2,inputs)
            #multiply with corresponding weight matrix, this is possible because vector addition and matrix multiplication can be distributed
            output_d1 = tf.tensordot(output_d1,weights_d1,[[2],[0]])
            output_d2 = tf.tensordot(output_d2,weights_d2,[[2],[0]])
            output_loop = tf.tensordot(gated_inputs_loop,weights_loop,[[2],[0]])

            #multiplication with adjacency_tags_mask sets bias vectors for 0 tag back to 0
            output_bias = tf.multiply(adjacency_tags_mask,tf.nn.embedding_lookup(tag_biases,adjacency_tags))
            output_bias = tf.reduce_sum(output_bias,axis=2)

            complete_sum = tf.add_n([output_loop,output_d1,output_d2,output_bias])
            complete_sum = self.add_activation(complete_sum,activation)
            return complete_sum

    def check_dropout_arguments(self,dropout_mode,dropout_keep_prob,depth):
        result_mode = None
        result_keep_prob = [None for i in range(depth)]
        if dropout_mode != "normal" and dropout_mode != "word":
            if not dropout_mode is None:
                print("Error: unknown dropout mode {}".format(dropout_mode))
            return result_mode, result_keep_prob

        if not isinstance(dropout_keep_prob,list):
            print("Error: value of dropout_keep_prob is not a list")
            return result_mode, result_keep_prob

        if len(dropout_keep_prob) != depth:
            print("Error: not enough values in dropout_keep_prob")
            return result_mode, result_keep_prob

        return dropout_mode, dropout_keep_prob

    def add_dropout_layer(self,inputs,keep_prob,mode="word"):
        """
        Adds a dropout layer to the model.

        'Inputs' may be a Tensor or a list of Tensors. The 'mode' keyword argument
        can be used to either apply normal or word dropout.

        Parameters
        ----------
        inputs : Tensor or list of Tensor
            Tensor(s) to apply dropout to
        keep_prob : float
            probability to keep elements in the input Tensor(s)
        mode : str, optional
            dropout mode to use, possible modes are "word" and "normal" (default is "word")
        """

        with self.graph.as_default():
            dropout_function = None
            if mode is None or keep_prob is None:
                return inputs
            elif mode == "word":
                dropout_function = self.add_word_dropout
            elif mode == "normal":
                dropout_function = self.add_dropout
            else:
                print("Warning: unknown dropout mode {}".format(mode))
                return inputs

            result = None
            #inputs may be a list of tensors, for e.g. in RNN functions with stack_output = False
            if isinstance(inputs,list):
                result = []
                for i in range(len(inputs)):
                    result.append(dropout_function(inputs[i],keep_prob))
            else:
                result = dropout_function(inputs,keep_prob)

            return result


    #Asusmes that inputs is of shape [batch_size,sentence_length,embedding_size]
    def add_word_dropout(self,inputs,keep_prob):
        """
        Applies word dropout to the input tensor.

        Input tensor should be 2 or 3-dimensional.
        If dimension is 3: For particular indices i,j of the first and second dimension the elements
        inputs[i,j,k] (for all k) are either all kept or all set to zero.
        If dimension is 2: For a particular index i of the first dimension the elements inputs[i,k]
        (for all k) are either all kept or all set to zero.
        A placeholder Tensor is created and added as key to the self.dropout_placeholders dictionary
        with a value with the argument 'keep_prob'.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            2 or 3-dimensional input tensor
        keep_prob : float
            probability to keep a word vector
        """

        with self.graph.as_default():
            dropout_placeholder = tf.placeholder(tf.float32,name=self.get_name("word_dropout_placeholder"))
            self.dropout_placeholders[dropout_placeholder] = keep_prob
            tensor_shape = tf.shape(inputs)
            num_dim = len(self.get_tensor_shape(inputs))
            noise_shape = None
            #when using unstacked outputs of RNN/BiRNN layer input might be 2-dimensional
            if num_dim == 2:
                noise_shape = [tensor_shape[0],1]
            #normal case
            elif num_dim == 3:
                noise_shape = [tensor_shape[0],tensor_shape[1],1]
            dropout = tf.nn.dropout(inputs,dropout_placeholder,noise_shape=noise_shape,name=self.get_name("word_dropout"))
            return dropout

    def add_gcn_dropout(self,inputs,word_dropout):
        """
        Applies dropout to a GCN adjacency matrix according to word dropout tensor.

        Input tensor should be 3-dimensional batch of adjacency matrices and a word dropout tensor.
        If a paritcular word vector was dropped out, all edges incident to this word in the adjacency matrix
        are also dropped.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            3-dimensional batch of adjacency matrices
        word_dropout : tensorflow.Tensor
            word dropout tensor
        """

        with self.graph.as_default():
            shape = self.get_tensor_shape(word_dropout)
            sentence_length = shape[1]

            #get word dropout tensor and create dropout mask for adjacency matrix, i.e. if a word is dropped the corresponding row of the adjacecny matrix is set to 0
            word_dropout_mask = self.graph.get_tensor_by_name(self.get_tensor_base_name(word_dropout) + "/Floor:0")
            word_dropout_mask = tf.cast(word_dropout_mask,tf.int32)
            gcn_dropout_mask = tf.slice(word_dropout_mask,[0,0,0],[-1,-1,1])
            gcn_dropout_mask = tf.tile(gcn_dropout_mask,[1,1,sentence_length])
            gcn_dropout_mask_t = tf.transpose(gcn_dropout_mask,perm=[0,2,1])
            result = tf.multiply(inputs,gcn_dropout_mask)
            result = tf.multiply(inputs,gcn_dropout_mask_t)
            return result

    def add_dropout(self,inputs,keep_prob):
        """
        Applies dropout to the input tensor.

        Input tensor can have any shape. Each element of the tensor is kept with probability
        'keep_prob' and set to zero with probability (1-'keep_prob').

        Parameters
        ----------
        inputs : tensorflow.Tensor
            input tensor of any shape
        keep_prob : float
            probability to keep an element of the input tensor
        """

        with self.graph.as_default():
            dropout_placeholder = tf.placeholder(tf.float32,name=self.get_name("dropout_placeholder"))
            self.dropout_placeholders[dropout_placeholder] = keep_prob
            dropout = tf.nn.dropout(inputs,dropout_placeholder)
            return dropout

    def add_l2_weight_loss(self,l2_lambda):
        """
        Adds a L2 weight loss function to this neural network model.

        Parameters
        ----------
        l2_lambda : float or Tensor
            multiplicative constant for L2 loss term
        """

        with self.graph.as_default():
            losses = []
            for v in self.variables:
                losses.append(tf.nn.l2_loss(v))
            l2_loss = tf.multiply(l2_lambda,tf.add_n(losses))
            self.losses.append(l2_loss)
            return l2_loss

    def add_cross_entropy_loss(self,logits,targets):
        """
        Adds a cross entropy loss function to this model.

        Parameters
        ----------
        logits : tensorflow.Tensor
            tensor representing the prediction of a neural network model
        targets : tensorflow.Tensor
            tensor containing correct predictions
        """

        with self.graph.as_default():
            cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=targets)
            cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
            self.losses.append(cross_entropy_loss)
            return cross_entropy_loss

    def add_sparse_cross_entropy_loss(self,logits,targets):
        """
        Adds a sparse cross entropy loss function to this model.

        Parameters
        ----------
        logits : tensorflow.Tensor
            tensor representing the prediction of a neural network model
        targets : tensorflow.Tensor
            tensor containing correct predictions
        """

        with self.graph.as_default():
            sparse_cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=targets)
            sparse_cross_entropy_loss = tf.reduce_mean(sparse_cross_entropy_loss)
            self.losses.append(sparse_cross_entropy_loss)
            return sparse_cross_entropy_loss

    def add_weighted_cross_entropy_loss(self,logits,targets,num_classes,weight_zero,weight_non_zero):
        """
        Adds a weighted cross entropy loss function to this model.

        Loss for zero and non-zero target values can be weighted with different multiplicative constants.

        Parameters
        ----------
        logits : tensorflow.Tensor
            tensor representing the prediction of a neural network model
        targets : tensorflow.Tensor
            tensor containing correct predictions
        num_classes : int
            number of possible target values
        weight_zero : float or Tensor
            constant multiplied with loss for zero target values
        weight_non_zero : float or Tensor
            constant multiplied with loss for non-zero target values
        """

        with self.graph.as_default():
            targets_shape = self.get_tensor_shape(targets)
            tile_shape = [1 for i in range(len(targets_shape))]
            tile_shape.append(num_classes)


            one_hot_targets = tf.one_hot(targets,num_classes,on_value=1.0,off_value=0.0,dtype=tf.float32,axis=-1)
            epsilon = 0.000001
            probabilities = tf.nn.softmax(tf.add(logits,epsilon),axis=-1)
            log_prob = tf.log(probabilities)

            # create masks for zero and non zero target values
            mask = tf.minimum(targets,1)
            reverse_mask = tf.subtract(tf.cast(1,tf.int32),mask)
            non_zero_count = tf.cast(tf.add(tf.reduce_sum(mask),1),tf.float32)
            zero_count = tf.cast(tf.add(tf.reduce_sum(reverse_mask),1),tf.float32)
            mask = tf.cast(tf.tile(tf.expand_dims(mask,-1),tile_shape),tf.float32)
            reverse_mask = tf.cast(tf.tile(tf.expand_dims(reverse_mask,-1),tile_shape),tf.float32)

            loss_non_zero = tf.multiply(tf.reduce_sum(tf.multiply(tf.multiply(mask,log_prob),one_hot_targets)),-1)
            loss_zero = tf.multiply(tf.reduce_sum(tf.multiply(tf.multiply(reverse_mask,log_prob),one_hot_targets)),-1)

            loss_non_zero = tf.divide(loss_non_zero,non_zero_count)
            loss_zero = tf.divide(loss_zero,zero_count)

            loss = tf.add(tf.multiply(loss_non_zero,weight_non_zero),tf.multiply(loss_zero,weight_zero))
            self.losses.append(loss)
            return loss

    def add_squared_error_loss(self,outputs,targets):
        """
        Adds a squared error loss function to this model.

        Parameters
        ----------
        outputs : tensorflow.Tensor
            tensor output of a neural network model
        targets : tensorflow.Tensor
            tensor containing correct outputs
        """

        with self.graph.as_default():
            squared_error = tf.nn.l2_loss(tf.subtract(outputs,targets))
            self.losses.append(squared_error)
            return squared_error


    def add_total_loss(self):
        """
        Returns a tensor representing the sum of all the loss tensors of this neural network model.
        """

        with self.graph.as_default():
            total_loss = tf.add_n(self.losses)
            return total_loss

    def add_optimizer(self,optimizer,loss,learning_rate,global_step=None,clip_gradients=None):
        """
        Adds an optimizer to this neural network model and applies it to a loss function.

        Returns the training operation to optimize the loss function and the global step tensor.

        Parameters
        ----------
        optimizer : str
            name of the optimizer to use, possible values 'adam' and 'adadelta'
        loss : tensorflow.Tensor
            loss tensor to optimize
        learning_rate : float or Tensor
            learning rate the optimizer should use
        global_step : Tensor, optional
            global step tensor used by the optimizer (default is None)
        clip_gradients : float or Tensor, optional
            if not None, all weight gradients are clipped to this value (default is None)
        """

        if optimizer == "adam":
            return self.add_adam_optimizer(loss,learning_rate,global_step=global_step,clip_gradients=clip_gradients)
        elif optimizer == "adadelta":
            return self.add_adadelta_optimizer(loss,learning_rate,global_step=global_step,clip_gradients=clip_gradients)

    def add_adam_optimizer(self,loss,learning_rate,global_step=None,clip_gradients=None):
        """
        Adds an Adam optimizer to this neural network model and applies it to a loss function.

        Returns the training operation to optimize the loss function and the global step tensor.

        Parameters
        ----------
        loss : tensorflow.Tensor
            loss tensor to optimize
        learning_rate : float or Tensor
            learning rate the optimizer should use
        global_step : Tensor, optional
            global step tensor used by the optimizer (default is None)
        clip_gradients : float or Tensor, optional
            if not None, all weight gradients are clipped to this value (default is None)
        """

        with self.graph.as_default():
            if global_step is None:
                global_step = tf.Variable(0,name="global_step", trainable=False)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss)

            #clip values to interval [-clip_gradients,clip_gradients]
            clipped_grads_and_vars = None
            if not clip_gradients is None:
                clip_max = clip_gradients
                clip_min = (-1) * clip_gradients
                clipped_grads_and_vars = [(tf.clip_by_value(gv[0],clip_min,clip_max),gv[1]) for gv in grads_and_vars]
            else:
                clipped_grads_and_vars = grads_and_vars

            train_op = optimizer.apply_gradients(clipped_grads_and_vars,global_step=global_step)
            return (train_op,global_step)

    def add_adadelta_optimizer(self,loss,learning_rate,global_step=None,clip_gradients=None):
        """
        Adds an AdaDelta optimizer to this neural network model and applies it to a loss function.

        Returns the training operation to optimize the loss function and the global step tensor.

        Parameters
        ----------
        loss : tensorflow.Tensor
            loss tensor to optimize
        learning_rate : float or Tensor
            learning rate the optimizer should use
        global_step : Tensor, optional
            global step tensor used by the optimizer (default is None)
        clip_gradients : float or Tensor, optional
            if not None, all weight gradients are clipped to this value (default is None)
        """

        with self.graph.as_default():
            if global_step is None:
                global_step = tf.Variable(0,name="global_step", trainable=False)

            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss)
            clipped_grads_and_vars = None
            if not clip_gradients is None:
                clip_max = clip_gradients
                clip_min = (-1) * clip_gradients
                clipped_grads_and_vars = [(tf.clip_by_value(gv[0],clip_min,clip_max),gv[1]) for gv in grads_and_vars]
            else:
                clipped_grads_and_vars = grads_and_vars

            train_op = optimizer.apply_gradients(clipped_grads_and_vars,global_step=global_step)
            return (train_op,global_step)

    def add_class_prediction(self,inputs):
        """
        Returns a tensor containing class predictions.

        Input tensor represents a probability distribution over possible classes. The argmax
        function is applied to the last dimension of this tensor to produce class predictions.
        I.e. the resulting tensor contains the indices of the classes with highest probability.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            input tensor to apply argmax function to
        """

        with self.graph.as_default():
            predictions = tf.argmax(inputs,axis=-1,output_type=tf.int32)
            return predictions

    #assumes that prediction input is of size [batch_size], if not it is computed to that using argmax
    def add_classification_accuracy(self,predictions,targets):
        """
        Returns the classification accuracy for a given pair of tensors representing
        predicted and correct classes.

        Parameters
        ----------
        predictions : tensorflow.Tensor
            tensor representing class predictions
        targets : tensorflow.Tensor
            tensor representing correct predictions
        """

        with self.graph.as_default():
            correct_pred = tf.equal(predictions,targets)
            accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
            return accuracy

    def add_weighted_classification_accuracy(self,predictions,targets):
        """
        Returns the classification accuracy for zero and non-zero target values separately.

        Parameters
        ----------
        predictions : tensorflow.Tensor
            tensor representing class predictions
        targets : tensorflow.Tensor
            tensor representing correct predictions
        """

        with self.graph.as_default():
            #create mask for zero and non-zero targets
            mask_non_zero = tf.minimum(targets,1)
            mask_zero = tf.maximum(tf.add(tf.multiply(targets,-1),1),0)
            count_non_zero = tf.cast(tf.reduce_sum(mask_non_zero),tf.float32)
            count_zero = tf.cast(tf.reduce_sum(mask_zero),tf.float32)

            targets_non_zero = tf.add(targets,tf.multiply(mask_zero,-1))
            correct_pred_non_zero = tf.cast(tf.equal(predictions,targets_non_zero),tf.int32)
            count_correct_pred_non_zero = tf.cast(tf.reduce_sum(correct_pred_non_zero),tf.float32)
            accuracy_non_zero = tf.divide(count_correct_pred_non_zero,count_non_zero)

            targets_zero = tf.multiply(targets,-1)
            correct_pred_zero = tf.cast(tf.equal(predictions,targets_zero),tf.int32)
            count_correct_pred_zero = tf.cast(tf.reduce_sum(correct_pred_zero),tf.float32)
            accuracy_zero = tf.divide(count_correct_pred_zero,count_zero)

            return (accuracy_zero, accuracy_non_zero, count_zero, count_non_zero)

    def initialize_variables(self):
        """
        Initializes all variables of this neural network model.
        """

        with self.graph.as_default():
            self.run_operations(tf.global_variables_initializer(),{})
            self.variables_initialized = True

    def train(self,inputs,num_steps,batch_size,quiet=False,print_frequency=10):
        """
        Trains this neural network model.

        The inputs and targets must be given as a dictionary with the same keys as
        self.train_placehoders. Training batches are created by randomly sampling from
        the input.

        Parameters
        ----------
        inputs : dict
            dictionary mapping strings to values, must contain the same keys as self.train_placeholders
        num_steps : int
            number of training steps to perform
        batch_size : int
            number of examples in a training batch
        quiet : bool, optional
            if true, don't output training progress (default is False)
        print_frequency : int, optional
            print training progress every x steps (default is 10)
        """

        if not self.variables_initialized:
            self.initialize_variables()

        for i in range(num_steps):
            #sample batch of random elements from inputs
            batch_inputs = self.get_batch(inputs,batch_size)
            feed_dict = {}
            for j in self.train_placeholders:
                feed_dict[self.train_placeholders[j]] = batch_inputs[j]

            #this could also be done with feed_dict.update(self.dropout_placeholders)
            for dropout_placeholder, keep_prob in self.dropout_placeholders.items():
                feed_dict[dropout_placeholder] = keep_prob

            results = self.run_operations(self.train_operations,feed_dict)
            if not quiet:
                if i % print_frequency == 0:
                    self.print_train_status(results)

    def print_train_status(self,results):
        """
        Print training status. This method should be overriden by subclasses.
        """

        return 0


    def test(self,inputs,quiet=False):
        """
        Tests this neural network model.

        The inputs and targets must be given as a dictionary with the same keys as
        self.train_placehoders.

        Parameters
        ----------
        inputs : dict
            dictionary mapping strings to values, must contain the same keys as self.train_placeholders
        quiet : bool, optional
            if true, dont' print anything (default is False)
        """

        if not self.variables_initialized:
            self.initialize_variables()

        feed_dict = {}
        for i in inputs:
            feed_dict[self.train_placeholders[i]] = inputs[i]

        for dropout_placeholder in self.dropout_placeholders:
            feed_dict[dropout_placeholder] = 1.0

        results = self.run_operations(self.test_operations,feed_dict)
        if not quiet:
            self.print_test_status(results)
        return results

    def test_in_batches(self,inputs,batch_size,quiet=False):
        """
        Tests this neural network model.

        The inputs and targets must be given as a dictionary with the same keys as
        self.train_placehoders. The test dataset is split into batches. Returns the
        outputs for each batch as a list.

        Parameters
        ----------
        inputs : dict
            dictionary mapping strings to values, must contain the same keys as self.train_placeholders
        batch_size : int
            number of examples in a testing batch
        quiet : bool, optional
            if true, don't print anything (default is False)
        """

        if not self.variables_initialized:
            self.initialize_variables()

        length = inputs[list(inputs.keys())[0]].shape[0]
        offset = 0
        results = []
        while offset < length:
            batch_inputs = self.get_next_batch(inputs,batch_size,offset)
            tmp_results = self.test(batch_inputs,quiet=True)
            results.append(tmp_results)
            offset = offset + batch_size

        return results

    def query(self,inputs):
        """
        Uses this neural network model to process a query.

        The inputsnd targets must be given as a dictionary with the same keys as
        self.query_placeholders. Returns the values of the tensors in self.query_operations.

        Parameters
        ----------
        inputs : dict
            dictionary mapping strings to values, must contain the same keys as self.train_placeholders
        """

        if not self.variables_initialized:
            self.initialize_variables()

        feed_dict = {}
        for i in inputs:
            feed_dict[self.query_placeholders[i]] = inputs[i]

        for dropout_placeholder in self.dropout_placeholders:
            feed_dict[dropout_placeholder] = 1.0

        results = self.run_operations(self.query_operations,feed_dict)
        return results

    def query_in_batches(self,inputs,batch_size,quiet=False):
        """
        Uses this neural network model to process multiple queries.

        The inputs must be given as a dictionary with the same keys as
        self.query_placehoders. The input dataset is split into batches. Returns the
        outputs for each batch as a list.

        Parameters
        ----------
        inputs : dict
            dictionary mapping strings to values, must contain the same keys as self.query_placeholders
        batch_size : int
            number of examples in a batch
        quiet : bool, optional
            if true, don't print anything (default is False)
        """

        if not self.variables_initialized:
            self.initialize_variables()

        length = inputs[list(inputs.keys())[0]].shape[0]
        offset = 0
        results = []
        while offset < length:
            batch_inputs = self.get_next_batch(inputs,batch_size,offset)
            tmp_results = self.query(batch_inputs)
            results.append(tmp_results)
            offset = offset + batch_size

        return results

    def print_test_status(self,results):
        """
        Print testing tatus. This method should be overriden by subclasses.
        """

        return 0

    def run_operations(self,operations,feed_dict):
        """
        Print training status. This method should be overriden by subclasses.

        Parameters
        ----------
        operations : list or dict
            list or dictionary of operations to run
        feed_dict : dict
            dictionary of inputs to feed the neural network
        """

        return self.session.run(operations,feed_dict)


    def get_batch(self,inputs,batch_size=None):
        """
        Randomly samples a batch of given size from a set of inputs.

        Parameters
        ----------
        inputs : dict
            dictionary of input values
        batch_size : int
            number of examples in the resulting batch
        """

        if batch_size is None:
            return inputs
        else:
            length = inputs[list(inputs.keys())[0]].shape[0]
            batch_indices = np.random.randint(0,length,batch_size);
            batch_inputs = {}
            for i in inputs:
                batch_inputs[i] = inputs[i][batch_indices]
            return batch_inputs

    def get_next_batch(self,inputs,batch_size,offset):
        """
        Returns a batch obtained by taking consecutive elements starting at a given offset.

        Parameters
        ----------
        inputs : dict
            dictionary of input values
        batch_size : int
            number of examples in the resulting batch
        offset : int
            index of first element of the batch
        """

        length = inputs[list(inputs.keys())[0]].shape[0]
        batch_indices = [j for j in range(offset,min(offset+batch_size,length))]
        batch_inputs = {}
        for i in inputs:
            batch_inputs[i] = inputs[i][batch_indices]
        return batch_inputs

    def get_tensor_shape(self,tensor):
        """
        Returns the shape of a tensor as a list.

        Parameters
        ----------
        tensor : tensorflow.Tensor
            tensor to return shape of
        """

        return tensor.get_shape().as_list()

    def add_variable(self, variable):
        """
        Adds a variable to this object's set of variables (self.variables).

        Parameters
        ----------
        variable : tensorflow.Tensor
            variable tensor
        """

        self.variables.append(variable)

    def add_variables(self,variables):
        """
        Adds a list of variables to this object's set of variables (self.variables).

        Parameters
        ----------
        variables : List of tensorflow.Tensor
            list of svariable tensors
        """

        self.variables.extend(variables)

    def get_rnn_cell_variables(self,cell):
        """
        Returns the variables of a RNN cell.

        Parameters
        ----------
        cell : tensorflow.nn.rnn_cell.*
            RNN cell
        """

        return cell.trainable_variables

    def add_saver(self):
        """
        Adds an operation to save the weights of this neural network model.
        """

        with self.graph.as_default():
            self.saver = tf.train.Saver(save_relative_paths=True)
            return self.saver

    def save_model(self,path):
        """
        Save the weights of this neural network model.

        Parameters
        ----------
        path : str
            directory to save model in
        """

        if self.variables_initialized:
            filepath = path.joinpath("nn")
            filename = self.get_string_path(filepath)
            result = self.saver.save(self.session,filename)
            return result
        else:
            print("Error: can't save model if variables are not initialized")
            return None


    def load_model(self,filename):
        """
        Load the weights of a saved neural network model.

        Parameters
        ----------
        filename : str
            base filename of the saved model
        """

        filename = self.get_string_path(filename)
        self.saver.restore(self.session,filename)
        self.variables_initialized = True

    def get_string_path(self,path):
        return str(path)

    def add_embedding_layer(self,inputs,num_classes,embedding_size,weight_initializer=None):
        """
        Adds an embedding layer to this neural network model.

        The values of the input tensor are the indices of classes. This function represents each class
        as a trainable vector of size 'embedding_size'. In the return value the class indices are
        replaced by their correspondig embedding vector.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            input tensor of dtype tensorflow.int32
        num_classes : int
            number of possible classes
        embedding_size : int
            size of embedding vectors
        weight_initializer : str, optional
            string denoting a weight initialization function (default is None)
        """

        with self.graph.as_default():
            if weight_initializer is None:
                weight_initializer = self.DEFAULT_WEIGHT_INITIALIZER

            shape = [num_classes, embedding_size]
            embedding = tf.Variable(self.get_initial_value(weight_initializer,shape),name=self.get_name("embedding"))
            self.variables.append(embedding)
            lookup = tf.nn.embedding_lookup(embedding,inputs)
            return lookup

    def get_name(self,base):
        """
        Returns a str obtained by concatenating the input base str with a unique integer index.

        Parameters
        ----------
        base : str
            base string name
        """

        name = base + "_" + str(self.next_name_id)
        self.next_name_id += 1
        return name

    def get_tensor_base_name(self,tensor):
        """
        Returns the base of a tensor name by splitting the string at "/".

        Parameters
        ----------
        tensor :tensorflow.Tensor
            tensor to return base name of
        """

        name = tensor.name
        return name.split("/")[0]


class MultiTaskModel(Model):
    """
    Class that builds a complete neural network model based on a JSON model configuration file.
    """

    def __init__(self,config,sentence_length,inputs,targets):
        """
        Create a model based on a JSON model configuration file.

        Parameters
        ----------
        config : dict
            dictionary contaning model configuration
        sentence_length : int
            maximum sentence_length to use for this model
        inputs : dict
            annotation definitions from dataset file for inputs specified in model configuration
        targets : dict
            annotation definitions from dataset file for targets specified in model configuration
        """

        super().__init__()
        self.sentence_length = sentence_length

        layers = config['layers']
        optimizer = config['optimizer']
        model_inputs = config['inputs']
        tasks = config['tasks']
        losses = None
        if 'losses' in config:
            losses = config['losses']
        initializers = None
        if 'initializers' in config:
            initializers = config['initializers']


        saved_model = False
        if "saved_model" in config:
            #if a saved model is loaded, the config files contains a key "saved_model" and "sentence_length" to use the sentence_length the model was originally created with
            saved_model = True
            sentence_length = config['sentence_length']
            self.sentence_length = sentence_length

        if not initializers is None:
            if 'weight' in initializers:
                self.DEFAULT_WEIGHT_INITIALIZER = initializers['weight']
            if 'bias' in initializers:
                self.DEFAULT_BIAS_INITIALIZER = initializers['bias']

        with self.graph.as_default():
            input_placeholders = {}
            gcn_placeholders = {}

            initial_input = None
            inputs_to_concat = []
            #inputs of type "vector_sequence" and "class_sequence" will be concatenated along last dimension to form a single input tensor
            for i in model_inputs:
                input_name = i['name']
                input_type = i['type']
                ph = None
                if input_type == "vector_sequence":
                    vector_length = None
                    if saved_model:
                        vector_length = i['vector_length']
                    else:
                        vector_length = inputs[input_name][1].shape[2]
                        i['vector_length'] = vector_length

                    ph = tf.placeholder(tf.float32,[None,sentence_length,vector_length],name=input_name)
                    input_placeholders[input_name] = ph
                elif input_type == "graph_structure":
                    num_classes = None
                    if saved_model:
                        num_classes = i['num_classes']
                    else:
                        num_classes = inputs[input_name][2]
                        i['num_classes'] = num_classes

                    ph = tf.placeholder(tf.int32,[None,sentence_length,sentence_length],name=input_name)
                    input_placeholders[input_name] = ph
                    gcn_placeholders[input_name] = (ph, num_classes)
                elif input_type == "class_sequence":
                    num_classes = None
                    if saved_model:
                        num_classes = i['num_classes']
                    else:
                        num_classes = inputs[input_name][2]
                        i['num_classes'] = num_classes

                    ph = tf.placeholder(tf.int32,[None,sentence_length],name=input_name)
                    input_placeholders[input_name] = ph
                    embedding_size = i['embedding_size']
                    ph = self.add_embedding_layer(ph,num_classes,embedding_size)

                if input_type != "graph_structure":
                    inputs_to_concat.append(ph)

            initial_input = tf.concat(inputs_to_concat,axis=2)

            #create list of training, test and query placeholders that need to be fed to the model
            self.train_placeholders = {}
            self.train_placeholders.update(input_placeholders)

            self.query_placeholders = {}
            self.query_placeholders.update(input_placeholders)

            #add dropout to the model if necessary
            current_inputs = initial_input
            dropout_op = None
            if "word_dropout" in config:
                dropout = config['word_dropout']
                current_inputs = self.add_word_dropout(current_inputs,dropout["keep_prob"])
                dropout_op = current_inputs

            #add neural network layers to the model
            for i in range(len(layers)):
                layer = layers[i]
                layer_type = layer['type']
                if layer_type == "rnn":
                    depth = layer['depth']
                    cell_type = layer['cell_type']
                    hidden_sizes = layer['sizes']
                    activation = None
                    if 'activation' in layer:
                        activation = layer['activation']
                    concat_internal_state = False
                    if 'concat_internal_state' in layer:
                        concat_internal_state = layer['concat_internal_state']
                    dropout_mode = None
                    if 'dropout_mode' in layer:
                        dropout_mode = layer['dropout_mode']
                    dropout_keep_prob = None
                    if 'dropout_keep_prob' in layer:
                        dropout_keep_prob = layer['dropout_keep_prob']

                    current_inputs = self.add_multi_rnn_layer(current_inputs,depth,cell_type,hidden_sizes,activation=activation,concat_internal_state=concat_internal_state,dropout_mode=dropout_mode,dropout_keep_prob=dropout_keep_prob)
                elif layer_type == "birnn":
                    depth = layer['depth']
                    cell_type = layer['cell_type']
                    hidden_sizes = layer['sizes']
                    activation = None
                    if 'activation' in layer:
                        activation = layer['activation']
                    concat_internal_state = False
                    if 'concat_internal_state' in layer:
                        concat_internal_state = layer['concat_internal_state']
                    dropout_mode = None
                    if 'dropout_mode' in layer:
                        dropout_mode = layer['dropout_mode']
                    dropout_keep_prob = None
                    if 'dropout_keep_prob' in layer:
                        dropout_keep_prob = layer['dropout_keep_prob']

                    current_inputs = self.add_multi_birnn_layer(current_inputs,depth,cell_type,hidden_sizes,activation=activation,concat_internal_state=concat_internal_state,dropout_mode=dropout_mode,dropout_keep_prob=dropout_keep_prob)
                elif layer_type == "conv":
                    num_filters = layer['num_filters']
                    filter_sizes = layer['filter_sizes']
                    use_max_pool = layer['use_max_pool']
                    activation = None
                    if 'activation' in layer:
                        activation = layer['activation']
                    preserve_vectors = False
                    if 'preserve_vectors' in layer:
                        preserve_vectors = layer['preserve_vectors']
                    current_inputs = self.add_cnn_layer(current_inputs,num_filters,filter_sizes,activation=activation,use_max_pool=use_max_pool,preserve_vectors=preserve_vectors)
                    if 'dropout_keep_prob' in layer and 'dropout_mode' in layer:
                        current_inputs = self.add_dropout_layer(current_inputs,layer['dropout_keep_prob'],mode=layer['dropout_mode'])
                elif layer_type == "gcn":
                    batch_adjacency = gcn_placeholders[layer['adjacency']][0]
                    #if dropout is used it must be applied to the activaiton function
                    if not dropout_op is None:
                        batch_adjacency = self.add_gcn_dropout(batch_adjacency,dropout_op)
                    num_tags = gcn_placeholders[layer['adjacency']][1]
                    use_gating = True
                    if 'use_gating' in layer:
                        use_gating = layer['use_gating']
                    activation = None
                    if 'activation' in layer:
                        activation = layer['activation']
                    current_inputs = self.add_gcn_layer(current_inputs,batch_adjacency,num_tags,activation=activation,use_gating=use_gating)
                    if 'dropout_keep_prob' in layer and 'dropout_mode' in layer:
                        current_inputs = self.add_dropout_layer(current_inputs,layer['dropout_keep_prob'],mode=layer['dropout_mode'])
                elif layer_type == "dense":
                    size = layer['size']
                    activation = None
                    if 'activation' in layer:
                        activation = layer['activation']
                    current_inputs = self.add_dense_layer(current_inputs,size,activation=activation)
                elif layer_type == "dense3":
                    size = layer['size']
                    activation = None
                    if 'activation' in layer:
                        activation = layer['activation']
                    current_inputs = self.add_dense3_layer(current_inputs,size,activation=activation)
                elif layer_type == "attention":
                    current_inputs = self.add_attention_layer(current_inputs)

            target_placeholders = {}
            loss_tensors = {}
            accuracy_tensors = {}

            self.query_operations = {}
            #create prediction, loss and accuracy tensors for each task the model should solve
            for task in tasks:
                task_type = task['type']
                if task_type == "sentence_class":
                    target_name = task['target']
                    loss = task['loss']
                    output_layer = "dense"
                    if "output_layer" in task:
                        output_layer = task['output_layer']

                    num_classes = None
                    if saved_model:
                        num_classes = task['num_classes']
                    else:
                        num_classes = targets[target_name][2]
                        task['num_classes'] = num_classes

                    ph = tf.placeholder(tf.int32,shape=[None],name=target_name)
                    target_placeholders[target_name] = ph

                    tensor_shape = self.get_tensor_shape(current_inputs)
                    logits = None
                    if output_layer == "dense":
                        logits = self.add_dense_layer(current_inputs,num_classes)


                    loss_tensor = None
                    if loss == "cross_entropy":
                        loss_tensor = self.add_sparse_cross_entropy_loss(logits,ph)
                    elif loss == "squared_error":
                        expanded_target = tf.one_hot(ph,num_classes,on_value=1.0,off_value=0.0,dtype=tf.float32,axis=-1)
                        loss_tensor = self.add_squared_error_loss(logits,expanded_target)

                    predictions = self.add_class_prediction(logits)
                    accuracy = self.add_classification_accuracy(predictions,ph)
                    self.query_operations[target_name] = predictions

                    loss_tensors[target_name] = loss_tensor
                    accuracy_tensors[target_name] = accuracy

                elif task_type == "class_sequence":
                    target_name = task['target']
                    loss = task['loss']
                    output_layer = "dense3"
                    if "output_layer" in task:
                        output_layer = task['output_layer']

                    num_classes = None
                    if saved_model:
                        num_classes = task['num_classes']
                    else:
                        num_classes = targets[target_name][2]
                        task['num_classes'] = num_classes
                    ph = tf.placeholder(tf.int32,shape=[None,sentence_length],name=target_name)
                    target_placeholders[target_name] = ph

                    logits = None
                    if output_layer == "dense3":
                        logits = self.add_dense3_layer(current_inputs,num_classes)

                    loss_tensor = None
                    if loss == "cross_entropy":
                        loss_tensor = self.add_sparse_cross_entropy_loss(logits,ph)
                    elif loss == "squared_error":
                        expanded_target = tf.one_hot(ph,num_classes,on_value=1.0,off_value=0.0,dtype=tf.float32,axis=-1)
                        loss_tensor = self.add_squared_error_loss(logits,expanded_target)
                    elif loss == "weighted_cross_entropy":
                        weight_zero = task['weight_zero']
                        weight_non_zero = task['weight_non_zero']
                        loss_tensor = self.add_weighted_cross_entropy_loss(logits,ph,num_classes,weight_zero,weight_non_zero)

                    predictions = self.add_class_prediction(logits)
                    accuracy_zero, accuracy_non_zero, count_zero, count_non_zero = self.add_weighted_classification_accuracy(predictions,ph)
                    self.query_operations[target_name] = predictions

                    loss_tensors[target_name] = loss_tensor
                    accuracy_tensors[target_name+"_zero"] = accuracy_zero
                    accuracy_tensors[target_name+"_non_zero"] = accuracy_non_zero
                    accuracy_tensors[target_name+"_zero_count"] = count_zero
                    accuracy_tensors[target_name+"_non_zero_count"] = count_non_zero

                elif task_type == "fixed_length_class_sequence":
                    target_name = task['target']
                    loss = task['loss']
                    output_layer = "dense"
                    if "output_layer" in task:
                        output_layer = task['output_layer']

                    num_classes = None
                    if saved_model:
                        num_classes = task['num_classes']
                    else:
                        num_classes = targets[target_name][2]
                        task['num_classes'] = num_classes

                    sequence_length = None
                    if saved_model:
                        sequence_length = task['sequence_length']
                    else:
                        sequence_length = targets[target_name][3]
                        task['sequence_length'] = sequence_length

                    ph = tf.placeholder(tf.int32,shape=[None,sequence_length],name=target_name)
                    target_placeholders[target_name] = ph

                    logits_list = []
                    for j in range(sequence_length):
                        if output_layer == "dense":
                            logits_list.append(self.add_dense_layer(current_inputs,num_classes))
                    logits = tf.stack(logits_list,axis=1)

                    loss_tensor = None
                    if loss == "cross_entropy":
                        loss_tensor = self.add_sparse_cross_entropy_loss(logits,ph)
                    elif loss == "squared_error":
                        expanded_target = tf.one_hot(ph,num_classes,on_value=1.0,off_value=0.0,dtype=tf.float32,axis=-1)
                        loss_tensor = self.add_squared_error_loss(logits,expanded_target)
                    elif loss == "weighted_cross_entropy":
                        weight_zero = task['weight_zero']
                        weight_non_zero = task['weight_non_zero']
                        loss_tensor = self.add_weighted_cross_entropy_loss(logits,ph,num_classes,weight_zero,weight_non_zero)

                    predictions = self.add_class_prediction(logits)
                    accuracy = self.add_classification_accuracy(predictions,ph)
                    self.query_operations[target_name] = predictions

                    loss_tensors[target_name] = loss_tensor
                    accuracy_tensors[target_name] = accuracy

                else:
                    print("Warning: unknown task type at model creation {}".format(task_type))


            self.train_placeholders.update(target_placeholders)

            if not losses is None:
                for i in range(len(losses)):
                    loss = losses[i]
                    loss_type = loss['type']
                    if loss_type == "l2_weight":
                        l2_lambda = loss['lambda']
                        self.add_l2_weight_loss(l2_lambda)

            total_loss = self.add_total_loss()

            optimizer_type = optimizer['type']
            learning_rate = optimizer['learning_rate']

            clip_gradients = None
            if 'clip_gradients' in optimizer:
                clip_gradients = optimizer['clip_gradients']

            # add optimizer to model to get final training operation
            (train_op,global_step) = self.add_optimizer(optimizer_type,total_loss,learning_rate,clip_gradients=clip_gradients)

            self.train_operations = {
                    "train_op" : train_op,
                    "total_loss" : total_loss,
                    "global_step" : global_step,
            }
            self.train_operations.update(loss_tensors)
            self.train_operations.update(accuracy_tensors)

            self.test_operations = accuracy_tensors
            self.loss_tensors = loss_tensors
            self.accuracy_tensors = accuracy_tensors

            self.add_saver()

        #create a deep copy of the JSON configuration used to create this model
        self.copy_config(config)
        self.config['sentence_length'] = self.sentence_length
        self.config['saved_model'] = True


    def print_train_status(self,results):
        """
        Prints training status, i.e. value of loss function, step number and accuracies on current batch.

        Parameters
        ----------
        results : dict
            result of running the training operations
        """

        total_loss = results["total_loss"]
        step = results["global_step"]
        print("step {}, total loss {:g}".format(step,total_loss))
        for acc in self.accuracy_tensors:
            print("Accuracy for task {}: {:g}".format(acc,results[acc]))

    def print_test_status(self,results):
        """
        Prints accuracies for test data.

        Parameters
        ----------
        results : dict
            result of running testing operations
        """

        for acc in results:
            print("Accuracy for task {}: {:g}".format(acc,results[acc]))

    def copy_config(self,config):
        """
        Creates a deep copy of a model configuration.

        Parameters
        ----------
        config : dict
            dictionary containing model configuration
        """

        self.config = json.loads(json.dumps(config))
