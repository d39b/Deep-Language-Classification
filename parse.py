import numpy as np
import json

def parse_json_file(filename,input_names,target_names):
    """
    Parses a datset file and returns numpy representations of the annotations given by the 'input_names' and 'target_names' arguments.

    The dataset file should be in the special format specified in the documentation of this project.
    The values of 'sentence_class', 'class_sequence', 'fixed_length_class_sequence' or
    'graph_structure' are usually strings. For each such annotation an index from strings to integers is created
    and returned with the numpy representations. If a model is loaded these indices can be used to get a consistent mapping
    from string values to integers.

    Parameters
    ----------
    filename : str
        filename of JSON dataset file to load
    input_names : list of str
        list of input annotation names of the dataset file to be parsed
    target_names : list of str
        list of target annotation names of the dataset file to be parsed
    """

    json_content = None
    with open(filename,'r') as fp:
        json_content = json.load(fp)

    metadata = json_content['metadata']
    data = json_content['data']
    num_examples = len(data)

    sentence_length = metadata['sentence_length']
    annotations = metadata['annotations']

    #create lists of necessary annotations to parse
    tmp_inputs = []
    tmp_targets = []
    for a in annotations:
        if a['name'] in input_names:
            tmp_inputs.append(a)
        elif a['name'] in target_names:
            tmp_targets.append(a)

    #parse inputs
    #create dictionary from annotation name to parsing result (numpy array, number of classes)
    #create dictionary from annotation name to (dictionary from class names to integers)
    name_to_name_to_indices = {}
    parsed_inputs = {}
    for i in tmp_inputs:
        input_type = i['type']
        input_name = i['name']
        if input_type == 'vector_sequence':
            vector_length = i['vector_length']
            if input_name in json_content:
                word_vectors = json_content[input_name]
                parsed_inputs[input_name] = parse_vector_sequence(input_name,data,sentence_length,vector_length,word_vectors)
            else:
                parsed_inputs[input_name] = parse_vector_sequence_without_list(input_name,data,sentence_length,vector_length)

        elif input_type == 'graph_structure':
            parsed = parse_graph_structure(input_name,data,sentence_length)
            parsed_inputs[input_name] = (parsed[0],parsed[1],parsed[2])
            name_to_name_to_indices[input_name] = parsed[3]
        elif input_type == 'class_sequence':
            parsed = parse_class_sequence(input_name,data,sentence_length)
            parsed_inputs[input_name] = (parsed[0],parsed[1],parsed[2])
            name_to_name_to_indices[input_name] = parsed[3]
        else:
            print("Parse error: unknown input type {}".format(input_type))
            print("Please check your data file")

    #parse targets
    #create dictionary from annotation name to parsing result (numpy array, number of classes)
    #create dictionary from annotation name to (dictionary from class names to integers)
    parsed_targets = {}
    for t in tmp_targets:
        target_type = t['type']
        target_name = t['name']
        if target_type == 'class_sequence':
            parsed = parse_class_sequence(target_name,data,sentence_length)
            parsed_targets[target_name] = (parsed[0],parsed[1],parsed[2])
            name_to_name_to_indices[target_name] = parsed[3]
        elif target_type == 'sentence_class':
            parsed = parse_sentence_class(target_name,data)
            parsed_targets[target_name] = (parsed[0],parsed[1],parsed[2])
            name_to_name_to_indices[target_name] = parsed[3]
        elif target_type == 'fixed_length_class_sequence':
            sequence_length = t['sequence_length']
            parsed = parse_fixed_length_class_sequence(target_name,data,sequence_length)
            parsed_targets[target_name] = (parsed[0],parsed[1],parsed[2],parsed[3])
            name_to_name_to_indices[target_name] = parsed[4]
        else:
            print("Parse error: unknown input type {}".format(input_type))
            print("Please check your data file")

    sentences = parse_sentences('text',data)

    return (sentence_length, num_examples, sentences, parsed_inputs, parsed_targets, name_to_name_to_indices)

def parse_json_file_with_index(filename,name_to_name_to_indices,input_names,target_names,sentence_length):
    """
    Parses a datset file and returns numpy representations of the annotations given by the 'input_names' and 'target_names' arguments.

    The dataset file should be in the special format specified in the documentation of this project.
    This method is used when a saved model that might have been trained with another dataset file is loaded again.
    To train the loaded model with another dataset file, string values need to be mapped to the same integer values
    that were used in the previous training session. For this purpose this method takes an additional
    set of string to integer dictionaries as argument.

    Parameters
    ----------
    filename : str or dict
        filename of JSON dataset file to load or dataset as dictionary
    name_to_name_to_indices : dict of dict
        a dictionary mapping annotation names to dictionaries mapping string values to integers
    input_names : list of str
        list of input annotation names of the dataset file to be parsed
    target_names : list of str
        list of target annotation names of the dataset file to be parsed
    sentence_length : int
        maximum sentence length of the model this dataset file is loaded for
    """

    #check if input is filename or dataset (as dictionary)
    json_content = None
    if isinstance(filename, str):
        with open(filename,'r') as fp:
            json_content = json.load(fp)
    else:
        json_content = filename

    metadata = json_content['metadata']
    data = json_content['data']
    num_examples = len(data)

    if "sentence_length" in metadata:
        tmp_sentence_length = metadata["sentence_length"]
        if tmp_sentence_length > sentence_length:
            print("Warning: sentence length of stored model smaller than sentence length of data file")


    annotations = metadata['annotations']

    #create list of annotations to parse
    tmp_inputs = []
    tmp_targets = []
    for a in annotations:
        if a['name'] in input_names:
            tmp_inputs.append(a)
        elif a['name'] in target_names:
            tmp_targets.append(a)

    parsed_inputs = {}
    for i in tmp_inputs:
        input_type = i['type']
        input_name = i['name']
        if input_type == 'vector_sequence':
            vector_length = i['vector_length']
            if input_name in json_content:
                word_vectors = json_content[input_name]
                parsed_inputs[input_name] = parse_vector_sequence(input_name,data,sentence_length,vector_length,word_vectors)
            else:
                parsed_inputs[input_name] = parse_vector_sequence_without_list(input_name,data,sentence_length,vector_length)
        elif input_type == 'graph_structure':
            parsed = parse_graph_structure(input_name,data,sentence_length,initial_name_to_index=name_to_name_to_indices[input_name])
            parsed_inputs[input_name] = (parsed[0],parsed[1],parsed[2])
        elif input_type == 'class_sequence':
            parsed = parse_class_sequence(input_name,data,sentence_length,initial_name_to_index=name_to_name_to_indices[input_name])
            parsed_inputs[input_name] = (parsed[0],parsed[1],parsed[2])
        else:
            print("Parse error: unknown input type {}".format(input_type))
            print("Please check your data file")


    parsed_targets = {}
    for t in tmp_targets:
        target_type = t['type']
        target_name = t['name']
        if target_type == 'class_sequence':
            parsed = parse_class_sequence(target_name,data,sentence_length,initial_name_to_index=name_to_name_to_indices[target_name])
            parsed_targets[target_name] = (parsed[0],parsed[1],parsed[2])
        elif target_type == 'sentence_class':
            parsed = parse_sentence_class(target_name,data,initial_name_to_index=name_to_name_to_indices[target_name])
            parsed_targets[target_name] = (parsed[0],parsed[1],parsed[2])
        elif target_type == 'fixed_length_class_sequence':
            sequence_length = t['sequence_length']
            parsed = parse_fixed_length_class_sequence(target_name,data,sequence_length,initial_name_to_index=name_to_name_to_indices[target_name])
            parsed_targets[target_name] = (parsed[0],parsed[1],parsed[2],parsed[3])
            name_to_name_to_indices[target_name] = parsed[4]
        else:
            print("Parse error: unknown input type {}".format(input_type))
            print("Please check your data file")

    sentences = parse_sentences('text',data)

    return (num_examples, sentences, parsed_inputs, parsed_targets)


def parse_sentences(name,data):
    """
    Returns a list of natural language sentence in a dataset file

    Parameters
    ----------
    name : str
        string key that contains sentences in the dataset file
    data : list of dict
        list of dictionaries where each dictionary contains a sentence and its annotations
    """

    num_examples = len(data)

    sentences = []
    for i in range(num_examples):
        sentence = data[i][name]
        sentences.append(sentence)

    return sentences

def parse_vector_sequence(name,data,sequence_length,vector_length,word_vectors):
    """
    Parses a 'vector_sequence' annotation and returns it as a numpy array.

    The returned numpy array has size num_sentences times
    'sequence_length' times 'vector_length'

    Parameters
    ----------
    name : str
        name of the annotation
    data : list of dict
        list of dictionaries where each dictionary contains a sentence and its annotations
    sequence_length : int
        size of the second dimension of the returned matrix, usually corresponds to the maximum sentence length of a model
    vector_length : int
        size of the third dimesion of the returned matrix
    """

    num_examples = len(data)
    arr = np.zeros([num_examples,sequence_length,vector_length])

    for i in range(num_examples):
        vector_sequence = data[i][name]
        vector_sequence_length = len(vector_sequence)
        for j in range(min(vector_sequence_length,sequence_length)):
            word_vector = word_vectors[vector_sequence[j]][1]
            real_vector_length = len(word_vector)
            for k in range(min(real_vector_length,vector_length)):
                arr[i,j,k] = word_vector[k]

    return ("vector_sequence", arr)

def parse_vector_sequence_without_list(name,data,sequence_length,vector_length):
    """
    Parses a 'vector_sequence' annotation and returns it as a numpy array.

    The returned numpy array has size num_sentences times
    'sequence_length' times 'vector_length'

    Parameters
    ----------
    name : str
        name of the annotation
    data : list of dict
        list of dictionaries where each dictionary contains a sentence and its annotations
    sequence_length : int
        size of the second dimension of the returned matrix, usually corresponds to the maximum sentence length of a model
    vector_length : int
        size of the third dimesion of the returned matrix
    """

    num_examples = len(data)
    arr = np.zeros([num_examples,sequence_length,vector_length])

    for i in range(num_examples):
        vector_sequence = data[i][name]
        vector_sequence_length = len(vector_sequence)
        for j in range(min(vector_sequence_length,sequence_length)):
            real_vector_length = len(vector_sequence[j])
            for k in range(min(real_vector_length,vector_length)):
                arr[i,j,k] = vector_sequence[j][k]

    return ("vector_sequence", arr)

def parse_graph_structure(name,data,sequence_length,initial_name_to_index=None):
    """
    Parses a 'graph_sequence' annotation and returns it as a numpy array.

    The returned numpy array has size num_sentences times 'sequence_length'
    times 'sequence_length' and corresponds to the adjacency matrices of the graphs.
    Creates an index from edge type names to indices. Suppose there is an edge of type
    k from vertex i to j and let adj be the adjacency matrix. Then adj[i,j] = k and
    adj[j,i] = -(k+1).

    Returns the numpy array, the number of edge types and the dictionary from edge type names to integers

    Parameters
    ----------
    name : str
        name of the annotation
    data : list of dict
        list of dictionaries where each dictionary contains a sentence and its annotations
    sequence_length : int
        size of the second dimension of the returned matrix, usually corresponds to the maximum sentence length of a model
    initial_name_to_index : dict, optional
        use this dictionary from edge type names to integers instead of creating a new one, necessary when using a saved model (default is None)
    """

    num_examples = len(data)

    arr = np.zeros([num_examples,sequence_length,sequence_length])
    index_fixed = False
    #"LOOP" key reserved for later use with GCN layer
    name_to_index = {"None" : 0, "LOOP" : 1}
    if not initial_name_to_index is None:
        name_to_index = initial_name_to_index
        index_fixed = True

    next_index = 2
    for i in range(num_examples):
        edges = data[i][name]
        for e in edges:
            #type of outgoing edge
            t = str(e['type'])
            #type of incoming edge
            t_r = t + "_r"
            origin = e['origin']
            target = e['target']
            if not index_fixed:
                if not t in name_to_index:
                    name_to_index[t] = next_index
                    name_to_index[t_r] = next_index + 1
                    next_index += 2

            value = 0
            value_r = 0
            #check if type exists, might not be case when using a previous name_to_index
            if t in name_to_index:
                value = name_to_index[t]
                value_r = name_to_index[t_r]

            if origin < sequence_length and target < sequence_length:
                arr[i,origin,target] = value
                #values of incoming edges are negative to easier distinguish between incoming and outgoing edges
                arr[i,target,origin] = (-1)*value_r

    num_classes = len(name_to_index)
    return ("graph_structure", arr, num_classes, name_to_index)

def parse_class_sequence(name,data,sequence_length,initial_name_to_index=None):
    """
    Parses a 'class_sequence' annotation and returns it as a numpy array.

    The returned numpy array has size num_sentences times 'sequence_length'.
    An index from class names to integers is created. Let arr be the returned
    numpy array. If word j in sentence i belongs to class k then arr[i,j] = k.

    Returns the numpy array, the number of class names and the dictionary from class names to integers.

    Parameters
    ----------
    name : str
        name of the annotation
    data : list of dict
        list of dictionaries where each dictionary contains a sentence and its annotations
    sequence_length : int
        size of the second dimension of the returned matrix, usually corresponds to the maximum sentence length of a model
    initial_name_to_index : dict, optinal
        use this dictionary from class names to integers instead of creating a new one, necessary when using a saved model
    """

    num_examples = len(data)

    arr = np.zeros([num_examples,sequence_length])
    index_fixed = False
    name_to_index = {"None" : 0}
    if not initial_name_to_index is None:
        name_to_index = initial_name_to_index
        index_fixed = True

    next_index = 1
    for i in range(num_examples):
        labels = data[i][name]
        for j in range(len(labels)):
            e = labels[j]
            class_name = None
            #value of 'class_sequence' annotation can either be list of strings or list of object with "index" and "type" keys
            is_dict = isinstance(e,dict)
            if is_dict:
                class_name = str(e['type'])
            else:
                class_name = str(e)

            if not index_fixed:
                if not class_name in name_to_index:
                    name_to_index[class_name] = next_index
                    next_index += 1

            value = 0
            #check if class_name exits, might not be the case when using a previous name_to_index
            if class_name in name_to_index:
                value = name_to_index[class_name]

            if is_dict:
                if 'index' in e:
                    index = e['index']
                    if index < sequence_length:
                        arr[i,index] = value
                else:
                    si = e['startIndex']
                    ei = e['endIndex']
                    for k in range(si,ei+1):
                        if k < sequence_length:
                            arr[i,k] = value

            else:
                if j < sequence_length:
                    arr[i,j] = value

    num_classes = len(name_to_index)
    return ("class_sequence", arr, num_classes, name_to_index)

def parse_fixed_length_class_sequence(name,data,sequence_length,initial_name_to_index=None):
    """
    Parses a 'fixed_length_class_sequence' annotation and returns it as a numpy array.

    The returned numpy array has size num_sentences times 'sequence_length'.
    An index from class names to integers is created. Let arr be the returned
    numpy array. If the j-th class of sentence i is k then arr[i,j] = k.

    Returns the numpy array, the number of class names and the dictionary from class names to integers.

    Parameters
    ----------
    name : str
        name of the annotation
    data : list of dict
        list of dictionaries where each dictionary contains a sentence and its annotations
    sequence_length : int
        size of the second dimension of the returned matrix, corresponding to the fixed length of the annotation
    initial_name_to_index : dict, optional
        use this dictionary from class names to integers instead of creating a new one, necessary when using a saved model (default is None)
    """

    num_examples = len(data)

    arr = np.zeros([num_examples,sequence_length])
    index_fixed = False
    name_to_index = {"None" : 0}
    if not initial_name_to_index is None:
        name_to_index = initial_name_to_index
        index_fixed = True

    next_index = 1
    for i in range(num_examples):
        labels = data[i][name]
        for j in range(len(labels)):
            class_name = str(labels[j])
            if not index_fixed:
                if not class_name in name_to_index:
                    name_to_index[class_name] = next_index
                    next_index += 1

            value = 0
            if class_name in name_to_index:
                value = name_to_index[class_name]

            if j < sequence_length:
                arr[i,j] = value

    num_classes = len(name_to_index)
    return ("class_sequence", arr, num_classes, sequence_length, name_to_index)

def parse_sentence_class(name,data,initial_name_to_index=None):
    """
    Parses a 'sentence_class' annotation and returns it as a numpy array.

    The returned numpy array has size num_sentences.
    An index from class names to integers is created. Let arr be the returned
    numpy array. If sentence i belongs to class k then arr[i] = k.

    Returns the numpy array, the number of class names and the dictionary from class names to integers.

    Parameters
    ----------
    name : str
        name of the annotation
    data : list of dict
        list of dictionaries where each dictionary contains a sentence and its annotations
    initial_name_to_index : dict, optional
        use this dictionary from class names to integers instead of creating a new one, necessary when using a saved model (default is None)
    """

    num_examples = len(data)

    arr = np.zeros([num_examples])
    name_to_index = {}
    index_fixed = False
    if not initial_name_to_index is None:
        name_to_index = initial_name_to_index
        index_fixed = True

    next_index = 0
    for i in range(num_examples):
        class_name = str(data[i][name])
        if not index_fixed:
            if not class_name in name_to_index:
                name_to_index[class_name] = next_index
                next_index += 1

        value = 0
        if class_name in name_to_index:
            value = name_to_index[class_name]
        else:
            print("Warning: unknown sentence class {}".format(class_name))

        arr[i] = value

    num_classes = len(name_to_index)
    return ("sentence_class" ,arr, num_classes, name_to_index)
