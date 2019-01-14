import argparse
import json
import numpy as np
from model import MultiTaskModel
import parse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network to perform NLP tasks")
    parser.add_argument("data_file", help="json file containing sentences and corresponding input and target data")
    parser.add_argument("config_or_model", help="json file containing model configuration or a directory containing a saved model")
    parser.add_argument("-sm","--save_model",help="directory to save trained model to")
    parser.add_argument("-bs","--batch_size",type=int,default=32,help="number of sentences in a single batch used for training")
    parser.add_argument("-s","--steps",type=int,default=1000,help="number of training steps to perform")
    parser.add_argument("-nr","--num_repetitions",type=int,default=1,help="number of repetitions of cross validation to perform")
    parser.add_argument("-np","--num_parts",type=int,default=10,help="number of parts to split dataset into for cross validation")
    parser.add_argument("-q","--quiet",action="store_true",help="don't print training loss and accuracy information")
    parser.add_argument("-pf","--print_frequency",type=int,default=10,help="how often to print training loss and accuracy information")
    parser.add_argument("-lm","--load_model",action="store_true",help="resume training on saved model")
    parser.add_argument("--test_data_fraction",type=float,default=0.1,help="fraction of examples that are used for testing the model")
    parser.add_argument("--train_all_examples",action="store_true",help="use complete dataset for training")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-cv","--cross_validate",action="store_true",help="run cross validation")
    group.add_argument("--test_data_file",help="file containing test data")
    args = parser.parse_args()
    return args


def main(args):
    config = None
    model = None
    sentence_length = None
    num_examples = None
    sentences = None
    inputs = None
    targets = None
    input_names = None
    target_names = None
    name_to_name_to_indices = None

    #load a saved model
    if args.load_model:
        load_path = Path(args.config_or_model)
        if (not load_path.exists()) or (not load_path.is_dir()):
            print("Error: directory doesn't exist")

        config_filename = load_path.joinpath("model_config.json")
        with config_filename.open('r') as fp:
            config = json.load(fp)

        index_filename = load_path.joinpath("name_to_index.json")
        with index_filename.open('r') as fp:
            name_to_name_to_indices = json.load(fp)

        sentence_length = config['sentence_length']
        model = MultiTaskModel(config,config['sentence_length'],{},{})
        model.load_model(load_path.joinpath("nn"))
        input_names = []
        target_names = []
        input_name_to_def = {}
        for i in config['inputs']:
            input_names.append(i['name'])
            input_name_to_def[i['name']] = i
        for t in config['tasks']:
            target_names.append(t['target'])

        num_examples, sentences, inputs, targets = parse.parse_json_file_with_index(args.data_file,name_to_name_to_indices,input_names,target_names,sentence_length)

        for input_name in input_names:
            if not input_name in inputs:
                print("problem: model input \"{}\" not found in dataset file, feeding zero values".format(input_name))
                input_def = input_name_to_def[input_name]
                input_type = input_def['type']
                array_shape = [] 
                if input_type == "vector_sequence":
                    array_shape = [num_examples,sentence_length,input_def['vector_length']]
                elif input_type == "class_sequence":
                    array_shape = [num_examples,sentence_length]
                elif input_type == "graph_structure":
                    array_shape = [num_examples,sentence_length,sentence_length]
                
                inputs[input_name] = (input_type,np.zeros(array_shape))                 
        
        for target_name in target_names:
            if not target_name in targets:
                print("problem: model target \"{}\" not found in dataset file".format(target_name))
                #a saved model should not be retrained without necessary target data
                #because if zero values are fed to the model instead of the real target values
                #the weights of the model will change to produce zero target value
                #this will destroy the previous training progress for this target
                print("Shutting down")
                return

        print("Model loaded from: {}".format(args.config_or_model))
    #create a new model
    else:  
        with open(args.config_or_model,'r') as fp:
            config = json.load(fp)

        input_names = []
        target_names = []
        for i in config['inputs']:
            input_names.append(i['name'])
        for t in config['tasks']:
            target_names.append(t['target'])

        sentence_length, num_examples, sentences, inputs, targets, name_to_name_to_indices = parse.parse_json_file(args.data_file,input_names,target_names)
        model = MultiTaskModel(config,sentence_length,inputs,targets)

    tmp_bs = args.batch_size
    if args.batch_size <= 0 or args.batch_size > num_examples:
        print("Error: batch size negative or greater than number of examples in dataset")
        tmp_bs = 32

    #split dataset into training and testing part
    if not args.cross_validate and (args.test_data_file is None):
        #split dataset into train and test parts
        permutation = np.random.permutation(num_examples)
        split_index = int(num_examples*(1.0-args.test_data_fraction))

        train_data = {}
        test_data = {}
        for x,y in inputs.items():
            if args.train_all_examples:
                train_data[x] = y[1]
            else:
                train_data[x] = y[1][permutation[0:split_index]]

            test_data[x] = y[1][permutation[split_index:]]

        for x,y in targets.items():
            if args.train_all_examples:
                train_data[x] = y[1]
            else:
                train_data[x] = y[1][permutation[0:split_index]]

            test_data[x] = y[1][permutation[split_index:]]

        test_sentences = []
        for i in range(split_index,num_examples):
            test_sentences.append(sentences[permutation[i]])

        model.train(train_data, args.steps, tmp_bs, quiet=args.quiet,print_frequency=args.print_frequency)
        test_data_length = test_data[list(test_data.keys())[0]].shape[0]
        results = model.test_in_batches(test_data,tmp_bs,quiet=args.quiet)
        results = combine_accuracy_results(results,target_names,tmp_bs,test_data_length)
        model.print_test_status(results)
    #use separate dataset file for testing
    elif not args.test_data_file is None:
        test_num_examples, test_sentences, test_inputs, test_targets = parse.parse_json_file_with_index(args.test_data_file,name_to_name_to_indices,input_names,target_names,sentence_length)

        train_data = {}
        test_data = {}
        for x,y in inputs.items():
            train_data[x] = y[1]

        for x,y in targets.items():
            train_data[x] = y[1]

        for x,y in test_inputs.items():
            test_data[x] = y[1]

        for x,y in test_targets.items():
            test_data[x] = y[1]


        model.train(train_data, args.steps, tmp_bs, quiet=args.quiet,print_frequency=args.print_frequency)
        test_data_length = test_data[list(test_data.keys())[0]].shape[0]
        results = model.test_in_batches(test_data,tmp_bs,quiet=args.quiet)
        results = combine_accuracy_results(results,target_names,tmp_bs,test_data_length)
        model.print_test_status(results)
    #run cross validation
    else:
        data = {}
        for x,y in inputs.items():
            data[x] = y[1]

        for x,y in targets.items():
            data[x] = y[1]


        accuracies = cross_validate(model,args.num_repetitions,args.num_parts,num_examples,data,args.steps,tmp_bs,target_names,quiet=args.quiet)

        for target_name in target_names:
            print_cv_summary(target_name,accuracies[target_name])

    #save model
    if not args.save_model is None:
        path = save_model(args.save_model,model,name_to_name_to_indices)
        if path is None:
            print("Error: model could not be saved")
        else:
            print("Model saved in directory {}".format(path))

def save_model(dir,model,name_to_name_to_indices):
    path = Path(dir)
    #check if parent directory exists
    if not path.parent.exists():
        print("Error: parent directory of save dir does not exist")
        print("Parent directory: {}".format(path.parent))
        return None

    #check if directory exists
    if path.exists() and (not path.is_dir()):
        print("Error: cannot create directory, there is already a file with the name '{}''".format(path.name))
        return None

    if path.exists():
        print("Warning: there already is a directory with this name, previously saved model will be overwritten")

    #create directory if it doesn't exist
    if not path.exists():
        path.mkdir()

    #save neural network variables
    result = model.save_model(path)
    if result == None:
        print("Error: neural network weights could not be saved")
        return None
    else:
        #save model configuration file, with this the model can be recreated
        model_config_path = path.joinpath("model_config.json")
        with model_config_path.open('w') as fp:
            json.dump(model.config,fp,indent=4)
        #save the dictionary from annotation names to dictionaries from strings to integers
        index_path = path.joinpath("name_to_index.json")
        with index_path.open('w') as fp:
            json.dump(name_to_name_to_indices,fp,indent=4)

    return path

def cross_validate(model,num_repetitions,num_parts,num_examples,data,steps,batch_size,target_names,quiet=False):
    part_size = int(float(num_examples) / float(num_parts))
    split_indices = []
    curr_index = 0
    while curr_index + part_size < num_examples:
        curr_index = curr_index + part_size
        split_indices.append(curr_index)

    accuracies = {}
    for target_name in target_names:
        accuracies[target_name] = np.zeros([num_repetitions,len(split_indices)+1])

    for i in range(num_repetitions):
        permutation = np.random.permutation(num_examples)

        for j in range(len(split_indices)+1):
            train_indices, test_indices = get_train_and_test_indices(permutation,split_indices,j)
            train_data = {}
            test_data = {}
            for name, value in data.items():
                train_data[name] = value[train_indices]
                test_data[name] = value[test_indices]

            model.initialize_variables()
            model.train(train_data,steps,batch_size,quiet=quiet,print_frequency=args.print_frequency)
            test_data_length = test_data[list(test_data.keys())[0]].shape[0]
            results = model.test_in_batches(test_data,batch_size,quiet=quiet)
            results = combine_accuracy_results(results,target_names,batch_size,test_data_length)
            for target_name in target_names:
                accuracy = results[target_name]
                accuracies[target_name][i,j] = accuracy

    return accuracies

def get_train_and_test_indices(permutation,split_indices,j):
    n = len(split_indices)
    train_indices = None
    test_indices = None
    if j == 0:
        train_indices = permutation[split_indices[0]:]
        test_indices = permutation[0:split_indices[0]]
    elif j == n:
        train_indices = permutation[0:split_indices[j-1]]
        test_indices = permutation[split_indices[j-1]:]
    else:
        train_indices = np.concatenate([permutation[0:split_indices[j-1]],permutation[split_indices[j]:]])
        test_indices = permutation[split_indices[j-1]:split_indices[j]]

    return (train_indices,test_indices)


def print_cv_summary(target_name,accuracies):
    mean_accuracy = np.mean(accuracies)
    global_min = np.min(accuracies)
    global_max = np.max(accuracies)
    std_dev = np.std(accuracies)
    print("Results for task: {}".format(target_name))
    print("Mean accuracy: {:g}".format(mean_accuracy))
    print("Min accuracy: {:g}".format(global_min))
    print("Max accuracy: {:g}".format(global_max))
    print("Standard deviation: {:g}".format(std_dev))

def combine_accuracy_results(results,target_names,batch_size,length):
    output = {}
    for target_name in results[0].keys():
        if not "count" in target_name:
            #for outputs of type 'class_sequence'
            if "zero" in target_name:
                accuracy = 0.0
                total_count = 0.0
                for i in range(len(results)):
                    count = results[i][target_name+"_count"]
                    accuracy += results[i][target_name]*count
                    total_count += count

                accuracy = accuracy / total_count
                output[target_name] = accuracy
            else:
                #for outputs of type 'sentence_class' or 'fixed_length_class_sequence'
                accuracy = 0
                for i in range(len(results)-1):
                    accuracy += results[i][target_name]*batch_size
                if (length % batch_size) == 0:
                    accuracy += results[-1][target_name]*batch_size
                else:
                    accuracy += results[-1][target_name]*(length % batch_size)
                accuracy = accuracy / length
                output[target_name] = accuracy

    return output

def check_save_path(args):
    if not args.save_model is None:
        path = Path(args.save_model)
        #check if parent directory exists
        if not path.parent.exists():
            print("Error: parent directory of save dir does not exist")
            print("Parent directory: {}".format(path.parent))
            return False

        #check if directory exists
        if path.exists() and (not path.is_dir()):
            print("Error: cannot create directory, there is already a file with the name '{}''".format(path.name))
            return False
    
    return True
    
if __name__ == '__main__':
    args = parse_args()
    #check if save directory is valid, otherwise we might spend some time training a model and then fail saving it
    if check_save_path(args):
        main(args)
