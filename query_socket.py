import argparse
import json
from pathlib import Path
import numpy as np
from model import MultiTaskModel
import parse
import create_data
import asyncio
import websockets
import logging
import logging.handlers

class QueryModel:

    def __init__(self,model_file,identifier):
        self.config = None
        self.model = None
        self.sentence_length = None
        self.name_to_name_to_indices = None
        self.logger = logging.getLogger('root.Model-{}'.format(identifier))
    
        load_path = Path(model_file)
        if (not load_path.exists()) or (not load_path.is_dir()):
            self.logger.error("model directory {} doesn't exist".format(model_file))

        config_filename = load_path.joinpath("model_config.json")
        with config_filename.open('r') as fp:
            self.config = json.load(fp)

        index_filename = load_path.joinpath("name_to_index.json")
        with index_filename.open('r') as fp:
            self.name_to_name_to_indices = json.load(fp)

        self.sentence_length = self.config['sentence_length']

        self.model = MultiTaskModel(self.config,self.sentence_length,{},{})
        self.model.load_model(load_path.joinpath("nn"))

        self.input_names = []
        self.target_name_to_def = {}
        self.input_name_to_def = {}
        self.name_to_index_to_name = {}
        for i in self.config['inputs']:
            input_name = i['name']
            self.input_names.append(input_name)
            self.input_name_to_def[input_name] = i
        for t in self.config['tasks']:
            target_name = t['target']
            self.target_name_to_def[target_name] = t
            index_to_name = {}
            for x,y in self.name_to_name_to_indices[target_name].items():
                index_to_name[y] = x
            self.name_to_index_to_name[target_name] = index_to_name

    def query(self,query_input):
        num_examples, sentences, inputs, targets = parse.parse_json_file_with_index(query_input,self.name_to_name_to_indices,self.input_names,[],self.sentence_length)

        for input_name in self.input_names:
            if not input_name in inputs:
                self.logger.warning("problem: model input \"{}\" not found in dataset file, feeding zero values".format(input_name))
                input_def = self.input_name_to_def[input_name]
                input_type = input_def['type']
                array_shape = [] 
                if input_type == "vector_sequence":
                    array_shape = [num_examples,self.sentence_length,input_def['vector_length']]
                elif input_type == "class_sequence":
                    array_shape = [num_examples,self.sentence_length]
                elif input_type == "graph_structure":
                    array_shape = [num_examples,self.sentence_length,self.sentence_length]
                
                inputs[input_name] = (input_type,np.zeros(array_shape))                 

        data = {}
        for x,y in inputs.items():
            data[x] = y[1]

        results = self.model.query(data)
        return results

class QueryServer:

    def __init__(self,args):
        self.query_config = None
        with open(args.query_config,'r') as fp:
            self.query_config = json.load(fp)

        log_filename = None
        if 'log_filename' in self.query_config:
            log_filename = self.query_config['log_filename']
        self.init_logging(log_filename)

        self.logger.info("initializing server...")

        #Load models
        self.models = []
        identifier = 0
        if not args.model_file is None:
            self.models.append(QueryModel(args.model_file))
        else:
            for m in self.query_config['models']:
                self.models.append(QueryModel(m,identifier))
                self.logger.info("loading model {} from {}".format(identifier,m))
                identifier += 1

        self.logger.info("models loaded successfully")

        self.target_name_to_models = {}
        for i in range(len(self.models)):
            m = self.models[i]
            for target_name in m.target_name_to_def:
                if not target_name in self.target_name_to_models:
                    self.target_name_to_models[target_name] = [i]
                else:
                    self.target_name_to_models[target_name].append(i)
        
        self.corenlp_server = None
        self.corenlp_port = 9000
        if 'corenlp_server' in self.query_config:
            self.corenlp_server = self.query_config['corenlp_server']
            if 'corenlp_port' in self.query_config:
                self.corenlp_port = self.query_config['corenlp_port']


        self.wordvector_file = self.query_config['wordvector_file']
        wv_path = Path(self.wordvector_file)
        if (not wv_path.exists()) or wv_path.is_dir():
            self.logger.critical("word vector file does not exist: {}".format(self.wordvector_file))
            raise FileNotFoundError

        self.hostname = self.query_config['hostname']
        self.port = self.query_config['port']

        self.dp = create_data.DataProcessor(self.wordvector_file)

    def query(self,text):
        query_input = self.dp.get_data(text,self.corenlp_server,self.corenlp_port,self.wordvector_file)

        results = []
        for i in range(len(self.models)):
            model = self.models[i]
            result = model.query(query_input)
            results.append(result)

        averaged_result = self.average_results(results,text)
        return json.dumps(averaged_result,indent=4,ensure_ascii=False)

    async def handler(self,websocket,path):
        while True:
            try:
                message = await websocket.recv()
                self.logger.info("received query: {}".format(message))
            except websockets.ConnectionClosed:
                break
            else:
                result = self.query(message)
                self.logger.info("sent reply: {}".format(result))
                await websocket.send(result)

    def start(self):
        start_server = websockets.serve(self.handler, self.hostname, self.port)
        self.logger.info("listening on {}:{}".format(self.hostname,self.port))
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

    #TODO break ties in a special way?
    def find_max_in_dict(self,dictionary):
        best_class = None
        max_votes = 0
        for class_name, num_votes in dictionary.items():
            if num_votes > max_votes:
                max_votes = num_votes
                best_class = class_name

        return best_class

    def average_results(self,results,query_string):
        query_string = self.dp.clean_string(query_string)
        query_string = query_string.split(" ")
        query_length = len(query_string)

        json_result = {}
        for target_name, model_indices in self.target_name_to_models.items():
            num_models = len(model_indices)
            target_type = self.models[model_indices[0]].target_name_to_def[target_name]['type']
            if target_type == "sentence_class":
                result = {}
                for i in model_indices:
                    model = self.models[i]
                    target_value = results[i][target_name]
                    predicted_class = target_value[0]
                    predicted_class_name = model.name_to_index_to_name[target_name][predicted_class]
                    if predicted_class_name in result:
                        result[predicted_class_name] += 1
                    else:
                        result[predicted_class_name] = 1

                best_class = self.find_max_in_dict(result)
                json_result[target_name] = best_class

            elif target_type == "class_sequence":
                result = []
                for i in range(query_length):
                    result.append({})

                for i in model_indices:
                    model = self.models[i]
                    target_value = results[i][target_name]
                    for j in range(query_length):
                        predicted_class = target_value[0,j]
                        predicted_class_name = model.name_to_index_to_name[target_name][predicted_class]
                        if predicted_class_name in result[j]:
                            result[j][predicted_class_name] += 1
                        else:
                            result[j][predicted_class_name] = 1

                best_class_name_list = []
                for i in range(query_length):
                    best_class = self.find_max_in_dict(result[i])
                    best_class_name_list.append((query_string[i],best_class))


                json_result[target_name] = best_class_name_list

            elif target_type == "fixed_length_class_sequence":
                #TODO possible error here if different models have different fixed lengths for this annotation, if they were trained on the same dataset this should not occur
                sequence_length = self.models[model_indices[0]].target_name_to_def[target_name]['sequence_length']
                result = []
                for i in range(sequence_length):
                    result.append({})

                for i in model_indices:
                    model = self.models[i]
                    target_value = results[i][target_name]
                    for j in range(sequence_length):
                        predicted_class = target_value[0,j]
                        predicted_class_name = model.name_to_index_to_name[target_name][predicted_class]
                        if predicted_class_name in result[j]:
                            result[j][predicted_class_name] += 1
                        else:
                            result[j][predicted_class_name] = 1

                best_class_name_list = []
                for i in range(sequence_length):
                    best_class = self.find_max_in_dict(result[i])
                    best_class_name_list.append(best_class)


                json_result[target_name] = best_class_name_list

        return json_result
    
    def init_logging(self,log_filename):
        #set up logging
        root_logger = logging.getLogger('root') 
        root_logger.setLevel(logging.DEBUG)

        #create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        #create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        root_logger.addHandler(ch) 

        #create rotating file handler
        try:
            if not log_filename is None:                
                maxBytes = 1024*1024*10
                backupCount=5
                fh = logging.handlers.RotatingFileHandler(log_filename, maxBytes=maxBytes, backupCount=backupCount,mode='a')
                fh.setFormatter(formatter)
                root_logger.addHandler(fh)
        except Exception as ex:
            root_logger.error("Could not create log file {}, reason: {}".format(log_filename,ex))


        self.logger = logging.getLogger('root.QueryServer')

        
def parse_args():
    parser = argparse.ArgumentParser(description="Perform NL classification with pre-trained neural networks")
    parser.add_argument("--query_config",type=str,default="query_config.json",help="json file containing configuration")
    parser.add_argument("--model_file", help="path to saved model, overrides models specified in query config file")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    query_server = QueryServer(args)
    query_server.start()
