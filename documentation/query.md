# Staring a query server

The script `query_socket.py` can be used to start a server that accepts user queries on websocket connections. To run a server you need a trained neural network model and a query server configuration file. It is possible to use multiple trained models with a single server. A query is then processed by all specified neural network models and the final output is decided by majority vote.

## Usage

> python3 query_socket.py  query-config

## Arguments
`query-config` : query server configuration file

## Configuration file

The query server configuration file contains a JSON object with the following keys:

`models` :  a list of paths to saved neural network models  
`wordvector_file` : file containing word vectors   
`hostname` : hostname of the query server  
`port` : port of the query server

The following keys are optional:  

`log_filename` : file to store logs in  
`corenlp_server` : address of Stanford CoreNLP server  
`corenlp_port` : port of Stanford CoreNLP server  

Example:

```javascript
{
    "models" : [
        "saved_neural_network"
    ],
    "corenlp_server" : "localhost",
    "corenlp_port" : 9000,
    "wordvector_file" : "word-vectors.txt",
    "log_filename" : "query_server.log",
    "hostname" : "localhost",
    "port" : 4567
}
```
