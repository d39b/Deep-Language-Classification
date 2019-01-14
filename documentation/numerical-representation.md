# Creating word vector annotations

The script `create_data.py` can be used to represent the sentences of a dataset file using word vectors. Takes a dataset and word vector file as input and adds word vectors with an annotation of type `vector_sequence` and name `"wordVectors"` to the dataset. If there are words in the dataset for which no vector exists in the word vector file, the program tries to find a similar word that has a vector. If there is no sufficiently similar word, a random vector is assigned to the unknown word. The program also outputs a new word vector file containing the vectors for the words in the dataset file and some similar words (see [Outputting additional word vectors](#outputting-additional-word-vectors) below).

[Facebook FastText](https://github.com/facebookresearch/fastText) offers word vector files for many languages.

## Usage

> python3 create_data.py find-vectors  dataset  word-vector-file

## Arguments
`dataset` : dataset file in the format specified in [Data format](data-format.md)  
`word-vector-file` : file containing word vectors, each line should contain a word followed by a sequence of numbers, separated by spaces  

By default the changes are written directly to the input dataset file. To use a separate output file, use the option `--output_file`.

## Outputting additional word vectors

Use the option `--augment-vector-file` to output additional word vectors in the new word vector file. For each word in the dataset, at most 20 words with a similarity score of at least 0.8 are added. Furthermore the first words from the input word vector file are added until the new word vector file contains 500000 words (word vector files from Facebook FastText list words by frequency).

> python3 create_data.py find-vectors dataset word-vector-file --augment-vector-file

Further options:

`--augment_cutoff` : minimum similarity, default 0.8  
`--augment_n` : maximum number of similar words to add, default 20  
`--augment_max_word_count` : maximum number of words in new word vector file, default 500000

# Creating POS and dependency annotations

The script `create_data.py` can also be used to add Part-Of-Speech tag and dependency information to a dataset file. Takes as input a dataset file and adds an annotation of type `class_sequence` for Part-Of-Speech tags and an annotation of type `graph_structure` for dependencies to it. The names of the annotations will be `"pos"` and `"dep"` respectively.

The program requires a running Stanford CoreNLP Server. See the [guide](corenlp-setup.md) on how to set one up for more information.

## Usage

> python3 create_data.py find-dependencies dataset

## Arguments
`dataset` : dataset file

By default changes are written directly to the dataset file. To specify a separate output file use the option `--output_file`.

The Stanford CoreNLP Server is assumed to be running on localhost port 9000. To use a different hostname or port use the options `--corenlp_server` and `--corenlp_port`.
