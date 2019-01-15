# Filling patterns
The script `create_data.py` can be used to generate complete sentences by filling sentences containing placeholders with short phrases. It takes as input two files `pattern-file` and `phrase-file` in the dataset format described in [Data format](data-format.md).

## Example
`pattern-file` contains the sentence `Tim _ before he _ ?`  
`phrase-file` contains the short phrases `watched a movie` and `read a book`.  
The resulting dataset file will contain the sentence `Tim watched a movie before he read a book`

For each sentence in `pattern-file` 10 complete sentences are created by replacing the placeholder symbols with randomly sampled sentences from `phrase-file`.

## Usage
> python3 create_data.py fill-patterns pattern-file phrase-file output-file

## Arguments
`pattern-file` : dataset file with sentences containing placeholder symbols  
`phrase-file` : dataset file  
`output-file` : file to write output to

To specify a different placeholder symbol use the option `--pattern_mask`.  
To change the number of output sentences created per sentence in `pattern-file` use the option `--num_sentences_per_pattern`.

> python3 create_data.py fill-patterns pattern-file phrase-file output-file --pattern_mask * --num_sentences_per_pattern 42
