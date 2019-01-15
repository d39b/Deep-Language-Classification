# Setting up a Stanford CoreNLP server for use with this project

* Download the [CoreNLP package](https://stanfordnlp.github.io/CoreNLP/index.html#download) and unzip the file.

* Download models for your language, e.g. the [German language model](https://stanfordnlp.github.io/CoreNLP/index.html#download) jar file and copy it into the CoreNLP root directory.

* Create a properties file, e.g. `my.properties` with the following content (change accordingly for different languages):

```properties
annotators = tokenize, ssplit, pos, depparse                                                             

tokenize.language = de

#split words only at whitespace
tokenize.whitespace = True

#one sentence per line
ssplit.eolonly = True

#use german models
pos.model = edu/stanford/nlp/models/pos-tagger/german/german-hgc.tagger

depparse.model    = edu/stanford/nlp/models/parser/nndep/UD_German.gz
depparse.language = german
```

Let `stanford-corenlp-full` be the path to the root directory of the CoreNLP server. To start the server on port 9000 run:

> java -cp "stanford-corenlp-full/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties my.properties -port 9000 -timeout 15000 -preload "tokenize, ssplit, pos, depparse"

The `-preload` argument forces the server to preload the needed annotator models. Without this the first query might take some time to process.
