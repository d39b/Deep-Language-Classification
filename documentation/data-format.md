# Data file format

To enable the simple creation and training of neural networks for natural language classifcation, dataset files must conform to a specific format.

Valid data files are in JSON format and contain a single object which in turn contains a `data` and `metadata` key.  
The value of the `data` key is an array of objects representing natural language sentences with annotations. Each object contains a key `text` whose string value is a natural language sentence. Additionally the objects can contain annotations of various types like simple strings or vector lists.    
The `metadata` key stores information about the annotations used in the dataset. It contains an array of objects `annotations`. Each object describes an annotation and must at least contain the string keys `name` and `type`. The value of the key `name` refers to the name of the key of the corresponding annotation in the `data` key of the dataset file. The value of key `type` is one of several possible types described below.  
Moreover, the `metadata` object should contain a key `sentence_length`, whose integer value is an upper bound on the number of words in sentences of the dataset file.

The following example shows a valid dataset file that contains multiple sentences annotated with a simple class name denoting the activity mentioned in the sentence. In particular, each sentence object in `data` contains a key `activity` with a string value. Accordingly there must be an annotation object in `metadata` with keys `name` and `type` and values `activity` and `sentence_class` respectively.

```javascript
{
    "metadata" : {
            "sentence_length" : 32,
            "annotations" : [
                {
                    "name" : "activity",
                    "type" : "sentence_class"
                }
            ]
    },
    "data" : [
        {
            "text" : "I watched the new Star Wars movie yesterday.",
            "activity" : "watchMovie"
        },
        {
            "text" : "Tim is reading a book on moral philosophy."
            "activity" : "readBook"
        },
        {
            "text" : "He is always listening to music.",
            "activity" : "listenMusic"
        }
    ]
}
```

## Annotation types

The following section describes the different types of annotations that can be used in dataset files.

### Sentence class
The simplest type of annotation is a `sentence_class`. It assigns a string value to each sentence. In the example above, the annotation named `"activity"` is of this type.
Sentence class annotations are usually used as targets for a neural network. For example given a sentence we want our neural network to predict the most fitting class.

### Fixed length class sequence
The value of an annotation of type `fixed_length_class_sequence` is an ordered list of class names, i.e. a list of strings. The number of class names in a list is bounded. The maximum number of elements must be specified in the `metadata` object for this annotation. If a list contains less strings than this maximum number, the missing elements are implicitly assumed to be `"None"`.

An annotation of this type can be seen as a combination of a fixed number of `sentence_class` annotations all using the same possible class names. The following example shows a dataset file containing an annotation of type `fixed_length_class_sequence`. The sentences in this dataset file mention up to two different activities. This can be modeled with an annotation of type `fixed_length_class_sequence` with maximum length 2.

```javascript
{
    "metadata" : {
        "sentence_length" : 32,
        "annotations" : [
            {
                "name" : "activities",
                "type" : "fixed_length_class_sequence",
                "sequence_length" : 2
            }
        ]
    },
    "data" : [
        {
            "text" : "After I watched the new Star Wars movie yesterday, I started reading a book on moral philosophy.",
            "activities" : [
                "watchMovie",
                "readBook"
            ]
        },
        {
            "text" : "Tomorrow after having lunch, we will visit my cousin in Berlin."
            "activities" : [
                "eat",
                "visit"
            ]
        },
        {
            "text" : "They are always listening to music.",
            "activities" : [
                "listenMusic"
            ]
        }
    ]
}
```

### Class sequence

The annotation type `class_sequence` behaves like `fixed_length_class_sequence` except that the fixed length is implicitly assumed to be the maximum sentence length defined in the `metadata` object of the dataset file. As such this annotation type is usually used to assign a class name to each word in a sentence.  

The value of an annotation of type `class_sequence` can be either a list of strings or a list of objects.  
For a list of strings the intended semantics is that the i-th string in the list is the class name of the i-th word of the sentence. The list doesn't have to be complete, missing values are implicitely assumed to be `"None"`.
In many cases we want to assign a special class only to a few words in a sentence and the rest of the words all belong to the `"None"` class. In this case it is more convenient to only explicitly define the class names for these few words. For this purpose, an annotation of type `class_sequence` can also take as value a list of objects. Each object contains a `type` and `index` key, corresponding to the class name and index of the word in the sentence to which the class name belongs (indices starting from 0).

The dataset file in the following example contains an annotation `"entities"` of type `class_sequence`.

```javascript
{
    "metadata" : {
        "sentence_length" : 32,
        "annotations" : [
            {
                "name" : "entities",
                "type" : "class_sequence"
            }
        ]
    },
    "data" : [
        {
            "text" : "Tomorrow after having lunch, we will visit my cousin in Berlin.",
            "entities" : [
                {
                    "type" : "location",
                    "index" : 10
                },
                {
                    "type" : "person",
                    "index" : 8
                },
                {
                    "type" : "time",
                    "index" : 0
                }
            ]
        }
    ]
}
```

### Vector sequence

The value of an annotation of type `vector_sequence` is a list integers, where each integer is the index of a vector in a separate vector list. The vector list is stored at the top level of the dataset file using as a key the annotation name. Each element in the vector list is a list containing an identifier and a vector. A vector sequence should contain at most as many integer indices as the maximum sentence length defined in the `metadata` object of the dataset file. Additionally all vectors in the vector list should have the same length. The description of a `vector_sequence` annotation in the `metadata` object of the dataset file specifies the vector length via the key `vector_length`.  
A `vector_sequence` annotation is typically used to represent each word of the sentence as a numerical vector. Accordingly the i-th vector of the sequence represents the i-th word of the sentence.

The following example contains an annotation `"wordVectors"` of type `vector_sequence`.

```javascript
{
    "data" : [
        {
            "text" : "I watched the new Star Wars movie yesterday."
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

### Graph structure

The annotation type `graph_structure` can be used to describe directed graphs. Its value is a list of objects, where each object describes a directed edge of the graph and contain 3 keys. `origin` and `target` specify the vertex indices of the edge and `type` specifies a type.  
Annotations of type `graph_structure` can for example be used to model dependencies between words. The `origin` and `target` indices correspond to indices of words that have a dependency.

The following example shows a datset file with an annotation `"dependencies"` of type `graph_structure`.

```javascript
{
    "metadata" : {
        "sentence_length" : 32,
        "annotations" : [
            {
                "type" : "graph_structure",
                "name" : "dependencies"
            }
        ]
    },
    "data" : [
        {
            "text" : "I watched the new Star Wars movie yesterday",
            "dependencies" : [
                {
                    "origin" : 1,
                    "target" : 6,
                    "type" : "nobj"
                },
                {
                    "origin" : 2,
                    "target" : 6,
                    "type" : "det"
                },
                ...
            ]
        }
    ]
}
```
