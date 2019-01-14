import numpy as np
import json
import re
import argparse
import http.client
import urllib.parse
import difflib
import logging

class DataProcessor:
    """
    Class that can be used to find word vectors or POS tags and syntactic dependencies.

    If an object of this class is used to find word vectors, a dictionary from words to
    line numbers in the word vector file are created. For large word vector files the
    vectors may not fit into memory. The word vectors for a specific set of words can then
    be found faster if the line numbers containing the necessary word vectors are already known.

    If an object of this class is used to find POS tags and syntactic dependencies,
    a running Stanford CoreNLP server is necessary.

    Attributes
    ----------
    vector_file : str
        filename of word vector file this object uses
    words_loaded : bool
        true if the words in the word vector file have been read in
    words : list of str
        list of words in the word vector file
    word_to_index : dict
        dictionary from word to line number in the word vector file
    vector_length : int
        length of word vectors of the word vector file
    """

    def __init__(self,vector_file=None):
        """
        Create a DataProcessor object.

        If an optional word vector file is provided as argument, the file is read and a dictionary
        from words to line numbers is created to speed up later processing.

        Parameters
        ----------
        vector_file : str, optional
            filename of word vector file (default is None)
        """

        self.logger = logging.getLogger('root.DataProcessor')
        self.vector_file = vector_file
        self.words_loaded = False
        self.words = []
        self.word_to_index = {}
        self.vector_length = 0
        if not vector_file is None:
            self.logger.info("loading words from file: {}".format(vector_file))
            self.load_words(vector_file)
            self.words_loaded = True

    def sample_phrases(self,phrases,count):
        """
        Sample a given number of random elements from a list.

        Parameters
        ----------
        phrases : list
            list to sample elements from
        count : int
            number of elements to sample
        """

        result = []
        for i in range(count):
            index = np.random.randint(0,len(phrases))
            result.append(phrases[index])
        return result

    def insert_phrases(self,pattern,phrases,pattern_mask):
        """
        Replaces placeholder symbols in a string with other strings.

        Example:
        pattern = "This is a _ with _"
        phrases = ["string", "placeholders"]
        pattern_mask = "_"
        -> "This is a string with placeholders"

        Parameters
        ----------
        pattern : str
            string containing placeholders to be replaced
        phrases : list of str
            list of strings used to replace placeholder symbols with
        pattern_mask : str
            placeholder symbol
        """

        #check if pattern string contains enough placeholder symbols
        count = len(phrases)
        real_count = pattern.count(pattern_mask)
        if count > real_count:
            self.logger.warning("pattern did not contain enough placeholder symbols");
            count = real_count

        result = pattern
        startIndices = []
        endIndices = []
        for i in range(count):
            phrase = phrases[i].split(" ")
            index = result.find(pattern_mask)
            pre_pattern_length = len(result[0:index].strip().split())
            phrase_length = len(phrase)
            #compute start and end word indices of inserted phrase
            startIndices.append(pre_pattern_length)
            endIndices.append(pre_pattern_length+phrase_length-1)
            result = self.insert_phrase(result,phrases[i],index)

        return (result, startIndices, endIndices)

    def insert_phrase(self,pattern,phrase,index):
        """
        Replace a character in a string with another string.

        Parameters
        ----------
        pattern : str
            string in which to replace a character
        phrase : str
            string inserted instead of character
        index : int
            index of character to be replaced
        """

        return str(pattern[0:index] + phrase + pattern[index+1:])

    def stringify_npvec(self,vec):
        result = []
        dim = len(vec)
        for i in range(dim):
            result.append(str(vec[i]))
        return result

    def get_unknown_vec(self,dimension):
        #TODO also sample negative values here, i.e. in range -1,1
        return np.random.rand(dimension).tolist()

    def get_zero_vec(self,dimension):
        return np.zeros(dimension).tolist()

    def clean_string(self,s,pattern_mask="_"):
        """
        Removes non alpha-numeric characters from a string.

        All characters except a-z, A-Z, 0-9, ä, ö, ü, Ä, Ö, Ü, ß and the character
        given via the argument 'pattern_mask' are removed from the input string.

        Parameters
        ----------
        s : str
            string to remove characters from
        pattern_mask : str, optional
            the character(s) specified by this string are not removed (default is "_")
        """

        chars_to_keep = r'[^a-zA-Z0-9äöüÄÖÜß '+pattern_mask+']'
        result = re.sub(chars_to_keep,'',s)
        result = result.strip()
        result = result.lower()
        #replace consecutive whitespaces with a single whitespace
        result = re.sub(' +',' ',result)
        return result

    def load_words(self,vector_file):
        """
        Reads a word vector file and creates a dictionary from words to line numbers as self.word_to_index.

        Parameters
        ----------
        vector_file : str
            filename of word vector file
        """
        
        self.words = []
        self.word_to_index = {}
        self.vector_length = 0
        line_number = 0
        with open(vector_file,'r') as fp:
            first_line = True
            for line in fp:
                if first_line:
                    first_line = False
                    if line.count(" ") <= 2:
                        continue

                tmp = line.rstrip().split(" ")
                word = tmp[0]
                if self.vector_length == 0:
                    vec = tmp[1:]
                    self.vector_length = len(vec)
                self.words.append(word)
                self.word_to_index[word] = line_number
                line_number += 1

    def find_word_vectors(self,words,vector_file,unknown_vec):
        """
        Finds word vectors in a word vector file for a given set of words.

        If a word is not contained in the word vector file, the vector of a word from the input set
        with similarity at least 0.8 is used.
        If no such word exists, the vector of a word from the
        word vector file with similarity at least 0.8 is used.
        If no similar word could be found, a zero or random vector is used, according to whether
        the 'unknown_vec' argument is set to "zero" or "random".
        For information on the computation of word similaries see the Python library difflib.

        Returns a dictionary from words to vectors.

        Parameters
        ----------
        words : set of str
            a set of words
        vector_file : str
            filename of a word vector file
        unknown_vec : str
            mode used to create new vectors for unknown words, possible values "zero" or "random"
        """

        word_to_vector = {}
        vector_length = self.vector_length

        word_indices = []
        known_words = []
        unknown_words = []
        for word in words:
            if word in self.word_to_index:
                word_indices.append(self.word_to_index[word])
                known_words.append(word)
            else:
                unknown_words.append(word)

        #first seach for similar words only among words in the dataset because those might be semantically more similar
        still_unknown = []
        now_known = []
        matched_words = []
        close_matches = self.find_similar_words(unknown_words,known_words,n=1,cutoff=0.8)
        for i in range(len(close_matches)):
            matches = close_matches[i]
            unknown_word = unknown_words[i]
            if len(matches) == 0:
                still_unknown.append(unknown_word)
            else:
                most_similar_word = matches[0]
                now_known.append(unknown_word)
                matched_words.append(most_similar_word)

        #if the above failed, search for similar words among all words
        tmp_still_unknown = []
        close_matches = self.find_similar_words(still_unknown,self.words)
        for i in range(len(close_matches)):
            matches = close_matches[i]
            unknown_word = still_unknown[i]
            if len(matches) == 0:
                tmp_still_unknown.append(unknown_word)
            else:
                most_similar_word = matches[0]
                now_known.append(unknown_word)
                matched_words.append(most_similar_word)

        still_unknown = tmp_still_unknown

        #create list of necessary line numbers to load from word vector file
        for word in matched_words:
            word_index = self.word_to_index[word]
            if not word_index in word_indices:
                word_indices.append(word_index)

        #load word vectors by line number
        if len(word_indices) > 0:
            word_indices.sort()
            list_index = 0
            current_index = 0
            next_index = word_indices[list_index]

            with open(vector_file,'r') as fp:
                first_line = True
                for line in fp:
                    if first_line:
                        first_line = False
                        if line.count(" ") <= 2:
                            continue

                    if current_index == next_index:
                        tmp = line.rstrip().split(" ")
                        word = tmp[0]
                        vec = []
                        for j in range(1,len(tmp)):
                            vec.append(float(tmp[j]))

                        if vector_length == 0:
                            vector_length = len(vec)
                        if not word in word_to_vector:
                            word_to_vector[word] = vec

                        list_index += 1
                        if list_index < len(word_indices):
                            next_index = word_indices[list_index]
                        else:
                            break

                    current_index += 1

        for i in range(len(now_known)):
            word_to_vector[now_known[i]] = word_to_vector[matched_words[i]]

        #create vectors for still unknown words (i.e. no word with minimum required similarity exists)
        for word in still_unknown:
            if unknown_vec == "random":
                word_to_vector[word] = self.get_unknown_vec(vector_length)
            elif unknown_vec == "zero":
                word_to_vector[word] = self.get_zero_vec(vector_length)

        return (word_to_vector, vector_length)


    def find_similar_words(self,words,known_words,n=1,cutoff=0.6):
        """
        Finds similar words for a given set of words.

        For each word in a given set of words, finds the most similar words
        from a second given set of words. Returns at most 'n' similar words
        and only words with a similarity of at least 'cutoff'.

        Parameters
        ----------
        words : list of str
            words for which similar words need to be found
        known_words : list of str
            list of candidate similar words
        n : int, optional
            maximum number of similar words to return for each word in 'words' (default is 1)
        cutoff : float, optional
            minimum similarity needed for a word to be considered (default is 0.6)
        """

        result = []
        for i in range(len(words)):
            word = words[i]
            self.logger.info("finding similar words for {}, {}/{}".format(word,i+1,len(words)))
            close_matches = difflib.get_close_matches(word,known_words,n=n,cutoff=cutoff)
            result.append(close_matches)
        return result


    def find_vectors(self,data,text_name,vector_file,new_vector_file=None,target_name="wordVectors",token_name="tokens",unknown_vec="random"):
        """
        Finds word vectors and adds them as an 'vector_sequence' annotation to the given dataset object.

        Parameters
        ----------
        data : list or dict
            dataset object containing sentences (with annotations)
        text_name : str
            string key of sentences
        vector_file : str
            filename of word vector file
        new_vector_file : str, optional
            filename to write found word vectors to or None if no such file should be created (default is None)
        target_name : str, optional
            name of the newly created word vector annotation (default is "wordVectors")
        token_name : str, optional
            name of the newly created 'class_sequence' annotation containing the individual words of sentences (default is "tokens")
        unknown_vec : str, optional
            mode to create vectors for unknown words with (default is "random")
        """

        #ensure that dataset conforms to dataset format, add metadata section if necessary
        result = None
        metadata = None
        if isinstance(data,dict):
            result = data
            data = data['data']
            metadata = result['metadata']
        else:
            metadata = self.get_metadata()
            result = self.add_metadata(data,metadata)

        sentences = []

        for i in range(len(data)):
            d = data[i]
            sentence = d[text_name]
            sentences.append(sentence)

        #create set of words of the dataset
        words = set()
        for s in sentences:
            for w in s.split(" "):
                words.add(w)

        #find word vectors
        word_to_vector, vector_length = self.find_word_vectors(words,vector_file,unknown_vec)

        vector_list = []
        word_to_index = {}

        curr_index = 0
        for word, vector in word_to_vector.items():
            vector_list.append([word,vector])
            word_to_index[word] = curr_index
            curr_index += 1

        tmp = []
        for i in range(len(sentences)):
            s = sentences[i]
            tmp.append(s.split(" "))
        sentences = tmp

        #add word vectors to dataset
        for i in range(len(sentences)):
            s = sentences[i]
            d = data[i]
            v = []
            tokens = []
            for j in range(len(s)):
                vec_index = word_to_index[s[j]]
                v.append(vec_index)
                tokens.append(s[j])

            d[target_name] = v
            d[token_name] = tokens

        #write found word vectors to new word vector file
        if not new_vector_file is None:
            with open(new_vector_file,'w') as fp:
                for word, vec in word_to_vector.items():
                    output_string = word
                    for i in range(len(vec)):
                        output_string += " " + str(vec[i])
                    output_string += "\n"
                    fp.write(output_string)

        result["wordVectors"] = vector_list

        metadata["annotations"].append({
                    "type" : "vector_sequence",
                    "name" : "wordVectors",
                    "vector_length" : vector_length
                })

        metadata["annotations"].append({
                    "type" : "class_sequence",
                    "name" : "tokens",
                })

        return result

    def find_dependencies_with_corenlp(self,data,text_name,server,port,pos_target_name="pos",dep_target_name="dep",character_limit=10000):
        """
        POS tags and syntactic dependencies are computed with a Stanford CoreNLP Server
        and added to the given dataset as annotations of type 'class_sequence'
        as 'graph_structure'.

        Parameters
        ----------
        data : list or dict
            dataset object containing sentences (with annotations)
        text_name : str
            string key of sentences
        server : str
            hostname of Stanford CoreNLP server
        port : int
            port of Stanford CoreNLP server
        pos_target_name : str, optional
            name of POS annotation (default is "pos")
        dep_target_name : str, optional
            name of dependency annotation (default is "dep")
        character_limit : int, optional
            maximum number of characters to send to Stanford CoreNLP server in one query (default is 10000)
        """

        result = data
        data = data['data']

        #buffer is filled with sentences until max character limit is reached
        #then buffer is send to Stanford CoreNLP server
        i = 0
        sentence_buffer = []
        buffer_start_index = 0
        buffer_length = 0
        try:
            while i < len(data):
                d = data[i]
                sentence = d[text_name]
                #add +1 here for newlines that are later added
                buffer_full = False
                if buffer_length + len(sentence) + 1 < character_limit:
                    if len(sentence_buffer) == 0:
                        buffer_start_index = i
                    sentence_buffer.append(sentence)
                    buffer_length += len(sentence) + 1
                else:
                    buffer_full = True

                if buffer_full or i == (len(data) - 1):
                    query_results = self.query_call(sentence_buffer,server,port)['sentences']

                    if len(query_results) != len(sentence_buffer):
                        self.logger.error("corenlp parser did not return the right number of sentences")

                    #add POS and dependencies to dataset
                    for j in range(len(sentence_buffer)):
                        d = data[buffer_start_index+j]
                        dep = query_results[j]
                        bd = dep['basicDependencies']
                        tokens = dep['tokens']
                        pos_tags = []
                        for token in tokens:
                            token_index = token['index'] - 1
                            tag = token['pos']
                            pos_tags.append({
                                "index" : token_index,
                                "type" : tag
                            })

                        d[pos_target_name] = pos_tags

                        dependencies = []
                        for x in bd:
                            origin = x['governor'] - 1
                            target = x['dependent'] - 1
                            tag = x['dep']
                            if origin >= 0:
                                dependencies.append({
                                    "origin" : origin,
                                    "target" : target,
                                    "type" : tag
                                })

                        d[dep_target_name] = dependencies

                    sentence_buffer = []
                    buffer_length = 0

                if not buffer_full:
                    i += 1

            metadata = result['metadata']
            metadata['annotations'].append({
                "type" : "graph_structure",
                "name": "dep"
            })
            metadata['annotations'].append({
                "type" : "class_sequence",
                "name": "pos"
            })

        except Exception:
            self.logger.error("could not compute pos-tags and dependencies")
                        
        return result

    def query_call(self,text,server,port,max_tries=5):
        """
        Sends a given list of strings to a Stanford CoreNLP server for processing.

        Parameters
        ----------
        text : list of str
            list of strings to send to server
        server : str
            hostname of Stanford CoreNLP server
        port : int
            port of Stanford CoreNLP server
        """

        #sends a list of sentences to Stanford CoreNLP server
        method = "POST"
        header = {}
        params = {"properties" : {"outputFormat":"json"}}
        params = urllib.parse.urlencode(params)
        path = "/?%s" % params
        body = ""
        for i in range(len(text)):
            body += text[i] + "\n"

        response_status = None
        response = None
        response_string = None
        try_number = 0
        while response_status != 200 and try_number < max_tries:
            if not response_status is None:
                self.logger.error("server responded with code {}".format(response_status))
                self.logger.info("trying again...")
            try:
                conn = http.client.HTTPConnection(server,port=port)
                conn.request(method,path,body,header)
                response = conn.getresponse()
                response_status = response.status
                response_string = response.read().decode('UTF-8')
                conn.close()
            except Exception as ex:
                self.logger.error("http request to CoreNLP server failed, reason: {}".format(ex))
                response_string = None

            try_number += 1
        
        if response_string is None:
            raise ValueError

        result = json.loads(response_string)
        return result

    def fill_patterns(self,pattern_file,phrase_file,pattern_mask,num_sentences_per_pattern,pattern_text_name,phrase_text_name,output_text_name):
        """
        Fills strings that contain placeholder symbols from one dataset
        with randomly sampled strings from another dataset and returns a new
        dataset object.

        Parameters
        ----------
        pattern_file : str
            filename of datset file containing strings with placeholder symbols
        phrase_file : str
            filename of datset file containing strings used to replace placeholder symbols with
        pattern_mask : str
            placeholder symbol to replace
        num_sentences_per_pattern : int
            number of complete strings to create per string of pattern_file
        pattern_text_name : str
            dictionary key of strings in pattern_file
        phrase_text_name : str
            dictionary key of strings in phrase_file
        output_text_name : str
            dictionary key of strings in output file
        """

        patterns = None
        with open(pattern_file,'r') as fp:
            patterns = json.load(fp)
        patterns_metadata = patterns['metadata']
        patterns = patterns['data']

        phrases = None
        with open(phrase_file,'r') as fp:
            phrases = json.load(fp)
        phrases_metadata = phrases['metadata']
        phrases = phrases['data']
        phrases_annotation_name_to_type = {}
        for at in phrases_metadata['annotations']:
            phrases_annotation_name_to_type[at['name']] = at['type']

        for pattern in patterns:
            clean_text = self.clean_string(pattern[pattern_text_name],pattern_mask=pattern_mask)
            pattern[pattern_text_name] = clean_text

        for phrase in phrases:
            clean_text = self.clean_string(phrase[phrase_text_name],pattern_mask=pattern_mask)
            phrase[phrase_text_name] = clean_text

        result = []
        max_sentence_length = 0
        max_pattern_arguments = 0
        metadata = self.get_metadata()
        metadata['annotations'] = patterns_metadata['annotations']
        for pattern in patterns:
            pattern_text = pattern[pattern_text_name]
            count = pattern_text.count(pattern_mask)
            if count > max_pattern_arguments:
                max_pattern_arguments = count

            #copy annotations from pattern sentence
            other = {}
            for key,value in pattern.items():
                if key != output_text_name:
                    other[key] = value

            #sample random phrases and insert them into pattern
            for i in range(num_sentences_per_pattern):
                phrases_to_insert = self.sample_phrases(phrases,count)
                phrases_text = []
                for phrase_to_insert in phrases_to_insert:
                    phrases_text.append(phrase_to_insert[phrase_text_name])

                complete_text, startIndices, endIndices = self.insert_phrases(pattern_text,phrases_text,pattern_mask)
                sentence_length = len(complete_text.split(" "))
                if sentence_length > max_sentence_length:
                    max_sentence_length = sentence_length

                #create object containing all information about newly created sentence
                new_dict = {
                    output_text_name : complete_text,
                    "startIndices" : startIndices,
                    "endIndices" : endIndices
                }
                #add annotations from pattern sentence
                new_dict.update(other)

                #combine annotations from phrases
                for at_name, at_type in phrases_annotation_name_to_type.items():
                    if at_type == "sentence_class":
                        new_list_value = []
                        for j in range(len(phrases_to_insert)):
                            phrase = phrases_to_insert[j]
                            new_list_value.append(phrase[at_name])

                        new_dict[at_name] = new_list_value

                    elif at_type == "class_sequence":
                        new_list_value = []
                        new_list_value_2 = []

                        for j in range(len(phrases_to_insert)):
                            phrase = phrases_to_insert[j]
                            old_value = phrase[at_name]
                            for x in old_value:
                                if isinstance(x,dict):
                                    copy = {}
                                    copy.update(x)
                                    if 'index' in copy:
                                        copy['index'] = startIndices[j] + copy['index']
                                        copy_2 = {}
                                        copy_2['index'] = copy['index']
                                        copy_2['type'] = "element" + str(j)
                                        new_list_value.append(copy)
                                        new_list_value_2.append(copy_2)
                                elif isinstance(x,str):
                                    new_list_value.append(x)
                                    new_list_value_2.append("element"+str(j))

                            new_dict[at_name] = new_list_value
                            new_dict[at_name+"PhraseIndex"] = new_list_value_2

                    elif at_type == "fixed_length_class_sequence":
                        new_list_value = []

                        for j in range(len(phrases_to_insert)):
                            phrase = phrases_to_insert[j]
                            new_list_value.extend(phrase[at_name])

                        new_dict[at_name] = new_list_value

                result.append(new_dict)


        for at in phrases_metadata['annotations']:
            at_name = at['name']
            at_type = at['type']
            if at_type == "sentence_class":
                metadata["annotations"].append(
                        {
                            "name" : at_name,
                            "type" : "fixed_length_class_sequence",
                            "sequence_length" : max_pattern_arguments
                        }
                )
            elif at_type == "class_sequence":
                metadata["annotations"].append(
                        {
                            "name" : at_name,
                            "type" : "class_sequence"
                        }
                )
                metadata["annotations"].append(
                        {
                            "name" : at_name+"PhraseIndex",
                            "type" : "class_sequence"
                        }
                )
            elif at_type == "fixed_length_class_sequence":
                metadata["annotations"].append(
                        {
                            "name" : at_name,
                            "type" : "fixed_length_class_sequence",
                            "sequence_length" : (at["sequence_length"]*max_pattern_arguments)
                        }
                )

        metadata["sentence_length"] = max_sentence_length

        return (result, metadata)

    def get_metadata(self,sentence_length=None):
        """
        Returns an empty metadata dictionary.

        Parameters
        ----------
        sentence_length : int, optional
            maximum sentence length (default is None)
        """

        metadata = {
            "annotations" : []
        }
        if not sentence_length is None:
            metadata["sentence_length"] = sentence_length

        return metadata

    def add_metadata(self,data,metadata):
        """
        Combine data list and metadata dictionary to form complete dataset.

        Parameters:
        data : list
            list of objects each of which represents a natural language sentence with annotations
        metadata : dict
            information about annotations used in the data list
        """

        result = {
            "metadata" : metadata,
            "data" : data
        }
        return result

    def sentence_to_json(self,sentence):
        result = [{
            "text" : sentence
        }]
        return result

    def get_data(self,sentence,server,port,vec_file):
        """
        Finds and returns word vectors, POS tags and dependencies for a given sentence.

        Result is dataset object containing only one sentence.

        Parameters
        ----------
        sentence : str
            string containing the sentence text
        server : str
            hostname of Stanford CoreNLP server
        port : int
            port of Stanford CoreNLP server
        vec_file : str
            filename of word vector file
        """

        sentence = self.clean_string(sentence)
        sentence_length = sentence.count(' ') + 1
        x = self.find_vectors(self.sentence_to_json(sentence),'text',vec_file,unknown_vec="zero")
        x['metadata']["sentence_length"] = sentence_length
        if not server is None:
            x = self.find_dependencies_with_corenlp(x,'text',server,port)
        return x

    def augment_word_vector_file(self,new_vector_file,old_vector_file,max_word_count,cutoff,n):
        """
        Adds similar words and most common words from one word vector file to another.

        For each word in 'new_vector_file' similar words are found among the words in 'old_vector_file'.
        At most 'n' words with similarity score at least 'cutoff' from 'old_vector_file' are added
        per word of 'new_vector_file'. All these similar words are appended to 'new_vector_file'.
        Furthermore the most common words from 'old_vector_file' are added to 'new_vector_file' until
        'new_vector_file' contains 'max_word_count' words with wordvectors.

        Parameters
        ----------
        new_vetor_file : str
            filename of vector file to be augmented with similar words
        old_vector_file : str
            filename of vector file from which to extract similar words
        max_word_count : int
            maximum word count for new_vector_file
        cutoff : float
            minimum word similarity
        n : int
            maximum number of similar words to be added per word of new_vector_file
        """

        #it is assumed that old_vector_file was already loaded with this DataProcessor object

        #load words from new_vetor_file
        words = []
        with open(new_vector_file,'r') as fp:
            for line in fp:
                tmp = line.rstrip().split(" ")
                word = tmp[0]
                words.append(word)

        # k is number of most common words that need to be added (if n similar words can be found for each word from new_vector_file)
        k = max_word_count - len(words)*n+1

        #add k most common names to list of line numbers thate need to be loaded from old_vector_file
        word_indices = set()
        for i in range(min(k,len(self.words))):
            if not self.words[i] in words:
                word_indices.add(i)

        #find similar words
        close_matches = self.find_similar_words(words,self.words,n=n,cutoff=cutoff)
        for i in range(len(close_matches)):
            matches = close_matches[i]
            for word in matches:
                if not word in words:
                    word_index = self.word_to_index[word]
                    if not word_index in word_indices:
                        if word_index >= k:
                            word_indices.add(word_index)

        word_indices = list(word_indices)
        word_indices.sort()

        #load all necessary words using the list of line numbers
        with open(new_vector_file,'a') as fp:
            list_index = 0
            current_index = 0
            next_index = 0

            if len(word_indices) > 0:
                next_index = word_indices[list_index]

            with open(old_vector_file,'r') as ofp:
                first_line = True
                for line in ofp:
                    if first_line:
                        first_line = False
                        if line.count(" ") <= 2:
                            continue

                    #if current_index < k:
                    #    fp.write(line)
                    if len(word_indices) > 0 and current_index == next_index:
                        fp.write(line)

                        list_index += 1
                        if list_index < len(word_indices):
                            next_index = word_indices[list_index]
                        else:
                            break

                    current_index += 1

    def load_file(self,input_file):
        result = None
        with open(input_file,'r') as fp:
            result = json.load(fp)
        return result

    def write_file(self,data,output_file):
        with open(output_file,'w') as fp:
            json.dump(data,fp,ensure_ascii=False,indent=4)


############################################################
# Functions for using this script as a command line program.
############################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Create sentences by filling patterns with phrases, find word vectors and dependencies")
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_fill_patterns = subparsers.add_parser('fill-patterns', help="Fill patterns help")
    parser_fill_patterns.add_argument("pattern_file", help="file containing pattern sentences")
    parser_fill_patterns.add_argument("phrase_file", help="file containing phrases")
    parser_fill_patterns.add_argument("output_file", help="file to write results in")
    parser_fill_patterns.add_argument("--pattern_mask", type=str, default="_", help="character that is used in pattern file to mark positions where phrases can be inserted")
    parser_fill_patterns.add_argument("--num_sentences_per_pattern", type=int, default="10", help="number of sentences to generate for each pattern")
    parser_fill_patterns.add_argument("--pattern_text_name", type=str, default="text", help="name of the json attribute for sentence text in the pattern file")
    parser_fill_patterns.add_argument("--phrase_text_name", type=str, default="text", help="name of the json attribute for sentence text in the phrase file")
    parser_fill_patterns.add_argument("--output_text_name", type=str, default="text", help="name of the json attribute for sentence text in the output file")
    parser_fill_patterns.set_defaults(func=fill_patterns)

    parser_find_vectors = subparsers.add_parser('find-vectors', help="Find vector help", aliases=['find-vec','vec'])
    parser_find_vectors.add_argument("input_file", help="file containing sentences")
    parser_find_vectors.add_argument("vec_file", help="file containing word vectors")
    parser_find_vectors.add_argument("--input_text_name", type=str, default="text", help="name of the json attribute for sentence text")
    parser_find_vectors.add_argument("--output_file", help="file to write result to, if not specified, input file is overwritten")
    parser_find_vectors.add_argument("--augment_vec_file",action="store_true",help="augment output word vector file with top k and similar words")
    parser_find_vectors.add_argument("--augment_max_word_count",type=int,default=500000,help="maximum number of words when augmenting output word vector file")
    parser_find_vectors.add_argument("--augment_cutoff",type=float,default=0.8,help="minimum similarity when finding similar words")
    parser_find_vectors.add_argument("--augment_n",type=int,default=20,help="maximum number of similar words to add for each original word")
    parser_find_vectors.set_defaults(func=find_vectors)

    parser_find_dep = subparsers.add_parser('find-dependencies', help="Find dependencies help", aliases=['find-dep','dep'])
    parser_find_dep.add_argument("input_file", help="file containing sentences")
    parser_find_dep.add_argument("--input_text_name", type=str, default="text", help="name of the json attribute for sentence text")
    parser_find_dep.add_argument("--output_file", help="file to write result to, if not specified, input file is overwritten")
    parser_find_dep.add_argument("--corenlp_server", type=str, default="localhost", help="host running corenlp server")
    parser_find_dep.add_argument("--corenlp_port", type=int, default=9000, help="port of corenlp server")
    parser_find_dep.add_argument("--corenlp_character_limit", type=int, default=10000, help="maximum number of characters sent in a single request")
    parser_find_dep.set_defaults(func=find_dep)

    args = parser.parse_args()
    args.func(args)

def fill_patterns(args):
    dp = DataProcessor()
    result, metadata = dp.fill_patterns(args.pattern_file,args.phrase_file,args.pattern_mask,args.num_sentences_per_pattern,args.pattern_text_name,args.phrase_text_name,args.output_text_name)
    result = dp.add_metadata(result,metadata)
    dp.write_file(result,args.output_file)

def find_vectors(args):
    dp = DataProcessor()
    result = dp.load_file(args.input_file)
    dp.load_words(args.vec_file)
    new_vector_file = None
    if not args.output_file is None:
        new_vector_file = args.output_file + ".vec"
    else:
        new_vector_file = args.input_file + ".vec"

    result = dp.find_vectors(result,args.input_text_name,args.vec_file,new_vector_file=new_vector_file)

    if args.augment_vec_file:
        dp.augment_word_vector_file(new_vector_file,args.vec_file,args.augment_max_word_count,args.augment_cutoff,args.augment_n)

    if not args.output_file is None:
        dp.write_file(result,args.output_file)
    else:
        dp.write_file(result,args.input_file)

def find_dep(args):
    dp = DataProcessor()
    result = dp.load_file(args.input_file)
    result = dp.find_dependencies_with_corenlp(result,args.input_text_name,args.corenlp_server,args.corenlp_port,character_limit=args.corenlp_character_limit)

    if not args.output_file is None:
        dp.write_file(result,args.output_file)
    else:
        dp.write_file(result,args.input_file)

def init_logging():
   #set up logging
   root_logger = logging.getLogger('root') 
   root_logger.setLevel(logging.DEBUG)

   #create console handler
   ch = logging.StreamHandler()
   ch.setLevel(logging.INFO)

   #create formatter
   formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   ch.setFormatter(formatter)

   root_logger.addHandler(ch) 

def main():
    init_logging()
    parse_args()

if __name__ == '__main__':
    main()
