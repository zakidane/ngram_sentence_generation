
# CSCI 3832, Spring 2021, CU Boulder

from collections import Counter
from collections import defaultdict
import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
import sys
import math


class LanguageModel:

    def __init__(self, ngram_order, is_laplace_smoothing):
        
        #initialise the variables
        self.ngram_order = ngram_order
        self.is_laplace_smoothing = is_laplace_smoothing
        self.token_freq = {} #this would store the unique words and their frequencies
        self.ngram_dict = {} #this stores unique ngrams and their frequencies
        self.ngram_minusone_dict = {} # for calculating the denominator count(the boy) for P(the |the boy) for instance
        self.n_sentences = 0 # this will store the total number of sentences in a file
        self.bad_vocab = {} #this is the list of vocabulary with frequency > 1
        

      #the train function maps tokens to their frequencies and ngrams to their frequencies
        #and outputs text files with the probabilities corresponding to each sentence
    def train(self,filepath):
        # first we empty the appropriate dictionaries so that training ngram counts from previous training files
        #don't overlap with training from the current file
        self.ngram_dict.clear()
        self.ngram_minusone_dict.clear()
        self.token_freq.clear()
        self.bad_vocab.clear()
        #we then declare tokens and ngrams as well as n-1 gram lists 
        tokens = []
        ngrams = []
        n_1grams = []
        # indicating the name of the output file based on n-gram
        with open(filepath) as lines:
            for line in lines:
                token = line.split()
                tokens.extend(token)
        
        freqs= Counter()
        freqs.update(tokens)
        self.token_freq = dict(freqs)
        
        
        #replace frequency = 1 words with <unk> in token_freq dictionary and place words in bad_vocab
        for key, value in self.token_freq.items():
            if value == 1: 
                self.bad_vocab[key] = value
                key = "<unk>"
                
                
        #create ngram lists
        ngrams = self.create_ngram(self.ngram_order,tokens)
        
        #create dictionaries with ngram tuples and their counts
        freqs2= Counter()
        freqs2.update(ngrams)
        self.ngram_dict = dict(freqs2)
        
        #create n-1 gram lists and get their counts if ngram>=2
        
        
        n_1grams = self.create_ngram(self.ngram_order-1, tokens)

        #create dictionaries with ngram-1 tuples and their counts
        freqs3= Counter()
        freqs3.update(n_1grams)
        self.ngram_minusone_dict = dict(freqs3)
        
        print('num of unique {}-grams :={}'.format(self.ngram_order, len(self.ngram_dict) ))#todo: put the proper variables
        return ngrams
    
    #helper function for train to create ngrams
    def create_ngram(self, n, tokens):
        ngrams = []
        for i in range(n-1,len(tokens)):
            ngram = tokens[i-n+1:i+1]
            ngrams.append(tuple(ngram))
        return ngrams
        
        
        #then use counter to create the dictionary with keys = (w1,w2) and values = frequency of (w1,w2)
        
    #for a given test sentence, this calculates P(sentence) = πP(wi|w i-n+1)
    def score(self, sentence):
        #tokenize sentence
        tokens = []
        log_p_list = []
        len_vocab = len(self.token_freq) # this is |V|
        tokens = sentence.lower().split()
 
        for i in range(len(tokens)): 
            #account for token with frequency 1 or those that don't exist in training data
            if tokens[i] in self.bad_vocab:
                token = "<unk>"
            if tokens[i] not in self.token_freq: 
                tokens[i] = "<unk>"
        #create ngrams get their frequencies stored in n_count and n-1 count       
        ngrams = self.create_ngram(self.ngram_order, tokens)
        for ngram in ngrams: 
            
            n_count = self.ngram_dict.get(ngram, 0)
            n_1count = self.ngram_minusone_dict.get(tuple(list(ngram)[:-1]), 0)
            
            #add to log_p_list based on whether is_laplace_smoothing is true
            if self.is_laplace_smoothing: 
                log_p_list.append(math.log((n_count+1)/(n_1count+len_vocab)))
            else: 
                log_p_list.append(math.log(n_count/n_1count))
                
     
        return sum(log_p_list)
        
        #returns perplexity of the file
        #formula is e^((-1/N)* sum of log(P_i)) for all i which is the same as Perplexity = P(w1, w2...)^(-1/N)
    def getPerplexity(self, filename):
        
        prob_file = 1
        sum_logs = 0
        N = len(self.token_freq)
        
        with open(filename) as lines:
            for sentence in lines:
                  sum_logs = sum_logs + self.score(sentence)
            perplexity = math.exp((-1/N)*sum_logs)
        
        

        print('Perplexity using {}-grams :={}'.format(self.ngram_order, perplexity))#todo: put the proper variables
        return perplexity
    
#this function generates random sentences by looping through range(Num of sentences)
    def generate(self,num_sentences):
        randomsentences=[]
        for i in range(num_sentences): 
            prev = "<s>"
            sentence = [prev]
            for i in range(50): 
                ngrams = []
                ngram_probs = []
                if prev == "</s>": # to ensure we don't continue past end of sentence tag
                    break
                for k, v in self.ngram_dict.items(): #get list of ngrams and their probabiilities
                    if k[0] == prev: 
                        ngrams.append(k)
                        ngram_probs.append(v)
                        
             #this picks random index from list of indices
                random_index = np.random.choice(len(ngrams), 1, ngram_probs)[0]
              
                #the following add the tokens in an ngram to a sentence but first remove the prev
                list_words = list(ngrams[random_index])
                list_words.pop(0)
                sentence.extend(list_words)
                prev = sentence[-1]
            #sentence added to random sentences
            randomsentences.append(' '.join(sentence))
        return randomsentences

#this function will generate unigram probabilities from test file and output to hw2-unigram-out.txt
#Note that it calculates log(P(sentence)) and not P(sentence)
def unigramOut(filepath):
    model = LanguageModel(1, True)
    output_file = "hw2-unigram-out.txt"
    model.train(filepath)
    input_f = open(filepath, "r")
    output_f = open(output_file, "w+")
    
    for line in input_f: 
        score = model.score(line)
        output_f.write(str(score)+'\n')
    input_f.close()
    output_f.close()
        
        #similarly, the following function generates hw2-bigram-out.txt
def bigramOut(filepath):
    model = LanguageModel(2, True)
    output_file = "hw2-bigram-out.txt"
    model.train(filepath)
    input_f = open(filepath, "r")
    output_f = open(output_file, "w+")
    
    for line in input_f: 
        score = model.score(line)
        output_f.write(str(score)+'\n')
    input_f.close()
    output_f.close()
        
 #the following file generates a 100 sentences from unigram model, one per line
def unigramGenerate(filepath):
    model = LanguageModel(1, True)
    output_file = "hw2-unigram-generated.txt"
    model.train(filepath)
    input_f = open(filepath, "r")
    output_f = open(output_file, "w+")
    
    randomsentences = model.generate(100)
    for sentence in randomsentences: 
        output_f.write(sentence+'\n')
    output_f.close()
    
#similarly this generates a 100 sentences for bigram model, one per line    
def bigramGenerate(filepath):
    model = LanguageModel(2, True)
    output_file = "hw2-bigram-generated.txt"
    model.train(filepath)
    input_f = open(filepath, "r")
    output_f = open(output_file, "w+")
    
    randomsentences = model.generate(100)
    for sentence in randomsentences: 
        output_f.write(sentence+'\n')
    output_f.close()

    #this plots a histogram with words (unigrams) and their frequencies
def plotHistogram(filepath):
    tokens = []
    with open(filepath, "r") as lines:
        for l in lines: 
            token = l.split()
            tokens.extend(token)
        lines.close()
    word_counts = Counter(tokens)
    df = pd.DataFrame.from_dict(word_counts, orient='index')
    df.plot(kind='hist')
    plt.show()
        
        

if __name__ == '__main__':

    # ADDED
    if len(sys.argv) != 3:
        print("Usage:", "python hw2_lm.py berp-training.txt hw2-test.txt ")
        sys.exit(1)

    trainingFilePath = sys.argv[1]
    testFilePath = sys.argv[2]
    
    #the instructions below train a bigram model, generate 3 sentences and  get perplexity
    model = LanguageModel(1, True)
    model.train(trainingFilePath)                            
    model.generate(10)
    model.getPerplexity(testFilePath)
    
    #the 1st two  lines generate probabilities (log(P)'s to prevent underflow) for unigrams and bigrams
    #the last two lines generate two files with 100 random sentences each, one for unigram, another for bigram
    unigramOut(trainingFilePath)
    bigramOut(trainingFilePath)
    unigramGenerate(trainingFilePath)
    bigramGenerate(trainingFilePath)
    #bonus plotting histogram
    plotHistogram(trainingFilePath)