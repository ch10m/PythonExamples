# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 15:01:55 2018

@author: ledmans
"""
    



#packages to be imported
import pandas as pd
from sklearn.externals import joblib
import re
import string
string.punctuation
import nltk
stopwords = nltk.corpus.stopwords.words('english')
stopwords.append("")
ps = nltk.PorterStemmer()
lm = nltk.WordNetLemmatizer()
import nltk.classify.util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from nltk import ngrams
import numpy as np

##install NLTK###

import pip


try:
    __import__("nltk")
except ImportError:
    pip.main(['install', "nltk"])   


  
#function to read in and prep any data needed  
def input(filename):
    #read in file to categorise
    data= pd.read_csv(filename, sep=',')
    data['Combined']=data['Title']+" "+data['Content']
    #csv file with all the features needed for the categorisation models 
    #puts into a data frame for use at a later stage
    features = pd.read_csv("final features.csv", sep=',')
    for word in features['words']:
        #print(word)
        features[word]=0
    features.pop(features.columns[0])
    features.pop(features.columns[0])
    features=features[0:0]
    for row in  range(0,len(features)):
        features.append(pd.Series([np.nan]), ignore_index = True)
        #for col in features:
            #features.loc[row,col]= 0
    #features.to_csv("empty features.csv")
    #print(features)
    return data,features

#initial clean removes punctuation and sets all text to lower case
def initial_clean(text):
    #remove punctuation
    ###my attempt at adding removing punct that has no space
    text = re.sub("(\')",'',text)
    text =re.sub(r"(\W)",r' ',text)
    text=re.sub(r" +",r' ',text)
     
    #######################################
    text = re.sub("\/"," ",text)
    nopunct = "".join([char for char in text if char not in string.punctuation])
    return nopunct.lower()

#stems words, applies the spelling corrector and removes any stopwords
def clean_data_stem(text):
    #split review into words tokenized
    tokens = re.split('\W+',text) 
    text = [ ps.stem(correction(word)) for word in tokens if word not in stopwords]
    return text

def data_for_ngram(text):
    new_text=""
    for word in text.split():
        if word not in stopwords:
            new_text=new_text+" "+ps.stem(correction(word))
    
    return new_text

##spelling corrector code
def words(text): return re.findall(r'\w+', text.lower())
#reads in the initial list of all reviews as a text to check against
WORDS = Counter(words(open('txt reviews all.txt').read()))
print(WORDS)

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)
    #trying to match with real words only 
    #return set(w for w in words if wn.synsets(w) != [])

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def ngram_count(words, text):
     
    count = 0
    for word in text:  
        if word == words:
            count +=1 
    return count
#
#def ngram_count1(words, text):
#     
#    count = 0
#    for word in text:  
#        if word == words:
#            count +=1 
#    return count

def ngram_list(text):
    #print(text)
    list_ngram=[]
    bigrams=ngrams(text.split(),2)
    for gram in bigrams:
        newgram=""
        for word in gram:
            newgram=newgram+word+" "
        list_ngram.append(newgram[0:len(newgram)-1])
    trigrams=ngrams(text.split(),3)
    for gram in trigrams:
        newgram=""
        for word in gram:
            newgram=newgram+word+" "
        list_ngram.append(newgram[0:len(newgram)-1])
   #print(list_ngram)
    return list_ngram

#creates feature matrix for the set of reveiws to analyse
def feature_selection(data,features):#,phrase,ngram):
    #gets the length of a review for the initial clean 
    features['Review Length']=data['Combined'].apply(lambda x: len(x))
    data['No num'] = data['Combined'].apply(lambda x: re.sub("(\d|£)","",x))
   
    #does the initial clean ready to find any double bonded words
    data['Initial Clean']=data['No num'].apply(lambda x: initial_clean(x))
    data['Stem']=data['Initial Clean'].apply(lambda x: clean_data_stem(x))
    data['ngram']=data['Initial Clean'].apply(lambda x: data_for_ngram(x))
    data['grams']=data['ngram'].apply(lambda x: ngram_list(x))
    for words in features:

        if words!= "Review Length":
            if " " in words:
                features[words]=data['grams'].apply(lambda x: ngram_count(words,x))
            else:
                features[words]=data['Stem'].apply(lambda x: ngram_count(words,x))
                
    return data,features


 
def main(file_to_analyse):
    data,features=input(file_to_analyse)
    #first return the datafrmae of steming and lemming
    data,features=feature_selection(data,features)
    final_results = pd.DataFrame(columns=['Title','Content',
                                          'Level 1: People','Level 1: Product','Level 1: Process',
                                          'Level 2: People','Level 2: Product','Level 2: Process',
                                          'Sentiment'])
    final_results['Title']=data['Title']
    final_results['Content']=data['Content']
    
##    #Level 1 models
    rf_model=joblib.load('Level 1 People6.pkl')
    pred = rf_model.predict(features)
    final_results['Level 1: People']=pred
#    
    rf_model=joblib.load('Level 1 Product8.pkl')
    pred = rf_model.predict(features)
    final_results['Level 1: Product']=pred
    
    rf_model=joblib.load('Level 1 Process6.pkl')
    pred = rf_model.predict(features)
    final_results['Level 1: Process']=pred
    
    #selects reviews which have been categorised as level 1 people to put through the level 2 model
    #then runs through level 2 model 
    input_people = []
    h=0
    for k in final_results['Level 1: People'] :
        if k=='People':
            input_people.append(h)
        h+=1
    if input_people != []:
        rf_model=joblib.load('Level 2 People87.pkl')
        pred = rf_model.predict(features.iloc[input_people])
        for row,i in zip(input_people,range(0,len(pred))):
            final_results.loc[row,'Level 2: People']=pred[i]

    
    #selects reviews which have been categorised as level 1 product to put through the level 2 model
    #then runs through level 2 model 
    input_product = []
    h=0
    for k in final_results['Level 1: Product'] :
        if k=='Product':
            input_product.append(h)
        h+=1
    if input_product != []:
        rf_model=joblib.load('Level 2 Product111.pkl')
        pred = rf_model.predict(features.iloc[input_product])
        for row,i in zip(input_product,range(0,len(pred))):
            final_results.loc[row,'Level 2: Product']=pred[i]
    
    #selects reviews which have been categorised as level 1 process to put through the level 2 model 
    #then runs through level 2 model 
    input_process = []
    h=0
    for k in final_results['Level 1: Process'] :
        if k=='Process':
            input_process.append(h)
        h+=1
    if input_process != []:
        rf_model=joblib.load('Level 2 Process4.pkl')
        pred = rf_model.predict(features.iloc[input_process])
        for row,i in zip(input_process,range(0,len(pred))):
            final_results.loc[row,'Level 2: Process']=pred[i]
#        
    #sentiment model applied to all reviews
    analyzer = SentimentIntensityAnalyzer()
    for sentence, row in zip(data['Combined'],range(0,len(data['Combined']))):
        #sentence = " ".join(sentence)
        vs = analyzer.polarity_scores(sentence)
        final_results.loc[row,'Sentiment']=vs.get('compound')
      
    #output of results to csv file
    #features.to_csv("FEATURES testing 3.csv")
    final_results.to_csv("Final Results Test Data SM.csv", sep=',')
    return final_results,data,features

final_results,data,features=main('Test data SM - discovery.csv')