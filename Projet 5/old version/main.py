#!/usr/bin/env python
# coding: utf-8

###############################################################################################################################


# pip install FastAPI
# pip install mylib
# pip install html5lib
# pip install uvicorn
# pip install requests
# pip install Flask


###############################################################################################################################

import re
import pickle

import pandas as pd
import numpy as np

# Pour supprimer les warnings :
import warnings
warnings.filterwarnings("ignore")

# pour le modelling des mots :
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Beautiful Soup :
import lxml
import html5lib
from bs4 import BeautifulSoup

# Pour supprimer les warnings :
import warnings
warnings.filterwarnings("ignore")

# Pour le BOW :
from nltk.tokenize import word_tokenize

# pour les algorithmes supervisés :
from sklearn.linear_model import LogisticRegression

# Pour la LDA : 
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Pour la visualisation des tokens :
from sklearn.feature_extraction.text import CountVectorizer

# nettoyage du texte :
import string  # permet d'avoir accés à toute les ponctuations.

from fastapi import FastAPI, Request, Header
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from pydantic import BaseModel
import uvicorn


###############################################################################################################################


app = FastAPI(title='Prediction des Tags pour StackOverFlow :',
              description="Permet de visualiser les tags d'un post :",
              version='0.0.1')


###############################################################################################################################


CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}


###############################################################################################################################


                ##############################################################################
                ######################## Fonction de nettoyage : #############################
                ##############################################################################
    

def clean_html(text_html):

    soup = BeautifulSoup(text_html, "html5lib")

    for element in soup.find_all("code"):
        element.decompose()

    return soup.get_text().replace("\n", " ")


                ##############################################################################
    

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                   flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)            if contraction_mapping.get(match)            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


                ##############################################################################

    
def text_cleaning(text):

    text = re.sub('\w*\d\w*', '', text)  # supprimer tout les chiffres
    text = re.sub(r'\n', '', text)  # retirer les fins de lignes
    text = re.sub(r'\s+', ' ', text)  # retirer les fins de lignes de corpus

    return text


                ##############################################################################

    
def tokenize(text):
    
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    try:
        res = word_tokenize(text, language='english')
    except TypeError:
        return text

    res = [token for token in res if token not in punctuation]
    res = [token for token in res if token not in stop_words]

    return res


                ##############################################################################

    
def filtering_nouns(text):

    res = nltk.pos_tag(text)

    res = [token[0]
           for token in res if token[1] == 'NN']  # Rajouter adverbe etc etc

    return res


                ##############################################################################

    
def lemmatisation(text):

    # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # res = [lemmatizer.lemmatize(word, wordnet.VERB) for word in text]
    # res = [lemmatizer.lemmatize(word, wordnet.ADJ) for word in res]
    # res = [lemmatizer.lemmatize(word, wordnet.NOUN) for word in res]
    res = [lemmatizer.lemmatize(word) for word in text]

    return res


                ##############################################################################

    
def bag_of_words(texts):
    
    data = texts 
    bow = bow_pickle.transform(data)
    
    return bow


###############################################################################################################################


                ##############################################################################
                ################################# Pickles : ##################################
                ##############################################################################


# Matrice Bow :
bow_pickle = pickle.load(
    open('cv.pkl', 'rb'))

# Supervisé model :
bow_reg_log = pickle.load(
    open('model_log.pkl', 'rb'))

# Multilabel :
multilabel = pickle.load(
    open('multi_lab.pkl', 'rb'))


                 ##############################################################################
                 ############################ Lancement de l'API : ############################
                 ##############################################################################
                
                
@app.get("/")
def root():
    return {"API de détection des Tags StackOverflow :"}




                 ##############################################################################        
                 ############################ Pour les input : ################################
                 ##############################################################################            
            

class Input(BaseModel):
    text : str         



                 ##############################################################################
                 ########################### Pour les requêtes : ##############################
                 ##############################################################################

"""
@app.get("/corpus/")
async def request_tags(corpus: str):
    results_list = []

    for index, row in corpus.iterrows():
    
        raw_text = row['Title'] + ' ' + row['Body']
        # Ensuite tu appel la fonction uniquement sur le text brut
        results = test(raw_text) # appel de l'API => modifier par une request avec body raw_text
        # Enfin tu sauves le resultat dans une liste
        results_list.append(results)
    
    return results_list
"""

                 ##############################################################################        
                 ############################ Pour la prédiction : ############################
                 ##############################################################################  
            
            
@app.post("/predict")
async def predictions(data: Input):
    

                 ##############################################################################
                 ############################### Nettoyage : ##################################
                 ##############################################################################
                     
# get data
    
    input_data = dict(data)
    
    text = input_data['text']
    
    clean = clean_html(text)
    
    print('CLEAN', clean)
    
    text_clean = text_cleaning(clean)
    #data['corpus'] = data['corpus'].apply(lambda x : text_cleaning(x))
    
    print('TEXTE CLEAN',text_clean)
    
    # text_contracted = expand_contractions(text_clean)
    #data['corpus'] = data['corpus'].apply(lambda x : expand_contractions(x))
    
    # print(text_contracted)
    
    text_tokenized = tokenize(text_clean)

    print('TEXT TOKENIZED', text_tokenized)
    
   # text_filtering_nouns = filtering_nouns(text_tokenized)

    #print(text_filtering_nouns)
    
    # text_lemmatized = lemmatisation(text_tokenized)

   #  print('TEXTE LEMMATIZED', text_lemmatized)
    

                  ##############################################################################
                  ############################### Prédiction : #################################
                  ##############################################################################
                
    
    bow_corpus = bag_of_words(text_tokenized)
    
    # print(bow_corpus)
    
    supervised_pred = bow_reg_log.predict(bow_corpus.toarray())

    
    # print(supervised_pred)
    
    
                  ##############################################################################
                  ######################### Visualisation des Tags : ###########################
                  ##############################################################################
    
    
    tags = multilabel.inverse_transform(supervised_pred)
    
    tags = [list(item)[0] for item in tags if len(item) != 0 ]
    
    if len(tags) == 0 :
        
        prediction = bow_reg_log.predict_proba(bow_corpus.toarray())
        df_proba = pd.DataFrame(supervised_pred, columns=list(multilabel.classes_))
    
        top_tags = []
        for i, row in df_proba.iterrows() :
            top = row.nlargest(2).index
            
        top_tags.append(','.join(top))
        
        tags = top_tags
    
    
    print(tags)

                  ##############################################################################
                  ############################## Encode en json : ##############################
                  ##############################################################################


    ## retoune la requête en json : ##
    return  JSONResponse(status_code=200, content={"Tags_supervisé" : tags })

    
                  ##############################################################################
                  ############################ Non supervisé : #################################
                  ##############################################################################
                
# Abandonné car non pertinent
"""    
    corpus_final = gensim_dictionary.doc2bow(text_lemmatized)
    
    # print(corpus_final)

    lda_predict = lda_model[corpus_final][0]
    
    # print(lda_predict)
    
    topic = np.argmax([x[1] for x in lda_predict]) 
    
    numero_de_topic = {0 : "Topic : Function/Class/Method",
   1 : "Topic : Model/List/Time",
   2 : "Topic : Service/Documentation/Input",
   3 : "Topic : Programmation_Web",
   4 : "Topic : Database/API/Server",
   5 : "Topic : Request/Application/Client",
   6 : "Topic : Programmation_Python",
   7 : "Topic : Array/Template/Header"}
    
 #   return numero_de_topic[topic]
    
    print(numero_de_topic[topic])
"""

###############################################################################################################################


if __name__ == "__main__":
   app.run()

"""
{
  "text": "Android Studio php connection I'm trying to get information from db using php in android studio java. The php code works 100%. The problems is that when I open the app from the Android emulator the app closes itself almost instantly."
}
"""