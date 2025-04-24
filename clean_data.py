import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(min_reviews, max_reviews):
    d = pd.read_csv('games.csv')
    
    # remove non-english games
    # remove unnecessary columns
    d = d.loc[~d['Genres'].isna(), ]
    d = d.loc[d['Genres'].str.contains('Indie'), ]
    d = d.loc[d['Supported languages'].str.contains('English'), ]

    cols = ['AppID', 'Name', 'Release date', 'Price', 'DLC count', 
     'About the game', 'Metacritic score', 'Positive', 'Negative',
     'Achievements', 'Recommendations', 'Average playtime forever', 'Median playtime forever', 'Categories',
     'Genres', 'Tags']

    d = d[cols]

    # create num reviews 
    d['Num Reviews'] = d['Positive'] + d['Negative']
    if max_reviews is None:
        d = d.loc[(d['Num Reviews'] >= min_reviews), ]
    else:
        d = d.loc[(d['Num Reviews'] >= min_reviews) & (d['Num Reviews'] <= max_reviews), ]

    d['Positive Ratio'] = d['Positive']/d['Num Reviews']
    d['Recommendations Ratio'] = d['Recommendations']/d['Num Reviews']

    d = d.reset_index(drop = True)
    
    # create review class
    d['Review Class'] = np.where(d['Positive Ratio'] >= 0.7, 1, 0)
    
    # dummy cols for genres
    genres = list(set((np.sum(d['Genres'] + ',', axis = 0)).split(',')))
    genres.remove('')
    genres.remove('Indie')
    genres_df = pd.DataFrame(np.zeros((len(d), len(genres))), columns = genres).astype(int)
    d = d.join(genres_df)

    for genre in genres:
        genre_list = np.zeros(len(d))
        for i in range(len(d)):
            if genre in d.iloc[i, 14]:
                genre_list[i] = 1
        d[genre] = genre_list

    return d

def preprocess(text, stop_words, engl_words, lemmatizer):
    cleaned_text = re.sub('^\s+|\W+|[0-9]|\s+|_$', ' ', str(text)).strip()
    cleaned_text = cleaned_text.strip().replace('\n', ' ')
    cleaned_text = cleaned_text.strip().replace('_', ' ')
    cleaned_text = cleaned_text.strip().replace('_', ' ')
    cleaned_text = cleaned_text.lower()

    
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(cleaned_text)
    word_tokens = [lemmatizer.lemmatize(w) for w in word_tokens if (w in engl_words or not w.isalpha()) and (not w.lower() in stop_words)]

    cleaned_text = ''
    for i in range(len(word_tokens)):
        if i != len(word_tokens) - 1:
            cleaned_text += word_tokens[i] + ' '
        else:
            cleaned_text += word_tokens[i]

    return cleaned_text

def clean_text(df):
    stop_words = set(stopwords.words("english"))
    engl_words = set(nltk.corpus.words.words())
    lemmatizer = WordNetLemmatizer()

    text_list = list(df['About the game'])
    text_list_clean = [preprocess(t, stop_words, engl_words, lemmatizer) for t in text_list]

    return text_list_clean

def tfidf(text, ngram, min_df):
    tfidf = TfidfVectorizer(ngram_range = ngram, min_df = min_df)
    text_tfidf = tfidf.fit_transform(text)
    text_tfidf_df = pd.DataFrame(text_tfidf.toarray().astype('float16'), columns=tfidf.get_feature_names_out())
    
    return text_tfidf_df
    
    
def top_words(d):
    d2 = pd.melt(d, id_vars = 'AppID', value_vars = list(d.columns[47:]), 
        var_name = 'word', value_name = 'tfidf')

    d2 = d2.merge(d[['AppID', 'Review Class']], on = ['AppID'])

    tops = d2.groupby(['Review Class', 'word'])['tfidf'].mean().reset_index()
    
    print(tops.loc[tops['Review Class'] == 0, ].sort_values(by = 'tfidf', ascending = False).iloc[:5, :])
    
    print(tops.loc[tops['Review Class'] == 1, ].sort_values(by = 'tfidf', ascending = False).iloc[:5, :])