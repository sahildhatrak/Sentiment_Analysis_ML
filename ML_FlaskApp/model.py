import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import math
import warnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import WhitespaceTokenizer
lemmatizer = WordNetLemmatizer()
w_tokenizer = WhitespaceTokenizer()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, plot_confusion_matrix
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report
import colored 
import pickle






data_file = 'reviews_final.csv'
data = pd.read_csv(data_file)



# Making Stopwords List

stopwords_eng = stopwords.words('english')
stopwords_eng2 = stopwords_eng
stopwords_eng2 = [x.capitalize() for x in stopwords_eng2]
stopwords_final = stopwords_eng + stopwords_eng2


# # Building a Pipeline

class ApplyRegex(BaseEstimator, TransformerMixin):
    
    def __init__(self, break_line=True, carriage_return=True, numbers=True, number_replacing='', 
                 special_char=True, additional_spaces=True):
        self.break_line = break_line
        self.carriage_return = carriage_return
        self.numbers = numbers
        self.number_replacing = number_replacing
        self.special_char = special_char
        self.additional_spaces = additional_spaces
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = []
        for c in X:
            if self.break_line:
                c = re.sub('\n', ' ', c)
            if self.carriage_return:
                c = re.sub('\r', ' ', c)
            if self.numbers:
                c = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', self.number_replacing , c)
            if self.special_char:
                c = re.sub(r'R\$', ' ', c)
                c = re.sub(r'\W', ' ', c)
            if self.additional_spaces:
                c = re.sub(r'\s+', ' ', c)
            X_transformed.append(c)
        return X_transformed
    


# In[27]:


class StopWordsRemoval(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def stopword_removal(self):
        y=[]
        review_no_stopword = []
        for idx, review in enumerate(self) :
            try:
                y=''
                for word in review.split():
                    if word not in stopwords_final:
                        y+= word + ' '
                review_no_stopword.append(y)
            except:
                print(idx)
        return review_no_stopword
        
    def transform(self, X, y=None): 
        X_transformed = StopWordsRemoval.stopword_removal(X)
        return X_transformed





class TextLemmatization(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def lemmatize_text(text):
        return ' '.join(lemmatizer.lemmatize(w, pos="v") for w in w_tokenizer.tokenize(text))


    def transform(self, X, y=None):
        X_transformed = list(map(lambda c: TextLemmatization.lemmatize_text(c), X))
        return X_transformed





preprocess_pipeline = Pipeline([
    ('regex_cleaner', ApplyRegex()),
    ('stopwords_remover', StopWordsRemoval()),
    ('lemmatization', TextLemmatization()),
])


# # Applying a Pipeline




X = data['reviews.text']
y = data['reviews.rating'].values
y = y.astype(int)

X_preprocessed = preprocess_pipeline.fit_transform(X)
reviews_vector = list(map(lambda c: nltk.word_tokenize(c), X_preprocessed))
vectorizer = CountVectorizer(max_features=300)
X_transformed = vectorizer.fit_transform(X_preprocessed).toarray()

X_transformed[0]





# # Labelling and Splitting of Data


bin_edges = [0, 2, 3, 5]
bin_names = ['Negative', 'Neutral', 'Positive']
data['class'] = pd.cut(data['reviews.rating'], bins=bin_edges, labels=bin_names)


# In[33]:



X_train, X_test, y_train, y_test = train_test_split(X_transformed, data['class'] , test_size=.20, random_state=42)


# # Training and Evaluating the Model

# In[34]:





# In[63]:


log_reg = LogisticRegression(max_iter = 200, multi_class = 'auto', solver = 'newton-cg', verbose=1)
log_reg.fit(X_train, y_train)
predictions = log_reg.predict(X_test)


# In[64]:


# Evaluating results

print(classification_report(y_test,predictions))


# # Class Prediction

# In[352]:


example = ['Bought in 2017. It is a bad product. The battery life sucks.' ]
example_preprocessed = preprocess_pipeline.fit_transform(example)

# Vectorizing
example_transformed = vectorizer.transform(example_preprocessed).toarray()

# Communicating
example_pred = log_reg.predict(example_transformed)


# In[377]:

print(example)
print(example_pred)




# # Sentiment Score Predictor

# In[442]:
print('\n')
def SentimentScorePredictor(text):
    text_preprocessed = preprocess_pipeline.fit_transform(text)
    text_transformed = vectorizer.transform(text_preprocessed).toarray()
    review_proba = log_reg.predict_proba(text_transformed)
    sentiment_score = round(review_proba[0,2]*100,2)
    sentiment_score = sentiment_score.item()
    print(text)
    if(sentiment_score <= 30):
        sentiment_score = str(sentiment_score)
        print(colored.fg("red") + sentiment_score)
    elif(30 < sentiment_score < 70):
        sentiment_score = str(sentiment_score)
        print(colored.fg("yellow") + sentiment_score)
    elif(70 <= sentiment_score <= 100):
        sentiment_score = str(sentiment_score)
        print(colored.fg("green") + sentiment_score)



SentimentScorePredictor(['Bought in 2017. It is a bad product.'])




pickle.dump(log_reg, open('model.pkl', 'wb'))

