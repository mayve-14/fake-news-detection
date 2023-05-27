#!/usr/bin/env python
# coding: utf-8

# 1) Install and import nedded libraries:

# In[112]:


import numpy as np
import pandas as pd
import nltk
import re
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm 
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from  nltk.stem import PorterStemmer
import scikitplot.estimators as esti
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import time
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier


# 2) Read Data set 

# In[113]:


#Read Data set
Data = pd.read_csv("news.csv",encoding='latin-1')
Data = Data[['title' , 'text' , 'label']]
Data


# In[114]:


#my work 

##change fake and real to 0 and 1 
Data['label']=np.where(Data['label']=='FAKE',0,1)

##plotting real and fake news numbers "to make sure they are almost equal"
plt.figure(figsize=(16,9))
sns.countplot(Data.label)

##checking that there is no null values 
Data.isna().sum()
#Data['text'] = Data['title'] + ' ' + Data['text']

del Data['title']

  
###############################################################################

stemmer = PorterStemmer()
stemmer2 = SnowballStemmer(language='english')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
def Clean(text):

  # Frist converting all letters to lower case
  text= text.lower()
  
  # removing unwanted digits ,special chracters from the text
  text= ' '.join(re.sub("(@[A-Za-z0-9]+)", " ", text).split()) #tags
  text= ' '.join(re.sub("^@?(\w){1,15}$", " ", text).split())
   
  #text= ' '.join(re.sub("â", " ", text).split())
 # text= ' '.join(re.sub("(\w+:\/\/\S+)", " ", text).split())   #Links
 # text= ' '.join(re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"," ", text).split()) 
  #text= ' '.join(re.sub(r'http\S+', '',text).split())
  
  
  #text= ' '.join(re.sub(r'www\S+', '',text).split())
  #text= ' '.join(re.sub("\s+", " ",text).split()) #Extrem white Space
 # text= ' '.join(re.sub("[^-9A-Za-z ]", "" ,text).split()) #digits 
  text= ' '.join(re.sub('-', ' ', text).split()) 
  text= ' '.join(re.sub('_', ' ', text).split()) #underscore 
  
  #Display available PUNCTUATION for examples for c in string.punctuation:
  #print(f"[{c}]")
  
  # removing stopwards and numbers from STRING library
  table= str.maketrans('', '', string.punctuation+string.digits)
  text = text.translate(table)
  
  # Split Sentence as tokens words 
  tokens = word_tokenize(text)
  
  # converting words to their root forms by STEMMING THE WORDS 
  stemmed2 = [lemmatizer.lemmatize(word) for word in tokens] #Covert words to their actual root
  #stemmed2 = [stemmer2.stem(word) for word in tokens] # Covert words to their rootbut not actual
  
  # Delete each stop words from English stop words
  #words = [w for w in stemmed1 if not w in n_words] #n_words contains English stop words
  words = [w for w in stemmed2 if not w in stop_words] #n_words contains English stop words

  text  = ' '.join(words)
    
  return text
########################################################3

Data.text=[Clean(x) for x in Data.text]
Data.head(10)


# In[115]:


#data preparation

#Splitting data into train and validation
train_x, test_x, train_y, test_y = train_test_split(Data['text'], Data['label'] , shuffle = False)

# TFIDF feature generation for a maximum of 5000 features
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(Data['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xtest_tfidf =  tfidf_vect.transform(test_x)

xtrain_tfidf.data


# In[116]:


# model training 
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    #esti.plot_learning_curve(classifier, feature_vector_train, label)
    #plt.show()
    #plot_confusion_matrix(classifier,  feature_vector_train, label)  
    #plt.show()
    return metrics.accuracy_score(predictions, test_y)
   


# In[91]:


# Naive Bayes trainig
accuracy = train_model(naive_bayes.MultinomialNB(alpha=0.2), xtrain_tfidf, train_y, xtest_tfidf)
print ("Accuracy: ", accuracy)


# In[92]:


# Linear Classifier on Word Level TF IDF Vectors " logistic regression"
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xtest_tfidf)
print ("Accuracy: ", accuracy)


# In[93]:


# polynomial svm classifier 
accuracy = train_model(svm.SVC(kernel='poly') ,xtrain_tfidf, train_y, xtest_tfidf )
print ("Accuracy: ", accuracy)


# In[94]:


# linear svm classifier 
accuracy = train_model(svm.SVC(kernel='linear') ,xtrain_tfidf, train_y, xtest_tfidf )
print ("Accuracy: ", accuracy)


# In[95]:


# decision tree classifier
accuracy = train_model(DecisionTreeClassifier(random_state=200) ,xtrain_tfidf, train_y, xtest_tfidf )
print ("Accuracy: ", accuracy)


# In[117]:


def TFIDFModels(Model,txt):
    
  
    
    vect      = TfidfVectorizer(min_df = 5, max_df =0.8, sublinear_tf = True, use_idf = True)
    train_vect= vect.fit_transform(train_x)
    test_vect = vect.transform(test_x)
    
    model     = Model
    t0        = time.time()
    model.fit(train_vect, train_y)
    t1        = time.time()
    predicted = model.predict(test_vect)
    t2        = time.time()
    time_train= t1-t0
    time_pred = t2-t1
    
    accuracy  = model.score(train_vect, train_y)
    predicted = model.predict(test_vect)
    
    report = classification_report(test_y, predicted, output_dict=True)
    
    print(txt)
    print("Training time: %fs; Prediction time: %fs \n" % (time_train, time_pred))
    print('Accuracy score train set :', accuracy)
    print('Accuracy score test set  :', accuracy_score(test_y, predicted),'\n')
    
    print('\n -------------------------------------------------------------------------------------- \n')


# In[118]:



SupportVectorClassifier=svm.SVC(kernel='linear')
print('Models with Tfidf Feature extraction Techniques : \n')
print('************************************************ \n')

#LogReg=TFIDFModels(Model=LogisticRegression(),txt='Logistic Regression Model : \n ')
svm=TFIDFModels(Model=SupportVectorClassifier,txt='Support Vector Classifier Model : \n ')
#DecTree=TFIDFModels(Model=tree.DecisionTreeClassifier(),txt='Decision Tree Classifier Model : \n ')
#knn_tfidf=KNN_TFIDF()


# In[ ]:





# In[ ]:




