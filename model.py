# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:05:42 2022

MODEL TYPE1 : LinearSVC(penalty='l2', loss='squared_hinge', *,
                        dual=True, tol=0.0001, C=1.0, 
                        multi_class='ovr', fit_intercept=True, 
                        intercept_scaling=1, 
                        class_weight=None, verbose=0, random_state=None, max_iter=1000)


MODEL TYPE1 : MultinomialNB(*, alpha=1.0, fit_prior=True, class_prior=None)
AccuRACY : 0.9793365959760739

Dataset : https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

@author: manish
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import pandas as pd 
import pickle


df= pd.read_csv("spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Features and Labels
df['label'] = df['class'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']
	
# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data




pickle.dump(cv, open('tranformations.pkl', 'wb'))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#Naive Bayes Classifier
#clf = MultinomialNB()



clf = LinearSVC()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
filename = 'sms_model.pkl'
pickle.dump(clf, open(filename, 'wb'))

'''
	Alternative Usage of Saved Model
	joblib.dump(clf, 'NB_spam_model.pkl')
	NB_spam_model = open('NB_spam_model.pkl','rb')
	clf = joblib.load(NB_spam_model)
'''