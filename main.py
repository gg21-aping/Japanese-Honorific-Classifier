#get data
import re

from google.colab import auth
auth.authenticate_user()

import gspread
from oauth2client.client import GoogleCredentials

gc = gspread.authorize(GoogleCredentials.get_application_default())

corpus = []

data = gc.open('train_data').sheet1.get_all_values()
data = [item for sublist in data for item in sublist]

for line in data:
  line = line.replace("。","").replace("、","")
  line = re.sub(r"\d+", "", line)
  corpus.append(line)

#informal = 0, polite = 1, formal = 2
def label_int(string_label):
  if string_label == "informal":
    return 0
  elif string_label == "polite":
    return 1
  else: 
    return 2

label = gc.open('train_label').sheet1.get_all_values()
label = [item for sublist in label for item in sublist]

y = list(map(lambda x: label_int(x), label))

#Tokenizer and Vectorizer
!pip install nagisa
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nagisa
import pandas as pd

def tokenizer_jp(text):
  words = nagisa.filter(text, filter_postags=['助詞'])
  return words.words

# stop_words = ['な', 'と', 'た', 'で', 'は']

vectorizer = TfidfVectorizer(tokenizer=tokenizer_jp)

corpus_tfidf = vectorizer.fit_transform(corpus)

frame = pd.DataFrame(corpus_tfidf.toarray(), columns= vectorizer.get_feature_names())
print(frame)

X_train, X_test, y_train, y_test = train_test_split(corpus_tfidf, y, test_size = 0.2, random_state = 34)

import pickle

pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

#Logistic Regression
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver = 'liblinear', penalty = 'l2', C =10)

model.fit(X_train, y_train)

#SGD classifier
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, alpha=0.0003, n_iter_no_change=10)

model.fit(X_train, y_train)

#Naive Bayes
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train, y_train)

#Support Vector Machine (SVM)
from sklearn.svm import SVC

model = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

model.fit(X_train, y_train)

#Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

cv = KFold(n_splits=10, random_state=1, shuffle=True)
#model = 
scores = cross_val_score(model, corpus_tfidf, y, scoring='accuracy', cv=cv, n_jobs=-1)

#Validation
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)

print(accuracy)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


import pickle

pickle.dump(model, open('model.pkl', 'wb'))
