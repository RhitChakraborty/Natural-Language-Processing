import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)  #delimiter tab is used
#print(dataset.head())

#Cleaning of Text
import re
import nltk
#nltk.download('stopwords')   #downloaded all stopwords
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])   #keeping the letters fro a-z
    review=review.lower().split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

print(corpus)
#Creating a BAG Of WORDS Model   ### Sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1050)
x=cv.fit_transform(corpus).toarray()

y= dataset.iloc[:,1].values

#splitting test train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import f1_score,precision_score,recall_score
print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(f1_score(y_test,y_pred))