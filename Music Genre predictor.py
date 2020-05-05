#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree


# In[2]:


music_data = pd.read_csv('music.csv')

X = music_data.drop(columns=['genre'])
y = music_data['genre']

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

tree.export_graphviz(model, out_file='genre-prediction.dot', feature_names=['age', 'gender'], 
                    class_names=sorted(y.unique()), label='all',
                    rounded=True, filled=True)


# In[3]:


predictions = model.predict(X_test)
accuracy = 100 * accuracy_score(y_test, predictions)
print(accuracy, "%")


# In[ ]:





# In[ ]:




