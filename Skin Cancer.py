#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# In[2]:


pip install matplotlib


# In[3]:


from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report


# In[4]:


X_train = np.load(r'C:\Users\lenovo\Downloads\x_train.npy')
Y_train= np.load(r'C:\Users\lenovo\Downloads\y_train.npy')
X_test=np.load( r'C:\Users\lenovo\Downloads\x_test.npy')
Y_test=np.load(r'C:\Users\lenovo\Downloads\y_test.npy')


# In[5]:


print(X_train.shape, X_test.shape , Y_train.shape, Y_test.shape)


# In[6]:


fig, axarr=plt.subplots(2,2, figsize=(15,20))
axarr[0,0].imshow(X_train[0],interpolation='nearest')
axarr[0,1].imshow(X_train[1],interpolation='nearest')
axarr[1,0].imshow(X_train[2],interpolation='nearest')
axarr[1,1].imshow(X_train[3],interpolation='nearest')


# In[7]:


pip install seaborn


# In[8]:


pip install sklearn


# In[9]:


Y_train=Y_train[:,0].astype(int)
Y_test=Y_test[:,0].astype(int)


# In[10]:


Y_test


# In[11]:


Y_train


# In[12]:


pca=PCA(n_components=2)
pca=pca.fit_transform(X_train.reshape(2000, 128*128))
plt.scatter(pca[:,0], pca[:,1],c=Y_train)
plt.show()


# In[13]:


pca=PCA(n_components=120)
X_train=pca.fit_transform(X_train.reshape(2000, 128*128))
X_test=pca.fit_transform(X_test.reshape(120, 128*128))
print(X_train.shape, X_test.shape)


# In[20]:


clf=KNeighborsClassifier(n_neighbors=2).fit(X_train, Y_train)
clf_score=clf.score(X_test, Y_test)
print('accuracy:', clf_score)
y_pred=clf.predict(X_test)
con_met=confusion_matrix(Y_test, y_pred, labels=clf.classes_)
disp=ConfusionMatrixDisplay(confusion_matrix=con_met, display_labels=clf.classes_)
classification_rep=classification_report(Y_test,y_pred)
print('Classification report', classification_rep)
disp.plot()
plt.show()


# In[18]:


clf=KNeighborsClassifier(n_neighbors=3).fit(X_train, Y_train)
clf_score=clf.score(X_test, Y_test)
print('accuracy:', clf_score)
y_pred=clf.predict(X_test)
con_met=confusion_matrix(Y_test, y_pred, labels=clf.classes_)
disp=ConfusionMatrixDisplay(confusion_matrix=con_met, display_labels=clf.classes_)
classification_rep=classification_report(Y_test,y_pred)
print('Classification report', classification_rep)
disp.plot()
plt.show()


# In[16]:


clf=KNeighborsClassifier(n_neighbors=8).fit(X_train, Y_train)
clf_score=clf.score(X_test, Y_test)
print('accuracy:', clf_score)
y_pred=clf.predict(X_test)
con_met=confusion_matrix(Y_test, y_pred, labels=clf.classes_)
disp=ConfusionMatrixDisplay(confusion_matrix=con_met, display_labels=clf.classes_)
classification_rep=classification_report(Y_test,y_pred)
print('Classification report', classification_rep)
disp.plot()
plt.show()


# In[19]:


clf=KNeighborsClassifier(n_neighbors=21, weights='uniform').fit(X_train, Y_train)
clf_score=clf.score(X_test, Y_test)
print('accuracy:', clf_score)
y_pred=clf.predict(X_test)
con_met=confusion_matrix(Y_test, y_pred, labels=clf.classes_)
disp=ConfusionMatrixDisplay(confusion_matrix=con_met, display_labels=clf.classes_)
classification_rep=classification_report(Y_test,y_pred)
print('Classification report\n', classification_rep)
disp.plot()
plt.show()


# In[ ]:




