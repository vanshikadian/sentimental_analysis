#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataset = pd.read_csv('amazon_alexa.tsv', delimiter = '\t')


# In[3]:


dataset.head()
dataset.shape


# In[4]:


dataset['variation'].value_counts()


# ### checking for null values

# In[5]:


dataset.isnull().sum()


# In[6]:


dataset[dataset['verified_reviews'] .isna() == True]


# In[7]:


dataset.dropna()


# ### calculating the length of the reviews

# In[8]:


dataset['length'] = dataset['verified_reviews'].astype('str').apply(len)


# In[9]:


dataset.head()


# ### Rating column

# In[10]:


print(f"Rating count: \n {dataset['rating'].value_counts()}")


# In[11]:


dataset['rating'].plot.hist(color='yellowgreen',bins=5)
plt.xticks([1,2,3,4,5])
plt.title('Rating frequency')
plt.xlabel('Rating')
plt.ylabel("Frequency")
plt.show()


# In[12]:


labels = ['1','2','3','4','5']
sizes = [161,96,152,455,2286]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','lightpink']
explode = (0.1, 0.1, 0.1, 0.1, 0.1)  

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True) 
plt.title('%wise distribution of rating')
plt.axis('equal')
plt.show()


# ### Feedback Column

# In[13]:


print(f"Feedback Frequency: \n{dataset['feedback'].value_counts()}")


# In[14]:


review_0 = dataset[dataset['feedback'] == 0].iloc[1]['verified_reviews']
print(review_0)
review_1 = dataset[dataset['feedback'] == 1].iloc[1]['verified_reviews']
print(review_1)
# this tells us that 1 feedbacks are positive and 0 feedbacks are negative


# In[15]:


dataset['feedback'].value_counts().plot.bar(color = 'lightblue')
plt.title('Feedback frequency distribution')
plt.xlabel('Feedback')
plt.ylabel('Frequency')
plt.show()


# In[16]:


labels = ['0','1']
sizes = [257,2893]
colors = ['yellowgreen', 'lightcoral']
explode = (0.1, 0.1)  

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True) 
plt.title('%wise distribution of feedback')
plt.axis('equal')
plt.show()


# In[17]:


dataset[dataset['feedback']==0]['rating'].value_counts()


# In[18]:


dataset[dataset['feedback']==1]['rating'].value_counts()


# ### So when the feedback is 0, ratings are 1 and 2. And when feedback is 1, ratings are 3,4,5.

# ### Variation Column

# In[19]:


print(f"Different variations of amazon echo: \n{dataset['variation'].value_counts()}")


# In[20]:


dataset['variation'].value_counts().plot.bar(color='yellow')
plt.xlabel('Variations')
plt.ylabel('Frequency')
plt.title('Variation distribution count')
plt.show()


# In[21]:


dataset.groupby('variation')['rating'].mean()


# In[22]:


dataset.groupby('variation')['rating'].mean().sort_values().plot.bar(color = 'pink', figsize=(11, 6))
plt.title("Mean rating according to variation")
plt.xlabel('Variation')
plt.ylabel('Mean rating')
plt.show()


# ### length analysis

# In[23]:


sns.histplot(dataset[dataset['feedback']==0]['length'],color='red').set(title='Distribution of length of review if feedback = 0')


# In[24]:


sns.histplot(dataset[dataset['feedback']==1]['length'],color='blue').set(title='Distribution of length of review if feedback = 1')


# In[25]:


dataset.groupby('length')['rating'].mean().plot.hist(color = 'green', figsize=(7, 6), bins = 20)
plt.title(" Review length wise mean ratings")
plt.xlabel('ratings')
plt.ylabel('length')
plt.show()


# In[32]:


import nltk
import re
import pickle


# In[33]:


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


# ### Data preprocessing and cleaning

# In[34]:


wordnet = WordNetLemmatizer()
corpus = []


# In[35]:


# data cleaning and preprocessing
for i in range (0,dataset.shape[0]):
    rev = re.sub('^a-zA-Z', ' ',dataset['verified_reviews'].astype('str')[i])
    rev = rev.lower()
    rev = rev.split()
    rev = [wordnet.lemmatize(word)for word in rev if word not in set(stopwords.words('english'))]
    rev = ' '.join(rev)
    corpus.append(rev)    


# In[38]:


cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray() #independent feature
y = dataset['feedback'].values #dependent feature


# In[39]:


pickle.dump(cv, open('sentimental_analysis_models/countVectorizer.pkl', 'wb'))


# ### Model Building

# In[40]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state=10)




# In[45]:


from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier


# In[46]:


scaler = MinMaxScaler()
X_train_new = scaler.fit_transform(X_train)
X_test_new = scaler.transform(X_test)


# In[47]:


pickle.dump(scaler, open('sentimental_analysis_models/scaler.pkl', 'wb'))


# In[48]:


xgb = XGBClassifier()
xgb.fit(X_train_new, y_train)


# In[51]:


from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
y_pred = xgb.predict(X_test_new)


# In[52]:


print("Training Accuracy :", xgb.score(X_train_new, y_train))
print("Testing Accuracy :", xgb.score(X_test_new, y_test))


# In[53]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[54]:


cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=xgb.classes_)
cm_display.plot()
plt.show()


# In[55]:


pickle.dump(xgb, open('sentimental_analysis_models/xgb.pkl', 'wb'))


# ## Wordcloud


# In[58]:


from wordcloud import WordCloud # type: ignore


# In[63]:


reviews = " ".join([review for review in dataset['verified_reviews'].astype('str')])
wc = WordCloud(background_color='pink', max_words=50)

plt.figure(figsize=(10,10))
plt.imshow(wc.generate(reviews))
plt.title('Wordcloud for all reviews', fontsize=10)
plt.axis('off')
plt.show()


# In[65]:


# Combine all reviews for each feedback category and splitting them into individual words
neg_reviews = " ".join([review for review in dataset[dataset['feedback'] == 0]['verified_reviews'].astype('str')])
neg_reviews = neg_reviews.lower().split()

pos_reviews = " ".join([review for review in dataset[dataset['feedback'] == 1]['verified_reviews'].astype('str')])
pos_reviews = pos_reviews.lower().split()

#Finding words from reviews which are present in that feedback category only
unique_negative = [x for x in neg_reviews if x not in pos_reviews]
unique_negative = " ".join(unique_negative)

unique_positive = [x for x in pos_reviews if x not in neg_reviews]
unique_positive = " ".join(unique_positive)


# In[67]:


wc = WordCloud(background_color='white', max_words=50)

plt.figure(figsize=(10,10))
plt.imshow(wc.generate(unique_negative))
plt.title('Wordcloud for negative reviews', fontsize=10)
plt.axis('off')
plt.show()


# In[68]:


wc = WordCloud(background_color='white', max_words=50)

plt.figure(figsize=(10,10))
plt.imshow(wc.generate(unique_positive))
plt.title('Wordcloud for positive reviews', fontsize=10)
plt.axis('off')
plt.show()


# In[ ]:




