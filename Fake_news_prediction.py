#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd


# In[63]:


df=pd.read_csv("train.csv")


# In[64]:


df.head()


# In[65]:


df.describe()


# In[66]:


df.info()


# In[67]:


df.isnull().sum()


# In[68]:


df=df.fillna('')


# In[69]:


df.isnull().sum()


# # Data Visualization

# In[70]:


# Visualize the distribution of target labels
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='label', data=df)
plt.title('Distribution of News Labels')
plt.show()


# In[71]:


import plotly.express as px

df = pd.read_csv('train.csv')

#Let's visualize the proportion of real and fake news!
#real vs fake
fig = px.pie(df, names='label', title='Proportion of Real vs. Fake News')
fig.show()


# In[72]:


# Visualize the correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[ ]:





# # Data Pre-processing

# In[9]:


df.columns


# In[10]:


df=df.drop(['id', 'title', 'author'], axis=1)


# In[11]:


df.head()


# # Stemming: 
# #Stemming is the process of reducing a word to its Root word
# 
# #example: actor, actress, acting --> act

# In[12]:


from nltk.corpus import stopwords


# In[13]:


from nltk.stem.porter import PorterStemmer


# In[14]:


import re


# In[15]:


port_stem=PorterStemmer()


# In[16]:


port_stem


# In[18]:


def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con


# In[20]:


df['text']= df['text'].apply(stemming)


# In[21]:


x=df['text']


# In[23]:


print(x)


# In[24]:


y=df['label']


# In[25]:


print(y)


# In[26]:


x.shape


# In[27]:


y.shape


# # Splitting the dataset to training & test data

# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train , x_test , y_train, y_test = train_test_split(x, y, test_size=0.20)


# # converting the textual data to numerical data

# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[31]:


vect=TfidfVectorizer()


# In[32]:


x_train=vect.fit_transform(x_train)
x_test=vect.transform(x_test)


# In[33]:


x_test.shape


# # Training the Model

# In[34]:


from sklearn.tree import DecisionTreeClassifier


# In[35]:


model=DecisionTreeClassifier()


# In[36]:


model.fit(x_train, y_train)


# In[37]:


prediction=model.predict(x_test)


# In[38]:


prediction


# # Evaluation Accuracy score

# In[39]:


model.score(x_test, y_test)


# In[61]:


import seaborn as sns
from sklearn import metrics
sns.set(rc = {'figure.figsize':(7,7)})
colormap = sns.color_palette("Reds")
p = sns.heatmap(metrics.confusion_matrix(y_test, prediction), annot=True, cmap=colormap)

p.set_xlabel('Predicted Values')
p.set_ylabel('Actual Values')


# # To save the trained vectorizer and model objects to disk 

# In[40]:


import pickle


# In[41]:


pickle.dump(vect, open('vector.pkl', 'wb'))


# In[42]:


pickle.dump(model, open('model.pkl', 'wb'))


# In[43]:


vector_form=pickle.load(open('vector.pkl', 'rb'))


# In[44]:


load_model=pickle.load(open('model.pkl', 'rb'))


# # Making a Predictive System

# In[45]:


def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction


# In[46]:


val=fake_news("""In these trying times, Jackie Mason is the Voice of Reason. [In this week’s exclusive clip for Breitbart News, Jackie discusses the looming threat of North Korea, and explains how President Donald Trump could win the support of the Hollywood left if the U. S. needs to strike first.  “If he decides to bomb them, the whole country will be behind him, because everybody will realize he had no choice and that was the only thing to do,” Jackie says. “Except the Hollywood left. They’ll get nauseous. ” “[Trump] could win the left over, they’ll fall in love with him in a minute. If he bombed them for a better reason,” Jackie explains. “Like if they have no transgender toilets. ” Jackie also says it’s no surprise that Hollywood celebrities didn’t support Trump’s strike on a Syrian airfield this month. “They were infuriated,” he says. “Because it might only save lives. That doesn’t mean anything to them. If it only saved the environment, or climate change! They’d be the happiest people in the world. ” Still, Jackie says he’s got nothing against Hollywood celebs. They’ve got a tough life in this country. Watch Jackie’s latest clip above.   Follow Daniel Nussbaum on Twitter: @dznussbaum """)


# In[47]:


if val==[0]:
    print('reliable')
else:
    print('unreliable')


# In[ ]:




