#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv('Wheat Countries Production.csv')
data.head()


# In[2]:


data.shape


# In[3]:


# melting the data set to convert it in a structure which is easier to analyze
#new variable           #columns to leave   #new column
df=pd.melt(frame=data,id_vars=['Country','Image'],var_name='year',value_name='wheatproduction')
df


# In[4]:


del df['Image']# deleting the imange column as there is maximum null value and we are not going to use this column for further analysis.
df.head(30)


# In[5]:


#Is there any null value in the dataset
df.isna().sum()


# In[6]:


df.describe()


# In[7]:


##confirming is there a normal distribution in wheat production
import seaborn as sns
sns.distplot(df['wheatproduction'],color='blue')


# In[8]:


df['wheatproduction']= df['wheatproduction'].fillna(df['wheatproduction'].median())


# In[9]:


df.isnull().sum()


# In[10]:


df.info()


# In[11]:


#analyzing dataset graphically
#!pip install pandasgui


# In[12]:


from pandasgui import show
show(df)


# # Creating new dataset with selected european countries

# In[13]:


import numpy as np
m1= df['Country'].values == 'Ireland'
m2=df['Country'].values=='Denmark' 
m3=df['Country'].values=='Germany'
m4=df['Country'].values=='France'
m5=df['Country'].values=='Italia'
# new dataframe
df1 = df[m1+m2+m3+m4+m5]
 
df1=pd.DataFrame(df1)


# In[14]:


df1


# In[15]:


import altair as alt
alt.Chart(df1).mark_bar().encode(
  x='year:T',
  y='wheatproduction:Q',
  color='Country:N'
)


# In[16]:


df['year'] = df.year.astype('int')


# In[17]:


df.info()


# In[18]:


# Now ready to creat a panel dashboard to compare different european countries according to year and wheat production.
import panel as pn
pn.extension()


# In[19]:


#!pip install hvplot
#!pip install altair
import numpy as np
pn.extension('tabulator')


import hvplot.pandas


# In[20]:


#make dataframe pipeline interactive 
from ipywidgets import interact
idf=df.interactive()
idf.head()


# In[21]:


#define panel widgets
year_slider=pn.widgets.IntSlider(name='year slider',start=1960, end=2018, step=5, value=1860)
year_slider


# In[ ]:


yaxis_wheat=pn.widgets.RadioButtonGroup(
              name='y axis',
              options=['wheatproduction'],
              button_type='success')


# In[ ]:



country=['Germany', 'France', 'Ireland','Denmark','Italia']
wheat_pipeline=(
idf[(idf.year <= year_slider) &
    (idf.Country.isin(country))]
    .groupby(['Country','year'])[yaxis_wheat].mean()
    .to_frame()
    .reset_index()
    .reset_index(drop=True)

    .sort_values(by='year')
    
)


# In[ ]:


wheat_plot=wheat_pipeline.hvplot(x='year',by='Country',y=yaxis_wheat,line_width=2,title='Wheat production by different countries of europe')
wheat_plot


# # Different Statical test according to the dataset

# : To ascertain whether the random variable follows the null or alternative hypothesis, a statistical test can be run. It essentially reveals if there are substantial differences between a sample and the population or between two or more samples. There are different types of statistical tests, and using the results of these analyses, we can draw conclusions based on the patterns we find in the data. 

# In[25]:


#using a ttest in wheat production column to satate that is there any statistically significient difference between population and sample
#H0=there is no statistical difference in the mean of the samples
#H1= there is statistical difference in the mean of samples
new=df['wheatproduction'] # creating a new dataframe with wheatproduction column form df data set


# In[26]:


sample_size=120
new_sample=np.random.choice(new,sample_size)


# In[27]:


from scipy.stats import ttest_1samp
ttest,p_value=ttest_1samp(new_sample,100)
p_value


# In[28]:


if p_value<0.05:#assuming the significence value of 0.05
    print('we can reject null hypothesis')
else:
    print('we can not reject null hypothesis')


# # Counclusion after t-test
# It proves that, there is a significient difference of mean in the samples.But as the dataset provides imformation about different countries
# wheat production, It is very ovious there will be difference.Just only sample mean difference can not justify the whole data set.
# For that reason there is ANOVA test for justifing the mean difference between different country and their wheat production.

# In[29]:


#ANOVA test between those countries to analysise the differences 
#The null hypothesis (H0): The mean is equal across all groups.

#The alternative hypothesis: (Ha): The mean is different at least one group across all


# In[30]:


# Subsetting:
df1 = df[(df.Country=='France')]
df2 = df[(df.Country=='Germany')]
df3 = df[(df.Country=='Ireland')]
df4 = df[(df.Country=='Denmark')]
df5=df[(df.Country=='Italia')]

# Let's perform the test:
# Import the library
import scipy.stats as stats

# Perform the one-way ANOVA test:
stats.f_oneway(df1['wheatproduction'], df2['wheatproduction'], df3['wheatproduction'],df4['wheatproduction'],df5['wheatproduction'])


# In[31]:


# p- value is less than 0.05 so we can reject the H0
#we have sufficient evidence to say that at least one of the means of the groups is different from the others.


# In[32]:


# lets plot a graph to understand why we have rejected the H0
import matplotlib.pyplot as plt

# Add three histograms to one plot
plt.hist(df1['wheatproduction'], alpha=0.5, label='France')
plt.hist(df2['wheatproduction'], alpha=0.5, label='Germany')
plt.hist(df3['wheatproduction'], alpha=0.5, label='Ireland')
plt.hist(df4['wheatproduction'], alpha=0.5, label='Denmark')
plt.hist(df5['wheatproduction'], alpha=0.5, label='Italia')



# Add plot title and axis labels
plt.title('wheat production by different european country')
plt.xlabel('Wheat production')
plt.ylabel('Frequency')

# Add legend
plt.legend(title='Production of wheat')

# Display plot
plt.show()


# In[33]:


from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# In[34]:


import scipy.stats as stats
stats.f_oneway(
 *(df.loc[df['Country']==Country, 'wheatproduction'] 
 for Country in df['Country'].unique())
 )


# In[35]:


tukey = pairwise_tukeyhsd(endog=df['wheatproduction'],
                          groups=df['Country'],
                          alpha=0.05)

#display results
print(tukey)


# # Applying non-perametric test on the dataset

# In[36]:


#Perform the Kruskal-Wallis Test.(non oerametric test)
#The null hypothesis (H0): The median is equal across all groups.

#The alternative hypothesis: (Ha): The median is not equal across all

df1 = df[(df.Country=='France')]
df2 = df[(df.Country=='Germany')]
df3 = df[(df.Country=='Ireland')]
df4 = df[(df.Country=='Denmark')]
df5=df[(df.Country=='Italia')]
stats.kruskal(df1['wheatproduction'], df2['wheatproduction'], df3['wheatproduction'],df4['wheatproduction'],df5['wheatproduction'])


# We have sufficient evidence to conclude that the production of wheat leads to statistically significant differences in median

# In[37]:


## Applying friedmanchisquare test(non-perametric test)
#The null hypothesis (H0): The mean for each population is equal.
#The alternative hypothesis: (Ha): At least one population mean is different from the rest.

df1 = df[(df.Country=='France')]
df2 = df[(df.Country=='Germany')]
df3 = df[(df.Country=='Ireland')]
df4 = df[(df.Country=='Ukraine')]
df5=df[(df.Country=='Russia')]
stats.friedmanchisquare(df1['wheatproduction'], df2['wheatproduction'], df3['wheatproduction'],df4['wheatproduction'],df5['wheatproduction'])


# #in other words we have enough evidence to conclude that,the amount of wheat production has signifucient difference between those groups.

# # predicting the wheat production of ireland in upcoming year

# In[38]:


##predicting irelands wheat production in future using the df dataset
df.head()


# In[39]:


df['year']=pd.to_datetime(df.year, format='%Y')#


# In[40]:


df.set_index('year',inplace=True)


# In[41]:


df.info()


# In[42]:


# using condition where entity == ireland
# condition with df.values property
mask = df['Country'].values == 'Ireland'
 
# new dataframe
df_new = df[mask]
 
print(df_new)


# In[43]:


df_new.info()


# # Visualize the data

# In[44]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(x='year', y='wheatproduction', data=df_new)

plt.xlabel('Year')
plt.ylabel('Amount of weat')
plt.title(' wheat production in ireland every 10 year duration')
plt.show()


# As we can see from the visualisation the data is not seasonal, there is no specific time between a year that the production has
# gone up.The has gone up gradually and it has gone down in the recent year.

# # Testing for Stationarity

# In[45]:


#testing Stationarity with agumented dickey fuller test
from statsmodels.tsa.stattools import adfuller


# In[46]:


test_result=adfuller(df_new['wheatproduction'])


# In[47]:


# H0=it is non stationary
# H1=it is stationary

def adfuller_test(wheatproduction):
    result=adfuller(wheatproduction)
    labels=['ADF Test Statistics','p-Value','lags used','Number of observation used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value))
    if result[1]<=0.05:
        print('Strong evidence againts the null hypothesis (H0).reject the null hypothesis.Data has no unit root and is stationary')
    else:
        print('weak evidence against null hypothesis,time series has a unit root,indicating it is non-stationary')


# In[48]:


adfuller_test(df_new['wheatproduction'])


# # Differencing

# Performing defferencing making the data stationary

# In[49]:


df_new['production difference']=df_new['wheatproduction']-df_new['wheatproduction'].shift(1)
#did shifting with only 1 position because the data is  seasonal(by year).


# In[50]:


df_new.head()


# In[51]:


adfuller_test(df_new['production difference'].dropna())


# As we can see the data has become stationary

# In[52]:


df_new['production difference'].plot()


# # Auto Regressive Model

# In[53]:


#!pip install statsmodels
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[54]:


fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig=sm.graphics.tsa.plot_acf(df_new['production difference'].dropna(),lags=20,ax=ax1)
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(df_new['production difference'].dropna(),lags=20,ax=ax2)


# In[55]:


import statsmodels.api as sm


# In[ ]:


model=sm.tsa.statespace.SARIMAX(df_new['wheatproduction'],order=(8,1,9),seasonal_order=(8,1,9,12))
results=model.fit()


# In[57]:


df_new['forecast']=results.predict(start=1,end=59,dynamic=True)
df_new[['wheatproduction','forecast']].plot(figsize=(10,8))


# In[58]:


from pandas.tseries.offsets import DateOffset
future=[df_new.index[-1]+ DateOffset(years=x)for x in range(0,10)]


# In[59]:


future_df=pd.DataFrame(index=future[1:],columns=df_new.columns)


# In[60]:


future_df.head()


# In[61]:


future_df1=pd.concat([df_new,future_df])


# In[62]:


future_df1['forecast']=results.predict(start=10,end=60,dynamic=True)
future_df1[['wheatproduction','forecast']].plot(figsize=(10,8))


# As we can see the in the prediction there will a increase in production of wheat in the upcoming year in Ireland

# # sentiment analysis using twitter api
# 

# In[342]:


consumer_key="MJihzonipkdVoRiV4YNr8aoqC"
consumer_secret="FKmux9T0gADErmPqiS0ytDjweffBs8hjhnKIpwYgW9nr3aOL2w"
access_token='1603704377507815424-fXmVKAQhedTkjeQASkOfiV9bITa9Os'
access_token_secret="mCZp9gp4nEiPzUvJxtesYtHPS2wGhEeT95N8q8Vh0Aw45"
bearer_token="AAAAAAAAAAAAAAAAAAAAAGnzkQEAAAAAwr244l%2F9PQft%2FVD4rXU9n%2BrhwHQ%3DDly51KA4xsLbCDg76fI1UJnQMS3DyOUjHSM7gtwEl7fq39s8lu"


# In[343]:


#Data Extraction
import pandas as pd
import tweepy
query = 'wheat Ireland -is:retweet'
tw_clnt=tweepy.Client(bearer_token="AAAAAAAAAAAAAAAAAAAAAGnzkQEAAAAAwr244l%2F9PQft%2FVD4rXU9n%2BrhwHQ%3DDly51KA4xsLbCDg76fI1UJnQMS3DyOUjHSM7gtwEl7fq39s8lu")
tweets=tweepy.Paginator(tw_clnt.search_recent_tweets,query,max_results=100).flatten(limit=5000)
df=pd.DataFrame(tweets)
df.head(2)


# In[344]:


#Check for nulls/blank fields
df.id.count(), df.isnull().sum()


# In[345]:


#Remove special characters/links
import numpy as np
import re
def tweet_cleaner(x):
    text=re.sub("[@&][A-Za-z0-9_]+","", x)     # Remove mentions
    text=re.sub(r"http\S+","", text)           # Remove media links
    return  pd.Series([text])
df[['plain_text']] = df.text.apply(tweet_cleaner)
df


# In[346]:


#Convert all text to lowercase
df.plain_text = df.plain_text.str.lower()
#Remove newline character
df.plain_text = df.plain_text.str.replace('\n', '')
df


# In[347]:


#Replacing any empty strings with null
df = df.replace(r'^\s*$', np.nan, regex=True)
if df.isnull().sum().plain_text == 0:
    print ('no empty strings')
else:
    df.dropna(inplace=True)


# In[348]:


#!pip install langdetect
#detect language of tweets
from langdetect import detect
def detect_textlang(text):
    try:
        src_lang = detect(text)
        if src_lang =='en':
            return 'en'
        else:
        #return "NA"    
            return src_lang
    except:
        return "NA"
df['text_lang']=df.plain_text.apply(detect_textlang)
df


# In[349]:


# Group tweets by language and list the top 10
import matplotlib.pyplot as plt
plt.figure(figsize=(4,3))
df.groupby(df.text_lang).plain_text.count().sort_values(ascending=False).head(10).plot.bar()
plt.show()


# In[350]:


#Remove un-important words from text
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
query_words={'wheat Ireland', '#' }
stop_words.update(query_words)
for word in query_words:
    df.translated_text = df.text.str.replace(word, '')
#Creating word cloud
from wordcloud import WordCloud, ImageColorGenerator
wc=WordCloud(stopwords=stop_words, collocations=False, max_font_size=55, max_words=25, background_color="black")
wc.generate(' '.join(df.translated_text))
plt.figure(figsize=(10,12))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")


# In[351]:


#Sentiment Check
#pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer=SentimentIntensityAnalyzer()        
df['polarity']=[analyzer.polarity_scores(text)['compound'] for text in df.plain_text]
def get_sentiment(polarity):
    if polarity < 0.0:
        return 'Negative'
    elif polarity > 0.2:
        return 'Positive'
    else:
        return 'Neutral'
df['sentiment']=df.polarity.apply(get_sentiment)
df


# In[352]:


#Sentiment Check
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer=SentimentIntensityAnalyzer()        
df['polarity']=[analyzer.polarity_scores(text)['compound'] for text in df.plain_text]
def get_sentiment(polarity):
    if polarity < 0.0:
        return 'Negative'
    elif polarity > 0.2:
        return 'Positive'
    else:
        return 'Neutral'
df['sentiment']=df.polarity.apply(get_sentiment)
plt.figure(figsize=(3,3))
df.sentiment.value_counts().plot.bar()


# In[353]:


df.head()


# In[354]:


df.shape


# In[355]:


#data pre-processing

# Store the column of the dataframe named as "text"
X = df['plain_text']

# Display the value "X"
print(X)


# In[356]:


# Store the column if the dataframe named as  'sentiment"
y = df['sentiment']

# Display the column of the dataframe named as "sentiment"
print(y)


# In[357]:


#lets clean the text data


# In[358]:


import nltk


# In[359]:


nltk.download('stopwords')

from nltk.corpus import stopwords


import string
from nltk.stem import PorterStemmer


# In[360]:


# Store the stopwords into the object named as "stop_words"
stop_words = stopwords.words('english')

# Store the string.punctuation into an object punct
punct = string.punctuation

# Initialise an object using a method PorterStemmer
stemmer = PorterStemmer()


# In[361]:


import re

cleaned_data=[]

# For loop from first value to length(X), ^a-zA-Z means include small and capital case letters

for i in range(len(X)):
    tweet = re.sub('[^a-zA-Z]', ' ', X.iloc[i])
    tweet = tweet.lower().split()
    tweet = [stemmer.stem(word) for word in tweet if (word not in stop_words) and (word not in punct)]
    tweet = ' '.join(tweet)
    cleaned_data.append(tweet)


# In[362]:


# Display the cleaned_data
cleaned_data


# In[363]:


print(y)


# In[364]:


# Collect all columns into dataframe named as sentiment_ordering
sentiment_ordering = ['Negative', 'Neutral', 'Positive']

# store all values into column named as "y"
y = y.apply(lambda x: sentiment_ordering.index(x))


# In[365]:


y.head()


# In[366]:


#Bag of words using count-vectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Instantiate an object cv by calling a method named as CountVectorzer()
cv    = CountVectorizer(max_features = 4000, stop_words = ['wheat', 'Ireland'])

# Train the dataset by calling a fit_transform() method
X_fin = cv.fit_transform(cleaned_data).toarray()

# Display the rows and colums
X_fin.shape


# In[367]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Instantiate an object model by calling a method MultinomialNB()
model = MultinomialNB()


# In[368]:


# Split the dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X_fin, y, test_size = 0.2)


# In[369]:


# Train the model by calling a method fit()
model.fit(X_train,y_train)


# In[370]:


# Call predict() method
y_pred = model.predict(X_test)


# In[371]:


from sklearn.metrics import classification_report

# Instantiate a mthod named as Cla
cf = classification_report(y_test, y_pred)

# Display the values of an object cf
print(cf)


# In[379]:


# is the model is underfitted or overfitted
print('Training set score: {:.2f}'.format(model.score(X_train, y_train)))

print('Test set score: {:.2f}'.format(model.score(X_test, y_test)))


# # using multinomial logistic regression for sentiment analysis

# In[382]:


from sklearn.linear_model import LogisticRegression
x_train, x_test, y_train, y_test = train_test_split(X_fin, y, test_size=0.2, random_state=0)
clf = LogisticRegression (solver='lbfgs',multi_class='multinomial')
clf.fit(x_train, y_train)
clf.score(x_test, y_test)


# In[383]:


# Call predict() method
y_pred = clf.predict(x_test)


# In[384]:


cf = classification_report(y_test, y_pred)
print(cf)


# In[386]:


# is the model is underfitted or overfitted
print('Training set score: {:.2f}'.format(clf.score(X_train, y_train)))

print('Test set score: {:.2f}'.format(clf.score(X_test, y_test)))


# In[387]:


from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
x_train, x_test, y_train, y_test = train_test_split(X_fin, y, test_size=0.2)
rfc.fit(x_train, y_train)


# In[388]:


y_pred = rfc.predict(x_test)


# In[389]:


cf = classification_report(y_test, y_pred)
print(cf)


# In[390]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier()
forest_params = [{'max_depth': list(range(1,2)), 'max_features': list(range(0,5))}]
clf = GridSearchCV(rfc, forest_params, cv = 2, scoring='accuracy')
clf.fit(x_train, y_train)
print(clf.best_params_)

print(clf.best_score_)


# # Sentiment analysis using LSTM model

# In[391]:


#importing necessary libraries
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# In[392]:


#we will be removing the nutral sentiment from the column, it is not necessary but can enhance the performance of the model
df = df.loc[(df['sentiment'] == 'Positive') | (df['sentiment'] == 'Negative')]

df.loc[df['sentiment'] == 'Positive','sentiment'] = 1
df.loc[df['sentiment'] == 'Negative','sentiment'] = 0


# In[393]:


#To build a binary classification model, the sentiment column should be converted to numeric values such that ‘positive’ sentiment is regarded as 1 and ‘negative’ sentiment as 0. We can achieve this easily by modifying the column of pandas DataFrame using np.where():

df['sentiment'] = np.where(df['sentiment'] == 'positive', 1, 0)


#  The next thing we need to do is to convert the labels and reviews to NumPy arrays as pre-processing methods favor arrays instead of pandas series:

# In[394]:


sentence = df['plain_text'].to_numpy()
label= df['sentiment'].to_numpy()


# In[395]:


X_train, X_test, y_train, y_test = train_test_split(sentence, label, test_size=0.25)
print("Training Data Input Shape: ", X_train.shape)
print("Training Data Output Shape: ", y_train.shape)
print("Testing Data Input Shape: ", X_test.shape)
print("Testing Data Output Shape: ", y_test.shape)


# In[396]:


vocab_size = 10000
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)


# In[397]:


tokenizer.fit_on_texts(X_train)
print("Number of Documents: ", tokenizer.document_count)
print("Number of Words: ", tokenizer.num_words)


# In[398]:


tokenizer.word_counts


# In[399]:


tokenizer.word_docs


# In[400]:


train_sequences = tokenizer.texts_to_sequences(X_train)
print(train_sequences[0])


# In[401]:


sequence_length = 200
train_padded = pad_sequences(train_sequences, maxlen=sequence_length, padding='post', truncating='post')


# In[402]:


test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences, maxlen=sequence_length, padding='post', truncating='post')


# # Sentiment Analysis using LSTM

# In[403]:


model = Sequential()


# In[404]:


embedding_dim = 16
model.add(Embedding(vocab_size, embedding_dim, input_length=sequence_length))


# In[405]:


lstm_out = 32
model.add(Bidirectional(LSTM(lstm_out)))


# In[406]:


model.add(Dense(10, activation='relu'))


# In[407]:


model.add(Dense(1, activation='sigmoid'))


# In[408]:



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[409]:


print(model.summary())


# In[410]:


import os
checkpoint_filepath = os.getcwd()
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)
callbacks = [EarlyStopping(patience=2), model_checkpoint_callback]


# In[411]:


history = model.fit(train_padded, y_train, epochs=3, validation_data=(test_padded, y_test), callbacks=callbacks)


# In[414]:


metrics_df = pd.DataFrame(history.history)
print(metrics_df)


# In[418]:


plt.figure(figsize=(10,5))
plt.plot(metrics_df.index, metrics_df.loss)
plt.plot(metrics_df.index, metrics_df.val_loss)
plt.title('Sentiment Analysis Model Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Binary Crossentropy')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()


# In[423]:


plt.figure(figsize=(10,5))
plt.plot(metrics_df.index, metrics_df.accuracy)
plt.plot(metrics_df.index, metrics_df.val_accuracy)
plt.title(' Sentiment Analysis Model Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




