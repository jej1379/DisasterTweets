import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
from pytorch_pretrained_bert import BertTokenizer

BERT_MODEL = 'bert-base-uncased'

tweet= pd.read_csv('./data/train.csv')
test=pd.read_csv('./data/test.csv')

"""
output=open('./data/train.tsv','w',encoding='utf-8')
for i, row in tweet.iterrows():
    k=row['keyword'] if type(row['keyword'])!=float else ''
    output.write('%d\t%s\t%s\t%s\n' %(row['id'], row['target'], k, re.sub(r'\n|\r|\r\n',' ',row['text'])))
"""
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

#
def keyword_location_ratio(typ='keyword', label=1):
    feats=tweet[typ].dropna().unique()
    ratios=[[],[]]
    for f in feats:
        x=tweet[tweet[typ]==f].target.value_counts()
        for label in x.index:
            ratios[label].append(x[label] / sum(x))

    plt.hist(ratios[label], label=['%d' %label])
    plt.legend()
    plt.title('tweets having %s w/ label=%d' %(typ, label))
    plt.show()

# Histogram
def label_count():
    x=tweet.target.value_counts()
    sns.barplot(x.index,x)
    plt.gca().set_ylabel('frequency')
    plt.show()

# no of characters
def num_characters():
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
    tweet_len=tweet[tweet['target']==1]['text'].str.len()
    ax1.hist(tweet_len,color='red')
    ax1.set_title('disaster tweets(avg=%.2f)' %(sum(list(tweet_len))/float(len(tweet_len))))
    tweet_len=tweet[tweet['target']==0]['text'].str.len()
    ax2.hist(tweet_len,color='green')
    ax2.set_title('Not disaster tweets(avg=%.2f)' %(sum(list(tweet_len))/float(len(tweet_len))))
    fig.suptitle('Characters in tweets')
    plt.show()

def num_stopwords(k=10):
    stop_words = set(stopwords.words('english'))
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
    stops=list()
    for x in tweet[tweet['target'] == 1]['text'].str.split():
        for w in x:
            w=w.lower()
            if w in stop_words: stops.append(w)
    cnt=Counter(stops).most_common(k)
    x,y=zip(*cnt)
    ax1.bar(x, y)
    ax1.set_title('disaster tweets')

    stops = list()
    for x in tweet[tweet['target'] == 0]['text'].str.split():
        for w in x:
            w = w.lower()
            if w in stop_words: stops.append(w)
    cnt = Counter(stops).most_common(k)
    x, y = zip(*cnt)
    ax2.bar(x, y,color='green')
    ax2.set_title('Not disaster')
    fig.suptitle('stopwords in tweets')
    plt.show()

def num_engdigit():
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
    engdigit=list()
    for x in tweet[tweet['target'] == 1]['text'].str.split():
        engdigit.append(len([w for w in x if re.search('[a-z0-9]', w.lower())]))

    ax1.hist(engdigit,color='red')
    ax1.set_title('disaster tweets(avg=%.2f)' %(sum(engdigit)/float(len(engdigit))))

    engdigit=list()
    for x in tweet[tweet['target'] == 0]['text'].str.split():
        engdigit.append(len([w for w in x if re.search('[a-z0-9]', w.lower())]))
    ax2.hist(engdigit,color='green')
    ax2.set_title('Not disaster tweets(avg=%.2f)' %(sum(engdigit)/float(len(engdigit))))
    fig.suptitle('eng/digit in tweets')
    plt.show()

if __name__ == '__main__':
    keyword_location_ratio('location', 0)
    '''
    label_count()
    num_characters()
    num_stopwords(k=10)
    num_engdigit()
    '''
