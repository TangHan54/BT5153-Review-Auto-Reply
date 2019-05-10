import time
import pickle 
import re
import numpy as np
import os
import pandas as pd
import pickle
from imblearn.over_sampling import RandomOverSampler

# load data
fpath = 'data/google_play_review.pickle'
with open(fpath, 'rb') as f:
    df = pickle.load(f)

# Over sampling
ros = RandomOverSampler(random_state=666)
ros.fit(df[['reviews','replies']],df[['ratings']])
X,y =  ros.fit_sample(df[['reviews','replies']],df[['ratings']])
df = pd.DataFrame(X,columns= ['reviews','replies'])

# clean text
def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"jennyhealintcom", "<EMAIL>", text)
    
    return text

reviews = [clean_text(i) for i in df.reviews if isinstance(i, str)]
replies = [clean_text(j) for (i,j) in zip(df.reviews, df.replies) if isinstance(i, str)]

minlen = 2
maxlen = 100

# Filter out the reviews/replies that are too short/long
review_reply = [(i,j) for (i,j) in zip(reviews, replies) if (minlen<=len(i.split())<=maxlen) & (minlen<=len(j.split())<=maxlen)]

all_vob = []
for (i,j) in review_reply:
    all_vob = all_vob + i.split() + j.split()

# Remove rare words from the vocabulary.
threshold = 2
vocab = {i:all_vob.count(i) for i in set(all_vob)}
vocab = {word:vocab[word] for word in vocab.keys() if vocab[word] > threshold}

# The ratio of remaining words
print(len(vocab)/len(set(all_vob)))

adds = ['<PAD>','<EOS>','<UNK>','<GO>']
vocab_reduced = adds + list(vocab.keys()) 
vocab_reduced = {word: idx for (idx, word) in enumerate(vocab_reduced)}

# replace rare words with <UNK>
# add <EOS> to the end of every reply.
# transform reviews/replies to a list of list of index
review_matrix = []
reply_matrix = []
for review, reply in review_reply:
    review_int = [vocab_reduced[word] if word in vocab_reduced.keys() else vocab_reduced['<UNK>'] for word in review.split()]
    reply_int = [vocab_reduced[word] if word in vocab_reduced.keys() else vocab_reduced['<UNK>'] for word in reply.split()]
    reply_int.append(vocab_reduced['<EOS>'])
    review_matrix.append(review_int)        
    reply_matrix.append(reply_int)


with open('data/vocab_reduced.pickle','wb') as f: 
    pickle.dump(vocab_reduced,f) 
with open('data/review_matrix.pickle','wb') as f:
    pickle.dump(review_matrix, f)
with open('data/reply_matrix.pickle','wb') as f:
    pickle.dump(reply_matrix, f)