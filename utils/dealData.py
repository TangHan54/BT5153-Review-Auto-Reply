import pickle
import os
import pandas as pd
from imblearn.over_sampling import RandomOverSampler


with open(os.path.abspath('data')+'/google_play_review.pickle','rb') as f:
    df = pickle.load(f)

ros = RandomOverSampler(random_state=666)
ros.fit(df[['reviews','replies']],df[['ratings']])
X,y =  ros.fit_sample(df[['reviews','replies']],df[['ratings']])
df = pd.DataFrame(X,columns= ['reviews','replies'])

train_size = int(len(df)*0.9)

with open(os.path.abspath('data')+'/train.txt','w') as f:
    for index, row in df[:train_size].iterrows():
        if isinstance(row['reviews'],float):
            continue
        f.write(row['reviews'].replace('\r\n',' '))
        f.write('\n')
        f.write(row['replies'].replace('\r\n',' '))
        f.write('\n')


with open(os.path.abspath('data')+'/train.reviews.txt','w') as f:
    for index, row in df[:train_size].iterrows():
        if isinstance(row['reviews'],float):
            continue
        f.write(row['reviews'].replace('\r\n',' '))
        f.write('\n')

with open(os.path.abspath('data')+'/train.replies.txt','w') as f:
    for index, row in df[:train_size].iterrows():
        if isinstance(row['reviews'],float):
            continue
        f.write(row['replies'].replace('\r\n',' '))
        f.write('\n')

with open(os.path.abspath('data')+'/test.reviews.txt','w') as f:
    for index, row in df[train_size:].iterrows():
        if isinstance(row['reviews'],float):
            continue
        f.write(row['reviews'].replace('\r\n',' '))
        f.write('\n')

with open(os.path.abspath('data')+'/test.replies.txt','w') as f:
    for index, row in df[train_size:].iterrows():
        if isinstance(row['reviews'],float):
            continue           
        f.write(row['replies'].replace('\r\n',' '))
        f.write('\n')

